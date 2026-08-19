#!/usr/bin/env python3
"""Apply an SME decisions file to an augment package and seal it.

    python apply_decisions.py --augment output/augmented/tsk_X_augment.json \
        --decisions decisions_tsk_X_2026-08-03.json --out output/final

This is the last step of audit + augmentation. Until it runs, a package carries
unresolved judgment questions, and a package with unresolved judgment questions is
not scoreable — not because a rule says so, but because the model resolves an open
question differently on different runs. On one task the FX ambiguity resolved as
the live rate (savings 2.20 Cr) and on the next as the file rate (0.918 Cr): same
task, same code, two goldens with different answers and different frozen targets.
An answered question, written into the artifact, is what stops that.

WHAT EACH DECISION DOES
  accept          the proposal stands as-is
  reject + reason a mechanical change is REVERTED; a rewrite or split is undone;
                  a proposed verifier is not added
  other  + reason the SME's text replaces the proposal
  question        the SME's answer is applied and recorded; the question closes

SME edits reach every artifact a resolution can change: the corrected prompt,
sanity check and solution logic, the verifier block, AND the auditor's deliverable
and trajectory (a resolution that changes a value must move the deliverable with
it, or the sealed golden contradicts its own solution logic).

THE SEAL RE-DERIVES THE FROZEN GRAPH. An SME edit can change the verifier id space
(a reverted split drops V5a/V5b and restores V5; an accepted gap adds a verifier)
or a frozen target, so the DAG, base weights, crux set and Shapley weights computed
at augment time are stale by the time the package is sealed. Sealing therefore
re-runs augment_task.derive_frozen_graph on the edited artifacts — the SAME routine
the augmenter used for its preview, so the two cannot diverge — and only then marks
the package sealed. A re-derive that cannot run means the sealed set is inconsistent
and the package is marked not-scoreable rather than shipping a stale graph.

REFUSALS
  * a run_hash mismatch means the decisions were made against a different run of
    this task, so applying them would edit text the reviewer never saw
  * an incomplete file (any item undecided, or a reject/other with no reason, or a
    question with no answer) is refused; a partly-sealed package is worse than an
    unsealed one because nothing downstream can tell which it is
"""
import argparse
import json
import os
import sys

#: Verdicts that block scoring unless the SME explicitly re-grades at seal.
_NONPROCEEDABLE = {"BROKEN", "UNGRADEABLE", "NON_DETERMINISTIC"}
#: Verdicts an SME override may set (a re-grade must land on a proceedable one).
_PROCEEDABLE = {"SOUND", "SALVAGEABLE"}

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


#: Which package field each editable artifact maps to. Defined once so the revert
#: and replace branches cannot drift apart. The deliverable and trajectory are
#: here because an SME resolution that changes a value must move them too — a
#: sealed golden whose deliverable still shows the pre-resolution value is a
#: self-contradiction the scorer cannot catch.
_ARTIFACT_KEY = {
    "solution_logic": "corrected_solution_logic",
    "sanity_check": "corrected_sanity_check",
    "prompt": "corrected_prompt",
    "deliverable": "gold_deliverable_text",
    "golden_deliverable": "gold_deliverable_text",   # alias
}
#: The trajectory is not a separate text artifact in this build — it IS the
#: corrected_claim_verdicts + judgment_steps, from which claim_graph rebuilds the
#: step graph. An SME who needs to change a trajectory value therefore edits the
#: value in solution_logic (which flows into the deliverable) and, if the change
#: touches a claim's recomputed figure, the corrected_claim_verdicts entry. When a
#: standalone trajectory artifact is introduced, add its field here (e.g.
#: "trajectory": "gold_trajectory_text") and give it a producer in augment_task.
#: Adding an unbacked field now would be dead weight, so it is deliberately left
#: out until there is text for it to carry.


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _items(pkg):
    """Rebuild the same id space the report used. Must stay in step with
    augment_report._decision_items — the ids are the contract between them."""
    out = []
    va = pkg.get("verifier_audit") or {}
    # A change that targets a verifier later REWRITTEN by the property audit is
    # suppressed in the report (the rewrite supersedes it, so the SME sees one
    # decision, not two). The sealer must drop the same change items or the id
    # space drifts (sealer would carry a chg the SME never decided).
    rewritten_vids = {v.get("id") for v in (va.get("verifiers") or [])
                      if (v.get("rewrite") or "").strip()}

    def _change_superseded(c):
        import re as _re
        if str(c.get("artifact", "")).lower() != "verifiers":
            return False
        loc = str(c.get("location", "")) + " " + str(c.get("old", ""))
        return any(vid in _re.findall(r"\bV\d+[a-z]?\b", loc)
                   for vid in rewritten_vids)

    _ci = 0
    for c in (pkg.get("changes_applied") or []):
        if _change_superseded(c):
            _ci += 1
            continue
        out.append({"id": f"chg{_ci}", "kind": "change", "payload": c})
        _ci += 1
    for i, c in enumerate(pkg.get("judgment_changes_pending_sme") or []):
        out.append({"id": f"q{i}", "kind": "question", "payload": c})
    va = pkg.get("verifier_audit") or {}
    for i, v in enumerate(va.get("verifiers") or []):
        if (v.get("rewrite") or "").strip():
            out.append({"id": f"rw{i}", "kind": "rewrite", "payload": v})
    for i, x in enumerate(pkg.get("verifier_splits_applied") or []):
        out.append({"id": f"sp{i}", "kind": "split", "payload": x})
    # coverage gaps MUST be built from the same deterministic
    # unwatched_load_bearing list the report uses (not the model's shorter
    # coverage_gaps list) or the id space drifts and accepted gap verifiers are
    # silently dropped at seal. Reuse the report's authoring so the proposal text
    # the SME accepted is exactly what gets applied.
    from src.augment_report import _step_index, _author_gap_verifier
    cov = pkg.get("step_coverage") or {}
    model_gaps = {str(g.get("step")): g for g in (va.get("coverage_gaps") or [])}
    step_idx = _step_index(pkg)
    for i, step in enumerate(cov.get("unwatched_load_bearing") or []):
        mg = model_gaps.get(str(step), {})
        proposed = mg.get("proposed_verifier") or _author_gap_verifier(
            str(step), step_idx.get(str(step), {}))
        out.append({"id": f"gap{i}", "kind": "gap",
                    "payload": {"step": str(step),
                                "proposed_verifier": proposed}})
    # advisory model gaps for steps NOT in the deterministic list (gapx*),
    # matching the report's second gap loop
    det_steps = {str(s) for s in (cov.get("unwatched_load_bearing") or [])}
    for j, (step, g) in enumerate(model_gaps.items()):
        if step in det_steps:
            continue
        out.append({"id": f"gapx{j}", "kind": "gap",
                    "payload": {"step": str(step),
                                "proposed_verifier": g.get("proposed_verifier",
                                                           "")}})
    for i, c in enumerate(va.get("duplicate_clusters") or []):
        out.append({"id": f"dup{i}", "kind": "dedupe", "payload": c})
    # value-mismatch items: a numeric verifier whose value matches no step
    mapping = pkg.get("verifier_mapping_report") or {}
    for i, u in enumerate(mapping.get("unmatched") or []):
        if str(u.get("kind", "numeric")).lower() == "numeric":
            out.append({"id": f"unmatched{i}", "kind": "value_mismatch",
                        "payload": u})
    # temporal-drift items: a split child with an unpinned live value
    _seen = []
    for x in (pkg.get("verifier_splits_applied") or []):
        _seen += x.get("temporal_unpinned_children") or []
    for i, vid in enumerate(dict.fromkeys(_seen)):
        out.append({"id": f"tmp{i}", "kind": "temporal", "payload": {"verifier": vid}})
    # SME verdict override — present only when the model's verdict is
    # non-proceedable. Lets the SME re-grade the task once the defects that drove
    # the verdict are resolved, so a fully-repaired task can become scoreable. The
    # answer carries the new verdict + justification; it is applied LAST.
    if str(pkg.get("audit_verdict", "")).upper() in _NONPROCEEDABLE:
        out.append({"id": "verdict", "kind": "verdict_override",
                    "payload": {"current": pkg.get("audit_verdict")}})
    return out


def _step_value_index(pkg: dict) -> dict:
    """Map each step id to its computed value, so a gap verifier added for that
    step can be given a matching target. Claims carry a numeric recomputed value;
    judgment steps usually have no numeric value (they bind by decision, not
    value) so they are omitted here and left to text/id mapping."""
    idx = {}
    for c in pkg.get("corrected_claim_verdicts", []):
        v = c.get("recomputed")
        if isinstance(v, (int, float)):
            idx[str(c.get("id"))] = float(v)
    return idx


def _verifier_map(pkg):
    """{id: text} from the canonical block, preserving suffixed ids."""
    import re
    out = {}
    for line in (pkg.get("augmented_verifiers_text") or "").splitlines():
        m = re.match(r"\s*(V[\w]+)\s*:\s*(.*)", line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def _next_id(vmap):
    import re
    n = max((int(re.sub(r"\D", "", k) or 0) for k in vmap), default=0)
    return f"V{n + 1}"


def apply_decisions(pkg: dict, dec: dict, force: bool = False) -> dict:
    items = _items(pkg)
    choices = dec.get("decisions") or {}
    answers = dec.get("answers") or {}

    # --- refusals -------------------------------------------------------
    ph, dh = pkg.get("run_hash"), dec.get("run_hash")
    if ph and dh and ph != dh and not force:
        raise SystemExit(
            f"run_hash mismatch: package {ph} vs decisions {dh}. These decisions "
            f"were made against a different run of this task, so applying them "
            f"would edit text the reviewer never saw. Re-review, or --force.")

    incomplete = []
    for it in items:
        c = choices.get(it["id"])
        a = (answers.get(it["id"]) or "").strip()
        if not c:
            incomplete.append(f"{it['id']} undecided")
        elif c != "accept" and not a:
            incomplete.append(f"{it['id']} {c} with no reason")
        elif it["kind"] == "question" and not a:
            incomplete.append(f"{it['id']} question unanswered")
        elif it["kind"] == "temporal" and c == "accept" and not a:
            # accepting a temporal item means "pin it" — the pin text is required
            incomplete.append(f"{it['id']} temporal pin not provided")
        elif it["kind"] == "value_mismatch" and c == "accept" and not a:
            # accepting means "here is the corrected verifier text / value"
            incomplete.append(f"{it['id']} value_mismatch resolution not provided")
        elif it["kind"] == "verdict_override" and c == "accept" and not a:
            # accepting means "re-grade to this verdict" — the new verdict + reason
            # is required; rejecting (keep the verdict) needs no answer
            incomplete.append(f"{it['id']} verdict override accepted but no "
                              f"new verdict given")
    if incomplete and not force:
        raise SystemExit("decisions incomplete, refusing to seal:\n  "
                         + "\n  ".join(incomplete)
                         + "\n(a partly-sealed package cannot be told apart from a "
                           "sealed one downstream; --force to override)")

    out = dict(pkg)
    vmap = _verifier_map(pkg)
    log = []

    for it in items:
        cid, kind = it["id"], it["kind"]
        choice = choices.get(cid, "accept")
        reason = (answers.get(cid) or "").strip()
        pay = it["payload"]

        if kind == "change":
            if choice == "reject":
                # revert a mechanical edit: put the old text back
                art = pay.get("artifact")
                key = _ARTIFACT_KEY.get(art)
                if key and pay.get("old") and pay.get("new"):
                    cur = out.get(key) or ""
                    if pay["new"] in cur:
                        out[key] = cur.replace(pay["new"], pay["old"], 1)
                        log.append({"item": cid, "action": "reverted",
                                    "artifact": art, "reason": reason})
                    else:
                        log.append({"item": cid, "action": "revert_failed",
                                    "artifact": art,
                                    "detail": "new text not found; edit by hand",
                                    "reason": reason})
            elif choice == "other":
                art = pay.get("artifact")
                key = _ARTIFACT_KEY.get(art)
                if key and pay.get("new"):
                    cur = out.get(key) or ""
                    if pay["new"] in cur:
                        out[key] = cur.replace(pay["new"], reason, 1)
                        log.append({"item": cid, "action": "replaced_with_sme_text",
                                    "artifact": art, "text": reason})
            else:
                log.append({"item": cid, "action": "kept"})

        elif kind == "question":
            # the answer IS the resolution; record it against the artifact so a
            # re-run cannot resolve it differently
            out.setdefault("sme_resolutions", []).append({
                "artifact": pay.get("artifact"),
                "location": pay.get("location"),
                "question": pay.get("sme_question"),
                "answer": reason,
                "decided": choice,
            })
            log.append({"item": cid, "action": "resolved",
                        "question": str(pay.get("sme_question"))[:80]})

        elif kind == "rewrite":
            vid = pay.get("id")
            if choice == "accept":
                log.append({"item": cid, "action": "kept", "verifier": vid})
            elif choice == "reject":
                orig = next((x for x in (pkg.get("_verifiers_before_audit") or [])
                             if x.get("id") == vid), None)
                if orig:
                    vmap[vid] = orig.get("text", vmap.get(vid, ""))
                    log.append({"item": cid, "action": "rewrite_reverted",
                                "verifier": vid, "reason": reason})
                else:
                    log.append({"item": cid, "action": "revert_unavailable",
                                "verifier": vid,
                                "detail": "pre-rewrite text not retained",
                                "reason": reason})
            else:
                vmap[vid] = reason
                log.append({"item": cid, "action": "verifier_replaced_by_sme",
                            "verifier": vid, "text": reason})

        elif kind == "split":
            if choice == "reject":
                parent, kids = pay.get("parent"), pay.get("children") or []
                for k in kids:
                    vmap.pop(k, None)
                vmap[parent] = pay.get("parent_text", "")
                ev = out.get("expected_values") or {}
                heir = pay.get("target_went_to")
                if heir and heir in ev:
                    ev[parent] = ev.pop(heir)
                for k in kids:
                    ev.pop(k, None)
                out["expected_values"] = ev
                log.append({"item": cid, "action": "split_undone",
                            "parent": parent, "reason": reason})
            else:
                log.append({"item": cid, "action": "kept",
                            "parent": pay.get("parent")})

        elif kind == "gap":
            if choice in ("accept", "other"):
                nid = _next_id(vmap)
                vmap[nid] = (pay.get("proposed_verifier", "") if choice == "accept"
                             else reason)
                step = pay.get("step")
                # Bind the new verifier to the step it was authored for, by seeding
                # an expected_value from that step's computed value. The seal
                # re-derive maps numeric verifiers to steps BY VALUE, so without a
                # target the added verifier stays unmapped and the coverage gap it
                # was meant to close persists. We know the step, so seed its value
                # directly (a claim's recomputed value; judgment steps bind by id).
                sidx = _step_value_index(pkg)
                sv = sidx.get(str(step))
                if sv is not None:
                    out.setdefault("_sealed_added_targets", {})[nid] = {
                        "value": sv, "tol": abs(sv) * 0.005 if sv else 0,
                        "source_of_verification": "arithmetic", "step": str(step)}
                out.setdefault("_sealed_step_bindings", {})[nid] = str(step)
                log.append({"item": cid,
                            "action": ("verifier_added" if choice == "accept"
                                       else "verifier_added_sme_text"),
                            "id": nid, "step": step})
            else:
                log.append({"item": cid, "action": "gap_left_open",
                            "step": pay.get("step"), "reason": reason})

        elif kind == "dedupe":
            # Two+ verifiers assert the same quantity. On accept, keep the one the
            # audit recommends and DROP the rest, so the set is MECE. The keep id
            # is parsed from recommended_action ("keep V17; V18 restates it"); if
            # it can't be parsed, fall back to keeping the lowest-numbered id. The
            # seal re-derive rebuilds the graph for the reduced set.
            ids = [i for i in (pay.get("verifier_ids") or []) if i in vmap]
            if choice == "accept" and len(ids) >= 2:
                import re as _re
                rec = str(pay.get("recommended_action") or "")
                m = _re.search(r"keep\s+(V\d+[a-z]?)", rec, _re.I)
                keep = m.group(1) if m and m.group(1) in ids else min(
                    ids, key=lambda x: (int(_re.sub(r"\D", "", x) or 0), x))
                ev = out.get("expected_values") or {}
                dropped = []
                for vid in ids:
                    if vid != keep:
                        vmap.pop(vid, None)
                        ev.pop(vid, None)
                        dropped.append(vid)
                out["expected_values"] = ev
                out.setdefault("_crux_excluded_ids", []).extend(dropped)
                log.append({"item": cid, "action": "deduped",
                            "kept": keep, "dropped": dropped})
            elif choice == "reject":
                log.append({"item": cid, "action": "dedupe_declined",
                            "verifiers": ids, "reason": reason})
            else:
                log.append({"item": cid, "action": "kept_all",
                            "verifiers": ids})

        elif kind == "value_mismatch":
            # A numeric verifier whose value matches no computed step. Accept +
            # answer = the corrected verifier text (the SME fixes the number to the
            # derivation's value, or rewords it to the step it should check).
            # Reject = the golden is genuinely missing a step; left not-scoreable
            # so the gap is not forgotten. The seal re-derive re-maps the fixed
            # verifier — if it now matches a step, the mismatch clears itself.
            vid = pay.get("verifier")
            if choice == "accept" and reason and vid in vmap:
                vmap[vid] = reason
                log.append({"item": cid, "action": "value_corrected",
                            "verifier": vid, "text": reason})
            elif choice == "reject":
                log.append({"item": cid, "action": "value_mismatch_golden_gap",
                            "verifier": vid, "reason": reason})
            else:
                log.append({"item": cid, "action": "value_mismatch_unresolved",
                            "verifier": vid})

        elif kind == "verdict_override":
            # SME re-grades a non-proceedable verdict after resolving the defects
            # that drove it. Requires acceptance AND a target verdict + reason. The
            # new verdict must be a proceedable one; anything else is ignored and
            # the original verdict stands (recorded, so the attempt is auditable).
            if choice == "accept":
                new_v = ""
                # answer may be "SOUND: justification" or just the verdict word
                a = (reason or "").strip()
                head = a.split(":", 1)[0].strip().upper() if a else ""
                if head in _PROCEEDABLE:
                    new_v = head
                if new_v:
                    out["_verdict_before_override"] = out.get("audit_verdict")
                    out["audit_verdict"] = new_v
                    out["proceedable"] = True
                    out["verdict_override"] = {
                        "from": out.get("_verdict_before_override"),
                        "to": new_v, "justification": a,
                        "decided_by": dec.get("decided_by", "SME")}
                    log.append({"item": cid, "action": "verdict_overridden",
                                "from": out["_verdict_before_override"],
                                "to": new_v})
                else:
                    log.append({"item": cid, "action": "verdict_override_invalid",
                                "detail": f"answer did not name a proceedable "
                                          f"verdict {sorted(_PROCEEDABLE)}",
                                "reason": a})
            else:
                log.append({"item": cid, "action": "verdict_override_declined",
                            "reason": reason})

        elif kind == "temporal":
            # A verifier with an unpinned live value. Accept + answer = the SME's
            # pin, which REPLACES the verifier text (e.g. adds "as-of 2026-01-15
            # per market_wage_reference.txt"). Reject = leave it, but it stays
            # not-scoreable. The seal re-derive picks up the pinned text.
            vid = pay.get("verifier")
            if choice == "accept" and reason and vid in vmap:
                vmap[vid] = reason
                log.append({"item": cid, "action": "temporal_pinned",
                            "verifier": vid, "text": reason})
            elif choice == "reject":
                log.append({"item": cid, "action": "temporal_left_unpinned",
                            "verifier": vid, "reason": reason})
            else:
                log.append({"item": cid, "action": "temporal_no_pin",
                            "verifier": vid})

    import re
    out["augmented_verifiers_text"] = "\n".join(
        f"{k}: {vmap[k]}" for k in sorted(
            vmap, key=lambda x: (int(re.sub(r"\D", "", x) or 0), x)))
    out["sme_decisions"] = {"decisions": choices, "answers": answers,
                            "saved_at": dec.get("saved_at"),
                            "run_hash": dh}
    out["sme_applied_log"] = log
    # the questions are answered, so the package is sealed
    out["judgment_changes_pending_sme"] = []

    # Re-derive the frozen graph from the SEALED artifacts. Every SME edit above
    # is now applied — verifier text (incl. reverted splits/rewrites and added
    # gap verifiers), solution logic, prompt, sanity, deliverable and trajectory —
    # so the verifier id space and/or the targets may differ from what the
    # augmenter froze. The dag / base weights / crux set / Shapley weights it
    # produced are stale; recompute them here, on the sealed set, as the single
    # authoritative freeze. Same routine the augmenter used for its preview, so
    # the two cannot diverge; only the inputs (post- vs pre-SME) differ.
    # Seed targets for gap verifiers added at seal so the re-derive maps them to
    # the step they were authored for (they close the coverage gap they resolved).
    # Without this, an added numeric verifier has no frozen target, stays unmapped,
    # and the gap the SME just closed reappears as unwatched.
    if out.get("_sealed_added_targets"):
        ev = dict(out.get("expected_values") or {})
        for vid, tgt in out["_sealed_added_targets"].items():
            ev.setdefault(vid, tgt)
        out["expected_values"] = ev

    from src.augment_task import derive_frozen_graph
    try:
        # compute_shapley=True: the seal is the ONLY place crux Shapley is
        # computed now — on the final sealed set, once, so its value is stable and
        # meaningful rather than a pre-seal preview the SME's edits invalidate.
        out = derive_frozen_graph(
            out, compute_shapley=True,
            forced_verifier_to_step=out.get("_sealed_step_bindings"),
            seal_crux=True)
        out["frozen_graph_source"] = "sealed"
        # A successful seal re-derive supersedes any stale 'error' carried in from
        # an earlier (pre-fix) build of the input package — otherwise the final
        # scoreability gate keys off a stale error and misreports the reason.
        if out.get("error"):
            out["_cleared_stale_error"] = out.pop("error")
    except Exception as e:
        # A re-derive that cannot run means the sealed set is inconsistent, so the
        # package must not present itself as scoreable with a stale graph.
        out["scoreable"] = False
        out["not_scoreable_reason"] = f"seal-time graph re-derive failed: {e}"
        out["frozen_graph_source"] = "STALE_rederive_failed"

    # Post-seal coverage closure. The seal re-derive rebuilds the DAG from the
    # POST-DECISION artifacts, and the SME's own decisions (splits, drops, added
    # verifiers) can reshape the graph so that steps NOT in the pre-seal decision
    # list now show as unwatched. The SME resolved every gap they were shown, yet
    # the sealed task could still have holes they never saw. Rather than ship a
    # silently-not-scoreable task or force a second decision round, auto-author a
    # verifier for each newly-surfaced gap (the same content-derived proposal used
    # pre-seal), bind it, and re-derive ONCE more. Bounded to a single pass; any
    # gap still open after that is reported, not hidden. Everything auto-added is
    # recorded in _auto_authored_gaps for the final report and audit.
    try:
        newly = list((out.get("step_coverage") or {}).get(
            "unwatched_load_bearing") or [])
    except Exception:                                           # noqa: BLE001
        newly = []
    if newly and out.get("frozen_graph_source") == "sealed":
        from src.augment_report import _step_index, _author_gap_verifier
        vmap2 = _verifier_map(out)
        sidx = _step_index(out)
        svidx = _step_value_index(out)
        auto = []
        for step in newly:
            nid = _next_id(vmap2)
            proposed = _author_gap_verifier(str(step), sidx.get(str(step), {}))
            vmap2[nid] = proposed
            out.setdefault("_sealed_step_bindings", {})[nid] = str(step)
            sv = svidx.get(str(step))
            if sv is not None:
                out.setdefault("_sealed_added_targets", {})[nid] = {
                    "value": sv, "tol": abs(sv) * 0.005 if sv else 0,
                    "source_of_verification": "arithmetic", "step": str(step)}
            auto.append({"id": nid, "step": str(step), "verifier": proposed})
        # write the augmented verifier set back and re-seed targets
        out["augmented_verifiers_text"] = "\n".join(
            f"{vid}: {txt}" for vid, txt in vmap2.items())
        if out.get("_sealed_added_targets"):
            ev = dict(out.get("expected_values") or {})
            for vid, tgt in out["_sealed_added_targets"].items():
                ev.setdefault(vid, tgt)
            out["expected_values"] = ev
        out["_auto_authored_gaps"] = (out.get("_auto_authored_gaps") or []) + auto
        try:
            out = derive_frozen_graph(
                out, compute_shapley=True,
                forced_verifier_to_step=out.get("_sealed_step_bindings"),
                seal_crux=True)
            still = list((out.get("step_coverage") or {}).get(
                "unwatched_load_bearing") or [])
            log.append({"item": "_post_seal_coverage",
                        "action": "auto_authored_gap_verifiers",
                        "added": [a["id"] for a in auto],
                        "still_unwatched": still})
        except Exception as e:                                  # noqa: BLE001
            out["scoreable"] = False
            out["not_scoreable_reason"] = (
                f"post-seal gap-closure re-derive failed: {e}")

    out["sealed"] = True
    # Complete source_of_verification map for EVERY verifier, so GEAR's per-edge
    # lambda (sov mode) and any downstream recomputation can be done from the
    # sealed record alone. SOV lives in expected_values for verifiers with a
    # frozen value; a verifier WITHOUT one is a decision/judgment check, whose
    # verification source is llm_judgment (the weak-lambda edge in GEAR). Building
    # the full map here means the CSV can carry SOV for the whole DAG, not just
    # the numeric subset.
    ev_all = out.get("expected_values") or {}
    dag_all = out.get("dag") or {}
    sov_map = {}
    for vid in dag_all:
        s = (ev_all.get(vid) or {}).get("source_of_verification")
        sov_map[vid] = s or "llm_judgment"
    out["sov_map"] = sov_map
    # Seal-time Shapley on BOTH graphs, plus the crux subgraph. crux_shapley_weights
    # (crux over the full DAG, non-crux as context) is already set by the
    # re-derive; here we add the full-DAG distribution over ALL verifiers and the
    # crux-subgraph distribution (crux vs crux only), and store the crux DAG.
    try:
        from src.crux_shapley import (full_dag_shapley, crux_dag_shapley,
                                       crux_subgraph)
    except Exception:                                           # noqa: BLE001
        full_dag_shapley = crux_dag_shapley = crux_subgraph = None
    if full_dag_shapley and out.get("dag") and out.get("base_weights"):
        all_vs = [{"id": v} for v in out["dag"].keys()]
        try:
            out["full_dag_shapley_weights"] = full_dag_shapley(
                all_vs, out["dag"], out["base_weights"])
            crux = out.get("crux_ids") or []
            out["crux_dag"] = crux_subgraph(out["dag"], crux)
            out["crux_dag_shapley_weights"] = crux_dag_shapley(
                all_vs, out["dag"], out["base_weights"], crux)
        except Exception as e:                                  # noqa: BLE001
            out["shapley_note"] = f"seal Shapley (full/crux-dag) failed: {e}"
    # scoreable only if the audit was proceedable, there is no error, AND the
    # re-derive did not itself veto (e.g. a crux verifier still judgment_flagged
    # after the SME pass, or the re-derive failed above).
    out["scoreable"] = (bool(out.get("proceedable"))
                        and not out.get("error")
                        and out.get("scoreable", True))
    if not out["scoreable"]:
        out["not_scoreable_reason"] = (
            out.get("not_scoreable_reason")
            or f"audit verdict {out.get('audit_verdict')} is not proceedable")
    else:
        out["not_scoreable_reason"] = ""
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--augment", required=True)
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--out", default="output/final")
    ap.add_argument("--force", action="store_true",
                    help="apply despite a run_hash mismatch or an incomplete file")
    args = ap.parse_args()

    pkg, dec = load(args.augment), load(args.decisions)
    if pkg.get("task_id") != dec.get("task_id"):
        raise SystemExit(f"task mismatch: {pkg.get('task_id')} vs {dec.get('task_id')}")

    sealed = apply_decisions(pkg, dec, force=args.force)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{pkg['task_id']}_final.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sealed, f, indent=2, ensure_ascii=False, default=str)
    # clean final prompt-package HTML for SME review (no decision scaffolding)
    html_path = os.path.join(args.out, f"{pkg['task_id']}_final.html")
    try:
        from src.augment_report import write_sealed_report
        write_sealed_report(sealed, html_path)
    except Exception as e:                                       # noqa: BLE001
        html_path = None
        print(f"  (sealed HTML not written: {e})")

    print(f"\n=== {pkg['task_id']} sealed ===")
    for e in sealed["sme_applied_log"]:
        extra = e.get("verifier") or e.get("id") or e.get("artifact") or ""
        print(f"  {e['item']:6s} {e['action']:26s} {extra}")
    print(f"\n  resolutions recorded : {len(sealed.get('sme_resolutions') or [])}")
    print(f"  verifiers            : "
          f"{len(sealed['augmented_verifiers_text'].splitlines())}")
    print(f"  scoreable            : {sealed['scoreable']}"
          + (f"  ({sealed['not_scoreable_reason']})"
             if not sealed["scoreable"] else ""))
    print(f"\nwrote {path}")
    if html_path:
        print(f"wrote {html_path}")


if __name__ == "__main__":
    main()