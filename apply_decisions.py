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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _items(pkg):
    """Rebuild the same id space the report used. Must stay in step with
    augment_report._decision_items — the ids are the contract between them."""
    out = []
    for i, c in enumerate(pkg.get("changes_applied") or []):
        out.append({"id": f"chg{i}", "kind": "change", "payload": c})
    for i, c in enumerate(pkg.get("judgment_changes_pending_sme") or []):
        out.append({"id": f"q{i}", "kind": "question", "payload": c})
    va = pkg.get("verifier_audit") or {}
    for i, v in enumerate(va.get("verifiers") or []):
        if (v.get("rewrite") or "").strip():
            out.append({"id": f"rw{i}", "kind": "rewrite", "payload": v})
    for i, x in enumerate(pkg.get("verifier_splits_applied") or []):
        out.append({"id": f"sp{i}", "kind": "split", "payload": x})
    for i, g in enumerate(va.get("coverage_gaps") or []):
        out.append({"id": f"gap{i}", "kind": "gap", "payload": g})
    return out


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
                key = {"solution_logic": "corrected_solution_logic",
                       "sanity_check": "corrected_sanity_check",
                       "prompt": "corrected_prompt"}.get(art)
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
                key = {"solution_logic": "corrected_solution_logic",
                       "sanity_check": "corrected_sanity_check",
                       "prompt": "corrected_prompt"}.get(art)
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
            if choice == "accept":
                nid = _next_id(vmap)
                vmap[nid] = pay.get("proposed_verifier", "")
                log.append({"item": cid, "action": "verifier_added", "id": nid,
                            "step": pay.get("step")})
            elif choice == "other":
                nid = _next_id(vmap)
                vmap[nid] = reason
                log.append({"item": cid, "action": "verifier_added_sme_text",
                            "id": nid, "step": pay.get("step")})
            else:
                log.append({"item": cid, "action": "gap_left_open",
                            "step": pay.get("step"), "reason": reason})

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
    out["sealed"] = True
    out["scoreable"] = bool(out.get("proceedable")) and not out.get("error")
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


if __name__ == "__main__":
    main()