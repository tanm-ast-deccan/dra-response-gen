#!/usr/bin/env python3
"""Write sealed packages into a COPY of the original authoring csv.

    python build_final_csv.py --csv ./SME_data/Non_IB_Prompt_Eval_Approved_tasks.csv \
        --final output_Fin_7/augmented --out output_Fin_7/final_tasks.csv

Every original column and every original row is preserved. For each task that has
a *_final.json, the corrected artifacts are written into appended columns; tasks
with no sealed package keep their row with those columns blank, so the sheet stays
a complete picture of the batch rather than only the finished part.

The corrected columns are placed next to their originals — Revised Prompt beside
Prompt — because a reviewer reads them as a pair.
"""
import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.auditor import build_header_map, get_field, read_task_csv

def _chain_weights(depths: dict) -> dict:
    """CHAIN weights: (depth+1)^alpha, alpha=2, normalized to sum 1.0. Deeper
    verifiers (further along the derivation) weigh more. Recomputable downstream
    from Verifier Depths JSON; precomputed here for convenience."""
    if not depths:
        return {}
    raw = {v: (int(d) + 1) ** 2 for v, d in depths.items()}
    tot = sum(raw.values()) or 1.0
    return {v: w / tot for v, w in raw.items()}


def _trajectory_text(p: dict) -> str:
    """The golden trajectory as one readable line per step.

    This is the SFT target and what CHAIN rests on, so it belongs in the sheet
    rather than only in the html report. One line per step keeps a task on one row
    while staying parseable: fields are pipe-separated in a fixed order.
    """
    steps = p.get("corrected_claim_verdicts") or []
    judg = p.get("judgment_steps") or []
    graph = p.get("step_graph") or {}
    v2s = p.get("verifier_to_step") or {}
    watched = {}
    for vid, sid in v2s.items():
        watched.setdefault(sid, []).append(vid)
    feeds = {}
    for child, parents in graph.items():
        for par in parents:
            feeds.setdefault(par, []).append(child)

    lines = []
    for c in steps:
        sid = c.get("id")
        parents = list(dict.fromkeys(
            x.get("from_claim") for x in (c.get("input_provenance") or [])
            if x.get("from_claim")))
        lines.append(" | ".join([
            str(sid), str(c.get("label", "")),
            f"{c.get('operation','')} = {c.get('recomputed')}",
            "from: " + (", ".join(parents) or "source files"),
            "feeds: " + (", ".join(feeds.get(sid, [])) or "TERMINAL"),
            "watched by: " + (", ".join(watched.get(sid, [])) or "NOBODY"),
            str(c.get("status", "")),
        ]))
    for j in judg:
        jid = j.get("id")
        lines.append(" | ".join([
            str(jid), str(j.get("question", "")), "judgement",
            "consumes: " + (", ".join(j.get("consumes") or []) or "nothing"),
            "feeds: " + (", ".join(feeds.get(jid, [])) or "TERMINAL"),
            "watched by: " + (", ".join(watched.get(jid, [])) or "NOBODY"),
            "ruling: " + str(j.get("ruling", "")),
        ]))
    return "\n".join(lines)


def _trajectory_health(p: dict) -> str:
    h = p.get("step_graph_health") or {}
    if not h:
        return ""
    return (f"{h.get('n_nodes', 0)} steps, {h.get('n_edges', 0)} dependencies, "
            f"{h.get('connected', 0)} connected to a terminal; "
            f"terminals: {', '.join(h.get('terminals') or []) or 'none'}; "
            f"cycles: {len(h.get('cycles') or [])}; "
            f"is_derivation: {h.get('is_derivation')}")


def _unwatched_steps(p: dict) -> str:
    """Steps no verifier checks — work a response can get wrong unopposed."""
    sc = p.get("step_coverage") or {}
    if not sc:
        return ""
    parts = []
    if sc.get("unwatched_terminals"):
        parts.append("terminal: " + ", ".join(sc["unwatched_terminals"]))
    if sc.get("unwatched_load_bearing"):
        parts.append("load-bearing: " + ", ".join(sc["unwatched_load_bearing"]))
    if sc.get("dead_weight"):
        parts.append("dead weight: " + ", ".join(sc["dead_weight"]))
    return "; ".join(parts)


#: (column, how to get it from the sealed package)
FINAL_COLUMNS = [
    ("Corrected Prompt",         lambda p: p.get("corrected_prompt", "")),
    ("Corrected Sanity Check",   lambda p: p.get("corrected_sanity_check", "")),
    ("Corrected Solution Logic", lambda p: p.get("corrected_solution_logic", "")),
    ("Augmented Verifiers",      lambda p: p.get("augmented_verifiers_text", "")),
    ("Golden Deliverable",       lambda p: p.get("gold_deliverable_text", "")),
    ("Golden Trajectory",        _trajectory_text),
    ("Trajectory Health",        _trajectory_health),
    ("Unwatched Steps",          _unwatched_steps),
    ("Trajectory JSON",          lambda p: json.dumps({
        "steps": p.get("corrected_claim_verdicts") or [],
        "judgment_steps": p.get("judgment_steps") or [],
        "step_graph": p.get("step_graph") or {},
        "verifier_to_step": p.get("verifier_to_step") or {}},
        ensure_ascii=False)),
    ("Expected Values JSON",     lambda p: json.dumps(p.get("expected_values") or {},
                                                      ensure_ascii=False)),
    ("DAG JSON",                 lambda p: json.dumps(p.get("dag") or {},
                                                      ensure_ascii=False)),
    ("Crux Verifier IDs",        lambda p: ",".join(p.get("crux_ids") or [])),
    ("Base Weights JSON",        lambda p: json.dumps(p.get("base_weights") or {},
                                                      ensure_ascii=False)),
    ("SOV Map JSON",             lambda p: json.dumps(
        p.get("sov_map") or {v: (ev or {}).get("source_of_verification",
                                               "llm_judgment")
                             for v, ev in (p.get("expected_values") or {}).items()},
        ensure_ascii=False)),
    ("Verifier Depths JSON",     lambda p: json.dumps(p.get("depths") or {},
                                                      ensure_ascii=False)),
    ("CHAIN Weights JSON",       lambda p: json.dumps(
        _chain_weights(p.get("depths") or {}), ensure_ascii=False)),
    ("Crux Shapley Weights JSON", lambda p: json.dumps(
        p.get("crux_shapley_weights") or {}, ensure_ascii=False)),
    ("Full-DAG Shapley Weights JSON", lambda p: json.dumps(
        p.get("full_dag_shapley_weights") or {}, ensure_ascii=False)),
    ("Crux-DAG Shapley Weights JSON", lambda p: json.dumps(
        p.get("crux_dag_shapley_weights") or {}, ensure_ascii=False)),
    ("Crux DAG JSON",             lambda p: json.dumps(
        p.get("crux_dag") or {}, ensure_ascii=False)),
    ("Verdict Override",          lambda p: (
        f"{(p.get('verdict_override') or {}).get('from','')} -> "
        f"{(p.get('verdict_override') or {}).get('to','')}: "
        f"{(p.get('verdict_override') or {}).get('justification','')}"
        if p.get("verdict_override") else "")),
    ("Auto-Authored Gap Verifiers", lambda p: "; ".join(
        f"{a.get('id')} watches {a.get('step')}"
        for a in (p.get("_auto_authored_gaps") or []))),
    ("Audit Verdict",            lambda p: p.get("audit_verdict", "")),
    ("Scoreable",               lambda p: "YES" if p.get("scoreable") else "NO"),
    ("Not Scoreable Reason",     lambda p: p.get("not_scoreable_reason", "")),
    ("SME Resolutions",          lambda p: "\n\n".join(
        f"[{r.get('artifact')} — {r.get('location')}]\nQ: {r.get('question')}\n"
        f"A: {r.get('answer')}" for r in (p.get("sme_resolutions") or []))),
    ("Verifiers Without Target", lambda p: ",".join(
        p.get("verifiers_without_target") or [])),
    ("Input Files Read",         lambda p: ",".join(
        (p.get("input_coverage") or {}).get("files_read") or [])),
    ("Sealed",                  lambda p: "YES" if p.get("sealed") else "NO"),
    ("Run Hash",                lambda p: p.get("run_hash", "")),
    ("Trap Anchors",             lambda p: ",".join(p.get("crux_anchors_trap") or [])),
    ("Expert Anchors",           lambda p: ",".join(p.get("crux_anchors_expert") or [])),
    ("Final Answer Verifiers",   lambda p: ",".join(
        p.get("final_answer_verifiers") or [])),
    ("Verifier Splits Applied",  lambda p: "; ".join(
        f"{x.get('parent')} -> {', '.join(x.get('children') or [])}"
        for x in (p.get("verifier_splits_applied") or []))),
    ("Verifier Rewrites Applied", lambda p: ",".join(
        p.get("verifier_rewrites_applied") or [])),
    ("Property Failures",        lambda p: json.dumps(
        (p.get("verifier_audit") or {}).get("fails_by_property") or {},
        ensure_ascii=False)),
    ("Skipped Inputs",           lambda p: ",".join(p.get("skipped_inputs") or [])),
    ("Input Coverage Complete",  lambda p: str(
        (p.get("input_coverage") or {}).get("complete"))),
    ("Decisions Saved At",       lambda p: (p.get("sme_decisions") or {}).get(
        "saved_at", "")),
]


def load_sealed(paths_or_dirs):
    """{task_id: package} from *_final.json files, and any that failed to load."""
    out, bad = {}, []
    files = []
    for p in paths_or_dirs:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "*_final.json")))
        else:
            files.append(p)
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                pkg = json.load(fh)
        except Exception as e:
            bad.append((os.path.basename(f), str(e)))
            continue
        tid = str(pkg.get("task_id") or "").strip()
        if not tid:
            bad.append((os.path.basename(f), "no task_id"))
            continue
        if not pkg.get("sealed"):
            # a package that has not been through apply_decisions still carries
            # unresolved questions; writing it would look finished when it is not
            bad.append((os.path.basename(f), "not sealed — run apply_decisions first"))
            continue
        out[tid] = pkg
    return out, bad


#: A narrower sheet for review and hand-off: the original columns, the corrected
#: artifacts, the golden, the trajectory in readable form, and the scoring
#: structure. The full sheet keeps everything; this one keeps what a person reads.
SLIM_COLUMNS = [
    "task_id", "prompt_id", "Domain", "Primary Prompt Type",
    "Secondary Prompt Type", "prompt", "Sanity Check", "Solution Logic",
    "drive_url", "Verifiers",
    "Estimated Time",
    "Corrected Prompt", "Corrected Sanity Check", "Corrected Solution Logic",
    "Augmented Verifiers", "Golden Deliverable",
    "Golden Trajectory", "Trajectory Health",
    "DAG JSON", "Crux Verifier IDs", "Base Weights JSON",
    "Crux Shapley Weights JSON", "Crux-DAG Shapley Weights JSON",
    "Verdict Override", "Audit Verdict", "Scoreable",
]


#: Columns the RESPONSE-GENERATION pipeline reads (it uses get_field aliases:
#: prompt, sanity_check, solution_logic, verifiers, drive_link, prompt_type). The
#: response-ready sheet puts the CORRECTED / augmented values into those canonical
#: names and carries drive_link through, so the pipeline generates responses
#: against the SEALED task, not the pre-correction original — and can still fetch
#: the input files. Only scoreable, sealed tasks belong here.
def _response_ready_row(original: dict, pkg: dict, hmap: dict) -> dict:
    """Build one response-ready row for the dra_harness response-generation
    pipeline. Verified against dra_harness/csv_loader.py: the harness reads only
    task_id, prompt, drive_url (aliases include drive_link), and optionally
    output_format + sme_name — it does NOT read sanity_check / solution_logic /
    verifiers (those are for scoring, not generation). The harness has no
    'Corrected Prompt' alias, so the CORRECTED prompt must occupy the literal
    'prompt' column for generation to run against the sealed task. Golden and
    verifiers are carried as extra columns for reference/scoring; the harness
    ignores them."""
    fmt = (pkg or {}).get("gold_deliverable_format", "")
    return {
        "task_id": get_field(original, hmap, "task_id"),
        "prompt": (pkg or {}).get("corrected_prompt")
        or get_field(original, hmap, "prompt"),
        "drive_link": get_field(original, hmap, "drive_link"),
        "output_format": fmt,
        "prompt_type": get_field(original, hmap, "prompt_type"),
        # reference-only (harness ignores; useful for the scoring side)
        "sanity_check": (pkg or {}).get("corrected_sanity_check")
        or get_field(original, hmap, "sanity_check"),
        "solution_logic": (pkg or {}).get("corrected_solution_logic")
        or get_field(original, hmap, "solution_logic"),
        "verifiers": (pkg or {}).get("augmented_verifiers_text")
        or get_field(original, hmap, "verifiers"),
        "golden_deliverable": (pkg or {}).get("gold_deliverable_text", ""),
        "audit_verdict": (pkg or {}).get("audit_verdict", ""),
        "scoreable": "YES" if (pkg or {}).get("scoreable") else "NO",
    }


RESPONSE_READY_FIELDS = [
    # harness-read columns first (task_id, prompt, drive_link, output_format)
    "task_id", "prompt", "drive_link", "output_format", "prompt_type",
    # reference columns for the scoring side (harness ignores these)
    "sanity_check", "solution_logic", "verifiers", "golden_deliverable",
    "audit_verdict", "scoreable",
]


def _write(path, fields, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if os.path.exists(path):
        import shutil
        shutil.copyfile(path, path + ".bak")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="the original authoring csv")
    ap.add_argument("--final", nargs="+", required=True,
                    help="*_final.json files, or a directory of them")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only-sealed", action="store_true",
                    help="write only rows that have a sealed package")
    ap.add_argument("--slim", default=None,
                    help="also write a narrower sheet here (for human review)")
    ap.add_argument("--response-ready", default=None,
                    help="also write a CSV the response-generation pipeline can "
                         "consume directly: canonical columns (prompt, "
                         "sanity_check, solution_logic, verifiers) hold the "
                         "corrected/sealed values, drive_link preserved. Only "
                         "scoreable sealed tasks are included.")
    args = ap.parse_args()

    headers, rows = read_task_csv(args.csv)
    hmap = build_header_map(headers)
    sealed, bad = load_sealed(args.final)

    for name, why in bad:
        print(f"  !! skipped {name}: {why}")
    if not sealed:
        raise SystemExit("no sealed packages found — nothing to write")

    fields = list(headers) + [c for c, _ in FINAL_COLUMNS if c not in headers]
    out_rows, written, blank = [], [], []
    response_rows = []
    for r in rows:
        tid = get_field(r, hmap, "task_id")
        pkg = sealed.get(tid)
        if pkg is None:
            if args.only_sealed:
                continue
            blank.append(tid)
            out_rows.append(dict(r))
            continue
        row = dict(r)
        for col, fn in FINAL_COLUMNS:
            try:
                row[col] = fn(pkg)
            except Exception as e:                              # noqa: BLE001
                row[col] = f"(error: {e})"
        out_rows.append(row)
        written.append(tid)
        # response-ready: only scoreable sealed tasks are fit to generate against
        if pkg.get("scoreable"):
            response_rows.append(_response_ready_row(r, pkg, hmap))

    # This REBUILDS from the original csv plus whatever is sealed, so it is safe to
    # re-run after each task. But that also means a *_final.json going missing
    # silently blanks its row, so keep one step back.
    _write(args.out, fields, out_rows)

    slim_path = args.slim or (os.path.splitext(args.out)[0] + "_slim.csv")
    missing = [c for c in SLIM_COLUMNS if c not in fields]
    slim_fields = [c for c in SLIM_COLUMNS if c in fields]
    _write(slim_path, slim_fields, out_rows)

    if args.response_ready:
        _write(args.response_ready, RESPONSE_READY_FIELDS, response_rows)
        print(f"\n{len(response_rows)} scoreable task(s) -> {args.response_ready}  "
              f"(response-ready: corrected prompt/logic/verifiers in canonical "
              f"columns, drive_link preserved)")
        skipped = len(written) - len(response_rows)
        if skipped:
            print(f"  ({skipped} sealed-but-not-scoreable task(s) excluded from "
                  f"response-ready)")

    print(f"\n{len(out_rows)} row(s) -> {slim_path}  ({len(slim_fields)} columns)"
          + (f"   [not present in this batch: {missing}]" if missing else ""))
    print(f"{len(out_rows)} row(s) -> {args.out}"
          + ("  (previous version saved as "
             f"{os.path.basename(args.out)}.bak)" if os.path.exists(args.out + ".bak")
             else ""))
    print(f"  {len(fields)} columns ({len(headers)} original "
          f"+ {len(fields) - len(headers)} appended)")
    print(f"  filled  : {written}")
    if blank:
        print(f"  no sealed package yet ({len(blank)}): {blank[:8]}"
              + ("..." if len(blank) > 8 else ""))
    for tid in written:
        p = sealed[tid]
        nv = len((p.get("augmented_verifiers_text") or "").splitlines())
        print(f"\n  {tid}: {nv} verifiers, {len(p.get('crux_ids') or [])} crux, "
              f"scoreable={p.get('scoreable')}, "
              f"{len(p.get('sme_resolutions') or [])} resolution(s)")
        if not p.get("scoreable"):
            print(f"    NOT SCOREABLE: {p.get('not_scoreable_reason','')[:90]}")


if __name__ == "__main__":
    main()