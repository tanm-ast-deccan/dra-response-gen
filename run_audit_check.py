#!/usr/bin/env python3
"""Run the real auditor on one or two tasks and report what Phases 1 and 2 did.

    python run_audit_check.py --csv ./SME_data/Augmented_tasks_SME_Delivery_-_sme_shortlist_with_links.csv \
        --task tsk_1874355798 --out ./audit_check

    # two tasks, and the frequency question
    python run_audit_check.py --csv ... --task tsk_1874355798 --task tsk_4902728654

This is a DIAGNOSTIC, not a pipeline stage. It makes the two audit calls, caches
the result, and answers the five things no offline test can:

  1. does call 2 actually return corrected_claims, and how many?
  2. what changed between the first and second verification pass?
  3. is from_claim populated — the field the whole chain rests on?
  4. did it emit judgment steps, and do they consume anything?
  5. does the gate pass on the CORRECTED derivation?

Plus the deferred frequency question: how often is a mismatch a scale defect?

Everything is cached to {task}_auditcheck.json, so a second run costs nothing
and the artifact can be diffed against the previous one.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.auditor import audit_task, build_header_map, get_field, HeaderError
from src.config import configure_api


def read_rows(path):
    """Banner-row tolerant; see src.auditor.read_task_csv."""
    return read_task_csv(path)


def _pct(n, d):
    return f"{n}/{d}" + (f" ({n/d:.0%})" if d else "")


def report(task_id, a):
    """a is the AuditResult, already computed."""
    first = a.claim_verdicts or []
    second = a.corrected_claim_verdicts or []
    line = "=" * 78

    print(f"\n{line}\n{task_id}\n{line}")
    print(f"verdict            : {a.verdict}   proceedable={a.proceedable}")
    print(f"primary_reason     : {a.primary_reason[:100]}")

    cov = a.input_coverage or {}
    print(f"\ninput coverage     : complete={cov.get('complete')}  "
          f"read={len(cov.get('files_read') or [])}  "
          f"skipped={cov.get('files_skipped') or []}  "
          f"truncated={cov.get('files_truncated') or []}")

    # --- 1. did call 2 rebuild the derivation? ---
    print(f"\n1  DERIVATION REBUILT?")
    print(f"   claims, pass 1   : {len(first)}")
    print(f"   claims, pass 2   : {len(second)}"
          + ("   <-- call 2 returned NONE; the chain does not exist"
             if not second else ""))
    if a.findings_note:
        print(f"   note             : {a.findings_note}")
    if second:
        ids1, ids2 = {c['id'] for c in first}, {c['id'] for c in second}
        if ids1 - ids2:
            print(f"   DROPPED in pass 2: {sorted(ids1 - ids2)}")
        if ids2 - ids1:
            print(f"   ADDED in pass 2  : {sorted(ids2 - ids1)}")

    # --- 2. what did correction change? ---
    print(f"\n2  WHAT THE CORRECTION CHANGED")
    def counts(vs):
        c = {}
        for v in vs:
            c[v['status']] = c.get(v['status'], 0) + 1
        return c
    print(f"   pass 1 statuses  : {counts(first)}")
    print(f"   pass 2 statuses  : {counts(second) if second else '(none)'}")
    if second:
        by1 = {c['id']: c for c in first}
        moved = [(c['id'], by1[c['id']]['status'], c['status'])
                 for c in second
                 if c['id'] in by1 and by1[c['id']]['status'] != c['status']]
        for cid, was, now in moved[:12]:
            print(f"     {cid:8s} {was:16s} -> {now}")
        if not moved:
            print("     (no status changed — correction may not have landed)")

    # --- 3. from_claim, the field the chain rests on ---
    print(f"\n3  from_claim COVERAGE")
    src = second or first
    tot_in = sum(len(c.get('input_provenance') or []) for c in src)
    with_fc = sum(1 for c in src for p in (c.get('input_provenance') or [])
                  if p.get('from_claim'))
    claims_linked = sum(1 for c in src
                        if any(p.get('from_claim')
                               for p in (c.get('input_provenance') or [])))
    print(f"   inputs with a parent named : {_pct(with_fc, tot_in)}")
    print(f"   claims with >=1 parent     : {_pct(claims_linked, len(src))}")
    if src and with_fc == 0:
        print("   !! NONE populated — there is no chain, only a list. Phase 3")
        print("      has nothing to derive a dependency graph from.")

    # --- 4. judgment steps ---
    js = a.judgment_steps or []
    print(f"\n4  JUDGMENT STEPS      : {len(js)}")
    for j in js[:6]:
        print(f"     {j.get('id','?'):4s} consumes={j.get('consumes')}  "
              f"{str(j.get('question',''))[:60]}")
    orphan = [j.get('id') for j in js if not (j.get('consumes') or [])]
    if orphan:
        print(f"   !! consuming nothing: {orphan} — not part of the derivation")

    # --- 5. the gate ---
    g = a.gate or {}
    print(f"\n5  GATE               : passed={g.get('passed')}  "
          f"derivation_available={g.get('derivation_available')}")
    if g.get("reason"):
        print(f"   {g['reason']}")
    for b in (g.get("blocking") or [])[:10]:
        print(f"     BLOCK {b['id']:8s} {b['status']:16s} {b.get('label','')[:44]}")
    for w in (g.get("warnings") or [])[:6]:
        print(f"     warn  {w['id']:8s} {w['status']}")

    # --- 6. frequency: the deferred question ---
    summ = a.corrected_arithmetic_summary or a.arithmetic_summary or {}
    repairs = [c for c in (a.changes or [])
               if c.get("source") == "arithmetic_verifier"]
    print(f"\n6  SCALE FREQUENCY")
    print(f"   summary          : {summ}")
    print(f"   auto-repairs     : {len(repairs)}")
    for r in repairs[:8]:
        print(f"     {r['location'][:40]:42s} {r['old']}  ->  {r['new']}")

    # --- 7. the three corrected artifacts ---
    print(f"\n7  CORRECTED ARTIFACTS (empty = unchanged)")
    for name, txt in (("solution_logic", a.corrected_solution_logic),
                      ("prompt", a.corrected_prompt),
                      ("sanity_check", a.corrected_sanity_check)):
        print(f"   {name:16s} {len(txt or '')} chars")
    va = getattr(a, "verifier_change_audit", None) or {}
    if va.get("corrected"):
        print(f"   verifiers        {va['n_before']} -> {va['n_after']}  "
              f"edited={va['changed']}")
        for k, label in (("undeclared_edits", "!! edited but not declared"),
                         ("declared_not_done", "!! declared but not done"),
                         ("judgment_resolved_silently",
                          "!! JUDGMENT resolved without asking"),
                         ("dropped", "!! verifiers dropped"),
                         ("added", "   verifiers added")):
            if va.get(k):
                print(f"     {label}: {va[k]}")
    elif va:
        print(f"   verifiers        {va.get('note', 'not corrected')}")

    mech = sum(1 for c in (a.changes or []) if c.get("type") == "MECHANICAL")
    judg = [c for c in (a.changes or []) if c.get("type") == "JUDGMENT_REQUIRED"]
    print(f"   changes          : {mech} mechanical, {len(judg)} judgment")
    for c in judg[:4]:
        print(f"     Q: {str(c.get('sme_question',''))[:88]}")

    return {
        "task_id": task_id, "verdict": a.verdict, "proceedable": a.proceedable,
        "n_claims_pass1": len(first), "n_claims_pass2": len(second),
        "from_claim_inputs": with_fc, "total_inputs": tot_in,
        "n_judgment_steps": len(js),
        "gate_passed": g.get("passed"),
        "gate_reason": g.get("reason", ""),
        "scale_repairs": len(repairs),
        "coverage_complete": cov.get("complete"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--task", action="append", required=True,
                    help="repeatable; 1-2 recommended")
    ap.add_argument("--out", default="./audit_check")
    ap.add_argument("--model", default="")
    ap.add_argument("--no-files", action="store_true")
    ap.add_argument("--force", action="store_true", help="ignore the cache")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    headers, rows = read_rows(args.csv)
    try:
        hmap = build_header_map(headers)
    except HeaderError as e:
        raise SystemExit(f"header problem: {e}")
    if not args.model:
        from src.prompt_evaluator import DEFAULT_JUDGE_MODEL
        args.model = DEFAULT_JUDGE_MODEL
    configure_api()

    summaries = []
    for tid in args.task:
        match = [r for r in rows if get_field(r, hmap, "task_id") == tid]
        if not match:
            print(f"!! {tid} not in the csv"); continue
        row = match[0]
        cache = os.path.join(args.out, f"{tid}_auditcheck.json")

        if os.path.exists(cache) and not args.force:
            print(f"\n{tid}: using cached audit ({cache}). --force to re-run.")
            d = json.load(open(cache, encoding="utf-8"))
            class _A:  # replay the cached result through the same reporter
                pass
            a = _A()
            for k, v in d["audit"].items():
                setattr(a, k, v)
        else:
            files_text, files_names, skipped = "", [], []
            if not args.no_files:
                dl = get_field(row, hmap, "drive_link")
                if dl.strip():
                    try:
                        from src.gdrive_raw_fetcher import fetch_gdrive_folder_raw
                        files_text, files_names, skipped = fetch_gdrive_folder_raw(dl)
                        print(f"{tid}: fetched {len(files_names)} input file(s)"
                              + (f", SKIPPED {skipped}" if skipped else ""))
                    except Exception as e:
                        print(f"{tid}: !! drive fetch failed: {e}")
            print(f"{tid}: running audit_task ({args.model}) ...")
            a = audit_task(row, hmap, input_files_text=files_text,
                           input_files_names=files_names,
                           model_name=args.model, skipped_inputs=skipped)
            keep = ("verdict", "primary_reason", "proceedable", "claim_verdicts",
                    "corrected_claim_verdicts", "arithmetic_summary",
                    "corrected_arithmetic_summary", "judgment_steps", "gate",
                    "changes", "findings", "leakage_findings", "input_coverage",
                    "corrected_solution_logic", "corrected_prompt",
                    "corrected_sanity_check", "corrected_verifiers",
                    "verifier_change_audit", "malformed_claims",
                    "findings_note")
            json.dump({"task_id": tid,
                       "audit": {k: getattr(a, k, None) for k in keep}},
                      open(cache, "w", encoding="utf-8"),
                      indent=2, ensure_ascii=False, default=str)
            print(f"{tid}: cached -> {cache}")

        for k in ("findings_note", "corrected_prompt", "corrected_sanity_check",
                  "corrected_solution_logic", "corrected_verifiers",
                  "primary_reason"):
            if not hasattr(a, k):
                setattr(a, k, "")
        summaries.append(report(tid, a))

    print("\n" + "=" * 78)
    print("ACROSS TASKS")
    print("=" * 78)
    for s in summaries:
        print(f"  {s['task_id']:22s} verdict={s['verdict']:14s} "
              f"gate={str(s['gate_passed']):5s} "
              f"claims {s['n_claims_pass1']}->{s['n_claims_pass2']}  "
              f"from_claim {s['from_claim_inputs']}/{s['total_inputs']}  "
              f"judg={s['n_judgment_steps']}  scale_repairs={s['scale_repairs']}")
    json.dump(summaries, open(os.path.join(args.out, "summary.json"), "w"),
              indent=2, default=str)
    print(f"\nwrote {os.path.join(args.out, 'summary.json')}")


if __name__ == "__main__":
    main()