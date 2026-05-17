#!/usr/bin/env python3
"""
diagnose_run.py — Batch run diagnostic tool for DRA benchmark results.

Usage:
    python diagnose_run.py --dispatch-dir ./dispatch_results_22_04_2026
    python diagnose_run.py --tasks-dir ./results_22_04_2026/tasks
    python diagnose_run.py --dispatch-dir ./dispatch_results_22_04_2026 --show-pass
    python diagnose_run.py --dispatch-dir ./dispatch_results_22_04_2026 --agent gemini
    python diagnose_run.py --dispatch-dir ./dispatch_results_22_04_2026 --summary-only
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict


# ── Colour helpers ────────────────────────────────────────────────────────────
def _c(code, text): return f"\033[{code}m{text}\033[0m"
RED    = lambda t: _c("31", t)
GREEN  = lambda t: _c("32", t)
YELLOW = lambda t: _c("33", t)
CYAN   = lambda t: _c("36", t)
BOLD   = lambda t: _c("1",  t)
DIM    = lambda t: _c("2",  t)


# ── Categorise failure reason ─────────────────────────────────────────────────
def categorise(error: str) -> str:
    e = (error or "").lower()
    if "stuck in 'in_progress'" in e or "staleinteractionerror" in e:
        return "Gemini API timeout (in_progress)"
    if "0 input_tokens" in e or "0 tokens" in e:
        return "Gemini API timeout (0 tokens)"
    if "invalid_prompt" in e or "400" in e and "invalid" in e:
        return "Content policy 400"
    if "there was a problem" in e and "400" in e:
        return "Content policy 400 (Gemini)"
    if "file_output_not_produced" in e or "did not generate the required file" in e:
        return "File output not produced (model skipped)"
    if "file generation failed" in e:
        sub = ""
        if "no such file" in e or "filenotfounderror" in e:
            sub = " — file not found"
        elif "keyerror" in e:
            sub = " — wrong column name"
        elif "assertionerror" in e:
            sub = " — wrong header arg"
        elif "attributeerror" in e:
            sub = " — wrong API usage"
        elif "modulenotfounderror" in e or "no module" in e:
            lib = ""
            for l in ["sklearn", "statsmodels", "pulp", "scipy", "xlsxwriter"]:
                if l in e:
                    lib = f" ({l})"
                    break
            sub = f" — missing library{lib}"
        return f"File generation failed{sub}"
    if "timeout" in e:
        return "Timeout"
    if "error" in e:
        return "Other error"
    return "Unknown failure"


# ── Load results from either dispatch dir or tasks dir ────────────────────────
def load_results(dispatch_dir=None, tasks_dir=None):
    results = []
    if dispatch_dir:
        pattern = os.path.join(dispatch_dir, "*.json")
    else:
        pattern = os.path.join(tasks_dir, "*.json")

    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"No JSON files found in {dispatch_dir or tasks_dir}")
        sys.exit(1)

    for path in paths:
        if os.path.basename(path) == "index.json":
            continue
        try:
            d = json.load(open(path))
            results.append(d)
        except Exception as e:
            print(f"Warning: could not read {path}: {e}")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="DRA benchmark run diagnostic")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--dispatch-dir", help="dispatch_results/ directory")
    src.add_argument("--tasks-dir",    help="results/tasks/ directory")
    parser.add_argument("--agent",        help="Filter to one agent (claude/openai/gemini)")
    parser.add_argument("--show-pass",    action="store_true", help="Also print PASS rows")
    parser.add_argument("--summary-only", action="store_true", help="Print summary table only")
    parser.add_argument("--err-width",    type=int, default=90, help="Max error text width")
    args = parser.parse_args()

    results = load_results(args.dispatch_dir, args.tasks_dir)

    # ── Collect rows ──────────────────────────────────────────────────────────
    rows = []
    for d in results:
        tid = d.get("task_id", "?")
        pkg = d.get("package", {})
        research_type = pkg.get("research_type", "")
        domain        = pkg.get("domain", "")

        for agent, passes in d.get("agent_results", {}).items():
            if args.agent and agent != args.agent:
                continue
            r = passes[0] if isinstance(passes, list) else passes
            completed   = r.get("completed", False)
            forced_stop = r.get("forced_stop", False)
            error       = r.get("error") or ""
            cost        = r.get("total_cost_usd") or 0.0
            tokens_in   = r.get("input_tokens") or 0
            tokens_out  = r.get("output_tokens") or 0
            out_files   = r.get("output_files") or []
            out_errors  = r.get("output_file_errors") or {}

            if completed and not forced_stop:
                status = "PASS"
                cat    = ""
                if out_errors:
                    status = "PASS*"
                    cat    = "File generation failed: " + "; ".join(
                        f"{fmt}: {categorise(msg)}"
                        for fmt, msg in out_errors.items()
                    )
            else:
                status = "FAIL"
                cat    = categorise(error)

            rows.append({
                "tid":           tid,
                "agent":         agent,
                "status":        status,
                "category":      cat,
                "error":         error,
                "cost":          cost,
                "tokens_in":     tokens_in,
                "tokens_out":    tokens_out,
                "research_type": research_type,
                "domain":        domain,
                "out_files":     out_files,
            })

    # ── Print detail table ────────────────────────────────────────────────────
    if not args.summary_only:
        print()
        print(BOLD("── Per-agent results ") + "─" * 60)
        print()

        col_tid    = 36
        col_agent  = 8
        col_status = 7
        col_cost   = 7

        hdr = (
            f"{'Task ID':<{col_tid}}  "
            f"{'Agent':<{col_agent}}  "
            f"{'Status':<{col_status}}  "
            f"{'Cost':>{col_cost}}  "
            f"Failure / Note"
        )
        print(DIM(hdr))
        print(DIM("─" * (col_tid + col_agent + col_status + col_cost + args.err_width + 10)))

        for r in rows:
            if r["status"].startswith("PASS") and not args.show_pass and not r["category"]:
                continue

            if r["status"] == "PASS":
                status_str = GREEN(f"{'PASS':<{col_status}}")
            elif r["status"] == "PASS*":
                status_str = YELLOW(f"{'PASS*':<{col_status}}")
            else:
                status_str = RED(f"{'FAIL':<{col_status}}")

            agent_str = {
                "claude": CYAN,
                "openai": lambda t: _c("34", t),
                "gemini": lambda t: _c("32", t),
            }.get(r["agent"], lambda t: t)(f"{r['agent']:<{col_agent}}")

            cat = r["category"][:args.err_width] if r["category"] else ""

            print(
                f"{r['tid'][:col_tid]:<{col_tid}}  "
                f"{agent_str}  "
                f"{status_str}  "
                f"${r['cost']:>{col_cost-1}.3f}  "
                f"{cat}"
            )

        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(BOLD("── Summary ") + "─" * 70)
    print()

    total       = len(rows)
    passes      = sum(1 for r in rows if r["status"].startswith("PASS"))
    fails       = sum(1 for r in rows if r["status"] == "FAIL")
    pass_star   = sum(1 for r in rows if r["status"] == "PASS*")
    total_cost  = sum(r["cost"] for r in rows)
    tasks       = len(results)

    print(f"  Tasks dispatched : {tasks}")
    print(f"  Agent runs       : {total}  ({passes} PASS  {f'{pass_star} PASS*  ' if pass_star else ''}{fails} FAIL)")
    print(f"  Total cost       : ${total_cost:.4f}")
    print()

    # Per-agent breakdown
    print(f"  {'Agent':<10}  {'Pass':>5}  {'Pass*':>5}  {'Fail':>5}  {'Cost':>8}")
    print(f"  {'─'*10}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*8}")
    for agent in ["claude", "openai", "gemini"]:
        agent_rows = [r for r in rows if r["agent"] == agent]
        if not agent_rows:
            continue
        ap  = sum(1 for r in agent_rows if r["status"] == "PASS")
        aps = sum(1 for r in agent_rows if r["status"] == "PASS*")
        af  = sum(1 for r in agent_rows if r["status"] == "FAIL")
        ac  = sum(r["cost"] for r in agent_rows)
        print(f"  {agent:<10}  {ap:>5}  {aps:>5}  {af:>5}  ${ac:>7.3f}")
    print()

    # Failure category breakdown
    fail_rows = [r for r in rows if r["status"] in ("FAIL", "PASS*")]
    if fail_rows:
        print(BOLD("── Failure breakdown ") + "─" * 60)
        print()
        cats = defaultdict(list)
        for r in fail_rows:
            cats[r["category"]].append(r)
        for cat, cat_rows in sorted(cats.items(), key=lambda x: -len(x[1])):
            print(f"  {RED(str(len(cat_rows))+'x')}  {cat}")
            for r in cat_rows:
                print(f"        {DIM(r['tid'][:40])}  {r['agent']}")
        print()

    # Tasks with all agents failed
    task_map = defaultdict(list)
    for r in rows:
        task_map[r["tid"]].append(r)
    all_fail = [
        tid for tid, rs in task_map.items()
        if all(r["status"] == "FAIL" for r in rs)
    ]
    if all_fail:
        print(BOLD("── Tasks where ALL agents failed ") + "─" * 40)
        for tid in all_fail:
            print(f"  {RED(tid)}")
        print()


if __name__ == "__main__":
    main()