#!/usr/bin/env python3
"""Diagnose a harness batch: effort, reliability, compliance, failure modes.

    python run_diagnostics.py './runs_dir/*/*.json'
    python run_diagnostics.py run.json --csv final_tasks.csv    # + compliance
    python run_diagnostics.py './runs_dir/*/*.json' --out diag.csv

Reads what the harness records and nothing else. No LLM calls, no network.

WHY EACH SECTION EXISTS - every one of these was a real miss on a live batch:

  finish_reason   Three wrong diagnoses were made of one stalled run before
                  anyone looked at this field. 'stop' means the model chose to
                  end; 'length' means it was truncated mid-message; 'tool_calls'
                  means it wanted a tool and the loop dropped it. Only the first
                  is a model failure. It is not on the result record, only inside
                  the trajectory, so it is easy to miss.

  effort          Effort tracked score at both ends of a real batch. The two
                  weakest analytical results were the two shallowest runs - one
                  took 5 turns and 8 tool calls with no code execution at all
                  against 17 verifiers. Turns-per-verifier flags under-engagement
                  before scoring does.

  errors          The tool_call 'error' field was 0 across a whole batch while a
                  Python traceback sat in result_preview. An exception that comes
                  back as OUTPUT is not counted as an error, so reliability
                  numbers understate. This counts both.

  compliance      --web-search was left enabled for a batch in which five of
                  seven prompts forbade external research. Nothing breached it,
                  but nothing prevented it either. This cross-checks each
                  prompt's stated protocol against what the run actually called.

  deliverable     A run that produces no deliverable still records completed=True
                  with no error. The scorer would grade it as a legitimate zero.
                  A content check catches it; a structural one does not.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import Counter

#: phrases in a prompt that forbid external retrieval
FORBIDS = re.compile(
    r"web\s*search\s*protocol\s*[:\-]?\s*closed"
    r"|no\s+external\s+(?:search|research|data|source)"
    r"|not\s+(?:use|introduce|consult)\s+(?:any\s+)?external"
    r"|external\s+research[^.]{0,40}prohibited"
    r"|only\s+the\s+(?:attached|provided|permitted)[^.]{0,60}(?:file|document)"
    r"|strictly\s+prohibited",
    re.IGNORECASE)
ALLOWS = re.compile(
    r"web\s*search\s*protocol\s*[:\-]?\s*open"
    r"|must\s+be\s+fetched\s+live|fetch(?:ed)?\s+(?:the\s+)?live"
    r"|permitted\s+external\s+(?:source|input)",
    re.IGNORECASE)
RETRIEVAL_TOOLS = {"web_search", "web_fetch", "browse", "search_web"}

#: a tool result that is really an exception, however the harness labelled it
TRACEBACK = re.compile(r"Traceback \(most recent call last\)|^\s*\w+Error:|Exception:",
                       re.MULTILINE)


def load_runs(patterns):
    out = []
    for pat in patterns:
        for f in sorted(glob.glob(pat)) or ([pat] if os.path.isfile(pat) else []):
            try:
                with open(f, encoding="utf-8") as fh:
                    out.append((f, json.load(fh)))
            except Exception as e:
                print(f"  !! could not read {f}: {e}", file=sys.stderr)
    return out


def prompt_policy(csv_path):
    """{task_id: 'forbidden'|'allowed'|'unstated'} from the task sheet."""
    if not csv_path or not os.path.exists(csv_path):
        return {}
    pol = {}
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            tid = (r.get("task_id") or "").strip()
            if not tid:
                continue
            txt = " ".join(str(r.get(k) or "") for k in
                           ("Corrected Prompt", "prompt", "Prompt",
                            "Corrected Solution Logic", "Solution Logic"))
            pol[tid] = ("forbidden" if FORBIDS.search(txt)
                        else "allowed" if ALLOWS.search(txt) else "unstated")
    return pol


def analyse(rec):
    tc = rec.get("tool_calls") or []
    tr = rec.get("trajectory") or []
    names = Counter(t.get("name") for t in tc)

    flagged = sum(1 for t in tc if t.get("error"))
    hidden = sum(1 for t in tc
                 if not t.get("error") and TRACEBACK.search(str(t.get("result_preview") or "")))

    text = str(rec.get("response_text") or "")
    files = rec.get("output_files") or []
    # a deliverable that exists but says nothing is not a deliverable
    thin = len(text.strip()) < 400 and not files

    return {
        "task_id": rec.get("task_id", ""),
        "provider": rec.get("provider", ""),
        "pass": rec.get("pass_index", ""),
        "completed": rec.get("completed"),
        "finish_reason": (tr[-1].get("finish_reason") if tr else ""),
        "forced_stop": rec.get("forced_stop"),
        "error": (str(rec.get("error") or "")[:80]),
        "turns": rec.get("turns", 0),
        "tool_calls": len(tc),
        "tools_flagged_error": flagged,
        "tools_traceback_in_output": hidden,
        "code_exec": sum(names.get(k, 0) for k in
                         ("python_execute", "bash_execute", "edit_file")),
        "retrieval_calls": sum(names.get(k, 0) for k in RETRIEVAL_TOOLS),
        "tool_mix": ", ".join(f"{k}:{v}" for k, v in names.most_common()),
        "input_tokens": rec.get("input_tokens", 0),
        "output_tokens": rec.get("output_tokens", 0),
        "cost_usd": round(float(rec.get("total_cost_usd") or 0), 4),
        "duration_sec": round(float(rec.get("total_duration_sec") or 0)),
        "response_chars": len(text),
        "n_output_files": len(files),
        "thin_deliverable": thin,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run json file(s) or a glob")
    ap.add_argument("--csv", default=None,
                    help="task sheet, to cross-check the web-search policy")
    ap.add_argument("--out", default=None, help="write the per-run table here")
    args = ap.parse_args()

    runs = load_runs(args.runs)
    if not runs:
        raise SystemExit("no run files matched")
    pol = prompt_policy(args.csv)

    recs = []
    for path, d in runs:
        for r in d.get("results") or []:
            row = analyse(r)
            row["run_file"] = os.path.basename(path)
            recs.append(row)
    done = [r for r in recs if r["completed"]]

    # ---------------------------------------------------------------- effort
    print("\n" + "=" * 108)
    print("EFFORT AND COST")
    print("=" * 108)
    print(f"  {'task':22s} {'prov':9s} {'turns':>5s} {'tools':>6s} {'code':>5s} "
          f"{'in-tok':>9s} {'out-tok':>8s} {'$':>8s} {'sec':>6s} {'finish':>10s}")
    print("  " + "-" * 104)
    for r in sorted(recs, key=lambda x: -x["turns"]):
        print(f"  {r['task_id'][:22]:22s} {r['provider'][:9]:9s} {r['turns']:>5d} "
              f"{r['tool_calls']:>6d} {r['code_exec']:>5d} {r['input_tokens']:>9,} "
              f"{r['output_tokens']:>8,} {r['cost_usd']:>8.4f} {r['duration_sec']:>6.0f} "
              f"{str(r['finish_reason'])[:10]:>10s}")
    if done:
        print("  " + "-" * 104)
        n = len(done)
        print(f"  {'MEAN':22s} {'':9s} {sum(r['turns'] for r in done)/n:>5.1f} "
              f"{sum(r['tool_calls'] for r in done)/n:>6.1f} "
              f"{sum(r['code_exec'] for r in done)/n:>5.1f} "
              f"{sum(r['input_tokens'] for r in done)//n:>9,} "
              f"{sum(r['output_tokens'] for r in done)//n:>8,} "
              f"{sum(r['cost_usd'] for r in done):>8.4f} "
              f"{sum(r['duration_sec'] for r in done)/n:>6.0f}   (total $)")

    # ------------------------------------------------------------ how it ended
    print("\n" + "=" * 108)
    print("HOW EACH RUN ENDED")
    print("=" * 108)
    fr = Counter(str(r["finish_reason"]) for r in recs)
    print(f"  finish_reason: {dict(fr)}")
    print("    stop        = the model chose to end. If the deliverable is empty, that is a MODEL failure.")
    print("    length      = truncated at max_tokens mid-message. Raise --max-tokens; not a model failure.")
    print("    tool_calls  = the model wanted a tool and the loop ended anyway. Harness bug.")
    for r in recs:
        why = []
        if not r["completed"]:
            why.append(f"NOT COMPLETED: {r['error'] or 'no error recorded'}")
        if r["forced_stop"]:
            why.append("forced_stop")
        if r["finish_reason"] == "length":
            why.append("TRUNCATED at max_tokens")
        if r["finish_reason"] == "tool_calls":
            why.append("ended on a pending tool call")
        if r["thin_deliverable"]:
            why.append(f"THIN DELIVERABLE: {r['response_chars']} chars, "
                       f"{r['n_output_files']} file(s) - would score as a false zero")
        if why:
            print(f"  !! {r['task_id']:22s} {'; '.join(why)}")

    # ----------------------------------------------------------- tool reliability
    print("\n" + "=" * 108)
    print("TOOL RELIABILITY")
    print("=" * 108)
    tf = sum(r["tools_flagged_error"] for r in recs)
    th = sum(r["tools_traceback_in_output"] for r in recs)
    tt = sum(r["tool_calls"] for r in recs)
    print(f"  {tt} tool calls: {tf} flagged as errors, {th} returned a traceback as OUTPUT")
    if th and not tf:
        print("  NOTE: the harness's error flag caught none of these. An exception that comes back")
        print("        as a tool result is not marked an error, so reliability numbers understate.")
    for r in recs:
        if r["tools_traceback_in_output"] or r["tools_flagged_error"]:
            print(f"    {r['task_id']:22s} flagged {r['tools_flagged_error']}, "
                  f"traceback-in-output {r['tools_traceback_in_output']} of {r['tool_calls']}")

    # --------------------------------------------------------------- compliance
    print("\n" + "=" * 108)
    print("RETRIEVAL COMPLIANCE")
    print("=" * 108)
    if not pol:
        print("  (pass --csv <task sheet> to cross-check each prompt's stated protocol)")
    breaches = []
    for r in recs:
        p = pol.get(r["task_id"], "unknown")
        if r["retrieval_calls"]:
            tag = "BREACH" if p == "forbidden" else ("ok" if p == "allowed" else "UNSTATED POLICY")
            print(f"  {tag:16s} {r['task_id']:22s} {r['retrieval_calls']} retrieval call(s), "
                  f"prompt says: {p}")
            if p == "forbidden":
                breaches.append(r["task_id"])
    if pol:
        forb = [t for t, v in pol.items() if v == "forbidden"]
        silent = [t for t, v in pol.items() if v == "unstated"]
        print(f"  prompts forbidding retrieval : {len(forb)}  {forb}")
        if silent:
            print(f"  prompts with NO stated policy: {len(silent)}  {silent}")
            print("    a task with no protocol line cannot be judged compliant either way - fix the prompt")
    if not breaches:
        print("  no breaches. NOTE: if --web-search was enabled, that is the models declining to")
        print("  search rather than the harness preventing it. Disable it, or set it per task.")

    # ------------------------------------------------------------------ effort flag
    print("\n" + "=" * 108)
    print("UNDER-ENGAGEMENT")
    print("=" * 108)
    print("  A run with few turns and no code execution on a task with many verifiers is worth")
    print("  looking at before scoring - on one real batch the two weakest results were the two")
    print("  shallowest runs.")
    for r in sorted(recs, key=lambda x: x["turns"]):
        if r["completed"] and (r["turns"] <= 6 or r["code_exec"] == 0):
            print(f"    {r['task_id']:22s} {r['turns']:>3d} turns, {r['tool_calls']:>3d} tool calls, "
                  f"{r['code_exec']} code executions, {r['response_chars']:,} response chars")

    if args.out:
        cols = ["run_file", "task_id", "provider", "pass", "completed", "finish_reason",
                "forced_stop", "error", "turns", "tool_calls", "code_exec",
                "tools_flagged_error", "tools_traceback_in_output", "retrieval_calls",
                "tool_mix", "input_tokens", "output_tokens", "cost_usd", "duration_sec",
                "response_chars", "n_output_files", "thin_deliverable"]
        with open(args.out, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in recs:
                w.writerow(r)
        print(f"\n{len(recs)} row(s) -> {args.out}")


if __name__ == "__main__":
    main()