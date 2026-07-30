#!/usr/bin/env python3
"""
consolidate_scores.py — collapse the two graded columns (M: marks, N: justifications)
into ONE canonical long-format table, one row per (task_id, model, verifier).

Both columns are per-verifier labelled with a `V# - ` prefix, so alignment is
deterministic (no fuzzy/LLM matching needed):
  M format:  "V5 - <verifier text> - <0|1>,"        (score TRAILS)
  N format:  "V5 - <0|1>: <justification prose>"     (score LEADS, then colon)

Output columns:
  task_id, model, verifier_id, mark, verifier_text, justification,
  n_score, score_mismatch, missing_justification

`score_mismatch` flags rows where M's mark and N's restated score disagree
(a grading-consistency red flag worth review). `missing_justification` flags
verifiers scored in M but not explained in N.

Usage:
  python consolidate_scores.py --csv <graded.csv> --out consolidated_scores.csv
"""
import argparse, csv, re, sys

HEADER_MARKER = "task_id"

# SMEs are inconsistent: some wrote "V1 - text", others "V1: text".
# Accept EITHER delimiter (dash or colon) after the verifier id, everywhere.
# A verifier block starts at "V<n><delim>" and runs to the next such marker.
M_SPLIT = re.compile(r'(?=V\d+\s*[-:]\s)')
M_ID    = re.compile(r'^V(\d+)\s*[-:]\s*(.*)$', re.S)
# score TRAILS in M: the last "- <0|1>" (or ": <0|1>") before comma/newline/EOL
M_TRAIL_SCORE = re.compile(r'[-:]\s*([01])\s*,?\s*$')

# N: "V<n><delim> <score>: <justification>"  (score LEADS, then a colon)
N_SPLIT = re.compile(r'(?=V\d+\s*[-:]\s)')
N_ID    = re.compile(r'^V(\d+)\s*[-:]\s*([01])\s*:\s*(.*)$', re.S)
N_ID_NOSCORE = re.compile(r'^V(\d+)\s*[-:]\s*(.*)$', re.S)  # fallback if no leading score


def parse_M(cell):
    """return {vid: (mark, verifier_text)}"""
    out = {}
    for block in M_SPLIT.split(cell or ""):
        block = block.strip()
        if not block:
            continue
        m = M_ID.match(block)
        if not m:
            continue
        vid = "V" + m.group(1)
        rest = m.group(2)
        sc = M_TRAIL_SCORE.search(rest)
        mark = int(sc.group(1)) if sc else None
        # verifier text = rest minus the trailing "- score,"
        text = M_TRAIL_SCORE.sub("", rest).strip().rstrip("-").strip().rstrip(",").strip()
        out[vid] = (mark, text)
    return out


def parse_N(cell):
    """return {vid: (n_score, justification)}"""
    out = {}
    for block in N_SPLIT.split(cell or ""):
        block = block.strip()
        if not block:
            continue
        m = N_ID.match(block)
        if m:
            out["V" + m.group(1)] = (int(m.group(2)), m.group(3).strip())
        else:
            m2 = N_ID_NOSCORE.match(block)
            if m2:
                out["V" + m2.group(1)] = (None, m2.group(2).strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="consolidated_scores.csv")
    a = ap.parse_args()

    raw = list(csv.reader(open(a.csv, encoding="utf-8-sig")))
    # find header row
    hrow = next((i for i, r in enumerate(raw[:6])
                 if any(c.strip() == HEADER_MARKER for c in r)), None)
    if hrow is None:
        sys.exit("could not find header row containing 'task_id'")
    hdr = [c.strip() for c in raw[hrow]]
    idx = {c: i for i, c in enumerate(hdr)}
    ci_task  = idx["task_id"]
    ci_model = idx["anon_model"]
    ci_M = idx.get("Augmented_verifiers with scores")
    ci_N = idx.get("Justification of the scores")
    if ci_M is None:
        sys.exit(f"no marks column; headers={hdr}")

    rows_out = []
    n_mismatch = n_missing_just = 0
    for r in raw[hrow + 1:]:
        if len(r) <= max(ci_task, ci_model, ci_M):
            continue
        task = r[ci_task].strip()
        model = r[ci_model].strip()
        if not task.startswith("tsk_"):
            continue
        M = parse_M(r[ci_M])
        N = parse_N(r[ci_N]) if ci_N is not None and len(r) > ci_N else {}
        for vid in sorted(M, key=lambda v: int(v[1:])):
            mark, vtext = M[vid]
            n_score, just = N.get(vid, (None, ""))
            mism = (n_score is not None and mark is not None and n_score != mark)
            miss = (not just)
            if mism: n_mismatch += 1
            if miss: n_missing_just += 1
            rows_out.append(dict(
                task_id=task, model=model, verifier_id=vid, mark=mark,
                verifier_text=vtext, justification=just,
                n_score=n_score,
                score_mismatch=int(mism), missing_justification=int(miss)))

    cols = ["task_id","model","verifier_id","mark","verifier_text",
            "justification","n_score","score_mismatch","missing_justification"]
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows_out)

    print(f"wrote {len(rows_out)} verifier-rows → {a.out}")
    print(f"  tasks×models: {len(set((r['task_id'],r['model']) for r in rows_out))}")
    print(f"  score mismatches (M vs N restated): {n_mismatch}")
    print(f"  verifiers with no justification in N: {n_missing_just}")
    if n_mismatch:
        print("  ⚠ review these — M and N disagree on the score:")
        for r in rows_out:
            if r["score_mismatch"]:
                print(f"     {r['task_id']} {r['model']} {r['verifier_id']}: "
                      f"M={r['mark']} N={r['n_score']}")


if __name__ == "__main__":
    main()
