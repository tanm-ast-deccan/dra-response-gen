#!/usr/bin/env python3
"""
score_crux_rates.py — from the SME-graded delivery CSV, compute per run:
  (1) overall verifier pass rate (all scored verifiers)
  (2) CRUX-only pass rate (only verifiers in the task's crux set)
  (3) Shapley-WEIGHTED crux pass rate (each crux verifier weighted by its
      crux-Shapley weight from the augment JSON)

Crux IDs and Shapley weights come from output/augmented/tsk_XXXX_augment.json.
The graded scores are parsed from the 'Justification of the scores' column,
where the SME writes 'V1 - 1: ...', 'V2 - 0: ...' etc.

    python score_crux_rates.py --csv <graded_delivery.csv> --aug-dir output/augmented
"""
import argparse, csv, json, os, re

SCORE_RE = re.compile(r'V(\d+)\s*-\s*([01])\b')          # N format: 'V1 - 1: ...'
TRAIL_RE = re.compile(r'-\s*([01])\s*$')                  # M format: 'V1 - <text> - 1'

def parse_scores(cell):
    """Parse SME verifier scores from either column format:
      M ('Augmented_verifiers with scores'): 'V1 - <text> - 1,' (score TRAILS)
      N ('Justification of the scores'):      'V1 - 1: <text>'   (score LEADS)
    Returns {verifier_id_int: 0/1}. Prefers M-style per-line trailing score;
    falls back to leading-score matches."""
    out = {}
    # M-style: one verifier per line, leading Vn + trailing 0/1
    for line in (cell or "").splitlines():
        line = line.strip().rstrip(",")
        m = re.match(r'V(\d+)\b', line)
        if not m:
            continue
        t = TRAIL_RE.search(line)
        if t:
            out[int(m.group(1))] = int(t.group(1))
    if out:
        return out
    # N-style fallback: 'Vn - 0/1' immediately
    for vid, sc in SCORE_RE.findall(cell or ""):
        out[int(vid)] = int(sc)
    return out

def find_header_row(rows):
    for i, r in enumerate(rows[:10]):
        if "task_id" in [c.strip() for c in r]:
            return i
    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--aug-dir", default="output/augmented")
    ap.add_argument("--score-col", default="Augmented_verifiers with scores",
                    help="column holding the graded verifier scores (M-style trailing "
                         "or N-style leading; both parsed)")
    args = ap.parse_args()

    raw = list(csv.reader(open(args.csv, encoding="utf-8-sig")))
    h = find_header_row(raw)
    header = [c.strip() for c in raw[h]]
    idx = {c: i for i, c in enumerate(header)}
    tcol = idx.get("task_id"); mcol = idx.get("anon_model")
    scol = idx.get(args.score_col)
    if scol is None:
        # fall back: last column
        scol = len(header) - 1
        print(f"(warning: '{args.score_col}' not found; using column {scol}: {header[scol]!r})")

    # cache augment crux + shapley per task
    aug_cache = {}
    def aug(task):
        if task not in aug_cache:
            p = os.path.join(args.aug_dir, f"{task}_augment.json")
            aug_cache[task] = json.load(open(p)) if os.path.exists(p) else {}
        return aug_cache[task]

    results = []
    for r in raw[h+1:]:
        if len(r) <= max(tcol, mcol, scol): continue
        task, model = r[tcol].strip(), r[mcol].strip()
        if not task.startswith("tsk_"): continue
        scores = parse_scores(r[scol])
        if not scores: continue

        a = aug(task)
        crux_ids = [int(v[1:]) for v in a.get("crux_ids", []) if v[1:].isdigit()]
        weights = {int(k[1:]): float(v) for k, v in
                   a.get("crux_shapley_weights", {}).items() if k[1:].isdigit()}

        # overall
        ov_p, ov_t = sum(scores.values()), len(scores)
        # crux-only: intersect scored verifiers with crux set
        crux_scored = {v: scores[v] for v in crux_ids if v in scores}
        cx_p, cx_t = sum(crux_scored.values()), len(crux_scored)
        # shapley-weighted crux: sum(weight*pass) / sum(weight) over scored crux
        wsum = sum(weights.get(v, 0) for v in crux_scored)
        wpass = sum(weights.get(v, 0) for v, s in crux_scored.items() if s == 1)
        results.append({
            "task": task, "model": model,
            "overall": (ov_p, ov_t),
            "crux": (cx_p, cx_t),
            "crux_missing_scores": [v for v in crux_ids if v not in scores],
            "wpass": wpass, "wsum": wsum,
        })

    if not results:
        print("No parseable 'Vn - 0/1' scores found. Is the grading done and in the right column?")
        return

    print(f"parsed {len(results)} graded run(s)\n")
    print(f"{'task':<18}{'model':<9}{'overall':>10}{'crux':>10}{'crux%':>8}{'shapley%':>10}")
    agg = {"ov_p":0,"ov_t":0,"cx_p":0,"cx_t":0,"wpass":0.0,"wsum":0.0}
    for r in results:
        op, ot = r["overall"]; cp, ct = r["crux"]
        cxr = (100*cp/ct) if ct else float('nan')
        shr = (100*r["wpass"]/r["wsum"]) if r["wsum"] else float('nan')
        print(f"{r['task']:<18}{r['model']:<9}{op}/{ot:<7}{cp}/{ct:<7}"
              f"{cxr:>7.0f}%{shr:>9.1f}%"
              + (f"  (crux not scored: {r['crux_missing_scores']})" if r['crux_missing_scores'] else ""))
        agg["ov_p"]+=op; agg["ov_t"]+=ot; agg["cx_p"]+=cp; agg["cx_t"]+=ct
        agg["wpass"]+=r["wpass"]; agg["wsum"]+=r["wsum"]

    print("\n=== AGGREGATE (all graded runs pooled) ===")
    print(f"overall verifier pass rate : {agg['ov_p']}/{agg['ov_t']} = {100*agg['ov_p']/agg['ov_t']:.1f}%")
    if agg["cx_t"]:
        print(f"crux verifier pass rate    : {agg['cx_p']}/{agg['cx_t']} = {100*agg['cx_p']/agg['cx_t']:.1f}%")
    if agg["wsum"]:
        print(f"Shapley-weighted crux rate : {100*agg['wpass']/agg['wsum']:.1f}%")

    # per-model split
    print("\n=== per-model (crux-weighted) ===")
    for m in ("Model_A","Model_B"):
        sub=[r for r in results if r["model"]==m]
        wp=sum(r["wpass"] for r in sub); ws=sum(r["wsum"] for r in sub)
        cp=sum(r["crux"][0] for r in sub); ct=sum(r["crux"][1] for r in sub)
        if ws:
            print(f"{m}: crux {cp}/{ct}={100*cp/ct:.0f}% | shapley-weighted {100*wp/ws:.1f}%  (n={len(sub)})")

if __name__ == "__main__":
    main()