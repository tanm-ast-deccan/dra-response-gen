#!/usr/bin/env python3
"""
gear_score.py — dependency-aware rubric scoring (GEAR, arxiv 2606.03361) over the
per-task canonical DAG, driven by the SME marks.

For each graded run it computes:
  flat      : flat Shapley-weighted crux pass rate  (baseline — the current metric)
  gear      : GEAR soft-suppression score at the chosen lambda
  sweep     : gear at lambda in {0.6, 0.4, 0.2, 0.0}  (0.0 == hard DAG gate)

GEAR core (their eq. 12), topological, one pass:
  q_hat_i = p_i * PROD over parents j [ q_hat_j + (1 - q_hat_j) * lambda_ji ]
  p_i = SME mark (1/0);  roots: q_hat = p_i.
Aggregate:  S = sum_i w_i * q_hat_i / sum_i w_i   over crux verifiers.

Lambda (retention when a parent is unsupported) is uniform by default (--lam),
overridable to an sov-derived scheme with --lam-mode sov
(arithmetic/source_file edges -> strong 0.2 ; llm_judgment edges -> weak 0.6).

Optional trivia removal: --drop "tsk_x:V7,V9;tsk_y:V3"  removes those verifiers
from BOTH numerator and denominator for that task (Conundrum-1 fix). Reported as
a separate column so you see the effect.

Inputs:
  --marks   consolidated_scores.csv   (task_id, model, verifier_id, mark, ...)
  --augment augment_index.json        (task -> {dag, crux, weights, sov})

Usage:
  python gear_score.py --marks consolidated_scores.csv --augment augment_index.json \
      --lam 0.2 --out gear_scores.csv
"""
import argparse, csv, json
from collections import defaultdict

SWEEP = [0.6, 0.4, 0.2, 0.0]


def ancestors_ok(dag):
    """topological order of crux nodes; returns ordering (parents before children)."""
    order, seen = [], set()
    def visit(n, stack):
        if n in seen:
            return
        if n in stack:      # cycle guard (shouldn't happen; DAG)
            return
        stack.add(n)
        for p in dag.get(n, []):
            visit(p, stack)
        stack.discard(n)
        seen.add(n); order.append(n)
    for n in dag:
        visit(n, set())
    return order


def lam_for_edge(child, sov, lam, lam_mode):
    if lam_mode == "sov":
        s = (sov.get(child) or "").lower()
        return 0.6 if s == "llm_judgment" else 0.2   # judgment = weak, computed = strong
    return lam


def gear_scores(marks, dag, weights, crux, sov, lam, lam_mode, drop=()):
    """return dict of qhat + aggregate S for a given lambda."""
    scored = [v for v in crux if v not in drop]
    order = [v for v in ancestors_ok(dag) if v in scored]
    # ensure any crux not in dag (isolated) still included
    for v in scored:
        if v not in order:
            order.append(v)
    qhat = {}
    for v in order:
        p = marks.get(v)
        p = 0.0 if p is None else float(p)
        parents = [j for j in dag.get(v, []) if j in scored]
        factor = 1.0
        for j in parents:
            qj = qhat.get(j, 0.0)
            lam_ji = lam_for_edge(v, sov, lam, lam_mode)
            factor *= (qj + (1.0 - qj) * lam_ji)
        qhat[v] = p * factor
    tot = sum(weights.get(v, 0.0) for v in scored) or 1.0
    S = sum(weights.get(v, 0.0) * qhat[v] for v in scored) / tot
    return qhat, S


def flat_score(marks, weights, crux, drop=()):
    scored = [v for v in crux if v not in drop]
    tot = sum(weights.get(v, 0.0) for v in scored) or 1.0
    return sum(weights.get(v, 0.0) for v in scored if marks.get(v) == 1) / tot


def parse_drop(s):
    d = {}
    for part in (s or "").split(";"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        t, vs = part.split(":", 1)
        d[t.strip()] = set(v.strip() for v in vs.split(",") if v.strip())
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--marks", required=True)
    ap.add_argument("--augment", required=True)
    ap.add_argument("--lam", type=float, default=0.2)
    ap.add_argument("--lam-mode", choices=["uniform", "sov"], default="uniform")
    ap.add_argument("--drop", default="", help='trivia to remove, e.g. "tsk_x:V7;tsk_y:V3,V9"')
    ap.add_argument("--out", default="gear_scores.csv")
    a = ap.parse_args()

    aug = json.load(open(a.augment))
    drop = parse_drop(a.drop)

    # marks grouped by (task, model) -> {vid: mark}
    m = defaultdict(dict)
    for r in csv.DictReader(open(a.marks, encoding="utf-8-sig")):
        try:
            mk = int(r["mark"]) if r["mark"] not in ("", "None") else None
        except ValueError:
            mk = None
        m[(r["task_id"], r["model"])][r["verifier_id"]] = mk

    rows = []
    for (task, model), marks in sorted(m.items()):
        A = aug.get(task)
        if not A:
            rows.append(dict(task_id=task, model=model, status="no augment"))
            continue
        dag, weights, crux, sov = A["dag"], A["weights"], A["crux"], A.get("sov", {})
        dr = drop.get(task, set())
        flat = flat_score(marks, weights, crux)
        flat_dropped = flat_score(marks, weights, crux, dr) if dr else flat
        _, g = gear_scores(marks, dag, weights, crux, sov, a.lam, a.lam_mode, dr)
        sweep = {}
        for L in SWEEP:
            _, s = gear_scores(marks, dag, weights, crux, sov, L, a.lam_mode, dr)
            sweep[L] = s
        rows.append(dict(
            task_id=task, model=model, status="ok",
            n_crux=len(crux), n_dropped=len(dr),
            flat=round(100*flat, 1),
            flat_trivia_removed=round(100*flat_dropped, 1),
            gear=round(100*g, 1),
            **{f"gear_lam{L}": round(100*sweep[L], 1) for L in SWEEP}))

    cols = ["task_id","model","status","n_crux","n_dropped","flat",
            "flat_trivia_removed","gear"] + [f"gear_lam{L}" for L in SWEEP]
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})

    ok = [r for r in rows if r.get("status") == "ok"]
    print(f"scored {len(ok)}/{len(rows)} runs (lambda={a.lam}, mode={a.lam_mode}) -> {a.out}\n")
    print(f"{'task':<17}{'model':<9}{'flat':>6}{'gear':>7}{'l0.6':>6}{'l0.4':>6}{'l0.2':>6}{'l0.0':>6}")
    for r in ok:
        print(f"{r['task_id']:<17}{r['model']:<9}{r['flat']:>6}{r['gear']:>7}"
              f"{r['gear_lam0.6']:>6}{r['gear_lam0.4']:>6}{r['gear_lam0.2']:>6}{r['gear_lam0.0']:>6}")
    # per-model aggregate at chosen lambda
    print("\n=== per-model mean (chosen lambda) ===")
    for md in ("Model_A", "Model_B"):
        sub = [r for r in ok if r["model"] == md]
        if sub:
            mf = sum(r["flat"] for r in sub)/len(sub)
            mg = sum(r["gear"] for r in sub)/len(sub)
            print(f"  {md}: flat {mf:.1f}%  gear {mg:.1f}%  (n={len(sub)})")
    # Tencent gate on gear
    print("\n=== Tencent difficulty gate (per-task, gear score) ===")
    bt = defaultdict(dict)
    for r in ok:
        bt[r["task_id"]][r["model"]] = r["gear"]
    qual = 0
    for t, mm in sorted(bt.items()):
        b, h = mm.get("Model_B"), mm.get("Model_A")
        if b is not None and h is not None:
            ok_gate = (b < 40 and h < 20)
            if ok_gate: qual += 1
            tag = "  <-- QUALIFIES" if ok_gate else ""
            print(f"  {t}: Doubao(B)={b}%  Hunyuan(A)={h}%{tag}")
    print(f"\nTasks meeting <40%/<20% on GEAR: {qual}/{len(bt)}")


if __name__ == "__main__":
    main()