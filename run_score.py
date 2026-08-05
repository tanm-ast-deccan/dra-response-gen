#!/usr/bin/env python3
"""
run_score.py — score harness responses against the augmented packages, in
parallel, and rank tasks by the three crux metrics.

    python run_score.py --augmented-csv output/augmented/augmented_prompt_packages.csv \
                        --results-json 'results/*.json' --workers 6

All folder/file names are overridable (see augment_score_config). staging paths
inside the results JSON are resolved tolerant of a renamed staging dir via
--staging-remap OLD=NEW (string rewrite of the stored staging folder name).

Outputs (in --out-dir):
  scores.csv        one row per (task, provider, pass) with the 3 crux metrics
  task_ranking.csv  per-task aggregate (mean over runs), sorted hardest-first
                    (low crux_shapley_score / low pass_ratio / not cleared)
"""

import argparse, csv, glob, json, os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import configure_api
from src.augment_score_config import DEFAULT
from src.prompt_evaluator import DEFAULT_JUDGE_MODEL
from src.score_task import score_task

SCORE_FIELDS = ["task_id", "provider", "model", "pass_index", "run_id",
                "crux_cleared", "crux_verifier_pass_ratio", "crux_shapley_score",
                "n_crux", "n_passed", "n_unobservable",
                "n_all", "n_all_passed", "all_verifier_pass_ratio",
                "n_no_frozen_target", "n_no_frozen_target_passed",
                "scratch_fallback", "dropped_as_scratch", "golden_divergence",
                "deliverable_truncated", "not_found", "error"]


def load_augmented(path):
    with open(path, encoding="utf-8-sig") as f:
        return {r.get("task_id"): r for r in csv.DictReader(f)}


def load_responses(results_arg):
    files = glob.glob(results_arg) if any(c in results_arg for c in "*?[") else [results_arg]
    out = []
    for fn in files:
        raw = open(fn, encoding="utf-8", errors="replace").read()
        try:
            recs = json.loads(raw).get("results", [])
        except Exception:
            recs = list(_tolerant_records(raw))
            print(f"  warn: {fn} truncated; recovered {len(recs)} record(s)")
        for r in recs:
            if r.get("completed") or r.get("response_text") or r.get("output_files"):
                out.append(r)
    return out


def _tolerant_records(raw):
    """Yield result records from possibly-truncated JSON via a brace-balanced
    scan of the results array (skips a trailing incomplete record)."""
    start = raw.find('"results"')
    if start < 0:
        return
    start = raw.find('[', start)
    i, n = start + 1, len(raw)
    while i < n:
        while i < n and raw[i] in ' \t\r\n,':
            i += 1
        if i >= n or raw[i] == ']' or raw[i] != '{':
            break
        depth = 0; j = i; instr = False; esc = False
        while j < n:
            c = raw[j]
            if esc: esc = False
            elif c == '\\': esc = True
            elif c == '"': instr = not instr
            elif not instr:
                if c == '{': depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        j += 1; break
            j += 1
        if depth != 0:
            break
        try:
            yield json.loads(raw[i:j])
        except Exception:
            pass
        i = j


def _success_set(results_arg):
    """task_ids that succeeded (completed AND non-empty output_files) in the given
    results file(s). Tolerant of truncated JSON. Used by --require-both."""
    files = glob.glob(results_arg) if any(c in results_arg for c in "*?[") else [results_arg]
    ok = set()
    for fn in files:
        raw = open(fn, encoding="utf-8", errors="replace").read()
        try:
            recs = json.loads(raw).get("results", [])
        except Exception:
            recs = list(_tolerant_records(raw))
            print(f"  warn: {fn} was malformed/truncated; recovered {len(recs)} record(s)")
        for r in recs:
            if r.get("completed") and (r.get("output_files")):
                ok.add(r.get("task_id"))
    return ok


def main():
    ap = argparse.ArgumentParser(description="Parallel crux scorer + task ranking")
    ap.add_argument("--augmented-csv", default=None)
    ap.add_argument("--results-json", default=None, help="path or glob")
    ap.add_argument("--staging-remap", default="staging=staging_1",
                    help="rewrite stored staging folder segment: OLD=NEW (default staging=staging_1)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--require-both", nargs="+", default=None, metavar="RESULTS",
                    help="only score tasks that SUCCEEDED in every listed results file "
                         "(path or glob per model). Intersect across models for the gate.")
    ap.add_argument("--exclude-tasks", nargs="*", default=None,
                    help="task_ids to skip (e.g. tsk_auto1 tsk_auto2)")
    a = ap.parse_args()

    cfg = DEFAULT.override(
        augmented_csv=a.augmented_csv, results_glob=a.results_json,
        out_dir_score=a.out_dir, model=a.model, workers=a.workers,
    )
    model = cfg.model or DEFAULT_JUDGE_MODEL
    _rm = a.staging_remap.split("=", 1)
    staging_remap = (_rm[0], _rm[1]) if len(_rm) == 2 else ("staging", "staging_1")

    aug = load_augmented(cfg.augmented_csv)
    responses = load_responses(cfg.results_glob)

    # --require-both: intersect success sets across the listed results files
    allowed = None
    if a.require_both:
        sets = [_success_set(rf) for rf in a.require_both]
        allowed = set.intersection(*sets) if sets else set()
        print(f"--require-both: {len(allowed)} task(s) succeeded in ALL "
              f"{len(a.require_both)} result file(s)")
    excl = set(a.exclude_tasks or [])
    if excl:
        print(f"excluding {len(excl)} task(s): {sorted(excl)}")

    def _keep(tid):
        if tid not in aug: return False
        if allowed is not None and tid not in allowed: return False
        if tid in excl: return False
        return True

    jobs = [(i, r) for i, r in enumerate(responses) if _keep(r.get("task_id"))]
    missing = sorted({r.get("task_id") for r in responses if r.get("task_id") not in aug})
    print(f"{len(jobs)} response(s) to score across {len(aug)} augmented task(s); "
          f"{len(missing)} task(s) had responses but no augmented package"
          + (f": {missing[:5]}..." if missing else ""))

    configure_api()
    os.makedirs(cfg.out_dir_score, exist_ok=True)

    def work(idx, resp):
        return idx, score_task(aug[resp["task_id"]], resp,
                               staging_remap=staging_remap, model_name=model)

    results_by_idx = {}
    with ThreadPoolExecutor(max_workers=max(1, cfg.workers)) as ex:
        futs = [ex.submit(work, i, r) for i, r in jobs]
        done = 0
        for fut in as_completed(futs):
            idx, sr = fut.result()
            results_by_idx[idx] = sr
            done += 1
            if sr.not_found:
                print(f"  [{done}/{len(jobs)}] {sr.task_id} {sr.provider} NOT FOUND — {sr.error}", flush=True)
            else:
                tag = "CLEARED" if sr.crux_cleared else f"{sr.n_passed}/{sr.n_crux}"
                print(f"  [{done}/{len(jobs)}] {sr.task_id} {sr.provider} "
                      f"shapley={sr.crux_shapley_score:.2f} {tag}"
                      + (f" ERR {sr.error}" if sr.error else ""), flush=True)

    # deterministic order = original job order
    scored = [results_by_idx[i] for i, _ in jobs if i in results_by_idx]

    # scores.csv
    sp = os.path.join(cfg.out_dir_score, "scores.csv")
    with open(sp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SCORE_FIELDS, extrasaction="ignore")
        w.writeheader()
        for s in scored:
            w.writerow(s.to_dict())

    # also dump per-verifier detail as JSONL for auditing
    jp = os.path.join(cfg.out_dir_score, "scores_detail.jsonl")
    with open(jp, "w", encoding="utf-8") as f:
        for s in scored:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")

    # task_ranking.csv (aggregate over runs per task)
    by_task = {}
    for s in scored:
        by_task.setdefault(s.task_id, []).append(s)
    rank_rows = []
    for tid, runs in by_task.items():
        rank_rows.append({
            "task_id": tid,
            "n_runs": len(runs),
            "any_cleared": any(r.crux_cleared for r in runs),
            "frac_cleared": round(mean(1.0 if r.crux_cleared else 0.0 for r in runs), 3),
            "mean_crux_shapley_score": round(mean(r.crux_shapley_score for r in runs), 3),
            "mean_pass_ratio": round(mean(r.crux_verifier_pass_ratio for r in runs), 3),
            "n_crux": runs[0].n_crux,
        })
    # hardest first: not cleared, lowest shapley, lowest pass ratio
    rank_rows.sort(key=lambda x: (x["frac_cleared"], x["mean_crux_shapley_score"],
                                  x["mean_pass_ratio"]))
    rp = os.path.join(cfg.out_dir_score, "task_ranking.csv")
    with open(rp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rank_rows[0].keys()) if rank_rows else
                           ["task_id", "n_runs", "any_cleared", "frac_cleared",
                            "mean_crux_shapley_score", "mean_pass_ratio", "n_crux"])
        w.writeheader()
        for r in rank_rows:
            w.writerow(r)

    nf = [s for s in scored if s.not_found]
    if nf:
        print(f"\n{len(nf)} task-response(s) NOT FOUND (deliverable file unresolved after remap); "
              f"see error column in scores.csv")
    print(f"\nWrote {len(scored)} score row(s) -> {sp}")
    print(f"Task ranking (hardest first) -> {rp}")
    hard = [r for r in rank_rows if not r["any_cleared"]]
    print(f"{len(hard)} task(s) with crux NEVER cleared across runs "
          f"(candidate hard set).")


if __name__ == "__main__":
    main()