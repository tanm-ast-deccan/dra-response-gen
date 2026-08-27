#!/usr/bin/env python3
"""
score_end_to_end.py — grade harness deliverables from scratch and write a CSV.

NO reference score anywhere. This reads ONLY the verifier definitions and their
expected values out of each augment package, LLM-grades every verifier against the
model's actual deliverable, and computes CHAIN + GEAR0 from those fresh marks. It
never reads, compares to, or requires any stored/hy3/"golden" score.

INPUTS
------
--augment-dir   folder of package JSONs, one per task-run:
                    {task_id}__{tag}__run{N}.json
                each carrying (inside the file):
                    crux_ids / crux_dag / crux_depth
                    crux_shapley_weights   (+ neglog_crux if present; else derived)
                    expected_values        {vid: {value,tol,unit,kind}}
                    augmented_verifiers    (the verifier id: text lines)
                The run label is read from run.run_label (authoritative), not the
                filename — a filename/label mismatch is reported, not obeyed.

--output-dir    root of per-run deliverables in subfolders:
                    <output-dir>/<task_id>/<model>/<run>/<the deliverable file(s)>
                'run' matches the package's run number (from run.run_label). Every
                file in that leaf folder is a candidate; the content classifier in
                score_task picks the real deliverable and rejects scratch files.

--input-csv     (optional) the batch CSV with a task_id column and a Drive-link
                column for input files. Only used to stamp the input-files link
                into the output rows for provenance; never fetched, never graded.

--models        (optional) comma list to restrict which model subfolders are scored.
                Default: every model subfolder found under each task.

OUTPUT (--out)
--------------
One row per (task, model, run):
    task_id, model, run, run_label, label_mismatch,
    n_crux, n_crux_passed, crux_passed_ids,
    chain, gear_lambda0,                         <- the headline metrics
    crux_pass_ratio, all_verifier_pass_ratio,
    graded_file, rejected_files, scratch_fallback,
    input_files_link, deliverable_chars, error
Plus a sibling *_per_verifier.csv with one row per (task, model, run, verifier):
    task_id, model, run, verifier_id, is_crux, verdict, unobservable, reason

GRADING
-------
Verdicts come from an LLM judge (Anthropic; needs the API key configured exactly
as the rest of the repo expects — see src.config.configure_api). For each verifier
the judge sees the deliverable text and the verifier's check text + expected value,
and returns PASS/FAIL. This reuses score_task() unchanged, so the file-selection,
prompt, and metric math are identical to the repo's grader — minus any reference.

USAGE
-----
    python score_end_to_end.py \
        --augment-dir  ./augment_jsons \
        --output-dir   ./model_outputs \
        --input-csv    ./finance_final_tasks_slim.csv \
        --out          ./scores.csv \
        --workers 6
"""
import argparse
import csv
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import configure_api
from src.prompt_evaluator import DEFAULT_JUDGE_MODEL
from src.score_task import score_task
from src.ingest_folders import ingest_augment_folder, group_by_task, run_label_num


# ---------------------------------------------------------------- input CSV link
def load_input_links(input_csv):
    """task_id -> input-files Drive link, for provenance stamping only."""
    if not input_csv or not os.path.exists(input_csv):
        return {}
    links = {}
    with open(input_csv, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            tid = (r.get("task_id") or "").strip()
            link = (r.get("Drive Link") or r.get("input_files_link")
                    or r.get("input_files_folder") or "").strip()
            if tid:
                links[tid] = link
    return links


# ---------------------------------------------------------- output-folder lookup
def find_run_dir(output_dir, task_id, model, run_num):
    """<output-dir>/<task_id>/<model>/<run>/ — tolerant of a few run-folder spellings
    ('1', 'run1', 'run_1', 'pass1'). Returns the leaf dir or None."""
    base = os.path.join(output_dir, task_id, model)
    if not os.path.isdir(base):
        return None
    candidates = [str(run_num), f"run{run_num}", f"run_{run_num}",
                  f"pass{run_num}", f"pass_{run_num}", f"r{run_num}"]
    for c in candidates:
        p = os.path.join(base, c)
        if os.path.isdir(p):
            return p
    # A single unnamed run folder is only accepted when the package itself has a
    # single run; with multiple runs we must NOT borrow another run's folder, so a
    # missing run folder is reported as missing.
    subs = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    if len(subs) == 1 and run_num in (1, None):
        return os.path.join(base, subs[0])
    return None


def list_deliverables(run_dir):
    """Every file in the leaf run folder is a candidate deliverable."""
    if not run_dir:
        return []
    return [os.path.join(run_dir, f) for f in sorted(os.listdir(run_dir))
            if os.path.isfile(os.path.join(run_dir, f))]


def discover_models(output_dir, task_id, restrict):
    base = os.path.join(output_dir, task_id)
    if not os.path.isdir(base):
        return []
    models = [d for d in sorted(os.listdir(base))
              if os.path.isdir(os.path.join(base, d))]
    if restrict:
        models = [m for m in models if m in restrict]
    return models


# --------------------------------------------------------------------- one score
def score_one(pkg, model, run_dir, input_link, judge_model):
    """Grade a single (task, model, run). Returns (summary_row, per_verifier_rows)."""
    augmented = pkg.to_augmented()
    deliverables = list_deliverables(run_dir)
    response = {
        "task_id": pkg.task_id,
        "model": model,
        "run_id": pkg.embedded_label,
        "pass_index": pkg.embedded_run_num or pkg.file_run_num or 0,
        "output_files": deliverables,
    }
    # score_task resolves files, selects the deliverable, LLM-grades every verifier,
    # and computes CHAIN/GEAR0 — all with no reference score.
    res = score_task(augmented, response, model_name=judge_model)

    graded = [g["file"] for g in res.graded_files if g["role"] == "graded"]
    rejected = [g["file"] for g in res.graded_files if g["role"] == "rejected"]
    summary = {
        "task_id": pkg.task_id,
        "model": model,
        "run": pkg.embedded_run_num or pkg.file_run_num or 0,
        "run_label": pkg.embedded_label,
        "label_mismatch": int(pkg.label_mismatch),
        "n_crux": res.n_crux,
        "n_crux_passed": res.n_passed,
        "crux_passed_ids": ";".join(res.crux_passed_ids),
        "chain": res.chain,
        "gear_lambda0": res.gear_lambda0,
        "crux_pass_ratio": round(res.crux_verifier_pass_ratio, 4),
        "all_verifier_pass_ratio": round(res.all_verifier_pass_ratio, 4),
        "graded_file": ";".join(graded),
        "rejected_files": ";".join(rejected),
        "scratch_fallback": int(res.scratch_fallback),
        "input_files_link": input_link,
        "deliverable_chars": res.deliverable_chars,
        "error": res.error,
    }
    per = []
    crux_set = set(augmented.get("crux_verifier_ids", []))
    for pv in res.per_verifier:
        # per_verifier records store the mark under 'result' (PASS/FAIL); the
        # judge's rationale is under 'why'.
        per.append({
            "task_id": pkg.task_id, "model": model,
            "run": summary["run"], "verifier_id": pv.get("id", ""),
            "is_crux": int(pv.get("is_crux", pv.get("id", "") in crux_set)),
            "verdict": pv.get("result", ""),
            "unobservable": int(bool(pv.get("unobservable"))),
            "reason": (pv.get("why", "") or "")[:300],
        })
    return summary, per


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="End-to-end verifier scorer (no reference score).")
    ap.add_argument("--augment-dir", required=True, help="folder of {task}__*__run{N}.json packages")
    ap.add_argument("--output-dir", required=True, help="root of <task>/<model>/<run>/ deliverables")
    ap.add_argument("--input-csv", default="", help="batch CSV, for input-link provenance only")
    ap.add_argument("--out", default="scores.csv")
    ap.add_argument("--models", default="", help="comma list to restrict models")
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true",
                    help="list the (task,model,run) plan and exit without grading")
    a = ap.parse_args()

    if not a.dry_run:
        configure_api()

    pkgs = ingest_augment_folder(local_dir=a.augment_dir)
    by_task = group_by_task(pkgs)
    restrict = set(m.strip() for m in a.models.split(",") if m.strip())
    links = load_input_links(a.input_csv)

    # Build the work plan: every (task-run package) x (model subfolder that exists)
    plan = []
    for task_id, tpkgs in by_task.items():
        models = discover_models(a.output_dir, task_id, restrict)
        if not models:
            print(f"  ! no model subfolders under {a.output_dir}/{task_id} — skipping")
        for pkg in tpkgs:
            run_num = pkg.embedded_run_num or pkg.file_run_num
            for model in models:
                run_dir = find_run_dir(a.output_dir, task_id, model, run_num)
                plan.append((pkg, model, run_dir, links.get(task_id, "")))
                if pkg.label_mismatch:
                    print(f"  ~ {task_id} {model} run{run_num}: "
                          f"filename/label mismatch — using embedded '{pkg.embedded_label}'")
                if run_dir is None:
                    print(f"  ! {task_id}/{model}/run{run_num}: no deliverable folder found")

    print(f"planned {len(plan)} (task,model,run) gradings across {len(by_task)} tasks")
    if a.dry_run:
        for pkg, model, run_dir, _ in plan:
            print(f"    {pkg.task_id:22} {model:14} run{pkg.embedded_run_num or pkg.file_run_num}  "
                  f"-> {run_dir or '(missing)'}")
        return

    summaries, per_rows = [], []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(score_one, pkg, model, run_dir, link, a.judge_model): (pkg, model)
                for (pkg, model, run_dir, link) in plan}
        for fut in as_completed(futs):
            pkg, model = futs[fut]
            try:
                s, per = fut.result()
                summaries.append(s)
                per_rows.extend(per)
                tag = "ok" if not s["error"] else f"ERR: {s['error'][:60]}"
                print(f"  scored {s['task_id']:22} {model:14} run{s['run']}  "
                      f"CHAIN={s['chain']:>6} GEAR0={s['gear_lambda0']:>6}  {tag}")
            except Exception as e:
                print(f"  FAILED {pkg.task_id} {model}: {e}")

    summaries.sort(key=lambda r: (r["task_id"], r["model"], r["run"]))
    per_rows.sort(key=lambda r: (r["task_id"], r["model"], r["run"], r["verifier_id"]))

    sum_cols = ["task_id", "model", "run", "run_label", "label_mismatch",
                "n_crux", "n_crux_passed", "crux_passed_ids",
                "chain", "gear_lambda0", "crux_pass_ratio", "all_verifier_pass_ratio",
                "graded_file", "rejected_files", "scratch_fallback",
                "input_files_link", "deliverable_chars", "error"]
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sum_cols)
        w.writeheader()
        w.writerows(summaries)

    per_path = a.out.rsplit(".", 1)[0] + "_per_verifier.csv"
    per_cols = ["task_id", "model", "run", "verifier_id", "is_crux",
                "verdict", "unobservable", "reason"]
    with open(per_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=per_cols)
        w.writeheader()
        w.writerows(per_rows)

    print(f"\nwrote {a.out} ({len(summaries)} runs) and {per_path} ({len(per_rows)} verifier rows)")


if __name__ == "__main__":
    main()
