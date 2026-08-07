#!/usr/bin/env python3
"""
finalize_tasks.py — one-command wrapper: apply decisions for many tasks, then build the final CSV.

The manual pipeline runs apply_decisions.py once per task (it only accepts a single
--augment / --decisions pair), then build_final_csv.py once. This wrapper does both in
one call:

  1. pair every <task_id>_augment.json in --augment-dir with the decisions_<task_id>_*.json
     in --decisions-dir that carries the same task_id,
  2. run the apply step for each pair, writing <task_id>_final.json into --final-dir,
  3. call build_final_csv.py on --final-dir to produce the final CSV.

No changes are needed to the three underlying tools. apply_decisions is imported and called
in-process; build_final_csv is invoked exactly as it is on the command line, so its output is
identical to the manual path.

Example:
  python finalize_tasks.py \\
    --augment-dir ./output/augmented \\
    --decisions-dir ./decisions \\
    --csv ./SME_data/tasks.csv \\
    --out ./output/final_tasks.csv
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import tempfile

import apply_decisions as ad  # apply_decisions.py must be importable (same folder)

HERE = os.path.dirname(os.path.abspath(__file__))


def _task_id_from_json(path):
    """Authoritative task_id: read it from inside the file."""
    try:
        with open(path, encoding="utf-8") as f:
            return str(json.load(f).get("task_id") or "").strip()
    except Exception:
        return ""


def index_augments(augment_dir):
    """{task_id: augment_path} for every *_augment.json in the folder."""
    out = {}
    for p in sorted(glob.glob(os.path.join(augment_dir, "*_augment.json"))):
        tid = _task_id_from_json(p)
        if not tid:
            # fall back to filename stem if the file has no task_id field
            m = re.match(r"(.+)_augment\.json$", os.path.basename(p))
            tid = m.group(1) if m else ""
        if tid:
            out[tid] = p
    return out


def index_decisions(decisions_dir):
    """{task_id: decisions_path}. If several decision files exist for one task,
    keep the newest by modification time (SMEs may re-save)."""
    out = {}
    for p in sorted(glob.glob(os.path.join(decisions_dir, "decisions_*.json"))):
        tid = _task_id_from_json(p)
        if not tid:
            # decisions_<task_id>_<date>.json -> strip prefix and trailing _<date>
            base = os.path.basename(p)[len("decisions_"):-len(".json")]
            base = re.sub(r"_\d{4}-\d{2}-\d{2}$", "", base)
            tid = base
        if not tid:
            continue
        if tid not in out or os.path.getmtime(p) > os.path.getmtime(out[tid]):
            out[tid] = p
    return out


def run(argv=None):
    ap = argparse.ArgumentParser(
        description="Apply decisions for many tasks and build the final CSV in one go.")
    ap.add_argument("--augment-dir", required=True,
                    help="folder of <task_id>_augment.json files (run_augment output)")
    ap.add_argument("--decisions-dir", required=True,
                    help="folder of decisions_<task_id>_<date>.json files saved by SMEs")
    ap.add_argument("--csv", required=True,
                    help="the original authoring csv (passed through to build_final_csv)")
    ap.add_argument("--out", required=True, help="path of the final CSV to write")
    ap.add_argument("--final-dir", default=None,
                    help="where to place the intermediate *_final.json files "
                         "(default: <out folder>/_final)")
    ap.add_argument("--slim", default=None,
                    help="also write a narrower review sheet here "
                         "(passed through to build_final_csv)")
    ap.add_argument("--only-sealed", action="store_true",
                    help="write only sealed tasks to the CSV (passed through)")
    ap.add_argument("--force", action="store_true",
                    help="apply despite incomplete/mismatched decisions (per-task)")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="skip a task whose apply step fails instead of stopping")
    args = ap.parse_args(argv)

    for d, what in [(args.augment_dir, "augment-dir"), (args.decisions_dir, "decisions-dir")]:
        if not os.path.isdir(d):
            raise SystemExit(f"ERROR: --{what} is not a directory: {d}")

    final_dir = args.final_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.out)) or ".", "_final")
    os.makedirs(final_dir, exist_ok=True)

    augs = index_augments(args.augment_dir)
    decs = index_decisions(args.decisions_dir)
    if not augs:
        raise SystemExit(f"ERROR: no *_augment.json found in {args.augment_dir}")

    paired = sorted(set(augs) & set(decs))
    aug_only = sorted(set(augs) - set(decs))   # awaiting SME decisions
    dec_only = sorted(set(decs) - set(augs))   # decisions with no matching augment

    print(f"augment tasks : {len(augs)}")
    print(f"decision files: {len(decs)}")
    print(f"paired        : {len(paired)}")
    if aug_only:
        print(f"  awaiting decisions ({len(aug_only)}): {aug_only}")
    if dec_only:
        print(f"  decisions with no augment ({len(dec_only)}): {dec_only}")

    if not paired:
        raise SystemExit("no augment/decisions pairs to apply — nothing to do")

    # ---- step 1: apply decisions for each pair (apply_decisions is single-task) ----
    sealed_ok, failed = [], []
    for tid in paired:
        try:
            pkg = ad.load(augs[tid])
            dec = ad.load(decs[tid])
            if pkg.get("task_id") != dec.get("task_id"):
                raise ValueError(
                    f"task mismatch inside files: "
                    f"{pkg.get('task_id')} vs {dec.get('task_id')}")
            sealed = ad.apply_decisions(pkg, dec, force=args.force)
            out_path = os.path.join(final_dir, f"{sealed['task_id']}_final.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(sealed, f, indent=2, ensure_ascii=False, default=str)
            state = ("scoreable" if sealed.get("scoreable")
                     else f"not scoreable: {sealed.get('not_scoreable_reason')}")
            print(f"  [applied] {tid} -> {os.path.basename(out_path)}  ({state})")
            sealed_ok.append(tid)
        except Exception as e:  # noqa: BLE001 - want the task id with the error
            print(f"  [FAILED ] {tid}: {e}")
            failed.append((tid, str(e)))
            if not args.continue_on_error:
                raise SystemExit(
                    f"stopping: apply failed for {tid} "
                    f"(use --continue-on-error to skip and go on)")

    if not sealed_ok:
        raise SystemExit("no tasks were sealed — not building a CSV")

    # ---- step 2: build the final CSV from the folder of *_final.json ----
    # build_final_csv already accepts a directory for --final and loops over the csv.
    cmd = [sys.executable, os.path.join(HERE, "build_final_csv.py"),
           "--csv", args.csv, "--final", final_dir, "--out", args.out]
    if args.only_sealed:
        cmd.append("--only-sealed")
    if args.slim:
        cmd += ["--slim", args.slim]
    print(f"\n$ {' '.join(cmd)}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"build_final_csv failed (exit {r.returncode})")

    print(f"\nDone. sealed {len(sealed_ok)}/{len(paired)} paired task(s)"
          + (f", {len(failed)} failed" if failed else "")
          + f". final json in {final_dir}, csv at {args.out}")
    if failed:
        print("failed tasks: " + ", ".join(t for t, _ in failed))
    return 0


if __name__ == "__main__":
    sys.exit(run())