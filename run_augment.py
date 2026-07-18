#!/usr/bin/env python3
"""
run_augment.py — augment prompt packages directly (no SME-approval gate).

For each row: audit -> apply corrections -> generate golden deliverable + DAG +
crux set + crux-only Shapley (single augment call) -> write an AUGMENTED CSV plus
per-task {task_id}_augment.html and {task_id}_golden.html.

    python run_augment.py --csv prompt_data.csv --out-dir output/augmented
    python run_augment.py --csv prompt_data.csv --row 1
    python run_augment.py --csv prompt_data.csv --from 1 --to 20 --no-files

The augmented CSV keeps every original column and ADDS:
  corrected_solution_logic, golden_deliverable, augmented_verifiers, dag_json,
  crux_verifier_ids, crux_shapley_weights_json, base_weights_json,
  audit_verdict, changes_applied_json, judgment_pending_json, augment_error
"""

import argparse, csv, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import configure_api
from src.auditor import build_header_map, get_field, HeaderError
from src.prompt_evaluator import DEFAULT_JUDGE_MODEL
from src.augment_task import augment_task
from src.augment_report import write_augment_report, write_golden_report

NEW_COLS = [
    "corrected_solution_logic", "golden_deliverable", "augmented_verifiers",
    "dag_json", "crux_verifier_ids", "crux_shapley_weights_json",
    "base_weights_json", "expected_values_json", "audit_verdict", "changes_applied_json",
    "judgment_pending_json", "augment_error",
]


def load_rows(path):
    with open(path, encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        return (r.fieldnames or []), list(r)


def main():
    ap = argparse.ArgumentParser(description="Direct prompt-package augmenter")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--row", type=int, default=None)
    ap.add_argument("--from", dest="row_from", type=int, default=None)
    ap.add_argument("--to", dest="row_to", type=int, default=None)
    ap.add_argument("--out-dir", default="output/augmented")
    ap.add_argument("--model", default=None)
    ap.add_argument("--no-files", action="store_true",
                    help="skip Drive input-file fetch (auditor provenance layer off)")
    ap.add_argument("--no-html", action="store_true")
    args = ap.parse_args()

    headers, rows = load_rows(args.csv)
    try:
        hmap = build_header_map(headers)
    except HeaderError as e:
        print(f"ERROR: {e}"); sys.exit(2)

    n = len(rows)
    if args.row is not None:
        idxs = [args.row]
    elif args.row_from or args.row_to:
        idxs = list(range((args.row_from or 1), (args.row_to or n) + 1))
    else:
        idxs = list(range(1, n + 1))
    idxs = [i for i in idxs if 1 <= i <= n]

    configure_api()
    os.makedirs(args.out_dir, exist_ok=True)
    model = args.model or DEFAULT_JUDGE_MODEL

    out_fields = list(headers) + [c for c in NEW_COLS if c not in headers]
    out_rows = []

    for i in idxs:
        row = dict(rows[i - 1])
        task_id = get_field(row, hmap, "task_id") or f"row{i}"
        print(f"[{i}] augmenting {task_id} ...", flush=True)

        # fetch input files (raw) so the auditor's provenance layer runs
        files_text, files_names = "", []
        if not args.no_files:
            dl = get_field(row, hmap, "drive_link")
            if dl.strip():
                try:
                    from src.gdrive_raw_fetcher import fetch_gdrive_folder_raw
                    files_text, files_names = fetch_gdrive_folder_raw(dl)
                except Exception as e:
                    print(f"    drive fetch failed: {e}")

        try:
            res = augment_task(row, hmap, input_files_text=files_text,
                               input_files_names=files_names, model_name=model)
            rd = res.to_dict()
        except Exception as e:
            print(f"    AUGMENT FAILED: {e}")
            row["augment_error"] = str(e)
            out_rows.append(row); continue

        # fill new columns
        row["corrected_solution_logic"] = rd["corrected_solution_logic"]
        row["golden_deliverable"] = rd["gold_deliverable_text"]
        row["augmented_verifiers"] = rd["augmented_verifiers_text"]
        row["dag_json"] = json.dumps(rd["dag"], ensure_ascii=False)
        row["crux_verifier_ids"] = json.dumps(rd["crux_ids"], ensure_ascii=False)
        row["crux_shapley_weights_json"] = json.dumps(rd["crux_shapley_weights"], ensure_ascii=False)
        row["base_weights_json"] = json.dumps(rd["base_weights"], ensure_ascii=False)
        row["expected_values_json"] = json.dumps(rd["expected_values"], ensure_ascii=False)
        row["audit_verdict"] = rd["audit_verdict"]
        row["changes_applied_json"] = json.dumps(rd["changes_applied"], ensure_ascii=False)
        row["judgment_pending_json"] = json.dumps(rd["judgment_changes_pending_sme"], ensure_ascii=False)
        row["augment_error"] = rd.get("error", "")
        out_rows.append(row)

        if not args.no_html:
            write_augment_report(rd, os.path.join(args.out_dir, f"{task_id}_augment.html"))
            write_golden_report(rd, os.path.join(args.out_dir, f"{task_id}_golden.html"))

        print(f"    -> {rd['audit_verdict']} | crux {len(rd['crux_ids'])} verifiers"
              + (f" | ERROR {rd['error']}" if rd.get("error") else ""))

    out_csv = os.path.join(args.out_dir, "augmented_prompt_packages.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"\nWrote {len(out_rows)} row(s) -> {out_csv}")


if __name__ == "__main__":
    main()