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
    "judgment_pending_json", "skipped_inputs", "scoreable", "not_scoreable_reason",
    "augment_error",
]


def _blank_new_cols(row):
    """Guarantee every augmented column exists on a row (even on failure) so the
    CSV never silently loses columns for rows that hit the except branch."""
    for c in NEW_COLS:
        row.setdefault(c, "")
    return row


def load_rows(path):
    with open(path, encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        raw_fields = r.fieldnames or []
        rows = list(r)
    # Drop phantom columns from trailing commas in the source header (Excel
    # exports often leave a long run of empty-named columns). Keep only named
    # columns; strip the corresponding keys from every row so they don't
    # re-serialize into the output CSV.
    fields = [h for h in raw_fields if h and h.strip()]
    dropped = [h for h in raw_fields if not (h and h.strip())]
    if dropped:
        print(f"[load] dropped {len(dropped)} empty/unnamed source column(s)",
              file=sys.stderr)
        for row in rows:
            for k in list(row.keys()):
                if k is None or not str(k).strip():
                    row.pop(k, None)
    return fields, rows


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
        files_text, files_names, skipped = "", [], []
        if not args.no_files:
            dl = get_field(row, hmap, "drive_link")
            if dl.strip():
                try:
                    from src.gdrive_raw_fetcher import fetch_gdrive_folder_raw
                    files_text, files_names, skipped = fetch_gdrive_folder_raw(dl)
                    if skipped:
                        print(f"    !! SKIPPED INPUTS (golden may be degraded): {skipped}")
                except Exception as e:
                    print(f"    drive fetch failed: {e}")

        try:
            res = augment_task(row, hmap, input_files_text=files_text,
                               input_files_names=files_names, model_name=model,
                               skipped_inputs=skipped)
            rd = res.to_dict()
        except Exception as e:
            print(f"    AUGMENT FAILED: {e}")
            _blank_new_cols(row)
            row["skipped_inputs"] = json.dumps(skipped, ensure_ascii=False)
            row["scoreable"] = False
            row["not_scoreable_reason"] = f"augment crashed: {e}"
            row["augment_error"] = str(e)
            out_rows.append(row); continue

        # permanent safeguard: lossless per-task JSON so the CSV can always be
        # rebuilt (via rebuild_csv.py) without re-calling Opus.
        try:
            json.dump(rd, open(os.path.join(args.out_dir, f"{task_id}_augment.json"),
                               "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"    warn: could not write {task_id}_augment.json: {e}")

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
        row["skipped_inputs"] = json.dumps(rd.get("skipped_inputs", []), ensure_ascii=False)
        row["scoreable"] = rd.get("scoreable", True)
        row["not_scoreable_reason"] = rd.get("not_scoreable_reason", "")
        row["augment_error"] = rd.get("error", "")
        out_rows.append(row)

        if not args.no_html:
            write_augment_report(rd, os.path.join(args.out_dir, f"{task_id}_augment.html"))
            write_golden_report(rd, os.path.join(args.out_dir, f"{task_id}_golden.html"))

        print(f"    -> {rd['audit_verdict']} | crux {len(rd['crux_ids'])} verifiers"
              + (f" | ERROR {rd['error']}" if rd.get("error") else ""))

    out_csv = os.path.join(args.out_dir, "augmented_prompt_packages.csv")
    # Diagnostic: confirm rows carry the augmented keys before writing.
    if out_rows:
        present = [c for c in NEW_COLS if c in out_rows[0]]
        print(f"[debug] out_rows={len(out_rows)} | new cols on row0: "
              f"{len(present)}/{len(NEW_COLS)} | header cols={len(out_fields)}")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"\nWrote {len(out_rows)} row(s) -> {out_csv}")


if __name__ == "__main__":
    main()