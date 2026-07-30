#!/usr/bin/env python3
"""
rebuild_csv.py — regenerate output/augmented/augmented_prompt_packages.csv from
the per-task {task_id}_augment.json dumps, WITHOUT re-calling Opus.

Requires that run_augment.py has written {task_id}_augment.json files (added as a
permanent safeguard). Merges each original prompt-CSV row (matched by task_id)
with the augmented fields from its JSON.

    python rebuild_csv.py --csv prompt_data.csv --aug-dir output/augmented
    python rebuild_csv.py --csv prompt_data.csv --aug-dir output/augmented \
        --out output/augmented/augmented_prompt_packages.csv
"""

import argparse, csv, glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.auditor import build_header_map, get_field, HeaderError

# JSON field (AugmentResult.to_dict) -> CSV column. Mirrors run_augment.NEW_COLS.
# Values that are dict/list are json.dumps'd; scalars pass through.
JSON_TO_CSV = [
    ("corrected_solution_logic",     "corrected_solution_logic", False),
    ("gold_deliverable_text",        "golden_deliverable",       False),
    ("augmented_verifiers_text",     "augmented_verifiers",      False),
    ("dag",                          "dag_json",                 True),
    ("crux_ids",                     "crux_verifier_ids",        True),
    ("crux_shapley_weights",         "crux_shapley_weights_json",True),
    ("base_weights",                 "base_weights_json",        True),
    ("expected_values",              "expected_values_json",     True),
    ("audit_verdict",                "audit_verdict",            False),
    ("changes_applied",              "changes_applied_json",     True),
    ("judgment_changes_pending_sme", "judgment_pending_json",    True),
    ("skipped_inputs",               "skipped_inputs",           True),
    ("scoreable",                    "scoreable",                False),
    ("not_scoreable_reason",         "not_scoreable_reason",     False),
    ("error",                        "augment_error",            False),
]
NEW_COLS = [csv_col for _, csv_col, _ in JSON_TO_CSV]


def load_rows(path):
    with open(path, encoding="utf-8-sig") as f:  # BOM-safe (see handoff §6)
        r = csv.DictReader(f)
        raw_fields = r.fieldnames or []
        rows = list(r)
    fields = [h for h in raw_fields if h and h.strip()]
    if len(fields) != len(raw_fields):
        for row in rows:
            for k in list(row.keys()):
                if k is None or not str(k).strip():
                    row.pop(k, None)
    return fields, rows


def _norm(s):
    return (s or "").lstrip("\ufeff").strip().lower()


def main():
    ap = argparse.ArgumentParser(description="Rebuild augmented CSV from JSON dumps")
    ap.add_argument("--csv", required=True, help="original prompt CSV")
    ap.add_argument("--aug-dir", default="output/augmented")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    headers, rows = load_rows(args.csv)
    try:
        hmap = build_header_map(headers)
    except HeaderError as e:
        print(f"ERROR: {e}"); sys.exit(2)

    # index original rows by normalized task_id
    by_id = {}
    for row in rows:
        tid = _norm(get_field(row, hmap, "task_id"))
        if tid:
            by_id[tid] = row

    out_fields = list(headers) + [c for c in NEW_COLS if c not in headers]
    out_rows, missing_orig, n = [], [], 0

    for jpath in sorted(glob.glob(os.path.join(args.aug_dir, "*_augment.json"))):
        try:
            rd = json.load(open(jpath, encoding="utf-8"))
        except Exception as e:
            print(f"  skip {os.path.basename(jpath)}: unreadable ({e})")
            continue
        tid = _norm(rd.get("task_id"))
        base = dict(by_id.get(tid, {}))
        if not base:
            missing_orig.append(rd.get("task_id"))
            base = {"task_id": rd.get("task_id")}  # still emit augmented fields

        for jkey, ccol, is_json in JSON_TO_CSV:
            val = rd.get(jkey, "" if not is_json else None)
            base[ccol] = json.dumps(val, ensure_ascii=False) if is_json else (val or "")
        out_rows.append(base)
        n += 1

    out_csv = args.out or os.path.join(args.aug_dir, "augmented_prompt_packages.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    print(f"Rebuilt {n} row(s) -> {out_csv} ({len(out_fields)} columns)")
    if missing_orig:
        print(f"  WARN: {len(missing_orig)} task(s) had a JSON but no original CSV row: "
              f"{missing_orig[:5]}{' ...' if len(missing_orig) > 5 else ''}")


if __name__ == "__main__":
    main()