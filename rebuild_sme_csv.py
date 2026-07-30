#!/usr/bin/env python3
"""
rebuild_sme_csv.py — regenerate the blind SME delivery CSV from the CURRENT
state of Drive, with split verifier columns.

Does NOT upload anything. It walks the existing parent -> tsk_XXXX -> Model_A/B
folders, reads the real (already-anonymized) file names + IDs live, and joins the
task text fields from the source CSV + augment JSONs. Safe to run repeatedly.

Generalized: no hardcoded model names, paths, or Drive IDs. The leak-check and
optional label validation derive the real provider names from --key (the
model_key.csv written by upload_sme_batch/build_manifest), so they work for any
provider set — not just doubao/hunyuan.

    python rebuild_sme_csv.py --parent <SHARED_DRIVE_ID> --shared-drive \\
        --aug-dir ./runs_dir/output_IB/augmented \\
        --src-csv ./SME_data/IB_SME_MasterSheet_New_Tasks.csv \\
        --key model_key.csv --out sme_shortlist_with_links.csv
"""
import argparse, csv, json, os, re, sys

FOLDER_MIME = "application/vnd.google-apps.folder"

def get_service():
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    key = os.environ.get("GOOGLE_SERVICE_ACCOUNT_KEY") or \
          os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    creds = service_account.Credentials.from_service_account_file(
        key, scopes=["https://www.googleapis.com/auth/drive"])
    return build("drive", "v3", credentials=creds)

def kw(shared):
    return {"includeItemsFromAllDrives": True, "supportsAllDrives": True} if shared else {}

def children(svc, parent, shared):
    out, token = [], None
    while True:
        r = svc.files().list(q=f"'{parent}' in parents and trashed=false",
                             fields="nextPageToken, files(id,name,mimeType)",
                             pageToken=token, pageSize=200, **kw(shared)).execute()
        out += r.get("files", []); token = r.get("nextPageToken")
        if not token: break
    return out

def file_link(fid): return f"https://drive.google.com/file/d/{fid}/view"
def folder_link(fid): return f"https://drive.google.com/drive/folders/{fid}"

def corrected_sanity_check(original, aug):
    text = original or ""
    for c in aug.get("changes_applied", []):
        if c.get("artifact") != "sanity_check": continue
        if (c.get("type") or "").upper() != "MECHANICAL": continue
        old, new = c.get("old") or "", c.get("new") or ""
        if old and old in text: text = text.replace(old, new)
    return text

def _resolve_col(fieldnames, *candidates):
    low = {c.lower().strip(): c for c in (fieldnames or [])}
    for cand in candidates:
        hit = low.get(cand.lower())
        if hit:
            return hit
    return None


def load_task_fields(csv_path):
    """task_id -> prompt-package fields. Self-contained csv.DictReader with
    case-insensitive header resolution (handles 'Sub Domain', drive-link
    variants). No run_augment/src.auditor dependency."""
    if not csv_path or not os.path.exists(csv_path):
        print(f"  (source CSV not found: {csv_path}; task fields blank)")
        return {}
    try:
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            r = csv.DictReader(f)
            fn = r.fieldnames or []
            c = {k: _resolve_col(fn, *v) for k, v in {
                "task_id": ("task_id", "task id", "taskid"),
                "domain": ("domain", "sub domain", "sub-domain", "subdomain"),
                "verifiers": ("verifiers", "verifier"),
                "prompt": ("prompt",),
                "sanity_check": ("sanity check", "sanity_check"),
                "solution_logic": ("solution logic", "solution_logic"),
                "drive_link": ("drive link", "drive_link", "google drive", "input files"),
            }.items()}
            if not c["task_id"]:
                print("  (source CSV: no task_id column; task fields blank)")
                return {}
            def v(row, k): return (row.get(c[k]) or "").strip() if c[k] else ""
            out = {}
            for row in r:
                tid = v(row, "task_id")
                if tid:
                    out[tid] = {"domain": v(row, "domain"), "verifiers": v(row, "verifiers"),
                                "prompt": v(row, "prompt"), "sanity_check": v(row, "sanity_check"),
                                "solution_logic": v(row, "solution_logic"),
                                "input_files_link": v(row, "drive_link")}
        return out
    except Exception as e:
        print(f"  (source CSV load skipped: {e})")
        return {}


def load_key(key_path):
    """Read model_key.csv -> ({anon_label: real_name}, [real_names]). Used to
    validate folder labels and to derive provider names for the blindness
    leak-check. Returns ({}, []) if not supplied."""
    if not key_path or not os.path.exists(key_path):
        return {}, []
    label2real = {}
    with open(key_path, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        fn = r.fieldnames or []
        c_anon = _resolve_col(fn, "anon", "anon_model", "label")
        c_real = _resolve_col(fn, "model", "provider", "real", "real_model")
        for row in r:
            a = (row.get(c_anon) or "").strip() if c_anon else ""
            m = (row.get(c_real) or "").strip() if c_real else ""
            if a and m:
                label2real[a] = m
    return label2real, [m for m in label2real.values() if m]

def main():
    ap = argparse.ArgumentParser(description="Rebuild the blind SME CSV from live Drive state")
    ap.add_argument("--parent", required=True)
    ap.add_argument("--shared-drive", action="store_true")
    ap.add_argument("--aug-dir", default="./runs_dir/output_IB/augmented",
                    help="dir of <task>_augment.json")
    ap.add_argument("--src-csv", required=True, help="source prompt-package CSV")
    ap.add_argument("--key", default=None,
                    help="model_key.csv (validates labels + derives provider names "
                         "for the blindness leak-check)")
    ap.add_argument("--out", default="sme_shortlist_with_links.csv")
    args = ap.parse_args()
    shared = args.shared_drive
    svc = get_service()
    meta = load_task_fields(args.src_csv)
    label2real, real_names = load_key(args.key)
    if label2real:
        print(f"key: {len(label2real)} label(s) -> {label2real}")

    rows_out = []
    seen_labels = set()
    for tf_folder in children(svc, args.parent, shared):
        if tf_folder["mimeType"] != FOLDER_MIME: continue
        task = tf_folder["name"]
        if not task.startswith("tsk_"): continue
        tf = meta.get(task, {})
        augp = os.path.join(args.aug_dir, f"{task}_augment.json")
        try:
            aug = json.load(open(augp, encoding="utf-8")) if os.path.exists(augp) else {}
        except Exception as e:
            aug = {}; print(f"  ! {task}: augment unreadable ({e})")
        crux = len(aug.get("crux_ids", []))

        for run_folder in children(svc, tf_folder["id"], shared):
            if run_folder["mimeType"] != FOLDER_MIME: continue
            anon = run_folder["name"]        # Model_A / Model_B
            seen_labels.add(anon)
            files = [c for c in children(svc, run_folder["id"], shared)
                     if c["mimeType"] != FOLDER_MIME]
            resp_cell = "\n".join(f"{c['name']}: {file_link(c['id'])}"
                                  for c in sorted(files, key=lambda x: x["name"]))
            rows_out.append({
                "task_id": task, "domain": tf.get("domain",""), "crux": crux,
                "anon_model": anon,
                "prompt": tf.get("prompt",""),
                "sanity_check": tf.get("sanity_check",""),
                "solution_logic": tf.get("solution_logic",""),
                "input_files_link": tf.get("input_files_link",""),
                "run_folder_link": folder_link(run_folder["id"]),
                "response_and_output_files": resp_cell,
                "golden_deliverable": aug.get("gold_deliverable_text",""),
                "golden_solution_logic": aug.get("corrected_solution_logic",""),
                "corrected_sanity_check": corrected_sanity_check(tf.get("sanity_check",""), aug),
                "original_verifiers": tf.get("verifiers",""),
                "augmented_verifiers": aug.get("augmented_verifiers_text",""),
            })

    rows_out.sort(key=lambda r: (r["task_id"], r["anon_model"]))
    cols = ["task_id","domain","crux","anon_model","prompt","sanity_check",
            "solution_logic","input_files_link","run_folder_link",
            "response_and_output_files","golden_deliverable","golden_solution_logic",
            "corrected_sanity_check","original_verifiers","augmented_verifiers"]
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig: preserve ₹ etc.
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows_out)
    print(f"wrote {len(rows_out)} row(s) -> {args.out}")

    # optional label validation against the key
    if label2real:
        expected = set(label2real)
        extra = seen_labels - expected
        missing = expected - seen_labels
        if extra:
            print(f"  ! labels on Drive not in key: {sorted(extra)}")
        if missing:
            print(f"  ! labels in key not seen on Drive: {sorted(missing)}")
        if not extra and not missing:
            print(f"  label check: OK (Drive labels match key: {sorted(seen_labels)})")

    # blindness leak-check: derive the real provider names from --key when given,
    # else fall back to scanning for ANY of the seen labels' typical model tokens.
    # No hardcoded doubao/hunyuan — works for any provider set.
    if real_names:
        pat = "|".join(re.escape(n) for n in real_names if n)
        leaks = [r["task_id"] for r in rows_out
                 if pat and re.search(pat, r["response_and_output_files"], re.I)]
        print(f"blindness leak-check (providers from key: {sorted(real_names)}): "
              f"{len(leaks)} row(s) leaking a provider name"
              + (f"  {sorted(set(leaks))}" if leaks else "  (clean)"))
    else:
        print("blindness leak-check: SKIPPED (pass --key model_key.csv to enable; "
              "cannot derive provider names to scan for without it)")

if __name__ == "__main__":
    main()