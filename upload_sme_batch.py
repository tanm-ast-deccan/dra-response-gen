#!/usr/bin/env python3
"""
upload_sme_batch.py — stage an SME blind-scoring batch to Google Drive.

Generalized: no model names, staging folders, task list, or Drive IDs are
hardcoded. Models are DISCOVERED (not declared): the results-trace JSON is the
source of truth for which models each task has, cross-checked against the
on-disk run folders, with mismatches reported. Anonymized labels (Model_A,
Model_B, ...) are assigned GLOBALLY and alphabetically across the whole batch,
so a task missing a model does not shift another task's lettering.

Filename anonymization is DATA-DRIVEN: for each discovered model it strips that
model's own (full) name from its filenames. It does NOT use short aliases —
a short alias like "hy" or "db" would match inside unrelated words (turning
"dashboard.xlsx" into "ashoard.xlsx"), silently corrupting legitimate filenames.
Stripping only the full model name avoids that class of bug.

Per task it creates <parent>/<task_id>/<Model_X>/ and uploads that model's files
with names stripped; it emits the SME delivery CSV and the private model_key.csv.

Dry-run by default; --commit to write. Idempotent via the upload manifest.

Auth: service account from GOOGLE_SERVICE_ACCOUNT_KEY (or
GOOGLE_APPLICATION_CREDENTIALS). The Drive parent must be shared with the SA.

Usage:
  python upload_sme_batch.py \
      --staging ./staging --results ./results/trace.json \
      --aug-dir ./output/augmented --src-csv prompt_data.csv \
      --parent <DRIVE_PARENT_ID> [--shared-drive] \
      [--tasks tsk_a,tsk_b] [--commit]
"""
import argparse, csv, glob, json, mimetypes, os, re, sys

FOLDER_MIME = "application/vnd.google-apps.folder"
P_GLOB = "*__p*"          # run-folder partition pattern: <model>__p1, <model>__p0, ...


# ============================================================================
# Discovery — models come from the trace (authoritative) + folder cross-check.
# ============================================================================

def tasks_from_trace(trace):
    """All task IDs in the trace, across shapes. Layout C: distinct task_id in
    the results list (NOT the top-level keys, which are csv/config/results/...)."""
    if isinstance(trace, dict) and isinstance(trace.get("results"), list):
        return sorted({(r.get("task_id") or r.get("task"))
                       for r in trace["results"]
                       if (r.get("task_id") or r.get("task"))})
    cont = trace.get("tasks", trace) if isinstance(trace, dict) else {}
    return sorted(cont.keys()) if isinstance(cont, dict) else []


def models_from_trace(trace, task_id):
    """Return the set of PROVIDER names the trace lists for a task (doubao,
    hunyuan, ...). Provider is the identity that names the run folders
    (<provider>__p1) and prefixes the files, so discovery agrees with the
    on-disk folders and with build_manifest's anonymization. Tolerant of a few
    trace shapes (see build_manifest.extract_from_trace)."""
    node = None
    if isinstance(trace, dict):
        node = (trace.get("tasks", {}) or {}).get(task_id) or trace.get(task_id)
    if isinstance(node, dict):
        models = node.get("models") or node.get("runs") or node
        if isinstance(models, dict):
            return set(models.keys())
    # Layout C (the real one): flat list of records. Key on PROVIDER, not model.
    if isinstance(trace, dict) and isinstance(trace.get("results"), list):
        return {r.get("provider") or r.get("model") or r.get("model_name")
                for r in trace["results"]
                if (r.get("task_id") or r.get("task")) == task_id}
    return set()


def models_from_folders(staging, task_id):
    """Discover model names from the on-disk run folders: the segment before
    '__p' in <staging>/<task_id>/runs/<model>__p*."""
    found = {}
    for rd in sorted(glob.glob(os.path.join(staging, task_id, "runs", P_GLOB))):
        base = os.path.basename(rd)
        m = base.split("__p")[0]
        if m:
            found.setdefault(m, []).append(rd)
    return found  # {model_name: [run_dir, ...]}


def discover(staging, trace, task_ids):
    """For every task, reconcile trace vs. folders. Returns:
        per_task: {task_id: {model: [run_dirs]}}   (folder truth, for uploading)
        all_models: sorted set across the batch
        warnings: list of mismatch strings
    """
    per_task, all_models, warnings = {}, set(), []
    for t in task_ids:
        want = {m for m in models_from_trace(trace, t) if m}
        have = models_from_folders(staging, t)
        have_set = set(have.keys())

        for m in want - have_set:
            warnings.append(f"{t}: trace lists model '{m}' but no run folder on disk")
        for m in have_set - want:
            warnings.append(f"{t}: run folder for '{m}' on disk but not in trace "
                            f"(orphan/stray run?)")

        # Upload what the trace authorizes AND exists on disk; if trace is empty
        # for this task, fall back to folder truth (with a warning).
        if want:
            use = {m: have[m] for m in (want & have_set)}
        else:
            warnings.append(f"{t}: no models in trace; falling back to folders")
            use = have
        per_task[t] = use
        all_models |= set(use.keys())
    return per_task, sorted(all_models), warnings


# ============================================================================
# Anonymization — data-driven: strip the model's own full name only.
# ============================================================================

def strip_model_name(filename, model):
    """Remove the model's own (full) name (case-insensitive) from a filename and
    tidy leftover separators. Uses the full name only — no short aliases — so it
    can't match inside unrelated words."""
    out = re.sub(rf'(?i){re.escape(model)}[_\-]*', '', filename)
    out = re.sub(r'__+', '_', out).lstrip('_-')
    return out or filename


def anon_labels(n):
    """Model_A, Model_B, ... Model_Z, Model_AA, ..."""
    import string
    labels = []
    for i in range(n):
        s, x = "", i
        while True:
            s = string.ascii_uppercase[x % 26] + s
            x = x // 26 - 1
            if x < 0:
                break
        labels.append(f"Model_{s}")
    return labels


def run_output_files(run_dirs, model):
    """All files under this model's run dirs. (No model-prefix filter — the
    folder is already model-specific; we anonymize names on upload.)"""
    files = []
    for rd in run_dirs:
        files += [x for x in sorted(glob.glob(os.path.join(rd, "*"))) if os.path.isfile(x)]
    return files


# ============================================================================
# Drive helpers (mirror the proven upload_sme_batch implementation)
# ============================================================================

def get_service():
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    key = os.environ.get("GOOGLE_SERVICE_ACCOUNT_KEY") or \
          os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not key or not os.path.exists(key):
        sys.exit("No service-account key. Set GOOGLE_SERVICE_ACCOUNT_KEY=/path/to/gcp_key.json")
    creds = service_account.Credentials.from_service_account_file(
        key, scopes=["https://www.googleapis.com/auth/drive"])
    return build("drive", "v3", credentials=creds)

def _kw(shared): return {"supportsAllDrives": True} if shared else {}


def with_retry(fn, *, what="drive call", tries=5, base=2.0, cap=30.0):
    """Run fn() with exponential backoff on transient network/HTTP errors.
    Retries TransportError, socket errors, and HTTP 429/5xx. Raises on the
    final failure or on non-transient errors (e.g. 404/403). Prints a countdown
    timer between attempts so a flaky network doesn't look like a hang."""
    import time, socket
    _http_errs = ()
    try:
        from googleapiclient.errors import HttpError
        _http_errs = (HttpError,)
    except Exception:
        HttpError = None
    _transient = [socket.gaierror, socket.timeout, ConnectionError, OSError]
    try:
        from google.auth.exceptions import TransportError
        _transient.append(TransportError)
    except Exception:
        pass
    transient_exc = tuple(_transient)
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except transient_exc as e:
            if attempt == tries:
                raise
            reason = type(e).__name__
        except Exception as e:
            # HTTP errors: retry only on 429/5xx; everything else re-raises.
            status = getattr(getattr(e, "resp", None), "status", None)
            is_http = HttpError is not None and isinstance(e, HttpError)
            if not is_http or status not in (429, 500, 502, 503, 504) or attempt == tries:
                raise
            reason = f"HTTP {status}"
        wait = min(cap, base * (2 ** (attempt - 1)))
        print(f"    {what}: transient failure ({reason}); "
              f"retry {attempt}/{tries - 1} in {wait:.0f}s", flush=True)
        # visible countdown so it doesn't look hung
        remaining = int(wait)
        while remaining > 0:
            print(f"      ...retrying in {remaining}s ", end="\r", flush=True)
            time.sleep(1)
            remaining -= 1
        print(" " * 40, end="\r")  # clear the line
    raise RuntimeError(f"{what}: exhausted {tries} attempts")

def ensure_folder(svc, parent, name, shared, manifest):
    key = f"folder::{parent}::{name}"
    if key in manifest:
        return manifest[key]
    q = (f"'{parent}' in parents and name = {json.dumps(name)} "
         f"and mimeType = '{FOLDER_MIME}' and trashed = false")
    lk = {"includeItemsFromAllDrives": True, "supportsAllDrives": True} if shared else {}
    fs = with_retry(lambda: svc.files().list(q=q, fields="files(id,name)", **lk).execute(),
                    what=f"list folder {name!r}").get("files", [])
    if fs:
        fid = fs[0]["id"]
    else:
        fid = with_retry(lambda: svc.files().create(
            body={"name": name, "mimeType": FOLDER_MIME, "parents": [parent]},
            fields="id", **_kw(shared)).execute(),
            what=f"create folder {name!r}")["id"]
    manifest[key] = fid
    return fid

def upload_file(svc, parent, path, upload_name, shared, manifest):
    key = f"file::{parent}::{upload_name}"
    if key in manifest:
        return manifest[key]
    # Query Drive for an existing file of this name in this folder BEFORE creating,
    # mirroring ensure_folder. Drive's create() does NOT overwrite by name — it
    # makes a duplicate — so without this check a lost/absent manifest would
    # silently double every file on re-run. Reuse the existing file if present.
    q = (f"'{parent}' in parents and name = {json.dumps(upload_name)} "
         f"and trashed = false")
    lk = {"includeItemsFromAllDrives": True, "supportsAllDrives": True} if shared else {}
    existing = with_retry(lambda: svc.files().list(q=q, fields="files(id,name)", **lk).execute(),
                          what=f"list file {upload_name!r}").get("files", [])
    if existing:
        fid = existing[0]["id"]
        manifest[key] = fid
        return fid
    from googleapiclient.http import MediaFileUpload
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    media = MediaFileUpload(path, mimetype=mime, resumable=True)
    fid = with_retry(lambda: svc.files().create(
        body={"name": upload_name, "parents": [parent]},
        media_body=media, fields="id", **_kw(shared)).execute(),
        what=f"upload {upload_name!r}")["id"]
    manifest[key] = fid
    return fid

def make_link_viewable(svc, fid, shared):
    try:
        with_retry(lambda: svc.permissions().create(
            fileId=fid, body={"type": "anyone", "role": "reader"}, **_kw(shared)).execute(),
            what="set link-sharing")
    except Exception as e:
        print(f"    warn: could not set link-sharing on {fid}: {e}")

def folder_link(fid): return f"https://drive.google.com/drive/folders/{fid}"
def file_link(fid):   return f"https://drive.google.com/file/d/{fid}/view"


# ============================================================================
# Task fields + augment (paths passed in)
# ============================================================================

def _resolve_col(fieldnames, *candidates):
    """Case-insensitive resolve of the first matching column name."""
    low = {c.lower().strip(): c for c in (fieldnames or [])}
    for cand in candidates:
        hit = low.get(cand.lower())
        if hit:
            return hit
    return None


def load_task_fields(csv_path):
    """task_id -> prompt-package fields from the source CSV.

    Self-contained: plain csv.DictReader with case-insensitive header resolution,
    no dependency on run_augment/src.auditor (which fail outside the repo).
    Resolves domain across "Domain"/"Sub Domain" and the drive-link variants.
    """
    if not csv_path or not os.path.exists(csv_path):
        return {}
    try:
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fn = reader.fieldnames or []
            c_tid = _resolve_col(fn, "task_id", "task id", "taskid")
            c_dom = _resolve_col(fn, "domain", "sub domain", "sub-domain", "subdomain")
            c_ver = _resolve_col(fn, "verifiers", "verifier")
            c_pr = _resolve_col(fn, "prompt")
            c_sc = _resolve_col(fn, "sanity check", "sanity_check")
            c_sl = _resolve_col(fn, "solution logic", "solution_logic")
            c_dl = _resolve_col(fn, "drive link", "drive_link", "google drive",
                                "input files", "input_files_link")
            if not c_tid:
                print("  (source-CSV: no task_id column found; task fields will be empty)")
                return {}
            def val(row, col):
                return (row.get(col) or "").strip() if col else ""
            out = {}
            for row in reader:
                tid = val(row, c_tid)
                if not tid:
                    continue
                out[tid] = {
                    "domain": val(row, c_dom),
                    "verifiers": val(row, c_ver),
                    "prompt": val(row, c_pr),
                    "sanity_check": val(row, c_sc),
                    "solution_logic": val(row, c_sl),
                    "input_files_link": val(row, c_dl),
                }
        return out
    except Exception as e:
        print(f"  (source-CSV load skipped: {e})")
        return {}


def load_aug(aug_dir, task):
    p = os.path.join(aug_dir, f"{task}_augment.json")
    if not os.path.exists(p):
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def corrected_sanity_check(original, aug):
    text = original or ""
    for c in aug.get("changes_applied", []):
        if c.get("artifact") != "sanity_check":
            continue
        if (c.get("type") or "").upper() != "MECHANICAL":
            continue
        old, new = c.get("old") or "", c.get("new") or ""
        if old and old in text:
            text = text.replace(old, new)
    return text


# ============================================================================
# main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="Generalized SME batch uploader")
    ap.add_argument("--staging", required=True, help="staging root: <staging>/<task>/runs/<model>__p*/")
    ap.add_argument("--results", required=True, help="combined run-trace JSON (or dir with one)")
    ap.add_argument("--aug-dir", required=True, help="dir of <task>_augment.json")
    ap.add_argument("--src-csv", required=True, help="source prompt-package CSV")
    ap.add_argument("--parent", required=True, help="Drive parent folder / Shared Drive ID")
    ap.add_argument("--shared-drive", action="store_true")
    ap.add_argument("--tasks", default=None, help="comma-separated task IDs (default: all in trace)")
    ap.add_argument("--out-dir", default=".", help="where to write the CSVs + manifest")
    ap.add_argument("--commit", action="store_true")
    args = ap.parse_args()

    results_path = args.results
    if os.path.isdir(results_path):
        js = sorted(glob.glob(os.path.join(results_path, "*.json")))
        if len(js) != 1:
            sys.exit(f"{results_path} has {len(js)} JSON files; point --results at the one trace file.")
        results_path = js[0]
    trace = json.load(open(results_path, encoding="utf-8"))

    # task list
    if args.tasks:
        task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
    else:
        task_ids = tasks_from_trace(trace)
        if not task_ids:
            sys.exit("No tasks found in trace and --tasks not given.")

    per_task, all_models, warnings = discover(args.staging, trace, task_ids)

    # GLOBAL alphabetical anonymization
    labels = anon_labels(len(all_models))
    anon = dict(zip(all_models, labels))          # real model -> Model_X
    print("Global model anonymization (alphabetical, discovered):")
    for m, a in anon.items():
        print(f"  {a} = {m}")
    if warnings:
        print(f"\n{len(warnings)} discovery warning(s):")
        for w in warnings:
            print(f"  ! {w}")
    print()

    meta = load_task_fields(args.src_csv)
    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "upload_manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    svc = get_service() if args.commit else None

    # Preflight: verify the parent is reachable BEFORE uploading 58 tasks, so a
    # bad/unshared parent or a shared-drive-flag mismatch fails in one clear line
    # instead of a mid-loop traceback on task 1.
    if args.commit:
        try:
            meta = with_retry(lambda: svc.files().get(
                fileId=args.parent, fields="id,name",
                supportsAllDrives=True).execute(), what="preflight parent")
            print(f"parent OK: {meta.get('name')!r} ({meta.get('id')})")
        except Exception as e:
            sys.exit(
                f"\nPARENT NOT REACHABLE: {args.parent}\n"
                f"  {e}\n"
                f"  If it's a Shared Drive, pass --shared-drive AND ensure the "
                f"service account is a member.\n"
                f"  If it's a folder, ensure it's shared with the service account."
            )

    print(f"{'COMMITTING' if args.commit else 'PLAN (dry-run)'} — {len(per_task)} task(s), parent={args.parent}\n")
    sme_rows = []

    def _persist_manifest():
        if args.commit:
            with open(manifest_path, "w") as mf:
                json.dump(manifest, mf, indent=2)

    try:
        for task in task_ids:
            models = per_task.get(task, {})
            aug = load_aug(args.aug_dir, task)
            tf = meta.get(task, {})
            domain = tf.get("domain", "")
            crux = len(aug.get("crux_ids", []))
            golden_text = aug.get("gold_deliverable_text", "")
            golden_logic = aug.get("corrected_solution_logic", "")
            corr_sanity = corrected_sanity_check(tf.get("sanity_check", ""), aug)
            verifiers_full = (f"=== ORIGINAL VERIFIERS ===\n{tf.get('verifiers','')}\n\n"
                              f"=== AUGMENTED (canonical) ===\n{aug.get('augmented_verifiers_text','')}").strip()

            task_fid = ensure_folder(svc, args.parent, task, args.shared_drive, manifest) if args.commit else "(task)"
            print(f"tsk {task}  [{domain}]  crux={crux}  models={sorted(anon[m] for m in models)}")

            for model in sorted(models.keys()):
                label = anon[model]
                files = run_output_files(models[model], model)
                if not files:
                    print(f"  {label}: NO FILES — skipping"); continue

                run_fid = ensure_folder(svc, task_fid, label, args.shared_drive, manifest) if args.commit else f"({label})"
                uploaded = []
                for fpath in files:
                    nn = strip_model_name(os.path.basename(fpath), model)
                    if args.commit:
                        fid = upload_file(svc, run_fid, fpath, nn, args.shared_drive, manifest)
                        uploaded.append((nn, file_link(fid)))
                    else:
                        uploaded.append((nn, "(link)"))
                if args.commit:
                    make_link_viewable(svc, run_fid, args.shared_drive)

                resp_cell = "\n".join(f"{n}: {l}" for n, l in uploaded)
                sme_rows.append({
                    "task_id": task, "domain": domain, "crux": crux, "anon_model": label,
                    "prompt": tf.get("prompt", ""), "sanity_check": tf.get("sanity_check", ""),
                    "solution_logic": tf.get("solution_logic", ""),
                    "input_files_link": tf.get("input_files_link", ""),
                    "run_folder_link": folder_link(run_fid) if args.commit else "(run-folder)",
                    "response_and_output_files": resp_cell,
                    "golden_deliverable": golden_text, "golden_solution_logic": golden_logic,
                    "corrected_sanity_check": corr_sanity, "verifiers_full": verifiers_full,
                })
                print(f"  {label}: {len(files)} file(s)  (stripped name: {model!r})")

            _persist_manifest()   # save progress after each task (idempotent resume)
    except (KeyboardInterrupt, Exception) as e:
        _persist_manifest()
        print(f"\nInterrupted/failed: {type(e).__name__}: {e}")
        print(f"Progress saved to {manifest_path} — re-run the SAME command to resume "
              f"(idempotent: existing folders/files are reused, not duplicated).")
        raise

    # CSVs
    cols = ["task_id","domain","crux","anon_model","prompt","sanity_check","solution_logic",
            "input_files_link","run_folder_link","response_and_output_files",
            "golden_deliverable","golden_solution_logic","corrected_sanity_check","verifiers_full"]
    sme_out = os.path.join(args.out_dir, "sme_shortlist_with_links.csv")
    with open(sme_out, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig: preserve ₹ etc.
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(sme_rows)
    key_out = os.path.join(args.out_dir, "model_key.csv")
    with open(key_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["anon_model", "real_model"])
        for m, a in sorted(anon.items(), key=lambda kv: kv[1]):
            w.writerow([a, m])

    print(f"\nrows (one per run): {len(sme_rows)}")
    print(f"wrote {sme_out}")
    print(f"wrote {key_out}  (PRIVATE — do not share)")
    if not args.commit:
        print("\nDRY RUN — nothing uploaded. Re-run with --commit to execute.")


if __name__ == "__main__":
    main()