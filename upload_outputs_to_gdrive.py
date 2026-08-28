#!/usr/bin/env python3
"""
upload_outputs_to_gdrive.py — push each task's OUTPUT/deliverable files into its
own Google Drive sub-folder, and write the sub-folder link back into a new column
of the tasks CSV (e.g. final_tasks.csv).

    python upload_outputs_to_gdrive.py \
        --csv final_tasks.csv \
        --results-json 'runs_dir/**/*_*tasks.json' \
        --parent-folder https://drive.google.com/drive/folders/<PARENT_ID> \
        --out final_tasks_with_links.csv

Source of truth is the RESULTS JSON, not the staging directory. The harness
writes deliverables and input files into the same per-task run folder, so
globbing the disk by extension would upload inputs too. Instead we read each
run's `output_files` list, which the runner builds by (a) harvesting only files
that are NOT inputs and (b) always adding {provider}_{task_id}_response.docx.
That list is exactly the deliverables + response, never inputs.

For every task_id in --csv this:
  1. looks up that task's run records in --results-json,
  2. collects the existing local paths named in their `output_files`,
  3. creates (or reuses) a Drive sub-folder named after the task_id inside the
     given parent folder,
  4. uploads those files into that sub-folder,
  5. fills the task's row in the new --link-column with the sub-folder's URL.

--only-success restricts uploads to runs that actually succeeded (completed, no
error, non-empty output_files) — the same pass/fail rule scan_failures.sh uses.

Auth: same service-account key the rest of the harness uses
(GOOGLE_SERVICE_ACCOUNT_KEY / GOOGLE_APPLICATION_CREDENTIALS), requested with the
read-WRITE Drive scope since uploading needs write access. The harness's own
GDriveClient is read-only-scoped, so this file builds its own writable client.

Idempotent: re-running reuses an existing task sub-folder (matched by name under
the parent) and, by default, skips files whose name already exists there. Use
--replace to delete-then-reupload matching names instead.
"""

import argparse
import csv
import glob
import json
import mimetypes
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reuse the harness's reference parser so folder links/ids are handled identically.
from src.file_resolver import parse_gdrive_reference

WRITE_SCOPES = ["https://www.googleapis.com/auth/drive"]
FOLDER_MIME = "application/vnd.google-apps.folder"


# ── writable Drive client ────────────────────────────────────────────────
class DriveWriter:
    def __init__(self):
        self._svc = None

    @staticmethod
    def _exec(request, what="drive call", max_tries=6):
        """Execute a Drive API request with exponential backoff on transient
        rate-limit / server errors. Re-raises anything non-retryable, and
        re-raises the last error if all retries are exhausted."""
        import random
        import time
        from googleapiclient.errors import HttpError
        RETRY_REASONS = {"userratelimitexceeded", "ratelimitexceeded",
                         "backenderror", "internalerror"}
        for attempt in range(max_tries):
            try:
                return request.execute()
            except HttpError as e:
                status = getattr(getattr(e, "resp", None), "status", None)
                body = (getattr(e, "content", b"") or b"").decode("utf-8", "replace").lower()
                reason_hit = any(r in body for r in RETRY_REASONS)
                transient = status in (403, 429, 500, 503) and (reason_hit or status in (429, 500, 503))
                if not transient or attempt == max_tries - 1:
                    raise
                # exp backoff: 2,4,8,16,32s (+jitter), capped
                wait = min(2 ** (attempt + 1), 32) + random.uniform(0, 1.5)
                print(f"   … {what}: {status} rate-limited, retry "
                      f"{attempt+1}/{max_tries-1} in {wait:.1f}s", file=sys.stderr)
                time.sleep(wait)
        return None  # unreachable

    def _service(self):
        if self._svc is not None:
            return self._svc
        try:
            from googleapiclient.discovery import build
            from google.oauth2 import service_account
        except ImportError:
            raise ImportError(
                "pip install google-api-python-client google-auth google-auth-oauthlib"
            )
        key_path = os.environ.get(
            "GOOGLE_SERVICE_ACCOUNT_KEY",
            os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", ""),
        )
        if not (key_path and os.path.exists(key_path)):
            raise RuntimeError(
                "No service-account key found. Set GOOGLE_SERVICE_ACCOUNT_KEY "
                "or GOOGLE_APPLICATION_CREDENTIALS to the JSON key path."
            )
        creds = service_account.Credentials.from_service_account_file(
            key_path, scopes=WRITE_SCOPES
        )
        self._svc = build("drive", "v3", credentials=creds)
        return self._svc

    def find_child_folder(self, parent_id: str, name: str):
        q = (f"'{parent_id}' in parents and name = {_q(name)} "
             f"and mimeType = '{FOLDER_MIME}' and trashed = false")
        resp = self._exec(self._service().files().list(
            q=q, fields="files(id,name)", pageSize=10,
            supportsAllDrives=True, includeItemsFromAllDrives=True,
        ), "find_folder")
        files = resp.get("files", [])
        return files[0]["id"] if files else None

    def create_folder(self, parent_id: str, name: str) -> str:
        meta = {"name": name, "mimeType": FOLDER_MIME, "parents": [parent_id]}
        f = self._exec(self._service().files().create(
            body=meta, fields="id", supportsAllDrives=True
        ), "create_folder")
        return f["id"]

    def get_or_create_folder(self, parent_id: str, name: str) -> str:
        return self.find_child_folder(parent_id, name) or \
            self.create_folder(parent_id, name)

    def list_folder(self, folder_id: str) -> list[dict]:
        out, token = [], None
        svc = self._service()
        while True:
            resp = self._exec(svc.files().list(
                q=f"'{folder_id}' in parents and trashed = false",
                fields="nextPageToken,files(id,name,size)",
                pageToken=token, pageSize=100,
                supportsAllDrives=True, includeItemsFromAllDrives=True,
            ), "list_folder")
            out.extend(resp.get("files", []))
            token = resp.get("nextPageToken")
            if not token:
                break
        return out

    def delete(self, file_id: str):
        self._exec(self._service().files().delete(
            fileId=file_id, supportsAllDrives=True), "delete")

    def upload(self, folder_id: str, local_path: str) -> str:
        from googleapiclient.http import MediaFileUpload
        name = os.path.basename(local_path)
        mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
        media = MediaFileUpload(local_path, mimetype=mime, resumable=True)
        f = self._exec(self._service().files().create(
            body={"name": name, "parents": [folder_id]},
            media_body=media, fields="id", supportsAllDrives=True,
        ), "upload")
        return f["id"]


def _q(s: str) -> str:
    """Quote a string for a Drive query 'name = ...' clause."""
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"


# ── output-file discovery: read the runner's own output_files list ───────
# The results JSON is the source of truth. The runner's _harvest_output_files
# already EXCLUDES input files (skips basenames matching the task's inputs) and
# _postprocess_outputs adds the {provider}_{task_id}_response.docx, so the
# output_files list is exactly the deliverables + response — never inputs.
# We do NOT glob the staging dir, precisely because those folders also hold
# input files and transient helper scripts.

def load_results(paths_or_globs: list[str]) -> dict[str, list[dict]]:
    """Return {task_id: [result_record, ...]} across all given results JSONs.

    Later files override earlier ones per (task_id, run_id) so re-runs win.
    """
    by_task: dict[str, list[dict]] = {}
    seen: dict[tuple, int] = {}
    files: list[str] = []
    for pg in paths_or_globs:
        hits = sorted(glob.glob(pg))
        files.extend(hits if hits else [pg])
    for f in files:
        if not os.path.isfile(f):
            print(f"!! results file not found: {f}", file=sys.stderr)
            continue
        try:
            data = json.load(open(f, encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"!! could not parse {f}: {e}", file=sys.stderr)
            continue
        for r in (data.get("results") or []):
            tid = r.get("task_id")
            if not tid:
                continue
            key = (tid, r.get("run_id") or r.get("pass_index"))
            recs = by_task.setdefault(tid, [])
            if key in seen:
                recs[seen[key]] = r        # override earlier run of same key
            else:
                seen[key] = len(recs)
                recs.append(r)
    return by_task


def task_output_files(records: list[dict], require_success: bool) -> list[str]:
    """Collect existing local output-file paths for a task's run records.

    `require_success` gates on the same definition scan_failures.sh uses:
    completed, no error, and a non-empty output_files list.
    """
    out: list[str] = []
    for r in records:
        if require_success:
            ok = r.get("completed") and not (r.get("error") or "") \
                and (r.get("output_files") or [])
            if not ok:
                continue
        for entry in (r.get("output_files") or []):
            # entries are path strings, or {"name"/"path": ...} objects
            p = entry if isinstance(entry, str) \
                else (entry.get("path") or entry.get("name") or "")
            if not p:
                continue
            if not os.path.isabs(p):
                p = os.path.abspath(p)
            if os.path.exists(p):
                out.append(p)
            else:
                print(f"   (listed but missing on disk: {p})", file=sys.stderr)
    # de-dupe, keep order
    seen_p, uniq = set(), []
    for p in out:
        if p not in seen_p:
            seen_p.add(p)
            uniq.append(p)
    return uniq


def folder_url(folder_id: str) -> str:
    return f"https://drive.google.com/drive/folders/{folder_id}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="tasks CSV (e.g. final_tasks.csv)")
    ap.add_argument("--results-json", required=True, nargs="+",
                    help="one or more results JSON files (globs ok). The runner's "
                         "output_files list is the source of truth — inputs are "
                         "already excluded, response.docx already included.")
    ap.add_argument("--parent-folder", required=True,
                    help="Drive parent folder URL or ID; task sub-folders go here")
    ap.add_argument("--out", help="output CSV path (default: <csv>_with_links.csv)")
    ap.add_argument("--link-column", default="Output Files Drive Link",
                    help="name of the new column to add/fill")
    ap.add_argument("--only-success", action="store_true",
                    help="upload only for runs that succeeded (completed, no error, "
                         "non-empty output_files); skip failed/partial runs")
    ap.add_argument("--replace", action="store_true",
                    help="overwrite same-named files in the task folder "
                         "(default: skip files already present)")
    ap.add_argument("--dry-run", action="store_true",
                    help="discover + plan only; no Drive writes, no CSV changes")
    ap.add_argument("--sleep", type=float, default=0.3,
                    help="seconds to pause between file uploads, to stay under "
                         "Google's per-user rate limit (default 0.3; raise to "
                         "0.5-1.0 if you still hit userRateLimitExceeded)")
    args = ap.parse_args(argv)

    ref = parse_gdrive_reference(args.parent_folder)
    if not ref:
        # parse_gdrive_reference only accepts full URLs; allow a bare folder id too.
        pid = args.parent_folder.strip()
        if pid and "/" not in pid and " " not in pid:
            ref = {"type": "folder", "id": pid}
    if not ref or ref["type"] != "folder":
        raise SystemExit(f"--parent-folder is not a Drive folder: {args.parent_folder}")
    parent_id = ref["id"]

    with open(args.csv, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if "task_id" not in fieldnames:
        raise SystemExit("CSV has no 'task_id' column")
    if args.link_column not in fieldnames:
        fieldnames.append(args.link_column)

    # Resume-safe: if the output CSV already exists (a prior/interrupted run),
    # preload its links so tasks we skip this run keep their links instead of
    # being blanked. Links are keyed by task_id.
    out_path = args.out or _default_out(args.csv)
    prior_links = {}
    if os.path.exists(out_path):
        try:
            with open(out_path, encoding="utf-8-sig", newline="") as f:
                for r in csv.DictReader(f):
                    lk = (r.get(args.link_column) or "").strip()
                    if r.get("task_id") and lk:
                        prior_links[r["task_id"]] = lk
            if prior_links:
                print(f"resuming: {len(prior_links)} existing link(s) preloaded "
                      f"from {out_path}")
        except Exception as e:  # noqa: BLE001
            print(f"(could not read existing {out_path}: {e})", file=sys.stderr)
    for row in rows:
        tid = (row.get("task_id") or "").strip()
        if tid in prior_links and not (row.get(args.link_column) or "").strip():
            row[args.link_column] = prior_links[tid]

    results = load_results(args.results_json)
    if not results:
        raise SystemExit("no usable results records found in --results-json")

    drive = None if args.dry_run else DriveWriter()

    def _flush_csv():
        if args.dry_run:
            return
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, out_path)  # atomic; a crash never leaves a half-file

    n_up = n_skip = n_task = 0
    try:
        for row in rows:
            tid = (row.get("task_id") or "").strip()
            if not tid:
                continue
            records = results.get(tid)
            if not records:
                # keep any preloaded link; only note if there's genuinely none
                if not (row.get(args.link_column) or "").strip():
                    print(f"[{tid}] not in results JSON — no link")
                continue
            files = task_output_files(records, require_success=args.only_success)
            if not files:
                if not (row.get(args.link_column) or "").strip():
                    why = "no successful run" if args.only_success else "no output files"
                    print(f"[{tid}] {why} in results JSON — no link")
                continue
            n_task += 1
            print(f"[{tid}] {len(files)} output file(s): "
                  + ", ".join(os.path.basename(p) for p in files))
            if args.dry_run:
                row[args.link_column] = "(dry-run)"
                continue

            folder_id = drive.get_or_create_folder(parent_id, tid)
            existing = {e["name"]: e for e in drive.list_folder(folder_id)}
            for p in files:
                name = os.path.basename(p)
                if name in existing:
                    if args.replace:
                        drive.delete(existing[name]["id"])
                    else:
                        n_skip += 1
                        continue
                drive.upload(folder_id, p)
                n_up += 1
                if args.sleep:
                    import time
                    time.sleep(args.sleep)
            # always set the link for a processed task (even if all files were
            # skip-because-present — the folder exists and is the deliverable set)
            row[args.link_column] = folder_url(folder_id)
            print(f"       → {folder_url(folder_id)}")
            _flush_csv()  # persist after each task so a crash keeps progress
    finally:
        _flush_csv()
        if not args.dry_run:
            print(f"\nWrote {out_path}")

    print(f"tasks with files : {n_task}")
    print(f"uploaded         : {n_up}")
    print(f"skipped (exists) : {n_skip}")


def _default_out(csv_path: str) -> str:
    base, ext = os.path.splitext(csv_path)
    return f"{base}_with_links{ext or '.csv'}"


if __name__ == "__main__":
    main()