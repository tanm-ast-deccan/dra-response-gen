#!/usr/bin/env python3
"""
rename_drive_files.py — fix the anonymization leak on ALREADY-UPLOADED files.

The first commit uploaded files with model-name prefixes (hunyuan_*, doubao_*)
inside the anon Model_A/Model_B folders, leaking which model is which. This walks
the parent tree and renames any file whose name starts with a model prefix,
stripping it (hunyuan_GTM_1Pager.pdf -> GTM_1Pager.pdf). Read-only by default.

    python rename_drive_files.py --parent <ID> --shared-drive            # dry run
    python rename_drive_files.py --parent <ID> --shared-drive --commit   # rename

Then regenerate the CSV so its cells match (the file IDs don't change on rename,
so existing links keep working, but the displayed names in the cell need updating):
    python upload_sme_batch.py --commit --shared-drive --parent <ID>
"""
import argparse, os, sys

FOLDER_MIME = "application/vnd.google-apps.folder"
MODELS = ("hunyuan", "doubao")

def get_service():
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    key = os.environ.get("GOOGLE_SERVICE_ACCOUNT_KEY") or \
          os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not key or not os.path.exists(key):
        sys.exit("No service-account key. Set GOOGLE_SERVICE_ACCOUNT_KEY.")
    creds = service_account.Credentials.from_service_account_file(
        key, scopes=["https://www.googleapis.com/auth/drive"])
    return build("drive", "v3", credentials=creds)

def strip_prefix(name):
    """Remove ANY model name wherever it appears (leading prefix or embedded
    run tag), so already-uploaded files stop leaking model identity."""
    import re
    n = name
    for m in MODELS:
        n = re.sub(rf"__{re.escape(m)}__", "__", n, flags=re.IGNORECASE)
        n = re.sub(rf"_{re.escape(m)}_", "_", n, flags=re.IGNORECASE)
        n = re.sub(rf"^{re.escape(m)}__?", "", n, flags=re.IGNORECASE)
        n = re.sub(re.escape(m), "", n, flags=re.IGNORECASE)
    n = re.sub(r"_{3,}", "__", n).strip("_ ")
    n = re.sub(r"^_+", "", n)
    return n or name

def list_children(svc, parent, shared):
    kw = {"includeItemsFromAllDrives": True, "supportsAllDrives": True} if shared else {}
    out, token = [], None
    while True:
        r = svc.files().list(
            q=f"'{parent}' in parents and trashed = false",
            fields="nextPageToken, files(id,name,mimeType)",
            pageToken=token, pageSize=200, **kw).execute()
        out += r.get("files", [])
        token = r.get("nextPageToken")
        if not token:
            break
    return out

def walk_and_rename(svc, parent, shared, commit, depth=0):
    n_renamed = 0
    for f in list_children(svc, parent, shared):
        if f["mimeType"] == FOLDER_MIME:
            n_renamed += walk_and_rename(svc, f["id"], shared, commit, depth+1)
        else:
            new = strip_prefix(f["name"])
            if new != f["name"]:
                print(f"  {'RENAME' if commit else 'would rename'}: {f['name']}  ->  {new}")
                if commit:
                    kw = {"supportsAllDrives": True} if shared else {}
                    svc.files().update(fileId=f["id"], body={"name": new}, **kw).execute()
                n_renamed += 1
    return n_renamed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True)
    ap.add_argument("--shared-drive", action="store_true")
    ap.add_argument("--commit", action="store_true")
    args = ap.parse_args()
    svc = get_service()
    print(f"{'RENAMING' if args.commit else 'DRY RUN'} under parent={args.parent}")
    n = walk_and_rename(svc, args.parent, args.shared_drive, args.commit)
    print(f"\n{'renamed' if args.commit else 'would rename'} {n} file(s)")
    if not args.commit:
        print("Re-run with --commit to apply.")

if __name__ == "__main__":
    main()
    