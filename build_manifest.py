#!/usr/bin/env python3
"""
build_manifest.py — build a generalized SME-batch manifest from a run.

Generalizes the old ib_select_and_stage.py. It joins three sources:

  1. --staging     (default ./staging)   per-task model outputs on disk:
                                          <staging>/<task_id>/runs/<model>__p1/<files>
  2. --augmented   (default ./output/augmented)  one augment JSON per task:
                                          <augmented>/<task_id>.json
                                          (a matching <task_id>.html may also exist)
  3. --results     (default ./results)    ONE combined run-trace JSON covering
                                          all tasks/models in the batch; the source
                                          of truth for which output files are the
                                          gradeable "deliverables" per task/model.

Output:
  --out       manifest.json    per-task entry: augment fields + a `models` dict
                               keyed by ANONYMIZED label (Model_A, Model_B, ...).
                               Real model names are NOT written here.
  --key-out   model_key.csv    the private anon->real-model mapping (keep OUT of
                               anything the SME sees).

Anonymization: model names are collected GLOBALLY across the whole batch, sorted
alphabetically, and assigned Model_A, Model_B, Model_C, ... in that order. This
is deterministic and reproducible, and — because assignment is global, not
per-task — a task missing one model does not shift the lettering for other tasks.

Supports any number of models per task. A task missing a model simply has fewer
entries in its `models` dict.

------------------------------------------------------------------------------
NOTE ON THE TWO ADAPTERS BELOW (extract_from_trace, extract_from_augment):
These two functions depend on the internal JSON shapes of the trace and augment
files. They have been VERIFIED against a live IB trace and real augment JSONs:
  - trace is Layout C: {"results": [ {task_id, provider, model, output_files}, ...]}
  - runs are keyed by PROVIDER ('doubao','hunyuan'), which names the staging
    folders and prefixes the files — NOT the full model string.
  - the trace's embedded output_files paths are relative to the OLD ./staging
    root and are STALE after the move to ./runs_dir/... ; only their basenames
    are used, and --staging (passed by the caller) is authoritative for the dir.
  - per-task augment files are <task_id>_augment.json (not <task_id>.json).
Re-verify if the trace/augment schema changes.
------------------------------------------------------------------------------
"""

import argparse
import csv
import glob
import json
import os
import string
import sys


P1_SUFFIX = "__p1"
DEFAULT_STAGING = "./staging"
DEFAULT_AUGMENTED = "./output/augmented"
DEFAULT_RESULTS = "./results"


# ============================================================================
# ADAPTERS — verify these two against real samples (see NOTE above).
# ============================================================================

def extract_from_trace(trace: dict) -> dict:
    """From the combined run-trace JSON, return:
        { task_id: { provider: [output_file_basename, ...] } }

    Keyed by PROVIDER (e.g. 'doubao', 'hunyuan') — the identity that names the
    staging run folders (<provider>__p1) and prefixes the output files, and the
    identity the rest of the SME cluster (sme_pool, upload_sme_batch) keys on.

    ALL output files are kept — nothing is filtered as scratch. Only basenames
    are returned; the on-disk path is resolved later from --staging, because the
    trace's embedded paths are relative to the OLD ./staging root and are stale
    after the move to ./runs_dir/... (see resolve_run_folder).

    ---- REAL SHAPE (verified against a live IB trace) ----
    Layout C: {"results": [ {task_id, provider, model, output_files:[...paths]}, ... ]}
    Each record is one (task, provider, pass) run. We also accept the older
    task-keyed layouts (A/B) as a fallback so historical traces still parse.
    """
    out = {}

    def basenames(rec):
        for key in ("output_files", "real_files", "deliverables", "files", "all_files"):
            if isinstance(rec, dict) and isinstance(rec.get(key), list):
                return [os.path.basename(str(x)) for x in rec[key]]
        return None

    # Layout C (the real one): flat list of per-run records under "results".
    if isinstance(trace, dict) and isinstance(trace.get("results"), list):
        for row in trace["results"]:
            t = row.get("task_id") or row.get("task")
            # PROVIDER first (not model): it names the folders and prefixes files.
            m = row.get("provider") or row.get("model") or row.get("model_name")
            bl = basenames(row)
            if t and m and bl is not None:
                # merge across passes; keep every file, de-dup by basename
                seen = out.setdefault(t, {}).setdefault(m, [])
                for b in bl:
                    if b not in seen:
                        seen.append(b)
        if out:
            return out

    # Fallback Layouts A/B: task-keyed dicts (older traces).
    container = trace.get("tasks", trace) if isinstance(trace, dict) else trace
    if isinstance(container, dict):
        for t, tval in container.items():
            if not isinstance(tval, dict):
                continue
            models = tval.get("models") or tval.get("runs") or tval
            if not isinstance(models, dict):
                continue
            for m, rec in models.items():
                bl = basenames(rec)
                if bl is not None:
                    out.setdefault(t, {})[m] = bl

    if not out:
        raise ValueError(
            "extract_from_trace: could not find any (task -> provider -> file-list) "
            "structure in the trace JSON. Inspect the trace and adjust this "
            "adapter to its real shape."
        )
    return out


def extract_from_augment(aug: dict) -> dict:
    """From a per-task augment JSON, return the SME-facing package fields.
    Missing fields default to "" so the manifest is uniform.

    ---- ASSUMED FIELD NAMES (verify!) ----
    """
    def g(*names, default=""):
        for n in names:
            if n in aug and aug[n] not in (None, ""):
                return aug[n]
        return default

    return {
        "domain": g("domain", "Domain"),
        "prompt": g("prompt", "Prompt"),
        "sanity_check": g("sanity_check", "Sanity Check", "corrected_sanity_check"),
        "solution_logic": g("corrected_solution_logic", "solution_logic", "Solution Logic"),
        "golden_deliverable": g("golden_deliverable", "golden"),
        "golden_solution_logic": g("golden_solution_logic"),
        "original_verifiers": g("original_verifiers", "verifiers", "Verifiers"),
        "augmented_verifiers": g("augmented_verifiers"),
        "crux": g("crux_ids", "crux", "crux_verifier_ids", default=[]),
        "crux_size": (len(aug["crux_ids"]) if isinstance(aug.get("crux_ids"), list)
                      else aug.get("crux_size",
                                   len(aug.get("crux", [])) if isinstance(aug.get("crux"), list) else 0)),
        "verdict": g("audit_verdict", "verdict"),
        "scoreable": aug.get("scoreable", True),
        "input_files_link": g("input_files_link", "input_files", "drive_link", "Drive Link"),
    }


# ============================================================================
# GENERALIZED CORE — shape-independent.
# ============================================================================

def anon_labels(n: int):
    """Model_A, Model_B, ... Model_Z, Model_AA, ... for n models."""
    labels = []
    for i in range(n):
        # base-26 lettering: A..Z, AA..AZ, ...
        s, x = "", i
        while True:
            s = string.ascii_uppercase[x % 26] + s
            x = x // 26 - 1
            if x < 0:
                break
        labels.append(f"Model_{s}")
    return labels


def resolve_run_folder(staging: str, task_id: str, provider: str) -> str:
    """The convention: <staging>/<task_id>/runs/<provider>__p1/.

    --staging is AUTHORITATIVE for the directory. The trace's embedded paths are
    NOT used here — they point at the pre-move ./staging root and are stale after
    the move to ./runs_dir/... So we always rebuild the folder from the --staging
    the caller passes now. Falls back to a glob if the exact __p1 folder isn't
    present (some runs use a different partition suffix)."""
    exact = os.path.join(staging, task_id, "runs", f"{provider}{P1_SUFFIX}")
    if os.path.isdir(exact):
        return exact
    cands = sorted(glob.glob(os.path.join(staging, task_id, "runs", f"{provider}*")))
    return cands[0] if cands else exact  # return exact (nonexistent) so caller can flag


def resolve_real_files(run_folder: str, file_basenames: list) -> list:
    """Map each output-file basename to its on-disk path under run_folder.
    Uses the basename only (the trace's original directory is stale post-move).
    Missing files keep their intended path so the caller can flag them."""
    found = []
    for bn in file_basenames:
        bn = os.path.basename(bn)  # defensive: ensure basename even if a path slipped through
        p = os.path.join(run_folder, bn)
        if os.path.isfile(p):
            found.append(p)
        else:
            hits = glob.glob(os.path.join(run_folder, "**", bn), recursive=True)
            found.append(hits[0] if hits else p)  # keep intended path if truly missing
    return found


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build(staging, augmented, results_path, verbose=True):
    trace = _load_json(results_path)
    task_provider_files = extract_from_trace(trace)

    # GLOBAL anonymization: collect every PROVIDER across the batch, sort, map.
    all_providers = sorted({m for mm in task_provider_files.values() for m in mm})
    labels = anon_labels(len(all_providers))
    anon = dict(zip(all_providers, labels))  # real provider -> Model_X
    if verbose:
        print("Global provider anonymization (alphabetical):")
        for m, a in anon.items():
            print(f"  {a} = {m}")

    manifest = {}
    warnings = []

    for task_id, providers in sorted(task_provider_files.items()):
        # per-task augment file is <task_id>_augment.json (matches sme_pool and
        # the real files on disk), NOT <task_id>.json.
        aug_path = os.path.join(augmented, f"{task_id}_augment.json")
        if os.path.isfile(aug_path):
            try:
                aug_fields = extract_from_augment(_load_json(aug_path))
            except Exception as e:
                aug_fields = extract_from_augment({})
                warnings.append(f"{task_id}: augment JSON unreadable ({e})")
        else:
            aug_fields = extract_from_augment({})
            warnings.append(f"{task_id}: no augment JSON at {aug_path}")

        entry = dict(aug_fields)
        entry["task_id"] = task_id
        entry["models"] = {}

        for provider, out_basenames in sorted(providers.items()):
            label = anon[provider]
            run_folder = resolve_run_folder(staging, task_id, provider)
            if not os.path.isdir(run_folder):
                warnings.append(f"{task_id}/{label}: run folder not found ({run_folder})")
            output_files = resolve_real_files(run_folder, out_basenames)
            if not out_basenames:
                warnings.append(f"{task_id}/{label}: 0 output files (model produced nothing)")
            missing = [f for f in output_files if not os.path.isfile(f)]
            if missing:
                warnings.append(f"{task_id}/{label}: {len(missing)} output file(s) not on disk")
            # ALL output files are kept (no scratch filter) — _response.docx and
            # every other produced file flow through; downstream decides use.
            entry["models"][label] = {
                "run_folder": run_folder,
                "output_files": output_files,
            }

        manifest[task_id] = entry

    return manifest, anon, warnings


def main():
    ap = argparse.ArgumentParser(description="Build a generalized SME-batch manifest")
    ap.add_argument("--staging", default=DEFAULT_STAGING,
                    help="per-task outputs: <staging>/<task_id>/runs/<model>__p1/ (default ./staging)")
    ap.add_argument("--augmented", default=DEFAULT_AUGMENTED,
                    help="per-task augment JSON: <augmented>/<task_id>.json (default ./output/augmented)")
    ap.add_argument("--results", default=DEFAULT_RESULTS,
                    help="combined run-trace JSON, or a directory containing exactly one (default ./results)")
    ap.add_argument("--out", default="manifest.json")
    ap.add_argument("--key-out", default="model_key.csv")
    args = ap.parse_args()

    # allow --results to be a dir containing one trace JSON
    results_path = args.results
    if os.path.isdir(results_path):
        jsons = sorted(glob.glob(os.path.join(results_path, "*.json")))
        if len(jsons) != 1:
            print(f"ERROR: {results_path} has {len(jsons)} JSON files; "
                  f"point --results at the single trace file.")
            sys.exit(2)
        results_path = jsons[0]

    manifest, anon, warnings = build(args.staging, args.augmented, results_path)

    # The manifest is an INTERNAL build artifact: it stores real run-folder paths
    # (which embed model names) so the uploader can locate files. It is NOT blind
    # and must never be handed to an SME. The blind boundary is the anonymized
    # Drive folders + the grading CSV built downstream.
    wrapped = {
        "_INTERNAL_not_blind": True,
        "_note": "Contains real model-named paths for the uploader. Do NOT give to SMEs.",
        "tasks": manifest,
    }
    json.dump(wrapped, open(args.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    with open(args.key_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["anon", "model"])
        for real_model, label in sorted(anon.items(), key=lambda kv: kv[1]):
            w.writerow([label, real_model])

    print(f"\n{len(manifest)} task(s) -> {args.out}")
    print(f"model key -> {args.key_out}  (KEEP PRIVATE — de-anonymization mapping)")
    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for w in warnings:
            print(f"  ! {w}")


if __name__ == "__main__":
    main()