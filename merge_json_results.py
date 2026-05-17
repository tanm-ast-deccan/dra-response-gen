"""
merge_json_results.py
─────────────────────
Merges JSON result files that share the same name prefix (e.g. krrish_chhablan_*)
across multiple source folders.

File naming convention:
    <prefix>_<random_suffix>.json
    e.g. krrish_chhablan_405991.json, krrish_chhablan_70b07c.json

Merge rules
-----------
1. Files with the SAME task_id found in multiple folders are deep-merged:
   - agent_results: union by agent key; prefer completed=True over failed
   - agents_attempted / succeeded / failed: recalculated from merged results
   - agent_errors: merged dict, cleared for agents that now have a success
   - total_cost_usd: summed across sources
   - total_duration_sec: max across sources
   - dispatched_at / completed_at: min / max across sources
   - extra keys (e.g. sme_metadata): preserved from whichever file has them
   - package / config: taken from the most complete source

2. Files with DIFFERENT task_ids but the same prefix are collected into a
   single output file as a JSON array.

3. NULL / dummy / empty values:
   - Agent result entries with response_length == 0 AND completed == False
     are treated as "empty shells" -- kept only if no better result exists
   - Agent-level error strings are preserved for diagnostics
   - Top-level null fields are overwritten by non-null values from other sources

Usage
-----
    python merge_json_results.py [OPTIONS]

Options:
    --source-dirs   Space-separated list of folders to scan (default: see SOURCE_DIRS)
    --output-dir    Where to write merged files (default: ./merged_output)
    --dry-run       Print what would be merged without writing files

Example:
    python merge_json_results.py --source-dirs ./results/json ./results_partial_run_6_30/json ./results_openai/json --output-dir ./merged_output/json
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any


# ── Default configuration ────────────────────────────────────────────────────

SOURCE_DIRS = [
    "./results/json",
    "./results_partial_run_6_30/json",
    "./results_openai/json",
]
OUTPUT_DIR = "./merged_output"

# Regex to detect the random suffix: underscore + 6 hex/alphanumeric chars at end
# Adjust the character class or length if your suffixes differ
SUFFIX_PATTERN = re.compile(r"^(.+)_([a-f0-9]{6})$")


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_prefix(task_id: str) -> str:
    """
    Split  krrish_chhablan_405991  ->  'krrish_chhablan'
    Falls back to the full task_id as prefix if the pattern does not match.
    """
    m = SUFFIX_PATTERN.match(task_id)
    return m.group(1) if m else task_id


def is_null_or_dummy(value: Any) -> bool:
    """Return True for None, empty string, 0, [], {}."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, (list, dict)) and len(value) == 0:
        return True
    return False


def coalesce(*values):
    """Return the first non-null/non-dummy value."""
    for v in values:
        if not is_null_or_dummy(v):
            return v
    return values[-1]  # fallback to last even if dummy


def parse_iso(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def fmt_iso(dt):
    return dt.isoformat() if dt else None


# ── Agent result merging ──────────────────────────────────────────────────────

def merge_agent_pass_list(existing, incoming):
    """
    Merge two lists of pass-level results for the SAME agent.

    Strategy per pass index:
      - If the incoming pass is better (completed=True, existing is not), replace it.
      - If both are complete, keep the one with the longer response (more content).
      - If incoming is also empty, keep existing but update error info if missing.
    """
    merged = list(existing)
    for i, inc in enumerate(incoming):
        if i < len(merged):
            ex = merged[i]
            # Prefer completed over not-completed
            if inc.get("completed") and not ex.get("completed"):
                merged[i] = deepcopy(inc)
            # Both completed: prefer longer response
            elif inc.get("completed") and ex.get("completed"):
                if inc.get("response_length", 0) > ex.get("response_length", 0):
                    merged[i] = deepcopy(inc)
            # Both failed: keep existing but take error message if missing
            elif not inc.get("completed") and not ex.get("completed"):
                if inc.get("error") and not ex.get("error"):
                    merged[i] = deepcopy(inc)
        else:
            # Extra pass in incoming that does not exist in existing
            merged.append(deepcopy(inc))
    return merged


def merge_agent_results(base, incoming):
    """
    Merge two agent_results dicts.
    Keys are agent names; values are lists of pass dicts.
    """
    merged = deepcopy(base)
    for agent, passes in incoming.items():
        if agent not in merged:
            merged[agent] = deepcopy(passes)
        else:
            merged[agent] = merge_agent_pass_list(merged[agent], passes)
    return merged


# ── Top-level task merging ────────────────────────────────────────────────────

def merge_two_tasks(base, incoming):
    """
    Merge `incoming` into `base` (both represent the same task_id).
    Returns a new merged dict.
    """
    result = deepcopy(base)

    # agent_results (core merge)
    result["agent_results"] = merge_agent_results(
        base.get("agent_results", {}),
        incoming.get("agent_results", {}),
    )

    # Recalculate derived agent lists from merged results
    attempted, succeeded, failed, errors = set(), set(), set(), {}

    for agent, passes in result["agent_results"].items():
        attempted.add(agent)
        any_success = any(p.get("completed") for p in passes)
        all_errors = [p.get("error") for p in passes if p.get("error")]
        if any_success:
            succeeded.add(agent)
        else:
            failed.add(agent)
            if all_errors:
                errors[agent] = all_errors[-1]  # keep most recent error

    result["agents_attempted"] = sorted(attempted)
    result["agents_succeeded"] = sorted(succeeded)
    result["agents_failed"] = sorted(failed)
    result["agent_errors"] = errors

    # Numeric aggregates
    result["total_cost_usd"] = (
        (base.get("total_cost_usd") or 0.0) +
        (incoming.get("total_cost_usd") or 0.0)
    )
    result["total_duration_sec"] = max(
        base.get("total_duration_sec") or 0.0,
        incoming.get("total_duration_sec") or 0.0,
    )

    # Timestamps
    dispatched = [parse_iso(base.get("dispatched_at")), parse_iso(incoming.get("dispatched_at"))]
    completed = [parse_iso(base.get("completed_at")), parse_iso(incoming.get("completed_at"))]
    dispatched = [d for d in dispatched if d]
    completed = [c for c in completed if c]
    if dispatched:
        result["dispatched_at"] = fmt_iso(min(dispatched))
    if completed:
        result["completed_at"] = fmt_iso(max(completed))

    # config / package: union, prefer non-null values from base
    for field in ("config", "package"):
        base_val = base.get(field) or {}
        inc_val = incoming.get(field) or {}
        if isinstance(base_val, dict) and isinstance(inc_val, dict):
            merged_field = deepcopy(inc_val)
            merged_field.update({k: v for k, v in base_val.items() if not is_null_or_dummy(v)})
            result[field] = merged_field
        else:
            result[field] = coalesce(base_val, inc_val)

    # Extra / unknown keys: carry over from incoming if not in base
    for key, value in incoming.items():
        if key not in result or is_null_or_dummy(result.get(key)):
            result[key] = deepcopy(value)

    return result


def merge_task_list(tasks):
    """Reduce a list of same-task_id records into one merged record."""
    result = tasks[0]
    for t in tasks[1:]:
        result = merge_two_tasks(result, t)
    return result


# ── File discovery & grouping ─────────────────────────────────────────────────

def discover_files(source_dirs):
    """
    Scan all source dirs for *.json files.
    Returns: { prefix -> [(source_path, parsed_json), ...] }
    """
    groups = defaultdict(list)
    found = 0
    errors = 0

    for folder in source_dirs:
        p = Path(folder)
        if not p.exists():
            print(f"  [WARN] Source dir not found, skipping: {folder}", file=sys.stderr)
            continue
        for fpath in sorted(p.glob("*.json")):
            try:
                with open(fpath, encoding="utf-8") as f:
                    data = json.load(f)
                task_id = data.get("task_id", fpath.stem)
                prefix = extract_prefix(task_id)
                groups[prefix].append((str(fpath), data))
                found += 1
            except (json.JSONDecodeError, OSError) as e:
                print(f"  [ERROR] Could not read {fpath}: {e}", file=sys.stderr)
                errors += 1

    print(
        f"  Discovered {found} files across {len(source_dirs)} dirs "
        f"({errors} unreadable) -> {len(groups)} unique prefixes"
    )
    return dict(groups)


# ── Output ────────────────────────────────────────────────────────────────────

def build_merged_output(prefix, entries):
    """
    For a given prefix, group entries by task_id and merge same-id groups.
    Returns a list of merged task records (one per unique task_id).
    """
    by_task_id = defaultdict(list)
    for _path, data in entries:
        tid = data.get("task_id", "unknown")
        by_task_id[tid].append(data)

    merged_tasks = []
    for tid, task_list in sorted(by_task_id.items()):
        merged = merge_task_list(task_list)
        merged_tasks.append(merged)

    return merged_tasks


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Merge JSON result files by name prefix across multiple folders."
    )
    parser.add_argument(
        "--source-dirs",
        nargs="+",
        default=SOURCE_DIRS,
        metavar="DIR",
        help="Folders to scan for *.json files",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        metavar="DIR",
        help="Where to write merged output files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print merge plan without writing any files",
    )
    args = parser.parse_args()

    print("\n-- JSON Result Merger --")
    print(f"  Source dirs : {args.source_dirs}")
    print(f"  Output dir  : {args.output_dir}")
    print(f"  Dry run     : {args.dry_run}\n")

    groups = discover_files(args.source_dirs)

    if not groups:
        print("No JSON files found. Check your --source-dirs.")
        return

    if not args.dry_run:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)

    total_written = 0
    for prefix, entries in sorted(groups.items()):
        by_tid = defaultdict(list)
        for path, data in entries:
            by_tid[data.get("task_id", "?")].append(path)

        print(f"\nPrefix: {prefix!r}")
        for tid, paths in sorted(by_tid.items()):
            print(f"  task_id={tid!r}  sources={len(paths)}")
            for p in paths:
                print(f"    <- {p}")

        if args.dry_run:
            continue

        merged_tasks = build_merged_output(prefix, entries)

        out_path = Path(args.output_dir) / f"{prefix}_merged.json"
        # Single task -> plain object; multiple tasks -> array
        payload = merged_tasks if len(merged_tasks) > 1 else merged_tasks[0]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        total_written += 1
        print(f"  -> Written: {out_path}  ({len(merged_tasks)} task(s))")

    if not args.dry_run:
        print(f"\nDone. {total_written} merged file(s) written to {args.output_dir!r}\n")
    else:
        print("\n[Dry run complete -- no files written]\n")


if __name__ == "__main__":
    main()
