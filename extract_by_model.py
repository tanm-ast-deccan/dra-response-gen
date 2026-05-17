"""
extract_by_model.py
-------------------
Extracts all agent result entries matching a specific model name from DRA
evaluation JSON files. Preserves the full task-level envelope (task_id,
package, sme_metadata, etc.) but filters agent_results down to only the
passes that match the requested model. Top-level derived fields
(agents_attempted, agents_succeeded, agents_failed, agent_errors,
total_cost_usd, total_duration_sec) are recalculated from the filtered set.

Matching modes  (--match)
-------------------------
  exact     Match the full model string literally            (default)
  contains  Model string contains the given value
  regex     Model string matches the given Python regex

Usage
-----
  # Single file
  python extract_by_model.py --model claude-opus-4-6 --input results.json

  # Single file, write to explicit output path
  python extract_by_model.py --model claude-opus-4-6 --input results.json \\
      --output ./out/results_claude.json

  # Batch: process every *.json in a folder
  python extract_by_model.py --model sonar-deep-research \\
      --input-dir ./results/json --output-dir ./extracted/perplexity

  # Substring match (useful for families like "claude" or "gemini")
  python extract_by_model.py --model claude --match contains \\
      --input-dir ./results/tasks --output-dir ./extracted/claude_all

  # Regex match
  python extract_by_model.py --model "o3.*research" --match regex \\
      --input-dir ./merged_output --output-dir ./extracted/openai_o3

Options
-------
  --model MODEL         Model name / pattern to filter by (required)
    options: claude-opus-4-6, deep-research-pro-preview-12-2025,
             o3-deep-research, sonar-deep-research  
  --match MODE          'exact' | 'contains' | 'regex'  (default: exact)
  --input FILE          Single input JSON file
  --input-dir DIR       Batch mode: folder of JSON files to process
  --output FILE         Output path for single-file mode
                        (default: <stem>_model_<model>.json next to input)
  --output-dir DIR      Output folder for batch mode
                        (default: ./extracted_<model> next to input-dir)
  --skip-empty          Skip output files where no matching passes were found
  --summary             Print a summary table at the end
  --dry-run             Show what would be written without creating files
"""

import argparse
import json
import re
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_null_or_dummy(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, (list, dict)) and len(value) == 0:
        return True
    return False


def safe_slug(s: str) -> str:
    """Turn an arbitrary string into a filename-safe slug."""
    return re.sub(r"[^a-zA-Z0-9_\-.]", "_", s)


# ── Model matching ────────────────────────────────────────────────────────────

def build_matcher(model: str, mode: str):
    """Return a function (model_value -> bool) based on the match mode."""
    if mode == "exact":
        return lambda v: (v or "") == model
    elif mode == "contains":
        lm = model.lower()
        return lambda v: lm in (v or "").lower()
    elif mode == "regex":
        pat = re.compile(model, re.IGNORECASE)
        return lambda v: bool(pat.search(v or ""))
    else:
        raise ValueError(f"Unknown match mode: {mode!r}")


# ── Core extraction ───────────────────────────────────────────────────────────

def extract_model_from_task(task: dict, matcher) -> dict | None:
    """
    Return a filtered copy of `task` containing only agent_results passes
    whose model field satisfies `matcher`.

    Returns None if no matching passes exist.
    """
    filtered_agent_results = {}

    for agent, passes in task.get("agent_results", {}).items():
        matching = [p for p in passes if matcher(p.get("model"))]
        if matching:
            filtered_agent_results[agent] = deepcopy(matching)

    if not filtered_agent_results:
        return None

    result = deepcopy(task)
    result["agent_results"] = filtered_agent_results

    # Recalculate derived fields from the filtered set
    attempted, succeeded, failed, errors = set(), set(), set(), {}
    total_cost = 0.0
    max_duration = 0.0

    for agent, passes in filtered_agent_results.items():
        attempted.add(agent)
        any_success = any(p.get("completed") for p in passes)
        all_errors = [p.get("error") for p in passes if p.get("error")]
        if any_success:
            succeeded.add(agent)
        else:
            failed.add(agent)
            if all_errors:
                errors[agent] = all_errors[-1]
        for p in passes:
            total_cost += p.get("total_cost_usd") or 0.0
            max_duration = max(max_duration, p.get("total_duration_sec") or 0.0)

    result["agents_attempted"] = sorted(attempted)
    result["agents_succeeded"] = sorted(succeeded)
    result["agents_failed"] = sorted(failed)
    result["agent_errors"] = errors
    result["total_cost_usd"] = round(total_cost, 8)
    result["total_duration_sec"] = max_duration

    # Recalculate timestamps from filtered passes
    all_started = []
    all_completed = []
    for passes in filtered_agent_results.values():
        for p in passes:
            try:
                if p.get("started_at"):
                    all_started.append(datetime.fromisoformat(p["started_at"]))
                if p.get("completed_at"):
                    all_completed.append(datetime.fromisoformat(p["completed_at"]))
            except ValueError:
                pass
    if all_started:
        result["dispatched_at"] = min(all_started).isoformat()
    if all_completed:
        result["completed_at"] = max(all_completed).isoformat()

    return result


# ── File-level operations ─────────────────────────────────────────────────────

def load_json(path: Path) -> list[dict] | dict | None:
    """Load a JSON file. Returns parsed content or None on error."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [ERROR] Could not read {path}: {e}", file=sys.stderr)
        return None


def normalise_to_list(data) -> list[dict]:
    """Accept either a single task dict or a list of task dicts."""
    if isinstance(data, dict):
        return [data]
    elif isinstance(data, list):
        return data
    else:
        return []


def process_file(
    input_path: Path,
    output_path: Path,
    matcher,
    skip_empty: bool,
    dry_run: bool,
) -> dict:
    """
    Process one input file.
    Returns a stats dict: {input, output, tasks_in, tasks_out, passes_out}.
    """
    stats = {
        "input": str(input_path),
        "output": str(output_path),
        "tasks_in": 0,
        "tasks_out": 0,
        "passes_out": 0,
        "skipped": False,
    }

    data = load_json(input_path)
    if data is None:
        stats["skipped"] = True
        return stats

    tasks = normalise_to_list(data)
    stats["tasks_in"] = len(tasks)

    extracted = []
    for task in tasks:
        filtered = extract_model_from_task(task, matcher)
        if filtered is not None:
            extracted.append(filtered)
            for passes in filtered["agent_results"].values():
                stats["passes_out"] += len(passes)

    stats["tasks_out"] = len(extracted)

    if not extracted and skip_empty:
        stats["skipped"] = True
        return stats

    if dry_run:
        return stats

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Mirror the input structure: single dict if one task, array if multiple
    payload = extracted[0] if len(extracted) == 1 and isinstance(data, dict) else extracted

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return stats


# ── Batch helpers ─────────────────────────────────────────────────────────────

def default_output_path(input_path: Path, model_slug: str) -> Path:
    """Derive a default single-file output path from the input path."""
    stem = input_path.stem
    return input_path.parent / f"{stem}_model_{model_slug}.json"


def default_output_dir(input_dir: Path, model_slug: str) -> Path:
    return input_dir.parent / f"extracted_{model_slug}"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract agent results for a specific model from DRA evaluation JSON files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Options")[0],
    )

    parser.add_argument(
        "--model",
        required=True,
        metavar="MODEL",
        help="Model name / pattern to extract (e.g. 'claude-opus-4-6')",
    )
    parser.add_argument(
        "--match",
        choices=["exact", "contains", "regex"],
        default="exact",
        help="How to match the model field: exact | contains | regex  (default: exact)",
    )

    # Input: single file OR directory
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", metavar="FILE", help="Single input JSON file")
    input_group.add_argument("--input-dir", metavar="DIR", help="Batch: folder of JSON files")

    # Output: single file OR directory (auto-derived if omitted)
    parser.add_argument(
        "--output",
        metavar="FILE",
        help="Output path for single-file mode (default: auto-derived next to input)",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        help="Output folder for batch mode (default: extracted_<model> next to input-dir)",
    )

    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="Skip output files where no matching passes were found",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a summary table after processing",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be written without creating any files",
    )

    args = parser.parse_args()

    # Validate: --output only makes sense with --input
    if args.output and args.input_dir:
        parser.error("--output can only be used with --input, not --input-dir")

    matcher = build_matcher(args.model, args.match)
    model_slug = safe_slug(args.model)

    print(f"\n-- Model Extractor --")
    print(f"  Model   : {args.model!r}  (match={args.match})")
    print(f"  Mode    : {'batch' if args.input_dir else 'single file'}")
    print(f"  Dry run : {args.dry_run}")
    print()

    # ── Collect input files ───────────────────────────────────────────────────
    if args.input:
        input_files = [Path(args.input)]
    else:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"[ERROR] Input dir not found: {input_dir}", file=sys.stderr)
            sys.exit(1)
        input_files = sorted(input_dir.glob("*.json"))
        if not input_files:
            print(f"[WARN] No *.json files found in {input_dir}", file=sys.stderr)
            sys.exit(0)

    # ── Resolve output paths ──────────────────────────────────────────────────
    if args.input:
        if args.output:
            out_paths = [Path(args.output)]
        else:
            out_paths = [default_output_path(Path(args.input), model_slug)]
    else:
        out_dir = Path(args.output_dir) if args.output_dir else default_output_dir(Path(args.input_dir), model_slug)
        out_paths = [out_dir / f.name for f in input_files]

    # ── Process ───────────────────────────────────────────────────────────────
    all_stats = []
    for in_path, out_path in zip(input_files, out_paths):
        stats = process_file(in_path, out_path, matcher, args.skip_empty, args.dry_run)
        all_stats.append(stats)

        status = "SKIP (empty)" if stats["skipped"] else ("DRY RUN" if args.dry_run else "OK")
        print(
            f"  [{status:11s}]  {in_path.name}"
            f"  tasks={stats['tasks_out']}/{stats['tasks_in']}"
            f"  passes={stats['passes_out']}"
            + (f"\n               -> {out_path}" if not stats["skipped"] else "")
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    if args.summary or args.input_dir:
        total_in = sum(s["tasks_in"] for s in all_stats)
        total_out = sum(s["tasks_out"] for s in all_stats)
        total_passes = sum(s["passes_out"] for s in all_stats)
        skipped = sum(1 for s in all_stats if s["skipped"])
        written = len(all_stats) - skipped

        print(f"\n-- Summary --")
        print(f"  Files processed  : {len(all_stats)}")
        print(f"  Files written    : {written}  (skipped={skipped})")
        print(f"  Tasks matched    : {total_out} / {total_in}")
        print(f"  Passes extracted : {total_passes}")
        if not args.dry_run and written > 0:
            out_loc = out_paths[0].parent if args.input_dir else out_paths[0]
            print(f"  Output location  : {out_loc}")

    print()


if __name__ == "__main__":
    main()