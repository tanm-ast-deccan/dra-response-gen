"""
export_completed_runs.py
------------------------
Reads DRA evaluation JSON files, filters to qualifying agent runs, and writes
each run as a human-readable .txt file.

Filtering rules
---------------
A run is included only when ALL three conditions hold:
  1. config.dry_run  == false          (task level)
  2. pass.completed  == true           (pass level)
  3. pass.forced_stop == false         (pass level)

Output files
------------
One .txt file per qualifying pass, named:
    <task_id>_<YYYYMMDD_HHMMSS>.txt
where the timestamp comes from the pass's own completed_at field.

If passes_per_agent > 1, or the same task appears across multiple source
files, each pass still gets its own file — the timestamp suffix keeps them
distinct.

Each file contains (in order):
  - Task Overview        top-level metadata
  - Task Configuration   config block
  - Task Package         prompt, domain, research type, files
  - SME Metadata         if present
  - Agent Run Details    per-pass stats (tokens, cost, duration, …)
  - Response             the full response_text
  - Citations            numbered list with title + URL

Usage
-----
  # Single file
  python export_completed_runs.py --input results/yukti_jain_eb583a.json

  # Single file, custom output folder
  python export_completed_runs.py --input results/yukti_jain_eb583a.json \\
      --output-dir ./reports

  # Batch: all *.json in a folder
  python export_completed_runs.py --input-dir ./results/tasks \\
      --output-dir ./reports

  # Dry run: show what would be created without writing
  python export_completed_runs.py --input-dir ./results/tasks --dry-run

Options
-------
  --input FILE        Single input JSON file
  --input-dir DIR     Batch mode: process all *.json files in this folder
  --output-dir DIR    Where to write .txt files  (default: ./exported_runs)
  --dry-run           Print file names that would be created, write nothing
  --summary           Print a summary table after processing (auto in batch)
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_OUTPUT_DIR = "./exported_runs"
LINE_WIDTH = 80


# ── Formatting helpers ────────────────────────────────────────────────────────

def banner(title: str, char: str = "=") -> str:
    bar = char * LINE_WIDTH
    return f"\n{bar}\n{title}\n{bar}\n"


def section(title: str) -> str:
    bar = "-" * LINE_WIDTH
    return f"\n{bar}\n{title}\n{bar}\n"


def field(label: str, value: Any, width: int = 22) -> str:
    label_str = f"{label:<{width}}: "
    value_str = format_value(value)
    # Indent continuation lines to align with the first value character
    indent = " " * (width + 2)
    lines = value_str.splitlines()
    if not lines:
        return f"{label_str}(none)\n"
    out = f"{label_str}{lines[0]}\n"
    for line in lines[1:]:
        out += f"{indent}{line}\n"
    return out


def format_value(v: Any) -> str:
    if v is None or v == "" or v == [] or v == {}:
        return "(none)"
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)
    if isinstance(v, float):
        return f"{v:,.6f}"
    return str(v)


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "(none)"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s  ({seconds:,.2f} s)"
    elif m:
        return f"{m}m {s}s  ({seconds:,.2f} s)"
    else:
        return f"{seconds:,.2f} s"


def format_timestamp(ts: str | None) -> str:
    if not ts:
        return "(none)"
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%Y-%m-%d  %H:%M:%S UTC")
    except ValueError:
        return ts


def format_tokens(n: int | None) -> str:
    if n is None:
        return "(none)"
    return f"{n:,}"


def timestamp_slug(ts: str | None) -> str:
    """Convert ISO timestamp to YYYYMMDD_HHMMSS for use in filenames."""
    if not ts:
        return "unknown_time"
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%Y%m%d_%H%M%S")
    except ValueError:
        clean = re.sub(r"[^0-9]", "", ts)[:14]
        return clean if clean else "unknown_time"


def safe_filename(s: str) -> str:
    return re.sub(r"[^\w\-.]", "_", s)


# ── Text block builders ───────────────────────────────────────────────────────

def build_task_overview(task: dict) -> str:
    out = banner("TASK OVERVIEW")
    out += field("Task ID",           task.get("task_id"))
    out += field("Dispatched At",     format_timestamp(task.get("dispatched_at")))
    out += field("Completed At",      format_timestamp(task.get("completed_at")))
    out += field("Total Cost (USD)",  task.get("total_cost_usd"))
    out += field("Total Duration",    format_duration(task.get("total_duration_sec")))
    out += field("Agents Attempted",  task.get("agents_attempted"))
    out += field("Agents Succeeded",  task.get("agents_succeeded"))
    out += field("Agents Failed",     task.get("agents_failed"))

    errors = task.get("agent_errors") or {}
    if errors:
        out += "\nAgent Errors:\n"
        for agent, msg in errors.items():
            out += f"  {agent}: {msg}\n"

    return out


def build_config(task: dict) -> str:
    cfg = task.get("config") or {}
    if not cfg:
        return ""
    out = section("TASK CONFIGURATION")
    out += field("Agents Configured", cfg.get("agents"))
    out += field("Passes Per Agent",  cfg.get("passes_per_agent"))
    out += field("Dry Run",           cfg.get("dry_run"))
    out += field("MCP Server URL",    cfg.get("mcp_server_url"))
    return out


def build_package(task: dict) -> str:
    pkg = task.get("package") or {}
    if not pkg:
        return ""
    out = section("TASK PACKAGE")
    out += field("Domain",             pkg.get("domain"))
    out += field("Research Type",      pkg.get("research_type"))
    out += field("IAT Type",           pkg.get("iat_type"))
    out += field("Decision Archetype", pkg.get("decision_archetype"))
    out += field("File Count",         pkg.get("file_count"))
    out += field("File Names",         pkg.get("file_names"))

    prompt = (pkg.get("prompt") or "").strip()
    if prompt:
        out += "\nPrompt:\n"
        out += "-" * 40 + "\n"
        out += prompt + "\n"
        out += "-" * 40 + "\n"

    return out


def build_sme_metadata(task: dict) -> str:
    sme = task.get("sme_metadata") or {}
    if not sme:
        return ""
    out = section("SME METADATA")
    out += field("SME Name",      sme.get("sme_name"))
    out += field("Submitted At",  sme.get("submitted_at"))
    out += field("Domain Detail", sme.get("domain_detail"))
    out += field("Drive URL",     sme.get("drive_url"))
    return out


def build_run_details(pass_data: dict, pass_index: int) -> str:
    agent = pass_data.get("agent", "unknown")
    model = pass_data.get("model", "unknown")
    out = banner(f"AGENT RUN  —  {agent.upper()}  |  {model}  (pass {pass_index})")

    out += field("Pass Task ID",    pass_data.get("task_id"))
    out += field("Agent",           pass_data.get("agent"))
    out += field("Model",           pass_data.get("model"))
    out += field("Status",          "Completed" if pass_data.get("completed") else "Failed")
    out += field("Forced Stop",     pass_data.get("forced_stop"))
    out += field("Error",           pass_data.get("error"))
    out += "\n"
    out += field("Started At",      format_timestamp(pass_data.get("started_at")))
    out += field("Completed At",    format_timestamp(pass_data.get("completed_at")))
    out += field("Duration",        format_duration(pass_data.get("total_duration_sec")))
    out += "\n"
    out += field("Input Tokens",    format_tokens(pass_data.get("input_tokens")))
    out += field("Output Tokens",   format_tokens(pass_data.get("output_tokens")))
    out += field("Cost (USD)",      pass_data.get("total_cost_usd"))
    out += field("Iterations",      pass_data.get("iterations"))
    out += field("Tool Calls",      pass_data.get("tool_calls_count"))
    out += field("Citations",       pass_data.get("citations_count"))
    out += field("Response Length", f"{pass_data.get('response_length', 0):,} characters")

    return out


def build_response(pass_data: dict) -> str:
    text = (pass_data.get("response_text") or "").strip()
    out = section("RESPONSE")
    if text:
        out += text + "\n"
    else:
        out += "(no response text)\n"
    return out


def build_citations(pass_data: dict) -> str:
    citations = pass_data.get("citations") or []
    if not citations:
        return ""

    out = section(f"CITATIONS  ({len(citations)})")
    for i, c in enumerate(citations, 1):
        # Citations can have different shapes across agents
        title = c.get("title") or ""
        url   = c.get("url")   or ""
        snip  = (c.get("snippet") or "").strip()
        idx   = c.get("index", i)

        out += f"  [{idx}]"
        if title:
            out += f" {title}\n"
            if url:
                out += f"       {url}\n"
        elif url:
            out += f" {url}\n"
        else:
            out += " (no details)\n"

        if snip:
            out += f"       {snip}\n"
        out += "\n"

    return out


# ── Core: build one txt document for one qualifying pass ──────────────────────

def build_txt(task: dict, pass_data: dict, pass_index: int) -> str:
    parts = [
        build_task_overview(task),
        build_config(task),
        build_package(task),
        build_sme_metadata(task),
        build_run_details(pass_data, pass_index),
        build_response(pass_data),
        build_citations(pass_data),
    ]
    header = (
        "=" * LINE_WIDTH + "\n"
        "DRA EVALUATION REPORT\n"
        f"Generated : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        + "=" * LINE_WIDTH + "\n"
    )
    return header + "".join(parts)


# ── Filtering ─────────────────────────────────────────────────────────────────

def qualifying_passes(task: dict) -> list[tuple[str, dict, int]]:
    """
    Return list of (agent, pass_data, pass_index) tuples for passes that
    satisfy all three filter conditions.
    """
    if (task.get("config") or {}).get("dry_run", False):
        return []  # entire task is a dry run

    results = []
    for agent, passes in (task.get("agent_results") or {}).items():
        for idx, p in enumerate(passes, 1):
            if p.get("completed") is True and p.get("forced_stop") is False:
                results.append((agent, p, idx))
    return results


# ── File processing ───────────────────────────────────────────────────────────

def normalise_to_list(data) -> list[dict]:
    if isinstance(data, dict):
        return [data]
    elif isinstance(data, list):
        return data
    return []


def process_file(
    input_path: Path,
    output_dir: Path,
    dry_run: bool,
) -> list[dict]:
    """
    Process one JSON file. Returns a list of result dicts (one per qualifying pass).
    Each dict: {task_id, agent, model, output_path, written, skipped_reason}
    """
    results = []

    try:
        with open(input_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [ERROR] Cannot read {input_path}: {e}", file=sys.stderr)
        return results

    tasks = normalise_to_list(data)

    for task in tasks:
        task_id = task.get("task_id", input_path.stem)
        passes  = qualifying_passes(task)

        if not passes:
            results.append({
                "task_id": task_id, "agent": None, "model": None,
                "output_path": None, "written": False,
                "skipped_reason": "no qualifying passes",
            })
            continue

        for agent, pass_data, pass_idx in passes:
            ts_slug   = timestamp_slug(pass_data.get("completed_at"))
            filename  = safe_filename(f"{task_id}_{ts_slug}.txt")
            out_path  = output_dir / filename
            model     = pass_data.get("model", "unknown")

            rec = {
                "task_id": task_id, "agent": agent, "model": model,
                "output_path": out_path, "written": False,
                "skipped_reason": None,
            }

            if not dry_run:
                output_dir.mkdir(parents=True, exist_ok=True)
                txt = build_txt(task, pass_data, pass_idx)
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(txt)
                rec["written"] = True

            results.append(rec)

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export completed, non-dry-run, non-force-stopped agent runs "
            "from DRA evaluation JSON files to human-readable .txt files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input",     metavar="FILE", help="Single input JSON file")
    input_group.add_argument("--input-dir", metavar="DIR",  help="Batch: folder of JSON files")

    parser.add_argument(
        "--output-dir", metavar="DIR", default=DEFAULT_OUTPUT_DIR,
        help=f"Where to write .txt files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be created without writing any files",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print a summary table (auto-enabled in batch mode)",
    )

    args = parser.parse_args()

    # Collect input files
    if args.input:
        input_files = [Path(args.input)]
    else:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"[ERROR] Input dir not found: {input_dir}", file=sys.stderr)
            sys.exit(1)
        input_files = sorted(input_dir.glob("*.json"))
        if not input_files:
            print(f"[WARN] No *.json files in {input_dir}", file=sys.stderr)
            sys.exit(0)

    output_dir = Path(args.output_dir)

    print(f"\n-- DRA Run Exporter --")
    print(f"  Mode       : {'batch' if args.input_dir else 'single file'}")
    print(f"  Filters    : completed=true  forced_stop=false  dry_run=false")
    print(f"  Output dir : {output_dir}")
    print(f"  Dry run    : {args.dry_run}")
    print()

    all_results = []

    for in_path in input_files:
        print(f"  Processing : {in_path.name}")
        file_results = process_file(in_path, output_dir, args.dry_run)
        all_results.extend(file_results)

        for rec in file_results:
            if rec["skipped_reason"]:
                print(f"    [SKIP]  {rec['task_id']}  ({rec['skipped_reason']})")
            else:
                status = "DRY RUN" if args.dry_run else "WRITTEN"
                print(
                    f"    [{status}]  {rec['task_id']}"
                    f"  agent={rec['agent']}"
                    f"  model={rec['model']}"
                )
                if rec["output_path"]:
                    print(f"             -> {rec['output_path'].name}")

    # Summary
    if args.summary or args.input_dir:
        written  = [r for r in all_results if r["output_path"] and not r["skipped_reason"]]
        skipped  = [r for r in all_results if r["skipped_reason"]]
        agents   = {}
        for r in written:
            agents[r["agent"]] = agents.get(r["agent"], 0) + 1

        print(f"\n-- Summary --")
        print(f"  Input files     : {len(input_files)}")
        print(f"  Files written   : {len(written)}")
        print(f"  Tasks skipped   : {len(skipped)}")
        if agents:
            print(f"  By agent:")
            for agent, count in sorted(agents.items()):
                print(f"    {agent:<20} {count} file(s)")
        if not args.dry_run and written:
            print(f"  Output location : {output_dir}/")

    print()


if __name__ == "__main__":
    main()