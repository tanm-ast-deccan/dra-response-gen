"""
csv_loader.py — Load SME prompt packages from a CSV file.

Reads the standard prompt_data.csv format produced by the SME intake
form, converts each row into a PromptPackage, optionally resolves
GDrive file links, and dispatches through the evaluation pipeline.

Column mapping is alias-based: COLUMN_ALIASES maps each logical field
to an ordered list of accepted header names (case-insensitive, after
stripping whitespace). First match wins. This handles the gap between
the original codebase column names ("POC Name", "Prompts", "SC", etc.)
and the actual benchmark CSV headers ("full_name", "prompt",
"sanity_check", etc.) without hardcoding either.

Output format detection is automatic: detect_output_formats(prompt)
is called for each row and populates PromptPackage.output_formats.
No extra CSV column is needed.

Usage:
    python csv_loader.py --csv prompt_data.csv --preview
    python csv_loader.py --csv prompt_data.csv --resolve-files --dispatch --dry-run
    python csv_loader.py --csv prompt_data.csv --resolve-files --dispatch --live \\
        --agents claude openai gemini perplexity \\
        --passes 1 --results-dir /data/eval_results --output dispatch_results/
"""

from __future__ import annotations

import os
import re
import sys
import csv
import json
import uuid
import asyncio
import logging
import argparse
from datetime import datetime, timezone
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env_loader import load_env
load_env()

from models import PromptPackage, DispatchConfig, DispatchResult
from file_generators import detect_output_formats
from file_resolver import FileResolver, parse_gdrive_reference
from task_dispatcher import (
    TaskDispatcher,
    dispatch_result_to_dict,
    save_dispatch_result,
)
from mcp_servers.results_store import ResultsStore

logger = logging.getLogger("dra.csv_loader")


# ─── Column alias map ─────────────────────────────────────────────────────────
# Maps logical field name → ordered list of accepted CSV header strings.
# Matching is case-insensitive after stripping whitespace.
# First match wins. Add new aliases to the END of each list.

COLUMN_ALIASES: dict[str, list[str]] = {
    # Core task fields
    "task_id":        ["task_id", "task id", "taskid", "id", "task"],
    "prompt":         ["prompt", "prompts", "prompt_text", "prompt text",
                       "question", "task description", "research question"],
    "prompt_id":      ["prompt_id", "prompt id", "prompt_no", "prompt no",
                       "s.no", "s. no", "serial no", "serial"],
    # SME / author fields
    "sme_name":       ["full_name", "full name", "poc name", "poc_name",
                       "sme_name", "sme name", "sme", "author", "name",
                       "annotator", "contributor"],
    "email":          ["ann_email", "email", "sme_email", "poc_email",
                       "e-mail", "contact email", "contact"],
    # Classification
    "research_type":  ["prompt_type", "prompt type", "category", "type",
                       "primary", "research_type", "research type",
                       "prompt category"],
    "domain":         ["domain", "domain_detail", "domain detail",
                       "vertical", "sector", "sub-domain"],
    # Evaluation metadata
    "sanity_check":   ["sanity_check", "sanity check", "sc",
                       "lazy ai", "lazy_ai", "lazy ai prediction",
                       "sanity", "trap check"],
    "solution_logic": ["solution_logic", "solution logic", "logic",
                       "solution", "answer logic", "solution steps",
                       "golden answer"],
    # File references
    "drive_url":      ["drive_link", "drive link", "drive", "drive_url",
                       "drive url", "gdrive", "gdrive_url", "gdrive url",
                       "files", "attachments", "file link", "google drive"],
    # Operational metadata
    "created_at":     ["created_at", "created at", "date", "timestamp",
                       "submission_date", "submission date", "submitted_at",
                       "submitted at"],
    "allocation_id":  ["allocation_id", "allocation id", "alloc_id",
                       "alloc id"],
    "estimated_time": ["estimated_time", "estimated time", "time",
                       "duration", "est. time", "est time", "time_mins",
                       "time (mins)", "time (minutes)"],
}

# Fields that must resolve for the loader to proceed
REQUIRED_FIELDS = {"task_id", "prompt"}


# ─── Research type alias map ──────────────────────────────────────────────────
# Maps canonical 3-letter code → all accepted spellings.
# Matching is case-insensitive after stripping whitespace.
# Add new aliases to the END of each list — first match wins on reverse lookup.

RESEARCH_TYPE_ALIASES: dict[str, list[str]] = {
    "CRP": ["CRP", "Constrained Research Prompt", "CONSTRAINED RESEARCH PROMPT",
            "constrained research prompt"],
    "RCP": ["RCP", "Relevance Compression Prompt", "RELEVANCE COMPRESSION PROMPT",
            "relevance compression prompt"],
    "SCP": ["SCP", "Structural Compliance Prompt", "STRUCTURAL COMPLIANCE PROMPT",
            "structural compliance prompt"],
    "LDP": ["LDP", "Latent Decomposition Prompt", "LATENT DECOMPOSITION PROMPT",
            "latent decomposition prompt"],
    "FSP": ["FSP", "Failure-Sensitive Prompt", "FAILURE-SENSITIVE PROMPT",
            "Failure Sensitive Prompt", "FAILURE SENSITIVE PROMPT",
            "failure-sensitive prompt", "failure sensitive prompt"],
}

# Reverse lookup: any alias (lowercased) → canonical code
_RESEARCH_TYPE_LOOKUP: dict[str, str] = {
    alias.strip().lower(): code
    for code, aliases in RESEARCH_TYPE_ALIASES.items()
    for alias in aliases
}

# IAT type per canonical code — single source of truth
_IAT_MAP: dict[str, str] = {
    "CRP": "IAT-2",
    "RCP": "IAT-1",
    "SCP": "IAT-3",
    "LDP": "IAT-1",
    "FSP": "IAT-3",
}


def normalize_research_type(raw: str) -> str:
    """
    Map any known alias to the canonical 3-letter code.

    Handles short codes (CRP), full names (Constrained Research Prompt),
    ALL CAPS (CONSTRAINED RESEARCH PROMPT), and any casing in between.
    Returns '' if the value is unrecognised.
    """
    return _RESEARCH_TYPE_LOOKUP.get(raw.strip().lower(), "")


def resolve_columns(actual_headers: list[str]) -> dict[str, str]:
    """
    Build a mapping: logical_field_name → actual_csv_column_name.

    Raises ValueError listing all missing required fields if any are
    unresolvable. Logs a warning for actual headers that match no alias
    (those columns will be ignored).

    Args:
        actual_headers: the raw fieldnames from csv.DictReader

    Returns:
        Dict mapping each logical field to its actual header string.
        Only includes fields that were successfully resolved.
    """
    norm = {h.strip().lower(): h for h in actual_headers}

    resolved: dict[str, str] = {}
    for field_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias.lower() in norm:
                resolved[field_name] = norm[alias.lower()]
                break

    # Hard check for required fields
    missing = REQUIRED_FIELDS - set(resolved)
    if missing:
        raise ValueError(
            f"Required column(s) not found: {missing}.\n"
            f"Available headers: {actual_headers}\n"
            f"Add an alias to COLUMN_ALIASES or rename the column."
        )

    # Soft warning for unmapped actual headers
    mapped_actual = set(resolved.values())
    unmapped = [h for h in actual_headers if h not in mapped_actual]
    if unmapped:
        logger.warning("Unrecognised CSV columns (ignored): %s", unmapped)

    return resolved


# ─── CSV Parsing ──────────────────────────────────────────────────────────────

def load_csv(csv_path: str) -> list[dict]:
    """
    Read the prompt CSV and return cleaned rows.

    Handles:
      - BOM-encoded UTF-8 (Excel exports)
      - Blank rows
      - Missing Prompt ID (auto-generated)
      - Whitespace cleanup
      - Arbitrary column name variations via COLUMN_ALIASES
    """
    rows = []

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        # Resolve column aliases once for the whole file
        col = resolve_columns(list(reader.fieldnames or []))
        logger.info(
            "Column mapping resolved for %s: %s",
            csv_path,
            {k: v for k, v in col.items() if k in REQUIRED_FIELDS},
        )

        for i, raw in enumerate(reader):
            # Skip fully empty rows
            if not any(v.strip() for v in raw.values() if isinstance(v, str)):
                continue

            # Skip rows with no prompt text
            prompt = raw.get(col["prompt"], "").strip()
            if not prompt:
                logger.warning("Row %d: empty prompt, skipping", i + 1)
                continue

            # ── task_id ──────────────────────────────────────────────
            prompt_id = raw.get(col.get("task_id", ""), "").strip()
            # Normalize Excel float-formatted IDs: "28115.0B" → "28115B"
            prompt_id = re.sub(r'^(\d+)\.0([A-Za-z]*)$', r'\1\2', prompt_id)
            if not prompt_id:
                sme = raw.get(col.get("sme_name", ""), "").strip()
                sme_slug = sme.lower().replace(" ", "_")[:15] if sme else "anon"
                prompt_id = f"{sme_slug}_{uuid.uuid4().hex[:6]}"

            row = {
                "task_id":        prompt_id,
                "sme_name":       raw.get(col.get("sme_name", ""), "").strip(),
                "email":          raw.get(col.get("email", ""), "").strip(),
                "submitted_at":   raw.get(col.get("created_at", ""), "").strip(),
                "research_type":  raw.get(col.get("research_type", ""), "").strip(),
                "domain_detail":  raw.get(col.get("domain", ""), "").strip(),
                "prompt":         prompt,
                "logic":          raw.get(col.get("solution_logic", ""), "").strip(),
                "sanity_check":   raw.get(col.get("sanity_check", ""), "").strip(),
                "drive_url":      raw.get(col.get("drive_url", ""), "").strip(),
                # Extra fields — stored in sme_metadata by dispatch_packages
                "prompt_id":      raw.get(col.get("prompt_id", ""), "").strip(),
                "allocation_id":  raw.get(col.get("allocation_id", ""), "").strip(),
                "estimated_time": raw.get(col.get("estimated_time", ""), "").strip(),
            }
            rows.append(row)

    logger.info("Loaded %d valid rows from %s", len(rows), csv_path)
    return rows


def row_to_package(
    row: dict,
    resolved_files: Optional[list[str]] = None,
) -> PromptPackage:
    """
    Convert a parsed CSV row into a PromptPackage.

    Also calls detect_output_formats(row["prompt"]) to auto-populate
    output_formats — no extra CSV column required.
    """
    research_type = normalize_research_type(row["research_type"])
    iat_type = _IAT_MAP.get(research_type, "")

    logic_text = row.get("logic", "")
    solution_steps = [
        line.strip()
        for line in logic_text.split("\n")
        if line.strip() and len(line.strip()) > 10
    ]

    file_paths = resolved_files or []
    if not file_paths and row.get("drive_url"):
        file_paths = [row["drive_url"]]

    # ── Auto-detect required output formats from the prompt ───────────
    # This is the key addition: no CSV column needed. Patterns are
    # validated to produce 0 FP / 0 FN on the benchmark CSV.
    output_formats = detect_output_formats(row["prompt"])
    if output_formats:
        logger.info(
            "[%s] Output formats detected from prompt: %s",
            row["task_id"], output_formats,
        )

    return PromptPackage(
        task_id=row["task_id"],
        prompt=row["prompt"],
        file_paths=file_paths,
        research_type=research_type,
        iat_type=iat_type,
        domain=row["domain_detail"],
        solution_steps=solution_steps,
        lazy_ai_prediction=row.get("sanity_check", ""),
        output_formats=output_formats,
    )


# ─── Batch Operations ─────────────────────────────────────────────────────────

def load_packages(
    csv_path: str,
    resolve_files: bool = False,
    staging_dir: Optional[str] = None,
    max_rows: Optional[int] = None,
    filter_type: Optional[str] = None,
    filter_sme: Optional[str] = None,
    task_ids: Optional[list[str]] = None,
) -> list[tuple[dict, PromptPackage]]:
    """
    Load all prompt packages from a CSV file.

    Args:
        csv_path:      path to the prompt CSV
        resolve_files: if True, download GDrive files to staging
        staging_dir:   local directory for downloaded files
        max_rows:      if set, only process the first N rows (applied BEFORE file resolution)
        filter_type:   only process rows of this research type
        filter_sme:    only process rows matching this SME name (substring)
        task_ids:      if set, only process rows whose task_id is in this list

    Returns:
        List of (raw_row, PromptPackage) tuples
    """
    rows = load_csv(csv_path)

    # ── Apply filters BEFORE file resolution ──────────────────────────
    if task_ids:
        task_id_set = set(task_ids)
        rows = [r for r in rows if r["task_id"] in task_id_set]
        logger.info("Pre-filtered to %d rows (task_ids=%s)", len(rows), task_ids)
    if filter_type:
        rows = [r for r in rows if r["research_type"] == filter_type]
        logger.info("Pre-filtered to %d rows (type=%s)", len(rows), filter_type)
    if filter_sme:
        rows = [r for r in rows if filter_sme.lower() in r.get("sme_name", "").lower()]
        logger.info("Pre-filtered to %d rows (sme=%s)", len(rows), filter_sme)
    if max_rows:
        rows = rows[:max_rows]
        logger.info("Limited to %d rows before resolution", max_rows)

    packages = []

    resolver = None
    if resolve_files:
        staging = staging_dir or f"/tmp/dra_staging_{uuid.uuid4().hex[:8]}"
        resolver = FileResolver(staging_dir=staging)
        logger.info("File resolver staging: %s", staging)

    for row in rows:
        resolved_files = None

        if resolve_files and row.get("drive_url"):
            try:
                resolved_files = resolver.resolve([row["drive_url"]])
                logger.info(
                    "[%s] Resolved %d files from GDrive",
                    row["task_id"], len(resolved_files),
                )
            except Exception as e:
                logger.error("[%s] File resolution failed: %s", row["task_id"], e)
                resolved_files = []

        package = row_to_package(row, resolved_files)
        packages.append((row, package))

    logger.info(
        "Created %d packages (%d with resolved files)",
        len(packages),
        sum(1 for _, p in packages if p.file_paths and os.path.exists(p.file_paths[0])),
    )
    return packages


async def dispatch_packages(
    packages: list[tuple[dict, PromptPackage]],
    config: DispatchConfig,
    results_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    max_parallel: int = 1,
) -> list[DispatchResult]:
    """
    Dispatch all packages through the evaluation pipeline.

    When max_parallel=1 (default): sequential, same as before.
    When max_parallel>1: bounded concurrency via semaphore.
    Each package still fans out to all agents concurrently within itself.
    """
    dispatcher = TaskDispatcher()
    store = None

    if results_dir:
        store = ResultsStore(results_dir)
        store.load_index()
        # Wire output_files_base_dir so the dispatcher creates per-task
        # subdirectories under results_dir/files/
        if config.output_files_base_dir is None:
            config.output_files_base_dir = os.path.join(results_dir, "files")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    total = len(packages)
    results = []
    sem = asyncio.Semaphore(max_parallel)
    results_lock = asyncio.Lock()

    if max_parallel > 1:
        logger.info("Parallel dispatch: max %d tasks concurrently", max_parallel)

    async def _run_one(i: int, row: dict, package: PromptPackage):
        """Run a single task, bounded by semaphore."""
        async with sem:
            logger.info(
                "━━━ Dispatching %d/%d: %s [%s, %s] by %s ━━━",
                i, total,
                package.task_id,
                package.research_type or "?",
                package.iat_type or "?",
                row.get("sme_name", "?"),
            )

            if row.get("drive_url") and not package.file_paths:
                logger.warning(
                    "[%s] Skipping: GDrive ref resolved to 0 files: %s",
                    package.task_id, row["drive_url"][:80],
                )
                return None

            try:
                result = await dispatcher.dispatch(package, config)

                if store:
                    result_dict = dispatch_result_to_dict(result)
                    result_dict["sme_metadata"] = {
                        "sme_name":       row.get("sme_name", ""),
                        "email":          row.get("email", ""),
                        "submitted_at":   row.get("submitted_at", ""),
                        "domain_detail":  row.get("domain_detail", ""),
                        "drive_url":      row.get("drive_url", ""),
                        "prompt_id":      row.get("prompt_id", ""),
                        "allocation_id":  row.get("allocation_id", ""),
                        "estimated_time": row.get("estimated_time", ""),
                    }
                    async with results_lock:
                        await store.store_result(result_dict)

                if output_dir:
                    out_path = os.path.join(output_dir, f"{package.task_id}.json")
                    save_dispatch_result(result, out_path)

                logger.info(
                    "  → %d/%d agents succeeded, $%.4f, %.1fs",
                    len(result.agents_succeeded), len(result.agents_attempted),
                    result.total_cost_usd, result.total_duration_sec,
                )
                return result

            except Exception as e:
                logger.error("  → FAILED: %s", e)
                return None

    if max_parallel <= 1:
        # Sequential (original behavior)
        for i, (row, package) in enumerate(packages, 1):
            result = await _run_one(i, row, package)
            if result:
                results.append(result)
    else:
        # Parallel with bounded concurrency
        tasks = [
            _run_one(i, row, package)
            for i, (row, package) in enumerate(packages, 1)
        ]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in raw_results:
            if isinstance(r, Exception):
                logger.error("Task exception: %s", r)
            elif r is not None:
                results.append(r)

    logger.info("━━━ Batch complete: %d/%d dispatched ━━━", len(results), total)
    if store:
        stats = store.get_stats()
        logger.info(
            "  Results store: %d total, %d scored, $%.2f total cost",
            stats["total_results"], stats["scored_results"], stats["total_cost_usd"],
        )

    return results


# ─── Preview ──────────────────────────────────────────────────────────────────

def preview_csv(csv_path: str):
    """Print a summary of what would be loaded from the CSV."""
    rows = load_csv(csv_path)

    print(f"\n{'═' * 70}")
    print(f"  CSV Preview: {csv_path}")
    print(f"  {len(rows)} prompt packages")
    print(f"{'═' * 70}\n")

    types = {}
    domains = {}
    smes = {}
    file_output_count = 0

    for r in rows:
        t = r["research_type"] or "unset"
        types[t] = types.get(t, 0) + 1
        d = r["domain_detail"] or "unset"
        domains[d] = domains.get(d, 0) + 1
        s = r["sme_name"] or "unknown"
        smes[s] = smes.get(s, 0) + 1
        if detect_output_formats(r["prompt"]):
            file_output_count += 1

    print(f"  By Research Type:")
    iat_map = {
        "CRP": "IAT-2", "FSP": "IAT-3", "SCP": "IAT-3",
        "RCP": "IAT-1", "LDP": "IAT-1",
        "Constrained Research Prompt": "IAT-2",
        "Failure-Sensitive Prompt": "IAT-3",
        "Structural Compliance Prompt": "IAT-3",
        "Relevance Compression Prompt": "IAT-1",
        "Latent Decomposition Prompt": "IAT-1",
    }
    for t, c in sorted(types.items()):
        iat = iat_map.get(t, "?")
        print(f"    {t:5s} ({iat:5s}): {c:2d} prompts")

    print(f"\n  By Domain ({len(domains)} unique):")
    for d, c in sorted(domains.items(), key=lambda x: -x[1])[:10]:
        print(f"    {d:25s}: {c:2d}")
    if len(domains) > 10:
        print(f"    ... and {len(domains) - 10} more")

    print(f"\n  By SME ({len(smes)} unique):")
    for s, c in sorted(smes.items(), key=lambda x: -x[1])[:10]:
        print(f"    {s:30s}: {c:2d}")
    if len(smes) > 10:
        print(f"    ... and {len(smes) - 10} more")

    files = sum(1 for r in rows if parse_gdrive_reference(r["drive_url"]) and
                parse_gdrive_reference(r["drive_url"])["type"] == "file")
    folders = sum(1 for r in rows if parse_gdrive_reference(r["drive_url"]) and
                 parse_gdrive_reference(r["drive_url"])["type"] == "folder")
    print(f"\n  GDrive Links: {files} files, {folders} folders")
    print(f"  File output required: {file_output_count} prompt(s)")

    print(f"\n{'─' * 70}")
    print(f"  Sample Packages (first 3):")
    print(f"{'─' * 70}")
    for row in rows[:3]:
        pkg = row_to_package(row)
        print(f"\n  [{pkg.task_id}]")
        print(f"    SME:          {row['sme_name']}")
        print(f"    Type:         {pkg.research_type} ({pkg.iat_type})")
        print(f"    Domain:       {row['domain_detail']}")
        print(f"    Prompt:       {pkg.prompt[:80]}...")
        print(f"    Logic:        {len(pkg.solution_steps)} steps")
        print(f"    SC:           {pkg.lazy_ai_prediction[:60]}...")
        print(f"    Drive:        {row['drive_url'][:60]}...")
        print(f"    Output fmts:  {pkg.output_formats or 'none'}")

    print(f"\n{'═' * 70}\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

async def cli_main():
    parser = argparse.ArgumentParser(
        description="Load SME prompts from CSV and dispatch to agents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python csv_loader.py --csv prompt_data.csv --preview
  python csv_loader.py --csv prompt_data.csv --dispatch --dry-run
  python csv_loader.py --csv prompt_data.csv --resolve-files --dispatch --dry-run
  python csv_loader.py --csv prompt_data.csv \\
      --resolve-files --staging-dir /tmp/eval_files \\
      --dispatch --live \\
      --agents claude openai \\
      --passes 1 \\
      --results-dir /data/eval_results \\
      --output dispatch_results/
  python csv_loader.py --csv prompt_data.csv --dispatch --dry-run \\
      --filter-type FSP --max-rows 5
        """,
    )

    parser.add_argument("--csv", required=True, help="Path to prompt CSV")
    parser.add_argument("--preview", action="store_true",
                        help="Preview CSV contents without processing")
    parser.add_argument("--dispatch", action="store_true",
                        help="Dispatch packages to agents")
    parser.add_argument("--resolve-files", action="store_true",
                        help="Download GDrive files to local staging")
    parser.add_argument("--staging-dir", default=None)
    parser.add_argument("--agents", nargs="*",
                        default=["claude", "openai", "gemini", "perplexity"])
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--max-parallel-tasks", type=int, default=1,
                        help="Max tasks to run concurrently (default: 1 = sequential). "
                             "Set to 3-5 for faster batch runs. OpenRouter handles 5+ "
                             "concurrent requests comfortably.")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--filter-type", default=None,
                        choices=["CRP", "RCP", "SCP", "LDP", "FSP",
                                 "Constrained Research Prompt",
                                 "Relevance Compression Prompt",
                                 "Structural Compliance Prompt",
                                 "Latent Decomposition Prompt",
                                 "Failure-Sensitive Prompt"])
    parser.add_argument("--filter-sme", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--task-ids", default=None,
                        help="Comma-separated task IDs to run (e.g. tsk_abc,tsk_def). "
                             "Skips all other tasks in the CSV.")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Per-task timeout in seconds (overrides model default of 900s). "
                             "Useful for complex tasks that need more time.")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.preview:
        preview_csv(args.csv)
        return

    task_ids = None
    if args.task_ids:
        task_ids = [t.strip() for t in args.task_ids.split(",") if t.strip()]

    packages = load_packages(
        args.csv,
        resolve_files=args.resolve_files,
        staging_dir=args.staging_dir,
        max_rows=args.max_rows,
        filter_type=args.filter_type,
        filter_sme=args.filter_sme,
        task_ids=task_ids,
    )

    if not packages:
        print("No packages to process after filtering.")
        return

    if args.dispatch:
        config = DispatchConfig(
            agents=args.agents,
            passes_per_agent=args.passes,
            dry_run=not args.live,
            # output_files_base_dir is set inside dispatch_packages
            # once results_dir is known
        )
        if args.timeout is not None:
            for _, pkg in packages:
                pkg.timeout_seconds = args.timeout
            logger.info("Timeout override: %ds per task", args.timeout)

        results = await dispatch_packages(
            packages,
            config,
            results_dir=args.results_dir,
            output_dir=args.output,
            max_parallel=args.max_parallel_tasks,
        )

        total_cost = sum(r.total_cost_usd for r in results)
        total_succeeded = sum(len(r.agents_succeeded) for r in results)
        total_attempted = sum(len(r.agents_attempted) for r in results)

        print(f"\n{'═' * 70}")
        print(f"  BATCH DISPATCH COMPLETE")
        print(f"{'═' * 70}")
        print(f"  Tasks dispatched:  {len(results)}/{len(packages)}")
        print(f"  Agent runs:        {total_succeeded}/{total_attempted} succeeded")
        print(f"  Total cost:        ${total_cost:.4f}")
        if args.results_dir:
            print(f"  Results stored:    {args.results_dir}")
        if args.output:
            print(f"  JSON files:        {args.output}/")
        print(f"{'═' * 70}\n")
    else:
        print(f"\n  Loaded {len(packages)} packages. Use --dispatch to run them.\n")
        for _, p in packages[:5]:
            fmts = f"  [{','.join(p.output_formats)}]" if p.output_formats else ""
            print(f"    {p.task_id:30s}  {p.research_type:3s}  {p.iat_type:5s}  "
                  f"{len(p.prompt):5d} chars{fmts}")
        if len(packages) > 5:
            print(f"    ... and {len(packages) - 5} more")
        print()


if __name__ == "__main__":
    asyncio.run(cli_main())