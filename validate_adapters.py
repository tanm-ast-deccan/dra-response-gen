"""
validate_adapters.py — Dry-run validation for all four agent adapters.

Runs two test tasks (one Closed, one Open) through all four adapters
in dry-run mode. Validates:
  1. All adapters import cleanly
  2. ResearchTask fields propagate correctly
  3. IAT enforcement logic fires for each agent
  4. AgentResult fields are populated
  5. Agent-specific behaviors are documented in output

Usage:
    cd deep_research_eval
    python validate_adapters.py
"""

import asyncio
import json
import os
import sys
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from models import ResearchTask, AgentResult
from adapters import (
    ClaudeAdapter,
    OpenAIAdapter,
    GeminiAdapter,
    PerplexityAdapter,
    AGENT_REGISTRY,
)


# ─── Test tasks ──────────────────────────────────────────────────────────

CLOSED_TASK = ResearchTask(
    task_id="VAL-001-CLOSED",
    prompt=(
        "Based on the provided financial statements and management commentary, "
        "determine the adjusted EBITDA for FY2024, excluding discontinued "
        "operations and one-time restructuring charges. Present your finding "
        "as a single JSON object with fields: adjusted_ebitda, adjustments_applied, "
        "source_references."
    ),
    file_paths=[
        "/tmp/eval_test/financial_statements_fy24.xlsx",
        "/tmp/eval_test/management_commentary.pdf",
        "/tmp/eval_test/restructuring_memo.docx",
    ],
    web_search_enabled=False,
    research_type="RCP",
    iat_type="IAT-1",
    decision_archetype="GoNoGo",
    domain="Financial Modeling",
)

OPEN_TASK = ResearchTask(
    task_id="VAL-002-OPEN",
    prompt=(
        "Analyze the current Euribor rate environment and its impact on the "
        "proposed variable-rate debt structure in the attached term sheet. "
        "The term sheet contains a placeholder rate of 3.50% — you MUST "
        "search for and use the actual current 3-month Euribor rate. "
        "Produce a risk assessment with Go/No-Go recommendation."
    ),
    file_paths=[
        "/tmp/eval_test/term_sheet_draft.pdf",
        "/tmp/eval_test/company_financials.xlsx",
    ],
    web_search_enabled=True,
    research_type="Failure-Sensitive Prompt",
    iat_type="IAT-3",
    decision_archetype="GoNoGo",
    domain="Financial Modeling",
)


def create_test_files():
    """Create dummy test files for validation."""
    os.makedirs("/tmp/eval_test", exist_ok=True)
    for task in [CLOSED_TASK, OPEN_TASK]:
        for fpath in task.file_paths:
            if not os.path.exists(fpath):
                with open(fpath, "w") as f:
                    f.write(f"[Test placeholder for {os.path.basename(fpath)}]")


def validate_result(result: AgentResult, task: ResearchTask, agent_name: str):
    """Validate that the AgentResult meets expectations."""
    errors = []

    # Basic field checks
    if result.task_id != task.task_id:
        errors.append(f"task_id mismatch: {result.task_id} != {task.task_id}")
    if result.agent != agent_name:
        errors.append(f"agent mismatch: {result.agent} != {agent_name}")
    if not result.model:
        errors.append("model is empty")
    if not result.response_text:
        errors.append("response_text is empty")
    if not result.started_at:
        errors.append("started_at is empty")
    if not result.completed_at:
        errors.append("completed_at is empty")
    if not result.completed:
        errors.append(f"completed=False (error: {result.error})")

    # IAT-specific checks
    if task.is_closed:
        # For Closed tasks, the dry-run report should mention IAT/Closed
        text_lower = result.response_text.lower()
        if "closed" not in text_lower and "iat" not in text_lower:
            errors.append("Closed task report doesn't mention IAT/Closed status")

    return errors


async def run_validation():
    """Run all four adapters in dry-run mode on both test tasks."""
    create_test_files()

    print("=" * 70)
    print("  DRA Adapter Validation — Dry Run Mode")
    print("=" * 70)
    print()

    # Instantiate all adapters in dry-run mode
    adapters = {
        "claude": ClaudeAdapter(dry_run=True),
        "openai": OpenAIAdapter(dry_run=True),
        "gemini": GeminiAdapter(dry_run=True),
        "perplexity": PerplexityAdapter(dry_run=True),
    }

    tasks = {
        "CLOSED (IAT-1, RCP)": CLOSED_TASK,
        "OPEN (IAT-3, FSP)": OPEN_TASK,
    }

    total_tests = 0
    total_passed = 0
    all_results = []

    for task_label, task in tasks.items():
        print(f"\n{'─' * 70}")
        print(f"  Task: {task_label}")
        print(f"  ID: {task.task_id}")
        print(f"  Type: {task.research_type} | IAT: {task.iat_type}")
        print(f"  Files: {len(task.file_paths)} | Web: {task.web_search_enabled}")
        print(f"{'─' * 70}")

        for agent_name, adapter in adapters.items():
            total_tests += 1
            print(f"\n  [{agent_name.upper()}]")

            try:
                result = await adapter.run(task)
                errors = validate_result(result, task, agent_name)

                if errors:
                    print(f"    ❌ FAIL: {len(errors)} error(s)")
                    for err in errors:
                        print(f"       • {err}")
                else:
                    print(f"    ✅ PASS")
                    total_passed += 1

                # Print key fields
                print(f"    Model:     {result.model}")
                print(f"    Duration:  {result.total_duration_sec:.3f}s")
                print(f"    Report:    {len(result.response_text)} chars")
                print(f"    Citations: {len(result.citations)}")
                if result.error:
                    print(f"    Note:      {result.error[:80]}")

                all_results.append({
                    "agent": agent_name,
                    "task": task.task_id,
                    "passed": len(errors) == 0,
                    "errors": errors,
                    "report_length": len(result.response_text),
                    "citations": len(result.citations),
                })

            except Exception as e:
                print(f"    ❌ EXCEPTION: {e}")
                all_results.append({
                    "agent": agent_name,
                    "task": task.task_id,
                    "passed": False,
                    "errors": [str(e)],
                })

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print(f"  RESULTS: {total_passed}/{total_tests} passed")
    print(f"{'═' * 70}")

    # ── Registry check ───────────────────────────────────────────────
    print(f"\n  Adapter Registry:")
    for name, cls in AGENT_REGISTRY.items():
        print(f"    {name:12s} → {cls.__name__}")

    # ── Structural comparison ────────────────────────────────────────
    print(f"\n  Agent Capabilities (from adapter design):")
    caps = {
        "claude":     {"file_upload": "Custom (inline/tools)", "mcp": "Yes (custom tools)", "context": "200K", "search_disable": "Yes"},
        "openai":     {"file_upload": "Files API",             "mcp": "Yes (native)",       "context": "200K", "search_disable": "Yes"},
        "gemini":     {"file_upload": "Files API",             "mcp": "No",                 "context": "1M",   "search_disable": "Yes"},
        "perplexity": {"file_upload": "Inline only",           "mcp": "No",                 "context": "128K", "search_disable": "No (instruction only)"},
    }
    fmt = "    {:<12s} {:>20s} {:>15s} {:>8s} {:>25s}"
    print(fmt.format("Agent", "File Upload", "MCP", "Context", "Search Disable"))
    print(fmt.format("─" * 12, "─" * 20, "─" * 15, "─" * 8, "─" * 25))
    for prov, cap in caps.items():
        print(fmt.format(prov, cap["file_upload"], cap["mcp"], cap["context"], cap["search_disable"]))

    print()
    return total_passed == total_tests


if __name__ == "__main__":
    success = asyncio.run(run_validation())
    sys.exit(0 if success else 1)
