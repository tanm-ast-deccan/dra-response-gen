"""
task_dispatcher.py — Orchestrates evaluation tasks across all agents.

This is the heart of the evaluation framework. It takes a PromptPackage
(what the SME created) and fans it out to all configured agent adapters
concurrently, collecting results for downstream scoring.

Architecture:
    ┌──────────────────────────────────────────────────────────────┐
    │                     TaskDispatcher                           │
    │                                                              │
    │  PromptPackage + DispatchConfig                              │
    │       │                                                      │
    │       ├── validate_package()                                 │
    │       ├── to_research_task()                                 │
    │       │                                                      │
    │       ├── asyncio.gather(                                    │
    │       │     _run_agent("claude",   task, passes=K),          │
    │       │     _run_agent("openai",   task, passes=K),          │
    │       │     _run_agent("gemini",   task, passes=K),          │
    │       │     _run_agent("perplexity", task, passes=K),        │
    │       │   )                                                  │
    │       │                                                      │
    │       ├── aggregate_results()                                │
    │       └── DispatchResult                                     │
    └──────────────────────────────────────────────────────────────┘

    Each _run_agent call:
      1. Creates adapter (from AGENT_REGISTRY) with per-agent config
      2. Runs the adapter K times (Pass@K) sequentially per agent
      3. Wraps each run in asyncio.wait_for for timeout isolation
      4. Catches all exceptions — one agent failing never kills others

Key design decisions:
    - Agents run concurrently, passes within an agent run sequentially.
      This is because passes should be independent samples, and some
      agents (OpenAI) have per-API-key rate limits.
    - Each agent gets its own timeout, isolated from others.
    - IAT enforcement is controlled by config.enforce_iat (default: False).
      When False, web search is always enabled regardless of IAT type,
      allowing responses to be collected for all tasks. Enable for
      benchmark-valid closed-corpus runs.
    - The dispatcher does NOT score results — that's the scorer's job.
      It only collects and aggregates.

Usage:
    dispatcher = TaskDispatcher()
    result = await dispatcher.dispatch(package, config)

    # Or with the convenience function:
    result = await dispatch_task(package, agents=["claude", "openai"])
"""

from __future__ import annotations

import os
import sys
import json
import asyncio
import logging
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Optional

from env_loader import load_env
load_env()


from models import (
    PromptPackage,
    DispatchConfig,
    DispatchResult,
    ResearchTask,
    AgentResult,
)
from adapters import AGENT_REGISTRY

logger = logging.getLogger("dra.dispatcher")


# ─── Validation ──────────────────────────────────────────────────────────

class PackageValidationError(Exception):
    """Raised when a PromptPackage fails pre-dispatch validation."""
    pass


def validate_package(package: PromptPackage) -> list[str]:
    """
    Validate a PromptPackage before dispatch.

    Returns a list of warning strings. Raises PackageValidationError
    for hard failures that should block dispatch.

    Checks:
      - Required fields present
      - Research type is valid
      - IAT type is valid
      - File paths exist (warning if missing)
      - Solution logic present (warning if missing — needed for scoring)
    """
    errors = []
    warnings = []

    # ── Hard requirements ─────────────────────────────────────────
    if not package.task_id:
        errors.append("task_id is required")
    if not package.prompt or len(package.prompt.strip()) < 20:
        errors.append("prompt is required (minimum 20 characters)")

    valid_types = {
        "CRP", "RCP", "SCP", "LDP", "FSP",
        "Constrained Research Prompt", "Relevance Compression Prompt",
        "Structural Compliance Prompt", "Latent Decomposition Prompt",
        "Failure-Sensitive Prompt", "",
    }
    if package.research_type not in valid_types:
        errors.append(
            f"research_type '{package.research_type}' invalid. "
            f"Must be one of: {valid_types - {''}}"
        )

    valid_iats = {"IAT-1", "IAT-2", "IAT-3", ""}
    if package.iat_type not in valid_iats:
        errors.append(
            f"iat_type '{package.iat_type}' invalid. "
            f"Must be one of: {valid_iats - {''}}"
        )

    if errors:
        raise PackageValidationError(
            "Package validation failed:\n  " + "\n  ".join(errors)
        )

    # ── Warnings (non-blocking) ───────────────────────────────────
    for fpath in package.file_paths:
        if not os.path.exists(fpath):
            warnings.append(f"File not found: {fpath}")

    if not package.file_paths:
        warnings.append("No files attached — most evaluation tasks require data files")

    if not package.research_type:
        warnings.append("research_type not set — scoring will use universal criteria only")

    if not package.iat_type:
        warnings.append("iat_type not set — web search will default to enabled")

    if not package.solution_steps:
        warnings.append("solution_steps empty — golden answer scoring will be unavailable")

    if not package.lazy_ai_prediction:
        warnings.append("lazy_ai_prediction empty — trap analysis will be skipped")

    return warnings


# ─── Agent eligibility ────────────────────────────────────────────────

def check_agent_eligibility(
    agent_name: str,
    package: PromptPackage,
) -> tuple[bool, str]:
    """
    Check whether an agent is eligible for this specific task.

    Returns (eligible, reason). Some agent-task combinations are
    structurally problematic (e.g., Perplexity on IAT-1). We still
    run them (per the user's decision), but log the structural caveat.
    """
    if agent_name not in AGENT_REGISTRY:
        return False, f"Unknown agent: {agent_name}"

    # Structural caveats (logged, not blocking)
    if agent_name == "perplexity" and package.is_closed:
        return True, (
            "STRUCTURAL CAVEAT: Perplexity cannot disable web search. "
            "IAT-1 enforcement uses instruction injection + post-hoc "
            "compliance audit. Results should be interpreted with caution."
        )

    # Gemini on very large corpora without MCP
    total_file_size = sum(
        os.path.getsize(f) for f in package.file_paths
        if os.path.exists(f)
    )
    if agent_name in ("openai", "gemini") and total_file_size > 3_000_000:
        return True, (
            "NOTE: Large corpus with native file upload. APEX finding: "
            "native file handling can massively inflate token counts. "
            "Consider MCP-based file access (Track B) for fairer comparison."
        )

    return True, ""


# ─── The Dispatcher ──────────────────────────────────────────────────────

class TaskDispatcher:
    """
    Orchestrates evaluation tasks across all agent adapters.

    Usage:
        dispatcher = TaskDispatcher()

        # Dispatch to all agents, 1 pass each
        result = await dispatcher.dispatch(package)

        # Dispatch with specific config
        config = DispatchConfig(
            agents=["claude", "openai"],
            passes_per_agent=3,
            dry_run=True,
        )
        result = await dispatcher.dispatch(package, config)
    """

    def __init__(self):
        self._adapter_cache: dict = {}  # (agent_name, dry_run) → adapter instance

    async def dispatch(
        self,
        package: PromptPackage,
        config: Optional[DispatchConfig] = None,
    ) -> DispatchResult:
        """
        Fan out a PromptPackage to all configured agents.

        This is the main entry point. It:
          1. Validates the package
          2. Checks agent eligibility
          3. Converts package → ResearchTask
          4. Runs all agents concurrently
          5. Aggregates and returns DispatchResult
        """
        config = config or DispatchConfig()
        started_at = datetime.now(timezone.utc)

        logger.info(
            "Dispatching %s to %d agents (Pass@%d, dry_run=%s)",
            package.task_id,
            len(config.agents),
            config.passes_per_agent,
            config.dry_run,
        )

        # ── Step 1: Validate ──────────────────────────────────────
        warnings = validate_package(package)
        for w in warnings:
            logger.warning("[%s] %s", package.task_id, w)

        # ── Step 2: Check eligibility ─────────────────────────────
        eligible_agents = []
        for agent_name in config.agents:
            ok, reason = check_agent_eligibility(agent_name, package)
            if ok:
                eligible_agents.append(agent_name)
                if reason:
                    logger.warning("[%s/%s] %s", package.task_id, agent_name, reason)
            else:
                logger.error("[%s/%s] Ineligible: %s", package.task_id, agent_name, reason)

        if not eligible_agents:
            raise PackageValidationError("No eligible agents for this task")

        # ── Step 3: Convert to ResearchTask ───────────────────────
        task = package.to_research_task()

        # IAT enforcement switch — when disabled (default), web search is
        # always on regardless of IAT type, so we get responses for all tasks.
        # Enable config.enforce_iat=True for benchmark-valid closed-corpus runs.
        if not config.enforce_iat:
            task.web_search_enabled = True

        # Set output_files_dir for EVERY task (not just when detector fires).
        # The model may generate deliverable files even when output_formats
        # is empty — the regex detector is advisory, not authoritative.
        # Each task gets its own subdirectory so multi-agent results
        # don't collide. Skip directory creation on dry runs.
        if not config.dry_run:
            if config.output_files_base_dir:
                task.output_files_dir = os.path.join(
                    config.output_files_base_dir, package.task_id
                )
            else:
                task.output_files_dir = os.path.join(
                    tempfile.gettempdir(), "dra_output_files", package.task_id
                )
            os.makedirs(task.output_files_dir, exist_ok=True)
            logger.info(
                "[%s] Output files dir: %s (detected formats: %s)",
                package.task_id, task.output_files_dir,
                task.output_formats or "(none — model decides)",
            )

        # ── Step 4: Fan out to agents ──────────────────────────
        # Use a semaphore to limit concurrency
        sem = asyncio.Semaphore(config.max_concurrent)

        async def run_with_sem(agent_name: str) -> tuple[str, list[AgentResult]]:
            async with sem:
                return await self._run_agent(
                    agent_name, task, config
                )

        # Run all agents concurrently
        agent_tasks = [
            run_with_sem(agent_name)
            for agent_name in eligible_agents
        ]

        raw_results = await asyncio.gather(
            *agent_tasks, return_exceptions=True
        )

        # ── Step 5: Aggregate results ─────────────────────────────
        completed_at = datetime.now(timezone.utc)

        agent_results = {}
        agents_succeeded = []
        agents_failed = []
        agent_errors = {}
        total_cost = 0.0

        for agent_name, result in zip(eligible_agents, raw_results):
            if isinstance(result, Exception):
                # Agent-level exception (not per-pass)
                agents_failed.append(agent_name)
                agent_errors[agent_name] = str(result)
                logger.error(
                    "[%s/%s] Agent failed: %s",
                    package.task_id, agent_name, result,
                )
                agent_results[agent_name] = []
            else:
                agent_name_out, results_list = result
                agent_results[agent_name_out] = results_list

                # Check if any pass succeeded
                any_success = any(
                    r.completed and not r.error
                    for r in results_list
                )
                if any_success:
                    agents_succeeded.append(agent_name_out)
                else:
                    agents_failed.append(agent_name_out)
                    # Collect error from first failed result
                    for r in results_list:
                        if r.error:
                            agent_errors[agent_name_out] = r.error
                            break

                # Accumulate cost
                total_cost += sum(r.total_cost_usd for r in results_list)

        dispatch_result = DispatchResult(
            task_id=package.task_id,
            package=package,
            config=config,
            agent_results=agent_results,
            agents_attempted=list(eligible_agents),
            agents_succeeded=agents_succeeded,
            agents_failed=agents_failed,
            agent_errors=agent_errors,
            total_cost_usd=round(total_cost, 6),
            total_duration_sec=(completed_at - started_at).total_seconds(),
            dispatched_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
        )

        # ── Log summary ───────────────────────────────────────────
        logger.info(
            "[%s] Dispatch complete: %d/%d agents succeeded, "
            "cost=$%.2f, duration=%.1fs",
            package.task_id,
            len(agents_succeeded),
            len(eligible_agents),
            total_cost,
            dispatch_result.total_duration_sec,
        )

        return dispatch_result

    # ── Per-agent timeout defaults (seconds) ─────────────────────────────
    # Gemini DR and o3 regularly take 30-60 min for hard FSP/CRP tasks.
    # Can be overridden per-agent via config.agent_overrides:
    #   config.agent_overrides = {"gemini": {"timeout_seconds": 7200}}
    AGENT_TIMEOUT_DEFAULTS = {
        "gemini":     3600,   # 60 min — Interactions API is slow
        "openai":     3600,   # 60 min — o3-deep-research can run long
        "claude":      900,   # 15 min — agentic loop, usually faster
        "qwen":       3600,   # 30 min — local tools + self-verification
        "perplexity":  300,   #  5 min — synchronous, fast
    }

    HEARTBEAT_INTERVAL = 300  # log a "still running" message every 5 minutes

    async def _run_with_heartbeat(
        self,
        coro,
        agent_name: str,
        task_id: str,
        timeout: float,
    ):
        """
        Run an adapter coroutine with:
          - A periodic heartbeat log so silent tasks are visible
          - A hard dispatcher-level timeout via asyncio.wait_for

        The heartbeat does NOT affect the timeout — it purely gives
        log visibility so you know whether the adapter is alive or stuck.
        Stale detection (no API response changing) belongs in the adapter.
        """
        async def _heartbeat():
            elapsed = 0
            while True:
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)
                elapsed += self.HEARTBEAT_INTERVAL
                logger.info(
                    "[%s/%s] Still running... (%dm elapsed, timeout=%dm)",
                    task_id, agent_name,
                    elapsed // 60,
                    int(timeout) // 60,
                )

        heartbeat_task = asyncio.create_task(_heartbeat())
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

    async def _run_agent(
        self,
        agent_name: str,
        task: ResearchTask,
        config: DispatchConfig,
    ) -> tuple[str, list[AgentResult]]:
        """
        Run a single agent for K passes.

        Passes run sequentially within an agent to:
          - Avoid per-API-key rate limits
          - Ensure independent samples (no request batching)
          - Make cost tracking predictable

        Each pass gets its own timeout via asyncio.wait_for (inside
        _run_with_heartbeat), which also emits a heartbeat log every
        5 minutes so silent tasks are immediately visible.
        """
        adapter = self._get_adapter(agent_name, config)
        results: list[AgentResult] = []

        # Resolve timeout: agent_overrides > per-agent default > task.timeout_seconds
        overrides = config.agent_overrides.get(agent_name, {})
        agent_timeout = overrides.get(
            "timeout_seconds",
            self.AGENT_TIMEOUT_DEFAULTS.get(agent_name, task.timeout_seconds),
        )
        dispatcher_timeout = agent_timeout + 60

        for pass_num in range(1, config.passes_per_agent + 1):
            pass_label = f"{agent_name}/pass{pass_num}"

            # Generate a unique task_id per pass for tracing
            pass_task = ResearchTask(
                task_id=f"{task.task_id}_{agent_name}_p{pass_num}",
                prompt=task.prompt,
                file_paths=task.file_paths,
                web_search_enabled=task.web_search_enabled,
                max_iterations=task.max_iterations,
                max_cost_usd=task.max_cost_usd,
                timeout_seconds=task.timeout_seconds,
                research_type=task.research_type,
                iat_type=task.iat_type,
                decision_archetype=task.decision_archetype,
                domain=task.domain,
                output_formats=task.output_formats,       # propagate file output requirements
                output_files_dir=task.output_files_dir,   # propagate output dir
            )

            logger.info("[%s] Starting %s", task.task_id, pass_label)

            try:
                result = await self._run_with_heartbeat(
                    adapter.run(pass_task),
                    agent_name=agent_name,
                    task_id=task.task_id,
                    timeout=dispatcher_timeout,
                )
                results.append(result)

                logger.info(
                    "[%s] %s complete: %s, cost=$%.4f, %d chars",
                    task.task_id,
                    pass_label,
                    "OK" if result.completed else "FAILED",
                    result.total_cost_usd,
                    len(result.response_text),
                )

            except asyncio.TimeoutError:
                logger.error(
                    "[%s] %s timed out (dispatcher-level)",
                    task.task_id, pass_label,
                )
                results.append(AgentResult(
                    task_id=pass_task.task_id,
                    agent=agent_name,
                    model="timeout",
                    response_text="",
                    completed=False,
                    forced_stop=True,
                    error=f"Dispatcher timeout after {dispatcher_timeout}s",
                    started_at=datetime.now(timezone.utc).isoformat(),
                    completed_at=datetime.now(timezone.utc).isoformat(),
                ))

                if config.fail_fast:
                    break

            except Exception as e:
                logger.error(
                    "[%s] %s exception: %s",
                    task.task_id, pass_label, e,
                )
                results.append(AgentResult(
                    task_id=pass_task.task_id,
                    agent=agent_name,
                    model="error",
                    response_text="",
                    completed=False,
                    forced_stop=True,
                    error=str(e),
                    started_at=datetime.now(timezone.utc).isoformat(),
                    completed_at=datetime.now(timezone.utc).isoformat(),
                ))

                if config.fail_fast:
                    break

        return agent_name, results

    def _get_adapter(self, agent_name: str, config: DispatchConfig):
        """
        Get or create an adapter instance for an agent.

        Adapter instances are cached by (agent_name, dry_run) so that
        connection pools and auth state are reused across passes.
        """
        cache_key = (agent_name, config.dry_run)
        if cache_key in self._adapter_cache:
            return self._adapter_cache[cache_key]

        AdapterClass = AGENT_REGISTRY[agent_name]

        # Build adapter kwargs based on agent and config
        kwargs = {"dry_run": config.dry_run}

        # Apply per-agent overrides
        overrides = config.agent_overrides.get(agent_name, {})

        # MCP server URL — only OpenAI supports this natively
        if agent_name == "openai" and config.mcp_server_url:
            kwargs["mcp_server_url"] = config.mcp_server_url

        # Gemini context caching
        if agent_name == "gemini":
            kwargs["use_context_cache"] = overrides.get(
                "use_context_cache", False
            )

        # Qwen agent configuration
        if agent_name == "qwen":
            kwargs["max_tool_rounds"] = overrides.get("max_tool_rounds", 100)

        adapter = AdapterClass(**kwargs)
        self._adapter_cache[cache_key] = adapter
        return adapter


# ─── Convenience function ────────────────────────────────────────────────

async def dispatch_task(
    package: PromptPackage,
    agents: Optional[list[str]] = None,
    passes: int = 1,
    dry_run: bool = False,
    mcp_server_url: Optional[str] = None,
) -> DispatchResult:
    """
    One-shot convenience function for dispatching a task.

    Usage:
        result = await dispatch_task(package, agents=["claude", "openai"])
        result = await dispatch_task(package, passes=3, dry_run=True)
    """
    config = DispatchConfig(
        agents=agents or ["claude", "openai", "gemini", "perplexity"],
        passes_per_agent=passes,
        dry_run=dry_run,
        mcp_server_url=mcp_server_url,
    )
    dispatcher = TaskDispatcher()
    return await dispatcher.dispatch(package, config)


# ─── Result serialization ────────────────────────────────────────────────

def dispatch_result_to_dict(result: DispatchResult) -> dict:
    """
    Serialize a DispatchResult to a JSON-compatible dict.

    This is the format saved to disk and consumed by the scorer.
    """
    return {
        "task_id": result.task_id,
        "dispatched_at": result.dispatched_at,
        "completed_at": result.completed_at,
        "total_cost_usd": result.total_cost_usd,
        "total_duration_sec": result.total_duration_sec,
        "agents_attempted": result.agents_attempted,
        "agents_succeeded": result.agents_succeeded,
        "agents_failed": result.agents_failed,
        "agent_errors": result.agent_errors,
        "config": {
            "agents": result.config.agents,
            "passes_per_agent": result.config.passes_per_agent,
            "dry_run": result.config.dry_run,
            "mcp_server_url": result.config.mcp_server_url,
            "output_files_base_dir": result.config.output_files_base_dir,  # ← from Spark2
        },
        "package": {
            "task_id": result.package.task_id,
            "prompt": result.package.prompt,
            "research_type": result.package.research_type,
            "iat_type": result.package.iat_type,
            "domain": result.package.domain,
            "decision_archetype": result.package.decision_archetype,
            "file_count": len(result.package.file_paths),
            "file_names": [
                os.path.basename(f) for f in result.package.file_paths
            ],
            "output_formats": result.package.output_formats,
        },
        "agent_results": {
            agent_name: [
                {
                    "task_id": r.task_id,
                    "agent": r.agent,
                    "model": r.model,
                    "completed": r.completed,
                    "forced_stop": r.forced_stop,
                    "error": r.error,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "total_cost_usd": r.total_cost_usd,
                    "total_duration_sec": r.total_duration_sec,
                    "iterations": r.iterations,
                    "citations_count": len(r.citations),
                    "tool_calls_count": len(r.tool_call_log),
                    "response_length": len(r.response_text),
                    "response_text": r.response_text,
                    "citations": r.citations,
                    "output_files": r.output_files or [],
                    "output_file_errors": r.output_file_errors or {},
                    "started_at": r.started_at,
                    "completed_at": r.completed_at,
                }
                for r in results
            ]
            for agent_name, results in result.agent_results.items()
        },
    }


def save_dispatch_result(result: DispatchResult, output_path: str):
    """Save a DispatchResult to a JSON file."""
    data = dispatch_result_to_dict(result)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Saved dispatch result to %s", output_path)


# ─── CLI entry point ─────────────────────────────────────────────────────

def _print_dispatch_summary(result: DispatchResult):
    """Pretty-print a dispatch result to terminal."""
    print(f"\n{'═' * 70}")
    print(f"  DISPATCH RESULT — {result.task_id}")
    print(f"{'═' * 70}")
    print(f"  Research Type:   {result.package.research_type or '—'}")
    print(f"  IAT Type:        {result.package.iat_type or '—'}")
    print(f"  Domain:          {result.package.domain or '—'}")
    print(f"  Files:           {len(result.package.file_paths)}")
    print(f"  Output Formats:  {result.package.output_formats or 'none'}")  # ← from Spark2
    print(f"  Passes/Agent:    {result.config.passes_per_agent}")
    print(f"  Duration:        {result.total_duration_sec:.1f}s")
    print(f"  Total Cost:      ${result.total_cost_usd:.4f}")
    print(f"{'─' * 70}")

    for agent_name in result.agents_attempted:
        results = result.agent_results.get(agent_name, [])
        status = "✅" if agent_name in result.agents_succeeded else "❌"
        error = result.agent_errors.get(agent_name, "")

        print(f"\n  {status} {agent_name.upper()}")
        if error:
            print(f"     Error: {error[:80]}")

        for i, r in enumerate(results, 1):
            flag = "✓" if r.completed and not r.error else "✗"
            files_info = f", {len(r.output_files)} file(s)" if r.output_files else ""  # ← from Spark2
            print(
                f"     Pass {i}: [{flag}] "
                f"{len(r.response_text):,} chars, "
                f"{len(r.citations)} cites, "
                f"${r.total_cost_usd:.4f}, "
                f"{r.total_duration_sec:.1f}s"
                f"{files_info}"
            )

    print(f"\n{'═' * 70}")
    print(
        f"  Summary: {len(result.agents_succeeded)}/{len(result.agents_attempted)} "
        f"agents succeeded"
    )
    print(f"{'═' * 70}\n")


async def cli_main():
    """CLI entry point for testing the dispatcher."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Dispatch a research task to all agents.",
    )
    parser.add_argument("--prompt", "-p", required=True)
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--files", "-f", nargs="*", default=[])
    parser.add_argument("--research-type", default="",
                        choices=[
                            "CRP", "RCP", "SCP", "LDP", "FSP",
                            "Constrained Research Prompt",
                            "Relevance Compression Prompt",
                            "Structural Compliance Prompt",
                            "Latent Decomposition Prompt",
                            "Failure-Sensitive Prompt", "",
                        ])
    parser.add_argument("--iat-type", default="",
                        choices=["IAT-1", "IAT-2", "IAT-3", ""])
    parser.add_argument("--domain", default="")   # free text — no restricted choices
    parser.add_argument("--agents", nargs="*",
                        default=["claude", "openai", "gemini", "perplexity"])
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--live", action="store_true",
                        help="Disable dry-run (make real API calls)")
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--output-files-dir", default=None,         # ← from Spark2
                        help="Base dir for generated output files (xlsx/docx/pptx)")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Build package — auto-detect output formats from prompt  ← from Spark2
    import uuid
    from file_generators import detect_output_formats

    package = PromptPackage(
        task_id=args.task_id or f"dispatch-{uuid.uuid4().hex[:8]}",
        prompt=args.prompt,
        file_paths=args.files,
        research_type=args.research_type,
        iat_type=args.iat_type,
        domain=args.domain,
        output_formats=detect_output_formats(args.prompt),   # ← from Spark2
    )

    # Build config
    config = DispatchConfig(
        agents=args.agents,
        passes_per_agent=args.passes,
        dry_run=not args.live,
        output_files_base_dir=args.output_files_dir,         # ← from Spark2
    )

    # Dispatch
    dispatcher = TaskDispatcher()
    result = await dispatcher.dispatch(package, config)

    # Display
    _print_dispatch_summary(result)

    # Save
    if args.output:
        save_dispatch_result(result, args.output)
        print(f"  📄 Saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(cli_main())