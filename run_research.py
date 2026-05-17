#!/usr/bin/env python3
"""
run_research.py — CLI entry point for running deep research tasks.

Usage:
    # Run a single research task with Claude
    python run_research.py --agent claude --prompt "Research the impact of AI on employment"

    # Dry run (no API calls, simulates the loop)
    python run_research.py --agent claude --prompt "..." --dry-run

    # With files and custom budget
    python run_research.py --agent claude \
        --prompt "Analyze these financial reports" \
        --files report_q1.pdf report_q2.pdf \
        --max-cost 10.0 \
        --max-iterations 15

    # Save results to JSON
    python run_research.py --agent claude --prompt "..." --output results.json
"""

import argparse
import asyncio
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from dataclasses import asdict

from env_loader import load_env
load_env()

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import ResearchTask, AgentResult
from adapters.claude_adapter import ClaudeAdapter


def build_task(args) -> ResearchTask:
    """Build a ResearchTask from CLI arguments."""
    return ResearchTask(
        task_id=args.task_id or f"eval-{uuid.uuid4().hex[:8]}",
        prompt=args.prompt,
        file_paths=args.files or [],
        web_search_enabled=not args.no_web_search,
        max_iterations=args.max_iterations,
        max_cost_usd=args.max_cost,
        timeout_seconds=args.timeout,
    )


async def run_live(task: ResearchTask, args) -> AgentResult:
    """Run the task against the real Claude API."""
    
    # Check for API key
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n❌ No API key found.")
        print("   Set ANTHROPIC_API_KEY env var or pass --api-key")
        print("   Use --dry-run to test without an API key.")
        sys.exit(1)
    
    model = args.model or "claude-sonnet-4-5-20250929"  # default to Sonnet for cost
    adapter = ClaudeAdapter(api_key=api_key, model=model)
    
    print(f"\n{'='*60}")
    print(f"  Deep Research Agent — Claude ({model})")
    print(f"{'='*60}")
    print(f"  Task ID:        {task.task_id}")
    print(f"  Max iterations: {task.max_iterations}")
    print(f"  Max cost:       ${task.max_cost_usd}")
    print(f"  Timeout:        {task.timeout_seconds}s")
    print(f"  Web search:     {'enabled' if task.web_search_enabled else 'disabled'}")
    print(f"  Files:          {len(task.file_paths)}")
    print(f"{'='*60}")
    print(f"  Prompt: {task.prompt[:100]}{'...' if len(task.prompt) > 100 else ''}")
    print(f"{'='*60}\n")
    
    result = await adapter.run(task)
    return result


async def run_dry(task: ResearchTask) -> AgentResult:
    """
    Dry run — simulates the agent loop without API calls.
    
    Useful for:
      - Testing the pipeline end-to-end without spending money
      - Validating that the runner, models, and output format work
      - Demonstrating the loop structure to teammates
    
    Simulates 3 iterations: search → fetch → final report.
    """
    from models import ToolCall
    
    print(f"\n{'='*60}")
    print(f"  Deep Research Agent — DRY RUN (no API calls)")
    print(f"{'='*60}")
    print(f"  Task ID:  {task.task_id}")
    print(f"  Prompt:   {task.prompt[:80]}...")
    print(f"{'='*60}\n")
    
    started_at = datetime.now(timezone.utc)
    
    # Simulate iteration 1: Claude decides to search
    print("  [ITER 1] Claude → web_search(\"AI employment impact 2025\")")
    print("  [TOOL]   web_search: found 8 results")
    await asyncio.sleep(0.3)  # simulate latency
    
    # Simulate iteration 2: Claude reads a specific page
    print("  [ITER 2] Claude → web_fetch(\"https://example.com/report\")")
    print("  [TOOL]   web_fetch: retrieved 4,200 tokens")
    await asyncio.sleep(0.3)
    
    # Simulate iteration 3: Claude does another search
    print("  [ITER 3] Claude → web_search(\"AI job displacement skeptics criticism\")")
    print("  [TOOL]   web_search: found 6 results")
    await asyncio.sleep(0.3)
    
    # Simulate final turn: Claude produces the report
    print("  [ITER 4] Claude → end_turn (producing report)")
    await asyncio.sleep(0.2)
    
    completed_at = datetime.now(timezone.utc)
    
    simulated_report = f"""# Research Report: {task.prompt[:60]}

## Executive Summary

This is a **simulated dry-run report**. In a live run, Claude would have:
1. Searched the web for relevant sources
2. Read full articles from authoritative sources
3. Synthesized findings into a comprehensive analysis

## Simulated Findings

The dry run simulated 3 tool calls across 4 iterations:
- 2 web searches (broad + targeted)
- 1 page fetch (deep reading)

## References

[1] Simulated Source — https://example.com/report

## Limitations

This is a dry run. No actual research was performed.
"""
    
    return AgentResult(
        task_id=task.task_id,
        agent="claude",
        model="dry-run",
        response_text=simulated_report,
        citations=[{"index": 1, "title": "Simulated Source", "url": "https://example.com/report"}],
        tool_call_log=[
            ToolCall(iteration=1, tool_name="web_search",
                     tool_input={"query": "AI employment impact 2025"},
                     result_preview="8 results found", result_tokens=500,
                     timestamp=started_at.isoformat(), duration_ms=200),
            ToolCall(iteration=2, tool_name="web_fetch",
                     tool_input={"url": "https://example.com/report"},
                     result_preview="Full article content...", result_tokens=4200,
                     timestamp=started_at.isoformat(), duration_ms=800),
            ToolCall(iteration=3, tool_name="web_search",
                     tool_input={"query": "AI job displacement skeptics criticism"},
                     result_preview="6 results found", result_tokens=400,
                     timestamp=started_at.isoformat(), duration_ms=180),
        ],
        input_tokens=12500,
        output_tokens=3200,
        total_cost_usd=0.0,  # dry run = free
        total_duration_sec=(completed_at - started_at).total_seconds(),
        iterations=4,
        completed=True,
        forced_stop=False,
        started_at=started_at.isoformat(),
        completed_at=completed_at.isoformat(),
    )


def display_result(result: AgentResult):
    """Pretty-print the result to terminal."""
    
    print(f"\n{'='*60}")
    print(f"  RESULT SUMMARY")
    print(f"{'='*60}")
    print(f"  Agent:      {result.agent} ({result.model})")
    print(f"  Task ID:     {result.task_id}")
    print(f"  Status:      {'✅ Completed' if result.completed else '❌ Failed'}")
    if result.forced_stop:
        print(f"  ⚠️  Forced stop (budget/iteration/timeout limit reached)")
    if result.error:
        print(f"  Error:       {result.error}")
    print(f"  Iterations:  {result.iterations}")
    print(f"  Tool calls:  {len(result.tool_call_log)}")
    print(f"  Citations:   {len(result.citations)}")
    print(f"  Tokens:      {result.input_tokens:,} in / {result.output_tokens:,} out")
    print(f"  Cost:        ${result.total_cost_usd:.4f}")
    print(f"  Duration:    {result.total_duration_sec:.1f}s")
    print(f"{'='*60}")
    
    # Tool call breakdown
    if result.tool_call_log:
        print(f"\n  Tool Call Log:")
        for tc in result.tool_call_log:
            input_preview = json.dumps(tc.tool_input, ensure_ascii=False)[:60]
            print(f"    [{tc.iteration}] {tc.tool_name}({input_preview})")
    
    # Report preview
    print(f"\n  Report Preview (first 500 chars):")
    print(f"  {'-'*56}")
    preview = result.response_text[:500]
    for line in preview.split("\n"):
        print(f"  {line}")
    if len(result.response_text) > 500:
        print(f"  ... ({len(result.response_text) - 500} more characters)")
    print(f"  {'-'*56}")


def save_result(result: AgentResult, output_path: str):
    """Save result as JSON for downstream processing."""
    result_dict = asdict(result)
    with open(output_path, "w") as f:
        json.dump(result_dict, f, indent=2, default=str)
    print(f"\n  📄 Result saved to: {output_path}")


async def main():
    parser = argparse.ArgumentParser(
        description="Run a deep research task using Claude's agent loop.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (no API key needed)
  python run_research.py --dry-run --prompt "Impact of AI on employment"

  # Live run with Sonnet (cheaper, good for testing)
  python run_research.py --prompt "Impact of AI on employment"

  # Live run with Opus (max quality, for evaluation)
  python run_research.py --prompt "..." --model claude-opus-4-6 --max-cost 15.0
        """,
    )
    
    parser.add_argument("--prompt", "-p", required=True,
                        help="Research prompt/question")
    parser.add_argument("--agent", default="claude",
                        choices=["claude"],
                        help="Agent to use (currently only claude)")
    parser.add_argument("--model", "-m", default=None,
                        help="Model string (default: claude-sonnet-4-5-20250929)")
    parser.add_argument("--files", "-f", nargs="*", default=[],
                        help="File paths to include as context")
    parser.add_argument("--max-iterations", type=int, default=25,
                        help="Max agent loop iterations (default: 25)")
    parser.add_argument("--max-cost", type=float, default=15.0,
                        help="Max cost in USD (default: 15.0)")
    parser.add_argument("--timeout", type=int, default=900,
                        help="Timeout in seconds (default: 900)")
    parser.add_argument("--no-web-search", action="store_true",
                        help="Disable web search (closed-corpus only)")
    parser.add_argument("--output", "-o", default=None,
                        help="Save result JSON to this path")
    parser.add_argument("--task-id", default=None,
                        help="Custom task ID (default: auto-generated)")
    parser.add_argument("--api-key", default=None,
                        help="Anthropic API key (or set ANTHROPIC_API_KEY)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate the loop without API calls")
    
    args = parser.parse_args()
    task = build_task(args)
    
    if args.dry_run:
        result = await run_dry(task)
    else:
        result = await run_live(task, args)
    
    display_result(result)
    
    if args.output:
        save_result(result, args.output)


if __name__ == "__main__":
    asyncio.run(main())
