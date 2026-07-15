"""
runner.py — The single unified run loop for one Task on one provider.

Architecture:
    1. Start MCPToolClient (which launches exec_server.py via stdio)
    2. Discover tools from the MCP server
    3. Run agentic loop: model calls → tool routing via MCP → results back
    4. Capture full trajectory (every turn, every tool call, every result)

Files are NOT inlined in the prompt. The model discovers and reads
them via MCP tools (python_execute, bash_execute, etc.).
"""

from __future__ import annotations

import os
import logging
from datetime import datetime, timezone

try:
    from . import file_gen
    from .provider import supports_tools
    from .models import RunResult, ToolCall, TurnRecord
    from .mcp_client import MCPToolClient
except ImportError:
    import file_gen
    from provider import supports_tools
    from models import RunResult, ToolCall, TurnRecord
    from mcp_client import MCPToolClient

logger = logging.getLogger("dra.runner")

SYSTEM_PROMPT = (
    "You are a deep research analyst. Produce a comprehensive, well-structured, "
    "well-cited report based on the user's question and any provided files. "
    "Cite non-trivial claims with numbered references. "
    "Do not fabricate sources.\n\n"
    "You have access to tools for reading files, executing code, running shell commands, "
    "searching the web, and writing output files. Use them as needed. "
)


# ─── Message construction ─────────────────────────────────────────────────────

def build_messages(task, params) -> list[dict]:
    """Build initial messages. Model discovers files via tools."""
    parts = [task.prompt]

    if task.file_paths:
        parts.append(
            "\n\nReference files have been provided in your working directory. "
            "Use your tools to list, read, and analyze them."
        )

    if params.file_output and task.output_formats:
        parts.append(file_gen.file_gen_instructions(
            task.output_formats[0], task.file_paths
        ))

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(parts)},
    ]

# def build_messages(task, params) -> list[dict]:
#     """
#     Build the initial [system, user] messages.

#     Files are NOT inlined. The model discovers and reads them via tools.
#     """
#     parts = [task.prompt]

#     if task.file_paths:
#         parts.append("\n" + "=" * 60)
#         parts.append("AVAILABLE REFERENCE FILES (use tools to read them)")
#         parts.append("=" * 60)
#         for fpath in task.file_paths:
#             basename = os.path.basename(fpath)
#             size = os.path.getsize(fpath) if os.path.exists(fpath) else 0
#             parts.append(f"  - {basename}  ({size:,} bytes)")
#         parts.append(
#             "\nUse python_execute or bash_execute to read and analyze these files. "
#             "They are in the current working directory."
#         )
#         parts.append("=" * 60)

#     if params.file_output and task.output_formats:
#         parts.append(file_gen.file_gen_instructions(task.output_formats[0], task.file_paths))

#     return [
#         {"role": "system", "content": SYSTEM_PROMPT},
#         {"role": "user", "content": "\n".join(parts)},
#     ]


# ─── Stage input files ───────────────────────────────────────────────────────

def _stage_input_files(file_paths: list, run_dir: str):
    """Symlink input files into the run directory. Fast, no copy."""
    os.makedirs(run_dir, exist_ok=True)
    for src in file_paths:
        if not os.path.exists(src):
            continue
        dst = os.path.join(run_dir, os.path.basename(src))
        if os.path.exists(dst):
            continue
        try:
            os.symlink(os.path.abspath(src), dst)
        except OSError:
            # Windows or filesystem doesn't support symlinks — fall back to copy
            import shutil
            shutil.copy2(src, dst)


# ─── Run one task ─────────────────────────────────────────────────────────────

async def run_task(task, params, driver) -> RunResult:
    started = datetime.now(timezone.utc)
    logger.info(
        "[%s] start (model=%s, dry_run=%s)",
        task.run_id, task.model_slug, driver.dry_run,
    )

    if driver.dry_run:
        return _dry_run_result(task, params, started)

    # ── Set up staging directory ──────────────────────────────────
    staging = task.output_dir or os.path.join(os.getcwd(), "staging", task.task_id)
    os.makedirs(staging, exist_ok=True)
    _stage_input_files(task.file_paths, staging)

    messages = build_messages(task, params)

    # ── Start MCP client ──────────────────────────────────────────
    mcp: MCPToolClient | None = None
    schemas = None

    if supports_tools(task.provider):
        try:
            mcp = MCPToolClient(staging_dir=staging)
            await mcp.start()
            schemas = mcp.openai_schemas
        except Exception as e:
            logger.warning(
                "[%s] MCP failed to start: %s — running without tools",
                task.run_id, e,
            )
            mcp = None
            schemas = None

    # ── Agentic loop ──────────────────────────────────────────────
    in_tok = out_tok = 0
    cost = 0.0
    turns = 0
    trajectory: list[TurnRecord] = []
    tool_log: list[ToolCall] = []
    citations: list = []
    seen_urls: set = set()
    final_text = ""
    forced_stop = False
    error = None

    def _add_citations(cits):
        for c in cits or []:
            url = c.get("url", "")
            if url and url not in seen_urls:
                citations.append(c)
                seen_urls.add(url)

    try:
        for turn in range(1, params.max_turns + 1):
            turns = turn
            turn_ts = datetime.now(timezone.utc).isoformat()

            resp = await driver.chat(task.model_slug, messages, params, schemas)

            in_tok += resp.input_tokens
            out_tok += resp.output_tokens
            cost += resp.cost_usd
            _add_citations(resp.citations)
            final_text = resp.text or final_text

            # ── Build turn record ─────────────────────────────────
            record = TurnRecord(
                turn=turn,
                timestamp=turn_ts,
                assistant_text=resp.text or "",
                finish_reason=resp.finish_reason,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                cost_usd=resp.cost_usd,
            )

            # Budget guard
            if cost > params.max_cost_usd:
                logger.warning(
                    "[%s] budget $%.2f exceeded ($%.2f) — stopping",
                    task.run_id, params.max_cost_usd, cost,
                )
                forced_stop = True
                trajectory.append(record)
                break

            # No tool calls → final answer
            if not resp.tool_calls or not schemas:
                trajectory.append(record)
                break

            # ── Execute tool calls via MCP ────────────────────────
            messages.append(resp.assistant_message)

            for tc in resp.tool_calls:
                tc_name = tc["name"]
                tc_args = tc["arguments"]
                tc_id = tc["id"]

                # Record the call
                record.tool_calls.append({
                    "name": tc_name,
                    "arguments": tc_args,
                    "id": tc_id,
                })

                # Route through MCP client
                if mcp and mcp.is_ready:
                    result = await mcp.call_tool(tc_name, tc_args)
                else:
                    result = f"[error] MCP not available for: {tc_name}"

                # Record the result
                record.tool_results.append({
                    "name": tc_name,
                    "result": result[:10000],
                    "server": "exec",
                })

                # Legacy flat log
                tool_log.append(ToolCall(
                    turn=turn,
                    name=tc_name,
                    arguments=tc_args,
                    result_preview=result[:500],
                ))

                # Feed back to model
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": tc_name,
                    "content": result,
                })

            trajectory.append(record)

        else:
            forced_stop = True
            logger.warning("[%s] max_turns (%d) reached", task.run_id, params.max_turns)

    except Exception as e:
        logger.exception("[%s] run error", task.run_id)
        error = str(e)
        forced_stop = True

    finally:
        if mcp:
            await mcp.stop()

    # ── File generation ───────────────────────────────────────────
    output_files: list = []
    output_file_errors: dict = {}
    if params.file_output and task.output_formats and final_text and not forced_stop:
        output_files, output_file_errors, fix_cost = await file_gen.generate(
            final_text, task, params, driver, task.model_slug
        )
        cost += fix_cost

    completed = bool(final_text) and not output_file_errors and not forced_stop
    if output_file_errors and not error:
        error = f"File generation failed: {list(output_file_errors.keys())}"

    return _finish(
        task, started,
        response_text=final_text,
        citations=citations,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost=cost,
        turns=turns,
        tool_log=tool_log,
        trajectory=trajectory,
        completed=completed,
        forced_stop=forced_stop,
        error=error,
        output_files=output_files,
        output_file_errors=output_file_errors,
    )


# ─── Result builders ──────────────────────────────────────────────────────────

def _finish(task, started, *, response_text, citations, input_tokens, output_tokens,
            cost, turns, tool_log, trajectory, completed, forced_stop, error,
            output_files=None, output_file_errors=None) -> RunResult:
    completed_at = datetime.now(timezone.utc)
    return RunResult(
        task_id=task.task_id,
        run_id=task.run_id,
        provider=task.provider,
        model=task.model_slug,
        pass_index=task.pass_index,
        response_text=response_text,
        citations=citations,
        output_files=output_files or [],
        output_file_errors=output_file_errors or {},
        tool_calls=[tc.to_dict() for tc in tool_log],
        trajectory=[t.to_dict() for t in trajectory],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_cost_usd=round(cost, 6),
        turns=turns,
        completed=completed,
        forced_stop=forced_stop,
        error=error,
        total_duration_sec=(completed_at - started).total_seconds(),
        started_at=started.isoformat(),
        completed_at=completed_at.isoformat(),
    )


def _dry_run_result(task, params, started) -> RunResult:
    fmts = ", ".join(task.output_formats) if task.output_formats else "none"
    files = ", ".join(os.path.basename(f) for f in task.file_paths) or "none"
    report = (
        f"# [DRY RUN] {task.provider} via OpenRouter\n\n"
        f"**Task:** {task.task_id}  \n**Model:** {task.model_slug}  \n"
        f"**Pass:** {task.pass_index}  \n**Web search:** {params.web_search}  \n"
        f"**Tools:** MCP (exec_server)  \n"
        f"**Output formats:** {fmts}  \n**Files:** {files}  \n"
        f"**max_turns:** {params.max_turns}  \n\n"
        f"## Prompt\n\n{task.prompt[:500]}{'...' if len(task.prompt) > 500 else ''}\n"
    )
    return _finish(
        task, started,
        response_text=report,
        citations=[],
        input_tokens=0,
        output_tokens=0,
        cost=0.0,
        turns=0,
        tool_log=[],
        trajectory=[],
        completed=True,
        forced_stop=False,
        error=None,
        output_files=[],
        output_file_errors=({task.output_formats[0]: "DRY_RUN"}
                           if task.output_formats else {}),
    )