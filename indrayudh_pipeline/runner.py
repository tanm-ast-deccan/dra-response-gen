"""
runner.py — The single unified run loop for one Task on one provider.

Same loop for every provider (they all go through OpenRouter):

    build_messages → [agentic turn loop with optional tools] → optional file gen

Every behaviour is driven by GenParams: turns, web search, tools, file
creation, sampling, timeouts, budget. There is NO provider-specific branching
beyond the model slug and the (registry-declared) tool capability.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

try:
    from . import tools as tools_mod
    from . import file_gen
    from .provider import supports_tools
    from .models import RunResult, ToolCall
except ImportError:  # run as a script from inside the package dir
    import tools as tools_mod
    import file_gen
    from provider import supports_tools
    from models import RunResult, ToolCall

logger = logging.getLogger("indrayudh.runner")

SYSTEM_PROMPT = (
    "You are a deep research analyst. Produce a comprehensive, well-structured, "
    "well-cited report based on the user's question and any provided files. "
    "Cite non-trivial claims with numbered references and end with a brief "
    "'Limitations' section. Treat provided files as primary sources. Do not "
    "fabricate sources."
)


# ─── Message construction ─────────────────────────────────────────────────────

def build_messages(task, params) -> list[dict]:
    """Build the initial [system, user] messages, inlining file text."""
    parts = [task.prompt]

    if task.file_paths:
        parts.append("\n" + "=" * 60)
        parts.append("PROVIDED REFERENCE DOCUMENTS")
        parts.append("=" * 60)
        budget = 90_000 * 4  # ~90K tokens worth of chars, split across files
        per_file = budget // max(len(task.file_paths), 1)
        for fpath in task.file_paths:
            import os
            parts.append(f"\n{'-' * 40}\nFILE: {os.path.basename(fpath)}\n{'-' * 40}")
            parts.append(file_gen.extract_file_text(fpath, max_chars=per_file))
        parts.append("\n" + "=" * 60 + "\nEND OF REFERENCE DOCUMENTS\n" + "=" * 60)

    if params.file_output and task.output_formats:
        parts.append(file_gen.file_gen_instructions(task.output_formats[0], task.file_paths))

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(parts)},
    ]


# ─── Run one task ─────────────────────────────────────────────────────────────

async def run_task(task, params, driver) -> RunResult:
    started = datetime.now(timezone.utc)
    logger.info("[%s] start (model=%s, dry_run=%s)",
                task.run_id, task.model_slug, driver.dry_run)

    if driver.dry_run:
        return _dry_run_result(task, params, started)

    messages = build_messages(task, params)

    # Tools: only if enabled AND the model supports them.
    schemas = None
    if params.enabled_tools and supports_tools(task.provider):
        schemas = tools_mod.schemas_for(params.enabled_tools)
    elif params.enabled_tools:
        logger.warning("[%s] tools enabled but provider '%s' lacks tool support — skipping",
                       task.run_id, task.provider)

    in_tok = out_tok = 0
    cost = 0.0
    turns = 0
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
            resp = await driver.chat(task.model_slug, messages, params, schemas)

            in_tok += resp.input_tokens
            out_tok += resp.output_tokens
            cost += resp.cost_usd
            _add_citations(resp.citations)
            final_text = resp.text or final_text

            # Budget guard
            if cost > params.max_cost_usd:
                logger.warning("[%s] budget $%.2f exceeded ($%.2f) — stopping",
                               task.run_id, params.max_cost_usd, cost)
                forced_stop = True
                break

            # No tool calls → this is the final answer.
            if not resp.tool_calls or not schemas:
                break

            # Execute tool calls locally and feed results back, then loop.
            messages.append(resp.assistant_message)
            for tc in resp.tool_calls:
                result = tools_mod.execute(tc["name"], tc["arguments"])
                tool_log.append(ToolCall(
                    turn=turn, name=tc["name"], arguments=tc["arguments"],
                    result_preview=result[:500],
                ))
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": tc["name"],
                    "content": result,
                })
        else:
            # Loop exhausted without a tool-free final answer.
            forced_stop = True
            logger.warning("[%s] max_turns (%d) reached", task.run_id, params.max_turns)

    except Exception as e:  # noqa: BLE001 — never let one run kill the batch
        logger.exception("[%s] run error", task.run_id)
        return _finish(task, started, response_text=final_text, citations=citations,
                       input_tokens=in_tok, output_tokens=out_tok, cost=cost,
                       turns=turns, tool_log=tool_log, completed=False,
                       forced_stop=True, error=str(e))

    # ── File generation ───────────────────────────────────────────────
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
        task, started, response_text=final_text, citations=citations,
        input_tokens=in_tok, output_tokens=out_tok, cost=cost, turns=turns,
        tool_log=tool_log, completed=completed, forced_stop=forced_stop,
        error=error, output_files=output_files, output_file_errors=output_file_errors,
    )


# ─── Result builders ──────────────────────────────────────────────────────────

def _finish(task, started, *, response_text, citations, input_tokens, output_tokens,
            cost, turns, tool_log, completed, forced_stop, error,
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
        tool_calls=tool_log,
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
    files = ", ".join(__import__("os").path.basename(f) for f in task.file_paths) or "none"
    report = (
        f"# [DRY RUN] {task.provider} via OpenRouter\n\n"
        f"**Task:** {task.task_id}  \n**Model:** {task.model_slug}  \n"
        f"**Pass:** {task.pass_index}  \n**Web search:** {params.web_search}  \n"
        f"**Tools:** {params.enabled_tools or 'none'}  \n"
        f"**Output formats:** {fmts}  \n**Files:** {files}  \n"
        f"**max_turns:** {params.max_turns}  \n\n"
        f"## Prompt\n\n{task.prompt[:500]}{'...' if len(task.prompt) > 500 else ''}\n"
    )
    return _finish(
        task, started, response_text=report, citations=[],
        input_tokens=0, output_tokens=0, cost=0.0, turns=0, tool_log=[],
        completed=True, forced_stop=False, error=None,
        output_files=[], output_file_errors=({task.output_formats[0]: "DRY_RUN"}
                                             if task.output_formats else {}),
    )
