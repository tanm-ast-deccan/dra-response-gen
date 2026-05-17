"""
claude_adapter.py — Transparent agentic research loop for Claude.

Architecture
────────────
  messages = [initial prompt + files]

  while True:
      response = call_claude(messages, tools)

      end_turn   → extract text + capture output files → done
      pause_turn → server loop hit iteration cap; append response, continue
      max_tokens → if server tools unresolved: force stop; else ask to continue
      tool_use   → all tools are server-side; append response, continue

Tool setup (per Anthropic docs)
────────────────────────────────
  web_search_20250305     — server-side, results inline in same response
  code_execution_20250825 — server-side, bash + file ops in Anthropic sandbox
    Pre-installed: openpyxl, python-docx, python-pptx, pandas, matplotlib, etc.
    Output files retrievable via client.beta.files.download(file_id)

Beta usage
──────────
  files-api-2025-04-14 — only when uploading input files or downloading
                         output files via the Files API.
  No other betas.

File capture (per Anthropic docs, code-execution-tool page)
────────────────────────────────────────────────────────────
  for block in response.content:
      if block.type == "bash_code_execution_tool_result":
          if block.content.type == "bash_code_execution_result":
              for output in block.content.content:
                  file_id = output.file_id   # download this
"""

from __future__ import annotations

import os
import re
import json
import asyncio
from datetime import datetime, timezone
from typing import Optional

from models import ResearchTask, AgentResult, ToolCall


def _xlsb_to_text(fpath: str) -> str:
    """
    Convert an Excel Binary Workbook (.xlsb) to labelled TSV text.
    Used before uploading to the Files API since the Anthropic sandbox
    and OpenAI containers do not have pyxlsb installed.
    Install: pip install pyxlsb
    """
    try:
        import pyxlsb
    except ImportError:
        raise ImportError("pip install pyxlsb  (needed for XLSB conversion)")
    sections = []
    with pyxlsb.open_workbook(fpath) as wb:
        for sheet_name in wb.sheets:
            with wb.get_sheet(sheet_name) as ws:
                rows = []
                for row in ws.rows():
                    vals = [str(r.v) if r.v is not None else "" for r in row]
                    if any(v.strip() for v in vals):
                        rows.append("\t".join(vals))
            if rows:
                sections.append(
                    f"=== Sheet: {sheet_name} ({len(rows)} rows) ===\n"
                    + "\n".join(rows)
                )
    return "\n\n".join(sections) if sections else "[Empty workbook]"



# ── Pricing ───────────────────────────────────────────────────────────────────
# Source: https://platform.claude.com/docs/en/about-claude/pricing

PRICING = {
    "claude-opus-4-6":   {"input_per_mtok": 5.00,  "output_per_mtok": 25.00},
    "claude-sonnet-4-6": {"input_per_mtok": 3.00,  "output_per_mtok": 15.00},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p = PRICING.get(model, PRICING["claude-opus-4-6"])
    return round(
        (input_tokens  / 1_000_000) * p["input_per_mtok"] +
        (output_tokens / 1_000_000) * p["output_per_mtok"],
        6,
    )


# ── Beta headers ──────────────────────────────────────────────────────────────
FILES_API_BETA = "files-api-2025-04-14"   # only beta we use


# ── System prompts ────────────────────────────────────────────────────────────

RESEARCH_SYSTEM_PROMPT = """\
You are a deep research agent. Your task is to produce a comprehensive, \
well-cited research report based on the user's prompt and any provided files.

## Research methodology

1. PLAN: Before searching, outline 3-5 key questions that need answering.
2. SEARCH: Use web_search to find relevant, authoritative sources. Prefer \
primary sources (official reports, peer-reviewed papers, government data) \
over aggregators.
3. READ: Use web_fetch to read full articles when search snippets are insufficient.
4. SYNTHESIZE: Cross-reference findings across multiple sources. Note contradictions.
5. ITERATE: After initial synthesis, identify gaps and search again to fill them.

## Citation format

Cite every non-trivial claim using numbered references: [1], [2], etc.
At the end of your report include a "References" section:
  [1] Title — URL

## Output format

Structure your report with Markdown headers.
Include a brief executive summary at the top.
End with a "Limitations" section noting gaps or uncertainty.

## Rules

- Never fabricate sources. If you cannot find evidence, say so explicitly.
- Distinguish facts from sources vs your own analysis.
- Treat provided files as primary sources; cite them as [File: filename].
- Depth over breadth: 5 well-analyzed sources beat 20 shallow ones.
"""

FILE_GEN_ADDENDUM = """\

## File generation requirement

The user requires a file deliverable. After completing your research you MUST \
use the code_execution tool to generate it.

The sandbox has pre-installed: openpyxl, python-docx, python-pptx, pandas, \
matplotlib, reportlab, pillow, and more.

Instructions:
- Write Python code to generate the file using the appropriate library.
- Base all content on your research — no placeholder text.
- Save with a descriptive filename matching the required format.
- The framework captures the file automatically.

Do not skip this step. A report without the required file is incomplete.
"""


def build_system_prompt(task) -> str:
    prompt = RESEARCH_SYSTEM_PROMPT
    if task.output_formats:
        prompt += FILE_GEN_ADDENDUM
        prompt += f"\nRequired format(s): {', '.join(task.output_formats)}\n"
    return prompt


# ── File extension routing ────────────────────────────────────────────────────
# document blocks → Claude reads natively (PDF parsed server-side, text inline)
# image blocks    → Claude sees image
# container_upload → lands at /input/<filename> in code execution sandbox
_DOCUMENT_EXTS = {".pdf", ".txt", ".md", ".html", ".xml"}
_IMAGE_EXTS    = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


# ── Adapter ───────────────────────────────────────────────────────────────────

class ClaudeAdapter:
    """
    Agentic research loop using Claude's Messages API.

    Tool choice:
      - web_search_20250305    (server-side, always on for non-IAT-1 tasks)
      - code_execution_20250825 (server-side, on when output_formats set)

    File flow for tasks with output_formats:
      upload:   local files → Files API → container_upload blocks in message
      download: bash_code_execution_tool_result.content.file_id → Files API
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-opus-4-6",
        dry_run: bool = False,
    ):
        self.model   = model
        self.dry_run = dry_run

        if not dry_run:
            try:
                import anthropic
                self.client = anthropic.AsyncAnthropic(
                    api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
                )
                self._anthropic = anthropic
            except ImportError:
                raise ImportError("pip install anthropic")

    # ── API call ──────────────────────────────────────────────────────────

    async def _call(self, needs_files_api: bool, **kwargs):
        """
        Route to beta or standard messages endpoint, always using streaming.

        Streaming is required by the SDK for requests that may run longer
        than 10 minutes (code execution + web research can exceed this).
        get_final_message() returns the same Message object as create(),
        so the rest of the adapter is unchanged.

        Uses files-api-2025-04-14 beta when the task involves file uploads
        (container_upload blocks) or file downloads (code execution output).
        No other betas. No Agent Skills. No container parameter.
        """
        if needs_files_api:
            async with self.client.beta.messages.stream(
                betas=[FILES_API_BETA],
                **kwargs,
            ) as stream:
                return await stream.get_final_message()
        async with self.client.messages.stream(**kwargs) as stream:
            return await stream.get_final_message()

    # ── run() ─────────────────────────────────────────────────────────────

    async def run(self, task) -> AgentResult:
        started_at = datetime.now(timezone.utc)

        print(f"[Claude] Starting {task.task_id} "
              f"(model={self.model}, dry_run={self.dry_run})")

        if self.dry_run:
            return self._dry_run_result(task, started_at)

        # Tasks with output_formats need the Files API for both
        # container_upload blocks (input) and file download (output).
        needs_files_api = bool(task.output_formats)

        # Upload input files only when code execution is involved
        file_refs: list[tuple[str, str, str]] = []
        if task.output_formats and task.file_paths:
            file_refs = await self._upload_files(task)

        messages      = self._build_messages(task, file_refs)
        tools         = self._build_tools(task)
        system_prompt = build_system_prompt(task)

        iterations        = 0
        total_input_tok   = 0
        total_output_tok  = 0
        tool_call_log: list[ToolCall] = []
        output_files: list[str]       = []
        final_text        = ""
        forced_stop       = False

        while True:
            iterations += 1

            # Guard: max iterations
            if iterations > task.max_iterations:
                print(f"  [GUARD] Max iterations ({task.max_iterations}) hit.")
                forced_stop = True
                break

            # Guard: budget
            current_cost = estimate_cost(self.model, total_input_tok, total_output_tok)
            if current_cost > task.max_cost_usd:
                print(f"  [GUARD] Budget ${task.max_cost_usd} exceeded (${current_cost:.2f}).")
                forced_stop = True
                break

            # Guard: timeout
            elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
            if elapsed > task.timeout_seconds:
                print(f"  [GUARD] Timeout {task.timeout_seconds}s exceeded.")
                forced_stop = True
                break

            print(f"  [ITER {iterations}] Calling... "
                  f"msgs={len(messages)}  cost=${current_cost:.2f}")

            try:
                response = await self._call(
                    needs_files_api,
                    model=self.model,
                    max_tokens=32768,
                    system=system_prompt,
                    tools=tools,
                    messages=messages,
                )
            except self._anthropic.APIError as e:
                print(f"  [ERROR] {e}")
                return self._error_result(
                    task, started_at, str(e),
                    total_input_tok, total_output_tok, tool_call_log, iterations,
                )

            total_input_tok  += response.usage.input_tokens
            total_output_tok += response.usage.output_tokens

            print(f"  [ITER {iterations}] stop={response.stop_reason}  "
                  f"in={response.usage.input_tokens}  out={response.usage.output_tokens}")

            # Observability: log server tool calls
            self._log_server_tools(response.content, iterations, tool_call_log)

            # ── end_turn ─────────────────────────────────────────────────
            if response.stop_reason == "end_turn":
                final_text = self._extract_text(response.content)
                new = await self._capture_files(response.content, task)
                output_files.extend(new)
                break

            # ── pause_turn ───────────────────────────────────────────────
            # Server-side loop hit its iteration cap.
            # Docs: "provide the response back as-is to let Claude continue."
            # Append as assistant turn. No user message.
            if response.stop_reason == "pause_turn":
                print(f"  [INFO] pause_turn — continuing.")
                new = await self._capture_files(response.content, task)
                output_files.extend(new)
                messages.append({
                    "role": "assistant",
                    "content": self._serialize(response.content),
                })
                continue

            # ── max_tokens ───────────────────────────────────────────────
            if response.stop_reason == "max_tokens":
                print(f"  [WARN] max_tokens hit.")

                # Identify unresolved server_tool_use blocks.
                # If any exist, continuing the conversation would fail —
                # the API requires bash_code_execution_tool_result in the
                # same response and we cannot produce it ourselves.
                resolved = {
                    getattr(b, "tool_use_id", None)
                    for b in response.content
                    if getattr(b, "type", None) in (
                        "bash_code_execution_tool_result",
                        "text_editor_code_execution_tool_result",
                        "web_search_tool_result",
                        "web_fetch_tool_result",
                    )
                }
                unresolved = [
                    b for b in response.content
                    if getattr(b, "type", None) == "server_tool_use"
                    and getattr(b, "id", None) not in resolved
                ]
                if unresolved:
                    print(f"  [WARN] {len(unresolved)} server tool(s) unresolved — forcing stop.")
                    forced_stop = True
                    break

                # All server tools have results; only the text output was cut.
                # Safe to continue conversation.
                messages.append({
                    "role": "assistant",
                    "content": self._serialize(response.content),
                })
                messages.append({
                    "role": "user",
                    "content": "Your response was cut off. Please continue.",
                })
                continue

            # ── tool_use ─────────────────────────────────────────────────
            # Our tools are all server-side; stop_reason="tool_use" is rare
            # but handled — append and loop.
            if response.stop_reason == "tool_use":
                print(f"  [INFO] tool_use — appending and continuing.")
                messages.append({
                    "role": "assistant",
                    "content": self._serialize(response.content),
                })
                continue

            # Unknown
            print(f"  [WARN] Unexpected stop_reason={response.stop_reason}.")
            forced_stop = True
            break

        # ── Forced-stop synthesis ─────────────────────────────────────────
        if forced_stop:
            print(f"  [SYNTH] Synthesizing from accumulated context...")
            messages.append({
                "role": "user",
                "content": (
                    "You have reached the research limit. Please synthesize all "
                    "findings gathered so far into a final well-cited research report. "
                    "Note any areas where research is incomplete."
                ),
            })
            try:
                synth = await self._call(
                    False,    # synthesis is text-only, no Files API needed
                    model=self.model,
                    max_tokens=8192,
                    system=system_prompt,
                    messages=messages,
                )
                total_input_tok  += synth.usage.input_tokens
                total_output_tok += synth.usage.output_tokens
                final_text = self._extract_text(synth.content)
            except self._anthropic.APIError as e:
                final_text = f"[Synthesis error: {e}]"

        completed_at = datetime.now(timezone.utc)
        total_cost   = estimate_cost(self.model, total_input_tok, total_output_tok)

        # File output validation
        output_file_errors: dict = {}
        completed = True
        if task.output_formats and not forced_stop:
            produced = {os.path.splitext(f)[1].lstrip(".") for f in output_files}
            for fmt in task.output_formats:
                if fmt not in produced:
                    output_file_errors[fmt] = (
                        f"FILE_OUTPUT_NOT_PRODUCED: model completed without "
                        f"generating the required .{fmt} file."
                    )
            if output_file_errors:
                completed = False
                print(f"  [WARN] Missing output files: {list(output_file_errors.keys())}")

        print(f"  [DONE] iters={iterations}  cost=${total_cost:.4f}  "
              f"files={len(output_files)}  forced_stop={forced_stop}")

        return AgentResult(
            task_id=task.task_id,
            agent="claude",
            model=self.model,
            response_text=final_text,
            citations=self._extract_citations(final_text),
            tool_call_log=tool_call_log,
            input_tokens=total_input_tok,
            output_tokens=total_output_tok,
            total_cost_usd=total_cost,
            total_duration_sec=(completed_at - started_at).total_seconds(),
            iterations=iterations,
            completed=completed,
            forced_stop=forced_stop,
            error=(
                f"File generation failed: {list(output_file_errors.keys())}"
                if output_file_errors else None
            ),
            output_files=output_files,
            output_file_errors=output_file_errors,
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
        )

    # ── File upload ───────────────────────────────────────────────────────

    async def _upload_files(self, task) -> list[tuple[str, str, str]]:
        """
        Upload task input files to the Files API.
        Returns list of (filename, file_id, extension).
        Requires files-api-2025-04-14 beta.
        """
        refs = []
        for fpath in task.file_paths:
            if not os.path.exists(fpath):
                print(f"[Claude] File not found, skipping: {fpath}")
                continue
            try:
                ext = os.path.splitext(fpath)[1].lower()
                # .xlsb cannot be parsed by the Anthropic sandbox — convert to text first
                if ext == ".xlsb":
                    text = _xlsb_to_text(fpath)
                    fname_txt = os.path.basename(fpath) + ".txt"
                    import io as _io
                    meta = await self.client.beta.files.upload(
                        file=(fname_txt, _io.BytesIO(text.encode("utf-8")), "text/plain"),
                        betas=[FILES_API_BETA],
                    )
                    ext = ".txt"
                    print(f"[Claude] {os.path.basename(fpath)} → xlsb converted to text, uploaded as {fname_txt}")
                else:
                    with open(fpath, "rb") as fh:
                        meta = await self.client.beta.files.upload(
                            file=(os.path.basename(fpath), fh),
                            betas=[FILES_API_BETA],
                        )
                refs.append((os.path.basename(fpath), meta.id, ext))
                print(f"[Claude] Uploaded {os.path.basename(fpath)} → {meta.id}")
            except Exception as e:
                print(f"[Claude] Upload failed for {fpath}: {e}")
        return refs

    # ── Message construction ──────────────────────────────────────────────

    def _build_messages(self, task, file_refs: list) -> list[dict]:
        """
        Build the initial user message.

        With file_refs (code_execution tasks):
          PDF/txt/md/html/xml → document block (Claude reads natively)
          Images              → image block
          xlsx/csv/docx/py/…  → container_upload (at /input/<name> in sandbox)

        Without file_refs (text-only tasks):
          Files embedded inline as text (original working behaviour).
        """
        parts = [{"type": "text", "text": task.prompt}]

        if file_refs:
            for filename, file_id, ext in file_refs:
                if ext in _IMAGE_EXTS:
                    parts.append({
                        "type": "image",
                        "source": {"type": "file", "file_id": file_id},
                    })
                    print(f"[Claude] {filename} → image block")
                elif ext in _DOCUMENT_EXTS:
                    parts.append({
                        "type": "document",
                        "source": {"type": "file", "file_id": file_id},
                        "title": filename,
                    })
                    print(f"[Claude] {filename} → document block")
                else:
                    parts.append({
                        "type": "container_upload",
                        "file_id": file_id,
                    })
                    print(f"[Claude] {filename} → container_upload (/input/{filename})")
        else:
            for fpath in task.file_paths:
                if not os.path.exists(fpath):
                    continue
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                        text = fh.read()
                    if len(text) > 50_000:
                        text = text[:50_000] + "\n\n[... truncated ...]"
                    parts.append({
                        "type": "text",
                        "text": f"\n\n--- File: {os.path.basename(fpath)} ---\n{text}",
                    })
                except Exception:
                    parts.append({
                        "type": "text",
                        "text": f"\n\n[File: {os.path.basename(fpath)} — unreadable as text]",
                    })

        return [{"role": "user", "content": parts}]

    # ── Tools ─────────────────────────────────────────────────────────────

    def _build_tools(self, task) -> list[dict]:
        """
        web_search_20250305    — server-side; on for IAT-2/IAT-3.
        code_execution_20250825 — server-side; on when output_formats set.
          GA tool. No extra beta header needed.
        """
        tools = []
        if task.web_search_enabled:
            tools.append({"type": "web_search_20250305", "name": "web_search"})
        if task.output_formats:
            tools.append({"type": "code_execution_20250825", "name": "code_execution"})
            print(f"[Claude] code_execution enabled for: {task.output_formats}")
        return tools

    # ── File capture ──────────────────────────────────────────────────────

    async def _capture_files(self, content_blocks, task) -> list[str]:
        """
        Find files produced by code_execution and download via Files API.

        Per Anthropic docs (code-execution-tool page, Python example):
            bash_code_execution_tool_result
              └─ .content  (BashCodeExecutionResult, type="bash_code_execution_result")
                   └─ .content[]  (List[BashCodeExecutionOutputBlock])
                        └─ .file_id   ← this is what we download

        Returns list of local paths.
        """
        if not task.output_files_dir:
            return []

        to_download: list[tuple[str, str]] = []  # (file_id, filename)

        for block in content_blocks:
            if getattr(block, "type", None) != "bash_code_execution_tool_result":
                continue
            result = getattr(block, "content", None)
            if result is None:
                continue
            if getattr(result, "type", None) != "bash_code_execution_result":
                continue
            for output in (getattr(result, "content", None) or []):
                file_id = getattr(output, "file_id", None)
                if not file_id:
                    continue
                name = getattr(output, "name", None) or f"output_{task.task_id}.bin"
                print(f"  [FILE] Found: {name} ({file_id})")
                to_download.append((file_id, name))

        if not to_download:
            return []

        results = await asyncio.gather(
            *[self._download_file(fid, name, task) for fid, name in to_download],
            return_exceptions=True,
        )

        saved = []
        for r in results:
            if isinstance(r, str):
                saved.append(r)
            elif isinstance(r, Exception):
                print(f"  [FILE] Download error: {r}")
        return saved

    async def _download_file(self, file_id: str, filename: str, task) -> str:
        """Download one output file from the Files API."""
        os.makedirs(task.output_files_dir, exist_ok=True)

        # BashCodeExecutionOutputBlock doesn't reliably expose a real filename —
        # the fallback name is output_{task_id}.bin. Use task.output_formats as
        # the authoritative extension source when the filename gives us nothing useful.
        ext = os.path.splitext(filename)[1]
        if not ext or ext == ".bin":
            ext = f".{task.output_formats[0]}" if task.output_formats else ".bin"

        local = os.path.join(task.output_files_dir, f"claude_{task.task_id}{ext}")

        raw = await self.client.beta.files.download(file_id)
        with open(local, "wb") as fh:
            async for chunk in raw.iter_bytes():
                fh.write(chunk)

        print(f"  [FILE] Saved → {local} ({os.path.getsize(local):,} bytes)")
        return local

    # ── Serialization ─────────────────────────────────────────────────────

    def _serialize(self, content_blocks) -> list[dict]:
        """
        Serialize response content for the messages list.

        Manual for known types (reliable). model_dump() fallback for anything
        new or unknown (future-proof for pause_turn continuation).
        """
        out = []
        for block in content_blocks:
            t = getattr(block, "type", None)
            if t == "text":
                out.append({"type": "text", "text": block.text})
            elif t == "tool_use":
                out.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
            elif t == "server_tool_use":
                out.append({
                    "type": "server_tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": getattr(block, "input", {}),
                })
            elif t == "web_search_tool_result":
                out.append({
                    "type": "web_search_tool_result",
                    "tool_use_id": getattr(block, "tool_use_id", ""),
                    "content": getattr(block, "content", []),
                })
            else:
                # Fallback: Pydantic model → dict (handles all other result types)
                if hasattr(block, "model_dump"):
                    out.append(block.model_dump(mode="json", exclude_unset=True))
        return out

    # ── Observability ─────────────────────────────────────────────────────

    def _log_server_tools(
        self,
        content_blocks,
        iteration: int,
        log: list[ToolCall],
    ) -> None:
        """Log server_tool_use calls (web searches, code execution) for observability."""
        for block in content_blocks:
            if getattr(block, "type", None) != "server_tool_use":
                continue
            name    = getattr(block, "name", "unknown")
            inp     = getattr(block, "input", {})
            preview = json.dumps(inp, ensure_ascii=False)[:120] if inp else ""
            print(f"  [SERVER TOOL] {name}: {preview}")
            log.append(ToolCall(
                iteration=iteration,
                tool_name=name,
                tool_input=inp if isinstance(inp, dict) else {},
                result_preview="(server-side)",
                result_tokens=0,
                timestamp=datetime.now(timezone.utc).isoformat(),
                duration_ms=0,
            ))

    # ── Parsing helpers ───────────────────────────────────────────────────

    def _extract_text(self, content_blocks) -> str:
        return "\n".join(
            b.text for b in content_blocks if hasattr(b, "text")
        )

    def _extract_citations(self, text: str) -> list[dict]:
        pattern = r'\[(\d+)\]\s+(.+?)\s*[—\-]+\s*(https?://\S+)'
        return [
            {
                "index": int(m.group(1)),
                "title": m.group(2).strip(),
                "url":   m.group(3).strip(),
            }
            for m in re.finditer(pattern, text)
        ]

    # ── Dry run / error ───────────────────────────────────────────────────

    def _dry_run_result(self, task, started_at) -> AgentResult:
        completed_at = datetime.now(timezone.utc)
        iat   = "CLOSED (web omitted)" if task.is_closed else "OPEN (web enabled)"
        fmts  = ", ".join(task.output_formats) if task.output_formats else "none"
        files = ", ".join(os.path.basename(f) for f in task.file_paths) or "none"

        report = (
            f"# [DRY RUN] Claude Deep Research\n\n"
            f"**Task:** {task.task_id}\n"
            f"**Model:** {self.model}\n"
            f"**IAT:** {iat}\n"
            f"**Output formats:** {fmts}\n"
            f"**Files:** {files}\n\n"
            f"## Prompt\n\n{task.prompt[:500]}"
            f"{'...' if len(task.prompt) > 500 else ''}\n"
        )
        print(f"[Claude] Dry run: {task.task_id} | IAT={iat} | formats={fmts}")

        return AgentResult(
            task_id=task.task_id,
            agent="claude",
            model=self.model,
            response_text=report,
            citations=[],
            tool_call_log=[],
            input_tokens=0,
            output_tokens=0,
            total_cost_usd=0.0,
            total_duration_sec=(completed_at - started_at).total_seconds(),
            iterations=0,
            completed=True,
            forced_stop=False,
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
        )

    def _error_result(
        self, task, started_at, error_msg,
        input_tokens, output_tokens, tool_call_log, iterations,
    ) -> AgentResult:
        completed_at = datetime.now(timezone.utc)
        return AgentResult(
            task_id=task.task_id,
            agent="claude",
            model=self.model,
            response_text="",
            tool_call_log=tool_call_log,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_cost_usd=estimate_cost(self.model, input_tokens, output_tokens),
            total_duration_sec=(completed_at - started_at).total_seconds(),
            iterations=iterations,
            completed=False,
            forced_stop=True,
            error=error_msg,
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
        )