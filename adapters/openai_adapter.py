"""
openai_adapter.py — Black-box managed agent adapter for OpenAI Deep Research.

File handling
-------------
When task.output_formats is non-empty:
  - Input files go into container.file_ids (NOT as input_file message blocks)
  - input_file blocks in the message cause invalid_file errors when code_interpreter is active
  - After completion, generated files are retrieved via the Containers API:
      GET /containers/{container_id}/files   → list files (filter source=assistant)
      GET /containers/{container_id}/files/{file_id}/content  → download bytes

When task.output_formats is empty (text-only tasks):
  - Files are passed as input_file blocks in the message (original working approach)
  - No code_interpreter tool
"""

from __future__ import annotations

import os
import time
import asyncio
import httpx
from datetime import datetime, timezone
from typing import Optional

from models import ResearchTask, AgentResult


# ─── Pricing constants ────────────────────────────────────────────────────────

PRICING = {
    "o3-deep-research": {
        "input_per_mtok": 10.00,
        "output_per_mtok": 40.00,
    },
    "o3-deep-research-2025-06-26": {
        "input_per_mtok": 10.00,
        "output_per_mtok": 40.00,
    },
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = None
    for key in PRICING:
        if model.startswith(key):
            prices = PRICING[key]
            break
    if prices is None:
        prices = PRICING["o3-deep-research"]
    input_cost = (input_tokens / 1_000_000) * prices["input_per_mtok"]
    output_cost = (output_tokens / 1_000_000) * prices["output_per_mtok"]
    return round(input_cost + output_cost, 6)


# ─── The adapter ──────────────────────────────────────────────────────────────

class OpenAIAdapter:

    _SUPPORTED_EXTENSIONS = {
        ".pdf", ".txt", ".md", ".csv", ".json",
        ".docx", ".xlsx", ".xlsb", ".pptx", ".doc", ".xls",
        ".html", ".xml", ".rtf",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "o3-deep-research",
        dry_run: bool = False,
        mcp_server_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.dry_run = dry_run
        self.mcp_server_url = mcp_server_url

        if not dry_run:
            try:
                from openai import AsyncOpenAI
                self.client = AsyncOpenAI(
                    api_key=self.api_key,
                    timeout=3600,
                )
            except ImportError:
                raise ImportError("pip install openai  (required for OpenAI adapter)")

    async def run(self, task: ResearchTask) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        print(f"[OpenAI] Starting task {task.task_id} "
              f"(model={self.model}, dry_run={self.dry_run})")

        if self.dry_run:
            return self._dry_run_result(task, started_at)

        try:
            file_ids = await self._upload_files(task)
            if file_ids:
                print(f"[OpenAI] Uploaded {len(file_ids)} files")

            # Two modes depending on whether file output is required:
            #
            # FILE OUTPUT MODE (output_formats set):
            #   - files go in container.file_ids inside the code_interpreter tool
            #   - input message contains ONLY text (no input_file blocks)
            #   - input_file + code_interpreter together causes invalid_file error
            #   - generated files retrieved via Containers API after completion
            #
            # TEXT-ONLY MODE (no output_formats):
            #   - files passed as input_file blocks in the message
            #   - no code_interpreter tool
            file_output_mode = bool(task.output_formats)

            tools = self._build_tools(task, file_ids if file_output_mode else [])
            input_content = self._build_input(task, [] if file_output_mode else file_ids)

            create_kwargs: dict = dict(
                model=self.model,
                input=input_content,
                tools=tools,
            )
            # include=code_interpreter_call.outputs gives us container_id
            if file_output_mode:
                create_kwargs["include"] = ["code_interpreter_call.outputs"]

            print(f"[OpenAI] Submitting to Responses API "
                  f"({'file output' if file_output_mode else 'text only'} mode)...")
            response = await self.client.responses.create(**create_kwargs)

            response = await self._poll_until_complete(response, task, started_at)
            return await self._extract_result(response, task, started_at)

        except Exception as e:
            print(f"[OpenAI] Error: {e}")
            return self._error_result(task, started_at, str(e))

    # ─── File upload ──────────────────────────────────────────────────

    async def _upload_files(self, task: ResearchTask) -> list[str]:
        file_ids = []
        for fpath in task.file_paths:
            if not os.path.exists(fpath):
                print(f"[OpenAI] File not found: {fpath}")
                continue
            ext = os.path.splitext(fpath)[1].lower()
            if ext not in self._SUPPORTED_EXTENSIONS:
                print(f"[OpenAI] Skipping unsupported file type {ext}: "
                      f"{os.path.basename(fpath)}")
                continue
            try:
                # .xlsb files are not natively supported by the Files API —
                # convert to TSV text and upload as a .txt file
                if ext == ".xlsb":
                    import pyxlsb, tempfile
                    tsv_lines = [f"# Converted from: {os.path.basename(fpath)}"]
                    with pyxlsb.open_workbook(fpath) as wb:
                        for sheet_name in wb.sheets:
                            tsv_lines.append(f"\n=== Sheet: {sheet_name} ===")
                            with wb.get_sheet(sheet_name) as ws:
                                for row in ws.rows():
                                    vals = [str(r.v) if r.v is not None else "" for r in row]
                                    if any(v.strip() for v in vals):
                                        tsv_lines.append("\t".join(vals))
                    tsv_bytes = "\n".join(tsv_lines).encode("utf-8")
                    txt_name = os.path.basename(fpath).replace(".xlsb", ".txt")
                    upload = await self.client.files.create(
                        file=(txt_name, tsv_bytes, "text/plain"),
                        purpose="assistants",
                    )
                else:
                    with open(fpath, "rb") as f:
                        upload = await self.client.files.create(
                            file=f,
                            purpose="assistants",
                        )
                file_ids.append(upload.id)
                print(f"[OpenAI] Uploaded {os.path.basename(fpath)} → {upload.id}")
            except Exception as e:
                print(f"[OpenAI] Failed to upload {fpath}: {e}")
        return file_ids

    # ─── Input + tools ────────────────────────────────────────────────

    def _build_input(self, task: ResearchTask, file_ids: list[str]) -> list[dict]:
        """
        Build message input. file_ids here are ONLY for text-only mode.
        In file output mode, file_ids go into container.file_ids instead.
        """
        prompt_text = task.prompt

        if task.output_formats:
            fmt = task.output_formats[0]
            prompt_text += (
                f"\n\n{'='*60}\n"
                f"MANDATORY FINAL STEP — YOUR RESPONSE IS INCOMPLETE WITHOUT THIS\n"
                f"{'='*60}\n\n"
                f"After completing your analysis, you MUST use the code interpreter "
                f"to generate a {fmt.upper()} file. Do NOT end your response without "
                f"producing this file. A response without a generated file is FAILED.\n\n"
                f"Write Python code that:\n"
                f"1. Uses the appropriate library: docx for DOCX, openpyxl for XLSX, python-pptx for PPTX\n"
                f"2. Writes all sections from your analysis directly into the file\n"
                f"3. Hardcodes your findings — do NOT attempt to re-read input files\n"
                f"4. Saves the output as: output.{fmt}\n\n"
                f"Execute this code using the code interpreter before finishing."
            )

        content = [{"type": "input_text", "text": prompt_text}]
        for fid in file_ids:
            content.append({"type": "input_file", "file_id": fid})
        return [{"role": "user", "content": content}]

    def _build_tools(self, task: ResearchTask, container_file_ids: list[str]) -> list[dict]:
        """
        Build tools list.

        container_file_ids: file IDs to pass into the code_interpreter container.
        Should be non-empty only when output_formats is set.
        In text-only mode, pass an empty list.
        """
        tools = []

        if task.web_search_enabled:
            tools.append({"type": "web_search_preview"})
        else:
            print(f"[OpenAI] IAT-1 (Closed): web_search disabled")

        if task.output_formats:
            container: dict = {"type": "auto"}
            if container_file_ids:
                container["file_ids"] = container_file_ids
            tools.append({
                "type": "code_interpreter",
                "container": container,
            })
            print(f"[OpenAI] code_interpreter enabled, "
                  f"{len(container_file_ids)} files in container, "
                  f"output: {task.output_formats}")

        if self.mcp_server_url:
            tools.append({
                "type": "mcp",
                "server_label": "eval_corpus",
                "server_url": self.mcp_server_url,
                "require_approval": "never",
            })
            print(f"[OpenAI] MCP server attached: {self.mcp_server_url}")

        return tools

    # ─── Polling ──────────────────────────────────────────────────────

    async def _poll_until_complete(self, response, task, started_at):
        STALE_THRESHOLD_SECONDS = 900
        poll_interval = 10
        max_poll_interval = 30
        last_status = None
        last_status_changed_at = time.monotonic()

        while hasattr(response, 'status') and response.status in ("queued", "in_progress"):
            elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
            if elapsed > task.timeout_seconds:
                print(f"[OpenAI] Timeout ({task.timeout_seconds}s) exceeded")
                break

            if response.status != last_status:
                last_status = response.status
                last_status_changed_at = time.monotonic()

            stale_for = time.monotonic() - last_status_changed_at
            if stale_for >= STALE_THRESHOLD_SECONDS:
                raise RuntimeError(
                    f"OpenAI response stuck in '{response.status}' for "
                    f"{stale_for / 60:.1f}m"
                )

            print(f"[OpenAI] Status: {response.status} (elapsed: {elapsed:.0f}s)")
            await asyncio.sleep(poll_interval)
            response = await self.client.responses.retrieve(response.id)
            poll_interval = min(poll_interval * 1.5, max_poll_interval)

        return response

    # ─── Result extraction ────────────────────────────────────────────

    async def _extract_result(self, response, task, started_at) -> AgentResult:
        completed_at = datetime.now(timezone.utc)

        report_text = ""
        citations = []
        output_files: list[str] = []
        output_file_errors: dict = {}
        seen_container_ids = set() 

        if hasattr(response, 'output') and response.output:
            for block in response.output:
                if not hasattr(block, 'type'):
                    continue

                if block.type == "message":
                    for content in (getattr(block, 'content', None) or []):
                        if hasattr(content, 'text'):
                            report_text += content.text
                        elif hasattr(content, 'type') and content.type == "output_text":
                            report_text += getattr(content, 'text', '')

                elif block.type == "text":
                    report_text += getattr(block, 'text', '')

                elif block.type == "code_interpreter_call":
                    # Files live in the container — retrieve via Containers API
                    container_id = getattr(block, 'container_id', None)
                    if container_id and task.output_files_dir:
                        if container_id in seen_container_ids: 
                            continue                         
                        seen_container_ids.add(container_id)
                        print(f"[OpenAI] code_interpreter container: {container_id}")
                        all_files = await self._list_container_files(container_id)
                        # Only assistant-generated files, not user uploads
                        files = [f for f in all_files if f.get("source") == "assistant"]
                        print(f"[OpenAI] Container: {len(all_files)} total, "
                              f"{len(files)} assistant-generated")
                        for file_info in files:
                            file_id = file_info.get("id")
                            filename = os.path.basename(
                                file_info.get("path") or f"openai_{task.task_id}.bin"
                            )
                            if file_id:
                                try:
                                    path = await self._download_container_file(
                                        container_id, file_id, filename,
                                        task.output_files_dir, task.task_id
                                    )
                                    output_files.append(path)
                                    print(f"[OpenAI] Saved → {path}")
                                except Exception as e:
                                    ext = os.path.splitext(filename)[1].lstrip(".") or "bin"
                                    output_file_errors[ext] = str(e)
                                    print(f"[OpenAI] Failed to download {file_id}: {e}")

        citations = self._extract_citations_from_response(response)

        input_tokens = 0
        output_tokens = 0
        if hasattr(response, 'usage') and response.usage:
            input_tokens = getattr(response.usage, 'input_tokens', 0)
            output_tokens = getattr(response.usage, 'output_tokens', 0)

        total_cost = estimate_cost(self.model, input_tokens, output_tokens)
        status = getattr(response, 'status', 'unknown')
        completed = status == "completed"
        forced_stop = status in ("incomplete", "failed")

        if task.output_formats and not forced_stop:
            produced_fmts = {os.path.splitext(f)[1].lstrip(".") for f in output_files}
            for fmt in task.output_formats:
                if fmt not in produced_fmts and fmt not in output_file_errors:
                    output_file_errors[fmt] = (
                        "FILE_OUTPUT_NOT_PRODUCED: o3-deep-research completed "
                        "but did not generate the required file."
                    )
            if output_file_errors:
                completed = False

        if hasattr(response, 'output') and response.output:
            for i, block in enumerate(response.output):
                btype = getattr(block, 'type', 'unknown')
                print(f"[OpenAI DEBUG] block[{i}] type={btype}")
                if btype == 'code_interpreter_call':
                    print(f"  container_id={getattr(block, 'container_id', 'MISSING')}")

        print(f"[OpenAI] Done. status={status}, "
              f"tokens=({input_tokens}in/{output_tokens}out), "
              f"cost=${total_cost:.4f}, "
              f"citations={len(citations)}, files={len(output_files)}")

        return AgentResult(
            task_id=task.task_id,
            agent="openai",
            model=self.model,
            response_text=report_text,
            citations=citations,
            tool_call_log=[],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_cost_usd=total_cost,
            total_duration_sec=(completed_at - started_at).total_seconds(),
            iterations=1,
            completed=completed,
            forced_stop=forced_stop,
            error=(
                f"File generation failed for: {list(output_file_errors.keys())}"
                if output_file_errors
                else (None if completed else f"Status: {status}")
            ),
            output_files=output_files,
            output_file_errors=output_file_errors,
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
        )

    # ─── Containers API ───────────────────────────────────────────────

    async def _list_container_files(self, container_id: str) -> list[dict]:
        """List files in the code_interpreter container via the Containers API."""
        url = f"https://api.openai.com/v1/containers/{container_id}/files"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "containers=v1",
        }
        try:
            async with httpx.AsyncClient(timeout=30) as http:
                resp = await http.get(url, headers=headers)
            if resp.status_code == 200:
                return resp.json().get("data", [])
            print(f"[OpenAI] Container files list failed: "
                  f"{resp.status_code} {resp.text[:200]}")
            return []
        except Exception as e:
            print(f"[OpenAI] Container files list error: {e}")
            return []

    async def _download_container_file(
        self,
        container_id: str,
        file_id: str,
        filename: str,
        output_dir: str,
        task_id: str,
    ) -> str:
        """Download a generated file from the container via the Containers API."""
        os.makedirs(output_dir, exist_ok=True)
        base, ext = os.path.splitext(filename)
        local_filename = f"openai_{task_id}{ext}" if ext else f"openai_{task_id}_{base}"
        local_path = os.path.join(output_dir, local_filename)

        url = f"https://api.openai.com/v1/containers/{container_id}/files/{file_id}/content"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "containers=v1",
        }
        async with httpx.AsyncClient(timeout=60) as http:
            resp = await http.get(url, headers=headers)
        resp.raise_for_status()

        with open(local_path, "wb") as f:
            f.write(resp.content)
        return local_path

    # ─── Citations ────────────────────────────────────────────────────

    def _extract_citations_from_response(self, response) -> list[dict]:
        citations = []
        seen_urls = set()

        if not hasattr(response, 'output'):
            return citations

        for block in response.output:
            content_list = []
            if hasattr(block, 'content'):
                content_list = block.content or []
            elif hasattr(block, 'text'):
                continue

            for content in content_list:
                annotations = getattr(content, 'annotations', [])
                for ann in annotations:
                    if hasattr(ann, 'url') and ann.url not in seen_urls:
                        citations.append({
                            "url": ann.url,
                            "title": getattr(ann, 'title', ''),
                            "snippet": getattr(ann, 'text', '')[:200],
                        })
                        seen_urls.add(ann.url)

        return citations

    # ─── Dry run ──────────────────────────────────────────────────────

    def _dry_run_result(self, task: ResearchTask, started_at) -> AgentResult:
        completed_at = datetime.now(timezone.utc)
        iat_status = "CLOSED (web disabled)" if task.is_closed else "OPEN (web enabled)"
        file_summary = ", ".join(os.path.basename(f) for f in task.file_paths) or "none"
        mode = "file output (container API)" if task.output_formats else "text only (input_file blocks)"

        report = (
            f"# [DRY RUN] OpenAI Deep Research Report\n\n"
            f"**Task:** {task.task_id}\n"
            f"**Model:** {self.model}\n"
            f"**IAT:** {iat_status}\n"
            f"**Files:** {file_summary}\n"
            f"**Mode:** {mode}\n"
            f"**MCP Server:** {self.mcp_server_url or 'none'}\n\n"
            f"## Prompt Received\n\n"
            f"{task.prompt[:500]}{'...' if len(task.prompt) > 500 else ''}\n\n"
            f"## Limitations\n\nDry-run mode — no actual research performed.\n"
        )

        print(f"[OpenAI] Dry run complete for {task.task_id} ({mode})")

        return AgentResult(
            task_id=task.task_id,
            agent="openai",
            model=self.model,
            response_text=report,
            citations=[{"url": "https://example.com/mock", "title": "Mock Citation", "snippet": "Dry run"}],
            tool_call_log=[],
            input_tokens=0,
            output_tokens=0,
            total_cost_usd=0.0,
            total_duration_sec=(completed_at - started_at).total_seconds(),
            iterations=1,
            completed=True,
            forced_stop=False,
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
        )

    def _error_result(self, task, started_at, error_msg) -> AgentResult:
        completed_at = datetime.now(timezone.utc)
        return AgentResult(
            task_id=task.task_id,
            agent="openai",
            model=self.model,
            response_text="",
            tool_call_log=[],
            input_tokens=0,
            output_tokens=0,
            total_cost_usd=0.0,
            total_duration_sec=(completed_at - started_at).total_seconds(),
            iterations=0,
            completed=False,
            forced_stop=True,
            error=error_msg,
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
        )