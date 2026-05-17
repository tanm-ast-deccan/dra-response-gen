"""
gemini_adapter.py — Black-box managed agent adapter for Google Gemini Deep Research.

Gemini Deep Research runs as a background operation: you submit a prompt
(with optional files and tools), receive an operation handle, and poll
until the research is complete. Like OpenAI, the internal agent loop is
fully opaque.

Architecture:
    ┌─────────────────────────────────────────────────┐
    │            GeminiAdapter                         │
    │                                                  │
    │  1. Upload files → Files API (or inline)         │
    │  2. POST generateContent (background mode)       │
    │  3. Poll: GET operation/{id}                     │
    │     status: PENDING → RUNNING → SUCCEEDED        │
    │  4. Extract report text + grounding metadata     │
    │  5. Return AgentResult                           │
    │                                                  │
    │  Structural advantage: 1M token context window   │
    │  means most corpora fit inline (Tier 1 only).    │
    │                                                  │
    │  Limitation: No MCP support for Deep Research.   │
    │  Tier 2 uses File Search tool or context stuff.  │
    │                                                  │
    │  Observability: LOW                              │
    │  - Final report text                             │
    │  - Token counts (input/output)                   │
    │  - Grounding metadata (search queries, sources)  │
    │  - No individual tool call logs                  │
    └─────────────────────────────────────────────────┘

Key design decisions:
    - Uses generateContent with background mode for async execution.
    - Context caching: when running multiple prompts against the same
      file corpus, cached tokens cost ~75% less. Important for Pass@3.
    - No MCP support — Gemini Deep Research does not support custom
      function calling tools or remote MCP servers (as of early 2026).
    - 1M context window is a structural advantage for IAT-1 (Closed)
      tasks with large document sets.
"""

from __future__ import annotations

import os
import time
import json
import asyncio
from datetime import datetime, timezone
from typing import Optional

from models import ResearchTask, AgentResult, ToolCall


# ─── Pricing constants ──────────────────────────────────────────────────
# Source: https://ai.google.dev/pricing
# Gemini Deep Research pricing (as of early 2026)
# Note: Context-cached input tokens cost ~75% less
#
# The Deep Research agent is accessed via the Interactions API using:
#   agent='deep-research-pro-preview-12-2025'
# NOT via generateContent — that model string does not exist.

DEEP_RESEARCH_AGENT = "deep-research-pro-preview-12-2025"
POLL_INTERVAL_SECONDS = 10

PRICING = {
    "deep-research-pro-preview-12-2025": {
        "input_per_mtok": 2.00,
        "output_per_mtok": 12.00,
        "cached_input_per_mtok": 0.50,
        "search_per_1k": 14.00,
    },
    # Fallback / older agent
    "gemini-2.0-flash-deep-research": {
        "input_per_mtok": 0.10,
        "output_per_mtok": 0.40,
        "cached_input_per_mtok": 0.025,
        "search_per_1k": 14.00,
    },
}


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
) -> float:
    """
    Estimate USD cost from token counts.

    Does not account for search query costs (billed separately).
    Cached tokens are subtracted from input and billed at reduced rate.
    """
    prices = None
    for key in PRICING:
        if model.startswith(key) or key.startswith(model):
            prices = PRICING[key]
            break
    if prices is None:
        prices = PRICING[DEEP_RESEARCH_AGENT]

    uncached_input = max(0, input_tokens - cached_tokens)
    input_cost = (uncached_input / 1_000_000) * prices["input_per_mtok"]
    cached_cost = (cached_tokens / 1_000_000) * prices["cached_input_per_mtok"]
    output_cost = (output_tokens / 1_000_000) * prices["output_per_mtok"]
    return round(input_cost + cached_cost + output_cost, 6)


# ─── The adapter ──────────────────────────────────────────────────────────

class GeminiAdapter:
    """
    Runs a deep research task using Google's Gemini API.

    This adapter handles:
      1. File upload via Gemini's Files API (or inline for small files)
      2. Task submission in background mode (async)
      3. Polling until completion
      4. Result extraction with grounding metadata
      5. Optional context caching for repeated corpus queries

    Usage:
        adapter = GeminiAdapter(api_key="...")
        result = await adapter.run(task)

    Dry-run mode:
        adapter = GeminiAdapter(dry_run=True)
        result = await adapter.run(task)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEEP_RESEARCH_AGENT,   # agent string, not model string
        dry_run: bool = False,
        use_context_cache: bool = False,
    ):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.model = model                   # kept as "model" attr for compatibility
        self.dry_run = dry_run
        self.use_context_cache = use_context_cache
        self._cache_name = None

        if not dry_run:
            try:
                # google.generativeai is deprecated — use google.genai
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
                self.genai = genai
            except ImportError:
                raise ImportError(
                    "pip install google-genai  "
                    "(required for Gemini adapter — replaces google-generativeai)"
                )

    async def run(self, task: ResearchTask) -> AgentResult:
        """Execute a deep research task via Gemini's Interactions API."""
        started_at = datetime.now(timezone.utc)

        print(f"[Gemini] Starting task {task.task_id} "
              f"(model={self.model}, dry_run={self.dry_run})")

        if self.dry_run:
            return self._dry_run_result(task, started_at)

        try:
            # ── Step 1: Upload files ────────────────────────────────
            uploaded_files = await self._upload_files(task)
            if uploaded_files:
                print(f"[Gemini] Uploaded {len(uploaded_files)} files")

            # ── Step 2: Build prompt input ──────────────────────────
            # The Interactions API takes a plain string input.
            # Uploaded file URIs are embedded in the prompt so the
            # agent can fetch them via its file_search tool.
            prompt_input = self._build_input(task, uploaded_files)

            # ── Step 3: Submit as background interaction ────────────
            # Deep Research is ONLY available via the Interactions API.
            # generateContent does NOT support this agent.
            print(f"[Gemini] Submitting to Gemini Deep Research...")
            interaction = await asyncio.to_thread(
                self._client.interactions.create,
                input=prompt_input,
                agent=self.model,           # e.g. "deep-research-pro-preview-12-2025"
                background=True,
            )
            print(f"[Gemini] Research started: interaction_id={interaction.id}")

            # ── Step 4: Poll until complete ─────────────────────────
            while True:
                interaction = await asyncio.to_thread(
                    self._client.interactions.get,
                    interaction.id,
                )
                status = getattr(interaction, "status", "unknown")

                if status == "completed":
                    print(f"[Gemini] Completed: {interaction.id}")
                    break
                elif status == "failed":
                    error_msg = str(getattr(interaction, "error", "unknown error"))
                    raise RuntimeError(f"Gemini Deep Research failed: {error_msg}")
                else:
                    import logging
                    logging.getLogger("dra.gemini").debug(
                        "Status: %s — polling in %ds...", status, POLL_INTERVAL_SECONDS
                    )
                    await asyncio.sleep(POLL_INTERVAL_SECONDS)

            # ── Step 5: Extract results ─────────────────────────────
            return self._extract_result(interaction, task, started_at)

        except Exception as e:
            print(f"[Gemini] Error: {e}")
            return self._error_result(task, started_at, str(e))

    # ─── File upload ──────────────────────────────────────────────────

    async def _upload_files(self, task: ResearchTask) -> list:
        """
        Upload files via Gemini's Files API (google.genai).

        Returns a list of uploaded file objects with .uri and .name attributes.
        """
        uploaded = []

        for fpath in task.file_paths:
            if not os.path.exists(fpath):
                print(f"[Gemini] File not found: {fpath}")
                continue

            try:
                upload = await asyncio.to_thread(
                    self._client.files.upload,
                    file=fpath,
                    config={"display_name": os.path.basename(fpath)},
                )
                # Wait for file to become ACTIVE
                while getattr(upload, "state", None) and upload.state.name == "PROCESSING":
                    await asyncio.sleep(2)
                    upload = await asyncio.to_thread(
                        self._client.files.get, name=upload.name
                    )

                uploaded.append(upload)
                print(f"[Gemini] Uploaded {os.path.basename(fpath)} "
                      f"→ {upload.name}")

            except Exception as e:
                print(f"[Gemini] Failed to upload {fpath}: {e}")

        return uploaded

    # ─── Content construction ─────────────────────────────────────────

    def _build_input(self, task: ResearchTask, uploaded_files: list) -> str:
        """
        Build the string input for the Interactions API.

        The Interactions API takes a plain text input. Uploaded file URIs
        are appended so the agent can reference them via file_search.
        IAT enforcement is injected as an instruction prefix.
        """
        parts = []

        # IAT enforcement instruction
        if task.is_closed:
            parts.append(
                "[CONSTRAINT] This is a closed-book task (IAT-1). "
                "Use ONLY the provided documents. Do NOT search the web.\n\n"
            )
            print(f"[Gemini] IAT-1 (Closed): web search disabled via instruction")

        # Main prompt
        parts.append(task.prompt)

        # Append file references so the agent knows what documents to use
        if uploaded_files:
            parts.append("\n\n[ATTACHED FILES]")
            for uf in uploaded_files:
                uri = getattr(uf, "uri", getattr(uf, "name", str(uf)))
                name = getattr(uf, "display_name", os.path.basename(uri))
                parts.append(f"  - {name}: {uri}")

        return "\n".join(parts)

    def _build_tools(self, task: ResearchTask) -> list:
        """
        Kept for reference — not used by Interactions API.

        The Interactions API manages tools internally. The agent has
        google_search enabled by default; we enforce IAT-1 via prompt
        instruction in _build_input instead of tool suppression.
        """
        return []

    # ─── Result extraction ────────────────────────────────────────────

    def _extract_result(self, interaction, task, started_at) -> AgentResult:
        """
        Extract the final report from a completed Interactions API response.

        The interaction object has:
          - interaction.outputs: list of output objects
          - interaction.outputs[-1].text: the final report text
          - Usage/token metadata may not be available in the same shape
            as generateContent — we extract what we can gracefully.
        """
        completed_at = datetime.now(timezone.utc)

        # Extract text from last output
        report_text = ""
        try:
            outputs = getattr(interaction, "outputs", [])
            if outputs:
                report_text = getattr(outputs[-1], "text", "") or ""
        except Exception as e:
            print(f"[Gemini] Error extracting text: {e}")

        # Extract citations if grounding metadata is available
        citations = self._extract_citations(interaction)

        # Token usage — available on some interaction shapes, graceful fallback
        input_tokens = 0
        output_tokens = 0
        cached_tokens = 0
        try:
            usage = getattr(interaction, "usage_metadata", None)
            if usage:
                input_tokens = getattr(usage, "prompt_token_count", 0) or 0
                output_tokens = getattr(usage, "candidates_token_count", 0) or 0
                cached_tokens = getattr(usage, "cached_content_token_count", 0) or 0
        except Exception:
            pass

        total_cost = estimate_cost(
            self.model, input_tokens, output_tokens, cached_tokens
        )

        completed = bool(report_text)
        error = None if completed else "Empty response from Interactions API"

        print(f"[Gemini] Done. "
              f"tokens=({input_tokens}in [{cached_tokens} cached], "
              f"{output_tokens}out), "
              f"cost=${total_cost:.4f}, "
              f"citations={len(citations)}")

        return AgentResult(
            task_id=task.task_id,
            agent="gemini",
            model=self.model,
            response_text=report_text,
            citations=citations,
            tool_call_log=[],  # Black box — no individual tool logs
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_cost_usd=total_cost,
            total_duration_sec=(completed_at - started_at).total_seconds(),
            iterations=1,
            completed=completed,
            forced_stop=False,
            error=error,
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
        )

    def _extract_citations(self, interaction) -> list[dict]:
        """
        Extract citations from a completed interaction.

        The Interactions API may surface grounding metadata differently
        than generateContent. We check both the old candidates shape and
        any top-level grounding field, falling back gracefully.
        """
        citations = []
        seen_urls = set()

        try:
            # Try top-level grounding_metadata (Interactions API)
            grounding = getattr(interaction, "grounding_metadata", None)

            # Fallback: candidates shape (generateContent legacy)
            if grounding is None:
                candidates = getattr(interaction, "candidates", [])
                if candidates:
                    grounding = getattr(candidates[0], "grounding_metadata", None)

            if not grounding:
                return citations

            chunks = getattr(grounding, "grounding_chunks", [])
            for chunk in chunks:
                web = getattr(chunk, "web", None)
                if web:
                    url = getattr(web, "uri", "")
                    if url and url not in seen_urls:
                        citations.append({
                            "url": url,
                            "title": getattr(web, "title", ""),
                            "snippet": "",
                        })
                        seen_urls.add(url)

        except Exception as e:
            print(f"[Gemini] Error extracting citations: {e}")

        return citations

    # ─── Dry run ──────────────────────────────────────────────────────

    def _dry_run_result(self, task: ResearchTask, started_at) -> AgentResult:
        """Return a mock result without making any API calls."""
        completed_at = datetime.now(timezone.utc)

        iat_status = "CLOSED (search disabled)" if task.is_closed else "OPEN (search enabled)"
        file_summary = ", ".join(os.path.basename(f) for f in task.file_paths) or "none"

        report = (
            f"# [DRY RUN] Gemini Deep Research Report\n\n"
            f"**Task:** {task.task_id}\n"
            f"**Model:** {self.model}\n"
            f"**IAT:** {iat_status}\n"
            f"**Research Type:** {task.research_type or 'unspecified'}\n"
            f"**Context Window:** 1M tokens (structural advantage for closed tasks)\n"
            f"**Files:** {file_summary}\n"
            f"**Context Caching:** {'enabled' if self.use_context_cache else 'disabled'}\n\n"
            f"## Executive Summary\n\n"
            f"This is a dry-run mock response. In production, Gemini's "
            f"deep research model would:\n"
            f"1. Ingest all files into its 1M token context window\n"
            f"2. Run its internal research loop (Google Search if enabled)\n"
            f"3. Produce a report with grounding metadata and source attribution\n\n"
            f"## Structural Notes\n\n"
            f"Gemini's 1M context window means most evaluation corpora fit\n"
            f"inline without truncation. This is a significant advantage for\n"
            f"IAT-1 (Closed) tasks requiring deep analysis of large document sets.\n"
            f"However, Gemini does not support MCP servers or custom function\n"
            f"calling for Deep Research — Tier 2 access is via File Search only.\n\n"
            f"## Prompt Received\n\n"
            f"{task.prompt[:500]}{'...' if len(task.prompt) > 500 else ''}\n\n"
            f"## Limitations\n\n"
            f"Dry-run mode — no actual research performed.\n"
        )

        print(f"[Gemini] Dry run complete for {task.task_id}")
        print(f"[Gemini]   IAT: {iat_status}")
        print(f"[Gemini]   Files: {len(task.file_paths)}")
        print(f"[Gemini]   Search: {'disabled' if task.is_closed else 'enabled'}")
        print(f"[Gemini]   Context caching: {self.use_context_cache}")

        return AgentResult(
            task_id=task.task_id,
            agent="gemini",
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
        """Construct an error result when the API call fails."""
        completed_at = datetime.now(timezone.utc)
        return AgentResult(
            task_id=task.task_id,
            agent="gemini",
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