"""
models.py — Shared data structures for the evaluation framework.

These dataclasses define the common language between:
  - The prompt author / SME (PromptPackage)
  - The task dispatcher (DispatchConfig, DispatchResult)
  - The agent adapters (ResearchTask → AgentResult)
  - The result normalizer / scorer (what comes OUT)

Data flow:
    PromptPackage → TaskDispatcher → ResearchTask → Adapter → AgentResult
                                                              ↓
                                                        DispatchResult
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class ResearchTask:
    """What the SME submits — agent-agnostic."""
    task_id: str
    prompt: str
    file_paths: list[str] = field(default_factory=list)
    web_search_enabled: bool = True
    max_iterations: int = 25
    max_cost_usd: float = 15.0
    timeout_seconds: int = 1800  # 15 minutes

    # ── Evaluation metadata ─────────────────────────────────────────
    research_type: Optional[str] = None    # CRP | RCP | SCP | LDP | FSP or full forms
    iat_type: Optional[str] = None         # IAT-1 (Closed) | IAT-2 (Semi-Open) | IAT-3 (Open)
    decision_archetype: Optional[str] = None
    domain: Optional[str] = None

    # ── File output requirements ────────────────────────────────────
    # Populated by csv_loader via detect_output_formats(prompt).
    # Empty list means no file output is required — all existing
    # behaviour is preserved exactly as before.
    output_formats: list[str] = field(default_factory=list)   # e.g. ["xlsx"] or ["docx", "pptx"]

    # Absolute local path where adapters should write generated files.
    # Set by TaskDispatcher before dispatching. None if output_formats
    # is empty or no results dir is configured.
    output_files_dir: Optional[str] = None

    @property
    def is_closed(self) -> bool:
        """IAT-1 tasks: agent must NOT use web search."""
        return self.iat_type == "IAT-1"

    @property
    def is_hybrid(self) -> bool:
        """IAT-2/IAT-3 tasks: agent may (or must) use web search."""
        return self.iat_type in ("IAT-2", "IAT-3")


@dataclass
class ToolCall:
    """
    One tool invocation by the agent.

    This is the unit of observability. For Claude, we capture every single
    tool call. For managed agents (OpenAI, Gemini), we only get what they
    expose in metadata — usually just counts, not details.
    """
    iteration: int
    tool_name: str
    tool_input: dict
    result_preview: str          # first 500 chars of result
    result_tokens: int           # approximate token count of the full result
    timestamp: str
    duration_ms: int             # how long the tool execution took


@dataclass
class AgentResult:
    """
    What every adapter returns — the common output format.

    Whether the agent is Claude (transparent loop) or OpenAI (black box),
    the result always looks like this.
    """
    task_id: str
    agent: str                           # "claude", "openai", "gemini", "perplexity"
    model: str                           # exact model string used
    response_text: str                   # the final research report (markdown)

    # Citations extracted from the response
    citations: list[dict] = field(default_factory=list)  # [{url, title, snippet}]

    # Output files the agent produced (absolute local paths)
    output_files: list[str] = field(default_factory=list)

    # Per-format errors when file generation was expected but failed.
    # Keys are format strings (e.g. "xlsx"), values are error descriptions.
    # Non-empty only when output_formats was set on the ResearchTask.
    output_file_errors: dict = field(default_factory=dict)

    # Observability — the full log of what the agent did
    tool_call_log: list[ToolCall] = field(default_factory=list)

    # Cost accounting
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost_usd: float = 0.0

    # Timing
    total_duration_sec: float = 0.0
    iterations: int = 0

    # Status
    completed: bool = True
    forced_stop: bool = False            # True if budget/iteration limit hit
    error: Optional[str] = None

    # Timestamps
    started_at: str = ""
    completed_at: str = ""


# ─── Prompt Package ──────────────────────────────────────────────────────────

@dataclass
class PromptPackage:
    """
    Complete evaluation unit created by the SME.

    Contains the four required components from the evaluation framework:
      1. The Prompt (context + task + constraints + output format)
      2. Data Files (min 3 files with embedded traps + noise)
      3. Solution Logic (step-by-step trace + golden answer)
      4. Sanity Check (lazy AI prediction + expert validation)
    """
    task_id: str
    prompt: str
    file_paths: list[str] = field(default_factory=list)

    # ── Classification ────────────────────────────────────────────────
    research_type: str = ""
    iat_type: str = ""
    decision_archetype: str = ""
    domain: str = ""

    # ── Solution Logic (Component 3) ──────────────────────────────────
    solution_steps: list[str] = field(default_factory=list)
    golden_answer_range: str = ""
    external_data_needed: list[dict] = field(default_factory=list)

    # ── Sanity Check (Component 4) ────────────────────────────────────
    lazy_ai_prediction: str = ""
    expert_validation: str = ""

    # ── File output requirements ──────────────────────────────────────
    # Auto-populated by csv_loader via detect_output_formats(prompt).
    output_formats: list[str] = field(default_factory=list)

    # ── Budget overrides ──────────────────────────────────────────────
    max_cost_usd: float = 15.0
    timeout_seconds: int = 1800
    max_iterations: int = 25

    @property
    def is_closed(self) -> bool:
        return self.iat_type == "IAT-1"

    @property
    def is_hybrid(self) -> bool:
        return self.iat_type in ("IAT-2", "IAT-3")

    def to_research_task(self) -> ResearchTask:
        """Convert to a ResearchTask for adapter consumption."""
        return ResearchTask(
            task_id=self.task_id,
            prompt=self.prompt,
            file_paths=list(self.file_paths),
            web_search_enabled=not self.is_closed,
            max_iterations=self.max_iterations,
            max_cost_usd=self.max_cost_usd,
            timeout_seconds=self.timeout_seconds,
            research_type=self.research_type,
            iat_type=self.iat_type,
            decision_archetype=self.decision_archetype,
            domain=self.domain,
            output_formats=list(self.output_formats),
            # output_files_dir is set by TaskDispatcher, not here
        )


# ─── Dispatch Configuration ──────────────────────────────────────────────────

@dataclass
class DispatchConfig:
    """
    Controls how the dispatcher runs a task across agents.
    """
    agents: list[str] = field(default_factory=lambda: [
        "claude", "openai", "gemini", "perplexity",
    ])
    passes_per_agent: int = 1
    dry_run: bool = False
    max_concurrent: int = 4
    fail_fast: bool = False

    # ── MCP / Tier 2 config ───────────────────────────────────────────
    mcp_server_url: Optional[str] = None
    corpus_dir: Optional[str] = None

    # ── File output config ────────────────────────────────────────────
    # Base directory for generated output files.
    # TaskDispatcher creates <output_files_base_dir>/<task_id>/ per task
    # and sets ResearchTask.output_files_dir accordingly.
    # If None and output_formats is non-empty, a temp dir is used.
    output_files_base_dir: Optional[str] = None

    # ── IAT enforcement ───────────────────────────────────────────
    # When False (default): web search is always enabled for all agents
    # regardless of IAT type. Use this during initial runs to get responses.
    # When True: IAT-1 tasks disable web search, enforcing closed-corpus
    # evaluation. Enable this for benchmark validity.
    enforce_iat: bool = False

    # ── Per-agent budget overrides ─────────────────────────────────
    agent_overrides: dict = field(default_factory=dict)


# ─── Dispatch Result ─────────────────────────────────────────────────────────

@dataclass
class DispatchResult:
    """
    Aggregated results from dispatching one PromptPackage to all agents.
    """
    task_id: str
    package: PromptPackage
    config: DispatchConfig

    # agent_name → [AgentResult_pass1, AgentResult_pass2, ...]
    agent_results: dict = field(default_factory=dict)

    # Dispatch metadata
    agents_attempted: list[str] = field(default_factory=list)
    agents_succeeded: list[str] = field(default_factory=list)
    agents_failed: list[str] = field(default_factory=list)
    agent_errors: dict = field(default_factory=dict)

    total_cost_usd: float = 0.0
    total_duration_sec: float = 0.0
    dispatched_at: str = ""
    completed_at: str = ""

    @property
    def all_results_flat(self) -> list[AgentResult]:
        flat = []
        for results_list in self.agent_results.values():
            flat.extend(results_list)
        return flat

    @property
    def best_per_agent(self) -> dict:
        best = {}
        for agent_name, results in self.agent_results.items():
            completed = [r for r in results if r.completed and not r.error]
            if completed:
                best[agent_name] = max(completed, key=lambda r: len(r.response_text))
            elif results:
                best[agent_name] = results[0]
        return best