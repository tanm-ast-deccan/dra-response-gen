"""
models.py — Lean, standalone data structures for the unified pipeline.

    Task        → one unit of work for ONE provider + ONE pass
    ToolCall    → an observability record of a local tool invocation
    RunResult   → the normalized output every provider produces

No imports from the repo root; this is the standalone language of the
indrayudh_pipeline package.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Task:
    """
    A single provider+pass unit of work, derived from a loaded PromptPackage.

    The same prompt fans out to N providers x K passes, each becoming one Task
    with a distinct run_id so results never collide.
    """
    task_id: str                 # original SME task id (shared across providers)
    prompt: str
    provider: str                # logical name: "claude" | "openai" | "gemini" | "qwen"
    model_slug: str              # resolved OpenRouter slug
    pass_index: int = 1
    file_paths: list = field(default_factory=list)
    output_formats: list = field(default_factory=list)
    output_dir: Optional[str] = None   # where generated files for this task go
    drive_url: str = ""
    sme_name: str = ""

    @property
    def run_id(self) -> str:
        """Unique id for this provider+pass run."""
        return f"{self.task_id}__{self.provider}__p{self.pass_index}"


@dataclass
class ToolCall:
    """One local tool invocation — observability for the agentic loop."""
    turn: int
    name: str
    arguments: dict
    result_preview: str = ""     # first ~500 chars of the tool result
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunResult:
    """
    The common output format for every provider. Whatever the model, the
    result always looks like this.
    """
    task_id: str
    run_id: str
    provider: str
    model: str
    pass_index: int

    response_text: str = ""
    citations: list = field(default_factory=list)        # [{url, title, snippet}]
    output_files: list = field(default_factory=list)     # local paths
    output_file_errors: dict = field(default_factory=dict)
    tool_calls: list = field(default_factory=list)       # [ToolCall]

    # Accounting
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost_usd: float = 0.0

    # Loop / status
    turns: int = 0
    completed: bool = True
    forced_stop: bool = False
    error: Optional[str] = None

    # Timing
    total_duration_sec: float = 0.0
    started_at: str = ""
    completed_at: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        # ToolCall objects are already dataclasses; asdict handles them,
        # but if any raw ToolCall slipped in, normalize defensively.
        d["tool_calls"] = [
            tc.to_dict() if isinstance(tc, ToolCall) else tc
            for tc in self.tool_calls
        ]
        return d
