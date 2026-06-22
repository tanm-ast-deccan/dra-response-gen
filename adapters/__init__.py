"""
adapters/ — Agent-specific implementations of the research agent interface.

Each adapter implements:  async run(task: ResearchTask) -> AgentResult

    ┌──────────────┬────────────────────┬──────────────────────┐
    │ Agent        │ Architecture       │ Observability        │
    ├──────────────┼────────────────────┼──────────────────────┤
    │ Claude       │ Open agent loop    │ FULL: every tool     │
    │              │ (we control it)    │ call, reasoning step │
    ├──────────────┼────────────────────┼──────────────────────┤
    │ OpenAI       │ Managed black box  │ LOW: final report +  │
    │              │ (Responses API)    │ structured citations │
    ├──────────────┼────────────────────┼──────────────────────┤
    │ Gemini       │ Managed black box  │ LOW: final report +  │
    │              │ (background mode)  │ grounding metadata   │
    ├──────────────┼────────────────────┼──────────────────────┤
    │ Qwen         │ Open agent loop    │ FULL: every tool     │
    │              │ (we control it)    │ call, local exec     │
    ├──────────────┼────────────────────┼──────────────────────┤
    │ Perplexity   │ Search-native      │ MINIMAL: report +    │
    │              │ (synchronous)      │ citation URLs only   │
    └──────────────┴────────────────────┴──────────────────────┘

All adapters support dry_run=True for pipeline testing without API costs.
"""

from .claude_adapter import ClaudeAdapter
from .openai_adapter import OpenAIAdapter
from .gemini_adapter import GeminiAdapter
from .qwen_adapter import QwenAdapter
from .perplexity_adapter import PerplexityAdapter

# Registry for the task dispatcher
AGENT_REGISTRY = {
    "claude": ClaudeAdapter,
    "openai": OpenAIAdapter,
    "gemini": GeminiAdapter,
    "qwen": QwenAdapter,
    "perplexity": PerplexityAdapter,
}

__all__ = [
    "ClaudeAdapter",
    "OpenAIAdapter",
    "GeminiAdapter",
    "QwenAdapter",
    "PerplexityAdapter",
    "AGENT_REGISTRY",
]