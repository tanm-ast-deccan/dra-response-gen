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
    │ Perplexity   │ Search-native      │ MINIMAL: report +    │
    │              │ (synchronous)      │ citation URLs only   │
    └──────────────┴────────────────────┴──────────────────────┘

All adapters support dry_run=True for pipeline testing without API costs.
"""

from .claude_adapter import ClaudeAdapter
from .openai_adapter import OpenAIAdapter
from .gemini_adapter import GeminiAdapter
from .perplexity_adapter import PerplexityAdapter

# Registry for the task dispatcher
AGENT_REGISTRY = {
    "claude": ClaudeAdapter,
    "openai": OpenAIAdapter,
    "gemini": GeminiAdapter,
    "perplexity": PerplexityAdapter,
}

__all__ = [
    "ClaudeAdapter",
    "OpenAIAdapter",
    "GeminiAdapter",
    "PerplexityAdapter",
    "AGENT_REGISTRY",
]
