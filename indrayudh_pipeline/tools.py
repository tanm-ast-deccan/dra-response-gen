"""
tools.py — Pluggable local tool registry for the agentic loop.

Two kinds of "tools" exist in this pipeline:

  1. WEB SEARCH — handled natively by OpenRouter (see GenParams.web_search /
     provider.py). It is NOT a local tool and needs nothing here.

  2. LOCAL TOOLS — Python functions the model can call via OpenAI-style
     function calling. The runner passes their schemas to the model, executes
     any tool calls locally, and feeds the results back. Register your own
     with @register_tool to extend the pipeline without touching the runner.

Enable tools per run via GenParams.enabled_tools = ["calculator", ...].
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict          # JSON schema for the function arguments
    func: Callable            # func(**arguments) -> str (or JSON-serializable)

    def schema(self) -> dict:
        """OpenAI-compatible tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def run(self, arguments: dict) -> str:
        result = self.func(**(arguments or {}))
        if isinstance(result, str):
            return result
        try:
            return json.dumps(result, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(result)


TOOL_REGISTRY: dict[str, Tool] = {}


def register_tool(name: str, description: str, parameters: dict):
    """Decorator to register a local tool callable in TOOL_REGISTRY."""
    def deco(func: Callable) -> Callable:
        TOOL_REGISTRY[name] = Tool(name, description, parameters, func)
        return func
    return deco


def schemas_for(names: list[str]) -> list[dict]:
    """Return OpenAI tool schemas for the given enabled tool names."""
    schemas = []
    for n in (names or []):
        tool = TOOL_REGISTRY.get(n)
        if tool is None:
            raise KeyError(
                f"Tool '{n}' is not registered. Available: {list(TOOL_REGISTRY)}"
            )
        schemas.append(tool.schema())
    return schemas


def execute(name: str, arguments: dict) -> str:
    """Execute a registered tool by name, returning a string result."""
    tool = TOOL_REGISTRY.get(name)
    if tool is None:
        return f"[tool error] unknown tool: {name}"
    try:
        return tool.run(arguments)
    except Exception as e:  # noqa: BLE001 — tool errors must not crash the loop
        return f"[tool error] {name} failed: {e}"


# ─── Built-in example tool ────────────────────────────────────────────────────
# A safe, offline arithmetic tool. Proves the tool-calling loop end to end and
# serves as a template for real tools. Not enabled unless added to enabled_tools.

@register_tool(
    name="calculator",
    description="Evaluate a basic arithmetic expression and return the result.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Arithmetic expression, e.g. '2 * (3 + 4) / 7'",
            }
        },
        "required": ["expression"],
    },
)
def _calculator(expression: str) -> str:
    import ast
    import operator as op

    ops = {
        ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
        ast.Div: op.truediv, ast.Pow: op.pow, ast.Mod: op.mod,
        ast.USub: op.neg, ast.UAdd: op.pos, ast.FloorDiv: op.floordiv,
    }

    def _eval(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("only numeric constants allowed")
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.operand))
        raise ValueError("unsupported expression")

    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval(tree.body))
    except Exception as e:  # noqa: BLE001
        return f"[calculator error] {e}"
