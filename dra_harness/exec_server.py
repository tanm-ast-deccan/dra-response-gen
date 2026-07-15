"""
exec_server.py — MCP server for agent execution tools.

Complements corpus_server.py (which serves files TO the agent) by
providing the tools the agent uses to DO work:

    python_execute  — run Python code in a subprocess
    bash_execute    — run shell commands
    write_file      — create output files
    web_search      — DuckDuckGo search (no API key)
    web_fetch       — fetch URL content
    calculator      — safe arithmetic evaluation

Architecture:
    ┌──────────────────────────────────────────────────────────────┐
    │  runner.py (MCP client)                                      │
    │     │                                                        │
    │     ├── corpus_server (port 9400) ← file access (existing)  │
    │     │     list_documents / search / fetch                    │
    │     │                                                        │
    │     ├── exec_server (stdio or port 9402) ← THIS FILE        │
    │     │     python_execute / bash_execute / write_file         │
    │     │     web_search / web_fetch / calculator                │
    │     │                                                        │
    │     └── results_server (port 9401) ← post-run storage       │
    └──────────────────────────────────────────────────────────────┘

Transports:
    stdio  — started as subprocess by runner.py (default, recommended)
    SSE    — standalone HTTP server for shared/remote use

Usage:
    # stdio (runner starts this automatically)
    python exec_server.py

    # stdio with custom staging dir
    INDRAYUDH_STAGING_DIR=/tmp/eval python exec_server.py

    # List available tools
    python exec_server.py --list-tools

    # SSE mode on custom port
    python exec_server.py --transport sse --port 9402

Dependencies:
    pip install fastmcp duckduckgo-search
"""

from __future__ import annotations

import os
import sys
import re
import ast
import json
import subprocess
import tempfile
import logging
import operator as op
import argparse
from typing import Optional

from fastmcp import FastMCP

logger = logging.getLogger("dra.exec_server")

STAGING_DIR = os.environ.get("INDRAYUDH_STAGING_DIR", os.getcwd())

mcp = FastMCP(
    name="dra-exec-tools",
    instructions=(
        "Execution tools for deep research analysis. Use python_execute "
        "for calculations and data processing, bash_execute for shell "
        "operations, web_search for external information, web_fetch to "
        "read web pages, write_file for producing deliverables, and "
        "calculator for quick arithmetic."
    ),
)


# ═══════════════════════════════════════════════════════════════════════════
#  TOOLS
# ═══════════════════════════════════════════════════════════════════════════


@mcp.tool()
def python_execute(code: str) -> str:
    """Execute Python code and return stdout/stderr.

    Use for calculations, data analysis, file processing, chart generation,
    and creating output files. Libraries available: pandas, numpy, scipy,
    openpyxl (xlsx), python-docx (docx), python-pptx (pptx), matplotlib,
    pdfplumber, json, csv, os. The working directory contains the task
    input files. To create output files (xlsx, docx, pptx, pdf), use the
    appropriate library and save to the current directory. Print results
    to stdout to see them.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", dir=STAGING_DIR, delete=False
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=STAGING_DIR,
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        return (output.strip() or "[no output]")[:15000]
    except subprocess.TimeoutExpired:
        return "[error] execution timed out (120s)"
    except Exception as e:
        return f"[error] {e}"
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@mcp.tool()
def bash_execute(command: str) -> str:
    """Execute a shell command and return stdout/stderr.

    Use for: file operations (mv, cp, ls), installing packages
    (pip install), running scripts, or any system command.
    The working directory is the task staging folder.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=STAGING_DIR,
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        return (output.strip() or "[no output]")[:15000]
    except subprocess.TimeoutExpired:
        return "[error] execution timed out (120s)"
    except Exception as e:
        return f"[error] {e}"


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """Write text content to a file in the staging directory.

    Use for: creating reports (.md, .txt), saving analysis results
    (.csv, .json), generating code (.py), or any text output.
    Path is relative to the staging directory.
    """
    if not os.path.isabs(path):
        path = os.path.join(STAGING_DIR, path)

    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"File written: {path} ({os.path.getsize(path):,} bytes)"
    except Exception as e:
        return f"[error] writing {path}: {e}"


@mcp.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo and return top results.

    Use for: finding current data, verifying facts, retrieving
    external information not available in the provided files.
    Returns titles, URLs, and snippets.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return "[error] pip install duckduckgo-search"

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return f"No results for '{query}'"

        parts = []
        for i, r in enumerate(results, 1):
            parts.append(
                f"[{i}] {r.get('title', '')}\n"
                f"    URL: {r.get('href', '')}\n"
                f"    {r.get('body', '')}"
            )
        return "\n\n".join(parts)
    except Exception as e:
        return f"[error] search failed: {e}"


@mcp.tool()
def web_fetch(url: str) -> str:
    """Fetch the text content of a web page.

    Use for: reading full articles/pages found via web_search,
    retrieving data from APIs, downloading publicly available
    documents. Returns content truncated to 30,000 characters.
    HTML tags are stripped for readability.
    """
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read(500_000)

            encoding = "utf-8"
            if "charset=" in content_type:
                encoding = (
                    content_type.split("charset=")[-1].split(";")[0].strip()
                )
            text = raw.decode(encoding, errors="replace")

            if "html" in content_type.lower():
                text = re.sub(
                    r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL
                )
                text = re.sub(
                    r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL
                )
                text = re.sub(r"<[^>]+>", " ", text)
                text = re.sub(r"\s+", " ", text).strip()

            return text[:30000] or "[empty response]"

    except urllib.error.HTTPError as e:
        return f"[error] HTTP {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return f"[error] URL error: {e.reason}"
    except Exception as e:
        return f"[error] fetching {url}: {e}"


@mcp.tool()
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression and return the result.

    Supports: +, -, *, /, **, //, %. Parentheses for grouping.
    Example: '2 * (3 + 4) / 7' → '2.0'
    """
    ops = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Pow: op.pow,
        ast.Mod: op.mod,
        ast.USub: op.neg,
        ast.UAdd: op.pos,
        ast.FloorDiv: op.floordiv,
    }

    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(
            node.value, (int, float)
        ):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.operand))
        raise ValueError("unsupported expression")

    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval(tree.body))
    except Exception as e:
        return f"[error] {e}"


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MCP Execution Server — agent tools for DRA"
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="MCP transport (default: stdio)",
    )
    parser.add_argument(
        "--port", type=int, default=9402, help="Port for SSE mode"
    )
    parser.add_argument(
        "--list-tools", action="store_true", help="Print tools and exit"
    )
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )

    if args.list_tools:
        for tool in mcp._tool_manager.tools.values():
            print(f"  {tool.name}: {tool.description[:80]}")
        sys.exit(0)

    if args.transport == "sse":
        mcp.run(transport="sse", port=args.port)
    else:
        mcp.run(transport="stdio")