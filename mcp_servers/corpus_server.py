"""
corpus_server.py — MCP-protocol server for Tier 2 corpus file access.

This server exposes evaluation corpus files to agents via the Model
Context Protocol (MCP). It provides three tools:
  - list_documents: enumerate available files with metadata
  - search: keyword search across all documents
  - fetch: retrieve full content of a specific document by ID

Transports:
  - SSE (Server-Sent Events) over HTTP — standard MCP transport
  - Compatible with OpenAI's native MCP client for deep research
  - Also usable by any MCP-compatible client (Claude Code, Cursor, etc.)

Architecture:
    ┌──────────────────────────────────────────────────────────────┐
    │                    Corpus MCP Server                         │
    │                                                              │
    │  ┌──────────────┐     ┌───────────────────────────────────┐ │
    │  │  SSE         │     │  Corpus (corpus_tools.py)         │ │
    │  │  Transport   │────▶│  - load files from directory      │ │
    │  │              │     │  - extract text (PDF/DOCX/XLSX)   │ │
    │  │  GET /sse    │     │  - keyword search                 │ │
    │  │  POST /msg   │     │  - fetch by document ID           │ │
    │  └──────────────┘     └───────────────────────────────────┘ │
    │                                                              │
    │  OpenAI connects:  "server_url": "http://host:9400/sse"     │
    │  Claude adapter:   calls corpus tools directly (no MCP)     │
    └──────────────────────────────────────────────────────────────┘

Usage:
    # Start the server
    python corpus_server.py --corpus-dir /path/to/eval/files --port 9400

    # In OpenAI adapter:
    tools=[{
        "type": "mcp",
        "server_label": "eval_corpus",
        "server_url": "http://localhost:9400/sse",
        "require_approval": "never",
    }]

Dependencies:
    pip install starlette uvicorn sse-starlette
    # Optional for document extraction:
    pip install pdfplumber python-docx openpyxl python-pptx

MCP Protocol Reference:
    https://spec.modelcontextprotocol.io/specification/
"""

from __future__ import annotations

import os
import sys
import json
import uuid
import asyncio
import logging
import argparse
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("dra.mcp.corpus")

# ─── MCP Protocol Constants ──────────────────────────────────────────────

JSONRPC_VERSION = "2.0"
MCP_PROTOCOL_VERSION = "2025-03-26"

SERVER_INFO = {
    "name": "dra-corpus-server",
    "version": "1.0.0",
}

SERVER_CAPABILITIES = {
    "tools": {},  # we support tools
}


# ─── Tool definitions (MCP format) ──────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "list_documents",
        "description": (
            "List all documents available in the evaluation corpus. "
            "Returns metadata for each file: ID, filename, type, size, "
            "and estimated token count. Use this first to understand "
            "what data is available before searching or reading."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "search",
        "description": (
            "Search across all documents in the corpus for relevant "
            "content. Returns ranked results with snippets showing "
            "context around matches. Use this to find which documents "
            "contain information relevant to your research question."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (keywords or phrases)",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch",
        "description": (
            "Fetch the full text content of a specific document by its "
            "ID. Use the ID from list_documents or search results. "
            "Returns the complete extracted text of the document."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": (
                        "Document ID (from list_documents or search results) "
                        "or filename"
                    ),
                },
            },
            "required": ["id"],
        },
    },
]


# ─── JSON-RPC message handling ───────────────────────────────────────────

class MCPHandler:
    """
    Handles MCP JSON-RPC messages against a loaded Corpus.

    This class is transport-agnostic — it takes a JSON-RPC request
    dict and returns a JSON-RPC response dict. The transport layer
    (SSE, stdio, etc.) wraps this.
    """

    def __init__(self, corpus):
        self.corpus = corpus
        self._initialized = False

    def handle_message(self, message: dict) -> Optional[dict]:
        """
        Process a JSON-RPC request and return a response.

        Returns None for notifications (no response needed).
        """
        method = message.get("method", "")
        msg_id = message.get("id")
        params = message.get("params", {})

        logger.debug("MCP request: method=%s, id=%s", method, msg_id)

        try:
            if method == "initialize":
                result = self._handle_initialize(params)
            elif method == "notifications/initialized":
                # Client notification — no response needed
                self._initialized = True
                return None
            elif method == "tools/list":
                result = self._handle_tools_list(params)
            elif method == "tools/call":
                result = self._handle_tools_call(params)
            elif method == "ping":
                result = {}
            else:
                return self._error_response(
                    msg_id, -32601, f"Method not found: {method}"
                )

            if msg_id is not None:
                return {
                    "jsonrpc": JSONRPC_VERSION,
                    "id": msg_id,
                    "result": result,
                }
            return None

        except Exception as e:
            logger.error("MCP handler error: %s", e, exc_info=True)
            return self._error_response(msg_id, -32603, str(e))

    def _handle_initialize(self, params: dict) -> dict:
        """Handle the 'initialize' handshake."""
        client_info = params.get("clientInfo", {})
        logger.info(
            "MCP initialize from %s v%s",
            client_info.get("name", "unknown"),
            client_info.get("version", "?"),
        )
        return {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": SERVER_CAPABILITIES,
            "serverInfo": SERVER_INFO,
        }

    def _handle_tools_list(self, params: dict) -> dict:
        """Handle 'tools/list' — return available tool definitions."""
        return {"tools": TOOL_DEFINITIONS}

    def _handle_tools_call(self, params: dict) -> dict:
        """Handle 'tools/call' — execute a tool and return results."""
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        logger.info("Tool call: %s(%s)", tool_name, json.dumps(arguments)[:100])

        if tool_name == "list_documents":
            docs = self.corpus.list_documents()
            content_text = json.dumps(docs, indent=2)

        elif tool_name == "search":
            query = arguments.get("query", "")
            max_results = arguments.get("max_results", 5)
            results = self.corpus.search_corpus(query, max_results=max_results)
            content_text = json.dumps(results, indent=2)

        elif tool_name == "fetch":
            doc_id = arguments.get("id", "")
            result = self.corpus.fetch_document(doc_id)
            content_text = json.dumps(result, indent=2)

        else:
            return {
                "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
                "isError": True,
            }

        return {
            "content": [{"type": "text", "text": content_text}],
        }

    def _error_response(self, msg_id, code: int, message: str) -> dict:
        """Build a JSON-RPC error response."""
        return {
            "jsonrpc": JSONRPC_VERSION,
            "id": msg_id,
            "error": {"code": code, "message": message},
        }


# ─── SSE Transport (Starlette/ASGI) ─────────────────────────────────────

def create_app(corpus_dir: str, file_paths: Optional[list[str]] = None):
    """
    Create an ASGI application implementing MCP over SSE.

    The MCP SSE transport works as follows:
      1. Client connects to GET /sse
      2. Server sends an 'endpoint' event with the POST URL
      3. Client sends JSON-RPC requests via POST to that URL
      4. Server sends JSON-RPC responses via SSE events

    Args:
        corpus_dir: directory containing corpus files
        file_paths: explicit file paths (overrides corpus_dir scan)

    Returns:
        Starlette ASGI application
    """
    try:
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import JSONResponse, Response
        from starlette.routing import Route
        from sse_starlette.sse import EventSourceResponse
    except ImportError:
        raise ImportError(
            "MCP SSE transport requires:\n"
            "  pip install starlette uvicorn sse-starlette\n"
        )

    # ── Load corpus ───────────────────────────────────────────────
    sys.path.insert(0, os.path.dirname(__file__))
    from corpus_tools import Corpus

    corpus = Corpus(corpus_dir)
    if file_paths:
        count = corpus.load_from_paths(file_paths)
    else:
        count = corpus.load()
    logger.info("Corpus loaded: %d documents, ~%d tokens", count, corpus.total_tokens_est)

    handler = MCPHandler(corpus)

    # ── Per-session state ─────────────────────────────────────────
    # Each SSE connection gets a session with its own message queue.
    # This supports multiple concurrent clients.
    sessions: dict[str, asyncio.Queue] = {}

    async def sse_endpoint(request: Request):
        """
        GET /sse — SSE connection endpoint.

        Creates a session and sends the 'endpoint' event telling
        the client where to POST messages.
        """
        session_id = uuid.uuid4().hex[:12]
        queue: asyncio.Queue = asyncio.Queue()
        sessions[session_id] = queue

        logger.info("SSE session opened: %s", session_id)

        async def event_generator():
            # First event: tell client the message endpoint
            endpoint_url = f"/message?session_id={session_id}"
            yield {
                "event": "endpoint",
                "data": endpoint_url,
            }

            # Then stream responses as they come
            try:
                while True:
                    try:
                        msg = await asyncio.wait_for(queue.get(), timeout=30)
                        yield {
                            "event": "message",
                            "data": json.dumps(msg),
                        }
                    except asyncio.TimeoutError:
                        # Send keepalive
                        yield {"event": "ping", "data": ""}
            except asyncio.CancelledError:
                pass
            finally:
                sessions.pop(session_id, None)
                logger.info("SSE session closed: %s", session_id)

        return EventSourceResponse(event_generator())

    async def message_endpoint(request: Request):
        """
        POST /message — Receive JSON-RPC requests from the client.

        Processes the request via MCPHandler and queues the response
        to be sent via the corresponding SSE session.
        """
        session_id = request.query_params.get("session_id", "")
        queue = sessions.get(session_id)

        if queue is None:
            return JSONResponse(
                {"error": "Invalid session_id"},
                status_code=400,
            )

        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                {"error": "Invalid JSON"},
                status_code=400,
            )

        # Handle the message
        response = handler.handle_message(body)
        if response is not None:
            await queue.put(response)

        return Response(status_code=202)  # Accepted

    async def health_endpoint(request: Request):
        """GET /health — Health check with corpus stats."""
        return JSONResponse({
            "status": "ok",
            "server": SERVER_INFO,
            "corpus": corpus.stats,
            "active_sessions": len(sessions),
        })

    async def docs_endpoint(request: Request):
        """GET /docs — Quick reference for the corpus contents."""
        docs = corpus.list_documents()
        return JSONResponse({
            "corpus_dir": corpus.corpus_dir,
            "documents": docs,
            "stats": corpus.stats,
        })

    app = Starlette(
        routes=[
            Route("/sse", sse_endpoint),
            Route("/message", message_endpoint, methods=["POST"]),
            Route("/health", health_endpoint),
            Route("/docs", docs_endpoint),
        ],
    )

    return app


# ─── Standalone HTTP mode (no ASGI framework needed) ─────────────────────

class StandaloneCorpusServer:
    """
    Minimal HTTP server for environments where installing Starlette
    isn't practical. Uses only the stdlib.

    Implements a simplified MCP-over-HTTP protocol:
      - POST /mcp — send JSON-RPC request, get JSON-RPC response
      - GET /health — health check
      - GET /docs — list corpus documents

    This is NOT SSE-based and won't work with OpenAI's native MCP
    client. Use the ASGI app (create_app) for that. This is useful
    for Claude's adapter (which calls tools directly via HTTP) and
    for testing.
    """

    def __init__(self, corpus_dir: str, port: int = 9400):
        sys.path.insert(0, os.path.dirname(__file__))
        from corpus_tools import Corpus

        self.corpus = Corpus(corpus_dir)
        self.corpus.load()
        self.handler = MCPHandler(self.corpus)
        self.port = port

    def run(self):
        """Start the HTTP server (blocking)."""
        from http.server import HTTPServer, BaseHTTPRequestHandler

        corpus = self.corpus
        handler_ref = self.handler

        class RequestHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/health":
                    self._json_response(200, {
                        "status": "ok",
                        "corpus": corpus.stats,
                    })
                elif self.path == "/docs":
                    self._json_response(200, {
                        "documents": corpus.list_documents(),
                    })
                else:
                    self._json_response(404, {"error": "Not found"})

            def do_POST(self):
                if self.path == "/mcp":
                    length = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(length))
                    response = handler_ref.handle_message(body)
                    self._json_response(200, response or {})
                else:
                    self._json_response(404, {"error": "Not found"})

            def _json_response(self, code, data):
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())

            def log_message(self, format, *args):
                logger.debug(format, *args)

        server = HTTPServer(("0.0.0.0", self.port), RequestHandler)
        logger.info(
            "Corpus server running on http://0.0.0.0:%d "
            "(%d documents, ~%d tokens)",
            self.port,
            len(self.corpus.documents),
            self.corpus.total_tokens_est,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Server stopped.")
            server.server_close()


# ─── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MCP Corpus Server — serves evaluation files to agents",
    )
    parser.add_argument(
        "--corpus-dir", "-d", required=True,
        help="Directory containing evaluation corpus files",
    )
    parser.add_argument(
        "--port", "-p", type=int, default=9400,
        help="Port to listen on (default: 9400)",
    )
    parser.add_argument(
        "--mode", choices=["sse", "http"], default="sse",
        help="Transport mode: 'sse' for MCP-over-SSE (OpenAI compatible), "
             "'http' for simple JSON-RPC (testing/Claude)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.mode == "http":
        server = StandaloneCorpusServer(args.corpus_dir, args.port)
        server.run()
    else:
        # SSE mode requires uvicorn + starlette
        try:
            import uvicorn
        except ImportError:
            logger.error(
                "SSE mode requires: pip install uvicorn starlette sse-starlette"
            )
            sys.exit(1)

        app = create_app(args.corpus_dir)
        uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
