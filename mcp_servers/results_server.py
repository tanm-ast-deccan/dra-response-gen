"""
results_server.py — MCP-protocol server for evaluation results collection.

Stores agent outputs from the Task Dispatcher and serves them to
the scoring pipeline, SME evaluation interface, and reporting tools.

Architecture:
    ┌──────────────────────────────────────────────────────────────┐
    │                 Results Collection Server                     │
    │                                                              │
    │  Writers:                     Readers:                       │
    │  ┌──────────────┐            ┌──────────────────────────┐   │
    │  │ Dispatcher    │──store──▶ │ SME Evaluation Interface  │   │
    │  │ (after run)   │           │ Reporting Pipeline        │   │
    │  └──────────────┘           │ Comparison Queries        │   │
    │  ┌──────────────┐           └──────────────────────────┘   │
    │  │ Scorer        │──score──▶                                │
    │  │ (after eval)  │                                          │
    │  └──────────────┘                                           │
    │                                                              │
    │  Storage: JSON files on disk (inspectable, git-friendly)     │
    │  Index:   In-memory for fast filtered queries                │
    └──────────────────────────────────────────────────────────────┘

MCP Tools:
    store_result      — save a DispatchResult (from dispatcher)
    store_scores      — attach evaluation scores (from scorer/SME)
    get_result        — retrieve full result by task_id
    get_scores        — retrieve scores by task_id
    list_results      — list all results (with metadata)
    query_results     — filtered search (by type, IAT, domain, agent)
    get_comparison    — side-by-side agent comparison for a task
    get_stats         — aggregate statistics across all results

Usage:
    # Start server
    python results_server.py --results-dir /path/to/results --port 9401

    # From dispatcher (Python):
    from mcp_servers.results_store import ResultsStore
    store = ResultsStore("/path/to/results")
    store.load_index()
    store.store_result_sync(dispatch_result_dict)

Dependencies:
    pip install starlette uvicorn sse-starlette  (for SSE mode)
    # No dependencies needed for HTTP mode or direct Python usage
"""

from __future__ import annotations

import os
import sys
import json
import uuid
import asyncio
import logging
import argparse
from typing import Optional

from env_loader import load_env
load_env()

logger = logging.getLogger("dra.mcp.results")

# ─── MCP Protocol Constants ──────────────────────────────────────────────

JSONRPC_VERSION = "2.0"
MCP_PROTOCOL_VERSION = "2025-03-26"

SERVER_INFO = {
    "name": "dra-results-server",
    "version": "1.0.0",
}

SERVER_CAPABILITIES = {
    "tools": {},
}


# ─── Tool definitions ────────────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "store_result",
        "description": (
            "Store a DispatchResult from the task dispatcher. "
            "Expects the full JSON output of dispatch_result_to_dict(). "
            "Returns the stored task_id."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "object",
                    "description": "Full DispatchResult as JSON object",
                },
            },
            "required": ["data"],
        },
    },
    {
        "name": "store_scores",
        "description": (
            "Attach evaluation scores to an existing result. "
            "Scores are stored separately from the agent output. "
            "Includes universal criteria (0-3), type-specific criteria, "
            "golden response selection, and comparative dimensions."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "Task ID to attach scores to",
                },
                "scores": {
                    "type": "object",
                    "description": "Evaluation scores object",
                },
            },
            "required": ["task_id", "scores"],
        },
    },
    {
        "name": "get_result",
        "description": (
            "Retrieve the full DispatchResult for a task by ID. "
            "Returns all agent responses, metadata, and config."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "Task ID to retrieve",
                },
            },
            "required": ["task_id"],
        },
    },
    {
        "name": "get_scores",
        "description": (
            "Retrieve evaluation scores for a task by ID."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "Task ID to retrieve scores for",
                },
            },
            "required": ["task_id"],
        },
    },
    {
        "name": "list_results",
        "description": (
            "List all stored results with lightweight metadata. "
            "Returns task IDs, research types, providers, costs, "
            "and scoring status. Does NOT return full response text."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "query_results",
        "description": (
            "Query results with filters. All filters are AND-combined. "
            "Filter by research_type (CRP/RCP/SCP/LDP/FSP/"
            "Constrained Research Prompt/Relevance Compression Prompt/"
            "Structural Compliance Prompt/Latent Decomposition Prompt/Failure-Sensitive Prompt). "
            "iat_type (IAT-1/IAT-2/IAT-3), domain (Multiple Choices), "
            "agent, or scoring status."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "research_type": {
                    "type": "string",
                    "description": "Filter by research type: CRP, RCP, SCP, LDP, FSP",
                },
                "iat_type": {
                    "type": "string",
                    "description": "Filter by IAT type: IAT-1, IAT-2, IAT-3",
                },
                "domain": {
                    "type": "string",
                    "description": "Filter by domain: Multiple choices",
                },
                "agent": {
                    "type": "string",
                    "description": "Filter by agent_name: claude, openai, gemini, perplexity",
                },
                "scored_only": {
                    "type": "boolean",
                    "description": "Only return scored results",
                    "default": False,
                },
                "unscored_only": {
                    "type": "boolean",
                    "description": "Only return unscored results",
                    "default": False,
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_comparison",
        "description": (
            "Get a side-by-side comparison of all agent results for "
            "a specific task. Shows best response preview, citations, "
            "cost, duration, and scores (if available) per agent."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "Task ID to compare providers for",
                },
            },
            "required": ["task_id"],
        },
    },
    {
        "name": "get_stats",
        "description": (
            "Get aggregate statistics across all stored results. "
            "Returns totals, breakdowns by research type/IAT/domain/"
            "agent, scoring progress, and total cost."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


# ─── MCP Handler ─────────────────────────────────────────────────────────

class ResultsMCPHandler:
    """Handles MCP JSON-RPC messages against the ResultsStore."""

    def __init__(self, store):
        self.store = store

    async def handle_message(self, message: dict) -> Optional[dict]:
        """Process a JSON-RPC request and return a response."""
        method = message.get("method", "")
        msg_id = message.get("id")
        params = message.get("params", {})

        logger.debug("MCP request: method=%s, id=%s", method, msg_id)

        try:
            if method == "initialize":
                result = self._handle_initialize(params)
            elif method == "notifications/initialized":
                return None
            elif method == "tools/list":
                result = {"tools": TOOL_DEFINITIONS}
            elif method == "tools/call":
                result = await self._handle_tools_call(params)
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
            logger.error("Handler error: %s", e, exc_info=True)
            return self._error_response(msg_id, -32603, str(e))

    def _handle_initialize(self, params: dict) -> dict:
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

    async def _handle_tools_call(self, params: dict) -> dict:
        """Route tool calls to the appropriate store method."""
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        logger.info("Tool call: %s", tool_name)

        if tool_name == "store_result":
            data = arguments.get("data", {})
            task_id = await self.store.store_result(data)
            return self._text_result(json.dumps({
                "stored": True, "task_id": task_id,
            }))

        elif tool_name == "store_scores":
            task_id = arguments.get("task_id", "")
            scores = arguments.get("scores", {})
            ok = await self.store.store_scores(task_id, scores)
            return self._text_result(json.dumps({
                "stored": ok, "task_id": task_id,
            }))

        elif tool_name == "get_result":
            task_id = arguments.get("task_id", "")
            result = self.store.get_result(task_id)
            if result is None:
                return self._text_result(json.dumps({
                    "error": f"Task not found: {task_id}",
                }))
            return self._text_result(json.dumps(result, default=str))

        elif tool_name == "get_scores":
            task_id = arguments.get("task_id", "")
            scores = self.store.get_scores(task_id)
            if scores is None:
                return self._text_result(json.dumps({
                    "error": f"Scores not found for: {task_id}",
                }))
            return self._text_result(json.dumps(scores, default=str))

        elif tool_name == "list_results":
            entries = self.store.list_results()
            return self._text_result(json.dumps(entries, default=str))

        elif tool_name == "query_results":
            entries = self.store.query_results(
                research_type=arguments.get("research_type"),
                iat_type=arguments.get("iat_type"),
                domain=arguments.get("domain"),
                agent=arguments.get("agent"),
                scored_only=arguments.get("scored_only", False),
                unscored_only=arguments.get("unscored_only", False),
            )
            return self._text_result(json.dumps(entries, default=str))

        elif tool_name == "get_comparison":
            task_id = arguments.get("task_id", "")
            comparison = self.store.get_comparison(task_id)
            if comparison is None:
                return self._text_result(json.dumps({
                    "error": f"Task not found: {task_id}",
                }))
            return self._text_result(json.dumps(comparison, default=str))

        elif tool_name == "get_stats":
            stats = self.store.get_stats()
            return self._text_result(json.dumps(stats, default=str))

        else:
            return {
                "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
                "isError": True,
            }

    def _text_result(self, text: str) -> dict:
        return {"content": [{"type": "text", "text": text}]}

    def _error_response(self, msg_id, code: int, message: str) -> dict:
        return {
            "jsonrpc": JSONRPC_VERSION,
            "id": msg_id,
            "error": {"code": code, "message": message},
        }


# ─── SSE Transport ───────────────────────────────────────────────────────

def create_app(results_dir: str):
    """Create an ASGI app implementing MCP-over-SSE for results."""
    try:
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import JSONResponse, Response
        from starlette.routing import Route
        from sse_starlette.sse import EventSourceResponse
    except ImportError:
        raise ImportError(
            "SSE transport requires:\n"
            "  pip install starlette uvicorn sse-starlette\n"
        )

    sys.path.insert(0, os.path.dirname(__file__))
    from results_store import ResultsStore

    store = ResultsStore(results_dir)
    count = store.load_index()
    logger.info("Results store loaded: %d results", count)

    handler = ResultsMCPHandler(store)
    sessions: dict[str, asyncio.Queue] = {}

    async def sse_endpoint(request: Request):
        session_id = uuid.uuid4().hex[:12]
        queue: asyncio.Queue = asyncio.Queue()
        sessions[session_id] = queue

        async def event_generator():
            yield {
                "event": "endpoint",
                "data": f"/message?session_id={session_id}",
            }
            try:
                while True:
                    try:
                        msg = await asyncio.wait_for(queue.get(), timeout=30)
                        yield {"event": "message", "data": json.dumps(msg)}
                    except asyncio.TimeoutError:
                        yield {"event": "ping", "data": ""}
            except asyncio.CancelledError:
                pass
            finally:
                sessions.pop(session_id, None)

        return EventSourceResponse(event_generator())

    async def message_endpoint(request: Request):
        session_id = request.query_params.get("session_id", "")
        queue = sessions.get(session_id)
        if queue is None:
            return JSONResponse({"error": "Invalid session"}, status_code=400)

        body = await request.json()
        response = await handler.handle_message(body)
        if response is not None:
            await queue.put(response)

        return Response(status_code=202)

    async def health_endpoint(request: Request):
        return JSONResponse({
            "status": "ok",
            "server": SERVER_INFO,
            "stats": store.get_stats(),
        })

    app = Starlette(routes=[
        Route("/sse", sse_endpoint),
        Route("/message", message_endpoint, methods=["POST"]),
        Route("/health", health_endpoint),
    ])

    return app


# ─── Standalone HTTP mode ────────────────────────────────────────────────

class StandaloneResultsServer:
    """
    Minimal HTTP server for the results collection.

    Routes:
      POST /mcp         — JSON-RPC MCP endpoint
      GET  /health      — server stats
      GET  /results     — list all results
      GET  /result/{id} — get one result
    """

    def __init__(self, results_dir: str, port: int = 9401):
        sys.path.insert(0, os.path.dirname(__file__))
        from results_store import ResultsStore

        self.store = ResultsStore(results_dir)
        self.store.load_index()
        self.handler = ResultsMCPHandler(self.store)
        self.port = port

    def run(self):
        from http.server import HTTPServer, BaseHTTPRequestHandler

        store = self.store
        handler_ref = self.handler
        loop = asyncio.new_event_loop()

        class RequestHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/health":
                    self._json(200, {
                        "status": "ok", "stats": store.get_stats(),
                    })
                elif self.path == "/results":
                    self._json(200, {"results": store.list_results()})
                elif self.path.startswith("/result/"):
                    task_id = self.path.split("/result/")[1]
                    result = store.get_result(task_id)
                    if result:
                        self._json(200, result)
                    else:
                        self._json(404, {"error": "Not found"})
                elif self.path.startswith("/comparison/"):
                    task_id = self.path.split("/comparison/")[1]
                    comp = store.get_comparison(task_id)
                    if comp:
                        self._json(200, comp)
                    else:
                        self._json(404, {"error": "Not found"})
                else:
                    self._json(404, {"error": "Not found"})

            def do_POST(self):
                if self.path == "/mcp":
                    length = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(length))
                    response = loop.run_until_complete(
                        handler_ref.handle_message(body)
                    )
                    self._json(200, response or {})
                else:
                    self._json(404, {"error": "Not found"})

            def _json(self, code, data):
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(data, default=str).encode())

            def log_message(self, fmt, *args):
                logger.debug(fmt, *args)

        server = HTTPServer(("0.0.0.0", self.port), RequestHandler)
        logger.info(
            "Results server on http://0.0.0.0:%d (%d results)",
            self.port, store.count,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.server_close()


# ─── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MCP Results Collection Server",
    )
    parser.add_argument(
        "--results-dir", "-d", required=True,
        help="Directory for storing results (created if needed)",
    )
    parser.add_argument("--port", "-p", type=int, default=9401)
    parser.add_argument(
        "--mode", choices=["sse", "http"], default="sse",
    )
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.mode == "http":
        server = StandaloneResultsServer(args.results_dir, args.port)
        server.run()
    else:
        try:
            import uvicorn
        except ImportError:
            logger.error("SSE mode requires: pip install uvicorn starlette sse-starlette")
            sys.exit(1)

        app = create_app(args.results_dir)
        uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
