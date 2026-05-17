"""
mcp_servers/ — MCP protocol servers for the evaluation framework.

Two servers, one for each direction of data flow:

    corpus_server.py   — Serves evaluation files TO agents (Tier 2 input)
    results_server.py  — Collects agent outputs FROM the dispatcher (output)

Shared modules:
    corpus_tools.py    — File access functions (list/search/fetch)
    results_store.py   — JSON-on-disk storage with in-memory index

Usage:
    # Corpus server (serves files to agents)
    python -m mcp_servers.corpus_server --corpus-dir /path/to/files --port 9400

    # Results server (collects agent outputs)
    python -m mcp_servers.results_server --results-dir /path/to/results --port 9401

    # Direct Python usage
    from mcp_servers.corpus_tools import Corpus
    from mcp_servers.results_store import ResultsStore
"""

from .corpus_tools import Corpus, CorpusDocument, SearchResult
from .corpus_server import MCPHandler as CorpusMCPHandler
from .corpus_server import create_app as create_corpus_app
from .corpus_server import StandaloneCorpusServer
from .results_store import ResultsStore
from .results_server import ResultsMCPHandler
from .results_server import create_app as create_results_app
from .results_server import StandaloneResultsServer
