"""
env_loader.py — Load environment variables from .env file.

Call load_env() at the top of any entry point to load API keys
and configuration from a .env file. Safe to call multiple times
(only loads once).

Usage:
    from env_loader import load_env
    load_env()  # loads .env from project root

Expected .env file:
    # ─── Agent API Keys ──────────────────────────────────
    ANTHROPIC_API_KEY=sk-ant-...
    OPENAI_API_KEY=sk-...
    GOOGLE_API_KEY=...
    PERPLEXITY_API_KEY=pplx-...

    # ─── Google Drive Access ─────────────────────────────
    # Option A: Service account (server/automated use)
    GOOGLE_SERVICE_ACCOUNT_KEY=/path/to/service-account.json
    # Option B: OAuth (interactive/personal use)
    # GOOGLE_CLIENT_SECRETS=/path/to/client_secrets.json

    # ─── Optional Overrides ──────────────────────────────
    # DRA_DEFAULT_MODEL=claude-opus-4-6
    # DRA_MAX_COST_USD=15.0
    # DRA_TIMEOUT_SECONDS=900
    # DRA_RESULTS_DIR=./results
    # DRA_STAGING_DIR=./staging

Dependencies:
    pip install python-dotenv
"""

from __future__ import annotations

import os
import logging

logger = logging.getLogger("dra.env")

_loaded = False


def load_env(env_path: str | None = None) -> dict[str, str]:
    """
    Load environment variables from a .env file.

    Search order for .env:
      1. Explicit env_path argument
      2. .env in the current working directory
      3. .env in the same directory as this file (project root)

    Returns a dict of variables that were loaded (not all env vars,
    just the ones from the .env file). Returns empty dict if no
    .env file found or dotenv not installed.

    Safe to call multiple times — only loads on first call.
    """
    global _loaded
    if _loaded:
        return {}

    # Find .env file
    candidates = []
    if env_path:
        candidates.append(env_path)
    candidates.append(os.path.join(os.getcwd(), ".env"))
    candidates.append(os.path.join(os.path.dirname(__file__), ".env"))

    dotenv_path = None
    for path in candidates:
        if os.path.isfile(path):
            dotenv_path = path
            break

    if dotenv_path is None:
        logger.debug("No .env file found (searched: %s)", candidates)
        _loaded = True
        return {}

    # Load with python-dotenv if available
    try:
        from dotenv import dotenv_values, load_dotenv as _load_dotenv

        # Load into os.environ
        _load_dotenv(dotenv_path, override=False)

        # Also return what was loaded for inspection
        loaded = dotenv_values(dotenv_path)
        _loaded = True

        # Log what we found (keys only, not values)
        key_names = [k for k in loaded if loaded[k]]
        logger.info("Loaded .env from %s (%d vars: %s)",
                     dotenv_path, len(key_names), ", ".join(key_names))
        return dict(loaded)

    except ImportError:
        # Fallback: manual parsing (basic .env format)
        logger.info("python-dotenv not installed, using basic .env parser")
        loaded = _basic_parse(dotenv_path)
        _loaded = True
        return loaded


def _basic_parse(dotenv_path: str) -> dict[str, str]:
    """
    Minimal .env parser for when python-dotenv isn't installed.

    Handles:
      - KEY=value
      - KEY="quoted value"
      - KEY='quoted value'
      - # comments
      - blank lines
      - export KEY=value (bash-compatible)

    Does NOT handle:
      - multiline values
      - variable interpolation (${VAR})
    """
    loaded = {}

    with open(dotenv_path, "r") as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Strip optional 'export ' prefix
            if line.startswith("export "):
                line = line[7:]

            # Split on first '='
            if "=" not in line:
                continue

            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()

            # Remove surrounding quotes
            if len(value) >= 2:
                if (value[0] == '"' and value[-1] == '"') or \
                   (value[0] == "'" and value[-1] == "'"):
                    value = value[1:-1]

            # Skip empty keys
            if not key:
                continue

            # Only set if not already in environment (don't override)
            if key not in os.environ:
                os.environ[key] = value

            loaded[key] = value

    key_names = [k for k in loaded if loaded[k]]
    logger.info("Loaded .env from %s (%d vars: %s)",
                 dotenv_path, len(key_names), ", ".join(key_names))
    return loaded


def require_keys(*keys: str) -> dict[str, str]:
    """
    Check that required environment variables are set.

    Call after load_env() to verify the keys you need are present.
    Raises RuntimeError listing all missing keys.

    Usage:
        load_env()
        require_keys("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
    """
    missing = [k for k in keys if not os.environ.get(k)]
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            f"Set them in your .env file or export them in your shell."
        )
    return {k: os.environ[k] for k in keys}


def get_env(key: str, default: str = "") -> str:
    """Get an environment variable with a default."""
    return os.environ.get(key, default)