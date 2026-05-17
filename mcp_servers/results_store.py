"""
results_store.py — Storage backend for evaluation results.

Stores DispatchResults and provides query/retrieval for downstream
scoring, comparison, and reporting. This is the "database" layer
that both the Results MCP Server and direct Python callers use.

Storage is file-based (JSON on disk) for simplicity and portability.
Each DispatchResult is stored as a single JSON file, indexed by task_id.
A lightweight in-memory index supports filtered queries without
loading every file.

Directory layout:
    results_dir/
    ├── index.json                     # lightweight index for queries
    ├── tasks/
    │   ├── TASK-001.json              # full DispatchResult
    │   ├── TASK-002.json
    │   └── ...
    └── scores/
        ├── TASK-001_scores.json       # evaluation scores (future)
        └── ...

Design decisions:
    - JSON files, not a database. Evaluation runs produce dozens of
      results, not millions. File-based storage is inspectable, portable,
      and git-friendly.
    - In-memory index for fast queries. Rebuilt from files on startup.
    - Scores stored separately from results so SMEs can score
      independently without modifying the original agent output.
    - Thread-safe via asyncio locks (one writer at a time).
"""

from __future__ import annotations

import os
import json
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("dra.results_store")


# ─── Index entry (lightweight metadata for queries) ──────────────────────

def _build_index_entry(data: dict) -> dict:
    """
    Extract queryable metadata from a full DispatchResult dict.

    This is what lives in the in-memory index and index.json.
    Kept small so we can load thousands without memory pressure.
    """
    package = data.get("package", {})
    config = data.get("config", {})

    # Per-agent summary
    agent_summary = {}
    for agent_name, results in data.get("agent_results", {}).items():
        passes = []
        for r in results:
            passes.append({
                "completed": r.get("completed", False),
                "error": r.get("error"),
                "response_length": r.get("response_length", 0),
                "citations_count": r.get("citations_count", 0),
                "total_cost_usd": r.get("total_cost_usd", 0),
                "total_duration_sec": r.get("total_duration_sec", 0),
            })
        agent_summary[agent_name] = passes

    return {
        "task_id": data.get("task_id", ""),
        "research_type": package.get("research_type", ""),
        "iat_type": package.get("iat_type", ""),
        "domain": package.get("domain", ""),
        "decision_archetype": package.get("decision_archetype", ""),
        "file_count": package.get("file_count", 0),
        "agents_attempted": data.get("agents_attempted", []),
        "agents_succeeded": data.get("agents_succeeded", []),
        "agents_failed": data.get("agents_failed", []),
        "total_cost_usd": data.get("total_cost_usd", 0),
        "total_duration_sec": data.get("total_duration_sec", 0),
        "dispatched_at": data.get("dispatched_at", ""),
        "completed_at": data.get("completed_at", ""),
        "passes_per_agent": config.get("passes_per_agent", 1),
        "dry_run": config.get("dry_run", False),
        "agent_summary": agent_summary,
        "scored": False,  # updated when scores are attached
    }


# ─── Results Store ───────────────────────────────────────────────────────


def _merge_dispatch_results(existing: dict, new: dict) -> dict:
    """
    Merge two DispatchResult dicts for the same task_id.

    Rules:
    - agent_results: union by agent name; new passes win per agent
    - total_cost_usd: summed (each run incurred real cost)
    - total_duration_sec: max (longest wall-clock time across runs)
    - dispatched_at: min (earliest dispatch timestamp)
    - completed_at: max (latest completion timestamp)
    - agents_attempted/succeeded/failed: recalculated from merged agent_results
    - package/config: prefer non-null values from new, fall back to existing
    """
    merged = dict(existing)  # shallow copy of existing as base

    # Merge agent_results — new run's passes replace existing per agent
    existing_agents = existing.get("agent_results", {})
    new_agents = new.get("agent_results", {})
    merged_agents = dict(existing_agents)
    merged_agents.update(new_agents)  # new agents overwrite existing per key
    merged["agent_results"] = merged_agents

    # Sum costs
    merged["total_cost_usd"] = (
        existing.get("total_cost_usd") or 0
    ) + (new.get("total_cost_usd") or 0)

    # Max duration
    merged["total_duration_sec"] = max(
        existing.get("total_duration_sec") or 0,
        new.get("total_duration_sec") or 0,
    )

    # Earliest dispatch, latest completion
    for ts_field, fn in [("dispatched_at", min), ("completed_at", max)]:
        existing_ts = existing.get(ts_field)
        new_ts = new.get(ts_field)
        if existing_ts and new_ts:
            merged[ts_field] = fn(existing_ts, new_ts)
        else:
            merged[ts_field] = existing_ts or new_ts

    # Recalculate derived agent lists from merged agent_results
    attempted, succeeded, failed, errors = [], [], [], {}
    for agent, passes in merged_agents.items():
        attempted.append(agent)
        pass_list = passes if isinstance(passes, list) else [passes]
        if any(p.get("completed") for p in pass_list):
            succeeded.append(agent)
        else:
            failed.append(agent)
            for p in pass_list:
                if p.get("error"):
                    errors[agent] = p["error"]
                    break
    merged["agents_attempted"] = attempted
    merged["agents_succeeded"] = succeeded
    merged["agents_failed"] = failed
    merged["agent_errors"] = errors

    # Field-level merge for package and config — preserve non-None values
    # from existing when new run has None for that field (e.g. reruns that
    # don't set output_formats will no longer wipe it from the stored result)
    for field in ("package", "config"):
        existing_dict = existing.get(field) or {}
        new_dict = new.get(field) or {}
        merged_dict = dict(existing_dict)
        for k, v in new_dict.items():
            if v is not None:
                merged_dict[k] = v
        merged[field] = merged_dict

    return merged

class ResultsStore:
    """
    File-based storage for evaluation results.

    Usage:
        store = ResultsStore("/path/to/results")
        store.load_index()

        # Store a result
        store.store_result(dispatch_result_dict)

        # Query results
        all_results = store.list_results()
        crp_only = store.query_results(research_type="CRP")
        one_task = store.get_result("TASK-001")

        # Store scores
        store.store_scores("TASK-001", scores_dict)
    """

    def __init__(self, results_dir: str):
        self.results_dir = os.path.abspath(results_dir)
        self.tasks_dir = os.path.join(self.results_dir, "tasks")
        self.scores_dir = os.path.join(self.results_dir, "scores")
        self.index_path = os.path.join(self.results_dir, "index.json")

        self._index: dict[str, dict] = {}  # task_id → index entry
        self._lock = asyncio.Lock()

    def _ensure_dirs(self):
        """Create storage directories if they don't exist."""
        os.makedirs(self.tasks_dir, exist_ok=True)
        os.makedirs(self.scores_dir, exist_ok=True)

    def load_index(self) -> int:
        """
        Load or rebuild the in-memory index.

        If index.json exists and is current, loads from it.
        Otherwise, rebuilds from individual task files.

        Returns the number of indexed results.
        """
        self._ensure_dirs()

        # Try loading existing index
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "r") as f:
                    self._index = json.load(f)
                logger.info("Loaded index: %d results", len(self._index))
                return len(self._index)
            except (json.JSONDecodeError, IOError):
                logger.warning("Index corrupted, rebuilding from files")

        # Rebuild from task files
        return self._rebuild_index()

    def _rebuild_index(self) -> int:
        """Rebuild index by scanning all task JSON files."""
        self._index.clear()

        if not os.path.isdir(self.tasks_dir):
            return 0

        for filename in sorted(os.listdir(self.tasks_dir)):
            if not filename.endswith(".json"):
                continue

            filepath = os.path.join(self.tasks_dir, filename)
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                task_id = data.get("task_id", filename[:-5])
                self._index[task_id] = _build_index_entry(data)

                # Check for scores
                scores_path = os.path.join(
                    self.scores_dir, f"{task_id}_scores.json"
                )
                if os.path.exists(scores_path):
                    self._index[task_id]["scored"] = True

            except Exception as e:
                logger.warning("Failed to index %s: %s", filename, e)

        self._save_index()
        logger.info("Rebuilt index: %d results", len(self._index))
        return len(self._index)

    def _save_index(self):
        """Persist the in-memory index to disk."""
        try:
            with open(self.index_path, "w") as f:
                json.dump(self._index, f, indent=2, default=str)
        except IOError as e:
            logger.error("Failed to save index: %s", e)

    # ─── Write operations ─────────────────────────────────────────

    async def store_result(self, data: dict) -> str:
        """
        Store a DispatchResult (as dict) and update the index.

        The data dict should be the output of dispatch_result_to_dict()
        from task_dispatcher.py.

        Returns the task_id.
        """
        async with self._lock:
            self._ensure_dirs()

            task_id = data.get("task_id", "unknown")

            # Merge with existing result if present (prevents overwrite on reruns)
            filepath = os.path.join(self.tasks_dir, f"{task_id}.json")
            if os.path.exists(filepath):
                try:
                    with open(filepath, "r") as f:
                        existing = json.load(f)
                    data = _merge_dispatch_results(existing, data)
                    logger.info("Merged result for existing task: %s", task_id)
                except Exception as e:
                    logger.warning("Could not merge existing result for %s: %s", task_id, e)

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=str)

            # Update index
            self._index[task_id] = _build_index_entry(data)
            self._save_index()

            logger.info("Stored result: %s", task_id)
            return task_id

    def store_result_sync(self, data: dict) -> str:
        """Synchronous version of store_result (for non-async contexts)."""
        self._ensure_dirs()

        task_id = data.get("task_id", "unknown")

        # Merge with existing result if present (prevents overwrite on reruns)
        filepath = os.path.join(self.tasks_dir, f"{task_id}.json")
        if os.path.exists(filepath):
            try:
                with open(filepath, "r") as f:
                    existing = json.load(f)
                data = _merge_dispatch_results(existing, data)
                logger.info("Merged result for existing task: %s", task_id)
            except Exception as e:
                logger.warning("Could not merge existing result for %s: %s", task_id, e)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        self._index[task_id] = _build_index_entry(data)
        self._save_index()

        logger.info("Stored result: %s", task_id)
        return task_id

    async def store_scores(self, task_id: str, scores: dict) -> bool:
        """
        Store evaluation scores for a task.

        Scores are stored separately from results so the original
        agent output is never modified.

        Expected scores format:
        {
            "task_id": "TASK-001",
            "scored_by": "sme_name",
            "scored_at": "2026-03-05T...",
            "agent_scores": {
                "claude": {
                    "pass_1": {
                        "universal": {...},      # 7 universal criteria, 0-3 each
                        "type_specific": {...},  # 4-5 type-specific criteria
                        "total_score": 2.5,
                        "notes": "..."
                    }
                },
                ...
            },
            "golden_response": "claude",    # or "none"
            "golden_justification": "...",
            "comparative": {               # 5 trade-off dimensions
                "instruction_following_vs_business_value": {...},
                ...
            }
        }
        """
        async with self._lock:
            self._ensure_dirs()

            scores["task_id"] = task_id
            scores.setdefault("scored_at", datetime.now(timezone.utc).isoformat())

            filepath = os.path.join(self.scores_dir, f"{task_id}_scores.json")
            with open(filepath, "w") as f:
                json.dump(scores, f, indent=2, default=str)

            # Update index
            if task_id in self._index:
                self._index[task_id]["scored"] = True
                self._save_index()

            logger.info("Stored scores: %s", task_id)
            return True

    # ─── Read operations ──────────────────────────────────────────

    def get_result(self, task_id: str) -> Optional[dict]:
        """
        Retrieve the full DispatchResult for a task.

        Returns None if not found.
        """
        filepath = os.path.join(self.tasks_dir, f"{task_id}.json")
        if not os.path.exists(filepath):
            return None

        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to read %s: %s", task_id, e)
            return None

    def get_scores(self, task_id: str) -> Optional[dict]:
        """Retrieve evaluation scores for a task."""
        filepath = os.path.join(self.scores_dir, f"{task_id}_scores.json")
        if not os.path.exists(filepath):
            return None

        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to read scores for %s: %s", task_id, e)
            return None

    def list_results(self) -> list[dict]:
        """
        List all stored results (index entries only, not full data).

        Returns list of lightweight index entries sorted by dispatched_at.
        """
        entries = list(self._index.values())
        entries.sort(
            key=lambda e: e.get("dispatched_at", ""),
            reverse=True,
        )
        return entries

    def query_results(
        self,
        research_type: Optional[str] = None,
        iat_type: Optional[str] = None,
        domain: Optional[str] = None,
        agent: Optional[str] = None,
        scored_only: bool = False,
        unscored_only: bool = False,
    ) -> list[dict]:
        """
        Query results with filters.

        All filters are AND-combined. Returns matching index entries.
        """
        results = []
        for entry in self._index.values():
            # Apply filters
            if research_type and entry.get("research_type") != research_type:
                continue
            if iat_type and entry.get("iat_type") != iat_type:
                continue
            if domain and entry.get("domain") != domain:
                continue
            if agent and agent not in entry.get("agents_attempted", []):
                continue
            if scored_only and not entry.get("scored"):
                continue
            if unscored_only and entry.get("scored"):
                continue
            results.append(entry)

        results.sort(
            key=lambda e: e.get("dispatched_at", ""),
            reverse=True,
        )
        return results

    def get_comparison(self, task_id: str) -> Optional[dict]:
        """
        Get a side-by-side comparison of all agent results for a task.

        Returns a dict with:
          - task metadata
          - per-agent: best response text, citations, cost, duration
          - scores (if available)
        """
        full_result = self.get_result(task_id)
        if full_result is None:
            return None

        scores = self.get_scores(task_id)

        comparison = {
            "task_id": task_id,
            "research_type": full_result.get("package", {}).get("research_type", ""),
            "iat_type": full_result.get("package", {}).get("iat_type", ""),
            "domain": full_result.get("package", {}).get("domain", ""),
            "prompt": full_result.get("package", {}).get("prompt", ""),
            "agents": {},
        }

        for agent_name, results in full_result.get("agent_results", {}).items():
            if not results:
                continue

            # Pick the best pass (longest completed response)
            completed = [
                r for r in results
                if r.get("completed") and not r.get("error")
            ]
            best = max(
                completed or results,
                key=lambda r: r.get("response_length", 0),
            )

            agent_data = {
                "model": best.get("model", ""),
                "completed": best.get("completed", False),
                "error": best.get("error"),
                "response_length": best.get("response_length", 0),
                "response_preview": best.get("response_text", "")[:500],
                "citations_count": best.get("citations_count", 0),
                "total_cost_usd": best.get("total_cost_usd", 0),
                "total_duration_sec": best.get("total_duration_sec", 0),
                "iterations": best.get("iterations", 0),
                "passes_run": len(results),
                "passes_succeeded": len(completed),
            }

            # Attach scores if available
            if scores:
                agent_scores = scores.get("agent_scores", {}).get(agent_name)
                if agent_scores:
                    agent_data["scores"] = agent_scores

            comparison["agents"][agent_name] = agent_data

        # Attach golden response selection
        if scores:
            comparison["golden_response"] = scores.get("golden_response", "")
            comparison["golden_justification"] = scores.get("golden_justification", "")

        return comparison

    # ─── Aggregate stats ──────────────────────────────────────────

    def get_stats(self) -> dict:
        """
        Aggregate statistics across all stored results.

        Useful for dashboard/reporting.
        """
        total = len(self._index)
        scored = sum(1 for e in self._index.values() if e.get("scored"))
        total_cost = sum(
            e.get("total_cost_usd", 0) for e in self._index.values()
        )

        # Breakdowns
        by_type = {}
        by_iat = {}
        by_domain = {}
        by_agent = {}

        for entry in self._index.values():
            rt = entry.get("research_type", "unset")
            by_type[rt] = by_type.get(rt, 0) + 1

            iat = entry.get("iat_type", "unset")
            by_iat[iat] = by_iat.get(iat, 0) + 1

            dom = entry.get("domain", "unset")
            by_domain[dom] = by_domain.get(dom, 0) + 1

            for p in entry.get("agents_attempted", []):
                by_agent[p] = by_agent.get(p, 0) + 1

        return {
            "total_results": total,
            "scored_results": scored,
            "unscored_results": total - scored,
            "total_cost_usd": round(total_cost, 2),
            "by_research_type": by_type,
            "by_iat_type": by_iat,
            "by_domain": by_domain,
            "by_agent": by_agent,
        }

    @property
    def count(self) -> int:
        return len(self._index)

    def __repr__(self) -> str:
        scored = sum(1 for e in self._index.values() if e.get("scored"))
        return (
            f"ResultsStore(dir={self.results_dir!r}, "
            f"results={len(self._index)}, scored={scored})"
        )