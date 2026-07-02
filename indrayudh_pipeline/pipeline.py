"""
pipeline.py — Orchestration: CSV → tasks → concurrent runs → results JSON.

    load_packages()  (csv_loader)        one PromptPackage per CSV row
        → fan out to providers x passes   one Task each
        → run_task() concurrently         one RunResult each
        → aggregate + save                results JSON

Concurrency is bounded by PipelineConfig.max_concurrent. One failing run never
affects the others.
"""

from __future__ import annotations

import os
import json
import asyncio
import logging
from datetime import datetime, timezone

try:
    from .config import PipelineConfig, load_env
    from .csv_loader import load_packages
    from .provider import OpenRouterDriver, resolve_slug
    from .runner import run_task
    from .models import Task
except ImportError:  # run as a script from inside the package dir
    from config import PipelineConfig, load_env
    from csv_loader import load_packages
    from provider import OpenRouterDriver, resolve_slug
    from runner import run_task
    from models import Task

logger = logging.getLogger("indrayudh.pipeline")


def build_tasks(packages, cfg: PipelineConfig) -> list[Task]:
    """Fan out each package to every provider and pass."""
    tasks: list[Task] = []
    for pkg in packages:
        files_dir = os.path.join(cfg.output_dir, "files", pkg.task_id)
        for provider in cfg.providers:
            slug = resolve_slug(provider, cfg.model_for(provider))
            for p in range(1, cfg.passes_per_provider + 1):
                tasks.append(Task(
                    task_id=pkg.task_id,
                    prompt=pkg.prompt,
                    provider=provider,
                    model_slug=slug,
                    pass_index=p,
                    file_paths=list(pkg.file_paths),
                    output_formats=list(pkg.output_formats),
                    output_dir=files_dir,
                    drive_url=pkg.drive_url,
                    sme_name=pkg.sme_name,
                ))
    return tasks


async def run_batch(
    csv_path: str,
    cfg: PipelineConfig,
    max_rows: int | None = None,
    task_ids: list[str] | None = None,
) -> dict:
    """Run the full pipeline for a CSV and return a results dict."""
    load_env()

    packages = load_packages(
        csv_path,
        resolve_files=cfg.resolve_files,
        staging_dir=cfg.staging_dir,
        max_rows=max_rows,
        task_ids=task_ids,
    )
    logger.info("Loaded %d package(s)", len(packages))

    tasks = build_tasks(packages, cfg)
    logger.info("Built %d run(s): %d package(s) x %d provider(s) x %d pass(es)",
                len(tasks), len(packages), len(cfg.providers), cfg.passes_per_provider)

    driver = OpenRouterDriver(
        api_key=cfg.api_key, base_url=cfg.base_url, dry_run=cfg.dry_run
    )

    # Per-provider params resolved once.
    params_by_provider = {p: cfg.params_for(p) for p in cfg.providers}

    sem = asyncio.Semaphore(cfg.max_concurrent)
    started = datetime.now(timezone.utc)

    async def _run(task: Task):
        async with sem:
            return await run_task(task, params_by_provider[task.provider], driver)

    results = await asyncio.gather(*[_run(t) for t in tasks], return_exceptions=True)

    # Normalize any unexpected exceptions into a record.
    run_dicts = []
    for task, res in zip(tasks, results):
        if isinstance(res, Exception):
            logger.error("[%s] unhandled: %s", task.run_id, res)
            run_dicts.append({
                "task_id": task.task_id, "run_id": task.run_id,
                "provider": task.provider, "model": task.model_slug,
                "pass_index": task.pass_index, "completed": False,
                "error": str(res),
            })
        else:
            run_dicts.append(res.to_dict())

    completed_at = datetime.now(timezone.utc)
    output = {
        "csv": csv_path,
        "started_at": started.isoformat(),
        "completed_at": completed_at.isoformat(),
        "duration_sec": (completed_at - started).total_seconds(),
        "config": cfg.to_dict(),
        "summary": _summarize(run_dicts, cfg),
        "results": run_dicts,
    }
    return output


def _summarize(run_dicts: list[dict], cfg: PipelineConfig) -> dict:
    by_provider: dict = {}
    total_cost = 0.0
    succeeded = 0
    for r in run_dicts:
        prov = r.get("provider", "?")
        slot = by_provider.setdefault(prov, {"runs": 0, "succeeded": 0, "cost": 0.0})
        slot["runs"] += 1
        if r.get("completed"):
            slot["succeeded"] += 1
            succeeded += 1
        c = r.get("total_cost_usd", 0.0) or 0.0
        slot["cost"] = round(slot["cost"] + c, 6)
        total_cost += c
    return {
        "total_runs": len(run_dicts),
        "succeeded": succeeded,
        "failed": len(run_dicts) - succeeded,
        "total_cost_usd": round(total_cost, 6),
        "by_provider": by_provider,
    }


def save_results(output: dict, path: str | None = None) -> str:
    """Write the results dict to JSON and return the path."""
    out_dir = output["config"].get("output_dir", "./results")
    if path is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out_dir, f"results_{ts}.json")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Saved results → %s", path)
    return path
