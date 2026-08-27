# src/ingest_folders.py
"""
Ingest the Drive/disk folder shapes this benchmark actually ships in, and hand
score_task a normalized (augmented_package, response) pair per (task, model, run).

THREE folder shapes are ingested, matching what is on Drive:

1. AUGMENT / GOLDEN folder  (one folder for the whole batch)
   Holds the crystallized packages, one per task-run:
       {task_id}__{ref_model}__run{N}.json          e.g. tsk_4520782367__hy3__run1.json
   Each package carries, inline:
       run.run_label                 <- AUTHORITATIVE label; filenames can be scrambled
       history.input_files_link      <- the task input folder
       hy_incorrect_response.output_files / run_folder_link
       hy_correct_response.golden_* + scoring_summary
       crux ids / dag / depths / crux_shapley_weights / expected_values
   (Also contains proposal .docx/.pdf and a .DS_Store — ignored.)

2. INPUT folder  (one per task)
   Flat folder of the task's input files (the .txt/.xlsx the agent was given).
   Linked from history.input_files_link; also present in the batch CSV Drive Link.

3. OUTPUT run folders  (per task, per model, per run)
   Where each model's actual deliverables live. The package's output_files[].url
   points at the graded deliverable(s); a run may also ship convenience/scratch
   files that score_task's content classifier must reject.

The RUN LABEL RULE is enforced here: the label comes from run.run_label inside the
JSON, never from the filename. On task 6 the filenames were shuffled relative to the
embedded labels; trusting the filename mis-assigns runs. ingest returns the embedded
label and flags any run where filename and embedded label disagree.

Nothing here calls an LLM. Drive fetching is pluggable: pass a `fetch_json(file_id)`
and `list_folder(folder_id)` (e.g. the repo gdrive_fetcher) or point at a local
mirror directory. If neither is given, only local paths are read.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any

PKG_RE = re.compile(r"^(?P<task>tsk_[A-Za-z0-9]+)__(?P<ref>[A-Za-z0-9]+)__run(?P<run>\d+)\.json$")
_DRIVE_ID_RE = re.compile(r"/(?:file/d|folders)/([A-Za-z0-9_-]+)|[?&]id=([A-Za-z0-9_-]+)")


def drive_id(url_or_id: str) -> str:
    """Extract a Drive file/folder id from a URL, or return the input unchanged if
    it already looks like a bare id."""
    if not url_or_id:
        return ""
    m = _DRIVE_ID_RE.search(url_or_id)
    if m:
        return m.group(1) or m.group(2)
    return url_or_id.strip()


def run_label_num(label: str) -> Optional[int]:
    """'Run 3 (crash)' -> 3 ; 'Run 1 (Alpha)' -> 1 ; None if unparseable."""
    if not label:
        return None
    m = re.search(r"run\s*(\d+)", label, re.I)
    return int(m.group(1)) if m else None


@dataclass
class RunPackage:
    """One normalized task-run ready for score_task."""
    task_id: str
    ref_model: str                    # e.g. 'hy3'
    filename: str
    file_run_num: Optional[int]       # run number in the FILENAME
    embedded_label: str               # run.run_label — authoritative
    embedded_run_num: Optional[int]   # run number parsed from the label
    label_mismatch: bool              # filename run != embedded run
    input_files_link: str = ""
    input_folder_id: str = ""
    output_files: List[dict] = field(default_factory=list)   # [{filename,url,id}]
    run_folder_link: str = ""
    golden: dict = field(default_factory=dict)               # full crystallized JSON
    problems: List[str] = field(default_factory=list)

    def to_augmented(self) -> dict:
        """Shape the crystallized JSON into the dict score_task.score_task expects.
        Pulls crux ids / dag / depths / shapley / expected / neglog / verifier text
        from wherever the crystallizer stored them, tolerating a couple of key
        spellings seen across versions."""
        g = self.golden
        def pick(*keys, default=None):
            for k in keys:
                if k in g and g[k] not in (None, "", {}, []):
                    return g[k]
            # also look one level down under common containers
            for container in ("augment", "augmented", "scoring", "package"):
                sub = g.get(container) or {}
                for k in keys:
                    if isinstance(sub, dict) and sub.get(k) not in (None, "", {}, []):
                        return sub[k]
            return default
        return {
            "task_id": self.task_id,
            "crux_verifier_ids": pick("crux_ids", "crux_verifier_ids", default=[]),
            "crux_shapley_weights_json": pick("crux_shapley_weights",
                                              "crux_shapley_weights_json", default={}),
            "expected_values_json": pick("expected_values", "expected_values_json",
                                         default={}),
            "augmented_verifiers": pick("augmented_verifiers", default=""),
            "crux_dag_json": pick("crux_dag", "dag", "crux_dag_json", default={}),
            "depths_json": pick("crux_depth", "depths", "depths_json", default={}),
            "neglog_shapley_json": pick("neglog_crux", "neglog_shapley",
                                        "neglog_shapley_json", default={}),
            "scoring_summary": pick("scoring_summary", default={}),
        }

    def to_response(self, model: str, output_paths: Optional[List[str]] = None) -> dict:
        """The response dict score_task expects. output_paths overrides the graded
        model's local deliverable paths; otherwise the package's own output_files
        (the reference run) are used."""
        files = (output_paths if output_paths is not None
                 else [f.get("filename") or f.get("url") for f in self.output_files])
        return {
            "task_id": self.task_id,
            "model": model,
            "run_id": self.embedded_label,
            "pass_index": self.embedded_run_num or self.file_run_num or 0,
            "output_files": files,
        }


def _load_json(fetch_json, local_dir, file_id, filename) -> Optional[dict]:
    if local_dir:
        p = os.path.join(local_dir, filename)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    if fetch_json and file_id:
        return fetch_json(file_id)
    return None


def ingest_augment_folder(
    folder_id: str = "",
    *,
    list_folder: Optional[Callable[[str], List[dict]]] = None,
    fetch_json: Optional[Callable[[str], dict]] = None,
    local_dir: str = "",
) -> List[RunPackage]:
    """Read every {task}__{ref}__run{N}.json in the augment/golden folder and return
    normalized RunPackages, with the run-label rule enforced.

    Provide EITHER a local_dir mirror of the folder, OR list_folder+fetch_json
    callables backed by the Drive connector / repo gdrive_fetcher.
    """
    entries: List[dict] = []
    if local_dir:
        for fn in sorted(os.listdir(local_dir)):
            if PKG_RE.match(fn):
                entries.append({"id": "", "title": fn})
    elif list_folder:
        for f in list_folder(folder_id):
            if PKG_RE.match(f.get("title", "")):
                entries.append({"id": f.get("id", ""), "title": f["title"]})
    else:
        raise ValueError("provide local_dir or list_folder")

    out: List[RunPackage] = []
    for e in entries:
        fn = e["title"]
        m = PKG_RE.match(fn)
        g = _load_json(fetch_json, local_dir, e["id"], fn)
        if g is None:
            out.append(RunPackage(
                task_id=m.group("task"), ref_model=m.group("ref"), filename=fn,
                file_run_num=int(m.group("run")), embedded_label="",
                embedded_run_num=None, label_mismatch=False,
                problems=["could not load package JSON"]))
            continue

        run = g.get("run", {}) or {}
        label = run.get("run_label", "") or ""
        emb_num = run_label_num(label)
        file_num = int(m.group("run"))
        hist = g.get("history", {}) or {}
        resp = (g.get("hy_incorrect_response") or g.get("hy_correct_response") or {})
        ofiles = []
        for of in (resp.get("output_files") or []):
            ofiles.append({
                "filename": of.get("filename", ""),
                "url": of.get("url", ""),
                "id": drive_id(of.get("url", "")),
            })

        pkg = RunPackage(
            task_id=m.group("task"),
            ref_model=m.group("ref"),
            filename=fn,
            file_run_num=file_num,
            embedded_label=label,
            embedded_run_num=emb_num,
            label_mismatch=(emb_num is not None and emb_num != file_num),
            input_files_link=hist.get("input_files_link", ""),
            input_folder_id=drive_id(hist.get("input_files_link", "")),
            output_files=ofiles,
            run_folder_link=resp.get("run_folder_link") or "",
            golden=g,
        )
        if pkg.label_mismatch:
            pkg.problems.append(
                f"RUN-LABEL MISMATCH: filename says run{file_num} but embedded "
                f"run_label is '{label}' (run {emb_num}). Using embedded label.")
        out.append(pkg)
    return out


def group_by_task(pkgs: List[RunPackage]) -> Dict[str, List[RunPackage]]:
    d: Dict[str, List[RunPackage]] = {}
    for p in pkgs:
        d.setdefault(p.task_id, []).append(p)
    for v in d.values():
        v.sort(key=lambda p: p.embedded_run_num or p.file_run_num or 0)
    return d


def resolve_output_run_folder(
    task_id: str, model: str, run_label: str,
    *,
    list_folder: Callable[[str], List[dict]],
    output_root_id: str,
) -> List[dict]:
    """Find the deliverable files for a given (task, model, run) under an output
    root. Convention observed in this corpus: files are named with the task id and
    a model tag; the graded deliverable is the named one, convenience files carry
    __p<N>. or <model>_tsk_<id>_response.* patterns. Returns candidate file dicts;
    score_task's content classifier makes the final graded/rejected call."""
    want_run = run_label_num(run_label)
    hits = []
    for f in list_folder(output_root_id):
        t = f.get("title", "")
        if task_id.split("_")[-1] not in t and task_id not in t:
            continue
        if model.split("_")[0] not in t.lower() and model not in t:
            continue
        hits.append({"id": f.get("id", ""), "title": t, "url": f.get("viewUrl", "")})
    return hits
