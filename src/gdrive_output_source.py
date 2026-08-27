"""
gdrive_output_source.py — let run_score.py pull a task's deliverable files from
a Google Drive FOLDER link stored in the augmented/tasks CSV, instead of (or in
addition to) local staging paths in the results JSON.

Design goals:
  * The scorer (src/score_task.py) is unchanged. It still consumes
    response["output_files"] as LOCAL paths and resolves them on disk. This
    module simply DOWNLOADS the folder to a temp dir and puts those local paths
    into response["output_files"] before score_task runs.
  * Files land in a temp staging dir and are removed after the task is scored
    (per the chosen behaviour). Nothing persists.
  * Reuses the harness's read-only FileResolver/GDriveClient and its
    parse_gdrive_reference, so a folder URL or bare id is handled identically to
    the rest of the pipeline.

Typical use inside run_score.work():

    src = GDriveOutputSource(column="Output Files Drive Link")
    link = src.link_for(aug_row)                 # None if no/blank link
    if link and not resp.get("output_files"):    # prefer explicit local paths
        local = src.stage(link)                  # downloads folder → temp dir
        resp = {**resp, "output_files": local}
    try:
        result = score_task(aug_row, resp, ...)
    finally:
        src.release(link)                         # cleanup that task's temp dir
"""

import logging
from typing import List, Optional

from src.file_resolver import FileResolver, parse_gdrive_reference

logger = logging.getLogger("dra.gdrive_output_source")

# Same set score_task's _scoreable_files accepts; we pre-filter so the temp dir
# only holds gradeable files (Google-native docs, images, etc. are ignored).
SCOREABLE_EXT = (".docx", ".txt", ".pdf", ".xlsx", ".csv", ".md", ".pptx", ".json")


class GDriveOutputSource:
    """Resolves per-task Drive output folders to local files, with cleanup."""

    def __init__(self, column: str = "Output Files Drive Link"):
        self.column = column
        # one resolver per active link so cleanup is scoped to a single task
        self._resolvers: dict = {}

    # -- CSV plumbing -------------------------------------------------------
    def link_for(self, aug_row: dict) -> Optional[str]:
        """Return the Drive folder link for this task, or None if absent/blank.

        Tolerant of header variants: exact column first, then a case/space/
        underscore-insensitive match, so it works whether the CSV kept the
        pretty header ('Output Files Drive Link') or a normalized one
        ('output_files_drive_link').
        """
        if not aug_row:
            return None
        v = aug_row.get(self.column)
        if v is None:
            want = _norm(self.column)
            for k, val in aug_row.items():
                if _norm(k) == want:
                    v = val
                    break
        v = (v or "").strip()
        return v or None

    # -- staging / cleanup --------------------------------------------------
    def stage(self, link: str) -> List[str]:
        """Download the folder behind `link` to a temp dir; return local paths.

        Returns only scoreable files. Empty list if the link is unusable or the
        folder yielded nothing (caller then sees a normal NOT-FOUND from the
        scorer, which is the correct signal).
        """
        info = parse_gdrive_reference(link)
        if not info:
            logger.warning("not a Drive reference, skipping: %s", link)
            return []
        if info["type"] != "folder":
            # A file link would also work via resolve(), but the agreed schema is
            # a per-task FOLDER. Accept a file link too rather than fail hard.
            logger.info("Drive link is a %s, not a folder: %s", info["type"], link)
        resolver = FileResolver()            # fresh temp staging dir
        self._resolvers[link] = resolver
        try:
            paths = resolver.resolve([link])
        except Exception as e:               # noqa: BLE001 - never break scoring
            logger.error("Drive resolve failed for %s: %s", link, e)
            resolver.cleanup()
            self._resolvers.pop(link, None)
            return []
        keep = [p for p in paths if p.lower().endswith(SCOREABLE_EXT)]
        logger.info("staged %d/%d scoreable file(s) from %s",
                    len(keep), len(paths), link)
        return keep

    def release(self, link: Optional[str]):
        """Remove the temp dir for `link` (safe to call with None/unknown)."""
        if not link:
            return
        r = self._resolvers.pop(link, None)
        if r is not None:
            try:
                r.cleanup()
            except Exception as e:           # noqa: BLE001
                logger.warning("cleanup failed for %s: %s", link, e)

    def release_all(self):
        for link in list(self._resolvers):
            self.release(link)


def _norm(s: str) -> str:
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())
