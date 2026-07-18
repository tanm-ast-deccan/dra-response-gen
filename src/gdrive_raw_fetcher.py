# src/gdrive_raw_fetcher.py
"""
Raw Google Drive fetcher for the AUDITOR.

Unlike gdrive_fetcher.fetch_gdrive_folder (which pre-summarizes large files with
an LLM before injection), this returns the FULL extracted text of every file,
with NO summarization. The auditor's provenance / source-checking layer needs the
literal numbers from the source (e.g. "170", "₹3,90,00,000"); an LLM summary would
destroy them and produce false found_in_source results. It is also faster and
costs no extra LLM calls.

Returns ("", []) on any failure — the audit then proceeds with provenance flagged
as not-checked, exactly as when no Drive link is present.

Per-file text is capped (MAX_CHARS_PER_FILE) only to bound the audit prompt size;
the cap is applied as head+tail truncation with a clear marker, never as a summary.
"""

import logging
import tempfile
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger("dra.gdrive_raw_fetcher")

# Bound per-file text so a giant workbook can't blow the audit prompt. This is a
# truncation (head+tail), NOT a summary — load-bearing numbers usually sit in the
# head (summary tables) and tail (appendices/footnotes).
MAX_CHARS_PER_FILE = 40_000


def fetch_gdrive_folder_raw(folder_url: str) -> Tuple[str, List[str]]:
    """Download all files from a GDrive reference and return (combined_raw_text,
    filenames) with full extraction and no summarization."""
    if not folder_url or not folder_url.strip():
        return "", []

    try:
        from src.file_resolver import FileResolver, parse_gdrive_reference
    except ImportError as e:
        logger.error("file_resolver not available: %s", e)
        return "", []
    try:
        from src.document_parser import read_document
    except ImportError as e:
        logger.error("document_parser not available: %s", e)
        return "", []

    gdrive_info = parse_gdrive_reference(folder_url)
    if gdrive_info is None or gdrive_info.get("type") not in ("folder", "file", "workspace_doc"):
        logger.warning("Not a usable GDrive reference: %s", folder_url)
        return "", []

    with tempfile.TemporaryDirectory(prefix="dra_audit_gdrive_") as staging_dir:
        resolver = FileResolver(staging_dir=staging_dir)
        try:
            local_paths = resolver.resolve([folder_url])
        except Exception as e:
            logger.error("GDrive resolution failed for %s: %s", folder_url, e)
            return "", []

        if not local_paths:
            return "", []

        sections: List[str] = []
        names: List[str] = []
        for local_path in local_paths:
            filename = Path(local_path).name
            try:
                raw = read_document(local_path)
            except NotImplementedError:
                logger.warning("Unsupported file type, skipping: %s", filename)
                continue
            except Exception as e:
                logger.error("Extraction failed for %s: %s", filename, e)
                continue
            if not raw or not raw.strip():
                continue

            if len(raw) > MAX_CHARS_PER_FILE:
                head = int(MAX_CHARS_PER_FILE * 0.7)
                tail = MAX_CHARS_PER_FILE - head
                omitted = len(raw) - MAX_CHARS_PER_FILE
                raw = (raw[:head]
                       + f"\n\n[... {omitted:,} chars truncated (NOT summarized) ...]\n\n"
                       + raw[-tail:])

            sections.append(f"### File: {filename}\n{raw.strip()}")
            names.append(filename)

    if not sections:
        return "", []
    return "\n\n".join(sections), names
