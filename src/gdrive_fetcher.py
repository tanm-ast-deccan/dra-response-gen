# src/gdrive_fetcher.py
"""
GDrive folder fetcher for LLMaJ cross-validation.

Downloads all files from a Google Drive folder URL, extracts their text via
document_parser, and applies Option C tiered injection logic:

  - Small files  (extracted text < SUMMARIZATION_THRESHOLD chars) → inject in full
  - Large files  (extracted text >= SUMMARIZATION_THRESHOLD chars) → pre-summarize
    via a focused LLM call before injection

The resulting combined text is appended to the Supporting Evidence section of
the scoring prompt, alongside Logic and SC fields.

Authentication:
    Set GOOGLE_SERVICE_ACCOUNT_KEY=/path/to/service-account.json in your .env file.
    The service account must have read access to the shared folder.

Failure behaviour:
    Any error during resolution or extraction is logged and the affected file is
    skipped. The function never raises — callers receive an empty string and an
    empty file list on total failure, which causes the pipeline to proceed without
    GDrive content (scoring continues with Logic/SC only).
"""

import logging
import tempfile
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger("dra.gdrive_fetcher")

# Files larger than this (chars after text extraction) are pre-summarized
SUMMARIZATION_THRESHOLD = 6_000

# Max chars fed into the summarization LLM call (head 80% + tail 20%)
MAX_RAW_CHARS_FOR_SUMMARY = 60_000

# Max tokens for each summarization response
SUMMARY_MAX_TOKENS = 1_000

# Prompt text is truncated to this length when used as summarization context
PROMPT_CONTEXT_MAX_CHARS = 3_000


def fetch_gdrive_folder(
    folder_url: str,
    prompt_text: str,
    model_name: str,
) -> Tuple[str, List[str]]:
    """
    Download all files from a GDrive folder, extract text, apply tiered logic.

    Args:
        folder_url:   Google Drive folder (or file) URL shared with the service account.
        prompt_text:  The prompt being evaluated — used as context for summarization
                      so the LLM knows what to focus on.
        model_name:   LLM model to use for pre-summarization of large files.

    Returns:
        (combined_text, processed_filenames)
        combined_text is an empty string if nothing could be extracted.
        processed_filenames lists all files that were successfully extracted.
    """
    # Lazy imports — avoids hard dependency when GDrive is not used
    try:
        from src.file_resolver import FileResolver, parse_gdrive_reference
    except ImportError as e:
        logger.error(
            "file_resolver not available — install google-api-python-client "
            "google-auth google-auth-oauthlib: %s", e
        )
        return "", []

    try:
        from src.document_parser import read_document
    except ImportError as e:
        logger.error("document_parser not available: %s", e)
        return "", []

    # Validate it's a recognised GDrive reference
    gdrive_info = parse_gdrive_reference(folder_url)
    if gdrive_info is None:
        logger.warning("Not a GDrive reference, skipping: %s", folder_url)
        return "", []

    if gdrive_info["type"] not in ("folder", "file", "workspace_doc"):
        logger.warning("Unrecognised GDrive reference type '%s': %s",
                       gdrive_info["type"], folder_url)
        return "", []

    # Use a temp staging dir — automatically cleaned up after extraction
    with tempfile.TemporaryDirectory(prefix="dra_gdrive_") as staging_dir:
        resolver = FileResolver(staging_dir=staging_dir)
        try:
            local_paths = resolver.resolve([folder_url])
        except RuntimeError as e:
            # Auth failure (no credentials configured)
            logger.error("GDrive auth failed for %s: %s", folder_url, e)
            raise  # Re-raise so the caller can skip this row
        except Exception as e:
            logger.error("GDrive resolution failed for %s: %s", folder_url, e)
            return "", []

        if not local_paths:
            logger.warning("No files resolved from GDrive reference: %s", folder_url)
            return "", []

        file_sections: List[str] = []
        processed_names: List[str] = []

        for local_path in local_paths:
            filename = Path(local_path).name
            section = _extract_single_file(
                local_path=local_path,
                filename=filename,
                prompt_text=prompt_text,
                model_name=model_name,
            )
            if section is not None:
                file_sections.append(section)
                processed_names.append(filename)

    if not file_sections:
        return "", []

    combined = "\n\n".join(file_sections)
    logger.info(
        "GDrive extraction complete: %d file(s), %d total chars",
        len(processed_names), len(combined),
    )
    return combined, processed_names


# =============================================================================
# Internal helpers
# =============================================================================

def _extract_single_file(
    local_path: str,
    filename: str,
    prompt_text: str,
    model_name: str,
) -> str | None:
    """
    Extract text from one downloaded file and apply tiered injection logic.

    Returns a formatted section string, or None if the file should be skipped.
    """
    from src.document_parser import read_document

    try:
        raw_text = read_document(local_path)
    except NotImplementedError:
        logger.warning("Unsupported file type — skipping: %s", filename)
        return None
    except Exception as e:
        logger.error("Text extraction failed for %s: %s", filename, e)
        return None

    if not raw_text.strip():
        logger.warning("Empty text extracted from: %s — skipping", filename)
        return None

    char_count = len(raw_text)

    if char_count < SUMMARIZATION_THRESHOLD:
        # Small file — inject full text
        logger.info("Injecting %s in full (%d chars)", filename, char_count)
        return (
            f"### File: {filename}\n"
            f"{raw_text.strip()}\n"
        )
    else:
        # Large file — pre-summarize
        logger.info(
            "Pre-summarizing %s (%d chars >= threshold %d)",
            filename, char_count, SUMMARIZATION_THRESHOLD,
        )
        summary = _summarize_file(
            raw_text=raw_text,
            filename=filename,
            prompt_text=prompt_text,
            model_name=model_name,
        )
        return (
            f"### File: {filename}  "
            f"[Pre-summarized — original {char_count:,} chars]\n"
            f"{summary}\n"
        )


def _summarize_file(
    raw_text: str,
    filename: str,
    prompt_text: str,
    model_name: str,
) -> str:
    """
    Call the LLM to produce a cross-validation-focused summary of a large file.

    The summary is constrained to aspects relevant to verifying the prompt:
    data fields, calculation steps, constraints, and discrepancies.

    Falls back to head+tail truncation if the LLM call fails.
    """
    from src.prompt_eval_templates import GDRIVE_SUMMARIZATION_TEMPLATE
    from src.prompt_evaluator import _call_llm

    # Truncate raw text before sending to the summarization call
    if len(raw_text) > MAX_RAW_CHARS_FOR_SUMMARY:
        head_chars = int(MAX_RAW_CHARS_FOR_SUMMARY * 0.8)
        tail_chars = MAX_RAW_CHARS_FOR_SUMMARY - head_chars
        omitted = len(raw_text) - MAX_RAW_CHARS_FOR_SUMMARY
        raw_text = (
            raw_text[:head_chars]
            + f"\n\n[... {omitted:,} chars omitted for summarization ...]\n\n"
            + raw_text[-tail_chars:]
        )

    llm_prompt = GDRIVE_SUMMARIZATION_TEMPLATE.format(
        filename=filename,
        prompt_text=prompt_text[:PROMPT_CONTEXT_MAX_CHARS],
        file_content=raw_text,
    )

    try:
        summary = _call_llm(llm_prompt, model_name, max_tokens=SUMMARY_MAX_TOKENS)
        return summary.strip()
    except Exception as e:
        logger.error("Summarization LLM call failed for %s: %s — using truncated excerpt", filename, e)
        # Fallback: head + tail truncation with a clear marker
        head = raw_text[:3_000]
        tail = raw_text[-1_000:] if len(raw_text) > 4_000 else ""
        fallback = head + ("\n\n[... truncated ...]\n\n" + tail if tail else "")
        return f"[Summarization failed — showing truncated excerpt]\n{fallback}"