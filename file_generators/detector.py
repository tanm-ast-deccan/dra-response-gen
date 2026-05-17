"""
detector.py — Regex-based output format detection for DRA prompts.

Detects when a prompt explicitly requires the agent to produce a file
as the deliverable (xlsx, docx, or pptx), as opposed to merely referencing
input files of those types.

Design principles
-----------------
1. Sentence-level matching: the output verb and format keyword must appear
   in the same sentence (no cross-sentence matching via [^.!?\\n]{0,120}).
   This eliminates false positives from "use X.xlsx to extract..." patterns.

2. Verb-gated for xlsx and pptx: requires an explicit output verb
   (present, output, create, produce, generate, deliver, build) before the
   format keyword. "excel" and "powerpoint" are the target keywords — not
   file extensions — because input file references use extensions (.xlsx,
   .pptx) while output instructions use the full word.

3. Simpler rule for docx: "word doc" or "word document" is short and
   semantically unambiguous in the evaluation context. No verb required.

Validated against the DRA benchmark CSV (10 rows, 4 true positives):
    task_id  tsk_260217210354676WR0MM  → ["xlsx"]  PASS
    task_id  tsk_260217210442744VSCW4  → ["xlsx"]  PASS
    task_id  tsk_260217210530999EJHE7  → ["docx"]  PASS
    task_id  tsk_260217210442744SUL2C  → ["pptx"]  PASS
    (all 6 text-only rows)             → []       PASS
Result: 0 false positives, 0 false negatives.

Limitations
-----------
If a future prompt writes "produce a results.xlsx file" without the word
"excel" (using only the extension), this detector will miss it. That edge
case should be handled by adding an LLM fallback, not by relaxing these
patterns (which would reintroduce false positives).
"""

import re
from typing import Optional


# ── Pattern definitions ───────────────────────────────────────────────────────

# Output verbs that distinguish "produce this file" from "read this file"
_OUTPUT_VERBS = r'(?:present|output|create|produce|generate|deliver|build)'

# [^.!?\n]{0,120}: stay within one sentence (period / ! / ? / newline as
# sentence boundary). 120 chars is wide enough to catch verbose instructions
# like "Create an output 2X2 matrix in an excel named..." (57 chars between
# verb and format keyword).

FORMAT_PATTERNS: dict[str, re.Pattern] = {
    "xlsx": re.compile(
        rf'(?i)\b{_OUTPUT_VERBS}\b[^.!?\n]{{0,120}}\bexcel\b'
    ),
    "docx": re.compile(
        r'(?i)\bword\s+doc(?:ument)?\b'
    ),
    "pptx": re.compile(
        rf'(?i)\b{_OUTPUT_VERBS}\b[^.!?\n]{{0,120}}\bpowerpoint\b'
    ),
}

# Human-readable library name for each format (used in adapter instructions)
FORMAT_LIBRARY: dict[str, str] = {
    "xlsx": "openpyxl",
    "docx": "python-docx (from docx import Document)",
    "pptx": "python-pptx (from pptx import Presentation)",
}

# MIME types for file saving
FORMAT_MIME: dict[str, str] = {
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


# ── Public API ────────────────────────────────────────────────────────────────

def detect_output_formats(prompt: str) -> list[str]:
    """
    Return a sorted list of file formats the prompt requires as output.

    Args:
        prompt: the raw SME prompt text (any length, may include file refs)

    Returns:
        Subset of ["docx", "pptx", "xlsx"] — alphabetically sorted.
        Empty list if no file output is required.

    Examples:
        >>> detect_output_formats("Present the final results in Excel format.")
        ['xlsx']
        >>> detect_output_formats("Produce a Word document deliverable.")
        ['docx']
        >>> detect_output_formats("Use Data Dump 1.xlsx to extract the data.")
        []
        >>> detect_output_formats("Use FlipAmaz Ordering Model.pptx to understand...")
        []
    """
    detected = []
    for fmt, pattern in FORMAT_PATTERNS.items():
        if pattern.search(prompt):
            detected.append(fmt)
    return sorted(detected)


def get_format_library(fmt: str) -> str:
    """Return the Python library name/import for a given format."""
    return FORMAT_LIBRARY.get(fmt, "unknown")


def get_format_mime(fmt: str) -> str:
    """Return the MIME type for a given format."""
    return FORMAT_MIME.get(fmt, "application/octet-stream")