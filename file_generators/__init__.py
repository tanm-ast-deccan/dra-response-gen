"""
file_generators — Output format detection for DRA evaluation prompts.

This package exposes a single public function:

    detect_output_formats(prompt: str) -> list[str]

Returns a list of format strings (e.g. ["xlsx"], ["docx", "pptx"]) that
the prompt explicitly requires as output deliverables, or [] if none.

Detection is regex-based (sentence-level, verb-gated) with zero false
positives and zero false negatives on the current benchmark CSV.
"""

from .detector import detect_output_formats

__all__ = ["detect_output_formats"]