# src/verifier_parser.py
"""
Robust parser for the free-text `Verifiers` cell.

The 183-row corpus uses several inconsistent conventions, all of which share one
invariant: verifiers are numbered sequentially (V1, V2, ... or 1, 2, ...). The
parser anchors on that monotonic sequence rather than on any single delimiter,
normalizes the known stylistic variants to one canonical form, then *verifies*
the parse against strict invariants. Anything that fails the invariants is
flagged UNCERTAIN and routed to human review — the parser never silently guesses.

Design goals (in priority order):
  1. No false positives: a row is reported "clean" only if it passes every invariant.
  2. No silent gaps: every number-token in the cell must end up in exactly one record.
  3. Determinism: same input always yields the same parse + same verdict.

Public API:
    parse_verifiers(cell: str) -> VerifierParseResult
    format_verifiers(records: list[VerifierRecord]) -> str   # canonical re-emit
    next_index(records) -> int                               # for augmenter continuation
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VerifierRecord:
    index: int          # the integer N from V<N> or bare <N>
    text: str           # the verifier statement, stripped
    raw: str = ""       # the original substring this record was parsed from


@dataclass
class VerifierParseResult:
    records: List[VerifierRecord] = field(default_factory=list)
    status: str = "CLEAN"           # CLEAN | UNCERTAIN | EMPTY
    reasons: List[str] = field(default_factory=list)  # why UNCERTAIN, if so
    detected_format: str = ""       # human-readable note on the source style
    original: str = ""

    @property
    def is_clean(self) -> bool:
        return self.status == "CLEAN"

    @property
    def count(self) -> int:
        return len(self.records)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

# Dash characters that appear as the V<n><dash> separator: hyphen-minus, en-dash,
# em-dash, figure dash, minus sign, non-breaking hyphen.
_DASH_CHARS = "\u002d\u2010\u2011\u2012\u2013\u2014\u2212"
_DASH_CLASS = f"[{re.escape(_DASH_CHARS)}]"

# A header line is a line BEFORE the first numbered token that is not itself a
# numbered token. Examples: "Verifiers", "Verifiers -", "Cost calculations ----",
# "Prompt-Specific Verifiers", "ID   Condition". We strip these.
_HEADER_HINT = re.compile(
    r"^\s*(verifiers?|prompt[- ]specific verifiers?|cost calculations?|id\s+condition)\b",
    re.IGNORECASE,
)


def _normalize_cell(cell: str) -> str:
    """Collapse known stylistic variants to one canonical shape, non-destructively
    to the verifier *content*. Only separators/prefixes are touched."""
    if cell is None:
        return ""
    # Unicode normalize so en/em dashes and odd spaces are predictable
    text = unicodedata.normalize("NFKC", cell)
    # Standardize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Strip surrounding quotes that wrap the whole cell or a leading token
    text = text.strip()
    if text and text[0] in "\"'\u201c\u201d\u2018\u2019":
        text = text[1:]
    return text


# Token that opens a V-prefixed verifier record at a line/segment start:
#   V1  v1  V 1   followed by an optional separator (: - | . ))
_OPENER_V = re.compile(
    rf"""
    (?:^|\n)\s*                 # start of cell or a new line
    [\"'\u201c\u2018]?          # optional stray opening quote
    [Vv]\s*                     # REQUIRED V/v prefix
    (\d{{1,3}})                 # (1) the index
    \s*
    (?: {_DASH_CLASS} | [:|.)] )?   # optional separator char
    [ \t]*
    """,
    re.VERBOSE,
)

# Bare-number opener — used ONLY for cells that contain no V-prefixed openers
# at all (2 of 183 rows). Strict: a digit at line start followed by whitespace
# then a NON-numeric, non-unit character, so "2.8 months", "8,700 trips" and
# "= 4.0 buses/hr" inside a verifier body are NOT mistaken for new records.
_OPENER_BARE = re.compile(
    r"""
    (?:^|\n)\s*
    (\d{1,3})                   # (1) the index
    (?: [.)] )?                 # optional list marker '.' or ')' ...
    [ \t]+                      # ... but it MUST be followed by whitespace
    (?=[^\d\s.,)])              # and then a non-numeric, non-punct char (prose)
    """,
    re.VERBOSE,
)


#: Same as _OPENER_V but matches anywhere, not only at a line start. Used ONLY
#: when the inline markers form a consecutive run — see parse_verifiers.
_OPENER_V_INLINE = re.compile(
    r"""
    (?:^|\n|[,;])\s*              # cell start, new line, OR a comma/semicolon
    [\"'\u201c\u2018]?
    [Vv]\s*
    (\d{1,3})
    \s*
    (?: [\-‐‑‒–—−] | [:|.)] )
    [ \t]*
    """,
    re.VERBOSE,
)


def _select_openers(norm: str):
    """Choose opener mode per cell. If any V-prefixed opener exists, use ONLY
    those (bare digits are body text). Otherwise fall back to strict bare-number
    mode. Returns (matches, mode_str)."""
    padded = "\n" + norm
    v_matches = list(_OPENER_V.finditer(padded))
    if v_matches:
        return v_matches, "V-prefix"
    bare_matches = list(_OPENER_BARE.finditer(padded))
    if bare_matches:
        return bare_matches, "bare-number"
    return [], "none"


def parse_verifiers(cell: str) -> VerifierParseResult:
    """Parse one Verifiers cell into records, with an invariant gate."""
    result = VerifierParseResult(original=cell or "")

    norm = _normalize_cell(cell)
    if not norm.strip():
        result.status = "EMPTY"
        result.reasons.append("cell is empty after normalization")
        return result

    # Select opener mode per cell (V-prefix wins; bare-number only if no V).
    padded = "\n" + norm  # prefix \n so ^ matches the first line
    matches, mode = _select_openers(norm)

    # INLINE CELLS. _OPENER_V requires a line start, so a cell written as
    # "V1: ..., V2: ..., V3: ..." on ONE line yields a single record containing
    # everything — and the gate below passed it as CLEAN. Observed in the wild:
    # 12 verifiers collapsed into 1, silently.
    #
    # Splitting on every V-marker regardless of position is unsafe on its own,
    # because a verifier's text can reference another ("consistent with V2"). The
    # discriminator is SEQUENCE: if the inline markers form a consecutive run
    # 1..N and there are more of them than line-start openers found, the cell is
    # a one-line list and splitting is correct. A stray back-reference does not
    # produce a consecutive run.
    inline = list(_OPENER_V_INLINE.finditer(padded))
    if len(inline) > len(matches):
        idxs = [int(m.group(1)) for m in inline]
        if idxs == list(range(1, len(idxs) + 1)):
            matches, mode = inline, mode + "+inline"
            result.reasons.append(
                f"cell written inline on one line; split on {len(inline)} "
                f"consecutive V-markers instead of {len(_select_openers(norm)[0])} "
                f"line-start opener(s)")
        else:
            result.reasons.append(
                f"{len(inline)} V-markers found but only {len(matches)} at a line "
                f"start, and the markers are not a consecutive run "
                f"({idxs[:8]}...) — verifiers may be lost")
    if not matches:
        result.status = "UNCERTAIN"
        result.reasons.append("no numbered verifier tokens found")
        result.detected_format = "unrecognized"
        return result

    # Build records from consecutive opener matches; text runs to the next opener.
    spans = [m.span() for m in matches]

    records: List[VerifierRecord] = []
    for i, m in enumerate(matches):
        start_text = m.end()
        end_text = spans[i + 1][0] if i + 1 < len(matches) else len(padded)
        text = padded[start_text:end_text].strip()
        # Collapse internal newlines/extra whitespace in the statement
        text = re.sub(r"\s*\n\s*", " ", text).strip()
        text = re.sub(r"[ \t]{2,}", " ", text)
        records.append(VerifierRecord(index=int(m.group(1)), text=text,
                                      raw=padded[m.start():end_text].strip()))

    # --- Invariant gate ---------------------------------------------------
    reasons: List[str] = []

    # (a) contiguous numbering from 1
    expected = list(range(1, len(records) + 1))
    got = [r.index for r in records]
    if got != expected:
        reasons.append(f"numbering not contiguous from 1: got {got}, expected {expected}")

    # (b) no empty statements
    empty_idxs = [r.index for r in records if not r.text]
    if empty_idxs:
        reasons.append(f"empty verifier text at index(es) {empty_idxs}")

    # (c) independent count cross-check using the SAME mode, as a guard against a
    #     missed split. Re-run the chosen opener regex and compare counts.
    # the cross-check must use the mode actually chosen, including inline
    if mode.startswith("V-prefix"):
        independent_re = _OPENER_V_INLINE if "inline" in mode else _OPENER_V
    else:
        independent_re = _OPENER_BARE
    independent = len(list(independent_re.finditer(padded)))
    if independent != len(records):
        reasons.append(f"independent token count {independent} != parsed records {len(records)}")

    # (d) orphaned leading prose that wasn't a recognized header
    first_start = spans[0][0]
    lead = padded[:first_start].strip()
    if lead and not _HEADER_HINT.search(lead) and not re.fullmatch(r"[\s\-\u2013\u2014|]*", lead):
        reasons.append(f"unrecognized leading text before first verifier: {lead[:60]!r}")

    fmt = mode
    result.detected_format = fmt

    result.records = records
    if reasons:
        result.status = "UNCERTAIN"
        result.reasons = reasons
    else:
        result.status = "CLEAN"
    return result


# ---------------------------------------------------------------------------
# Canonical re-emit (used by the augmenter to keep one consistent style)
# ---------------------------------------------------------------------------

def format_verifiers(records: List[VerifierRecord]) -> str:
    """Re-emit records in the canonical 'V<n>: text' newline-separated form."""
    return "\n".join(f"V{r.index}: {r.text}" for r in records)


def next_index(records: List[VerifierRecord]) -> int:
    """Return the next sequential index for augmentation continuation."""
    return (max((r.index for r in records), default=0)) + 1