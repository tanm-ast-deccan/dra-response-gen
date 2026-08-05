# src/input_corpus.py
"""
Two views of a task's input files, because code and the model need different things.

MEASURED ON A REAL TASK (tsk_1079445371, a 21.8 MB annual report):
    extracted text          3,137,117 chars   (~784,000 tokens)
    old 40k-per-file cap       84,756 chars   -> the auditor saw 2.7%
    provenance               20 of 26 findable values found  (77%)
    excerpt window             23,735 chars   (~5,900 tokens, 0.8%)
    provenance with excerpts 26 of 26 findable values found  (100%)

The model cannot receive 784k tokens at any cap, so raising the cap was never the
answer. The two needs are simply different:

  full_text    every character, searched IN CODE by _number_appears and by the
               leakage scan. Costs no tokens and must never be truncated.
  prompt_view  a file inventory plus a window around each number the solution
               logic declares. Enough to judge whether a value was read or
               computed, at under 1% of the size.

WHY EXCERPTS AND NOT A SEARCH TOOL
    A search tool would let the auditor look things up, but the audit would stop
    being reproducible — the same task could take different paths on different
    runs, which is the property the whole derivation design protects. Excerpts are
    built deterministically from the solution logic, so the same inputs always
    produce the same prompt.

THE CACHE
    Extraction of a 21.8 MB pdf is slow and identical every run. Text is cached on
    disk keyed by (filename, size), under DRA_CACHE. Point that at whatever the
    agent harness uses and both phases read the SAME text, so a value the agent
    legitimately found cannot be flagged unsourced by the auditor.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.arithmetic_verifier import parse_number

DEFAULT_CACHE = os.environ.get("DRA_CACHE", ".dra_cache")
EXCERPT_RADIUS = 500
MAX_PROMPT_CHARS = 120_000          # ~30k tokens; the whole corpus can be millions
MAX_EXCERPTS = 200
MIN_ANCHOR_VALUE = 100              # below this a number matches almost anywhere


@dataclass
class InputCorpus:
    #: Every character. Searched in code; NEVER put in a prompt.
    full_text: str = ""
    #: Inventory plus excerpts. Bounded, deterministic, safe to prompt with.
    prompt_view: str = ""
    files: List[dict] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    n_excerpts: int = 0
    prompt_chars: int = 0
    full_chars: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        # the full text is the point of this object but has no place in a record
        d.pop("full_text", None)
        d.pop("prompt_view", None)
        return d


def _cache_path(cache_dir: str, name: str, size: int) -> str:
    h = hashlib.sha256(f"{name}:{size}".encode()).hexdigest()[:20]
    return os.path.join(cache_dir, "text", f"{h}.txt")


def extract_cached(path: str, cache_dir: str = DEFAULT_CACHE
                   ) -> Tuple[Optional[str], bool]:
    """(text, was_cached). None means extraction is not possible for this type."""
    from src.document_parser import read_document

    name, size = Path(path).name, os.path.getsize(path)
    cp = _cache_path(cache_dir, name, size)
    if os.path.exists(cp):
        try:
            with open(cp, encoding="utf-8") as f:
                return f.read(), True
        except Exception:
            pass
    try:
        raw = read_document(path) or ""
    except NotImplementedError:
        return None, False
    except Exception:
        return None, False
    os.makedirs(os.path.dirname(cp), exist_ok=True)
    try:
        with open(cp, "w", encoding="utf-8") as f:
            f.write(raw)
    except Exception:
        pass
    return raw, False


def anchor_values(text: str, limit: int = MAX_EXCERPTS) -> List[float]:
    """Numbers the solution logic declares — the anchors an excerpt is built around.

    Deterministic and available BEFORE call 1, which matters: the model has not
    extracted its claims yet, so the excerpts cannot be built from them. Numbers
    under MIN_ANCHOR_VALUE and bare years are dropped because they match almost
    anywhere and would carry meaningless windows.
    """
    out, seen = [], set()
    for tok in re.findall(r"\d[\d,]*\.?\d*", text or ""):
        v = parse_number(tok)
        if v is None or abs(v) < MIN_ANCHOR_VALUE:
            continue
        if 1900 <= v <= 2100 and float(v).is_integer():
            continue
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
        if len(out) >= limit:
            break
    return out


def _renderings(v: float) -> List[str]:
    """The forms a number is plausibly written in, widest match first."""
    forms = [f"{v:,.2f}", f"{v:,.1f}", f"{v:.2f}", f"{v:.1f}"]
    if float(v).is_integer():
        forms += [f"{int(v):,}", str(int(v))]
    forms.append(f"{v:g}")
    seen, out = set(), []
    for f in forms:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def _windows(value: float, sections: List[Tuple[str, str]], radius: int
             ) -> List[Tuple[str, int, str]]:
    """(filename, offset, text) for the first hit of this value in each file."""
    hits = []
    for name, text in sections:
        for form in _renderings(value):
            i = text.find(form)
            if i >= 0:
                hits.append((name, i,
                             text[max(0, i - radius): i + len(form) + radius]))
                break
    return hits


def build_prompt_view(sections: List[Tuple[str, str]], anchors: List[float],
                      radius: int = EXCERPT_RADIUS,
                      max_chars: int = MAX_PROMPT_CHARS) -> Tuple[str, int]:
    """Inventory + one excerpt per anchor. Returns (text, n_excerpts).

    Says plainly that this is a partial view and that provenance is checked
    against the whole file, so the model does not treat an absent value as
    evidence of anything.
    """
    inv = ["## INPUT FILES — INVENTORY",
           "(You are shown an INDEX plus excerpts, not the whole text: these files",
           " total far more than any context window holds. Whether a value appears",
           " in a file is checked IN CODE against the COMPLETE text, so do not",
           " conclude a value is absent because it is not in an excerpt below.)",
           ""]
    for name, text in sections:
        head = " ".join((text[:300] or "").split())
        inv.append(f"- {name}: {len(text):,} chars.  starts: {head[:160]}")
    parts = ["\n".join(inv), ""]
    used = sum(len(p) for p in parts)
    n = 0

    parts.append("## EXCERPTS AROUND EACH FIGURE THE SOLUTION LOGIC STATES")
    for v in anchors:
        for name, off, win in _windows(v, sections, radius):
            block = (f"\n--- {name} @ char {off:,} — context for "
                     f"{v:,.6g}".rstrip("0").rstrip(".") + " ---\n"
                     + " ".join(win.split()))
            if used + len(block) > max_chars:
                parts.append(f"\n[excerpt budget reached after {n} excerpt(s); "
                             f"{len(anchors) - n} figure(s) not shown. Code still "
                             f"searches the complete text.]")
                return "\n".join(parts), n
            parts.append(block)
            used += len(block)
            n += 1
    if n == 0:
        parts.append("(no figure in the solution logic was located in any file)")
    return "\n".join(parts), n


def build_corpus(local_paths: List[str], anchor_text: str,
                 cache_dir: str = DEFAULT_CACHE,
                 radius: int = EXCERPT_RADIUS,
                 max_prompt_chars: int = MAX_PROMPT_CHARS) -> InputCorpus:
    """Build both views from already-resolved local file paths."""
    c = InputCorpus()
    sections: List[Tuple[str, str]] = []
    for p in local_paths:
        name = Path(p).name
        try:
            size = os.path.getsize(p)
        except OSError:
            c.skipped.append(name)
            continue
        text, hit = extract_cached(p, cache_dir)
        if text is None or not text.strip():
            c.skipped.append(name)
            c.files.append({"name": name, "bytes": size, "chars": 0,
                            "extracted": False})
            continue
        c.cache_hits += int(hit)
        c.cache_misses += int(not hit)
        c.files.append({"name": name, "bytes": size, "chars": len(text),
                        "extracted": True, "from_cache": hit})
        sections.append((name, text))

    # full text: complete, for code only
    c.full_text = "\n\n".join(f"### File: {n}\n{t.strip()}" for n, t in sections)
    c.full_chars = len(c.full_text)

    anchors = anchor_values(anchor_text)
    c.prompt_view, c.n_excerpts = build_prompt_view(
        sections, anchors, radius, max_prompt_chars)
    c.prompt_chars = len(c.prompt_view)
    return c


def build_corpus_from_drive(drive_link: str, anchor_text: str,
                            staging_dir: Optional[str] = None,
                            cache_dir: str = DEFAULT_CACHE,
                            **kw) -> InputCorpus:
    """Resolve a Drive reference, then build both views.

    staging_dir defaults under the cache rather than a fresh temp dir, so a 21.8 MB
    pdf is downloaded once instead of on every run — and pointing DRA_CACHE at the
    agent harness's directory makes both phases read identical text.
    """
    from src.file_resolver import FileResolver

    if not (drive_link or "").strip():
        return InputCorpus()
    staging = staging_dir or os.path.join(cache_dir, "staging")
    os.makedirs(staging, exist_ok=True)
    try:
        paths = FileResolver(staging_dir=staging).resolve([drive_link])
    except Exception:
        return InputCorpus()
    return build_corpus(paths or [], anchor_text, cache_dir=cache_dir, **kw)