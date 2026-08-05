# src/verifier_grammar.py
"""
Read the target clause off a verifier's own text.

WHY THIS EXISTS
    A verifier's text already carries its standard — the `toleranced` property
    obliges it to, because a grader reading only the verifier has to know what
    counts as close enough. So `expected_values` is not a second standard; it is a
    machine-readable index over the one in the text. Deriving it means there is a
    single source and nothing to keep in sync.

    Three separate defects came from treating the frozen record as primary: split
    children inheriting one target between three, a negative verifier frozen as a
    positive target for the trap value, and 20 of 31 verifiers unscoreable while
    the report said the task was fine. All of them dissolve when the text is the
    source.

MEASURED BEFORE AND AFTER
    On a real 29-verifier task, 17 of 29 target clauses could be read. The 12
    failures were mostly punctuation drift — a derivation between the value and the
    band, or "to" instead of "=" — which is why the spec now fixes the clause shape
    rather than the prose.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

#: numeric: "= <number> <unit> (+/- <band>)", band immediately after the value
_NUM = re.compile(
    r"=\s*[$\u20b9\u20ac\u00a3]?\s*(?P<val>-?[\d,]+\.?\d*)\s*"
    r"(?P<unit>%|pp|mln|bn|cr\b|crore|lakh|days?|hours?|units?|/kg|/hr|kWh[^\s]*)?"
    r"\s*\(\s*(?:\+/-|\u00b1)\s*(?P<band>[\d,]+\.?\d*)\s*(?P<bunit>pp|%)?\s*\)",
    re.IGNORECASE)

#: numeric with no band stated — a toleranced failure, but still a readable target
_NUM_NOBAND = re.compile(
    r"=\s*[$\u20b9\u20ac\u00a3]?\s*(?P<val>-?[\d,]+\.?\d*)\s*"
    r"(?P<unit>%|pp|mln|bn|cr\b|crore|lakh|days?|hours?|units?|/kg|/hr)?(?!\d)",
    re.IGNORECASE)

#: string: '= "exact"'
_STR = re.compile(r'=\s*"(?P<val>[^"]+)"')

#: decision: '= TOKEN' where TOKEN is upper-case words. Must follow the "=", so a
#: decision word inside prose does not count — a format verifier saying "each with
#: its own explicit FLAG / DO NOT FLAG conclusion" was mis-read as a decision.
#: Real decision tokens include a slash ("AT/ABOVE") and a hyphen ("NO-GO"), so the
#: class must allow both. "= AT/ABOVE" failed to parse when it did not.
_DEC = re.compile(r"=\s*(?P<val>[A-Z][A-Z0-9_/\-. ]{1,38}?)\s*(?:[.;,]|$)")

_FAILIF = re.compile(r"FAIL\s+IF\s+(?P<cond>.+?)(?:$)", re.IGNORECASE | re.DOTALL)

#: A value token: a currency-prefixed or unit-suffixed number. Counting these
#: after the "=" is how a multi-value clause is detected. An earlier single-regex
#: version used [^.;] to stop at sentence boundaries, which meant it could not
#: cross the decimal point in "2.565" and never matched.
_VALUE_TOKEN = re.compile(
    r"[$₹€£]\s*[\d,]+\.?\d*"
    r"|[\d,]+\.?\d*\s*(?:%|pp|mln|bn|cr\b|crore|lakh|days?|hours?|units?)",
    re.IGNORECASE)


def _n_values_after_eq(t: str) -> int:
    """How many value tokens follow the first '='."""
    i = t.find("=")
    if i < 0:
        return 0
    tail = t[i + 1:].split(". ")[0]          # first sentence only
    # the band is part of ONE target, not a second value: "= 6.67% (+/- 0.05pp)"
    # counted as two and wrongly read as multi-value
    tail = re.sub(r"\(\s*(?:\+/-|\u00b1)[^)]*\)", " ", tail)
    return len(_VALUE_TOKEN.findall(tail))


def _f(x: str) -> Optional[float]:
    try:
        return float(str(x).replace(",", ""))
    except (TypeError, ValueError):
        return None


def parse_target(text: str) -> dict:
    """Read one verifier's target clause.

    Returns {kind, value, tol, unit, fail_if, form, note}. kind is numeric,
    decision, string or presence. A presence check has no value by design and is
    not a failure.
    """
    t = (text or "").strip()
    out = {"kind": "presence", "value": None, "tol": 0.0, "unit": "",
           "fail_if": "", "form": "none", "note": ""}

    fi = _FAILIF.search(t)
    if fi:
        out["fail_if"] = " ".join(fi.group("cond").split())
        t = t[:fi.start()].rstrip()

    if "=" not in t:
        out["note"] = "no target clause; presence check"
        return out

    if _n_values_after_eq(t) >= 2 and not _STR.search(t):
        out["note"] = ("several values in one clause: no single target. Split the "
                       "verifier, or write it as a presence check.")
        out["form"] = "multi"
        return out

    m = _STR.search(t)
    if m:
        out.update(kind="string", value=m.group("val"), form="string")
        return out

    m = _NUM.search(t)
    if m:
        unit = (m.group("unit") or "").strip()
        band = _f(m.group("band")) or 0.0
        out.update(kind="numeric", value=_f(m.group("val")), tol=band,
                   unit=unit or (m.group("bunit") or ""), form="numeric")
        return out

    m = _DEC.search(t)
    if m:
        out.update(kind="decision", value=m.group("val").strip(), form="decision")
        return out

    m = _NUM_NOBAND.search(t)
    if m:
        out.update(kind="numeric", value=_f(m.group("val")), tol=0.0,
                   unit=(m.group("unit") or "").strip(), form="numeric_no_band",
                   note="value stated with NO band — a toleranced failure")
        return out

    out["note"] = "an '=' clause that matches no target form"
    out["form"] = "unreadable"
    return out


def derive_expected_values(verifier_text_block: str
                           ) -> Tuple[Dict[str, dict], List[dict]]:
    """{vid: frozen target} from the canonical verifier block, plus problems.

    A presence check yields no entry, which is correct: it has nothing to compare.
    """
    expected, problems = {}, []
    for line in (verifier_text_block or "").splitlines():
        line = line.strip()
        m = re.match(r"(V[\w]+)\s*:\s*(.+)", line)
        if not m:
            continue
        vid, txt = m.group(1), m.group(2)
        p = parse_target(txt)
        if p["kind"] == "presence" or p["value"] is None:
            if p["form"] in ("multi", "unreadable"):
                problems.append({"verifier": vid, "form": p["form"],
                                 "detail": p["note"], "text": txt[:120]})
            continue
        entry = {"value": p["value"], "tol": p["tol"], "unit": p["unit"],
                 "kind": p["kind"], "source_of_verification": "arithmetic",
                 "derived_from": "verifier_text"}
        if p["fail_if"]:
            entry["fail_if"] = p["fail_if"]
        expected[vid] = entry
        if p["form"] == "numeric_no_band":
            problems.append({"verifier": vid, "form": p["form"],
                             "detail": p["note"], "text": txt[:120]})
    return expected, problems