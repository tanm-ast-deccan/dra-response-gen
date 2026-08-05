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
#: CUR matches a currency symbol OR a word form ("Rs.", "INR", "USD", "Rs").
_CUR = r"(?:[$\u20b9\u20ac\u00a3]|Rs\.?|INR|USD|EUR|GBP)"
#: UNIT includes a valuation multiple ("20.71x"), which is not a currency amount.
#: "Mn" is how the Indian-notation tasks write million; "mln" is the other form
#: seen. Both appear in real verifier text, so both must be units or the band
#: after them is unreachable.
_UNIT = (r"%|pp|x\b|mln\b|mn\b|bn\b|cr\b|crore|lakh|lacs?|"
         r"days?|hours?|units?|/kg|/hr|kWh[^\s]*")

_NUM = re.compile(
    rf"=\s*(?P<neg1>-)?\s*{_CUR}?\s*\(?\s*(?P<neg2>-)?\s*{_CUR}?\s*"
    r"(?P<val>[\d,]+\.?\d*)\s*\)?\s*"
    rf"(?P<unit>{_UNIT})?"
    rf"\s*\(\s*(?:\+/-|\u00b1)\s*{_CUR}?\s*(?P<band>[\d,]+\.?\d*)\s*"
    rf"(?P<bunit>{_UNIT})?\s*\)",
    re.IGNORECASE)

#: numeric with no band stated — a toleranced failure, but still a readable target
_NUM_NOBAND = re.compile(
    rf"=\s*(?P<neg1>-)?\s*{_CUR}?\s*\(?\s*(?P<neg2>-)?\s*"
    r"(?P<val>[\d,]+\.?\d*)\s*\)?\s*"
    rf"(?P<unit>{_UNIT})?(?!\d)",
    re.IGNORECASE)

#: A stated RANGE is a legitimate way to write a tolerance, and the frozen record
#: encodes it as a centre plus half-span. Seen in the wild: "a total need in the
#: range of $800,000-$900,000" against a frozen 864000 +/- 50000.
#: A range needs an explicit CUE, or an en/em dash. A bare ASCII hyphen with no
#: cue is ambiguous with subtraction, and it bit: "= median (V7 = 20.71x) x
#: (1 - 25% discount) = 15.53x" parsed "1 - 25" as a range and produced 13 +/- 12.
_RANGE = re.compile(
    r"(?:(?:in\s+the\s+range\s+of|between|range\s*:?|roughly|approximately|"
    r"approx\.?)\s*"
    r"[$\u20b9\u20ac\u00a3]?\s*(?P<lo>[\d,]+\.?\d*)\s*"
    r"(?:%|pp|mln|mn\b|cr\b|crore|lakh)?\s*"
    r"(?:\u2013|\u2014|-|to)"
    r"|[$\u20b9\u20ac\u00a3]?\s*(?P<lo2>[\d,]+\.?\d*)\s*"
    r"(?:%|pp|mln|mn\b|cr\b|crore|lakh)?\s*(?:\u2013|\u2014|\s+to\s+))\s*"
    r"[$\u20b9\u20ac\u00a3]?\s*(?P<hi>[\d,]+\.?\d*)\s*"
    r"(?P<unit>%|pp|mln|bn|cr\b|crore|lakh|days?|units?)?",
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

    # A verifier often shows its working: "Adjusted Multiple = median (V7 = 20.71x)
    # x (1 - 25% discount) = 15.53x". The TARGET is the value after the LAST "=",
    # not the first — the first is the start of the derivation and may quote
    # another verifier's figure. Read right to left and take the first clause that
    # yields a value.
    heads = [t[i:] for i in range(len(t)) if t[i] == "="]
    for head in reversed(heads[1:]):
        sub = _read_one(head, out)
        if sub is not None:
            return sub

    got = _read_one(t, out)
    if got is not None:
        return got
    # there WAS an "=" but no clause parsed: that is unreadable, not a presence
    # check. Losing this distinction hides a malformed target.
    out["form"] = "unreadable"
    out["note"] = "an '=' clause that matches no target form"
    return out


def _read_one(t: str, base: dict):
    """Read ONE '=' clause. Returns a filled dict, or None if nothing parses."""
    out = dict(base)
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
        v = _f(m.group("val"))
        bunit = (m.group("bunit") or "").strip().lower()
        # A "%" band on a value that is NOT itself a percentage is RELATIVE:
        # "= 15,967.90 Mn (+/- 1%)" means +/-159.68, not +/-1. A "pp" band on a
        # percentage value is absolute, which is what pp exists to say.
        if (bunit == "%" and unit.lower() not in ("%", "pp") and v is not None):
            band = abs(v) * band / 100.0
        # "= Rs. (-138.3) Cr" and "= -51.68%" both mean a negative target; the
        # sign may sit before the currency symbol, after it, or in brackets
        if v is not None and (m.group("neg1") or m.group("neg2")
                              or re.search(r"=\s*\(", t)):
            v = -abs(v)
        out.update(kind="numeric", value=v, tol=band,
                   unit=unit or (m.group("bunit") or ""), form="numeric")
        return out

    m = _DEC.search(t)
    if m:
        out.update(kind="decision", value=m.group("val").strip(), form="decision")
        return out

    i = t.find("=")
    tail = t[i + 1:] if i >= 0 else t
    m = _RANGE.search(tail)
    if m:
        lo, hi = _f(m.group("lo") or m.group("lo2")), _f(m.group("hi"))
        if lo is not None and hi is not None and hi > lo:
            out.update(kind="numeric", value=(lo + hi) / 2.0,
                       tol=(hi - lo) / 2.0, unit=(m.group("unit") or "").strip(),
                       form="numeric_range",
                       note=f"stated as a range {lo:g}-{hi:g}; frozen as centre +/- half-span")
            return out

    m = _NUM_NOBAND.search(t)
    if m:
        v = _f(m.group("val"))
        if v is not None and (m.group("neg1") or m.group("neg2")
                              or re.search(r"=\s*[^\d(]*\(\s*-", t)):
            v = -abs(v)
        out.update(kind="numeric", value=v, tol=0.0,
                   unit=(m.group("unit") or "").strip(), form="numeric_no_band",
                   note="value stated with NO band — a toleranced failure")
        return out

    return None


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