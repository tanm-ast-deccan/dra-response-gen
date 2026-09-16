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


#: Relative tolerance applied to a numeric target that the verifier stated with
#: NO explicit band. Small enough to catch a genuine value error (38 vs 42, ~10%)
#: while absorbing rounding of a full-precision figure (3.333 vs 3.33333). An
#: explicit band in the text always wins over this default.
_DEFAULT_REL_TOL = 0.005  # 0.5%
#: Absolute floor so a tiny fractional value still gets a usable band.
_DEFAULT_ABS_FLOOR = 1e-6


def _is_whole_number(v: Optional[float]) -> bool:
    """A value with no meaningful fractional part (4, 42.0, -3). Whole numbers —
    counts, headcount, FTE, clerk counts — must stay EXACT: a "4 FTE" verifier
    should fail on 3 or 5, not tolerate 3.98–4.02. Only fractional values
    (rates, ratios, currency with paise/cents, percentages) get a default band.
    """
    if v is None:
        return False
    return abs(v - round(v)) < 1e-9


def _default_band_for(v: Optional[float]) -> float:
    """Tolerance for a band-less numeric target. Zero for whole numbers (kept
    exact, per the counting-quantity rule); a relative band for fractional ones.
    """
    if v is None or _is_whole_number(v):
        return 0.0
    return max(abs(v) * _DEFAULT_REL_TOL, _DEFAULT_ABS_FLOOR)


# ---------------------------------------------------------------------------
# CONSERVATIVE no-"=" target extraction  (Link 2)
# ---------------------------------------------------------------------------
#: Many authored SEED verifiers state their target as prose without an "=":
#: "Correct gas price used must be 3.047 USD/gal", "PPI Dec 2025 used must be
#: 181.077". The "=" grammar reads these as presence checks with no target, so
#: they never freeze, never map to a step, and the verifier DAG loses the edges
#: they would carry. This routine reads ONE such clause — deliberately
#: CONSERVATIVELY, because a wrong freeze is worse than a miss:
#:
#:   * fires ONLY on an explicit "must be / must equal(s) <number>" assertion —
#:     not "calculate X as N", not "convert to N", not a bare "of N" — so the
#:     number it freezes is one the verifier explicitly asserts as the target;
#:   * a NEGATION anywhere in the clause blocks it entirely. This is the trap
#:     guard: "Do NOT use 67.5", "must NOT be netted", "exclude ... 4" must never
#:     freeze their embedded number as a positive target, or a trapped response
#:     scores as correct. Missing a real target on a negated line is the safe
#:     failure;
#:   * a STRUCTURAL COUNT ("exactly 7 rows", "5 sections") is a format assertion,
#:     not a computed quantity, and is skipped — freezing it would grade a
#:     presence check as an arithmetic target.
#:
#: Anything this declines stays a presence check, exactly as before. This never
#: runs on text that already contains an "=" (parse_target tries the "=" forms
#: first and only falls through to here when there is no "=" clause at all), so
#: it cannot change any existing verdict.
_CONS_NEG = re.compile(
    r"\b(?:not|never|avoid|reject|exclude|do\s+not|must\s+not|don't|cannot|"
    r"no\s+longer|excluding|without)\b", re.IGNORECASE)
#: units we accept immediately after the number (real units only — never an
#: arbitrary following word, which would pollute the frozen unit field)
_CONS_UNIT = (r"%|pp|USD(?:/[A-Za-z]+)?|EUR|GBP|/[A-Za-z]+|"
              r"patients?(?:/[A-Za-z]+)?|percent")
_CONS_MUSTBE = re.compile(
    r"\bmust\s+(?:be|equal|equals)\s+"
    r"(?:approximately\s+|about\s+|~\s*|a\s+|the\s+|exactly\s+)?"
    rf"(?P<val>-?[\d,]+\.?\d*)\s*(?P<unit>{_CONS_UNIT})?",
    re.IGNORECASE)
#: a number immediately followed by one of these nouns is a structural count
_CONS_STRUCT_NOUN = re.compile(
    r"^\s*(?:rows?|columns?|cols?|sections?|items?|lines?|entries|checks?|"
    r"fields?|bullets?|steps?|tables?|cells?|pages?|decimals?|places?)\b",
    re.IGNORECASE)


def _read_conservative_mustbe(t: str, base: dict):
    """Read a single 'must be/equal <number>' target from text with NO '='.
    Returns a filled dict, or None if nothing safe parses. See the module note
    above for the conservatism rules (negation guard, structural-count guard)."""
    if _CONS_NEG.search(t):
        return None
    m = _CONS_MUSTBE.search(t)
    if not m:
        return None
    # structural count? inspect what immediately follows the number
    if _CONS_STRUCT_NOUN.match(t[m.end("val"):]):
        return None
    v = _f(m.group("val"))
    if v is None:
        return None
    out = dict(base)
    band = _default_band_for(v)
    out.update(kind="numeric", value=v, tol=band,
               unit=(m.group("unit") or "").strip(),
               form="numeric_mustbe_no_eq",
               note=("target read from a 'must be' clause with no '=' "
                     + (f"(default ±{band:g})" if band else "(exact)")))
    return out


# ---------------------------------------------------------------------------
# WIDENED compute-result form  (Link 2, guarded)
# ---------------------------------------------------------------------------
#: A step-computation verifier states its target as the RESULT of a computation:
#: "Calculate triage capacity as 60 patients per hour", "Convert peak demand to
#: 85 patients per hour". These describe the DERIVATION's interior steps — the
#: verifiers whose targets, once frozen and mapped, give the verifier DAG its
#: edges — but they carry no "=" and no "must be", so both the "=" grammar and
#: the conservative must-be reader decline them and the interior stays unwatched.
#:
#: This form is LOOSER than must-be and therefore carries a real trap risk: a
#: verifier can state the TRAPPED computation in the same shape ("Calculate the
#: demand rate as 67.5"), and freezing that value would grade a trapped response
#: as correct — the exact failure the benchmark exists to catch. So this form is
#: DISABLED unless the caller supplies the task's known trap values, and it
#: refuses to freeze any value equal to one of them. It keeps the same negation
#: and structural-count guards as must-be, and additionally requires an explicit
#: COMPUTE VERB (calculate/compute/convert/derive/determine) so it fires on a
#: stated computed result, not on an incidental number in prose.
_COMPUTE = re.compile(
    r"\b(?:calculate[sd]?|compute[sd]?|convert[sd]?|derive[sd]?|determine[sd]?)\b"
    r"[^.]*?\b(?:as|to)\s+(?:approximately\s+|about\s+|~\s*)?"
    rf"(?P<val>-?[\d,]+\.?\d*)\s*(?P<unit>{_CONS_UNIT})?",
    re.IGNORECASE)
#: A value within this relative distance of a known trap value is treated AS the
#: trap value and refused. Tight, because trap values are exact figures.
_TRAP_MATCH_REL = 1e-3


def _is_trap_value(v: float, trap_values) -> bool:
    for tv in (trap_values or ()):
        try:
            tvf = float(tv)
        except (TypeError, ValueError):
            continue
        if abs(v - tvf) <= max(abs(tvf) * _TRAP_MATCH_REL, 1e-9):
            return True
    return False


def _read_compute_result(t: str, base: dict, trap_values):
    """Read a 'Calculate/Convert X as/to <number>' target from text with NO '='.
    Refuses negated clauses, structural counts, AND any value matching a known
    trap value. Returns None (falls through to presence) when unsafe or absent.
    Only ever called when trap_values was supplied by the caller."""
    if _CONS_NEG.search(t):
        return None
    m = _COMPUTE.search(t)
    if not m:
        return None
    if _CONS_STRUCT_NOUN.match(t[m.end("val"):]):
        return None
    v = _f(m.group("val"))
    if v is None:
        return None
    if _is_trap_value(v, trap_values):
        return None                       # trap guard — never freeze a trap value
    out = dict(base)
    band = _default_band_for(v)
    out.update(kind="numeric", value=v, tol=band,
               unit=(m.group("unit") or "").strip(),
               form="numeric_compute_no_eq",
               note=("target read from a compute-result clause with no '=' "
                     + (f"(default ±{band:g})" if band else "(exact)")))
    return out


def parse_target(text: str, trap_values=None) -> dict:
    """Read one verifier's target clause.

    Returns {kind, value, tol, unit, fail_if, form, note}. kind is numeric,
    decision, string or presence. A presence check has no value by design and is
    not a failure.

    trap_values (optional): the task's known trap values. When supplied, the
    WIDENED compute-result reader ("Calculate X as N") is enabled for no-"="
    text, guarded so it never freezes a value equal to a trap value. When None
    (the default), only the strict must-be reader runs on no-"=" text, so
    behaviour is unchanged for any caller that does not pass trap values.
    """
    t = (text or "").strip()
    out = {"kind": "presence", "value": None, "tol": 0.0, "unit": "",
           "fail_if": "", "form": "none", "note": ""}

    fi = _FAILIF.search(t)
    if fi:
        out["fail_if"] = " ".join(fi.group("cond").split())
        t = t[:fi.start()].rstrip()

    if "=" not in t:
        # No "=" clause. Before calling this a presence check, try the no-"="
        # target readers (Link 2): authored seed verifiers routinely state a
        # target in prose without an "=".
        #   1. the strict "must be/equal <number>" reader — always on;
        #   2. the widened compute-result reader ("Calculate X as N") — ONLY when
        #      the caller supplied trap_values, and it refuses any trap value.
        # Both decline negated clauses and structural counts; anything they
        # decline falls through to presence, exactly as before.
        cons = _read_conservative_mustbe(t, out)
        if cons is not None:
            return cons
        if trap_values is not None:
            comp = _read_compute_result(t, out, trap_values)
            if comp is not None:
                return comp
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
        # No band was stated. A whole-number target (counts, FTE) stays EXACT; a
        # fractional target gets a small default relative band, so a rounded but
        # correct figure (3.333 vs a computed 3.33333) is not scored as a miss
        # and a genuine error (38 vs 42) still is. An explicit band, handled in
        # the _NUM branch above, always takes precedence over this default.
        band = _default_band_for(v)
        out.update(kind="numeric", value=v, tol=band,
                   unit=(m.group("unit") or "").strip(), form="numeric_no_band",
                   note=("value stated with NO band — default "
                         + (f"±{band:g} applied (fractional)" if band
                            else "exact (whole number)")))
        return out

    return None


def derive_expected_values(verifier_text_block: str,
                           trap_values=None
                           ) -> Tuple[Dict[str, dict], List[dict]]:
    """{vid: frozen target} from the canonical verifier block, plus problems.

    A presence check yields no entry, which is correct: it has nothing to compare.

    trap_values (optional): the task's known trap values. Passed straight to
    parse_target, where it enables the guarded compute-result reader ("Calculate
    X as N") for no-"=" verifiers while refusing any trap value. When None, only
    the strict must-be reader runs on no-"=" text, so behaviour is unchanged.
    """
    expected, problems = {}, []
    for line in (verifier_text_block or "").splitlines():
        line = line.strip()
        m = re.match(r"(V(?:\d+[a-z]?|_[A-Za-z0-9][A-Za-z0-9_.]*))\s*"
                     r"(?:\[[^\]]*\])?\s*:\s*(.+)", line)
        if not m:
            continue
        vid, txt = m.group(1), m.group(2)
        p = parse_target(txt, trap_values=trap_values)
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