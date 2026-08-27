# src/score_task.py
"""
Score one response against an augmented package. Cheaper all-at-once grading:
a single Opus call reads the deliverable text once and returns PASS/FAIL/
UNOBSERVED for every crux verifier, matched against the FROZEN expected values
from the augmenter (the model does not invent the standard; it reports whether
the deliverable meets the given target). Then score_crux() -> 3 metrics.

Path resolution for output_files is tolerant of renamed staging folders.
"""

from __future__ import annotations

import json, os, re, glob, logging, math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any

from src.prompt_evaluator import _call_llm, DEFAULT_JUDGE_MODEL, _repair_and_parse_json
from src.crux_shapley import score_crux, CruxMetrics
from gear_score import gear_scores, chain_weights, dag_depths

logger = logging.getLogger("dra.score")

# Opus 4.7/4.8 use adaptive thinking, and thinking tokens share this budget with
# the visible text (stream.text_stream returns only the text blocks). A fixed
# 8000 was therefore unsafe on long verifier sets: thinking could consume most of
# it and the JSON would be cut mid-object. Size it from the check count instead.
GRADE_TOKENS_BASE = 12000        # headroom for adaptive thinking
GRADE_TOKENS_PER_CHECK = 400     # verdict + found + why, generously
#: 128k is the Opus 4.8 output ceiling; left under it for headroom. Thinking
#: shares the budget, which is why this is well above the size of the JSON.
GRADE_TOKENS_CAP = 96000


def grade_token_budget(n_checks: int) -> int:
    return min(GRADE_TOKENS_CAP,
               GRADE_TOKENS_BASE + GRADE_TOKENS_PER_CHECK * max(n_checks, 1))

GRADE_SYSTEM = """\
You are a strict grader. You are given a response deliverable and a fixed list of \
crux checks, each with a FROZEN expected value. For each check, return a BINARY \
verdict — PASS or FAIL — using ONLY the deliverable text. You do NOT recompute the \
correct answer and you do NOT invent tolerances: the expected value and tolerance \
are given; you only report whether the deliverable's stated value meets it.
- PASS: the deliverable states a value that matches the expected value within tol
  (for decisions/strings: exact match, case-insensitive).
- FAIL: everything else. This includes two distinct situations, and you must tag
  which one applies:
    * the deliverable states a value that does NOT match  -> unobservable = false
    * the deliverable does not expose this value at all (not shown / not computed
      / omitted)                                          -> unobservable = true
  An unobservable check is a FAIL, not a separate verdict. Never guess a value the
  deliverable does not state in order to award a PASS.

Some checks are marked "NO FROZEN TARGET". For those there is no pre-agreed value,
so judge ONLY whether the deliverable satisfies the check as the check's own text
states it. Do not derive a target of your own and do not treat a missing target as
licence to be lenient: if the check names a condition and the deliverable does not
meet it, that is a FAIL.

=== CALIBRATION RULES ===
These five rules come from cross-checking hand-scoring against SME annotations on
21 tasks; each names a divergence that actually occurred. Apply them literally.

1. LITERAL PRESENCE vs EQUIVALENT PATH — this depends on what the check tests.
   * If the check asserts an INTERMEDIATE DERIVATION (a named factor, ratio,
     baseline or reference value) or an EXPLICIT EXCLUSION (a stated constraint
     the response must be shown to have applied), the named quantity or statement
     must be LITERALLY PRESENT in the deliverable. A response that reaches the
     right end number by a different route, without stating the intermediate the
     check names, is a FAIL. The check exists to test whether that step was taken
     and shown, not whether the arithmetic happened to work out.
   * Only for DECISION checks and FINAL-RESULT checks may an equivalent valid
     path earn a PASS: what matters there is the committed answer, not the route.
   Do not generalise the equivalent-path allowance beyond those two kinds.

2. TRAP USE vs TRAP AVOIDANCE — never infer trap-failure from the mere appearance
   of a trap figure. A response may legitimately cite the decoy in order to reject
   it ("the workbook's 999.9 placeholder was not used; the live index is 157.039")
   and that is trap AVOIDANCE, so PASS. Read what the figure is doing: FAIL only
   when the trap value is actually carried into the forbidden calculation or
   presented as the answer. Conversely, silence is not avoidance — if the check
   requires the response to STATE that the decoy was rejected and it says nothing,
   that is a FAIL under rule 1.

3. FABRICATED OR MIS-CONSTRUCTED SUPPORT — a correct final decision does not
   rescue the verifiers underneath it. If a supporting figure is invented, pulled
   from the wrong source, or built by the wrong operation, the verifier covering
   that figure is a FAIL even when the headline recommendation matches the golden.
   Grade each check on its own referent.

4. THE DELIVERABLE MAY SPAN SEVERAL FILES — you are given the concatenated text of
   every file judged to be part of the deliverable, each under a "### FILE:"
   header. A value stated in ANY of them counts as stated. Do not mark a check
   unobservable because the value is absent from the first file.

5. GOLDEN-CORRECTION DIVERGENCE — if the frozen expected value looks wrong against
   the deliverable's own internal evidence (for instance the deliverable derives a
   figure correctly from stated inputs and the frozen target contradicts it), still
   grade against the FROZEN target, and additionally record the concern in
   "golden_divergence". Do not silently absorb the difference by passing the
   response, and do not re-derive a target of your own.

=== ROUNDING AND COMMITMENT ===
* Rounding is not an error. A value presented to fewer decimals than the target
  PASSES when it is within tolerance (96.81 against a target of 96.8085 passes).
  Unit-scale restatements likewise pass when the magnitude agrees (INR 15.45 Cr
  against 154,500,000).
* Where the response offers several candidate answers, grade the one it COMMITS
  to. A discarded alternative mentioned in passing neither earns nor forfeits.
* Never infer a value the deliverable does not state in order to award a PASS.
  If you cannot point to it, it is unobservable, which is a FAIL.
"""

GRADE_TEMPLATE = """\
## RESPONSE DELIVERABLE (extracted text; may include tables)
{deliverable_text}

## CRUX CHECKS (id · criterion · expected)
{checks_text}

For EACH check id, output a verdict. Return ONLY this JSON:
{{
  "grades": {{
     "V1": {{"verdict": "PASS|FAIL", "unobservable": false,
             "found": "what the deliverable stated (or null)", "why": "one line"}}
  }},
  "golden_divergence": [
     {{"id": "V1", "frozen": "the frozen target", "deliverable_derives": "what the
       deliverable's own evidence supports", "concern": "one line"}}
  ]
}}
Every check id above MUST appear as a key. "unobservable" must be true only when
the deliverable does not expose the value at all. "golden_divergence" may be an
empty list; use it only for calibration rule 5. No extra text.
"""


@dataclass
class ScoreResult:
    task_id: str
    provider: str = ""
    model: str = ""
    pass_index: int = 0
    run_id: str = ""

    crux_cleared: bool = False
    crux_verifier_pass_ratio: float = 0.0
    crux_shapley_score: float = 0.0
    n_crux: int = 0
    n_passed: int = 0
    n_unobserved: int = 0

    # CHAIN = (depth+1)^2-weighted DAG-gated crux score; GEAR0 = -log(shapley)-
    # weighted DAG-gated crux score. Both at lambda=0 (hard gate), 0-100. These are
    # the headline metrics the report uses and were previously not computed here:
    # score_task emitted only the flat pass-ratio and raw-shapley score.
    chain: float = 0.0
    gear_lambda0: float = 0.0
    crux_passed_ids: List[str] = field(default_factory=list)

    n_unobservable: int = 0          # FAILs whose cause was "not stated at all"

    # all-verifier companion (crux metrics above stay the headline)
    n_all: int = 0
    n_all_passed: int = 0
    all_verifier_pass_ratio: float = 0.0
    n_no_frozen_target: int = 0      # graded against their own text only
    n_no_frozen_target_passed: int = 0
    golden_divergence: List[dict] = field(default_factory=list)

    per_verifier: List[dict] = field(default_factory=list)
    deliverable_files: List[str] = field(default_factory=list)
    resolved_files: List[str] = field(default_factory=list)
    scratch_fallback: bool = False    # every file read as process narration; the
                                     # least-bad one was scored anyway
    dropped_as_scratch: List[str] = field(default_factory=list)
    file_classifications: List[dict] = field(default_factory=list)
    # Explicit, audit-facing record of WHICH files were opened and scored vs
    # rejected and why — the disambiguation a human grader would state. Each entry:
    # {"file","role":"graded|rejected","score","reason"}.
    graded_files: List[dict] = field(default_factory=list)
    deliverable_chars: int = 0
    deliverable_truncated: bool = False
    not_found: bool = False
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Path resolution (tolerant of renamed staging folder)
# ---------------------------------------------------------------------------

def resolve_output_path(stored: str, staging_remap: tuple = ("staging", "staging_1")) -> Optional[str]:
    """Resolve a stored output_files path by STRING REMAP only.

    The stored paths look like "./staging/<task>/runs/<provider>__p<n>/<file>".
    We rewrite ONLY the leading staging folder segment: the stored prefix
    (staging_remap[0], e.g. "staging") is replaced with the actual folder name
    (staging_remap[1], e.g. "staging_1"). Nothing on disk is renamed and the
    results JSON is not modified — this is an in-memory path-string rewrite.

    No fallback: if the remapped path does not exist, return None (the caller
    reports the task as NOT FOUND and moves on).
    """
    if not stored:
        return None
    old, new = staging_remap
    s = stored[2:] if stored.startswith("./") else stored     # drop leading ./
    parts = s.split("/")
    # swap only the first segment if it matches the stored staging folder name
    if parts and parts[0] == old:
        parts[0] = new
    remapped = "/".join(parts)
    return remapped if os.path.exists(remapped) else None


def _extract_text(path: str) -> str:
    """Extract text from a deliverable file via the repo document_parser."""
    try:
        from src.document_parser import read_document
        return read_document(path) or ""
    except Exception as e:
        logger.warning("extract failed for %s: %s", path, e)
        # fallback: plain text read
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception:
            return ""


# ---------------------------------------------------------------------------
# Deliverable vs scratch — decided from CONTENT, not from the filename
# ---------------------------------------------------------------------------
#
# Filename rules were wrong in both directions. They dropped "output.json" (a
# real and sometimes SOLE deliverable) and they dropped "<task>_response.docx"
# wholesale — but that file sometimes CONTAINS a complete memo, because agents
# are documented to return the deliverable inline instead of writing an artifact.
# Whether a file is the deliverable is a property of what is in it.

_SCRATCH_SIGNALS = [
    # agent process narration: talking about the work instead of doing it
    (re.compile(r"\b(?:let me|let's|i'?ll|i will|i need to|i'?m going to|"
                r"next,? i|now i|first,? i)\b", re.I), 3, "process narration"),
    (re.compile(r"\bi (?:found|see|notice|located|opened|read|checked)\b", re.I),
     3, "first-person exploration"),
    # tooling residue
    (re.compile(r"Traceback \(most recent call last\)|"
                r"\b(?:ModuleNotFoundError|AttributeError|KeyError|ValueError|"
                r"FileNotFoundError|ImportError)\b"), 6, "python traceback"),
    (re.compile(r"```(?:python|bash|sh|js)|^\s*(?:import |from \w+ import |"
                r"print\(|df\s*=)", re.I | re.M), 4, "source code / repl output"),
    # inability / abandonment
    (re.compile(r"\b(?:unable to (?:read|parse|open)|could not (?:read|parse|open)|"
                r"appears? to be corrupt|cannot be (?:read|parsed)|"
                r"file (?:is )?(?:empty|unreadable))\b", re.I), 5, "reports failure to read"),
]

_DELIVERABLE_SIGNALS = [
    (re.compile(r"^\s*DECISION\s*:|\bDECISION\s*:\s*(?:SIGN|USE|ACCEPT|REJECT|"
                r"GO|NO[- ]?GO|WALK)", re.I | re.M), 6, "committed decision line"),
    (re.compile(r"\b(?:MEETS_TARGET|DOES_NOT_MEET_TARGET|justification_flag|"
                r"recommendation)\b"), 4, "decision schema field"),
    (re.compile(r"^\s*\|.+\|\s*$", re.M), 3, "markdown/pipe table"),
    (re.compile(r"^\s*#{1,3} \S|^\s*(?:Section|Table|Step|Exhibit)\s+[A-Z0-9]",
                re.M), 2, "section structure"),
    (re.compile(r"^### Sheet:|^### FILE:", re.M), 2, "structured extract"),
    (re.compile(r"(?:INR|USD|EUR|GBP|₹|\$|€|£)\s?[\d,]+(?:\.\d+)?|"
                r"[\d,]+(?:\.\d+)?\s?(?:Cr|crore|lakh|bps|%)", re.I),
     1, "quantified figures"),
]

# Below this many characters a file cannot be a deliverable in its own right.
_MIN_DELIVERABLE_CHARS = 200


def classify_deliverable(name: str, text: str) -> dict:
    """Score how much a file reads like the deliverable rather than scratch.

    Returns {"name","score","chars","reasons"}. Positive score leans deliverable,
    negative leans scratch. Deterministic and inspectable on purpose: no extra
    model call, and every point of the score names the evidence that produced it.
    """
    t = text or ""
    reasons, score = [], 0.0

    # Valid JSON is a strong deliverable signal for schema-constrained tasks, and
    # a correct one can be genuinely short — a strict-JSON SCP task may answer in
    # 150 characters. So the JSON check runs FIRST and exempts the length floor;
    # otherwise the tersest correct deliverables would be classed as scratch.
    stripped = t.strip()
    json_ok = False
    if stripped.startswith(("{", "[")):
        try:
            obj = json.loads(stripped)
            json_ok = isinstance(obj, (dict, list)) and bool(obj)
        except Exception:
            json_ok = False
    if json_ok:
        score += 6
        reasons.append("+6 parses as JSON (length floor waived)")

    if not json_ok and len(stripped) < _MIN_DELIVERABLE_CHARS:
        reasons.append(f"-8 near-empty ({len(stripped)} chars)")
        score -= 8

    for rx, w, why in _DELIVERABLE_SIGNALS:
        n = len(rx.findall(t))
        if n:
            pts = w * min(n, 3) / 3.0          # saturate: presence, not volume
            score += pts
            reasons.append(f"+{pts:.1f} {why} (x{n})")

    for rx, w, why in _SCRATCH_SIGNALS:
        n = len(rx.findall(t))
        if n:
            # narration scales with density, not raw count, so a long report that
            # says "I'll" once is not condemned for it
            per_kchar = n / max(len(t) / 1000.0, 1.0)
            pts = w * min(per_kchar / 2.0, 1.0)
            score -= pts
            reasons.append(f"-{pts:.1f} {why} (x{n}, {per_kchar:.1f}/kchar)")

    return {"name": name, "score": round(score, 2),
            "chars": len(t), "reasons": reasons}


def select_by_content(extracted: List[tuple],
                      scratch_threshold: float = 0.0) -> tuple:
    """extracted: [(name, text), ...] -> (kept, dropped, fallback, classifications)

    A file is dropped as scratch only when it scores below the threshold AND some
    other file scores at or above it. The last resort is never discarded: if every
    file looks like scratch, all are kept and ``fallback`` is True, because
    "the agent shipped only process narration" is itself a scoreable outcome and
    must not silently become "no deliverable found".
    """
    cls = [classify_deliverable(n, t) for n, t in extracted]
    if not cls:
        return [], [], False, []
    good = [c for c in cls if c["score"] >= scratch_threshold]
    if not good:
        return ([c["name"] for c in sorted(cls, key=lambda c: -c["score"])],
                [], True, cls)
    good.sort(key=lambda c: -c["score"])
    dropped = [c["name"] for c in cls if c["score"] < scratch_threshold]
    return [c["name"] for c in good], dropped, False, cls


def _scoreable_files(output_files: List[Any]) -> List[str]:
    """Files whose format the parser can read at all. No deliverable judgment.

    Deduplicates by full PATH, not by basename. A response may legitimately ship
    two distinct files that share a name (observed in this corpus: two different
    Strategy_Memo.docx under different run directories). Collapsing those by name
    would silently discard one; extracting the same path twice would double its
    weight in the grader prompt.
    """
    names = [f if isinstance(f, str) else f.get("name", "")
             for f in (output_files or [])]
    out, seen = [], set()
    for n in names:
        if not n.lower().endswith(
                (".docx", ".txt", ".pdf", ".xlsx", ".csv", ".md", ".pptx", ".json")):
            continue
        if n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out


def _file_label(path: str, used: Dict[str, int]) -> str:
    """A unique, human-readable label for the '### FILE:' header.

    Two files with the same basename would otherwise produce two identical
    headers, and the grader could not tell which one a value came from.
    """
    base = os.path.basename(path)
    used[base] = used.get(base, 0) + 1
    if used[base] == 1:
        return base
    parent = os.path.basename(os.path.dirname(path)) or "?"
    return f"{base} [{parent}]"


def _neglog_from_shapley(shapley: Dict[str, float], crux_ids: List[str]) -> Dict[str, float]:
    """-log(shapley) over the crux set, normalized. Raw Shapley is root-heavy by
    construction; the -log inversion redistributes weight toward the leaves, which
    is the intended GEAR weighting. A zero/absent shapley falls back to uniform."""
    raw = {}
    for v in crux_ids:
        s = shapley.get(v, 0.0) or 0.0
        # guard: -log needs 0<s<1; clamp into range so a stored 0 or 1 is finite
        s = min(max(float(s), 1e-9), 1 - 1e-9)
        raw[v] = -math.log(s)
    tot = sum(raw.values()) or 1.0
    return {v: w / tot for v, w in raw.items()}


def compute_chain_gear(crux_ids: List[str],
                       crux_dag: Dict[str, list],
                       depths: Dict[str, int],
                       neglog: Dict[str, float],
                       shapley: Dict[str, float],
                       results: Dict[str, bool]) -> dict:
    """CHAIN and GEAR0 for a crux verdict set, using the SAME engine gear_score.py
    uses. CHAIN = (depth+1)^2 weights; GEAR0 = -log(shapley) weights; both DAG-gated
    at lambda=0. Falls back to a derived DAG / derived neglog when the package
    predates those fields, so the metric is always computed rather than skipped."""
    dag = crux_dag or {v: [] for v in crux_ids}
    # keep the dag restricted to crux nodes (the metric is defined on the crux subgraph)
    dag = {v: [p for p in dag.get(v, []) if p in crux_ids] for v in crux_ids}
    dep = depths or dag_depths(dag)
    nl = neglog or _neglog_from_shapley(shapley, crux_ids)
    marks = {v: (1 if results.get(v) else 0) for v in crux_ids}
    cw = chain_weights(crux_ids, dep)
    _, chain = gear_scores(marks, dag, cw, crux_ids, {}, 0.0, "uniform")
    _, gear0 = gear_scores(marks, dag, nl, crux_ids, {}, 0.0, "uniform")
    passed = [v for v in crux_ids if results.get(v)]
    return {"chain": round(100 * chain, 2),
            "gear_lambda0": round(100 * gear0, 2),
            "crux_passed_ids": passed}


def score_task(
    augmented: dict,
    response: dict,
    staging_remap: tuple = ("staging", "staging_1"),
    model_name: str = DEFAULT_JUDGE_MODEL,
    max_deliverable_chars: int = 200000,
) -> ScoreResult:
    task_id = response.get("task_id") or augmented.get("task_id") or "(no id)"
    res = ScoreResult(
        task_id=task_id,
        provider=response.get("provider", ""),
        model=response.get("model", ""),
        pass_index=int(response.get("pass_index", 0) or 0),
        run_id=response.get("run_id", ""),
    )

    # frozen package fields
    def _j(k, default):
        v = augmented.get(k)
        if isinstance(v, (dict, list)):
            return v
        try:
            return json.loads(v) if v else default
        except Exception:
            return default

    crux_ids = _j("crux_verifier_ids", [])
    shapley = _j("crux_shapley_weights_json", {})
    expected = _j("expected_values_json", {})
    # For CHAIN/GEAR0 we need the crux DAG, depths, and the -log(shapley) vector.
    # All are frozen package fields; each has a safe fallback so the grader still
    # runs on older packages that predate them.
    crux_dag = _j("crux_dag_json", {}) or _j("dag_json", {})
    depths = _j("depths_json", {}) or _j("crux_depth_json", {})
    neglog = _j("neglog_shapley_json", {}) or _j("neglog_crux_json", {})
    # verifier text map for the grader prompt
    verifiers_text = augmented.get("augmented_verifiers", "") or ""
    vtext = {}
    for line in verifiers_text.splitlines():
        line = line.strip()
        if ":" in line and line.split(":", 1)[0].strip().startswith("V"):
            vid, txt = line.split(":", 1)
            vtext[vid.strip()] = txt.strip()

    if not crux_ids:
        res.error = "no crux verifiers in augmented package"
        return res

    # A crux verifier with no frozen expected value cannot be matched, so the
    # grader would be asked to invent the standard. The augmenter is supposed to
    # exclude these (crux_dropped_no_expected); if one reaches here the package
    # is inconsistent and the task is not scoreable.
    no_target = [c for c in crux_ids if not (expected.get(c) or {})]
    if no_target:
        res.error = ("package inconsistent: crux verifier(s) with no frozen "
                     "expected value: " + ", ".join(no_target))
        return res

    # Grade EVERY verifier, not only the crux subset. The crux metrics stay the
    # headline, but the all-verifier companion figure is required alongside them,
    # and verifiers dropped from the crux set for having no frozen target
    # (AugmentResult.crux_dropped_no_expected) were previously graded by nobody.
    def _vnum(v):
        m = re.match(r"V(\d+)", v)
        return int(m.group(1)) if m else 10**6
    graded_ids = sorted(vtext.keys(), key=_vnum) or list(crux_ids)
    for c in crux_ids:                       # never silently drop a crux verifier
        if c not in graded_ids:
            graded_ids.append(c)
    res.n_all = len(graded_ids)
    crux_set = set(crux_ids)

    # Resolve and extract EVERY scoreable file first — the deliverable/scratch
    # decision needs the content, so it cannot be made before extraction.
    files = _scoreable_files(response.get("output_files"))
    res.deliverable_files = files
    extracted, unresolved = [], []
    for f in files:
        rp = resolve_output_path(f, staging_remap)
        if rp:
            res.resolved_files.append(rp)
            extracted.append((rp, _extract_text(rp)))
        else:
            unresolved.append(f)

    # NOT FOUND: no deliverable file could be resolved on disk -> report & skip.
    if files and not res.resolved_files:
        res.not_found = True
        res.error = ("NOT FOUND: no deliverable file resolved after remap "
                     f"{staging_remap[0]}->{staging_remap[1]} "
                     f"(e.g. {unresolved[0] if unresolved else '?'})")
        return res
    if not files:
        res.not_found = True
        res.error = "NOT FOUND: response has no scoreable output_files"
        return res

    kept, dropped, res.scratch_fallback, res.file_classifications = \
        select_by_content(extracted)
    res.dropped_as_scratch = [os.path.basename(d) for d in dropped]
    # Explicit audit trail: which file(s) were graded, which rejected, and why.
    # This is the disambiguation a human grader states out loud ("scored the named
    # memo; rejected the convenience scratch file that only narrates the work").
    _cls_by_name = {c["name"]: c for c in res.file_classifications}
    kept_set = set(kept)
    for c in res.file_classifications:
        role = "graded" if c["name"] in kept_set else "rejected"
        top_reason = (c.get("reasons") or ["(no signal)"])[0]
        res.graded_files.append({
            "file": os.path.basename(c["name"]),
            "role": role,
            "score": c["score"],
            "reason": ("scored as deliverable" if role == "graded"
                       else "rejected as scratch/non-deliverable")
                      + f"; top signal: {top_reason}"
                      + ("; FALLBACK: all files looked like scratch, least-bad scored"
                         if res.scratch_fallback and role == "graded" else ""),
        })
    by_path = dict(extracted)
    used: Dict[str, int] = {}
    texts = [f"### FILE: {_file_label(k, used)}\n" + by_path.get(k, "")
             for k in kept]

    full_text = "\n\n".join(texts)
    res.deliverable_chars = len(full_text)
    res.deliverable_truncated = len(full_text) > max_deliverable_chars
    deliverable_text = full_text[:max_deliverable_chars]

    if not deliverable_text.strip():
        # files resolved but extraction yielded nothing -> crux all UNOBSERVED
        results = {c: False for c in crux_ids}
        m = score_crux(crux_ids, shapley, results)
        _fill(res, m)
        res.n_unobservable = len(graded_ids)
        res.n_all_passed = 0
        res.all_verifier_pass_ratio = 0.0
        res.n_no_frozen_target = sum(
            1 for c in graded_ids if not (expected.get(c) or {}))
        for pv in res.per_verifier:
            pv["unobservable"] = True
        res.error = ("deliverable files resolved but no text could be extracted "
                     "(all verifiers FAIL, cause=unobservable)")
        return res

    # build checks text
    checks = []
    for c in graded_ids:
        ev = expected.get(c, {}) or {}
        if ev:
            exp_str = (f"value={ev.get('value')} tol={ev.get('tol')} "
                       f"unit={ev.get('unit','')} kind={ev.get('kind','numeric')}")
        else:
            exp_str = "NO FROZEN TARGET — judge against the check text alone"
        checks.append(f"- {c}: {vtext.get(c,'(criterion text missing)')}  ||  EXPECTED: {exp_str}")
    checks_text = "\n".join(checks)

    prompt = GRADE_TEMPLATE.format(deliverable_text=deliverable_text, checks_text=checks_text)
    try:
        raw = _call_llm(prompt, model_name,
                        max_tokens=grade_token_budget(len(graded_ids)),
                        system_prompt=GRADE_SYSTEM)
    except Exception as e:
        res.error = f"grade call failed: {e}"
        return res
    parsed = _repair_and_parse_json(raw, task_id)
    if parsed is None:
        res.error = "grade call unparseable"
        return res

    grades = parsed.get("grades", {}) or {}
    res.golden_divergence = parsed.get("golden_divergence", []) or []

    # A crux id absent from the grader's JSON is NOT an unobservable check — it is
    # a grader failure (most often output truncation). Silently folding it into
    # FAIL would bias every score downward and be indistinguishable from a real
    # miss, so it is a hard error.
    missing = [c for c in graded_ids if c not in grades]
    if missing:
        res.error = ("grader returned no verdict for verifier(s): "
                     + ", ".join(missing)
                     + " — likely output truncation; raise GRADE_TOKENS_PER_CHECK "
                       "or split the check list. Not scored.")
        return res

    results, per = {}, {}
    n_unobs = 0
    for c in graded_ids:
        g = grades.get(c, {}) or {}
        verdict = (g.get("verdict") or "FAIL").upper()
        unobs = bool(g.get("unobservable"))
        # Legacy tolerance: an older grader may still emit UNOBSERVED.
        if verdict not in ("PASS", "FAIL"):
            unobs, verdict = True, "FAIL"
        results[c] = (verdict == "PASS")
        if not results[c] and unobs:
            n_unobs += 1
        ev = expected.get(c, {}) or {}
        per[c] = {"found": g.get("found"), "why": g.get("why"),
                  "unobservable": (not results[c]) and unobs,
                  "is_crux": c in crux_set,
                  "no_frozen_target": not bool(ev),
                  "source_of_verification": ev.get("source_of_verification", "")}

    # headline: crux metrics over the crux subset of the same verdict set
    crux_results = {c: results.get(c, False) for c in crux_ids}
    m = score_crux(crux_ids, shapley, crux_results)
    _fill(res, m)

    # CHAIN + GEAR0 — the headline metrics the report uses, on the crux verdicts.
    cg = compute_chain_gear(crux_ids, crux_dag, depths, neglog, shapley, crux_results)
    res.chain = cg["chain"]
    res.gear_lambda0 = cg["gear_lambda0"]
    res.crux_passed_ids = cg["crux_passed_ids"]

    # companion: all-verifier figures
    res.n_unobservable = n_unobs
    res.n_all = len(graded_ids)
    res.n_all_passed = sum(1 for c in graded_ids if results.get(c))
    res.all_verifier_pass_ratio = (res.n_all_passed / res.n_all) if res.n_all else 0.0
    untargeted = [c for c in graded_ids if not (expected.get(c) or {})]
    res.n_no_frozen_target = len(untargeted)
    res.n_no_frozen_target_passed = sum(1 for c in untargeted if results.get(c))

    # per_verifier from score_crux covers only the crux subset; extend it to all
    by_id = {pv["id"]: pv for pv in res.per_verifier}
    full = []
    for c in graded_ids:
        pv = by_id.get(c) or {
            "id": c, "shapley_weight": 0.0,
            "result": "PASS" if results.get(c) else "FAIL"}
        pv.update(per.get(c, {}))
        full.append(pv)
    res.per_verifier = full
    return res


def _fill(res: ScoreResult, m: CruxMetrics):
    res.crux_cleared = m.crux_cleared
    res.crux_verifier_pass_ratio = m.crux_verifier_pass_ratio
    res.crux_shapley_score = m.crux_shapley_score
    res.n_crux = m.n_crux
    res.n_passed = m.n_passed
    # score_crux counts only None results, which the binary grader no longer
    # produces; n_unobservable carries the real count and is set by the caller.
    res.n_unobserved = m.n_unobserved
    res.per_verifier = m.per_verifier