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

import json, os, glob, logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any

from src.prompt_evaluator import _call_llm, DEFAULT_JUDGE_MODEL, _repair_and_parse_json
from src.crux_shapley import score_crux, CruxMetrics

logger = logging.getLogger("dra.score")

GRADE_MAX_TOKENS = 8000

GRADE_SYSTEM = """\
You are a strict grader. You are given a response deliverable and a fixed list of \
crux checks, each with a FROZEN expected value. For each check, decide PASS / FAIL \
/ UNOBSERVED using ONLY the deliverable text. You do NOT recompute the correct \
answer and you do NOT invent tolerances — the expected value and tolerance are \
given; you only report whether the deliverable's stated value meets it.
- PASS: the deliverable states a value that matches the expected value within tol
  (for decisions/strings: exact match, case-insensitive).
- FAIL: the deliverable states a value that does NOT match.
- UNOBSERVED: the deliverable does not expose this value at all (not shown / not
  computed / omitted). Do not guess; if you cannot find it, it is UNOBSERVED.
"""

GRADE_TEMPLATE = """\
## RESPONSE DELIVERABLE (extracted text; may include tables)
{deliverable_text}

## CRUX CHECKS (id · criterion · expected)
{checks_text}

For EACH check id, output a verdict. Return ONLY this JSON:
{{
  "grades": {{
     "V1": {{"verdict": "PASS|FAIL|UNOBSERVED", "found": "what the deliverable stated (or null)", "why": "one line"}}
  }}
}}
Every check id above MUST appear as a key. No extra text.
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

    per_verifier: List[dict] = field(default_factory=list)
    deliverable_files: List[str] = field(default_factory=list)
    resolved_files: List[str] = field(default_factory=list)
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


def _select_deliverable_files(output_files: List[Any]) -> List[str]:
    """Prefer named deliverables (memo/report/output/bridge/decision), fall back
    to the generic *_response.docx, then everything scoreable."""
    names = [f if isinstance(f, str) else f.get("name", "") for f in (output_files or [])]
    scoreable = [n for n in names if n.lower().endswith(
        (".docx", ".txt", ".pdf", ".xlsx", ".csv", ".md", ".pptx"))]
    if not scoreable:
        return []
    pri = [n for n in scoreable if any(k in n.lower() for k in
           ("memo", "report", "output", "decision", "bridge", "deliverable", "1pager", "board"))]
    resp = [n for n in scoreable if "_response." in n.lower()]
    # dedupe preserving order: prioritized, then response, then rest
    ordered, seen = [], set()
    for group in (pri, resp, scoreable):
        for n in group:
            if n not in seen:
                ordered.append(n); seen.add(n)
    return ordered


def score_task(
    augmented: dict,
    response: dict,
    staging_remap: tuple = ("staging", "staging_1"),
    model_name: str = DEFAULT_JUDGE_MODEL,
    max_deliverable_chars: int = 40000,
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

    # resolve + extract deliverable(s) — STRING REMAP only, no fallback
    files = _select_deliverable_files(response.get("output_files"))
    res.deliverable_files = files
    texts = []
    unresolved = []
    for f in files:
        rp = resolve_output_path(f, staging_remap)
        if rp:
            res.resolved_files.append(rp)
            texts.append(f"### FILE: {os.path.basename(rp)}\n" + _extract_text(rp))
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

    deliverable_text = "\n\n".join(texts)[:max_deliverable_chars]

    if not deliverable_text.strip():
        # files resolved but extraction yielded nothing -> crux all UNOBSERVED
        results = {c: None for c in crux_ids}
        m = score_crux(crux_ids, shapley, results)
        _fill(res, m)
        res.error = "deliverable files resolved but no text could be extracted (all UNOBSERVED)"
        return res

    # build checks text
    checks = []
    for c in crux_ids:
        ev = expected.get(c, {})
        exp_str = (f"value={ev.get('value')} tol={ev.get('tol')} "
                   f"unit={ev.get('unit','')} kind={ev.get('kind','numeric')}") if ev else "(no frozen target)"
        checks.append(f"- {c}: {vtext.get(c,'(criterion text missing)')}  ||  EXPECTED: {exp_str}")
    checks_text = "\n".join(checks)

    prompt = GRADE_TEMPLATE.format(deliverable_text=deliverable_text, checks_text=checks_text)
    try:
        raw = _call_llm(prompt, model_name, max_tokens=GRADE_MAX_TOKENS, system_prompt=GRADE_SYSTEM)
    except Exception as e:
        res.error = f"grade call failed: {e}"
        return res
    parsed = _repair_and_parse_json(raw, task_id)
    if parsed is None:
        res.error = "grade call unparseable"
        return res

    grades = parsed.get("grades", {}) or {}
    results = {}
    per = {}
    for c in crux_ids:
        g = grades.get(c, {}) or {}
        verdict = (g.get("verdict") or "UNOBSERVED").upper()
        results[c] = True if verdict == "PASS" else (False if verdict == "FAIL" else None)
        per[c] = {"found": g.get("found"), "why": g.get("why"),
                  "source_of_verification": (expected.get(c, {}) or {}).get("source_of_verification", "")}

    m = score_crux(crux_ids, shapley, results)
    _fill(res, m)
    # enrich per_verifier with grader's found/why
    for pv in res.per_verifier:
        pv.update(per.get(pv["id"], {}))
    return res


def _fill(res: ScoreResult, m: CruxMetrics):
    res.crux_cleared = m.crux_cleared
    res.crux_verifier_pass_ratio = m.crux_verifier_pass_ratio
    res.crux_shapley_score = m.crux_shapley_score
    res.n_crux = m.n_crux
    res.n_passed = m.n_passed
    res.n_unobserved = m.n_unobserved
    res.per_verifier = m.per_verifier