# src/verifier_qc.py
"""
Deterministic verifier-QC checks, adapted from the gdpval-sample-generator
skill's checklist (T/R/D/S codes). Only the checks that can be done reliably
WITHOUT semantic understanding live here — these are hard, false-positive-free
structural checks. The semantic checks (R4 subjective, R5 TRAP-matches-specific-
failure, S3 names-methods) are left to the auditor LLM as soft flags, since they
need understanding rather than regex.

Each check returns zero or more QCFinding objects with a code, severity, and
message. Severity mirrors the skill: BLOCK (disqualifying) or REVISE (fixable).

These run on the EXISTING verifiers parsed from the task row, so the auditor's
findings include concrete verifier defects (e.g. "no decision-outcome verifier",
"no TRAP verifier", "a dimension is unrepresented") rather than only prose.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import List, Optional

from src.verifier_weights import classify_dim

FIVE_DIMS = ("DI", "AR", "RF", "EP", "FD")

# Decision-outcome keywords (R2): a task with a decision must have a verifier
# that checks the decision was stated.
_DECISION_RE = re.compile(
    r"\b(go|no[\s\-]?go|accept|reject|approve|recommend|select|choose|"
    r"option|line\s+[ab]|decision)\b", re.IGNORECASE)

# Tolerance markers (S2): an EP/numeric verifier should carry a tolerance/range.
_TOLERANCE_RE = re.compile(r"±|\+/-|\btolerance\b|\brange\b|[≥≤<>]|\bwithin\b")
_NUMERIC_RE = re.compile(r"\d")


@dataclass
class QCFinding:
    code: str
    severity: str       # BLOCK | REVISE
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


def run_verifier_qc(
    verifier_texts: List[str],
    has_decision_task: bool = True,
) -> List[QCFinding]:
    """Run the deterministic checklist over a list of verifier criterion strings.

    verifier_texts: the parsed verifier statements (without the V<n>: prefix).
    has_decision_task: whether the task culminates in a decision/recommendation
        (if known). When True, R2 (decision verifier exists) applies.
    """
    findings: List[QCFinding] = []
    if not verifier_texts:
        findings.append(QCFinding("R0", "BLOCK", "No verifiers present at all."))
        return findings

    # Classify each verifier into a dimension (best-effort, regex-based).
    dims = [classify_dim(t) for t in verifier_texts]

    # R2 — a decision-outcome verifier must exist (BLOCK) when the task decides.
    if has_decision_task:
        if not any(_DECISION_RE.search(t) for t in verifier_texts):
            findings.append(QCFinding(
                "R2", "BLOCK",
                "No decision-outcome verifier found — the task appears to require a "
                "decision/recommendation but no verifier checks that it was stated."))

    # R6 — at least one TRAP verifier (BLOCK). We detect a TRAP verifier
    # heuristically: it references rejecting/avoiding a wrong/stale/placeholder
    # value or the specific failure mode. (Existing verifiers rarely tag TRAP,
    # so this is a heuristic flag, not a hard claim.)
    trap_like = any(
        re.search(r"\b(reject|stale|placeholder|wrong|incorrect|do not use|"
                  r"avoid|must not|trap|naive|lazy)\b", t, re.IGNORECASE)
        for t in verifier_texts
    )
    if not trap_like:
        findings.append(QCFinding(
            "R6", "REVISE",
            "No TRAP-style verifier detected — none of the existing verifiers "
            "appears to test the cognitive trap's specific failure mode. Consider "
            "adding a TRAP verifier naming the exact wrong value/method to reject."))

    # R_DIM — all 5 dimensions represented (REVISE).
    present = set(dims)
    missing = [d for d in FIVE_DIMS if d not in present]
    if missing:
        findings.append(QCFinding(
            "R_DIM", "REVISE",
            f"Dimensions not represented among existing verifiers: {missing}. "
            f"(Present: {sorted(present)}.) The augmenter should add verifiers for "
            f"the missing dimensions."))

    # S2 — numeric/EP verifiers should carry a tolerance/range (REVISE).
    ep_without_tol = [
        i + 1 for i, (t, d) in enumerate(zip(verifier_texts, dims))
        if d == "EP" and _NUMERIC_RE.search(t) and not _TOLERANCE_RE.search(t)
    ]
    if ep_without_tol:
        findings.append(QCFinding(
            "S2", "REVISE",
            f"Numeric (EP) verifier(s) at position(s) {ep_without_tol} have a value "
            f"but no tolerance/range — a response slightly off would fail a brittle "
            f"exact-match. Add ± tolerance or an acceptable range."))

    # FD presence (verify()-style): at least one format/deliverability verifier.
    if "FD" not in present:
        findings.append(QCFinding(
            "R_FD", "REVISE",
            "No Format & Deliverability (FD) verifier — nothing checks the output "
            "is in the required format/structure."))

    return findings


def qc_summary(findings: List[QCFinding]) -> dict:
    blocks = [f for f in findings if f.severity == "BLOCK"]
    revises = [f for f in findings if f.severity == "REVISE"]
    return {
        "total": len(findings),
        "block": len(blocks),
        "revise": len(revises),
        "has_blocking_defect": len(blocks) > 0,
        "codes": [f.code for f in findings],
    }
