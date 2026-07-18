# src/auditor.py
"""
Two-call task auditor for the DRA prompt-QC pipeline.

Flow per task:
  0. Parse + gate the existing verifiers (verifier_parser). Uncertain -> flag.
  1. CALL 1 (Opus): audit + emit structured arithmetic claims + preliminary findings.
  2. CODE: recompute the claims (arithmetic_verifier). Authoritative for numbers.
  3. Short-circuit: if call 1 is clean (SOUND, no claims errored, no findings),
     skip call 2 and emit a SOUND result.
  4. CALL 2 (Opus): finalize verdict + corrections, grounded in the code results.
  5. Assemble AuditResult (JSON-serializable) + prose.

Column access is by HEADER NAME via a resolver (never by index).

The Opus judge has no fallback (hard-stop) — see prompt_evaluator.MODEL_FALLBACK_CHAIN.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any

from src.prompt_evaluator import _call_llm, DEFAULT_JUDGE_MODEL, _repair_and_parse_json
from src.auditor_templates import (
    AUDITOR_SYSTEM_PROMPT,
    AUDIT_CLAIM_TEMPLATE,
    CORRECTION_TEMPLATE,
)
from src.arithmetic_verifier import (
    ArithmeticClaim,
    verify_claims,
    summarize,
)
from src.verifier_parser import parse_verifiers
from src.verifier_qc import run_verifier_qc, qc_summary

logger = logging.getLogger("dra.auditor")

# Output token ceilings. These are safety rails against truncation, NOT budgets —
# Opus 4.8 supports up to 128K output, and you pay only for tokens actually
# generated, so a high ceiling you don't hit is free. They are set generously so
# a legitimate audit never truncates, but low enough that a response which
# actually hits the ceiling trips `likely_truncated` and signals an abnormally
# long (rambling / over-claiming) response worth investigating rather than
# silently absorbing. Call 1 (analysis + all claims + findings) is the large one;
# Call 2 (corrections) is smaller.
CALL1_MAX_TOKENS = 32000
CALL2_MAX_TOKENS = 16000

# Verdicts that mean "fit to proceed downstream" (scoring/augmentation)
PROCEEDABLE = {"SOUND", "SALVAGEABLE"}


# ---------------------------------------------------------------------------
# Header resolution (by name, tolerant of case/whitespace variants)
# ---------------------------------------------------------------------------

# canonical field -> list of acceptable header spellings (lowercased, stripped)
_HEADER_ALIASES = {
    "task_id": ["task_id", "task id", "id"],
    "prompt": ["prompt", "prompt text"],
    "sanity_check": ["sanity check", "sanity_check", "sc"],
    "solution_logic": ["solution logic", "solution_logic", "logic"],
    "verifiers": ["verifiers", "verifier", "verifiers "],
    "drive_link": ["drive link", "drive_link", "drive_url", "drive url", "gdrive", "drive"],
    "prompt_type": ["prompt type", "prompt_type", "type"],
    "final_task_tags": ["final task tags", "final_task_tags", "task tags"],
}


class HeaderError(KeyError):
    pass


def build_header_map(headers: List[str]) -> Dict[str, str]:
    """Map canonical field names to the actual header strings present.
    Raises HeaderError listing any REQUIRED field that cannot be resolved."""
    norm = { (h or "").strip().lower(): h for h in headers }
    resolved: Dict[str, str] = {}
    for canon, aliases in _HEADER_ALIASES.items():
        for a in aliases:
            if a in norm:
                resolved[canon] = norm[a]
                break
    required = ["task_id", "prompt", "sanity_check", "solution_logic", "verifiers"]
    missing = [r for r in required if r not in resolved]
    if missing:
        raise HeaderError(
            f"Required column(s) not found by header name: {missing}. "
            f"Present headers: {list(headers)}"
        )
    return resolved


def get_field(row: dict, header_map: Dict[str, str], canon: str, default: str = "") -> str:
    col = header_map.get(canon)
    if col is None:
        return default
    val = row.get(col, default)
    return "" if val is None else str(val)


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------

@dataclass
class AuditResult:
    task_id: str
    verdict: str = "UNKNOWN"
    primary_reason: str = ""
    proceedable: bool = False
    decision_inversion: bool = False

    # verifier gate
    verifier_parse_status: str = ""          # CLEAN | UNCERTAIN | EMPTY
    verifier_count: int = 0
    verifier_parse_reasons: List[str] = field(default_factory=list)
    verifier_qc_findings: List[dict] = field(default_factory=list)
    verifier_qc_summary: Dict[str, Any] = field(default_factory=dict)

    # arithmetic
    arithmetic_summary: Dict[str, Any] = field(default_factory=dict)
    claim_verdicts: List[dict] = field(default_factory=list)

    # input-file provenance: did the auditor actually receive the model-facing
    # input files, and did the source-checking (C-layer) run?
    input_files_supplied: bool = False
    input_files_names: List[str] = field(default_factory=list)
    provenance_checked: bool = False
    provenance_note: str = ""

    # corrections + findings
    corrected_solution_logic: str = ""
    changes: List[dict] = field(default_factory=list)
    findings: List[dict] = field(default_factory=list)
    leakage_findings: List[dict] = field(default_factory=list)
    temporal_drift_findings: List[dict] = field(default_factory=list)
    missing_inputs: List[dict] = field(default_factory=list)
    trap_assessment: Dict[str, Any] = field(default_factory=dict)

    prose_findings: str = ""

    # provenance / debugging
    model_used: str = ""
    calls_made: int = 0
    skipped_call2: bool = False
    error: str = ""
    raw_response_debug: str = ""
    raw_response_len: int = 0
    likely_truncated: bool = False

    def to_dict(self, include_raw: bool = False) -> dict:
        d = asdict(self)
        if not include_raw:
            # Keep the bulky raw model dump out of the JSON; it's written to a
            # separate .raw.txt file by the CLI. Retain the len/truncation flags.
            d.pop("raw_response_debug", None)
        return d

    def to_json(self, indent: int = 2, include_raw: bool = False) -> str:
        return json.dumps(self.to_dict(include_raw=include_raw),
                          indent=indent, ensure_ascii=False)


# ---------------------------------------------------------------------------
# The audit
# ---------------------------------------------------------------------------

def audit_task(
    row: dict,
    header_map: Dict[str, str],
    input_files_text: str = "",
    input_files_names: Optional[List[str]] = None,
    model_name: str = DEFAULT_JUDGE_MODEL,
    max_tokens: int = 8000,
) -> AuditResult:
    """Run the two-call audit on one task row. input_files_text is the extracted
    text of the model-facing input files (e.g. from gdrive_fetcher); pass "" if
    not available — the audit still runs on prompt/logic/SC."""

    task_id = get_field(row, header_map, "task_id") or "(no id)"
    prompt_text = get_field(row, header_map, "prompt")
    sanity_check_text = get_field(row, header_map, "sanity_check")
    solution_logic_text = get_field(row, header_map, "solution_logic")
    verifiers_text = get_field(row, header_map, "verifiers")
    prompt_type = get_field(row, header_map, "prompt_type") or "(unspecified)"

    result = AuditResult(task_id=task_id)
    result.input_files_supplied = bool(input_files_text and input_files_text.strip())
    result.input_files_names = input_files_names or []
    if not result.input_files_supplied:
        result.provenance_note = (
            "Input files were NOT supplied to the auditor; the source-checking "
            "(provenance) layer did not run. Arithmetic was checked only for "
            "internal consistency, not against the actual input data. Declared "
            "input values are UNVERIFIED against source."
        )

    # --- Stage 0: verifier gate + QC checklist ---------------------------
    vparse = parse_verifiers(verifiers_text)
    result.verifier_parse_status = vparse.status
    result.verifier_count = vparse.count
    result.verifier_parse_reasons = vparse.reasons

    # Deterministic verifier-QC checklist (R2/R6/R_DIM/S2/...). Runs on the
    # parsed records when available; on the raw split otherwise.
    if vparse.records:
        v_texts = [r.text for r in vparse.records]
    else:
        v_texts = [s.strip() for s in (verifiers_text or "").splitlines() if s.strip()]
    qc_findings = run_verifier_qc(v_texts)
    result.verifier_qc_findings = [f.to_dict() for f in qc_findings]
    result.verifier_qc_summary = qc_summary(qc_findings)

    # --- Stage 1: audit + claims (call 1) --------------------------------
    # Format QC findings for the prompt
    if result.verifier_qc_findings:
        qc_text = "\n".join(
            f"- [{f['code']} {f['severity']}] {f['message']}"
            for f in result.verifier_qc_findings
        )
    else:
        qc_text = "(no deterministic verifier-QC defects found)"

    call1_prompt = AUDIT_CLAIM_TEMPLATE.format(
        task_id=task_id,
        prompt_type=prompt_type,
        prompt_text=prompt_text or "(empty)",
        input_files_text=input_files_text or "(no input files supplied to auditor)",
        sanity_check_text=sanity_check_text or "(none)",
        solution_logic_text=solution_logic_text or "(none)",
        verifiers_text=verifiers_text or "(none)",
        verifier_qc_text=qc_text,
    )
    try:
        # Call 1 (analysis scratchpad + structured claims) can be long, especially
        # once input files are in the prompt. Generous ceiling — see constant note.
        call1_tokens = max(max_tokens, CALL1_MAX_TOKENS)
        raw1 = _call_llm(call1_prompt, model_name, max_tokens=call1_tokens,
                         system_prompt=AUDITOR_SYSTEM_PROMPT)
    except Exception as e:
        result.error = f"call 1 failed: {e}"
        result.verdict = "AUDIT_FAILED"
        return result
    result.calls_made = 1

    call1 = _repair_and_parse_json(raw1, task_id)
    if call1 is None:
        result.error = "call 1 returned unparseable JSON"
        result.verdict = "AUDIT_FAILED"
        # Preserve the raw response for diagnosis (truncation vs malformed).
        result.raw_response_debug = raw1
        result.raw_response_len = len(raw1 or "")
        # Heuristic: if it doesn't end with a closing brace, it was likely truncated.
        tail = (raw1 or "").rstrip()
        result.likely_truncated = not tail.endswith("}")
        return result

    result.trap_assessment = call1.get("trap_assessment", {})
    result.leakage_findings = call1.get("leakage_findings", []) or []
    result.temporal_drift_findings = call1.get("temporal_drift_findings", []) or []
    result.missing_inputs = call1.get("missing_inputs", []) or []

    # --- Stage 2: recompute claims in code (authoritative) ---------------
    claims = [ArithmeticClaim.from_dict(c) for c in call1.get("arithmetic_claims", [])]
    verdicts = verify_claims(claims, source_text=(input_files_text or None))
    result.claim_verdicts = [v.to_dict() for v in verdicts]
    result.arithmetic_summary = summarize(verdicts)

    # provenance_checked is True only if the source-check actually evaluated at
    # least one declared input (found_in_source is not None somewhere).
    any_checked = any(
        p.get("found_in_source") is not None
        for v in verdicts for p in v.input_provenance
    )
    result.provenance_checked = bool(result.input_files_supplied and any_checked)
    if result.provenance_checked:
        result.provenance_note = "Declared input values were checked against the supplied input files."

    # --- Stage 3: short-circuit if call 1 is clean -----------------------
    prelim = (call1.get("preliminary_verdict") or "").upper().replace("-", "_")
    no_arith_errors = not result.arithmetic_summary.get("any_error", False)
    no_findings = not (result.leakage_findings or result.temporal_drift_findings
                       or result.missing_inputs)
    if prelim == "SOUND" and no_arith_errors and no_findings:
        result.verdict = "SOUND"
        result.primary_reason = "No defects found; arithmetic confirmed; no leakage/drift."
        result.proceedable = True
        result.skipped_call2 = True
        result.model_used = model_name
        result.prose_findings = (
            "Task audited SOUND. All load-bearing figures recomputed and confirmed; "
            "cognitive trap valid; no answer leakage or unpinned time-sensitive values."
        )
        return result

    # --- Stage 4: corrections grounded in code results (call 2) ----------
    arithmetic_results_text = _format_arithmetic_for_prompt(verdicts)
    call1_findings = {
        "preliminary_verdict": call1.get("preliminary_verdict"),
        "trap_assessment": result.trap_assessment,
        "leakage_findings": result.leakage_findings,
        "temporal_drift_findings": result.temporal_drift_findings,
        "missing_inputs": result.missing_inputs,
        "preliminary_notes": call1.get("preliminary_notes", ""),
    }
    call2_prompt = CORRECTION_TEMPLATE.format(
        task_id=task_id,
        call1_findings_json=json.dumps(call1_findings, indent=2, ensure_ascii=False),
        arithmetic_results_text=arithmetic_results_text,
        prompt_text=prompt_text or "(empty)",
        sanity_check_text=sanity_check_text or "(none)",
        solution_logic_text=solution_logic_text or "(none)",
    )
    try:
        raw2 = _call_llm(call2_prompt, model_name, max_tokens=max(max_tokens, CALL2_MAX_TOKENS),
                         system_prompt=AUDITOR_SYSTEM_PROMPT)
    except Exception as e:
        # We have call-1 + arithmetic; degrade to a partial result rather than lose it.
        result.error = f"call 2 failed: {e}"
        result.verdict = prelim or "SALVAGEABLE"
        result.primary_reason = "Correction call failed; verdict is preliminary only."
        result.proceedable = result.verdict in PROCEEDABLE
        result.model_used = model_name
        return result
    result.calls_made = 2

    call2 = _repair_and_parse_json(raw2, task_id)
    if call2 is None:
        result.error = "call 2 returned unparseable JSON"
        result.verdict = prelim or "SALVAGEABLE"
        result.primary_reason = "Correction call unparseable; verdict is preliminary only."
        result.proceedable = result.verdict in PROCEEDABLE
        result.model_used = model_name
        return result

    result.verdict = (call2.get("verdict") or prelim or "UNKNOWN").upper().replace("-", "_")
    result.primary_reason = call2.get("primary_reason", "")
    result.decision_inversion = bool(call2.get("decision_inversion", False))
    result.corrected_solution_logic = call2.get("corrected_solution_logic", "") or ""
    result.changes = call2.get("changes", []) or []
    result.findings = call2.get("findings", []) or []
    result.prose_findings = call2.get("prose_findings", "")
    result.proceedable = result.verdict in PROCEEDABLE
    result.model_used = model_name
    return result


def _format_arithmetic_for_prompt(verdicts) -> str:
    """Render claim verdicts as a compact text table for call 2."""
    if not verdicts:
        return "(no arithmetic claims were emitted in call 1)"
    lines = []
    for v in verdicts:
        rec = f"{v.recomputed:g}" if v.recomputed is not None else "n/a"
        clm = f"{v.claimed:g}" if v.claimed is not None else "n/a"
        trap = " [MATCHES TRAP VALUE]" if v.matches_trap else ""
        lines.append(
            f"- {v.id} ({v.label}): {v.status} | claimed={clm} recomputed={rec}{trap}\n"
            f"    {v.detail}"
        )
    return "\n".join(lines)