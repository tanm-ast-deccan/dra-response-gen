# src/augment_task.py
"""
Task augmenter. Runs on top of the existing two-call auditor.

Per task:
  0. audit_task(...)         -> verified arithmetic + corrected_solution_logic + changes
  1. apply corrections       -> corrected package fields (no SME gate; JUDGMENT edits
                                applied now but tagged for later SME review)
  2. AUGMENT Opus call       -> golden deliverable + DAG edges + Sanity-Check anchors
                                + any augmented verifiers   (single call)
  3. build verifier set + DAG + base weights (verifier_weights.compute_weights)
  4. select_crux(...)        -> deterministic crux set from anchors + DAG
  5. crux_shapley(...)       -> crux-only Shapley weights
  6. assemble AugmentResult (JSON-serializable) for CSV + HTML emit

The three crux metrics are computed later per-response by the scorer; the
augmenter freezes the crux set + Shapley weights so scoring is pure matching.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any

from src.prompt_evaluator import _call_llm, DEFAULT_JUDGE_MODEL, _repair_and_parse_json
from src.auditor import audit_task, get_field, AuditResult
from src.augment_templates import AUGMENT_SYSTEM_PROMPT, AUGMENT_TEMPLATE
from src.verifier_parser import parse_verifiers, format_verifiers, VerifierRecord
from src.verifier_weights import compute_weights, classify_dim, cascade_analysis
from src.crux_shapley import select_crux, crux_shapley, ancestors

logger = logging.getLogger("dra.augment")

AUGMENT_MAX_TOKENS = 32000


@dataclass
class AugmentResult:
    task_id: str
    audit_verdict: str = ""
    proceedable: bool = False

    # corrected/augmented package
    corrected_solution_logic: str = ""
    augmented_verifiers_text: str = ""          # canonical V<n>: text (existing + added)
    changes_applied: List[dict] = field(default_factory=list)
    judgment_changes_pending_sme: List[dict] = field(default_factory=list)

    # golden deliverable
    gold_deliverable_format: str = ""
    gold_deliverable_sections: List[dict] = field(default_factory=list)
    gold_deliverable_text: str = ""             # flattened, for CSV cell + scoring

    # structure
    dag: Dict[str, List[str]] = field(default_factory=dict)
    base_weights: Dict[str, float] = field(default_factory=dict)
    amzn_weights: Dict[str, float] = field(default_factory=dict)
    depths: Dict[str, int] = field(default_factory=dict)

    # crux
    crux_ids: List[str] = field(default_factory=list)
    crux_anchors_trap: List[str] = field(default_factory=list)
    crux_anchors_expert: List[str] = field(default_factory=list)
    crux_shapley_weights: Dict[str, float] = field(default_factory=dict)
    expected_values: Dict[str, dict] = field(default_factory=dict)  # frozen scoring targets
    crux_dropped_no_expected: List[str] = field(default_factory=list)  # reachable but no target
    scoreable: bool = True                       # False if any crux verifier is judgment_flagged
    not_scoreable_reason: str = ""               # human-readable why-excluded

    # provenance
    model_used: str = ""
    skipped_inputs: List[str] = field(default_factory=list)  # inputs that could not be read
    error: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def _flatten_deliverable(fmt: str, sections: List[dict]) -> str:
    parts = [f"[FORMAT: {fmt}]"]
    for s in sections or []:
        parts.append(f"## {s.get('title','')}\n{s.get('content','')}")
    return "\n\n".join(parts)


def _apply_changes(base_logic: str, corrected_logic: str, changes: List[dict]):
    """Apply corrections directly. Returns (final_logic, applied, pending_sme).

    MECHANICAL changes are applied silently. JUDGMENT_REQUIRED changes are also
    applied now (no SME gate per current instruction) but recorded separately so
    a later SME pass can review them.
    """
    final_logic = corrected_logic.strip() or base_logic
    applied, pending = [], []
    for ch in changes or []:
        (applied if ch.get("type") == "MECHANICAL" else pending).append(ch)
    return final_logic, applied, pending


def _merge_verifiers(existing_records: List[VerifierRecord],
                     augmented: List[dict]) -> (str, List[dict]):
    """Append augmented verifiers after the existing ones, renumbering continuously.
    Returns (canonical_text, full_verifier_dicts)."""
    vs = []
    for r in existing_records:
        vs.append({"id": f"V{r.index}", "text": r.text,
                   "dim": classify_dim(r.text), "type": None, "is_decision": False})
    nxt = (max((r.index for r in existing_records), default=0)) + 1
    for a in augmented or []:
        vs.append({"id": f"V{nxt}", "text": a.get("text", ""),
                   "dim": a.get("dim") or classify_dim(a.get("text", "")),
                   "type": a.get("type"), "is_decision": bool(a.get("is_decision"))})
        nxt += 1
    recs = [VerifierRecord(index=int(v["id"][1:]), text=v["text"]) for v in vs]
    return format_verifiers(recs), vs


def _remap_dag(dag_edges: Dict[str, list], valid_ids: set) -> Dict[str, List[str]]:
    """Keep only edges among known verifier ids; ensure every id is a key."""
    dag = {vid: [] for vid in valid_ids}
    for child, parents in (dag_edges or {}).items():
        if child not in valid_ids:
            continue
        dag[child] = [p for p in (parents or []) if p in valid_ids and p != child]
    return dag


def augment_task(
    row: dict,
    header_map: Dict[str, str],
    input_files_text: str = "",
    input_files_names: Optional[List[str]] = None,
    model_name: str = DEFAULT_JUDGE_MODEL,
    audit: Optional[AuditResult] = None,
    skipped_inputs: Optional[List[str]] = None,
) -> AugmentResult:
    task_id = get_field(row, header_map, "task_id") or "(no id)"
    prompt_text = get_field(row, header_map, "prompt")
    sanity_check_text = get_field(row, header_map, "sanity_check")
    verifiers_text = get_field(row, header_map, "verifiers")

    res = AugmentResult(task_id=task_id, model_used=model_name)
    res.skipped_inputs = list(skipped_inputs or [])

    # --- Stage 0/1: audit (reuse existing auditor) + apply corrections ---
    if audit is None:
        audit = audit_task(row, header_map, input_files_text=input_files_text,
                           input_files_names=input_files_names, model_name=model_name)
    res.audit_verdict = audit.verdict
    res.proceedable = audit.proceedable

    base_logic = get_field(row, header_map, "solution_logic")
    final_logic, applied, pending = _apply_changes(
        base_logic, audit.corrected_solution_logic, audit.changes)
    res.corrected_solution_logic = final_logic
    res.changes_applied = applied
    res.judgment_changes_pending_sme = pending

    # --- Stage 2: single AUGMENT call (golden deliverable + DAG + anchors) ---
    from src.auditor import _format_arithmetic_for_prompt  # reuse formatter
    arithmetic_results_text = ""
    if audit.claim_verdicts:
        # rebuild a compact text view from stored verdict dicts
        lines = []
        for v in audit.claim_verdicts:
            lines.append(f"- {v.get('id')} ({v.get('label')}): {v.get('status')} "
                         f"| claimed={v.get('claimed')} recomputed={v.get('recomputed')}")
        arithmetic_results_text = "\n".join(lines)
    else:
        arithmetic_results_text = "(no arithmetic claims)"

    prompt = AUGMENT_TEMPLATE.format(
        task_id=task_id,
        prompt_text=prompt_text or "(empty)",
        solution_logic_text=final_logic or "(none)",
        sanity_check_text=sanity_check_text or "(none)",
        verifiers_text=verifiers_text or "(none)",
        arithmetic_results_text=arithmetic_results_text,
    )
    try:
        raw = _call_llm(prompt, model_name, max_tokens=AUGMENT_MAX_TOKENS,
                        system_prompt=AUGMENT_SYSTEM_PROMPT)
    except Exception as e:
        res.error = f"augment call failed: {e}"
        return res

    aug = _repair_and_parse_json(raw, task_id)
    if aug is None:
        res.error = "augment call returned unparseable JSON"
        return res

    # golden deliverable
    gd = aug.get("gold_deliverable", {}) or {}
    res.gold_deliverable_format = gd.get("format", "")
    res.gold_deliverable_sections = gd.get("sections", []) or []
    res.gold_deliverable_text = _flatten_deliverable(
        res.gold_deliverable_format, res.gold_deliverable_sections)
    res.notes = aug.get("notes", "")

    # --- Stage 3: verifier set + DAG + base weights ---
    vparse = parse_verifiers(verifiers_text)
    aug_text, all_vs = _merge_verifiers(vparse.records, aug.get("augmented_verifiers", []))
    res.augmented_verifiers_text = aug_text
    valid_ids = {v["id"] for v in all_vs}

    dag = _remap_dag(aug.get("dag_edges", {}), valid_ids)
    res.dag = dag

    dw, aw, depths = compute_weights(all_vs, dag)
    # compute_weights returns percentages summing to 100; convert to 0..1 base
    res.base_weights = {k: v / 100.0 for k, v in dw.items()}
    res.amzn_weights = aw
    res.depths = depths

    # frozen scoring targets (keep only those for known verifier ids) — needed
    # BEFORE crux selection because the crux is filtered to verifiers that have one.
    ev = aug.get("expected_values", {}) or {}
    res.expected_values = {k: v for k, v in ev.items() if k in valid_ids}

    # --- Stage 4: deterministic crux selection (expected-value filtered) ---
    trap_anchors = [a for a in aug.get("trap_anchor_ids", []) if a in valid_ids]
    expert_anchors = [a for a in aug.get("expert_anchor_ids", []) if a in valid_ids]
    sel = select_crux(all_vs, dag,
                      trap_anchor_ids=trap_anchors or None,
                      expert_anchor_ids=expert_anchors or None,
                      expected_value_ids=list(res.expected_values.keys()))
    res.crux_ids = sel.crux_ids
    res.crux_anchors_trap = sel.anchors_trap
    res.crux_anchors_expert = sel.anchors_expert
    res.crux_dropped_no_expected = sel.dropped_no_expected

    # --- Stage 5: crux-only Shapley ---
    res.crux_shapley_weights = crux_shapley(
        all_vs, dag, res.base_weights, res.crux_ids)

    # --- Scoreability gate: a crux verifier tied to an unresolved JUDGMENT_REQUIRED
    # question (source_of_verification == "judgment_flagged") means the task's
    # answer is contested and must NOT be scored until an SME resolves it (handoff
    # §3). Surface it as a first-class column so list_clean_tasks can filter
    # automatically instead of relying on a hand-maintained --exclude-tasks list.
    flagged = [vid for vid in res.crux_ids
               if (res.expected_values.get(vid, {}) or {}).get(
                   "source_of_verification") == "judgment_flagged"]
    if flagged:
        res.scoreable = False
        res.not_scoreable_reason = ("crux verifier(s) judgment_flagged: "
                                    + ", ".join(sorted(flagged)))
    elif res.error:
        res.scoreable = False
        res.not_scoreable_reason = f"augment error: {res.error}"
    elif not res.crux_ids:
        res.scoreable = False
        res.not_scoreable_reason = "no crux verifiers selected"

    return res