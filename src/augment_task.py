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
from typing import Tuple, Dict, List, Optional, Any

from src.prompt_evaluator import _call_llm, DEFAULT_JUDGE_MODEL, _repair_and_parse_json
from src.auditor import audit_task, get_field, AuditResult
from src.augment_templates import AUGMENT_SYSTEM_PROMPT, AUGMENT_TEMPLATE
from src.verifier_parser import parse_verifiers, format_verifiers, VerifierRecord
from src.verifier_audit import format_verifiers_ids
from src.verifier_weights import compute_weights, classify_dim, cascade_analysis
from src.crux_shapley import select_crux, crux_shapley, ancestors

logger = logging.getLogger("dra.augment")

AUGMENT_MAX_TOKENS = 32000


@dataclass
class AugmentResult:
    task_id: str
    audit_verdict: str = ""
    # Verified arithmetic, carried through instead of discarded. These are the
    # trajectory's only input (src/trajectory.py) and re-deriving them costs an
    # Opus call, so they belong in the persisted package.
    claim_verdicts: List[dict] = field(default_factory=list)
    arithmetic_summary: Dict[str, Any] = field(default_factory=dict)
    proceedable: bool = False

    # corrected/augmented package
    corrected_solution_logic: str = ""
    corrected_prompt: str = ""
    corrected_sanity_check: str = ""
    corrected_verifiers_applied: bool = False
    verifier_change_audit: dict = field(default_factory=dict)
    #: Cycles found in the ASSERTED dag_edges and broken. A dependency graph with
    #: a cycle is not one; before the guard this crashed select_crux.
    dag_cycles_broken: List[list] = field(default_factory=list)
    #: The trajectory graph and the verifier graph derived from it. `dag` IS the
    #: derived graph now — the augmenter's asserted dag_edges is no longer used.
    step_graph: Dict[str, List[str]] = field(default_factory=dict)
    step_graph_health: dict = field(default_factory=dict)
    verifier_to_step: Dict[str, str] = field(default_factory=dict)
    verifier_mapping_report: dict = field(default_factory=dict)
    dag_derived: Dict[str, List[str]] = field(default_factory=dict)
    dag_source: str = "derived"
    #: The pre-correction text of each correctable artifact, so a report can
    #: distinguish "unchanged" from "rewritten to the same thing".
    _originals: dict = field(default_factory=dict)
    _verifiers_before_audit: List[dict] = field(default_factory=list)
    #: Identifies this run. A decisions file carries it back, so decisions made
    #: against a different run are refused rather than applied to text the
    #: reviewer never saw.
    run_hash: str = ""
    sealed: bool = False
    #: Which derivation steps no verifier watches. An unwatched step is real work
    #: a response can skip with nothing objecting. Read alongside
    #: verifier_mapping_report: a step can look unwatched only because the
    #: verifier that watches it failed to map.
    step_coverage: dict = field(default_factory=dict)
    #: Fourth-call output: property verdicts, the frozen verifier->step mapping it
    #: resolved, rewrites applied, and splits PROPOSED (never applied — a split
    #: mints ids and the frozen targets, crux set and scores are keyed by id).
    verifier_audit: dict = field(default_factory=dict)
    verifier_rewrites_applied: List[str] = field(default_factory=list)
    #: Verifiers with no frozen target. They cannot be scored, and select_crux
    #: drops them silently, so they must be counted somewhere visible.
    verifiers_without_target: List[str] = field(default_factory=list)
    #: The subset whose text states a value, so a missing target IS a defect.
    verifiers_value_bearing_without_target: List[str] = field(default_factory=list)
    #: Verifiers testing the final answer — terminal-mapped or decision-kind.
    #: One of the two direct grounds for crux; the other is the anchors.
    final_answer_verifiers: List[str] = field(default_factory=list)
    #: Verifiers whose frozen target came from the model's emission because their
    #: text carries no readable target clause.
    targets_emitted_only: List[str] = field(default_factory=list)
    #: Where the model's emitted target and the verifier's own text named
    #: DIFFERENT quantities. The text wins; this records what was overridden.
    target_disagreements: List[dict] = field(default_factory=list)
    #: Target clauses that could not be read — multi-value, or malformed.
    target_grammar_problems: List[dict] = field(default_factory=list)
    #: [{parent, children, target_went_to, parent_text}] — splits APPLIED. Children
    #: take suffixed ids (V5 -> V5a, V5b) so no existing id is renumbered.
    verifier_splits_applied: List[dict] = field(default_factory=list)
    corrected_claim_verdicts: List[dict] = field(default_factory=list)
    judgment_steps: List[dict] = field(default_factory=list)
    gate: dict = field(default_factory=dict)
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
    input_coverage: dict = field(default_factory=dict)       # what the auditor saw
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


def final_answer_verifiers(expected_values: Dict[str, dict],
                           verifier_to_step: Dict[str, str],
                           step_graph: Dict[str, List[str]]) -> List[str]:
    """Verifiers that test the FINAL ANSWER.

    Two grounds, both direct:
      * mapped to a TERMINAL step — one nothing else consumes, so it is an output
        of the derivation rather than working towards one
      * asserting a DECISION — a decision is a final answer by definition

    The second is also insurance against the first: mapping prose verifiers to
    steps is the weakest link in the pipeline, and on one task it placed no
    verifier on a terminal at all while six were decisions.
    """
    consumed = {p for ps in (step_graph or {}).values() for p in ps}
    terminals = {k for k in (step_graph or {}) if k not in consumed}
    out = {v for v, sid in (verifier_to_step or {}).items() if sid in terminals}
    out |= {v for v, ev in (expected_values or {}).items()
            if str((ev or {}).get("kind", "")).lower() == "decision"}
    return sorted(out)


def _run_hash(res) -> str:
    """Stable fingerprint of what a reviewer was shown.

    Covers the artifacts and the decision surface, so re-running the task
    produces a different hash and stale decisions are caught.
    """
    import hashlib

    h = hashlib.sha256()
    for part in (res.task_id, res.corrected_solution_logic, res.corrected_prompt,
                 res.corrected_sanity_check, res.augmented_verifiers_text,
                 json.dumps(res.changes_applied, sort_keys=True, default=str),
                 json.dumps(res.judgment_changes_pending_sme, sort_keys=True,
                            default=str)):
        h.update(str(part or "").encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()[:12]


def _or_original(corrected: str, original: str) -> str:
    """An empty correction means UNCHANGED, not blank.

    Call 2 returns an empty string when it changed nothing, so writing the raw
    value through would erase the artifact. Named once here so the rule is
    testable and cannot drift between the three corrected artifacts.
    """
    return (corrected or "").strip() or (original or "")


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


def _break_cycles(dag: Dict[str, List[str]]) -> Tuple[Dict[str, List[str]], List[list]]:
    """Remove the edges that close a cycle, and report them.

    A dependency graph with a cycle is not a dependency graph. Self-loops were
    already stripped, but a two-node cycle survived and reached select_crux.
    The edge that closes a cycle is dropped rather than the whole task failing,
    because an asserted graph is advisory — but it is reported, never silent.
    """
    found: List[list] = []
    colour: Dict[str, int] = {}

    def visit(n: str, path: List[str]) -> None:
        colour[n] = 1
        for p in list(dag.get(n, [])):
            if colour.get(p, 0) == 1:                 # p is on the current path
                found.append(path[path.index(p):] + [p] if p in path else [n, p])
                dag[n] = [x for x in dag[n] if x != p]
            elif colour.get(p, 0) == 0:
                visit(p, path + [p])
        colour[n] = 2

    for n in list(dag):
        if colour.get(n, 0) == 0:
            visit(n, [n])
    return dag, found


def _remap_dag(dag_edges: Dict[str, list], valid_ids: set) -> Dict[str, List[str]]:
    """UNUSED in the main path since the graph became derived. Kept because
    load_augment and the older per-task artifacts still contain asserted graphs
    that a reader may want to clean."""
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
    run_verifier_audit: bool = True,
    audit: Optional[AuditResult] = None,
    skipped_inputs: Optional[List[str]] = None,
    input_corpus=None,
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
                           input_files_names=input_files_names,
                           input_corpus=input_corpus,
                           model_name=model_name,
                           # the auditor's leakage verdict is an absence claim,
                           # so it must know which files it never saw
                           skipped_inputs=skipped_inputs)
    res.audit_verdict = audit.verdict
    res.claim_verdicts = list(audit.claim_verdicts or [])
    res.input_coverage = dict(audit.input_coverage or {})
    res.arithmetic_summary = dict(audit.arithmetic_summary or {})
    res.proceedable = audit.proceedable
    res.corrected_claim_verdicts = list(audit.corrected_claim_verdicts or [])
    res.judgment_steps = list(audit.judgment_steps or [])
    res.gate = dict(audit.gate or {})

    # HONOUR THE GATE. It was previously computed and recorded but never acted
    # on, so a golden whose own arithmetic did not reconcile still produced
    # verifiers, targets and a crux set, and shipped to an SME to hand-score.
    if res.gate and not res.gate.get("passed", True):
        res.scoreable = False
        res.not_scoreable_reason = (
            "arithmetic gate: " + res.gate.get("reason", "derivation does not "
                                               "reconcile"))
        res.error = res.not_scoreable_reason
        return res

    base_logic = get_field(row, header_map, "solution_logic")
    final_logic, applied, pending = _apply_changes(
        base_logic, audit.corrected_solution_logic, audit.changes)
    res.corrected_solution_logic = final_logic
    # Same fall-back-to-original rule as the solution logic: an empty correction
    # means unchanged, not blank.
    res.verifier_change_audit = dict(audit.verifier_change_audit or {})

    # keep the inputs so the report can say "unchanged" as a fact rather than
    # rendering an untouched artifact as though it had been rewritten
    res._originals = {
        "solution_logic": base_logic,
        "sanity_check": get_field(row, header_map, "sanity_check"),
        "prompt": get_field(row, header_map, "prompt"),
        "verifiers": verifiers_text,
    }
    res.corrected_prompt = _or_original(
        audit.corrected_prompt, get_field(row, header_map, "prompt"))
    res.corrected_sanity_check = _or_original(
        audit.corrected_sanity_check, get_field(row, header_map, "sanity_check"))
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
    # Augmentation extends the CORRECTED verifier block when call 2 produced one.
    # Extending the stale originals instead would leave a verifier the correction
    # retargeted still pinning the old figure, and that verifier then goes on to
    # score responses against a number the golden no longer claims.
    corrected_vtext = (audit.corrected_verifiers or "").strip()
    vparse = parse_verifiers(corrected_vtext or verifiers_text)
    if corrected_vtext and vparse.records:
        res.corrected_verifiers_applied = True
    elif corrected_vtext:
        # emitted but unparseable — fall back and say so rather than losing it
        vparse = parse_verifiers(verifiers_text)
        res.verifier_change_audit = dict(res.verifier_change_audit or {})
        res.verifier_change_audit["parse_failed"] = True
    aug_text, all_vs = _merge_verifiers(vparse.records, aug.get("augmented_verifiers", []))
    res.augmented_verifiers_text = aug_text
    valid_ids = {v["id"] for v in all_vs}

    # --- frozen scoring targets ---------------------------------------------
    # DERIVED FROM THE VERIFIER TEXT, not taken from the model's separate
    # emission. The text already carries the standard — the `toleranced` property
    # obliges it to — so a second copy is not a second standard, it is an index
    # over the first, and the two can drift.
    #
    # They did. On one task the augment call rewrote all 14 verifiers, reassigning
    # which quantity each id carries, while emitting expected_values against the
    # OLD mapping: V4's text said 15,967.90 and its target held 28,329.84, V5 said
    # 18,374.82 and held 15,967.90, and so on down the chain. Every numeric crux
    # verifier scored a different quantity than it stated. audit_verifier_changes
    # flagged 14 of 14 undeclared edits and the property call wrote "re-freeze
    # target" 14 times; nothing acted on either. A target parsed from the text it
    # describes cannot desynchronise from it.
    from src.verifier_grammar import derive_expected_values

    emitted = {k: v for k, v in (aug.get("expected_values") or {}).items()
               if k in valid_ids}
    derived, grammar_problems = derive_expected_values(
        format_verifiers_ids(all_vs))
    res.target_grammar_problems = grammar_problems

    merged, disagreed = {}, []
    for vid in valid_ids:
        dv, em = derived.get(vid), emitted.get(vid)
        if dv:
            # keep the emitted metadata (unit, source_of_verification) but the
            # value and band come from the text
            base = dict(em) if em else {}
            base.update(dv)
            if em and isinstance(em.get("value"), (int, float)) \
                    and isinstance(dv.get("value"), (int, float)):
                tol = max(abs(dv["value"]) * 0.02, abs(dv.get("tol") or 0), 0.5)
                if abs(em["value"] - dv["value"]) > tol:
                    disagreed.append({"verifier": vid, "emitted": em["value"],
                                      "from_text": dv["value"],
                                      "detail": ("the model's target and the "
                                                 "verifier's own text name "
                                                 "different quantities; the text "
                                                 "wins")})
            merged[vid] = base
        elif em:
            # no readable target clause: keep what the model emitted, but say so
            merged[vid] = em
            res.targets_emitted_only.append(vid)
    res.expected_values = merged
    res.target_disagreements = disagreed

    # --- Stage 3: the graph is DERIVED from the trajectory ------------------
    # The augmenter's asserted dag_edges is gone. It was one model's opinion,
    # produced in the same reply that invented the verifiers, with nothing
    # checking it — and it decided crux selection, weights and CHAIN. Given the
    # trajectory the graph is computable, and computable means repeatable.
    from src.derive_dag import (claim_graph, graph_health, map_verifiers_to_steps,
                                derive_verifier_dag, step_coverage)

    step_graph, step_nodes = claim_graph(
        audit.corrected_claim_verdicts, audit.judgment_steps)
    if not step_graph:
        res.scoreable = False
        res.not_scoreable_reason = (
            "no trajectory: call 2 produced no corrected claims, so there is "
            "nothing to derive a dependency graph from")
        res.error = res.not_scoreable_reason
        return res
    res.step_graph = step_graph
    res.step_graph_health = graph_health(step_graph, step_nodes)

    vmap = map_verifiers_to_steps(
        res.expected_values, {v["id"]: v.get("text", "") for v in all_vs},
        step_nodes)
    res.verifier_mapping_report = {
        "n_mapped": len(vmap.verifier_to_step), "n_verifiers": len(all_vs),
        "unmatched": vmap.unmatched, "ambiguous": vmap.ambiguous,
        "near_misses": vmap.near_misses, "detail": vmap.detail}
    frozen_map = dict(vmap.verifier_to_step)
    res.step_coverage = step_coverage(step_nodes, step_graph, frozen_map)

    # --- Stage 4: verifier property audit; rewrites AND splits both apply ----
    if run_verifier_audit:
        from src.verifier_audit import (audit_verifiers, apply_rewrites,
                                        apply_splits)
        va = audit_verifiers(
            task_id=task_id, verifiers=all_vs,
            expected_values=res.expected_values, step_nodes=step_nodes,
            solution_logic=res.corrected_solution_logic,
            sanity_check=res.corrected_sanity_check,
            mapping_report=res.verifier_mapping_report,
            coverage=res.step_coverage, verifier_to_step=frozen_map,
            model=model_name)
        res.verifier_audit = va.to_dict()
        if not va.error:
            if va.rewrites:
                # keep the pre-rewrite text so an SME rejecting a rewrite can
                # actually get the original back
                res._verifiers_before_audit = [dict(v) for v in all_vs]
                all_vs = apply_rewrites(all_vs, va.rewrites)
                res.verifier_rewrites_applied = sorted(va.rewrites)
            if va.splits:
                # A split mints suffixed ids (V5 -> V5a, V5b) so the parent's
                # lineage is readable and no existing id is renumbered. Everything
                # downstream is a function of (verifiers, dag, targets), so it all
                # recomputes below rather than needing patching.
                all_vs, ev2, split_log = apply_splits(
                    all_vs, va.splits, res.expected_values)
                res.expected_values = ev2
                res.verifier_splits_applied = split_log
            valid_ids = {v["id"] for v in all_vs}
            res.expected_values = {k: v for k, v in res.expected_values.items()
                                   if k in valid_ids}
            # re-map: split children are new verifiers and need placing
            vmap = map_verifiers_to_steps(
                res.expected_values,
                {v["id"]: v.get("text", "") for v in all_vs}, step_nodes)
            frozen_map = dict(vmap.verifier_to_step)
            # a link the call resolved only ADDS; a deterministic value match wins
            for vid, sid in va.tests_step.items():
                if vid in valid_ids:
                    frozen_map.setdefault(vid, sid)
            res.verifier_mapping_report = {
                "n_mapped": len(frozen_map), "n_verifiers": len(all_vs),
                "unmatched": vmap.unmatched, "ambiguous": vmap.ambiguous,
                "near_misses": vmap.near_misses, "detail": vmap.detail,
                "resolved_by_audit": sorted(
                    set(va.tests_step) & set(frozen_map) - set(vmap.verifier_to_step))}
            res.step_coverage = step_coverage(step_nodes, step_graph, frozen_map)

    res.augmented_verifiers_text = format_verifiers_ids(all_vs)
    res.verifier_to_step = frozen_map
    dag = derive_verifier_dag(step_graph, frozen_map,
                              all_verifier_ids=[v["id"] for v in all_vs])
    res.dag = dag
    res.dag_derived = dag
    res.dag_source = "derived"

    if not dag:
        res.scoreable = False
        res.not_scoreable_reason = (
            "no verifier could be placed on a derivation step, so no dependency "
            "graph could be derived")
        res.error = res.not_scoreable_reason
        return res

    dw, aw, depths = compute_weights(all_vs, dag)
    res.base_weights = {k: v / 100.0 for k, v in dw.items()}
    res.amzn_weights = aw
    res.depths = depths

    # --- Stage 4: deterministic crux selection (expected-value filtered) ---
    trap_anchors = [a for a in aug.get("trap_anchor_ids", []) if a in valid_ids]
    expert_anchors = [a for a in aug.get("expert_anchor_ids", []) if a in valid_ids]
    res.final_answer_verifiers = final_answer_verifiers(
        res.expected_values, res.verifier_to_step, res.step_graph)
    sel = select_crux(all_vs, dag,
                      trap_anchor_ids=trap_anchors or None,
                      expert_anchor_ids=expert_anchors or None,
                      expected_value_ids=list(res.expected_values.keys()),
                      final_answer_ids=res.final_answer_verifiers)
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
    # An UNRESOLVED judgment change is the same blocker as a judgment_flagged
    # crux target, and for a sharper reason: the model resolves an open question
    # differently on different runs. On one task the FX ambiguity was resolved as
    # the live rate (savings 2.20 Cr) and on the next as the file rate (0.918 Cr)
    # — same task, same code, two goldens with different terminal answers and
    # therefore different frozen targets. Nothing downstream can be trusted until
    # a human pins it.
    # Always recorded, whatever the verdict: a targetless verifier is invisible
    # otherwise, since select_crux simply drops it.
    res.verifiers_without_target = [v["id"] for v in all_vs
                                    if v["id"] not in res.expected_values]
    # Only a verifier whose TEXT STATES A VALUE needs a frozen target. A presence
    # or decision check stated in prose legitimately has none and is graded against
    # its own text — counting those as defects would fail a package for doing the
    # right thing.
    from src.verifier_audit import _VALUE_IN_TEXT
    res.verifiers_value_bearing_without_target = [
        v["id"] for v in all_vs
        if v["id"] not in res.expected_values
        and _VALUE_IN_TEXT.search(v.get("text", "") or "")]
    pending = res.judgment_changes_pending_sme or []
    if not res.proceedable:
        res.scoreable = False
        res.not_scoreable_reason = (
            f"audit verdict {res.audit_verdict or '?'} is not proceedable"
            + (f" — {res.gate.get('reason')}" if not res.gate.get("passed", True)
               else ""))
    elif pending:
        res.scoreable = False
        res.not_scoreable_reason = (
            f"{len(pending)} unresolved judgment question(s): "
            + "; ".join(str(c.get("location") or c.get("artifact"))[:40]
                        for c in pending[:4])
            + ". Left open, the next run may resolve them differently and change "
              "the golden's answer.")
    elif flagged:
        res.scoreable = False
        res.not_scoreable_reason = ("crux verifier(s) judgment_flagged: "
                                    + ", ".join(sorted(flagged)))
    elif res.error:
        res.scoreable = False
        res.not_scoreable_reason = f"augment error: {res.error}"
    elif not res.crux_ids:
        res.scoreable = False
        res.not_scoreable_reason = "no crux verifiers selected"
    elif (len(((res.verifier_change_audit or {}).get("undeclared_edits")) or [])
          >= max(3, int(0.8 * ((res.verifier_change_audit or {}).get("n_before")
                               or 10 ** 6)))):
        # The verifier block was rewritten wholesale without declaring any of it.
        # On one task all 14 were silently rewritten, reassigning which quantity
        # each id carried — detected here, acted on nowhere, and the package was
        # reported SALVAGEABLE with every numeric target one position out.
        ue = (res.verifier_change_audit or {}).get("undeclared_edits") or []
        res.scoreable = False
        res.not_scoreable_reason = (
            f"{len(ue)} of "
            f"{(res.verifier_change_audit or {}).get('n_before')} verifiers were "
            f"rewritten without being declared, so the verifier set cannot be "
            f"reconciled with the authored one: {', '.join(sorted(ue)[:8])}"
            + ("..." if len(ue) > 8 else ""))
    elif (res.verifiers_value_bearing_without_target
          and len(res.verifiers_value_bearing_without_target) > len(all_vs) // 3):
        # A verifier with no frozen target cannot be scored at all. select_crux
        # drops those silently, so crux_ids stays non-empty and every check above
        # passes — which reported scoreable=True on a package where 20 of 31
        # verifiers had nothing to compare a response against.
        res.scoreable = False
        res.not_scoreable_reason = (
            f"{len(res.verifiers_without_target)} of {len(all_vs)} verifiers have "
            f"no frozen target and cannot be scored: "
            + ", ".join(res.verifiers_without_target[:8])
            + ("..." if len(res.verifiers_without_target) > 8 else ""))

    res.run_hash = _run_hash(res)
    return res