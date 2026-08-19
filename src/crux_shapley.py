# src/crux_shapley.py
"""
Crux-verifier selection and Shapley-based crux scoring.

Two responsibilities, kept separate:

  1. select_crux(...)  — DETERMINISTIC selection of the crux verifier set from the
     DAG plus the Sanity-Check linkage. A verifier is a crux verifier iff it is
     graph-connected (ancestor OR descendant, transitively) to a Sanity-Check
     anchor. Anchors come from BOTH halves of the Sanity Check:
        - Lazy-AI / trap side  -> TRAP-type verifiers + verifiers the auditor's
          trap_assessment / QC flagged as trap-linked
        - Expert-path side     -> the analytical spine and the decision/deliverable
          verifiers (the main analytical act the expert must perform)
     No LLM judgement at selection time: given the anchors + DAG, the set is a
     pure reachability computation and is fully reproducible.

  2. crux_shapley(...) — Monte-Carlo Shapley values over the crux set only
     (non-crux verifiers held as fixed context). The coalition value function
     credits a verifier ONLY when all its DAG ancestors are present, which is
     what prevents a shared root cause from being counted once per descendant
     (Shapley Efficiency axiom => weights sum to 100%, no double counting).

The three reported metrics (all co-equal, no single headline):
    crux_cleared              : bool  — every crux verifier passed (AND). An
                                        unobserved crux verifier is NOT a pass.
    crux_verifier_pass_ratio  : float — plain k/n passed over the crux set.
    crux_shapley_score        : float — sum of crux-only Shapley weights of the
                                        passed crux verifiers (0..1).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Iterable


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def ancestors(dag: Dict[str, List[str]]) -> Dict[str, Set[str]]:
    """Transitive ancestors (dependencies) of every node."""
    anc: Dict[str, Set[str]] = {n: set() for n in dag}

    # `in_progress` is a cycle guard. Without it a cyclic dag recurses until the
    # stack dies, and _remap_dag upstream strips self-loops but not longer cycles,
    # so a two-node cycle in the asserted dag_edges took down select_crux.
    in_progress: Set[str] = set()

    def rec(n: str) -> Set[str]:
        if anc[n]:
            return anc[n]
        if n in in_progress:
            return set()
        in_progress.add(n)
        s: Set[str] = set()
        for p in dag.get(n, []):
            if p in dag:
                s.add(p)
                s |= rec(p)
        in_progress.discard(n)
        anc[n] = s
        return s

    for n in dag:
        rec(n)
    return anc


def descendants(dag: Dict[str, List[str]]) -> Dict[str, Set[str]]:
    """Transitive descendants of every node (reverse of ancestors)."""
    anc = ancestors(dag)
    desc: Dict[str, Set[str]] = {n: set() for n in dag}
    for n, ancs in anc.items():
        for a in ancs:
            desc[a].add(n)
    return desc


def connected_to_anchors(dag: Dict[str, List[str]], anchors: Iterable[str]) -> Set[str]:
    """All nodes reachable from the anchors following edges in EITHER direction
    (a verifier is crux if it feeds, or is fed by, a Sanity-Check anchor)."""
    anc = ancestors(dag)
    desc = descendants(dag)
    out: Set[str] = set()
    for a in anchors:
        if a not in dag:
            continue
        out.add(a)
        out |= anc[a]
        out |= desc[a]
    return out


# ---------------------------------------------------------------------------
# Crux selection
# ---------------------------------------------------------------------------

@dataclass
class CruxSelection:
    crux_ids: List[str] = field(default_factory=list)
    anchors_trap: List[str] = field(default_factory=list)      # Lazy-AI side
    anchors_expert: List[str] = field(default_factory=list)    # Expert-path side
    dropped_no_expected: List[str] = field(default_factory=list)  # reachable but unscoreable
    method: str = ""

    def to_dict(self) -> dict:
        return {
            "crux_ids": self.crux_ids,
            "anchors_trap": self.anchors_trap,
            "anchors_expert": self.anchors_expert,
            "dropped_no_expected": self.dropped_no_expected,
            "method": self.method,
        }


def select_crux(
    all_vs: List[dict],
    dag: Dict[str, List[str]],
    trap_anchor_ids: Optional[Iterable[str]] = None,
    expert_anchor_ids: Optional[Iterable[str]] = None,
    expected_value_ids: Optional[Iterable[str]] = None,
    final_answer_ids: Optional[Iterable[str]] = None,
    include_immediate_parents: bool = False,
    excluded_ids: Optional[Iterable[str]] = None,
) -> CruxSelection:
    """Deterministic crux selection.

    all_vs: verifier dicts with at least 'id', optionally 'type', 'dim', 'is_decision'.
    dag:    {id: [dependency_ids]}.
    trap_anchor_ids / expert_anchor_ids: explicit anchors supplied by the
        augmenter (derived from the Sanity Check's two halves). When not given,
        we fall back to structural heuristics:
          trap side   -> verifiers with type == 'TRAP'
          expert side -> the max-depth analytical node(s) + any decision verifier
    final_answer_ids: verifiers that test the FINAL ANSWER — mapped to a terminal
        step of the trajectory, or asserting a decision. Together with the anchors
        these are the crux; there is no graph expansion.
    expected_value_ids: verifiers that have a FROZEN EXPECTED VALUE. When given,
        a verifier with no target is dropped UNLESS it is an anchor or a
        final-answer verifier — those are crux by definition and are graded against
        their own text, which is a sufficient standard for a decision or presence
        check. Filtering on a number alone excluded the trap discriminator on a
        real task and let a trapped response score 1.0.
    """
    ids = [v["id"] for v in all_vs]
    by_id = {v["id"]: v for v in all_vs}
    excluded = {i for i in (excluded_ids or []) if i in by_id}

    # --- trap-side anchors (Lazy-AI test) ---
    if trap_anchor_ids is None:
        trap = [v["id"] for v in all_vs if (v.get("type") == "TRAP")
                or v["id"].upper().startswith("TRAP")]
    else:
        trap = [i for i in trap_anchor_ids if i in by_id]

    # --- expert-side anchors (main analytical act + decision/deliverable) ---
    if expert_anchor_ids is None:
        # depth of each node
        anc = ancestors(dag)
        depth = {n: len(anc[n]) for n in dag}  # proxy depth = #ancestors
        # decision verifiers
        decision = [v["id"] for v in all_vs
                    if v.get("is_decision") or v.get("dim") == "AR"]
        # deepest analytical (non-format) node
        analytical = [v["id"] for v in all_vs if v.get("dim") not in ("FD",)]
        deepest = max(analytical, key=lambda i: depth.get(i, 0)) if analytical else None
        expert = list(decision)
        if deepest and deepest not in expert:
            expert.append(deepest)
    else:
        expert = [i for i in expert_anchor_ids if i in by_id]

    anchors = set(trap) | set(expert)

    # --- DIRECT relation only, no reachability expansion --------------------
    # THE RULE: a verifier is crux when it is directly related to
    #   (a) the SANITY CHECK's two halves  -> the trap and expert anchors, and
    #   (b) the FINAL ANSWER               -> a terminal step of the derivation,
    #                                         or a decision.
    # These define the core analytical task.
    #
    # It was connected_to_anchors(dag, anchors): every verifier connected to any
    # anchor, ancestors AND descendants. On a fully connected derived graph that
    # reaches nearly everything — 11 anchors on one task became 19 of 22.
    #
    # The golden DELIVERABLE is deliberately not a ground. It restates the whole
    # working, so "the target appears in the golden" matched 13 of 15 verifiers on
    # one task and made the crux wider than connectivity did.
    # This was connected_to_anchors(dag, anchors): every verifier connected to any
    # anchor, ancestors AND descendants. On a fully connected derived graph that
    # reaches nearly everything — 11 anchors on one task expanded to 19 of 22
    # verifiers, which is the graph minus its isolated roots, not a crux set.
    #
    # The crux is the verifiers that define the core analytical task:
    #   * those directly related to the GOLDEN DELIVERABLE's answers, and
    #   * those directly related to the SANITY CHECK's two halves (the anchors).
    # Both are direct properties of a verifier, not of its position in a graph.
    direct = set(anchors)
    if final_answer_ids is not None:
        direct |= {i for i in final_answer_ids if i in by_id}
    # SEAL-ONLY one-hop widening: add the immediate parents (direct DAG
    # dependencies) of each directly-selected crux verifier. Exactly one hop —
    # not transitive — so the crux grows to the verifiers the core answers
    # DIRECTLY rest on, without ballooning to the whole connected component.
    if include_immediate_parents:
        parents = set()
        for i in list(direct):
            for p in dag.get(i, []):
                if p in by_id:
                    parents.add(p)
        direct |= parents
    else:
        parents = set()
    # SEAL-ONLY exclusion: remove dropped duplicates / rejected verifiers so the
    # crux never double-counts a quantity (e.g. the dropped split child V5a).
    direct -= excluded
    reachable_ids = [i for i in ids if i in direct]

    # --- expected-value filter ---
    # Keep only crux verifiers that have a frozen expected value (scoreable).
    # A reachable verifier with no target can never PASS (it stays UNOBSERVED and
    # blocks crux_cleared), so it is dropped from the crux here.
    # A verifier the scorer cannot match would sit UNOBSERVED and permanently block
    # crux_cleared, so an unscoreable one is dropped — but "unscoreable" is not the
    # same as "has no numeric target". A presence or decision verifier stated in
    # prose has nothing to freeze and is graded against its own text, which the
    # scorer does perfectly well.
    #
    # Filtering on a frozen value alone dropped exactly the verifiers that encode
    # the trap. On one task "States, unconditionally, DO NOT FLAG Industrial
    # Coatings" is a sanity-check ANCHOR and the sole discriminator of the whole
    # task, and it was excluded for lack of a number — so a response that fell for
    # the trap scored crux_cleared=YES and 1.0000 on every metric.
    #
    # So: keep a targetless verifier when it is an ANCHOR or a FINAL-ANSWER
    # verifier, because that is the definition of crux and its text is a
    # sufficient standard. Drop a targetless one that qualified some other way.
    dropped: List[str] = []
    if expected_value_ids is not None:
        ev = set(expected_value_ids)
        # anchors, final-answer verifiers, AND immediate parents are crux by
        # role: a judgment/decision parent (e.g. "how many clerks required") has
        # no number to freeze but is graded against its text, so it is kept even
        # without a frozen value. Only a verifier that qualified some OTHER way
        # and lacks a target is dropped as unscoreable.
        keep_without_target = (set(anchors) | set(final_answer_ids or ())
                               | parents)
        crux_ids = [i for i in reachable_ids
                    if i in ev or i in keep_without_target]
        dropped = [i for i in reachable_ids
                   if i not in ev and i not in keep_without_target]
    else:
        crux_ids = reachable_ids

    return CruxSelection(
        crux_ids=crux_ids,
        anchors_trap=sorted(set(trap)),
        anchors_expert=sorted(set(expert)),
        dropped_no_expected=dropped,
        method=("direct+parents: sanity-check anchors + final-answer + immediate "
                "parents, minus excluded, filtered to frozen-value verifiers"
                if include_immediate_parents else
                "direct: sanity-check anchors + final-answer verifiers, "
                "filtered to verifiers with a frozen expected value"),
    )


# ---------------------------------------------------------------------------
# Shapley over the crux set
# ---------------------------------------------------------------------------

def crux_subgraph(dag: Dict[str, List[str]],
                  crux_ids: List[str]) -> Dict[str, List[str]]:
    """The DAG induced on the crux verifiers only. An edge u->p is kept when both
    are crux; when a crux node's parent is NOT crux, we lift the edge to that
    parent's nearest crux ancestors, so the crux subgraph preserves reachability
    among crux nodes even where intermediate non-crux verifiers are skipped."""
    crux = set(crux_ids)
    anc = ancestors(dag)
    sub: Dict[str, List[str]] = {c: [] for c in crux_ids}
    for c in crux_ids:
        parents = set()
        for p in dag.get(c, []):
            if p in crux:
                parents.add(p)
            else:
                parents |= {a for a in anc.get(p, set()) if a in crux}
        sub[c] = sorted(parents)
    return sub


def full_dag_shapley(all_vs: List[dict], dag: Dict[str, List[str]],
                     base_weights: Dict[str, float],
                     iters: int = 60000, seed: int = 12345) -> Dict[str, float]:
    """Shapley value of EVERY verifier over the full DAG (all verifiers are
    players, no fixed context) — the whole-set importance distribution,
    renormalized to sum to 1.0."""
    all_ids = [v["id"] for v in all_vs]
    return crux_shapley(all_vs, dag, base_weights, all_ids, iters=iters, seed=seed)


def crux_dag_shapley(all_vs: List[dict], dag: Dict[str, List[str]],
                     base_weights: Dict[str, float], crux_ids: List[str],
                     iters: int = 60000, seed: int = 12345) -> Dict[str, float]:
    """Shapley within the CRUX SUBGRAPH: the induced DAG on crux verifiers, with
    the crux as the players and no external context. Isolates each crux verifier's
    marginal against the other crux verifiers alone (vs crux_shapley, which holds
    the non-crux set as fixed context)."""
    sub = crux_subgraph(dag, crux_ids)
    crux_vs = [v for v in all_vs if v["id"] in set(crux_ids)]
    return crux_shapley(crux_vs, sub, base_weights, crux_ids,
                        iters=iters, seed=seed)


def _coalition_value(S: Set[str], base: Dict[str, float], anc: Dict[str, Set[str]]) -> float:
    """Sum of base weights of verifiers whose ALL ancestors are also present."""
    return sum(base[n] for n in S if anc[n] <= S)


def crux_shapley(
    all_vs: List[dict],
    dag: Dict[str, List[str]],
    base_weights: Dict[str, float],
    crux_ids: List[str],
    iters: int = 60000,
    seed: int = 12345,
) -> Dict[str, float]:
    """Crux-only Shapley values, renormalized to sum to 1.0 over the crux set.

    Non-crux verifiers are held as fixed context (always present), so each crux
    verifier's value is its marginal contribution GIVEN the rest of the task.
    Exact enumeration for <=8 crux nodes; Monte-Carlo permutation estimate above.
    """
    nodes = [v["id"] for v in all_vs]
    anc = ancestors(dag)
    base = {n: float(base_weights.get(n, 0.0)) for n in nodes}
    players = [c for c in crux_ids if c in base]
    context = set(n for n in nodes if n not in players)

    sh = {p: 0.0 for p in players}

    if len(players) <= 8:
        import itertools, math
        for perm in itertools.permutations(players):
            S = set(context)
            prev = _coalition_value(S, base, anc)
            for p in perm:
                S.add(p)
                cur = _coalition_value(S, base, anc)
                sh[p] += cur - prev
                prev = cur
        denom = math.factorial(len(players)) if players else 1
        sh = {p: sh[p] / denom for p in players}
    else:
        rng = random.Random(seed)
        for _ in range(iters):
            perm = players[:]
            rng.shuffle(perm)
            S = set(context)
            prev = _coalition_value(S, base, anc)
            for p in perm:
                S.add(p)
                cur = _coalition_value(S, base, anc)
                sh[p] += cur - prev
                prev = cur
        sh = {p: sh[p] / iters for p in players}

    tot = sum(sh.values()) or 1.0
    return {p: sh[p] / tot for p in players}   # renormalized to 1.0


# ---------------------------------------------------------------------------
# The three metrics
# ---------------------------------------------------------------------------

@dataclass
class CruxMetrics:
    crux_cleared: bool
    crux_verifier_pass_ratio: float
    crux_shapley_score: float
    n_crux: int
    n_passed: int
    n_unobserved: int
    per_verifier: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "crux_cleared": self.crux_cleared,
            "crux_verifier_pass_ratio": round(self.crux_verifier_pass_ratio, 4),
            "crux_shapley_score": round(self.crux_shapley_score, 4),
            "n_crux": self.n_crux,
            "n_passed": self.n_passed,
            "n_unobserved": self.n_unobserved,
            "per_verifier": self.per_verifier,
        }


def score_crux(
    crux_ids: List[str],
    shapley_weights: Dict[str, float],
    results: Dict[str, Optional[bool]],
) -> CruxMetrics:
    """Compute the three co-equal crux metrics for one response.

    results[id] is True (pass), False (fail), or None (unobservable in the
    deliverable). Unobservable counts as NOT passed for all three metrics, and
    blocks crux_cleared.
    """
    n = len(crux_ids)
    passed = [c for c in crux_ids if results.get(c) is True]
    unobs = [c for c in crux_ids if results.get(c) is None]

    cleared = (n > 0) and all(results.get(c) is True for c in crux_ids)
    pass_ratio = (len(passed) / n) if n else 0.0
    shap = sum(shapley_weights.get(c, 0.0) for c in passed)

    per = []
    for c in crux_ids:
        r = results.get(c)
        per.append({
            "id": c,
            "shapley_weight": round(shapley_weights.get(c, 0.0), 4),
            "result": "PASS" if r is True else ("UNOBSERVED" if r is None else "FAIL"),
        })

    return CruxMetrics(
        crux_cleared=cleared,
        crux_verifier_pass_ratio=pass_ratio,
        crux_shapley_score=shap,
        n_crux=n,
        n_passed=len(passed),
        n_unobserved=len(unobs),
        per_verifier=per,
    )