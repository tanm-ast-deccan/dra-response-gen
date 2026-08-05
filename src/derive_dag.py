# src/derive_dag.py
"""
Derive the verifier dependency graph from the trajectory, and compare it with the
one the augmenter asserted.

WHY DERIVE IT
    The augmenter used to assert `dag_edges` in the same reply that invented the
    verifiers, with `_remap_dag` only stripping malformed entries — it could not
    tell a wrong edge from a right one, and that graph decided crux selection,
    base weights, Shapley weights and CHAIN. Given the trajectory (which claim fed
    which, plus judgment steps and what they consume) the graph is computable, and
    computable means repeatable: same trajectory in, same graph out, assertable in
    a test. The asserted graph is no longer produced or consulted.

WHAT IT NEEDS, ALL ALREADY PRODUCED
    corrected_claim_verdicts  — from_claim per input, and the operation
    judgment_steps            — consumes
    expected_values           — value, tol, unit, kind per verifier
    verifier texts            — for tie-breaking by name

NO MODEL CALL. Every function here is pure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple

from src.arithmetic_verifier import parse_number

# --------------------------------------------------------------------------
# Step graph
# --------------------------------------------------------------------------


def claim_graph(corrected_claim_verdicts: List[dict],
                judgment_steps: Optional[List[dict]] = None
                ) -> Tuple[Dict[str, List[str]], Dict[str, dict]]:
    """child -> [parents] over claim and judgment ids, plus a node lookup.

    Claim parents come from `from_claim` on each input; judgment parents from
    `consumes`. Both are stated by call 2, so this is assembly, not inference.
    """
    graph: Dict[str, List[str]] = {}
    nodes: Dict[str, dict] = {}

    for c in corrected_claim_verdicts or []:
        cid = c.get("id")
        if not cid:
            continue
        nodes[cid] = {"id": cid, "kind": "arithmetic", "label": c.get("label", ""),
                      "value": c.get("recomputed"), "status": c.get("status", ""),
                      "operation": c.get("operation", ""),
                      # the wrong answer, so a verifier on this step can name it
                      "trap_value": c.get("trap_value"),
                      "solution_step": c.get("solution_step", "")}
        graph.setdefault(cid, [])
        for p in c.get("input_provenance") or []:
            par = p.get("from_claim")
            if par and par != cid and par not in graph[cid]:
                graph[cid].append(par)

    for j in judgment_steps or []:
        jid = j.get("id")
        if not jid:
            continue
        nodes[jid] = {"id": jid, "kind": "judgment",
                      "label": j.get("question", ""), "value": None,
                      "ruling": j.get("ruling", ""),
                      "solution_step": j.get("solution_step", "")}
        graph.setdefault(jid, [])
        for p in j.get("consumes") or []:
            if p and p != jid and p not in graph[jid]:
                graph[jid].append(p)

    # a parent naming a node that does not exist is dropped and reported by
    # graph_health, never silently kept
    for k in list(graph):
        graph[k] = [p for p in graph[k] if p in nodes]
    return graph, nodes


def _find_cycles(graph: Dict[str, List[str]]) -> List[List[str]]:
    cycles, colour = [], {}

    def visit(n: str, path: List[str]) -> None:
        colour[n] = 1
        for p in graph.get(n, []):
            if colour.get(p, 0) == 1:
                cycles.append(path[path.index(p):] + [p] if p in path else [p, n])
            elif colour.get(p, 0) == 0:
                visit(p, path + [p])
        colour[n] = 2

    for n in graph:
        if colour.get(n, 0) == 0:
            visit(n, [n])
    return cycles


def _ancestors(graph: Dict[str, List[str]]) -> Dict[str, Set[str]]:
    """Transitive ancestors, safe on a cyclic graph.

    crux_shapley.ancestors() is the canonical implementation and is reused when
    the graph is acyclic, so there is one code path for the normal case. But it
    recurses with no visited guard, so a cycle blows the stack — which would crash
    graph_health() on exactly the input it exists to report. NOTE the same
    exposure exists upstream: _remap_dag strips self-loops but not longer cycles,
    so a cyclic asserted dag_edges reaching select_crux would fail the same way.
    """
    if not _find_cycles(graph):
        from src.crux_shapley import ancestors
        return ancestors(graph)

    out: Dict[str, Set[str]] = {}
    for start in graph:
        seen: Set[str] = set()
        stack = list(graph.get(start, []))
        while stack:
            n = stack.pop()
            if n in seen or n not in graph:
                continue
            seen.add(n)
            stack.extend(graph.get(n, []))
        out[start] = seen
    return out


def graph_health(graph: Dict[str, List[str]], nodes: Dict[str, dict]) -> dict:
    """Is this a derivation? Reports; never repairs.

    A terminal is a node nothing consumes. Orphans matter because a node no
    terminal reaches contributes to no answer — either the link is missing or the
    step is dead weight.
    """
    consumed = {p for ps in graph.values() for p in ps}
    terminals = sorted(k for k in graph if k not in consumed)
    anc = _ancestors(graph)
    reachable: Set[str] = set()
    for t in terminals:
        reachable |= anc.get(t, set()) | {t}
    cycles = _find_cycles(graph)
    return {
        "n_nodes": len(graph),
        "n_edges": sum(len(v) for v in graph.values()),
        "terminals": terminals,
        "terminal_kinds": {t: nodes.get(t, {}).get("kind", "?") for t in terminals},
        "connected": len(reachable),
        "orphans": sorted(set(graph) - reachable),
        "cycles": cycles,
        "is_derivation": not cycles and not (set(graph) - reachable),
    }


# --------------------------------------------------------------------------
# Verifier -> step
# --------------------------------------------------------------------------

_STOP = {"the", "a", "an", "of", "for", "in", "at", "to", "and", "per", "total",
         "value", "amount", "figure", "number", "result", "from", "by", "on",
         "must", "be", "is", "are", "equal", "equals", "should", "stated"}
_TOK = re.compile(r"[a-z0-9]+")
_SUF = ("ations", "ation", "ments", "ment", "ings", "ing", "ies", "ied",
        "ers", "er", "ed", "es", "s")


def _stem(t: str) -> str:
    for suf in _SUF:
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            b = t[:-len(suf)]
            return b[:-1] if b.endswith("i") else b
    return t


def _tokens(s: str) -> Set[str]:
    return {_stem(t) for t in _TOK.findall((s or "").lower())
            if t not in _STOP and len(t) > 1}


def name_agreement(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    if ta <= tb or tb <= ta:
        return 1.0
    return len(ta & tb) / len(ta | tb)


NAME_MIN = 0.30
#: Below NAME_MIN but not negligible. A prose verifier and the judgment step it
#: tests can share almost no vocabulary and still be the same thing — "reject
#: adding permanent front end staffing" against "Option A add front-end staff or
#: Option B pool" scores 0.18. Lowering NAME_MIN to catch those buys false
#: matches, so they are reported as near misses for adjudication instead.
NEAR_MISS_MIN = 0.12


def _within(v: Optional[float], target: Optional[float], tol: float,
            unit: str = "") -> bool:
    """Does a step value match a frozen target, honouring the frozen tolerance?

    Reads `tol` from the frozen record rather than assuming one. Also tries the
    x100 form, because a target carrying unit "percent" is routinely stored
    against a step that computed the ratio.
    """
    if v is None or target is None:
        return False
    band = abs(tol) if tol else 0.0
    if abs(v - target) <= band or (band == 0 and abs(v - target) <= 1e-9):
        return True
    if "percent" in (unit or "").lower() or "%" in (unit or ""):
        return abs(v * 100.0 - target) <= max(band, 0.5)
    return False


@dataclass
class VerifierMapping:
    verifier_to_step: Dict[str, str] = field(default_factory=dict)
    detail: Dict[str, str] = field(default_factory=dict)
    unmatched: List[dict] = field(default_factory=list)
    ambiguous: List[dict] = field(default_factory=list)
    #: Plausible but below threshold. Resolve these with a model call ONCE, freeze
    #: the result, and the graph derived from the frozen mapping stays repeatable.
    near_misses: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def map_verifiers_to_steps(expected_values: Dict[str, dict],
                           verifier_texts: Dict[str, str],
                           nodes: Dict[str, dict]) -> VerifierMapping:
    """Which trajectory step does each verifier test?

    Numeric verifiers match a step whose value equals their frozen target within
    the FROZEN tolerance; value ties break on name agreement and unbroken ties are
    reported rather than guessed. Decision and presence verifiers match a judgment
    step, scored against the frozen decision value as well as the verifier prose —
    "ADD_DOCTORS" matches a ruling far more cleanly than the criterion text does.

    A verifier matching no step is a finding: it tests something the derivation
    does not produce.
    """
    m = VerifierMapping()
    arith = [n for n in nodes.values()
             if n["kind"] == "arithmetic" and n.get("value") is not None]
    judg = [n for n in nodes.values() if n["kind"] == "judgment"]

    for vid, ev in (expected_values or {}).items():
        ev = ev or {}
        vtext = verifier_texts.get(vid, "") or ""
        kind = str(ev.get("kind", "numeric")).lower()
        target = parse_number(ev.get("value"))
        tol = parse_number(ev.get("tol")) or 0.0
        unit = str(ev.get("unit", "") or "")

        if kind == "numeric" and target is not None:
            cands = [n for n in arith if _within(n["value"], target, tol, unit)]
            if not cands:
                m.unmatched.append({"verifier": vid, "kind": kind,
                                    "target": ev.get("value"),
                                    "reason": "no step produces this value"})
                continue
            if len(cands) == 1:
                m.verifier_to_step[vid] = cands[0]["id"]
                m.detail[vid] = f"unique value match on {cands[0]['label'][:40]!r}"
                continue
            scored = sorted(((name_agreement(vtext, n["label"]), n) for n in cands),
                            key=lambda x: -x[0])
            best, second = scored[0], (scored[1] if len(scored) > 1 else (0.0, None))
            if best[0] >= NAME_MIN and best[0] > second[0]:
                m.verifier_to_step[vid] = best[1]["id"]
                m.detail[vid] = (f"{len(cands)} steps share the value; name "
                                 f"agreement {best[0]:.2f} separated them")
            else:
                m.ambiguous.append({
                    "verifier": vid, "target": ev.get("value"),
                    "candidates": [n["id"] for n in cands],
                    "reason": "several steps produce this value; text cannot separate"})
            continue

        pool = judg or arith
        if not pool:
            m.unmatched.append({"verifier": vid, "kind": kind,
                                "reason": "no judgment step to attach to"})
            continue
        val_str = str(ev.get("value", "") or "")

        def _score(n: dict) -> float:
            side = f"{n.get('label','')} {n.get('ruling','')}"
            return max(name_agreement(val_str, side), name_agreement(vtext, side))

        scored = sorted(((_score(n), n) for n in pool), key=lambda x: -x[0])
        if scored[0][0] >= NAME_MIN:
            m.verifier_to_step[vid] = scored[0][1]["id"]
            m.detail[vid] = (f"{kind} verifier matched by name agreement "
                             f"{scored[0][0]:.2f}")
        else:
            m.unmatched.append({
                "verifier": vid, "kind": kind,
                "reason": f"no step resembles it (best {scored[0][0]:.2f})"})
            if scored[0][0] >= NEAR_MISS_MIN:
                m.near_misses.append({
                    "verifier": vid, "verifier_text": vtext[:160],
                    "candidate": scored[0][1]["id"],
                    "candidate_text": (f"{scored[0][1].get('label','')} "
                                       f"{scored[0][1].get('ruling','')}")[:160],
                    "agreement": round(scored[0][0], 3),
                    "reason": ("plausible but below threshold — a prose verifier "
                               "and its judgment step can share little vocabulary")})
    return m


# --------------------------------------------------------------------------
# Verifier DAG
# --------------------------------------------------------------------------

def derive_verifier_dag(step_graph: Dict[str, List[str]],
                        verifier_to_step: Dict[str, str],
                        all_verifier_ids: Optional[List[str]] = None
                        ) -> Dict[str, List[str]]:
    """A verifier's parents are the verifiers whose steps are its step's ancestors.

    EVERY verifier is a key, including ones mapped to no step. A verifier that
    tests no derivation step — a format requirement, or something the report must
    state — depends on nothing and is therefore a ROOT. Omitting it instead makes
    it vanish from the graph, so it gets no weight, and select_crux can still
    return it, at which point crux_shapley dies on the missing key.

    Only NEAREST parents are emitted; transitive closure is left to ancestors() in
    crux_shapley, which already computes it and is what select_crux uses.
    """
    anc = _ancestors(step_graph)
    step_to_vs: Dict[str, List[str]] = {}
    for vid, sid in (verifier_to_step or {}).items():
        step_to_vs.setdefault(sid, []).append(vid)

    keys = list(all_verifier_ids) if all_verifier_ids else list(
        verifier_to_step or {})
    for vid in (verifier_to_step or {}):
        if vid not in keys:
            keys.append(vid)
    dag: Dict[str, List[str]] = {vid: [] for vid in keys}
    for vid, sid in (verifier_to_step or {}).items():
        for a in anc.get(sid, set()):
            for pv in step_to_vs.get(a, []):
                if pv != vid and pv not in dag[vid]:
                    dag[vid].append(pv)
    v_anc = _ancestors(dag)
    for vid, parents in list(dag.items()):
        dag[vid] = sorted(p for p in parents
                          if not any(p in v_anc.get(q, set())
                                     for q in parents if q != p))
    return dag


# --------------------------------------------------------------------------
# Coverage: which derivation steps has nobody written a verifier for
# --------------------------------------------------------------------------

def step_coverage(nodes: Dict[str, dict],
                  step_graph: Dict[str, List[str]],
                  verifier_to_step: Dict[str, str]) -> dict:
    """The inverse of the verifier->step mapping.

    map_verifiers_to_steps answers "does this verifier test a step". This answers
    "does anybody test this step", which is the question that matters: an unwatched
    step is real work a response can skip or get wrong with nothing objecting.

    Read alongside verifier_mapping_report: a step can look unwatched only because
    the verifier that watches it failed to map.

    Two things are graded differently on purpose:
      * A step a TERMINAL depends on is load-bearing: the answer rests on it.
      * A step nothing depends on and no verifier watches is dead weight, and
        that is a defect in the derivation rather than in the verifier set.
    """
    watched: Dict[str, List[str]] = {}
    for vid, sid in (verifier_to_step or {}).items():
        watched.setdefault(sid, []).append(vid)

    consumed = {p for ps in step_graph.values() for p in ps}
    terminals = [k for k in step_graph if k not in consumed]
    anc = _ancestors(step_graph)
    load_bearing: Set[str] = set(terminals)
    for t in terminals:
        load_bearing |= anc.get(t, set())

    unwatched = []
    for sid, n in nodes.items():
        if sid in watched:
            continue
        unwatched.append({
            "step": sid, "kind": n.get("kind", "?"),
            "label": (n.get("label") or "")[:120], "value": n.get("value"),
            "solution_step": n.get("solution_step", ""),
            "load_bearing": sid in load_bearing,
            "is_terminal": sid in terminals,
        })
    unwatched.sort(key=lambda u: (not u["is_terminal"], not u["load_bearing"],
                                  u["step"]))
    return {
        "n_steps": len(nodes), "n_watched": len(watched),
        "n_unwatched": len(unwatched),
        "coverage": round(len(watched) / len(nodes), 4) if nodes else 1.0,
        "unwatched": unwatched,
        "unwatched_terminals": [u["step"] for u in unwatched if u["is_terminal"]],
        "unwatched_load_bearing": [u["step"] for u in unwatched
                                   if u["load_bearing"] and not u["is_terminal"]],
        "dead_weight": [u["step"] for u in unwatched if not u["load_bearing"]],
        "double_watched": {k: v for k, v in watched.items() if len(v) > 1},
    }