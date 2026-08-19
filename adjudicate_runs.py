"""Reconcile N pipeline runs of the SAME task into one final JSON + HTML.

The audit/augment stages are LLM calls and vary run to run: 3 vs 5 claims, V4 vs
V4a/V4b, a trap value of 3 one run and 1 the next, a split that appears or not.
Scoring a single run inherits whichever way that run fell. This module takes
several runs and produces one reconciled golden by SELECT AND MERGE — it chooses
among the real alternatives the runs produced and never authors new content.

Pipeline (agreed design):
  1. CLUSTER (master LLM): group each run's claims/verifiers into conceptual
     ROLES. A keyword key fragments ~6 real roles into ~15 (labels vary run to
     run), so clustering is semantic, done once by the LLM. The LLM only assigns
     existing items to groups; it writes no verifier text.
  2. VOTE within each cluster (deterministic): the modal value / text wins, and
     the tally is recorded. Majority = >= ceil(N/2) agreeing.
  3. ADJUDICATE VALUES (master LLM, can OVERRIDE majority): majority counts, it
     does not judge correctness — if the model repeats a mistake in most runs,
     majority entrenches it (observed: 2 runs say wait=38, 1 says 42; 38 is
     wrong). So for each value the LLM is shown the derivation context and the
     vote, and may override the majority. Every override is logged with its
     reason, in BOTH the JSON (`adjudication.overrides`) and the HTML.
  4. MERGE + EMIT: assemble the winning items into one package, re-id verifiers
     V1..Vn, and render an HTML with the decision + override log visible.

The master LLM does two jobs (cluster, adjudicate) — both SELECT/JUDGE over real
run content, never author. Each is one non-deterministic call; to make the
adjudicator itself reproducible, run it K times and majority-vote its outputs
(a later option, not done here). Both LLM entry points accept an injected
callable, so the module is fully testable without a live model.
"""
from __future__ import annotations

import html as _html
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


def _majority_needed(n: int) -> int:
    return (n // 2) + 1


def _num_key(v) -> Optional[float]:
    try:
        return round(float(v), 4)
    except (TypeError, ValueError):
        return None


def _first_number(text: str) -> Optional[float]:
    """The first standalone number in a verifier's text — the value it asserts.
    Used to tell a value disagreement ('wait = 38' vs '= 42') from a pure wording
    difference. Skips ids like 'V3' and pure ordinals."""
    for m in re.finditer(r"(?<![A-Za-z])(\d[\d,]*\.?\d*)", text or ""):
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            continue
    return None


def _vote(values: List, n_runs: int) -> Tuple[Optional[object], bool, dict]:
    """(winner, has_majority, tally). Numeric values compared by rounded key."""
    norm = []
    for v in values:
        nk = _num_key(v)
        norm.append(nk if nk is not None else str(v).strip())
    tally = Counter(norm)
    if not tally:
        return None, False, {}
    winner, count = tally.most_common(1)[0]
    return winner, count >= _majority_needed(n_runs), {str(k): c for k, c in tally.items()}


_ROLE_VOCAB = [
    "utilization", "offered load", "minimum server", "required", "fte", "clerk",
    "redesign", "residual", "wait", "recommend", "hire", "stability", "wage",
    "market", "target", "queue", "erlang", "arrival", "service", "capacity",
]


def _keyword_role(label: str) -> str:
    t = re.sub(r"[^a-z0-9 ]", " ", (label or "").lower())
    keys = sorted({kw for kw in _ROLE_VOCAB if kw in t})
    if keys:
        return "|".join(keys)
    toks = [w for w in t.split() if len(w) > 3]
    return "|".join(toks[:3]) or "_empty_"


@dataclass
class Run:
    idx: int
    data: dict

    @property
    def gate_ok(self) -> bool:
        return bool((self.data.get("gate") or {}).get("passed"))

    @property
    def claims(self) -> List[dict]:
        return self.data.get("corrected_claim_verdicts") or []

    def verifiers(self) -> List[dict]:
        out = []
        for line in (self.data.get("augmented_verifiers_text") or "").splitlines():
            m = re.match(r"\s*(V\d+[a-z]?)\s*:\s*(.*)", line)
            if m:
                out.append({"id": m.group(1), "text": m.group(2).strip()})
        return out


@dataclass
class Adjudication:
    n_runs: int = 0
    gate_ok_runs: List[int] = field(default_factory=list)
    clustering_method: str = ""
    claim_clusters: List[dict] = field(default_factory=list)
    verifier_clusters: List[dict] = field(default_factory=list)
    majority_decisions: List[dict] = field(default_factory=list)
    overrides: List[dict] = field(default_factory=list)
    residual_decisions: List[dict] = field(default_factory=list)
    dropped_minority: List[dict] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in (
            "n_runs", "gate_ok_runs", "clustering_method", "claim_clusters",
            "verifier_clusters", "majority_decisions", "overrides",
            "residual_decisions", "dropped_minority", "notes")}


def _loads(raw: str):
    if not raw:
        return None
    s = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, re.S)
        try:
            return json.loads(m.group(0)) if m else None
        except (json.JSONDecodeError, AttributeError):
            return None


def _cluster_with_llm(items, kind, llm_cluster) -> Dict[int, str]:
    listing = "\n".join(f"{i}: {it['label']}" for i, it in enumerate(items))
    prompt = (
        f"These are {kind} drawn from several runs of the SAME task. Group them by "
        f"the CONCEPTUAL ROLE each one checks — the quantity or claim it is about, "
        f"NOT its wording.\n"
        f"CRITICAL: merge DIFFERENT NAMES for the SAME quantity into one group. "
        f"Runs phrase the same thing differently, e.g. 'utilization with 3 clerks' "
        f"and 'stability check with 3 clerks' are the SAME role (both are rho at "
        f"3 servers); 'offered load' and 'arrival load in Erlangs' are the same; "
        f"'required FTE' and 'minimum feasible servers' are the same. If two items "
        f"compute or assert the same underlying quantity, they go in ONE group "
        f"even if the words differ. Prefer FEWER, broader groups over many narrow "
        f"ones. Do NOT rewrite any item.\n"
        f"{listing}\n\n"
        'Return ONLY JSON: {"groups":[{"role":"short name","items":[indices]}]}. '
        "Every index appears in exactly one group.")
    obj = _loads(llm_cluster(prompt))
    mapping: Dict[int, str] = {}
    for g in (obj or {}).get("groups", []):
        role = str(g.get("role") or "").strip() or "unnamed"
        for i in g.get("items", []):
            if isinstance(i, int) and 0 <= i < len(items):
                mapping[i] = role
    return mapping


def _group(items, kind, llm_cluster, value_of=None) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    if llm_cluster and items:
        try:
            mapping = _cluster_with_llm(items, kind, llm_cluster)
        except Exception:                                       # noqa: BLE001
            mapping = {}
        if len(mapping) == len(items):
            for i, role in mapping.items():
                groups[role].append(i)
            groups = _value_postmerge(dict(groups), items, value_of)
            return groups
    for i, it in enumerate(items):
        groups[_keyword_role(it["label"])].append(i)
    return _value_postmerge(dict(groups), items, value_of)


def _value_postmerge(groups: Dict[str, List[int]], items,
                     value_of) -> Dict[str, List[int]]:
    """Deterministic safety net for under-merged clustering. Two clusters that
    carry the SAME computed value are the same quantity the clusterer split under
    different names ('utilization with 3 clerks' vs 'stability check with 3
    clerks', both rho=1.111). Merge clusters whose value-sets overlap on a shared
    value. Only runs when a value_of(item)->number extractor is supplied (claims,
    trap values); text-only items (verifiers) are left to the LLM + keyword key.
    """
    if value_of is None or len(groups) < 2:
        return groups
    # signature of each group = the set of rounded values it contains, plus the
    # set of runs it spans (to avoid over-merging two DIFFERENT quantities that
    # coincidentally share a value)
    sig: Dict[str, set] = {}
    runs_of: Dict[str, set] = {}
    for role, idxs in groups.items():
        vals = set()
        rns = set()
        for i in idxs:
            v = value_of(items[i])
            if v is not None:
                vals.add(round(float(v), 4))
            ref = items[i].get("ref")
            if isinstance(ref, tuple):
                rns.add(ref[0])
        sig[role] = vals
        runs_of[role] = rns
    roles = list(groups)
    parent = {r: r for r in roles}

    def find(r):
        while parent[r] != r:
            parent[r] = parent[parent[r]]
            r = parent[r]
        return r

    for a in range(len(roles)):
        for b in range(a + 1, len(roles)):
            ra, rb = roles[a], roles[b]
            # merge only if they share a concrete value AND do NOT co-occur in the
            # same run — co-occurrence means they are distinct claims in that run
            # (e.g. two real quantities that happen to equal), not one role the
            # clusterer split across runs.
            if (sig[ra] and sig[rb] and (sig[ra] & sig[rb])
                    and not (runs_of[ra] & runs_of[rb])):
                parent[find(rb)] = find(ra)
    merged: Dict[str, List[int]] = defaultdict(list)
    # keep the name of the largest original cluster in each merged component
    comp_name: Dict[str, str] = {}
    for r in sorted(roles, key=lambda x: -len(groups[x])):
        comp_name.setdefault(find(r), r)
    for r in roles:
        merged[comp_name[find(r)]].extend(groups[r])
    return dict(merged)


def adjudicate(
    run_jsons: List[dict],
    llm_cluster: Optional[Callable[[str], str]] = None,
    llm_judge: Optional[Callable[[str], str]] = None,
) -> Tuple[dict, Adjudication]:
    """Reconcile runs into one final package + an Adjudication log."""
    if len(run_jsons) < 3:
        raise ValueError("adjudication needs at least 3 runs (default 5)")

    runs = [Run(i, d) for i, d in enumerate(run_jsons)]
    n = len(runs)
    adj = Adjudication(n_runs=n, gate_ok_runs=[r.idx for r in runs if r.gate_ok])
    value_runs = [r for r in runs if r.gate_ok] or runs
    vr_idx = {r.idx for r in value_runs}
    if len(value_runs) < n:
        adj.notes.append(f"{n - len(value_runs)} run(s) failed the gate; kept for "
                         f"structural votes, excluded from value votes")
    adj.clustering_method = "llm" if llm_cluster else "keyword_fallback"

    final: dict = {"task_id": run_jsons[0].get("task_id", ""),
                   "adjudicated": True, "n_runs": n}

    # CLAIMS
    claim_items = [{"ref": (r.idx, c), "label": c.get("label", "")}
                   for r in runs for c in r.claims]
    final_claims = []
    for role, idxs in _group(claim_items, "claims", llm_cluster,
                             value_of=lambda it: it["ref"][1].get("recomputed")).items():
        occ = [claim_items[i]["ref"] for i in idxs]
        present = {ri for ri, _ in occ}
        adj.claim_clusters.append({"role": role, "runs_present": sorted(present)})
        if len(present) < _majority_needed(n):
            adj.dropped_minority.append(
                {"kind": "claim", "role": role, "runs_present": sorted(present)})
            continue
        vals = [c.get("recomputed") for ri, c in occ
                if ri in vr_idx and c.get("recomputed") is not None]
        winner, has_maj, tally = _vote(vals, len(value_runs)) if vals else (None, False, {})
        chosen = winner
        if vals:
            chosen = _judge_value(adj, llm_judge, role=role, tally=tally,
                                  majority=winner, has_majority=has_maj,
                                  occ=occ, kind="claim_value")
        final_claims.append(_rep(occ, vr_idx, chosen))
    final["corrected_claim_verdicts"] = final_claims

    # VERIFIERS
    ver_items = [{"ref": (r.idx, v), "label": v["text"]}
                 for r in runs for v in r.verifiers()]
    final_verifiers = []
    for role, idxs in _group(ver_items, "verifiers", llm_cluster).items():
        occ = [ver_items[i]["ref"] for i in idxs]
        present = {ri for ri, _ in occ}
        adj.verifier_clusters.append({"role": role, "runs_present": sorted(present)})
        if len(present) < _majority_needed(n):
            adj.dropped_minority.append(
                {"kind": "verifier", "role": role, "runs_present": sorted(present)})
            continue
        texts = [v["text"] for _, v in occ]
        winner, has_maj, tally = _vote(texts, n)
        # Does the disagreement carry a NUMERIC value that differs across
        # phrasings? If so it is a value question, not a wording one — the model
        # can embed a wrong number in most runs' phrasings (observed: "wait = 38"
        # in 2 runs, "= 42" in 1; 38 is the planted error). Route it through value
        # judgment, which can OVERRIDE the majority, rather than picking the modal
        # phrasing (which would entrench 38). A pure wording difference (same
        # numbers) stays on the phrasing path.
        nums = [_first_number(t) for t in dict.fromkeys(texts)]
        distinct_nums = {x for x in nums if x is not None}
        if len(distinct_nums) > 1:
            num_tally = _vote([_first_number(t) for t in texts], n)[2]
            num_winner, num_maj, _ = _vote([_first_number(t) for t in texts], n)
            picked_val = _judge_value(
                adj, llm_judge, role=f"{role} (embedded value)", tally=num_tally,
                majority=num_winner, has_majority=num_maj, occ=occ,
                kind="verifier_embedded_value")
            # keep the phrasing whose number matches the adjudicated value
            chosen = next((t for _, v in occ for t in [v["text"]]
                           if _num_key(_first_number(t)) == _num_key(picked_val)),
                          winner)
        elif has_maj:
            chosen = winner
            adj.majority_decisions.append(
                {"kind": "verifier_text", "role": role, "text": winner,
                 "tally": tally})
        else:
            chosen = _judge_text(adj, llm_judge, role=role, tally=tally,
                                 options=texts)
        final_verifiers.append({"role": role, "text": chosen})
    final["augmented_verifiers_text"] = "\n".join(
        f"V{i+1}: {v['text']}" for i, v in enumerate(final_verifiers))
    final["adjudicated_verifier_roles"] = [v["role"] for v in final_verifiers]

    # STRUCTURAL
    for kind, series in (
            ("any_split", [bool(r.data.get("verifier_splits_applied")) for r in runs]),
            ("verdict", [r.data.get("audit_verdict") for r in runs])):
        w, maj, tally = _vote(series, n)
        adj.majority_decisions.append(
            {"kind": kind, "value": w, "majority": maj, "tally": tally})
        if kind == "verdict":
            final["audit_verdict"] = w

    # TRAP VALUES
    trap_items = [{"ref": (r.idx, c), "label": c.get("label", "")}
                  for r in value_runs for c in r.claims
                  if c.get("trap_value") is not None]
    trap_final = {}
    for role, idxs in _group(trap_items, "trap values", llm_cluster,
                             value_of=lambda it: it["ref"][1].get("trap_value")).items():
        occ = [trap_items[i]["ref"] for i in idxs]
        vals = [c.get("trap_value") for _, c in occ]
        w, maj, tally = _vote(vals, len(value_runs))
        chosen = _judge_value(adj, llm_judge, role=role, tally=tally, majority=w,
                              has_majority=maj, occ=occ, kind="trap_value")
        trap_final[role] = {"value": chosen, "tally": tally}
    final["adjudicated_trap_values"] = trap_final

    # JUDGMENT STEPS: align by role, keep a representative from a value run ----
    j_items = [{"ref": (r.idx, j), "label": j.get("question", "")}
               for r in runs for j in (r.data.get("judgment_steps") or [])]
    final_judgments = []
    for role, idxs in _group(j_items, "judgment steps", llm_cluster).items():
        occ = [j_items[i]["ref"] for i in idxs]
        present = {ri for ri, _ in occ}
        if len(present) < _majority_needed(n):
            adj.dropped_minority.append(
                {"kind": "judgment", "role": role, "runs_present": sorted(present)})
            continue
        rep = next((dict(j) for ri, j in occ if ri in vr_idx), dict(occ[0][1]))
        final_judgments.append(rep)
    final["judgment_steps"] = final_judgments

    # Carry the fields a full re-derivation + SME report need. These are taken
    # from a representative gate-ok run (structure is reconciled; these are the
    # supporting artifacts the report renders). The re-derivation below rebuilds
    # dag / weights / crux / trajectory from the MERGED claims+verifiers, so the
    # adjudicated HTML has the same sections a single-run report does.
    rep_run = next((r for r in value_runs), runs[0]).data
    # expected_values must be re-derived from the MERGED, re-ided verifier text —
    # the runs' own expected_values key on their old ids (V4a etc.) and would be
    # dropped by the re-derivation's id filter. derive_expected_values reads the
    # canonical text and produces fresh V1..Vn targets that match the merged set.
    try:
        from src.verifier_grammar import derive_expected_values
        ev, _ = derive_expected_values(final["augmented_verifiers_text"])
        final["expected_values"] = ev
    except Exception:                                           # noqa: BLE001
        final["expected_values"] = {}
    final["trap_anchor_ids"] = rep_run.get("trap_anchor_ids", [])
    final["expert_anchor_ids"] = rep_run.get("expert_anchor_ids", [])
    final["gold_deliverable_format"] = rep_run.get("gold_deliverable_format", "")
    final["gold_deliverable_sections"] = rep_run.get("gold_deliverable_sections", [])
    final["gold_deliverable_text"] = rep_run.get("gold_deliverable_text", "")
    final["model_used"] = rep_run.get("model_used", "")
    final["input_coverage"] = rep_run.get("input_coverage", {})

    final["adjudication"] = adj.to_dict()
    return final, adj


def _rep(occ, vr_idx, chosen):
    key = _num_key(chosen)
    if key is not None:
        for ri, c in occ:
            if ri in vr_idx and _num_key(c.get("recomputed")) == key:
                return dict(c)
    for ri, c in occ:
        if ri in vr_idx:
            return dict(c)
    return dict(occ[0][1])


def _judge_value(adj, llm_judge, *, role, tally, majority, has_majority, occ, kind):
    if llm_judge is None:
        if not has_majority:
            adj.residual_decisions.append(
                {"kind": kind, "role": role, "tally": tally,
                 "resolved_by": "majority_fallback", "choice": str(majority)})
        return majority
    sample = occ[0][1] if occ else {}
    ctx = (f"role={role}; operation={sample.get('operation','')!r}; "
           f"inputs={[(i.get('name'), i.get('value')) for i in (sample.get('input_provenance') or [])]}")
    prompt = (
        "Independent runs of the same task disagree on a value. Majority vote is "
        "NOT proof of correctness — the model can repeat a mistake. Judge which "
        "option is CORRECT from the derivation, and pick it FROM THE LIST (do not "
        "invent a value).\n"
        f"{kind} for {ctx}\n"
        f"Options (value: #runs): {tally}\n"
        f"Majority-by-count = {majority}.\n"
        'Return ONLY JSON: {"choice": <one option value>, "reason": "...", '
        '"overrides_majority": true/false}.')
    obj = _loads(llm_judge(prompt)) or {}
    choice = obj.get("choice")
    if choice is None or str(choice) not in {str(k) for k in tally}:
        adj.residual_decisions.append(
            {"kind": kind, "role": role, "tally": tally,
             "resolved_by": "llm_off_menu_rejected", "llm_said": choice,
             "choice": str(majority)})
        return majority
    if str(choice) != str(majority):
        adj.overrides.append(
            {"kind": kind, "role": role, "tally": tally,
             "majority_was": str(majority), "chosen": str(choice),
             "reason": obj.get("reason", "")})
    else:
        adj.majority_decisions.append(
            {"kind": kind, "role": role, "value": str(choice), "tally": tally,
             "llm_confirmed": True})
    return _num_key(choice) if _num_key(choice) is not None else choice


def _judge_text(adj, llm_judge, *, role, tally, options):
    if llm_judge is None:
        w = Counter(options).most_common(1)[0][0]
        adj.residual_decisions.append(
            {"kind": "verifier_text", "role": role, "tally": tally,
             "resolved_by": "majority_fallback", "choice": w})
        return w
    prompt = (
        "Independent runs phrase the same verifier differently. Pick the single "
        "clearest, most precise phrasing FROM THE LIST — do not rewrite it.\n"
        f"role={role}\n" + "\n".join(f"- {t}" for t in dict.fromkeys(options)) +
        '\nReturn ONLY JSON: {"choice": "<exact option text>"}.')
    obj = _loads(llm_judge(prompt)) or {}
    choice = obj.get("choice")
    if choice in options:
        adj.residual_decisions.append(
            {"kind": "verifier_text", "role": role, "resolved_by": "llm",
             "choice": choice})
        return choice
    w = Counter(options).most_common(1)[0][0]
    adj.residual_decisions.append(
        {"kind": "verifier_text", "role": role,
         "resolved_by": "llm_off_menu_rejected", "choice": w})
    return w


def _pick_representative(run_jsons: List[dict]) -> Tuple[int, dict]:
    """Choose the run whose STRUCTURE (trajectory/DAG) is the skeleton for the
    adjudicated report. Ranked by: gate passed, scoreable, has a derived DAG,
    crux size, then claim+judgment count. The runs disagree structurally and no
    merged trajectory is coherent (their claim-id spaces differ), so one real
    run's trajectory is used as the skeleton and the adjudicated verifiers/values
    are overlaid on it (Option A)."""
    def score(d):
        return (
            1 if (d.get("gate") or {}).get("passed") else 0,
            1 if d.get("scoreable") else 0,
            1 if d.get("dag") else 0,
            len(d.get("crux_ids") or []),
            len(d.get("corrected_claim_verdicts") or [])
            + len(d.get("judgment_steps") or []),
        )
    best_i = max(range(len(run_jsons)), key=lambda i: score(run_jsons[i]))
    return best_i, run_jsons[best_i]


def build_sme_package(final: dict, adj: Adjudication,
                      run_jsons: Optional[List[dict]] = None) -> dict:
    """Option A1: use a representative run as a COHERENT WHOLE — its trajectory,
    claims, judgment steps, verifiers, and DAG — and apply only the adjudicated
    VALUE OVERRIDES (e.g. a claim that read 38 becomes 42) and the majority
    VERDICT on top.

    Why the whole run, not the merged verifier set: a verifier DAG only exists
    when the verifiers were derived against those specific claims. The merged
    verifiers come from many runs and have no single claim graph to attach to, so
    overlaying them flattens the DAG. Using one run's verifiers keeps the
    dependency graph intact and the document coherent for the SME. The
    reconciliation that most affects correctness — the value overrides — is still
    applied, and the banner records the full adjudication (including the merged
    verifier set and every override) so nothing is hidden.
    """
    if not run_jsons:
        pkg = dict(final)
    else:
        rep_i, rep = _pick_representative(run_jsons)
        pkg = dict(rep)                        # the representative run, whole
        adj.notes.append(f"structure + verifiers from run index {rep_i} "
                         f"(representative); value overrides + verdict applied")
        pkg["_structure_from_run"] = rep_i
        # The persisted run stores the anchors under the OUTPUT keys
        # (crux_anchors_trap / crux_anchors_expert) but derive_frozen_graph reads
        # the INPUT keys (trap_anchor_ids / expert_anchor_ids). Without restoring
        # them, the re-derivation gets no anchors and select_crux collapses the
        # crux to just the final-answer verifier(s) — turning a 4-verifier crux
        # into 1. Map the persisted anchors back to the input fields so the
        # rebuild reproduces the representative run's actual crux.
        if not pkg.get("trap_anchor_ids") and pkg.get("crux_anchors_trap"):
            pkg["trap_anchor_ids"] = pkg["crux_anchors_trap"]
        if not pkg.get("expert_anchor_ids") and pkg.get("crux_anchors_expert"):
            pkg["expert_anchor_ids"] = pkg["crux_anchors_expert"]
        # majority verdict (structural) overlays the representative's
        pkg["audit_verdict"] = final.get("audit_verdict", pkg.get("audit_verdict"))
        pkg["task_id"] = final.get("task_id", pkg.get("task_id"))
        # keep the merged set + overrides visible in the package for the banner /
        # downstream, WITHOUT replacing the representative's own verifiers
        pkg["adjudicated_verifier_set"] = final.get("augmented_verifiers_text", "")
        pkg["adjudicated_trap_values"] = final.get("adjudicated_trap_values", {})
        # apply value overrides into BOTH the representative's claim values and,
        # where the value appears in its verifier text, the verifier text — so the
        # coherent trajectory shows the adjudicated (corrected) numbers
        _apply_overrides_to_claims(pkg, adj)
        _apply_overrides_to_verifier_text(pkg, adj)

    try:
        from src.augment_task import derive_frozen_graph
        pkg = derive_frozen_graph(pkg, compute_shapley=True)
    except Exception as e:                                       # noqa: BLE001
        pkg["error"] = f"adjudicated re-derivation failed: {e}"
    pkg["_adjudication"] = adj.to_dict()
    pkg["adjudicated"] = True
    return pkg


def _apply_overrides_to_verifier_text(pkg: dict, adj: Adjudication):
    """Where a value override's old number appears in the representative run's
    verifier text, replace it with the adjudicated value — so the verifier the
    SME reads carries the corrected figure (e.g. 'wait = 38' -> 'wait = 42') while
    keeping its DAG position. Only replaces a standalone number to avoid touching
    ids or unrelated figures."""
    text = pkg.get("augmented_verifiers_text") or ""
    if not text:
        return
    for ov in adj.overrides:
        mv, cv = ov.get("majority_was"), ov.get("chosen")
        mvf, cvf = _num_key(mv), _num_key(cv)
        if mvf is None or cvf is None:
            continue
        # format the numbers as they'd appear (int if whole)
        old_s = str(int(mvf)) if mvf == int(mvf) else str(mvf)
        new_s = str(int(cvf)) if cvf == int(cvf) else str(cvf)
        text = re.sub(rf"(?<![.\d]){re.escape(old_s)}(?![.\d])", new_s, text)
    pkg["augmented_verifiers_text"] = text


def _apply_overrides_to_claims(pkg: dict, adj: Adjudication):
    """Write each value override into the representative run's matching claim, so
    the skeleton trajectory shows the adjudicated value (e.g. a claim that read 38
    becomes 42). Matched by the overridden 'majority_was' value; unmatched
    overrides are left for the banner to surface."""
    claims = pkg.get("corrected_claim_verdicts") or []
    for ov in adj.overrides:
        mv = _num_key(ov.get("majority_was"))
        cv = _num_key(ov.get("chosen"))
        if mv is None or cv is None:
            continue
        for c in claims:
            if _num_key(c.get("recomputed")) == mv:
                c["recomputed"] = cv
                c["_adjudicated_from"] = ov.get("majority_was")
                c["_adjudicated_reason"] = ov.get("reason", "")


def _override_banner(adj: Adjudication) -> str:
    """An SME-facing banner listing where the adjudicator overrode the majority,
    plus the reconciliation summary. Prepended to the standard report."""
    parts = [
        "<div style='border:1px solid #e7e2d8;border-radius:8px;padding:12px 14px;"
        "margin:10px 0;background:#fff'>",
        f"<b>Adjudicated from {adj.n_runs} runs</b> "
        f"(gate-ok: {_html.escape(str(adj.gate_ok_runs))}; "
        f"clustering: {_html.escape(adj.clustering_method)}). "
        f"Trajectory/DAG shown is from the representative run; verifiers and "
        f"values below are the reconciled (adjudicated) set.",
    ]
    if adj.overrides:
        parts.append("<p style='margin:8px 0 4px'><b>Value overrides "
                     "(adjudicator judged against the majority):</b></p>")
        for ov in adj.overrides:
            parts.append(
                f"<div style='background:#fdecea;border-left:4px solid #b4413c;"
                f"padding:6px 10px;margin:5px 0;border-radius:6px'>"
                f"<span class=mono>{_html.escape(str(ov['role']))}</span> — "
                f"majority-by-count was <b>{_html.escape(str(ov['majority_was']))}"
                f"</b>, chose <b>{_html.escape(str(ov['chosen']))}</b> · "
                f"tally {_html.escape(str(ov['tally']))}<br>"
                f"<i>{_html.escape(str(ov.get('reason','')))}</i></div>")
    else:
        parts.append(" No value overrides — majority and adjudicator agreed.")
    if adj.dropped_minority:
        drops = ", ".join(f"{d['kind']}:{d['role']}" for d in adj.dropped_minority)
        parts.append(f"<p style='margin:6px 0 0;color:#9a7400'><b>Dropped "
                     f"(minority, too few runs):</b> {_html.escape(drops)}</p>")
    parts.append("</div>")
    return "".join(parts)


def render_html(final: dict, adj: Adjudication, out_path: Optional[str] = None,
                run_jsons: Optional[List[dict]] = None,
                pkg: Optional[dict] = None) -> str:
    """Render the adjudicated golden as an SME report IDENTICAL in shape to a
    per-run augment report, with an override banner prepended. If out_path is
    given, writes there; always returns the HTML string.

    Pass `pkg` (the output of build_sme_package) to render the SAME object saved
    as JSON — keeps adjudicated.json and adjudicated.html in sync and avoids
    building the package twice. If None, builds from final + run_jsons (Option
    A1)."""
    import os
    import tempfile
    if pkg is None:
        pkg = build_sme_package(final, adj, run_jsons=run_jsons)
    try:
        from src.augment_report import write_augment_report
        tmp = out_path or os.path.join(tempfile.mkdtemp(), "adjudicated.html")
        write_augment_report(pkg, tmp)
        html = open(tmp, encoding="utf-8").read()
    except Exception as e:                                       # noqa: BLE001
        # fallback: minimal doc so we never lose the adjudication if the shared
        # renderer is unavailable
        html = (f"<!doctype html><meta charset=utf-8><h1>"
                f"{_html.escape(str(final.get('task_id')))} — adjudicated</h1>"
                f"<pre>{_html.escape(json.dumps(final, indent=1, default=str))}</pre>"
                f"<p>renderer unavailable: {_html.escape(str(e))}</p>")
        if out_path:
            open(out_path, "w", encoding="utf-8").write(html)
        return html
    # inject the override banner right after the <h1> line
    banner = _override_banner(adj)
    marker = "</h1>"
    idx = html.find(marker)
    if idx != -1:
        cut = idx + len(marker)
        html = html[:cut] + "\n" + banner + html[cut:]
    if out_path:
        open(out_path, "w", encoding="utf-8").write(html)
    return html
