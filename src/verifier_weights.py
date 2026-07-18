# src/verifier_weights.py
"""
DAG-weighted scoring, AMZN-aligned weighting, cascade analysis, redundancy
detection, and the 5-dimension classifier.

PORTED (not imported) from the gdpval-sample-generator skill's
golden_sample_pipeline_v3.py so the pipeline has no runtime dependency on the
skill directory. This is the pure-logic core only — the skill's docx builder and
xlsx extraction are intentionally NOT ported (the augmenter writes V{n}: text,
not golden-sample docx tables).

The 5 dimensions:
  DI Data Integrity | AR Analytical Rigor | RF Relevance & Focus
  EP Execution Precision | FD Format & Deliverability
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Dimension classifier (patterns ported verbatim from the skill)
# ---------------------------------------------------------------------------

DIM_PATTERNS = {
    'FD': r'\bformat\b|\bsection\b|\bstructur\b|\boutput\b|\bdeliver\b|\bslide\b|\bmemo\b|\bpage\b|\bcolumn.*order\b|\brow.*count\b|\bfile.*name\b',
    'RF': r'\brelevant\b|\bfocus\b|\btangential\b|\bnoise\b|\bscope\b|\bexternal\b.*\bnot\b|\bonly\b.*\bprovided\b|\bexclud\b|\bignor\b.*\bnoise\b|\bno.*web\b',
    'EP': r'\bcalculat\b|\bsum\b|\bequal\b|\brange\b|\b±\b|[≥≤<>]|\bpercent\b|\barithmetic\b|\bcost\b.*[=≈]|\birr\b|\bformula\b|\bprofit\b.*[=≈]|\bsaving\b.*[=≈]|\btotal\b.*[=≈]|\b\d+\.\d{2,}\b',
    'AR': r'\branking\b|\bdriver\b|\bdominant\b|\bidentif\b|\bconclud\b|\brecommend\b|\bdecision\b|\bgo.*no.go\b|\baccept\b|\breject\b|\bapprove\b|\bbottleneck\b|\bconstraint\b|\bstrategy\b',
    'DI': r'\brate\b|\bvalue\b|\bdata\b|\bsource\b|\bcorrelat\b|\bcolumn\b|\bmatch\b|\bunit\b|\bfetch\b|\bretriev\b|\bextract\b|\buse.*file\b|\bplaceholder\b',
}


def classify_dim(text: str) -> str:
    """Classify a verifier criterion into one of the 5 dimensions."""
    t = (text or "").lower()
    for dim, pattern in DIM_PATTERNS.items():
        if re.search(pattern, t):
            return dim
    if re.search(r'\d+%|\d+\.\d+', t):
        return 'EP'
    return 'AR'


# ---------------------------------------------------------------------------
# Weights (DAG depth-proportional + AMZN tier-based), summing to exactly 100%
# ---------------------------------------------------------------------------

def compute_weights(all_vs: List[dict], dag: Dict[str, list]) -> Tuple[dict, dict, dict]:
    """Return (dag_weights, amzn_weights, depths). Each weight dict sums to 100.0.

    all_vs: list of verifier dicts, each with at least 'id' and optionally 'type'.
    dag:    {verifier_id: [dependency_ids]}.
    """
    def depth(node, memo):
        if node in memo:
            return memo[node]
        deps = dag.get(node, [])
        if not deps:
            memo[node] = 0
            return 0
        d = max((depth(dep, memo) + 1 for dep in deps if dep in dag), default=0)
        memo[node] = d
        return d

    memo: Dict[str, int] = {}
    depths = {v['id']: depth(v['id'], memo) for v in all_vs}
    max_d = max(depths.values()) if depths else 1

    # DAG weights: proportional to (depth + 1) — deeper verifiers weigh more.
    raw = {v['id']: depths[v['id']] + 1 for v in all_vs}
    tot = sum(raw.values()) or 1
    dw = {k: round(v / tot * 100, 1) for k, v in raw.items()}

    # AMZN weights: tier-based. final synthesis 6x, deliverable 4x(t2->4),
    # analytical 3.5x, data extraction 2x. (tiers map: 1:6,2:4,3:3.5,4:2)
    tiers = {1: 6, 2: 4, 3: 3.5, 4: 2}
    raw_a = {}
    for v in all_vs:
        d = depths[v['id']]
        if v.get('type') == 'TRAP':
            t = 3
        elif v['id'].startswith('FD'):
            t = 2
        elif d >= max_d - 1 and d > 0:
            t = 1
        elif d > 0:
            t = 3
        else:
            t = 4
        raw_a[v['id']] = tiers[t]
    tot_a = sum(raw_a.values()) or 1
    aw = {k: round(v / tot_a * 100, 1) for k, v in raw_a.items()}

    # Force each to exactly 100% by adjusting the last entry.
    for w in (dw, aw):
        if w:
            diff = 100.0 - sum(w.values())
            last = list(w.keys())[-1]
            w[last] = round(w[last] + diff, 1)

    return dw, aw, depths


# ---------------------------------------------------------------------------
# Cascade analysis — transitive downstream impact of each root
# ---------------------------------------------------------------------------

def cascade_analysis(dag: Dict[str, list], all_ids: List[str]) -> List[tuple]:
    """Return [(root_id, [downstream_ids], count)] sorted by descending impact."""
    downstream = defaultdict(set)
    for vid, deps in dag.items():
        for dep in deps:
            downstream[dep].add(vid)

    def full_cascade(root, visited=None):
        if visited is None:
            visited = set()
        for child in downstream.get(root, []):
            if child not in visited:
                visited.add(child)
                full_cascade(child, visited)
        return visited

    roots = [vid for vid in all_ids if not dag.get(vid, [])]
    cascades = []
    for r in sorted(roots):
        fc = full_cascade(r)
        if fc:
            cascades.append((r, sorted(fc), len(fc)))
    return sorted(cascades, key=lambda x: -x[2])


# ---------------------------------------------------------------------------
# Redundancy detection (DAG-aware, 4-axis) — ported and simplified
# ---------------------------------------------------------------------------

def _extract_numbers(text: str) -> set:
    return set(re.findall(r'-?\d+\.?\d*', text or ""))


def _keywords(text: str) -> set:
    return set(re.findall(r'[a-z]{4,}', (text or "").lower()))


def _text_jaccard(a: str, b: str) -> float:
    ka, kb = _keywords(a), _keywords(b)
    if not ka or not kb:
        return 0.0
    return len(ka & kb) / len(ka | kb)


def detect_redundancy(all_vs: List[dict], dag: Dict[str, list],
                      text_threshold: float = 0.55) -> List[dict]:
    """Flag verifier pairs that look redundant (same dim + high text overlap +
    same numbers). Conservative — reports as advisory, does not delete."""
    issues = []
    n = len(all_vs)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = all_vs[i], all_vs[j]
            if a.get('dim') != b.get('dim'):
                continue
            jac = _text_jaccard(a.get('criterion', ''), b.get('criterion', ''))
            nums_a = _extract_numbers(a.get('criterion', ''))
            nums_b = _extract_numbers(b.get('criterion', ''))
            same_nums = bool(nums_a) and nums_a == nums_b
            if jac >= text_threshold and same_nums:
                issues.append({
                    "pair": [a['id'], b['id']],
                    "jaccard": round(jac, 2),
                    "message": f"Possible redundant pair {a['id']}/{b['id']} "
                               f"(same dim {a.get('dim')}, text overlap {jac:.2f}, same numbers)",
                })
    return issues
