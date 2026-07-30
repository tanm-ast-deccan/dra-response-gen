#!/usr/bin/env python3
"""
DRA crux selection + GEAR scoring.
Implements SKILL_crux golden-anchored procedure (encoded as deterministic rules
over expected_values + golden text, with human-review flags) and the
crux_gear_score transitive-closure + soft-suppression scorer (HANDOFF sec 4).

Selection rules (from SKILL_crux "The definition" + "Apply the exclusions"):
  Candidate crux = verifiers whose value the golden REPORTS AS AN ANSWER:
    computed results (numeric/decision) the memo states as conclusions.
  EXCLUDE:
    - sov == 'source_file'         -> given input (strongest exclusion)
    - kind == 'string' AND value looks like filename/format/section/tautology
    - decision values that are RULE-FLAGS (NO_*, EXPANSION_*, *_BRANCH, EXCLUDED
      as a process instruction) rather than the final recommendation
    - tautologies (numeric value in {0,1,100} with trivial meaning, e.g. prob sum=1)
    - restatement-only strings (the 'formula' text verifiers)
  KEEP:
    - numeric with sov in {arithmetic, llm_judgment} that appears as a reported figure
    - the final decision/recommendation (one, dedup cosmetic equivalents)
  FLAG (keep-but-review):
    - arithmetic numeric whose value equals a trivially-computed input (e.g. 2.5x10)
    - decision restatement of an already-selected numeric comparison
  GATE:
    - scoreable == False -> SKIP
    - any expected_values[v].sov == 'judgment_flagged' -> SKIP whole task
"""
import json, re, sys, os
from collections import defaultdict

FORMAT_HINT = re.compile(r'(file name|filename|\.docx|\.xlsx|\.pptx|\.pdf|word document|format shall|structured as|contains all|sections?:|deliverable is)', re.I)
RULEFLAG = re.compile(r'^(NO_|EXPANSION_|NON_|EXCLUDE|EXCLUDED|.*_BRANCH$|.*_HIGH$|.*_LOW$|ACCEPT$|REJECT$|PASS$|FAIL$)', re.I)

def load(path):
    with open(path) as f:
        return json.load(f)

def is_tautology_num(v):
    try:
        f=float(v)
    except: return False
    return f in (0.0,1.0,100.0)

def select_crux(aug):
    ev = aug.get('expected_values', {})
    gold = (aug.get('gold_deliverable_text','') or '') + '\n' + (aug.get('corrected_solution_logic','') or '')
    dag = aug.get('dag', {})
    depths = aug.get('depths', {})

    # GATE 1: scoreable
    if aug.get('scoreable', True) is False:
        return {'skip': True, 'reason': aug.get('not_scoreable_reason','scoreable=false'), 'crux': [], 'flags': []}
    # GATE 2: judgment_flagged anywhere in crux-relevant values
    for vid,meta in ev.items():
        if meta.get('source_of_verification')=='judgment_flagged':
            return {'skip': True, 'reason': f'{vid} judgment_flagged (authorial contradiction)', 'crux': [], 'flags':[]}

    dropped=set(aug.get('crux_dropped_no_expected',[]))
    crux=[]; flags=[]; rationale={}
    # identify the decision recommendation (final): prefer llm_judgment decision that is a
    # real recommendation token appearing in the memo, dedup cosmetic equivalents by value.
    decision_final=[]
    seen_decision_val={}
    for vid,meta in ev.items():
        kind=meta.get('kind'); val=meta.get('value'); sov=meta.get('source_of_verification')
        if vid in dropped or val is None:
            continue  # no frozen expected value -> not scoreable as crux
        sval=str(val)
        # ---- EXCLUSIONS (given inputs / format / tautology / process-rule flags) ----
        if sov=='source_file':
            continue  # given input / rule read from file -> not a computed answer
        # format / structure / filename checks (any kind) -> never a reported answer
        if sval.lower() in ('structure_only','schema_compliant','format_ok','well_formed',
                            'used_and_cited','pilot_excluded','present'):
            continue
        if FORMAT_HINT.search(sval):
            continue
        if kind=='numeric':
            if is_tautology_num(val):
                continue
            if sov in ('arithmetic','llm_judgment'):
                crux.append(vid); rationale[vid]=f'computed numeric {val} ({meta.get("unit","")})'.strip()
            continue
        # kind in {string, decision}: a computed (arithmetic/llm_judgment) value the golden
        # reports is an ANSWER -- keep, regardless of string-vs-decision kind. Only drop
        # genuine process rule-flags (a tight known set), and dedup cosmetic decision repeats.
        RULEFLAG_TOKENS={'NO_DISCOUNTING','NO_EXPANSION_LOW_BRANCH','EXPANSION_HIGH_BRANCH',
                         'EXCLUDED','EXCLUDE','BOTTLENECK_GOVERNS','NOISE_IGNORED_3_FINALISTS',
                         'EXCLUDE_EXCEL_EXCEPT_EXPORT_NOTE','L1-L4_ONLY','BENCHMARKED_ASSUMPTIONS',
                         'DECISION_ON_TARGET_ONLY','before_consultation','underst ated'}
        if sval in RULEFLAG_TOKENS:
            continue
        # go/no-go & final recommendation / computed-answer strings: keep, dedup by value
        if sval in seen_decision_val:
            continue  # cosmetic dup of already-kept decision/string
        seen_decision_val[sval]=vid
        if kind=='decision':
            decision_final.append(vid); rationale[vid]=f'final recommendation "{sval}"'
        else:  # kind == string
            # Keep ONLY if it reads as a reported answer/identity, not a method/description.
            # Description markers -> exclude (formula text, scenario counts, restatements).
            desc = bool(re.search(r'(sum|scenario|calcul|\+|\*|=|\d+\s*(sub-?)?scenario|per unit|breakdown)', sval, re.I))
            longish = len(sval.split())>6
            if desc or longish:
                continue  # describes method, not a reported answer
            crux.append(vid); rationale[vid]=f'computed string answer "{sval}"'
            flags.append((vid,f'string-answer "{sval}" — verify it is a reported conclusion, not a description'))
    # include one decision recommendation (they were deduped by value already)
    for vid in decision_final:
        crux.append(vid)
        # restatement leniency flag if a numeric comparison drives it
        flags.append((vid,'decision restatement — lenient (trapped model may also pass)'))

    # residual-input flag: arithmetic numeric whose value is a round product (capex-like)
    for vid in list(crux):
        meta=ev.get(vid,{})
        if meta.get('kind')=='numeric' and meta.get('source_of_verification')=='arithmetic':
            unit=(meta.get('unit') or '').lower()
            val=meta.get('value')
            # heuristic: 'capex' in nearby gold text and value is round -> possible input
            try:
                fv=float(val)
            except: fv=None
            if fv is not None and abs(fv-round(fv))<1e-9 and re.search(r'\bcapex\b', gold, re.I):
                # only flag, do not auto-drop (skill says flag for human)
                if any(f'{int(fv)}' in seg for seg in re.findall(r'capex[^.]{0,40}', gold, re.I)):
                    flags.append((vid,f'arithmetic {val} may be trivially-computed capex input — REVIEW'))

    crux=sorted(set(crux), key=lambda x:int(re.sub(r'\D','',x) or 0))
    return {'skip':False,'reason':'','crux':crux,'flags':flags,'rationale':rationale}

# ---------- GEAR scoring ----------
def transitive_parents(node, dag, crux_set, memo):
    """Return crux-ancestors reachable through (possibly dropped) non-crux nodes."""
    if node in memo: return memo[node]
    res=set()
    for p in dag.get(node, []):
        if p in crux_set:
            res.add(p)
        else:
            res |= transitive_parents(p, dag, crux_set, memo)
    memo[node]=res
    return res

def gear_score(crux, marks, dag, lam=0.2):
    """marks: {vid:0/1}. Returns (flat, gear). Equal weights."""
    crux=[c for c in crux if c in marks]  # only score verifiers that have a mark
    if not crux:
        return None, None, 0, 0
    cs=set(crux)
    memo={}
    subgraph={c: transitive_parents(c,dag,cs,memo) for c in crux}
    p={c: float(marks[c]) for c in crux}
    # topo by number of crux-ancestors (roots first)
    order=sorted(crux, key=lambda c: len(subgraph[c]))
    qhat={}
    for c in order:
        parents=subgraph[c]
        prod=1.0
        for j in parents:
            qj=qhat.get(j, p[j])
            prod*= (qj + (1-qj)*lam)
        qhat[c]=p[c]*prod
    flat=sum(p.values())/len(crux)
    gear=sum(qhat.values())/len(crux)
    npass=sum(1 for c in crux if marks[c]==1)
    return flat, gear, npass, len(crux)

if __name__=='__main__':
    print("module ok")