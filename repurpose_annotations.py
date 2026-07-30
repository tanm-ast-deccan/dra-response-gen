#!/usr/bin/env python3
"""
repurpose_annotations.py — from existing SME marks (no re-grading), derive:
  (1) APEX-style ANSWER-ACCURACY (flat mean over answer-bearing verifiers)
  (2) full-rubric mean and DAG-dependency-gated score
  (3) a deterministic FAILURE-MODE label per run
  (4) reliability flags (runs whose marks look internally inconsistent)

Answer verifiers = those whose expected_value.kind is numeric/decision (value-bearing),
i.e. kind in {number, decision} OR source_of_verification == 'arithmetic'.
Process verifiers = everything else (llm_judgment, source_file string checks, etc).

    python repurpose_annotations.py --csv <graded.csv> --aug-dir output/augmented
"""
import argparse, csv, json, os, re

TRAIL = re.compile(r'-\s*([01])\s*$')

def parse_marks(cell):
    out={}
    for line in (cell or "").splitlines():
        line=line.strip().rstrip(",")
        m=re.match(r'V(\d+)\b',line)
        if not m: continue
        t=TRAIL.search(line)
        if t: out["V"+m.group(1)]=int(t.group(1))
    if out: return out
    for vid,sc in re.findall(r'V(\d+)\s*-\s*([01])\b',cell or ""):
        out["V"+vid]=int(sc)
    return out

def ancestors(dag):
    anc={n:set() for n in dag}
    def rec(n):
        if anc[n]: return anc[n]
        s=set()
        for p in dag.get(n,[]):
            if p in dag: s.add(p); s|=rec(p)
        anc[n]=s; return s
    for n in dag: rec(n)
    return anc

def is_answer(ev):
    k=(ev or {}).get("kind","")
    sov=(ev or {}).get("source_of_verification","")
    return k in ("number","decision") or sov=="arithmetic"

def failure_mode(marks, crux, ev, anc, answer_ids):
    passed=lambda v: marks.get(v)==1
    ans_fail=[v for v in answer_ids if v in marks and not passed(v)]
    proc_ids=[v for v in crux if v not in answer_ids]
    proc_fail=[v for v in proc_ids if v in marks and not passed(v)]
    if not ans_fail and not proc_fail: return "CLEAN (all crux passed)"
    if ans_fail and not proc_fail:     return f"EXECUTION ERROR: wrong answer value(s) {ans_fail}, process ok"
    if ans_fail and proc_fail:         return f"STRUCTURAL: wrong answer {ans_fail} + broken step(s) {proc_fail}"
    if proc_fail and not ans_fail:     return f"METHOD FLAW: answer ok but step(s) {proc_fail} failed"
    return "MIXED"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv",required=True); ap.add_argument("--aug-dir",default="output/augmented")
    ap.add_argument("--score-col",default="Augmented_verifiers with scores")
    ap.add_argument("--answer-map",default="",
                    help="path to JSON {task_id:[V-ids]} listing the FINAL-answer verifiers "
                         "per task. If unset, auto-detects kind in {numeric,decision} but that "
                         "OVER-INCLUDES trivial arithmetic checks (e.g. probabilities-sum-to-1). "
                         "Supply this for a defensible answer-accuracy number.")
    a=ap.parse_args()
    answer_map=json.load(open(a.answer_map)) if a.answer_map and os.path.exists(a.answer_map) else {}
    raw=list(csv.reader(open(a.csv,encoding="utf-8-sig")))
    h=next((i for i,r in enumerate(raw[:10]) if "task_id" in [c.strip() for c in r]),0)
    hdr=[c.strip() for c in raw[h]]; idx={c:i for i,c in enumerate(hdr)}
    tcol,mcol=idx["task_id"],idx["anon_model"]
    scol=idx.get(a.score_col,len(hdr)-1)

    cache={}
    def aug(t):
        if t not in cache:
            p=os.path.join(a.aug_dir,f"{t}_augment.json")
            cache[t]=json.load(open(p)) if os.path.exists(p) else {}
        return cache[t]

    rows=[]
    for r in raw[h+1:]:
        if len(r)<=max(tcol,mcol,scol): continue
        t,model=r[tcol].strip(),r[mcol].strip()
        if not t.startswith("tsk_"): continue
        marks=parse_marks(r[scol])
        if not marks: continue
        A=aug(t); crux=A.get("crux_ids",[]); ev=A.get("expected_values",{})
        w=A.get("crux_shapley_weights",{}); dag=A.get("dag",{}); anc=ancestors(dag)
        answer_ids=answer_map.get(t) or [v for v in crux if is_answer(ev.get(v,{}))]
        # (1) answer accuracy
        ans_scored=[v for v in answer_ids if v in marks]
        acc=(sum(marks[v] for v in ans_scored)/len(ans_scored)) if ans_scored else None
        # (2) full mean + shapley + dag-gated
        scored=[v for v in crux if v in marks]
        full=(sum(marks[v] for v in scored)/len(scored)) if scored else 0
        shap=sum(w.get(v,0) for v in crux if marks.get(v)==1)
        def earns(v):
            if marks.get(v)!=1: return False
            return all(marks.get(p,1)==1 for p in anc.get(v,()) if p in marks)
        tot=sum(w.values()) or 1
        dagg=sum(w.get(v,0) for v in crux if earns(v))/tot
        # (3) failure mode
        fm=failure_mode(marks,crux,ev,anc,answer_ids)
        # (4) reliability flag: answer verifiers all pass but full<1 and SME... just flag odd combos
        flag=""
        if acc is not None and acc==1.0 and full<0.6: flag="ODD: answers pass but many process fail"
        rows.append(dict(task=t,model=model,acc=acc,full=full,shap=shap,dagg=dagg,
                         n_ans=len(ans_scored),fm=fm,flag=flag))

    print(f"{'task':<17}{'model':<9}{'ANS-ACC':>8}{'full':>7}{'shap':>7}{'dag':>7}  failure_mode")
    ta=[]; 
    for r in sorted(rows,key=lambda x:(x['task'],x['model'])):
        acc="n/a" if r['acc'] is None else f"{100*r['acc']:.0f}%"
        print(f"{r['task']:<17}{r['model']:<9}{acc:>8}{100*r['full']:>6.0f}%{100*r['shap']:>6.0f}%{100*r['dagg']:>6.0f}%  {r['fm']}")
        if r['flag']: print(f"{'':>48}!! {r['flag']}")

    # aggregate + Tencent gate
    print("\n=== AGGREGATE ===")
    for m in ("Model_A","Model_B"):
        sub=[r for r in rows if r['model']==m and r['acc'] is not None]
        if not sub: continue
        acc=sum(r['acc'] for r in sub)/len(sub)
        shap=sum(r['shap'] for r in sub)/len(sub)
        print(f"{m}: mean ANSWER-ACC {100*acc:.1f}%  |  mean Shapley {100*shap:.1f}%  (n={len(sub)})")
    # Tencent difficulty: per-task answer-acc; ModelB(doubao)<40%, ModelA(hunyuan)<20%
    print("\n=== TENCENT DIFFICULTY GATE (per-task answer-accuracy) ===")
    bytask={}
    for r in rows:
        if r['acc'] is None: continue
        bytask.setdefault(r['task'],{})[r['model']]=r['acc']
    qual=0
    for t,mm in sorted(bytask.items()):
        b=mm.get('Model_B'); h_=mm.get('Model_A')
        tag=""
        if b is not None and h_ is not None:
            ok = (b<0.40 and h_<0.20)
            if ok: qual+=1; tag="  <-- QUALIFIES (hard enough)"
            print(f"  {t}: Doubao(B)={100*b:.0f}%  Hunyuan(A)={100*h_:.0f}%{tag}")
    print(f"\nTasks meeting Tencent difficulty bar (answer-acc): {qual}/{len(bytask)}")

if __name__=="__main__": main()