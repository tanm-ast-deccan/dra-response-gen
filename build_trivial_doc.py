#!/usr/bin/env python3
"""
build_trivia_doc.py — enriched verifier-triviality review worksheet.

Reads (all on server):
  --marks    consolidated_scores.csv     (verifier_id, verifier_text per task)
  --augment  augment_index.json          (crux membership + weights per task)
  --csv      delivery CSV                 (prompt col E for task summary)
  --aug-dir  dir of {task}_augment.json   (gold_deliverable_text for soln summary)

For each task: a 1-2 line TASK summary (from prompt opening) and SOLUTION summary
(from golden deliverable exec-summary / conclusion), then a table of every verifier
with: ID, CRUX (yes/weight), proposed trivial-flag+reason, full text, KEEP/DROP col.

Emits verifier_trivia_audit.docx. Requires: node + docx (for rendering step, separate).
This script writes an intermediate JSON that the node builder consumes.

Usage:
  python build_trivia_doc.py --marks consolidated_scores.csv --augment augment_index.json \
     --csv Augmented_tasks_SME_Delivery_-_sme_shortlist_with_links.csv --aug-dir ./output_2/augmented
  node make_trivia_doc2.js     # then renders the docx
"""
import argparse, csv, re, json, os
from collections import OrderedDict

TRIV=[
 (r'sum.*(is|=|equal|to)\s*1\b|probabilit.*(sum|is 1|equal 1)|sum of .*(shares|weights|probabil)','tautology / internal-consistency check'),
 (r'file ?name|file ?format|\.docx|\.xlsx|word document|output file|naming convention|shall be named','file/format requirement'),
 (r'rounded|decimal place|one decimal|two decimal|format compliance|contains table|table \d|number of rows|word limit|≤\s*\d+ ?word','presentation/format'),
 (r'no (use of )?discount|discounting (methodolog|framework|rate)|shall (not|be no)','rule-adherence (avoid-doing)'),
 (r'must be (used and )?cited|cite[sd]?|citation|reference (rate|url|source)|explicitly cites','sourcing/citation of a given input'),
 (r'design target (used )?=|target (used )?=|= \d+ (orders|units)|stated in .* as|given (in|as)','restates a given input value (not derived)'),
]
def classify(t):
    tl=t.lower()
    for pat,r in TRIV:
        if re.search(pat,tl): return 'TRIVIAL?',r
    return 'decisive',''

def first_sentences(text, n=2, maxlen=320):
    text=re.sub(r'\s+',' ',text or '').strip()
    parts=re.split(r'(?<=[.!?])\s+',text)
    out=' '.join(parts[:n])
    return (out[:maxlen]+'…') if len(out)>maxlen else out

def exec_summary(gold):
    """pull the exec-summary / recommendation sentence from golden deliverable text."""
    g=re.sub(r'\s+',' ',gold or '')
    # look for a Recommended/Recommendation/Summary marker
    m=re.search(r'(Recommended Strategy|Recommendation|Summary|Chosen Strategy)[:\s].{0,300}', g, re.I)
    return first_sentences(m.group(0),2) if m else first_sentences(g,2)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--marks',required=True); ap.add_argument('--augment',required=True)
    ap.add_argument('--csv',required=True); ap.add_argument('--aug-dir',default='.')
    ap.add_argument('--out-json',default='/tmp/trivia_audit2.json')
    a=ap.parse_args()

    aug=json.load(open(a.augment))
    # verifier texts per task
    byt=OrderedDict()
    for r in csv.DictReader(open(a.marks,encoding='utf-8-sig')):
        byt.setdefault(r['task_id'],OrderedDict())
        if r['verifier_id'] not in byt[r['task_id']]:
            byt[r['task_id']][r['verifier_id']]=r['verifier_text']
    # prompt per task from delivery csv
    raw=list(csv.reader(open(a.csv,encoding='utf-8-sig')))
    hrow=next(i for i,rr in enumerate(raw[:6]) if any(c.strip()=='task_id' for c in rr))
    idx={c.strip():i for i,c in enumerate(raw[hrow])}
    prompts={}
    for rr in raw[hrow+1:]:
        if len(rr)<=idx['prompt']: continue
        t=rr[idx['task_id']].strip()
        if t.startswith('tsk_') and t not in prompts: prompts[t]=rr[idx['prompt']]

    out=OrderedDict()
    for t,vs in byt.items():
        A=aug.get(t,{})
        crux=set(A.get('crux',[])); weights=A.get('weights',{})
        # solution summary from gold deliverable json
        gold=''
        p=os.path.join(a.aug_dir,f'{t}_augment.json')
        if os.path.exists(p):
            try: gold=json.load(open(p)).get('gold_deliverable_text','')
            except: pass
        task_sum=first_sentences(prompts.get(t,''),2)
        soln_sum=exec_summary(gold) if gold else '(golden deliverable not found)'
        vlist=[]
        for vid,txt in vs.items():
            flag,reason=classify(txt)
            is_crux=vid in crux
            w=weights.get(vid,0.0)
            vlist.append(dict(vid=vid,crux=is_crux,weight=round(100*w,1),
                              flag=flag,reason=reason,text=txt))
        out[t]=dict(task_summary=task_sum,solution_summary=soln_sum,verifiers=vlist)
    json.dump(out,open(a.out_json,'w'),indent=1)
    nt=sum(len(v['verifiers']) for v in out.values())
    nf=sum(1 for v in out.values() for x in v['verifiers'] if x['flag']=='TRIVIAL?')
    ncrux_flag=sum(1 for v in out.values() for x in v['verifiers'] if x['flag']=='TRIVIAL?' and x['crux'])
    print(f'{len(out)} tasks, {nt} verifiers, {nf} flagged ({ncrux_flag} of them crux) -> {a.out_json}')
    print('Now run:  node make_trivia_doc2.js')

if __name__=='__main__': main()