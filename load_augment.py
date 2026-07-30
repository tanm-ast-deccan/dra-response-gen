#!/usr/bin/env python3
"""
load_augment.py — for each graded task, find the CANONICAL augment source
(the one whose verifier texts match the delivery CSV's column L) and extract
its DAG, crux set, Shapley weights, and sov tags.

Priority: {task}_augment.json  →  {task}_augment.html  →  {task}_augment_*.{json,html}
A candidate is accepted ONLY if its verifier texts match column L (guards against
stale augments with drifted weights, e.g. the Zinc _augment_hunyuan leftover).
If none match, the task is flagged 'canonical augment MISSING' — never silently
falls back to a stale one.

Emits augment_index.json: {task: {source, dag, crux, weights, sov, verifier_texts}}
and prints a per-task status table.

Usage:
  python load_augment.py --csv <delivery.csv> --aug-dir <dir with augment files> --out augment_index.json
"""
import argparse, csv, re, json, glob, os, html as H

def vtexts(cell):
    """{Vid: normalized text} from a 'V# - text' / 'V#: text' blob."""
    d = {}
    for blk in re.split(r'(?=V\d+\s*[-:]\s)', cell or ""):
        m = re.match(r'^V(\d+)\s*[-:]\s*(.*)', blk.strip(), re.S)
        if not m:
            continue
        t = re.sub(r'\s+', ' ', m.group(2)).strip()
        t = re.sub(r'\s*-\s*[01]\s*,?\s*$', '', t).rstrip(',').strip()  # strip trailing score
        d['V' + m.group(1)] = t
    return d

def norm(t):
    return re.sub(r'[^a-z0-9]', '', (t or '').lower())[:40]

# ---- extract structured augment from JSON ----
def from_json(path):
    j = json.load(open(path))
    vt = j.get('augmented_verifiers_text', '')
    vtx = vtexts(vt) if isinstance(vt, str) else {f'V{i+1}': t for i, t in enumerate(vt)}
    return dict(source=path, dag=j.get('dag', {}), crux=j.get('crux_ids', []),
                weights=j.get('crux_shapley_weights', {}),
                sov={k: (v or {}).get('source_of_verification', '')
                     for k, v in j.get('expected_values', {}).items()},
                verifier_texts=vtx)

# ---- extract structured augment from HTML ----
def from_html(path):
    h = open(path).read()
    # DAG/weights/sov table rows
    dag, weights, sov, crux = {}, {}, {}, []
    rowpat = re.compile(
        r"<td class='mono'>(V\d+)</td><td>(.*?)</td><td class='mono'>(.*?)</td>"
        r"<td class='mono'>[\d.]+%</td><td class='mono'>([\d.]+)%.*?</td>"
        r"<td class='mono'>(.*?)</td>", re.S)
    for vid, cruxcell, dep, sw, s in rowpat.findall(h):
        deps = [d.strip() for d in dep.replace('root', '').split(',')
                if d.strip().startswith('V')]
        dag[vid] = deps
        weights[vid] = float(sw) / 100.0
        sov[vid] = s.strip()
        if 'CRUX' in cruxcell:
            crux.append(vid)
    # canonical verifier texts block
    m = re.search(r'Augmented verifiers \(canonical\)</h2>\s*<pre>(.*?)</pre>', h, re.S)
    vtx = vtexts(H.unescape(m.group(1))) if m else {}
    return dict(source=path, dag=dag, crux=crux, weights=weights, sov=sov,
                verifier_texts=vtx)

def texts_match(cand, csv_v):
    shared = set(cand) & set(csv_v)
    if not shared:
        return 0.0
    ok = sum(1 for k in shared if norm(cand[k]) == norm(csv_v[k]))
    # require both same ID set and high text agreement
    idset = (set(cand) == set(csv_v))
    return (ok / len(shared)) * (1.0 if idset else 0.9)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--aug-dir", default=".")
    ap.add_argument("--out", default="augment_index.json")
    ap.add_argument("--match-threshold", type=float, default=0.85)
    a = ap.parse_args()

    raw = list(csv.reader(open(a.csv, encoding="utf-8-sig")))
    hrow = next(i for i, r in enumerate(raw[:6]) if any(c.strip() == "task_id" for c in r))
    idx = {c.strip(): i for i, c in enumerate(raw[hrow])}
    ci_t, ci_L = idx["task_id"], idx["augmented_verifiers"]
    csv_verifiers = {}
    for r in raw[hrow + 1:]:
        if len(r) <= ci_L:
            continue
        t = r[ci_t].strip()
        if t.startswith("tsk_") and t not in csv_verifiers:
            csv_verifiers[t] = vtexts(r[ci_L])

    index, status = {}, []
    for t, csv_v in csv_verifiers.items():
        # candidate augment files, in priority order
        cands = ([os.path.join(a.aug_dir, f"{t}_augment.json")]
                 + [os.path.join(a.aug_dir, f"{t}_augment.html")]
                 + sorted(glob.glob(os.path.join(a.aug_dir, f"{t}_augment_*.json")))
                 + sorted(glob.glob(os.path.join(a.aug_dir, f"{t}_augment_*.html"))))
        best, best_score = None, 0.0
        for c in cands:
            if not os.path.exists(c):
                continue
            try:
                cand = from_json(c) if c.endswith(".json") else from_html(c)
            except Exception as e:
                status.append((t, c, f"parse-error: {e}")); continue
            sc = texts_match(cand.get("verifier_texts", {}), csv_v)
            if sc > best_score:
                best, best_score = cand, sc
        if best and best_score >= a.match_threshold:
            index[t] = best
            status.append((t, os.path.basename(best["source"]),
                           f"OK match={best_score:.2f} crux={len(best['crux'])}"))
        elif best:
            status.append((t, os.path.basename(best["source"]),
                           f"BELOW THRESHOLD match={best_score:.2f} — REVIEW"))
        else:
            status.append((t, "-", "NO AUGMENT FOUND — needs regeneration"))

    json.dump(index, open(a.out, "w"), indent=1)
    print(f"{'task':<17}{'source':<40}status")
    for t, s, st in sorted(status):
        print(f"{t:<17}{s:<40}{st}")
    ok = sum(1 for _, _, st in status if st.startswith("OK"))
    print(f"\n{ok}/{len(csv_verifiers)} tasks matched a canonical augment → {a.out}")
    miss = [t for t, _, st in status if not st.startswith("OK")]
    if miss:
        print("Tasks needing attention:", ", ".join(miss))

if __name__ == "__main__":
    main()