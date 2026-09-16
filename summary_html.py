#!/usr/bin/env python3
"""
summary_html.py — render a stripped-down summary HTML from an adjudicated (or
augment) package JSON, showing ONLY:

    1. Golden Trajectory          (corrected_claim_verdicts + judgment_steps)
    2. Decisions required from you (judgment_changes_pending_sme)
    3. Corrected Solution Logic   (corrected_solution_logic)
    4. Corrected Prompt           (corrected_prompt)
    5. Augmented verifiers        (augmented_verifiers_text)

Usage:
    python summary_html.py <adjudicated.json> [-o out.html]

If -o is omitted, writes <input_stem>_summary.html next to the input.
Read-only; touches nothing in the pipeline.
"""
import argparse
import html
import json
import os
import re


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def _pre(text: str) -> str:
    """Whitespace-preserving block for prose artifacts."""
    return f"<pre class='art'>{esc(text)}</pre>" if (text or "").strip() \
        else "<p class='muted'>(empty)</p>"


def _trajectory(d: dict) -> str:
    claims = d.get("corrected_claim_verdicts", []) or []
    judg = d.get("judgment_steps", []) or []
    if not claims and not judg:
        return "<p class='muted'>(no trajectory)</p>"
    rows = []
    for c in claims:
        prov = c.get("input_provenance") or []
        deps = ", ".join(
            esc(p.get("from_claim") or p.get("name") or "")
            for p in prov if isinstance(p, dict)) or "—"
        st = esc(c.get("status") or "")
        st_cls = ("ok" if st == "CONFIRMED"
                  else ("bad" if st and st != "CONFIRMED" else ""))
        rows.append(
            f"<tr><td class='mono'>{esc(c.get('id'))}</td>"
            f"<td>{esc(c.get('label'))}</td>"
            f"<td class='mono'>{esc(c.get('recomputed'))}</td>"
            f"<td class='mono'>{esc(c.get('operation'))}</td>"
            f"<td class='mono'>{deps}</td>"
            f"<td class='{st_cls}'>{st}</td></tr>")
    for j in judg:
        cons = ", ".join(esc(x) for x in (j.get("consumes") or [])) or "—"
        rows.append(
            f"<tr class='judg'><td class='mono'>{esc(j.get('id'))}</td>"
            f"<td>{esc(j.get('question'))}</td>"
            f"<td class='mono'>{esc(j.get('ruling'))}</td>"
            f"<td class='mono'>judgment</td>"
            f"<td class='mono'>{cons}</td><td></td></tr>")
    return (
        "<table><thead><tr><th>ID</th><th>Label / Question</th>"
        "<th>Value / Ruling</th><th>Operation</th><th>Inputs</th>"
        "<th>Status</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>")


def _decisions(d: dict) -> str:
    items = d.get("judgment_changes_pending_sme", []) or []
    if not items:
        return "<p class='muted'>(no decisions pending)</p>"
    out = []
    for i, it in enumerate(items):
        out.append(
            "<div class='card'>"
            f"<div class='chip'>q{i} &middot; {esc(it.get('artifact'))}"
            f" &middot; {esc(it.get('type'))}</div>"
            f"<p class='q'>{esc(it.get('sme_question'))}</p>"
            + (f"<p class='loc'><b>Location:</b> {esc(it.get('location'))}</p>"
               if it.get('location') else "")
            + (f"<p><b>Rationale:</b> {esc(it.get('rationale'))}</p>"
               if it.get('rationale') else "")
            + (f"<p><b>Old:</b> <span class='mono'>{esc(it.get('old'))}</span></p>"
               if it.get('old') else "")
            + "</div>")
    return "".join(out)


def _verifiers(d: dict) -> str:
    txt = d.get("augmented_verifiers_text", "") or ""
    lines = [ln for ln in txt.splitlines() if ln.strip()]
    if not lines:
        return "<p class='muted'>(no verifiers)</p>"
    crux = set(d.get("crux_ids", []) or [])
    out = []
    for ln in lines:
        m = re.match(r"\s*(V\S+?)\s*(?:\[[^\]]*\])?\s*[:\-]\s*(.*)", ln)
        vid = m.group(1) if m else ""
        is_crux = vid in crux
        tag = " <span class='cruxtag'>crux</span>" if is_crux else ""
        out.append(f"<div class='vrow'><span class='mono vid'>{esc(vid)}</span>"
                   f"{tag} {esc(m.group(2) if m else ln)}</div>")
    return "".join(out)


CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1000px;
margin:24px auto;padding:0 16px;color:#1a1a1a;line-height:1.5}
h1{font-size:20px;border-bottom:2px solid #175FFF;padding-bottom:6px}
h2{font-size:16px;margin-top:28px;background:#f4f6fb;padding:6px 10px;
border-left:3px solid #175FFF;border-radius:4px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border:1px solid #e2e5ea;padding:5px 8px;text-align:left;vertical-align:top}
th{background:#f4f6fb}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
.ok{color:#1a7f4b;font-weight:600}.bad{color:#c0392b;font-weight:600}
.judg td{background:#fbfaf4}
.muted{color:#999;font-style:italic}
.art{background:#f8f9fb;border:1px solid #e2e5ea;border-radius:6px;padding:12px;
white-space:pre-wrap;font-size:12.5px;font-family:ui-monospace,Menlo,monospace}
.card{border:1px solid #e2e5ea;border-left:3px solid #d39e00;border-radius:6px;
padding:10px 12px;margin:8px 0;background:#fffdf7}
.chip{font-size:11px;color:#8a6d00;font-weight:600;margin-bottom:4px}
.q{font-weight:600;margin:4px 0}.loc{font-size:12px;color:#555}
.vrow{padding:4px 0;border-bottom:1px solid #eef0f3;font-size:13px}
.vid{color:#175FFF;font-weight:600}
.cruxtag{background:#e7f3ec;color:#2e7d5b;font-size:10px;padding:1px 5px;
border-radius:4px;font-weight:600}
"""


def build(d: dict) -> str:
    tid = esc(d.get("task_id") or "task")
    sc = d.get("scoreable")
    verdict = esc(d.get("audit_verdict") or "")
    badge = ("<span class='ok'>scoreable</span>" if sc
             else f"<span class='bad'>not scoreable</span>")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>{tid} — summary</title><style>{CSS}</style></head><body>
<h1>{tid} — summary &nbsp; {badge} &nbsp; <span class='mono'>{verdict}</span></h1>
<h2>1. Golden Trajectory</h2>{_trajectory(d)}
<h2>2. Decisions required from you</h2>{_decisions(d)}
<h2>3. Corrected Solution Logic</h2>{_pre(d.get('corrected_solution_logic'))}
<h2>4. Corrected Prompt</h2>{_pre(d.get('corrected_prompt'))}
<h2>5. Augmented verifiers (canonical)</h2>{_verifiers(d)}
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()
    d = json.load(open(a.json_path, encoding="utf-8"))
    out = a.out or (os.path.splitext(a.json_path)[0] + "_summary.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(build(d))
    print("wrote", out)


if __name__ == "__main__":
    main()