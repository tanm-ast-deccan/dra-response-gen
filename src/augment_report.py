# src/augment_report.py
"""
Per-task HTML emitters for the augmenter:
  write_augment_report(res, path)   -> {task_id}_augment.html  (corrections, DAG,
                                        crux set + Shapley weights, verifiers)
  write_golden_report(res, path)    -> {task_id}_golden.html    (golden deliverable)

Both are self-contained, dependency-free HTML for SME eyeballing.
"""

from __future__ import annotations
import html
from typing import Dict


def _esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


_CSS = """
<style>
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1000px;margin:24px auto;
padding:0 20px;color:#1d1b16;line-height:1.55;background:#fbfaf7}
h1{font-size:26px} h2{font-size:19px;margin-top:28px;border-bottom:1px solid #e7e2d8;padding-bottom:4px}
code,.mono{font-family:ui-monospace,Menlo,monospace;font-size:13px}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:14px}
th,td{border:1px solid #e7e2d8;padding:6px 9px;text-align:left;vertical-align:top}
th{background:#f2eee6;font-size:12px;text-transform:uppercase;letter-spacing:.03em}
.tag{display:inline-block;background:#eceaf6;border-radius:5px;padding:1px 7px;margin:1px;font-size:12px;font-family:monospace}
.crux{background:#e7f3ec;border-left:3px solid #2e7d5b;padding:2px 6px}
.pill{border-radius:999px;padding:2px 10px;font-size:12px;font-weight:600}
.ok{background:#e4f3ea;color:#1a7a4a}.warn{background:#fbf3da;color:#9a7400}.bad{background:#f6e4e2;color:#b4413c}
pre{background:#f6f3ec;border:1px solid #e7e2d8;border-radius:8px;padding:12px;white-space:pre-wrap;font-size:13px}
.bar{height:7px;background:#175FFF;border-radius:4px;display:inline-block;vertical-align:middle}
</style>
"""


def write_augment_report(res: dict, out_path: str) -> str:
    tid = _esc(res.get("task_id"))
    verdict = _esc(res.get("audit_verdict"))
    vcls = "ok" if verdict in ("SOUND", "SALVAGEABLE") else "bad"

    crux = set(res.get("crux_ids", []))
    shap = res.get("crux_shapley_weights", {})
    dag = res.get("dag", {})
    base = res.get("base_weights", {})
    expected = res.get("expected_values", {})

    # verifier table
    rows = ""
    for vid in dag.keys():
        is_crux = vid in crux
        w = shap.get(vid, 0.0) * 100 if is_crux else 0.0
        deps = ", ".join(dag.get(vid, [])) or "root"
        sov = (expected.get(vid, {}) or {}).get("source_of_verification", "") if is_crux else ""
        rows += (f"<tr><td class='mono'>{_esc(vid)}</td>"
                 f"<td>{'<span class=crux>CRUX</span>' if is_crux else ''}</td>"
                 f"<td class='mono'>{deps}</td>"
                 f"<td class='mono'>{base.get(vid,0)*100:.1f}%</td>"
                 f"<td class='mono'>{w:.1f}%"
                 + (f" <span class='bar' style='width:{max(2,w):.0f}px'></span>" if is_crux else "")
                 + "</td>"
                 f"<td class='mono'>{_esc(sov)}</td></tr>")

    changes = res.get("changes_applied", []) + res.get("judgment_changes_pending_sme", [])
    chg_rows = ""
    for c in changes:
        pend = "warn" if c.get("type") != "MECHANICAL" else "ok"
        chg_rows += (f"<tr><td>{_esc(c.get('artifact'))}</td><td>{_esc(c.get('location'))}</td>"
                     f"<td><span class='pill {pend}'>{_esc(c.get('type'))}</span></td>"
                     f"<td>{_esc(c.get('old'))[:120]}</td><td>{_esc(c.get('new'))[:120]}</td>"
                     f"<td>{_esc(c.get('rationale'))[:160]}</td></tr>")

    trap = " ".join(f"<span class='tag'>{_esc(a)}</span>" for a in res.get("crux_anchors_trap", []))
    exp = " ".join(f"<span class='tag'>{_esc(a)}</span>" for a in res.get("crux_anchors_expert", []))

    doc = f"""<!doctype html><html><head><meta charset=utf-8>
<title>{tid} · augmentation</title>{_CSS}</head><body>
<h1>{tid} — Augmentation & Corrections</h1>
<p>Audit verdict: <span class='pill {vcls}'>{verdict}</span>
&nbsp; Crux size: <b>{len(crux)}</b> verifiers &nbsp; Model: <span class=mono>{_esc(res.get('model_used'))}</span></p>
{"<p class='pill bad'>ERROR: "+_esc(res.get('error'))+"</p>" if res.get('error') else ""}
{"<p class='pill warn'>&#9888; DEGRADED GOLDEN — these input files could not be read and were NOT used: "+", ".join("<span class=tag>"+_esc(s)+"</span>" for s in res.get('skipped_inputs',[]))+"</p>" if res.get('skipped_inputs') else ""}

<h2>Crux anchors (from Sanity Check)</h2>
<p><b>Lazy-AI / trap side:</b> {trap or '<i>none</i>'}<br>
<b>Expert-path side:</b> {exp or '<i>none</i>'}</p>

<h2>Verifiers · DAG · crux-only Shapley weights</h2>
<table><tr><th>ID</th><th>Crux</th><th>Depends on</th><th>Base wt</th><th>Crux Shapley wt</th><th>Source of verification</th></tr>
{rows}</table>
{"<p class='pill warn'>Dropped from crux (reachable but no frozen expected value, so unscoreable): "+", ".join("<span class=tag>"+_esc(d)+"</span>" for d in res.get('crux_dropped_no_expected',[]))+"</p>" if res.get('crux_dropped_no_expected') else ""}

<h2>Corrections applied</h2>
{"<table><tr><th>Artifact</th><th>Where</th><th>Type</th><th>Old</th><th>New</th><th>Why</th></tr>"+chg_rows+"</table>" if chg_rows else "<p><i>No changes.</i></p>"}
<p class='pill warn'>{len(res.get('judgment_changes_pending_sme',[]))} judgment change(s) applied now, flagged for later SME review.</p>

<h2>Corrected solution logic</h2>
<pre>{_esc(res.get('corrected_solution_logic'))}</pre>

<h2>Augmented verifiers (canonical)</h2>
<pre>{_esc(res.get('augmented_verifiers_text'))}</pre>

{"<h2>Notes</h2><pre>"+_esc(res.get('notes'))+"</pre>" if res.get('notes') else ""}
</body></html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)
    return out_path


def write_golden_report(res: dict, out_path: str) -> str:
    tid = _esc(res.get("task_id"))
    fmt = _esc(res.get("gold_deliverable_format"))
    secs = res.get("gold_deliverable_sections", [])
    body = ""
    for s in secs:
        body += f"<h2>{_esc(s.get('title'))}</h2><pre>{_esc(s.get('content'))}</pre>"
    doc = f"""<!doctype html><html><head><meta charset=utf-8>
<title>{tid} · golden deliverable</title>{_CSS}</head><body>
<h1>{tid} — Golden Deliverable</h1>
<p>Format: <span class='pill ok'>{fmt}</span> &nbsp; {len(secs)} section(s)</p>
{body if body else "<p><i>No deliverable generated.</i></p>"}
</body></html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)
    return out_path