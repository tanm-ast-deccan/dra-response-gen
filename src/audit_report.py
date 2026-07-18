# src/audit_report.py
"""
Generate a self-contained HTML review report from an audit JSON (the AuditResult
schema produced by auditor.py).

Design constraints (deliberate, for this use):
  - SINGLE self-contained .html file: all CSS + JS inline, no external requests,
    works offline from a file:// path (SMEs double-click it).
  - Write-back via DOWNLOAD (Option A): the SME accepts/rejects each change and
    answers JUDGMENT_REQUIRED questions; "Save decisions" triggers a JSON download
    keyed by the audit's run identity (task_id + a content hash) so a stale
    decisions file can't be silently applied to a changed audit.
  - No localStorage/sessionStorage (not available; also irrelevant for download flow).
  - Quiet, scannable, functional — this is an internal QC tool, not a brand page.

Public API:
    render_report(audit: dict) -> str            # returns HTML
    write_report(audit: dict, out_path: str)     # writes the .html file
    render_batch(audits: list[dict]) -> str       # one page, collapsible per task
"""

from __future__ import annotations

import hashlib
import html
import json
from typing import List


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _esc(s) -> str:
    return html.escape("" if s is None else str(s))


def _audit_hash(audit: dict) -> str:
    """Stable short hash of the audit's change set + verdict, so a downloaded
    decisions file can be checked against the audit it was made for."""
    basis = json.dumps({
        "task_id": audit.get("task_id"),
        "verdict": audit.get("verdict"),
        "changes": audit.get("changes", []),
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:12]


_VERDICT_CLASS = {
    "SOUND": "v-sound", "SALVAGEABLE": "v-salvage", "BROKEN": "v-broken",
    "UNGRADEABLE": "v-ungrade", "NON_DETERMINISTIC": "v-nondet",
    "AUDIT_FAILED": "v-failed",
}

_CLAIM_CLASS = {
    "CONFIRMED": "c-ok", "ARITHMETIC_ERROR": "c-err",
    "INPUT_ERROR": "c-warn", "UNVERIFIABLE": "c-unv",
}


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _render_claims(audit: dict) -> str:
    cvs = audit.get("claim_verdicts", [])
    if not cvs:
        return "<p class='muted'>No arithmetic claims were emitted.</p>"
    rows = []
    for v in cvs:
        rec = v.get("recomputed")
        clm = v.get("claimed")
        rec_s = f"{rec:g}" if isinstance(rec, (int, float)) else "—"
        clm_s = f"{clm:g}" if isinstance(clm, (int, float)) else "—"
        trap = " <span class='trap-badge'>matches trap</span>" if v.get("matches_trap") else ""
        cls = _CLAIM_CLASS.get(v.get("status", ""), "c-unv")
        # provenance summary
        prov = v.get("input_provenance", [])
        checked = [p for p in prov if p.get("found_in_source") is not None]
        derived = [p for p in prov if p.get("derived")]
        prov_bits = []
        if checked:
            ok = sum(1 for p in checked if p["found_in_source"])
            prov_bits.append(f"{ok}/{len(checked)} inputs found in source")
        if derived:
            prov_bits.append(f"{len(derived)} derived")
        prov_s = _esc("; ".join(prov_bits)) if prov_bits else "—"
        rows.append(
            f"<tr class='{cls}'>"
            f"<td class='mono'>{_esc(v.get('id'))}</td>"
            f"<td>{_esc(v.get('label'))}</td>"
            f"<td class='status'>{_esc(v.get('status'))}{trap}</td>"
            f"<td class='mono'>{clm_s}</td>"
            f"<td class='mono'>{rec_s}</td>"
            f"<td class='small'>{prov_s}</td>"
            f"</tr>"
        )
    return (
        "<table class='claims'><thead><tr>"
        "<th>ID</th><th>Figure</th><th>Status</th><th>Claimed</th>"
        "<th>Recomputed</th><th>Provenance</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _render_changes(audit: dict) -> str:
    changes = audit.get("changes", [])
    if not changes:
        return "<p class='muted'>No changes proposed.</p>"
    cards = []
    for i, c in enumerate(changes):
        ctype = c.get("type", "MECHANICAL")
        type_cls = "t-mech" if ctype == "MECHANICAL" else "t-judge"
        old = _esc(c.get("old", ""))
        new = _esc(c.get("new", ""))
        diff = ""
        if old or new:
            diff = (
                f"<div class='diff'>"
                f"<div class='old'><span class='lbl'>old</span> {old or '<em>—</em>'}</div>"
                f"<div class='new'><span class='lbl'>new</span> {new or '<em>(none)</em>'}</div>"
                f"</div>"
            )
        question = ""
        if ctype == "JUDGMENT_REQUIRED" and c.get("sme_question"):
            question = f"<div class='sme-q'><strong>SME decision needed:</strong> {_esc(c['sme_question'])}</div>"
        controls = (
            f"<div class='controls' data-change='{i}'>"
            f"<button class='btn accept' onclick=\"decide({i},'accept')\">Accept</button>"
            f"<button class='btn reject' onclick=\"decide({i},'reject')\">Reject</button>"
            f"<span class='decision-state' id='ds{i}'></span>"
            f"</div>"
        )
        cards.append(
            f"<div class='change' id='change{i}'>"
            f"<div class='change-head'>"
            f"<span class='type-badge {type_cls}'>{_esc(ctype.replace('_',' '))}</span>"
            f"<span class='artifact'>{_esc(c.get('artifact'))}</span>"
            f"<span class='loc small'>{_esc(c.get('location'))}</span>"
            f"</div>"
            f"<div class='rationale'>{_esc(c.get('rationale'))}</div>"
            f"{diff}{question}{controls}"
            f"</div>"
        )
    return "".join(cards)


def _render_findings(audit: dict) -> str:
    out = []
    findings = audit.get("findings", [])
    if findings:
        items = "".join(
            f"<li><span class='cat'>{_esc(f.get('category'))}</span> "
            f"<span class='sev {('conf' if f.get('status')=='confirmed' else 'susp')}'>{_esc(f.get('status'))}</span> "
            f"{_esc(f.get('evidence'))}</li>"
            for f in findings
        )
        out.append(f"<ul class='findings'>{items}</ul>")
    # leakage / drift / missing
    for key, label in [("leakage_findings", "Leakage"),
                       ("temporal_drift_findings", "Temporal drift"),
                       ("missing_inputs", "Missing inputs")]:
        lst = audit.get(key, [])
        if lst:
            items = "".join(f"<li>{_esc(json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else x)}</li>" for x in lst)
            out.append(f"<h4>{label}</h4><ul class='findings sub'>{items}</ul>")
    return "".join(out) if out else "<p class='muted'>No findings.</p>"


def _render_qc(audit: dict) -> str:
    qc = audit.get("verifier_qc_findings", [])
    if not qc:
        return "<p class='muted'>No verifier-QC defects.</p>"
    items = "".join(
        f"<li><span class='qc-code'>{_esc(f.get('code'))}</span> "
        f"<span class='sev {('conf' if f.get('severity')=='BLOCK' else 'susp')}'>{_esc(f.get('severity'))}</span> "
        f"{_esc(f.get('message'))}</li>"
        for f in qc
    )
    return f"<ul class='findings'>{items}</ul>"


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def render_report(audit: dict) -> str:
    task_id = audit.get("task_id", "unknown")
    verdict = audit.get("verdict", "UNKNOWN")
    vcls = _VERDICT_CLASS.get(verdict, "v-failed")
    run_hash = _audit_hash(audit)
    asum = audit.get("arithmetic_summary", {})

    prov_supplied = audit.get("input_files_supplied")
    prov_checked = audit.get("provenance_checked")
    if not prov_supplied:
        prov_banner = ("<div class='banner warn'>Input files were NOT supplied — "
                       "the source-checking (provenance) layer did not run. "
                       "Declared inputs are unverified against source.</div>")
    elif not prov_checked:
        prov_banner = ("<div class='banner warn'>Input files were supplied but no "
                       "input was source-checked (all derived, or no numeric inputs). "
                       "Provenance not established.</div>")
    else:
        files = ", ".join(audit.get("input_files_names", []))
        prov_banner = (f"<div class='banner ok'>Provenance checked against input files: "
                       f"{_esc(files)}</div>")

    if audit.get("error"):
        err = (f"<div class='banner err'>Audit error: {_esc(audit['error'])}"
               + (" — response looks truncated." if audit.get("likely_truncated") else "")
               + "</div>")
    else:
        err = ""

    arith_line = ""
    if asum:
        arith_line = (
            f"<span>{asum.get('confirmed',0)} confirmed</span>"
            f"<span class='warn-t'>{asum.get('input_error',0)} input-error</span>"
            f"<span class='err-t'>{asum.get('arithmetic_error',0)} arithmetic-error</span>"
            f"<span class='muted'>{asum.get('unverifiable',0)} unverifiable</span>"
        )

    corrected = audit.get("corrected_solution_logic", "")
    corrected_block = ""
    if corrected:
        corrected_block = (
            f"<details class='corrected'><summary>Corrected solution logic</summary>"
            f"<pre>{_esc(corrected)}</pre></details>"
        )

    body = f"""
<div class="report" data-task="{_esc(task_id)}" data-hash="{run_hash}">
  <header>
    <div class="title-row">
      <h1>{_esc(task_id)}</h1>
      <span class="verdict {vcls}">{_esc(verdict)}</span>
      <span class="meta">{audit.get('calls_made',0)} call(s){' · call 2 skipped' if audit.get('skipped_call2') else ''} · {_esc(audit.get('model_used') or '—')}</span>
    </div>
    <p class="primary-reason">{_esc(audit.get('primary_reason'))}</p>
    {err}
    {prov_banner}
  </header>

  <section>
    <h2>Summary</h2>
    <p class="prose">{_esc(audit.get('prose_findings'))}</p>
    <div class="arith-summary">{arith_line}</div>
  </section>

  <section>
    <h2>Arithmetic claims <span class="count">{len(audit.get('claim_verdicts',[]))}</span></h2>
    {_render_claims(audit)}
  </section>

  <section>
    <h2>Findings</h2>
    {_render_findings(audit)}
  </section>

  <section>
    <h2>Verifier QC <span class="count">{len(audit.get('verifier_qc_findings',[]))}</span>
      <span class="vstatus">parse: {_esc(audit.get('verifier_parse_status'))} ({audit.get('verifier_count',0)} verifiers)</span>
    </h2>
    {_render_qc(audit)}
  </section>

  <section>
    <h2>Proposed changes <span class="count">{len(audit.get('changes',[]))}</span></h2>
    <p class="hint">Accept or reject each. Mechanical = a concrete edit; Judgment required = answer the question, no auto-edit.</p>
    {_render_changes(audit)}
    {corrected_block}
  </section>

  <div class="save-bar">
    <span id="unsaved" class="unsaved"></span>
    <button class="btn save" onclick="saveDecisions()">Save decisions</button>
  </div>
</div>
"""

    return _PAGE_TEMPLATE.format(
        title=_esc(task_id),
        body=body,
        task_id=json.dumps(task_id),
        run_hash=json.dumps(run_hash),
        n_changes=len(audit.get("changes", [])),
        css=_CSS,
    )


def write_report(audit: dict, out_path: str) -> str:
    html_text = render_report(audit)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_text)
    return out_path


# ---------------------------------------------------------------------------
# Static assets (CSS + page shell with the download-based write-back JS)
# ---------------------------------------------------------------------------

_CSS = """
:root{--bg:#fbfaf8;--fg:#1a1a1a;--muted:#777;--line:#e4e0d8;--card:#fff;
  --ok:#2e7d4f;--warn:#b06a00;--err:#b3261e;--accent:#175fff;--judge:#7a4cc0;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Inter,sans-serif;
  line-height:1.5;font-size:15px}
.report{max-width:920px;margin:0 auto;padding:28px 22px 90px}
header{border-bottom:1px solid var(--line);padding-bottom:16px;margin-bottom:8px}
.title-row{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
h1{font-size:20px;margin:0;font-family:ui-monospace,monospace}
h2{font-size:15px;text-transform:uppercase;letter-spacing:.06em;color:#555;
  margin:26px 0 10px;display:flex;align-items:center;gap:10px}
h4{margin:12px 0 4px;font-size:13px;color:#555}
.count{background:#eee;border-radius:10px;padding:1px 9px;font-size:12px;color:#555}
.vstatus{font-size:12px;color:var(--muted);text-transform:none;letter-spacing:0;font-weight:400}
.verdict{padding:4px 12px;border-radius:6px;font-weight:600;font-size:13px}
.v-sound{background:#e3f3e9;color:var(--ok)} .v-salvage{background:#fdf0dc;color:var(--warn)}
.v-broken{background:#fbe4e2;color:var(--err)} .v-ungrade{background:#eee;color:#555}
.v-nondet{background:#eef;color:#446} .v-failed{background:#fbe4e2;color:var(--err)}
.meta{color:var(--muted);font-size:12px;margin-left:auto}
.primary-reason{font-size:15px;margin:10px 0 0}
.prose{color:#333}
.banner{padding:9px 13px;border-radius:7px;margin-top:10px;font-size:13.5px}
.banner.ok{background:#e8f3ec;color:#235} .banner.warn{background:#fdf3e3;color:#653}
.banner.err{background:#fbe4e2;color:#822}
.arith-summary{display:flex;gap:14px;flex-wrap:wrap;font-size:13px;margin-top:8px}
.arith-summary .warn-t{color:var(--warn)} .arith-summary .err-t{color:var(--err)}
table.claims{width:100%;border-collapse:collapse;font-size:13px}
table.claims th{text-align:left;color:#888;font-weight:600;border-bottom:1px solid var(--line);padding:6px 8px}
table.claims td{padding:6px 8px;border-bottom:1px solid #f0ece4;vertical-align:top}
.mono{font-family:ui-monospace,monospace}
.small{font-size:12px;color:#666}
tr.c-ok .status{color:var(--ok)} tr.c-err .status{color:var(--err);font-weight:600}
tr.c-warn .status{color:var(--warn)} tr.c-unv .status{color:var(--muted)}
.trap-badge{background:#fbe4e2;color:var(--err);font-size:10px;padding:1px 6px;border-radius:8px;margin-left:5px}
ul.findings{margin:6px 0;padding-left:0;list-style:none}
ul.findings li{padding:6px 0;border-bottom:1px solid #f0ece4;font-size:13.5px}
ul.findings.sub li{padding:3px 0;border:none}
.cat,.qc-code{font-family:ui-monospace,monospace;background:#eef;color:#446;padding:1px 7px;border-radius:5px;font-size:12px;margin-right:6px}
.sev{font-size:11px;padding:1px 7px;border-radius:8px;margin-right:6px}
.sev.conf{background:#fbe4e2;color:var(--err)} .sev.susp{background:#fdf0dc;color:var(--warn)}
.change{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:13px 15px;margin-bottom:11px}
.change-head{display:flex;align-items:center;gap:9px;margin-bottom:6px;flex-wrap:wrap}
.type-badge{font-size:11px;padding:2px 9px;border-radius:6px;font-weight:600}
.t-mech{background:#e8eeff;color:var(--accent)} .t-judge{background:#f0e9fb;color:var(--judge)}
.artifact{font-family:ui-monospace,monospace;font-size:12.5px;color:#444}
.loc{margin-left:auto}
.rationale{font-size:13.5px;color:#333;margin-bottom:7px}
.diff{font-size:13px;margin:6px 0}
.diff .old{color:#822;background:#fcecea;padding:3px 8px;border-radius:5px;margin-bottom:3px}
.diff .new{color:#235;background:#e8f3ec;padding:3px 8px;border-radius:5px}
.diff .lbl{font-size:10px;text-transform:uppercase;opacity:.6;margin-right:6px}
.sme-q{background:#f7f3fd;border-left:3px solid var(--judge);padding:8px 11px;border-radius:0 6px 6px 0;font-size:13.5px;margin:7px 0}
.controls{display:flex;align-items:center;gap:8px;margin-top:9px}
.btn{border:1px solid var(--line);background:#fff;border-radius:6px;padding:5px 15px;font-size:13px;cursor:pointer}
.btn:hover{background:#f4f1ec}
.btn.accept:hover{background:#e8f3ec;border-color:var(--ok);color:var(--ok)}
.btn.reject:hover{background:#fcecea;border-color:var(--err);color:var(--err)}
.change.accepted{border-color:var(--ok);background:#f6fbf8}
.change.rejected{opacity:.55}
.decision-state{font-size:12px;font-weight:600}
.change.accepted .decision-state{color:var(--ok)}
.change.rejected .decision-state{color:var(--err)}
.corrected{margin-top:14px}
.corrected summary{cursor:pointer;font-size:13px;color:var(--accent)}
.corrected pre{white-space:pre-wrap;background:#fff;border:1px solid var(--line);border-radius:8px;padding:13px;font-size:13px;line-height:1.6}
.save-bar{position:fixed;bottom:0;left:0;right:0;background:#fff;border-top:1px solid var(--line);
  padding:11px 22px;display:flex;align-items:center;justify-content:flex-end;gap:14px}
.btn.save{background:var(--accent);color:#fff;border-color:var(--accent);font-weight:600;padding:7px 20px}
.unsaved{font-size:13px;color:var(--warn)}
.hint{font-size:12.5px;color:var(--muted);margin:0 0 11px}
.muted{color:var(--muted)}
"""

_PAGE_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Audit · {title}</title>
<style>{css}</style>
</head><body>
{body}
<script>
const TASK_ID = {task_id};
const RUN_HASH = {run_hash};
const N_CHANGES = {n_changes};
const decisions = {{}};
const questions = {{}};

function decide(i, choice) {{
  decisions[i] = choice;
  const card = document.getElementById('change'+i);
  const state = document.getElementById('ds'+i);
  card.classList.remove('accepted','rejected');
  card.classList.add(choice === 'accept' ? 'accepted' : 'rejected');
  state.textContent = choice === 'accept' ? '✓ accepted' : '✗ rejected';
  markUnsaved();
}}

function markUnsaved() {{
  const done = Object.keys(decisions).length;
  const el = document.getElementById('unsaved');
  if (done < N_CHANGES) {{
    el.textContent = done + ' of ' + N_CHANGES + ' decided — unsaved';
  }} else {{
    el.textContent = 'all ' + N_CHANGES + ' decided — remember to Save';
  }}
}}

function saveDecisions() {{
  const payload = {{
    task_id: TASK_ID,
    run_hash: RUN_HASH,         // ties decisions to THIS audit; reject if stale
    saved_at: new Date().toISOString(),
    decisions: decisions,
    questions: questions
  }};
  const blob = new Blob([JSON.stringify(payload, null, 2)], {{type:'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  const stamp = new Date().toISOString().slice(0,10);
  a.download = 'decisions_' + TASK_ID + '_' + stamp + '.json';
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  document.getElementById('unsaved').textContent = 'saved ✓';
}}

window.addEventListener('beforeunload', function(e) {{
  if (Object.keys(decisions).length > 0 &&
      document.getElementById('unsaved').textContent.indexOf('saved') === -1) {{
    e.preventDefault(); e.returnValue = '';
  }}
}});
</script>
</body></html>"""


def render_batch(audits: List[dict]) -> str:
    """One page with a collapsible section per task (triage view)."""
    # Minimal: reuse per-task render inside <details>. Kept simple on purpose.
    blocks = []
    for a in audits:
        inner = render_report(a)
        # extract just the .report div from the full page
        start = inner.find('<div class="report"')
        end = inner.rfind('</div>\n<script>')
        report_div = inner[start:end] if start != -1 and end != -1 else inner
        blocks.append(
            f"<details class='task-block'><summary>{_esc(a.get('task_id'))} — "
            f"{_esc(a.get('verdict'))}</summary>{report_div}</details>"
        )
    return ("<!doctype html><html><head><meta charset='utf-8'>"
            f"<style>{_CSS}</style></head><body><div class='report'>"
            "<h1>Audit batch</h1>" + "".join(blocks) + "</div></body></html>")
