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


_PROPS = ("atomic", "quantifiable", "self_contained", "falsifiable",
          "toleranced", "content_not_ordinal")
_PROP_LABEL = {"atomic": "Atomic", "quantifiable": "Quantifiable",
               "self_contained": "Self-contained", "falsifiable": "Falsifiable",
               "toleranced": "Toleranced", "content_not_ordinal": "By content"}


DECISION_JS = """
<script>
const TASK_ID  = "%(task_id)s";
const RUN_HASH = "%(run_hash)s";
const ITEMS    = %(items_json)s;   // [{id, kind, needs_answer}]
const decisions = {};
const answers   = {};

function decide(id, choice) {
  decisions[id] = choice;
  const card = document.getElementById('it_' + id);
  const st   = document.getElementById('st_' + id);
  card.classList.remove('accepted','rejected','other','undecided');
  card.classList.add(choice === 'accept' ? 'accepted'
                    : choice === 'reject' ? 'rejected' : 'other');
  // fill the clicked button and dim its siblings, so the click is unmistakable
  card.querySelectorAll('.btns button').forEach(function(b) {
    b.classList.remove('on-accept','on-reject','on-other','dim');
    if (b.dataset.choice === choice) { b.classList.add('on-' + choice); }
    else { b.classList.add('dim'); }
  });
  st.className = 'pill ' + (choice === 'accept' ? 'ok'
                          : choice === 'reject' ? 'bad' : 'warn');
  st.textContent = choice === 'accept' ? '\u2713 accepted'
                 : choice === 'reject' ? '\u2717 rejected'
                 : '\u25c6 something else';
  // Reject and Something Else REQUIRE a reason: a bare rejection tells the
  // pipeline to drop a change without recording why, which is unauditable.
  const box = document.getElementById('why_' + id);
  if (box) { box.style.display = (choice === 'accept') ? 'none' : 'block'; }
  tally();
}

function note(id, el) { answers[id] = el.value; tally(); }

function tally() {
  let done = 0, blocked = [];
  ITEMS.forEach(function(it) {
    const c = decisions[it.id];
    const a = (answers[it.id] || '').trim();
    if (!c) return;
    if (c !== 'accept' && !a) { blocked.push(it.id + ' needs a reason'); return; }
    if (it.needs_answer && !a) { blocked.push(it.id + ' needs an answer'); return; }
    done++;
  });
  const el = document.getElementById('tally');
  el.textContent = done + ' of ' + ITEMS.length + ' complete'
                 + (blocked.length ? '  \u2014  ' + blocked.slice(0,3).join('; ') : '')
                 + (done === ITEMS.length ? '  \u2014 ready to save' : '  \u2014 unsaved');
  el.className = 'pill ' + (done === ITEMS.length ? 'ok' : 'warn');
}

function saveDecisions() {
  const missing = ITEMS.filter(function(it) {
    const c = decisions[it.id]; const a = (answers[it.id] || '').trim();
    return !c || (c !== 'accept' && !a) || (it.needs_answer && !a);
  }).map(function(it) { return it.id; });
  if (missing.length) {
    var NL = String.fromCharCode(10);
    if (!confirm('Incomplete: ' + missing.join(', ') + NL + NL
                 + 'Save anyway? apply_decisions.py will refuse to apply '
                 + 'an incomplete file.')) return;
  }
  const payload = {task_id: TASK_ID, run_hash: RUN_HASH,
                   saved_at: new Date().toISOString(),
                   decisions: decisions, answers: answers, incomplete: missing};
  const blob = new Blob([JSON.stringify(payload, null, 2)], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'decisions_' + TASK_ID + '_'
             + new Date().toISOString().slice(0,10) + '.json';
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  document.getElementById('tally').textContent = 'saved \u2713';
  document.getElementById('tally').className = 'pill ok';
}

window.addEventListener('beforeunload', function(e) {
  if (Object.keys(decisions).length &&
      document.getElementById('tally').textContent.indexOf('saved') === -1) {
    e.preventDefault(); e.returnValue = '';
  }
});
window.addEventListener('load', tally);
</script>
"""

DECISION_CSS = """
.card{border:1px solid #e7e2d8;border-radius:8px;padding:12px;margin:10px 0;background:#fff}
.card.accepted{border-left:6px solid #1a7a4a;background:#eef8f2}
.card.rejected{border-left:6px solid #b4413c;background:#fbeceb}
.card.other{border-left:6px solid #9a7400;background:#fdf6e3}
.card.undecided{border-left:6px solid #d6d0c4}
.card h4{margin:0 0 6px 0;font-size:15px}
.btns{margin-top:8px}
.btns button{font-size:13px;padding:5px 14px;margin-right:6px;border-radius:6px;
  border:1px solid #d6d0c4;background:#f6f3ec;cursor:pointer;font-weight:600;
  color:#4a463d}
.btns button:hover{background:#ece7dc}
/* the clicked button fills in. Previously only the CARD changed, with a tint of
   #f6fbf8 on a #fbfaf7 page — invisible — so a click looked like it did nothing. */
.btns button.on-accept{background:#1a7a4a;border-color:#166b41;color:#fff}
.btns button.on-reject{background:#b4413c;border-color:#9c3833;color:#fff}
.btns button.on-other{background:#9a7400;border-color:#836300;color:#fff}
.btns button.dim{opacity:.45}
.why{display:none;margin-top:8px}
.why textarea{width:100%;min-height:52px;font-family:inherit;font-size:13px;
  padding:6px;border:1px solid #d6d0c4;border-radius:6px}
.ask{margin-top:8px}
.ask textarea{width:100%;min-height:64px;font-family:inherit;font-size:13px;
  padding:6px;border:1px solid #9a7400;border-radius:6px;background:#fffdf5}
.sticky{position:sticky;top:0;background:#fbfaf7;padding:8px 0;z-index:5;
  border-bottom:1px solid #e7e2d8}
"""


def _decision_items(res: dict):
    """Everything a human must rule on, as uniform cards.

    Four kinds, because they apply differently downstream:
      change   — a MECHANICAL edit already applied; reject to revert it
      question — an unresolved JUDGMENT_REQUIRED item; needs a written answer, and
                 until it has one the task is not scoreable, because the next run
                 may resolve it the other way and change the golden's answer
      rewrite  — a verifier reworded in place by the property audit
      split    — a verifier divided into children
      gap      — a proposed new verifier for an unwatched step
    """
    items = []
    for i, c in enumerate(res.get("changes_applied") or []):
        items.append({"id": f"chg{i}", "kind": "change", "needs_answer": False,
                      "title": f"{c.get('artifact')} — {c.get('location')}",
                      "body": (f"<b>was:</b> {_esc(c.get('old'))}<br>"
                               f"<b>now:</b> {_esc(c.get('new'))}<br>"
                               f"<i>{_esc(c.get('rationale'))}</i>"),
                      "meta": "mechanical, already applied"})
    for i, c in enumerate(res.get("judgment_changes_pending_sme") or []):
        items.append({"id": f"q{i}", "kind": "question", "needs_answer": True,
                      "title": f"{c.get('artifact')} — {c.get('location')}",
                      "body": (f"<b>{_esc(c.get('sme_question'))}</b><br>"
                               f"<i>{_esc(c.get('rationale'))}</i>"),
                      "meta": "NOT applied — artifact keeps its original wording"})
    va = res.get("verifier_audit") or {}
    for i, v in enumerate(va.get("verifiers") or []):
        rw = (v.get("rewrite") or "").strip()
        if not rw:
            continue
        fails = [k for k, x in (v.get("properties") or {}).items()
                 if (x or {}).get("verdict") == "FAIL"]
        items.append({"id": f"rw{i}", "kind": "rewrite", "needs_answer": False,
                      "title": f"{v.get('id')} reworded ({', '.join(fails)})",
                      "body": f"<b>now:</b> {_esc(rw)}",
                      "meta": "applied in place, id unchanged"})
    # the children's text and their targets, because "V4 split into V4a, V4b, V4c"
    # with only the parent's old text shown gives no way to judge the split
    vtext, ev = {}, res.get("expected_values") or {}
    for line in (res.get("augmented_verifiers_text") or "").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            vtext[k.strip()] = v.strip()
    for i, x in enumerate(res.get("verifier_splits_applied") or []):
        kids = x.get("children") or []
        rows = []
        for k in kids:
            e = ev.get(k)
            tgt = (f"<span class='pill ok'>target {e.get('value')}"
                   f"{' ±' + str(e.get('tol')) if e.get('tol') else ''}</span>"
                   if e else "<span class='pill bad'>NO TARGET — unscoreable</span>")
            rows.append(f"<li><span class='mono'>{_esc(k)}</span> {tgt}<br>"
                        f"{_esc(vtext.get(k, '(text not found)'))}</li>")
        tl = x.get("targetless_children") or []
        items.append({"id": f"sp{i}", "kind": "split", "needs_answer": False,
                      "title": (f"{x.get('parent')} split into "
                                f"{', '.join(kids)}"),
                      "body": (f"<b>was:</b> {_esc(x.get('parent_text'))}"
                               f"<br><b>becomes:</b><ul>{''.join(rows)}</ul>"),
                      "meta": (f"target to {x.get('target_went_to')}"
                               + (f" — {len(tl)} child(ren) with no target: "
                                  f"{', '.join(tl)}" if tl else ""))})
    for i, g in enumerate(va.get("coverage_gaps") or []):
        items.append({"id": f"gap{i}", "kind": "gap", "needs_answer": False,
                      "title": f"step {g.get('step')} has no verifier",
                      "body": (f"{_esc(g.get('why_it_matters'))}<br>"
                               f"<b>proposed:</b> {_esc(g.get('proposed_verifier'))}"),
                      "meta": "NOT added — accept to add it"})
    return items


def _decision_section(res: dict) -> str:
    import json as _json
    items = _decision_items(res)
    if not items:
        return "<p><i>Nothing to decide.</i></p>"
    cards = []
    for it in items:
        ask = ""
        if it["needs_answer"]:
            ask = (f"<div class='ask'><b>Your answer (required):</b><br>"
                   f"<textarea id='why_{it['id']}' "
                   f"oninput=\"note('{it['id']}', this)\" "
                   f"placeholder='State the decision. This is what gets applied.'>"
                   f"</textarea></div>")
        else:
            ask = (f"<div class='why' id='why_{it['id']}'>"
                   f"<b>Reason (required for reject / something else):</b><br>"
                   f"<textarea oninput=\"note('{it['id']}', this)\"></textarea></div>")
        cards.append(
            f"<div class='card undecided' id='it_{it['id']}'>"
            f"<h4>{_esc(it['title'])} "
            f"<span class='pill warn'>{it['kind']}</span> "
            f"<span id='st_{it['id']}' class='pill warn'>undecided</span></h4>"
            f"<div>{it['body']}</div>"
            f"<div class='mono' style='color:#7a7568;font-size:12px'>{_esc(it['meta'])}</div>"
            f"<div class='btns'>"
            f"<button data-choice='accept' "
            f"onclick=\"decide('{it['id']}','accept')\">Accept</button>"
            f"<button data-choice='reject' "
            f"onclick=\"decide('{it['id']}','reject')\">Reject</button>"
            f"<button data-choice='other' "
            f"onclick=\"decide('{it['id']}','other')\">Something else</button>"
            f"</div>{ask}</div>")
    slim = _json.dumps([{"id": i["id"], "kind": i["kind"],
                         "needs_answer": i["needs_answer"]} for i in items])
    js = DECISION_JS % {"task_id": res.get("task_id", ""),
                        "run_hash": res.get("run_hash", ""),
                        "items_json": slim}
    return (f"<div class='sticky'><span id='tally' class='pill warn'></span> "
            f"&nbsp;<button onclick='saveDecisions()'>Save decisions</button>"
            f"&nbsp;<span class='mono' style='font-size:12px'>run "
            f"{_esc(res.get('run_hash'))}</span></div>"
            + "".join(cards) + js)


def _trajectory(res: dict) -> str:
    """The golden trajectory: the ordered derivation, with what each step feeds.

    Built since call 2 gained corrected_claims with from_claim links and
    judgment_steps with consumes, but never rendered — so the artifact the whole
    design turns on was invisible in the report.
    """
    steps = res.get("corrected_claim_verdicts") or []
    judg = res.get("judgment_steps") or []
    graph = res.get("step_graph") or {}
    health = res.get("step_graph_health") or {}
    v2s = res.get("verifier_to_step") or {}
    if not steps and not judg:
        return "<p><i>No trajectory: call 2 emitted no corrected claims.</i></p>"

    watched = {}
    for vid, sid in v2s.items():
        watched.setdefault(sid, []).append(vid)
    consumed = {p for ps in graph.values() for p in ps}
    feeds = {}
    for child, parents in graph.items():
        for par in parents:
            feeds.setdefault(par, []).append(child)

    rows = []
    for c in steps:
        sid = c.get("id")
        parents = [p.get("from_claim") for p in (c.get("input_provenance") or [])
                   if p.get("from_claim")]
        st = c.get("status", "")
        pill = ("ok" if st == "CONFIRMED"
                else "bad" if st in ("ARITHMETIC_ERROR", "UNVERIFIABLE")
                else "warn")
        rows.append(
            f"<tr><td class='mono'>{_esc(sid)}</td>"
            f"<td>{_esc(c.get('label'))}</td>"
            f"<td class='mono'>{_esc(c.get('operation'))}</td>"
            f"<td class='mono'>{_esc(c.get('recomputed'))}</td>"
            f"<td class='mono'>{_esc(', '.join(dict.fromkeys(parents)) or '—')}</td>"
            f"<td class='mono'>{_esc(', '.join(feeds.get(sid, [])) or 'TERMINAL')}</td>"
            f"<td class='mono'>{_esc(', '.join(watched.get(sid, [])) or '—')}</td>"
            f"<td><span class='pill {pill}'>{_esc(st)}</span></td></tr>")
    for j in judg:
        jid = j.get("id")
        rows.append(
            f"<tr><td class='mono'>{_esc(jid)}</td>"
            f"<td><b>{_esc(j.get('question'))}</b><br>"
            f"<i>{_esc(j.get('ruling'))}</i></td>"
            f"<td class='mono'>judgement</td><td class='mono'>—</td>"
            f"<td class='mono'>{_esc(', '.join(j.get('consumes') or []) or '—')}</td>"
            f"<td class='mono'>{_esc(', '.join(feeds.get(jid, [])) or 'TERMINAL')}</td>"
            f"<td class='mono'>{_esc(', '.join(watched.get(jid, [])) or '—')}</td>"
            f"<td><span class='pill warn'>judgement</span></td></tr>")

    hdr = ("<tr><th>step</th><th>what it establishes</th><th>operation</th>"
           "<th>value</th><th>from</th><th>feeds</th><th>watched by</th>"
           "<th>status</th></tr>")
    note = (f"<p>{health.get('n_nodes', 0)} steps, {health.get('n_edges', 0)} "
            f"dependencies, {health.get('connected', 0)} connected to a terminal"
            + (f", <span class='pill bad'>{len(health.get('cycles') or [])} "
               f"cycle(s)</span>" if health.get("cycles") else "")
            + (f". Terminals: <span class='mono'>"
               f"{_esc(', '.join(health.get('terminals') or []))}</span>"
               if health.get("terminals") else "")
            + ". A step with no verifier in <i>watched by</i> is work a response "
              "can get wrong with nothing objecting.</p>")
    return note + f"<table>{hdr}{''.join(rows)}</table>"


def _property_table(res: dict) -> str:
    """Per-verifier property verdicts, plus duplication and coverage gaps.

    The property audit ran from the first real task onward but nothing rendered
    it, so the atomicity failures that caused a split were invisible — a reader
    saw the split children with no stated reason.
    """
    va = res.get("verifier_audit") or {}
    if va.get("error"):
        return f"<p class='pill bad'>property audit failed: {_esc(va['error'])}</p>"
    vs = va.get("verifiers") or []
    if not vs:
        return "<p><i>Property audit did not run.</i></p>"

    head = ("<tr><th>ID</th>"
            + "".join(f"<th>{_PROP_LABEL[p]}</th>" for p in _PROPS)
            + "<th>Tests step</th><th>Finding</th></tr>")
    body = []
    for v in vs:
        cells = []
        why = []
        for p in _PROPS:
            d = (v.get("properties") or {}).get(p) or {}
            verdict = d.get("verdict", "")
            if verdict == "FAIL":
                cells.append("<td><span class='pill bad'>FAIL</span></td>")
                why.append(f"<b>{_PROP_LABEL[p]}</b>: {_esc(d.get('why',''))}")
            elif verdict == "NOT_APPLICABLE":
                cells.append("<td class='mono'>n/a</td>")
            elif verdict == "PASS":
                cells.append("<td class='mono'>ok</td>")
            else:
                cells.append("<td class='mono'>?</td>")
        step = v.get("tests_step") or "—"
        note = "<br>".join(why)
        if v.get("route_to_rubric"):
            note += (f"<br><span class='pill warn'>route to rubric: "
                     f"{_esc(v.get('rubric_dimension'))}</span>")
        body.append(f"<tr><td class='mono'>{_esc(v.get('id'))}</td>"
                    + "".join(cells)
                    + f"<td class='mono'>{_esc(step)}</td><td>{note}</td></tr>")

    extra = []
    for c in va.get("duplicate_clusters") or []:
        extra.append(f"<li><b>{_esc(', '.join(c.get('verifier_ids') or []))}</b> "
                     f"both assert {_esc(c.get('quantity'))} "
                     f"(values agree: {c.get('values_agree')}) — "
                     f"{_esc(c.get('recommended_action'))}</li>")
    for g in va.get("coverage_gaps") or []:
        extra.append(f"<li>step <b>{_esc(g.get('step'))}</b> has no verifier: "
                     f"{_esc(g.get('why_it_matters'))}<br>"
                     f"<i>proposed:</i> {_esc(g.get('proposed_verifier'))}</li>")
    for t in va.get("trap_passes_band") or []:
        extra.append(f"<li><span class='pill bad'>not falsifiable</span> "
                     f"<b>{_esc(t.get('verifier'))}</b>: {_esc(t.get('detail'))}</li>")
    sp = res.get("verifier_splits_applied") or []
    for x in sp:
        tl = x.get("targetless_children") or []
        extra.append(
            f"<li>split <b>{_esc(x.get('parent'))}</b> → "
            f"{_esc(', '.join(x.get('children') or []))}; target to "
            f"{_esc(x.get('target_went_to'))}"
            + (f" — <span class='pill bad'>no target: "
               f"{_esc(', '.join(tl))}</span>" if tl else "") + "</li>")

    return (f"<table>{head}{''.join(body)}</table>"
            + (f"<ul>{''.join(extra)}</ul>" if extra else ""))


def _corrected_blocks(res: dict) -> str:
    """Render every artifact the auditor may correct, not just the solution logic.

    All four were threaded onto the result but only the solution logic was ever
    rendered, so a corrected sanity check or prompt was invisible in the report a
    reviewer reads. An artifact identical to the input is labelled unchanged
    rather than shown as if it had been rewritten.
    """
    orig = res.get("_originals") or {}
    out = []
    for key, label in (("corrected_solution_logic", "Solution logic"),
                       ("corrected_sanity_check", "Sanity check"),
                       ("corrected_prompt", "Prompt"),
                       ("corrected_verifiers", "Verifiers (in-place edits)")):
        txt = res.get(key) or ""
        was = orig.get(key.replace("corrected_", ""), None)
        if not txt.strip():
            note = "<span class='pill ok'>no in-place edits</span>"
            body = ""
        elif was is not None and txt.strip() == str(was).strip():
            note = "<span class='pill ok'>unchanged</span>"
            body = f"<pre>{_esc(txt)}</pre>"
        else:
            note = "<span class='pill warn'>corrected</span>"
            body = f"<pre>{_esc(txt)}</pre>"
        out.append(f"<h3>{label} &nbsp; {note}</h3>{body}")
    return "\n".join(out)


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


def _css_with(extra: str) -> str:
    """Insert extra rules INSIDE the shared style block.

    _CSS carries its own </style>, so concatenating after it dropped the decision
    rules into the head as raw text — present in the file, styling nothing, which
    is why every button stayed white however it was clicked.
    """
    assert "</style>" in _CSS
    return _CSS.replace("</style>", extra + "\n</style>")


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

    _CSS_WITH_DECISIONS = _css_with(DECISION_CSS)
    doc = f"""<!doctype html><html><head><meta charset=utf-8>
<title>{tid} · augmentation</title>{_CSS_WITH_DECISIONS}</head><body>
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
{("<p class='pill warn'>"+str(len(res.get('judgment_changes_pending_sme',[])))+" judgment change(s) NOT applied — the artifact keeps its original wording and a question is posed for you below. Mechanical changes above are applied.</p>") if res.get('judgment_changes_pending_sme') else ""}
{("<h2>Questions for the SME</h2><ol>"+"".join("<li><b>"+_esc(c.get('artifact'))+"</b> — "+_esc(c.get('location'))+"<br>"+_esc(c.get('sme_question'))+"</li>" for c in res.get('judgment_changes_pending_sme',[]))+"</ol>") if res.get('judgment_changes_pending_sme') else ""}

<h2>Golden trajectory</h2>
{_trajectory(res)}

<h2>Decisions required from you</h2>
{_decision_section(res)}

<h2>Verifier properties</h2>
{_property_table(res)}

<h2>Corrected artifacts</h2>
{_corrected_blocks(res)}

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