#!/usr/bin/env python3
"""
sme_pool.py — task inventory and selection for an SME batch. Folds in the old
ib_select_and_stage.py selection funnel, generalized: no hardcoded task lists,
model names, Drive IDs, or credentials paths.

Reads the per-task augment JSON dumps, reports the real task inventory (clean
tsk_<10digits> IDs), filters to a scoreable/gradeable pool, SELECTS a shortlist
by one of two strategies, and tags each shortlisted task with an audit-coverage
flag derived from the auditor's own leakage/provenance signals.

Selection strategies (--strategy):
  rank       (default) verdict-priority then crux size; top --n. Simple.
  crux-split the ib "Option C" funnel: require SALVAGEABLE + (optionally) both/all
             providers completed with >=1 real deliverable + crux>0, then split at
             the median crux into HIGH (by crux size) and LOW (by min turns),
             taking --n-high + --n-low. Needs --results for provider/turn data.

Leakage / audit-coverage surfacing: instead of a format-based input gate, this
reads the auditor's own signals from each task's augment JSON when present —
whether input files were supplied to the auditor (input_files_supplied), whether
the provenance layer ran (provenance_checked), and any leakage_findings. Tasks
audited BLIND (no input files reached the auditor) are flagged, because their
input-file leakage check could not have fired. This replaces the old
--check-inputs .pptx/.json format gate: format is not the risk, leakage is, and
leakage is a content property the auditor checks — but only when it actually
saw the files.

Usage:
  # simple ranked shortlist
  python sme_pool.py --aug-dir output/augmented --csv prompt_data.csv --n 30

  # ib-style high/low crux split (needs the results trace for provider/turns)
  python sme_pool.py --strategy crux-split --aug-dir output/augmented \
      --csv prompt_data.csv --results results_trace.json --n-high 5 --n-low 5
"""
import argparse, csv, glob, json, os, re

CLEAN = re.compile(r"^tsk_\d{10}$")
# Verdicts use the auditor's own vocabulary (see auditor_templates.py):
#   SOUND | SALVAGEABLE | BROKEN | UNGRADEABLE | NON_DETERMINISTIC
# "CONFIRMED" was an older name for the clean-pass verdict and is kept as an
# alias so pre-rename augment data still classifies correctly.
GRADEABLE_VERDICTS = {"SOUND", "CONFIRMED", "SALVAGEABLE"}
VERDICT_ORDER = {"SOUND": 0, "CONFIRMED": 0, "SALVAGEABLE": 1,
                 "BROKEN": 2, "UNGRADEABLE": 3, "NON_DETERMINISTIC": 4}
DELIV = re.compile(r"\.(docx|xlsx|pdf|pptx)$", re.I)
SCRATCH = re.compile(r"(ocr_output|_test\.|/test\.|intermediate|tmp|scratch|_response\.docx$)", re.I)


# ---------------------------------------------------------------------------
# source-CSV helpers (domain + drive link), paths passed in
# ---------------------------------------------------------------------------

def _resolve_col(fieldnames, *candidates):
    """Case-insensitive resolve of the first matching column name."""
    low = {c.lower().strip(): c for c in (fieldnames or [])}
    for cand in candidates:
        hit = low.get(cand.lower())
        if hit:
            return hit
    return None

def load_source_fields(csv_path):
    """task_id -> {domain, drive_link} from the source CSV (best-effort).

    Self-contained: uses only csv.DictReader with case-insensitive header
    resolution, so it has no dependency on run_augment/src.auditor and works
    in isolation. Domain resolves across the several column names seen across
    domains ("Domain", "Sub Domain", ...); drive link across its variants.
    """
    if not csv_path or not os.path.exists(csv_path):
        return {}
    try:
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fn = reader.fieldnames or []
            tid_col = _resolve_col(fn, "task_id", "task id", "taskid")
            dom_col = _resolve_col(fn, "domain", "sub domain", "sub-domain", "subdomain")
            link_col = _resolve_col(fn, "drive link", "drive_link", "google drive",
                                    "drive url", "drive_url")
            if not tid_col:
                print("  (source-CSV join skipped: no task_id column found)")
                return {}
            out = {}
            for row in reader:
                tid = (row.get(tid_col) or "").strip()
                if not tid:
                    continue
                out[tid] = {
                    "domain": (row.get(dom_col) or "").strip() if dom_col else "",
                    "drive_link": (row.get(link_col) or "").strip() if link_col else "",
                }
        return out
    except Exception as e:
        print(f"  (source-CSV join skipped: {e})")
        return {}


# ---------------------------------------------------------------------------
# inventory
# ---------------------------------------------------------------------------

def load_inventory(aug_dir, src_fields):
    clean, malformed, unreadable = [], [], []
    for p in sorted(glob.glob(os.path.join(aug_dir, "*_augment.json"))):
        try:
            with open(p) as fh:
                rd = json.load(fh)
        except Exception as e:
            unreadable.append((os.path.basename(p), str(e)))
            continue
        tid = (rd.get("task_id") or "").strip()
        # Audit-coverage signals from the auditor (if the augment JSON carries
        # them). These may be nested under an "audit"/"audit_result" block or be
        # top-level, depending on how the augment record was assembled — read
        # defensively and record UNKNOWN when absent rather than assuming clean.
        audit = rd.get("audit") or rd.get("audit_result") or rd
        has_audit_fields = any(k in audit for k in
                               ("input_files_supplied", "provenance_checked", "leakage_findings"))
        if has_audit_fields:
            files_supplied = bool(audit.get("input_files_supplied", False))
            prov_checked = bool(audit.get("provenance_checked", False))
            leaks = audit.get("leakage_findings") or []
            n_leaks = len(leaks) if isinstance(leaks, list) else 0
            # file-scoped leaks are the ones the format gate used to gesture at
            n_file_leaks = sum(1 for f in leaks
                               if isinstance(f, dict)
                               and str(f.get("location", "")).startswith("file:"))
        else:
            files_supplied = prov_checked = None   # unknown — auditor fields not present
            n_leaks = n_file_leaks = None

        rec = {
            "task_id": tid,
            "domain": rd.get("domain", "") or src_fields.get(tid, {}).get("domain", ""),
            "verdict": rd.get("audit_verdict", ""),
            "crux": len(rd.get("crux_ids", [])),
            "scoreable": bool(rd.get("scoreable", True)),
            "reason": rd.get("not_scoreable_reason", ""),
            # audit coverage / leakage
            "files_supplied": files_supplied,     # True / False / None(unknown)
            "provenance_checked": prov_checked,
            "leaks": n_leaks,                      # total leakage findings (or None)
            "file_leaks": n_file_leaks,            # input-file-scoped leaks (or None)
        }
        (clean if CLEAN.match(tid) else malformed).append(rec)
    return clean, malformed, unreadable


def audit_flag(rec) -> str:
    """One-word coverage flag for a task, from the auditor's own signals.
      LEAK        — auditor found answer leakage (any location)
      BLIND       — auditor ran WITHOUT input files → input-file leakage check
                    could not fire; treat leakage status as unverified
      UNKNOWN     — augment JSON carries no auditor coverage fields
      OK          — files supplied, provenance checked, no leaks found
      (empty)     — files supplied, no leaks, provenance not confirmed
    """
    if rec.get("leaks"):
        loc = "file+prompt" if rec.get("file_leaks") else "prompt"
        return f"LEAK({rec['leaks']};{loc})"
    fs = rec.get("files_supplied")
    if fs is None:
        return "UNKNOWN"
    if fs is False:
        return "BLIND"          # audited without input files
    if rec.get("provenance_checked"):
        return "OK"
    return ""


# ---------------------------------------------------------------------------
# selection strategies
# ---------------------------------------------------------------------------

def select_rank(pool, n):
    ranked = sorted(pool, key=lambda r: (VERDICT_ORDER.get(r["verdict"], 4), -r["crux"]))
    return ranked[:n], {"strategy": "rank"}


def real_deliverables(files):
    return [f for f in files if DELIV.search(f) and not SCRATCH.search(f)]


def select_crux_split(pool, results_path, n_high, n_low, require_all_providers=True):
    """The ib Option-C funnel, generalized to any provider set discovered from the
    results trace (no hardcoded doubao/hunyuan)."""
    if not results_path:
        raise SystemExit("--strategy crux-split needs --results (the run trace).")
    res = json.load(open(results_path))
    records = res.get("results", res if isinstance(res, list) else [])
    # provider run data per task, provider discovered from the trace
    runs = {}
    turns_seen = False   # did the trace actually carry any turn counts?
    for x in records:
        t = x.get("task_id") or x.get("task")
        prov = x.get("provider") or x.get("model") or x.get("model_name")
        if not t or not prov:
            continue
        allf = x.get("output_files") or []
        if "turns" in x:
            turns_seen = True
        runs.setdefault(t, {})[prov] = dict(
            completed=x.get("completed", True),
            real=real_deliverables(allf),
            turns=x.get("turns", 0))
    if not turns_seen:
        print("  !  crux-split WARNING: the results trace carries no 'turns' field; "
              "the LOW tier's min-turns ranking is meaningless (all zeros). "
              "LOW-tier order is effectively arbitrary — verify the trace schema.")

    cand = []
    for r in pool:
        t = r["task_id"]
        if r["verdict"] != "SALVAGEABLE":
            continue
        rr = runs.get(t, {})
        if not rr:
            continue
        provs = list(rr.keys())
        ok = [m for m in provs if rr[m]["completed"] and len(rr[m]["real"]) >= 1]
        if require_all_providers and len(ok) < len(provs):
            continue
        if not ok:
            continue
        if r["crux"] == 0:
            continue
        cand.append(dict(task=t, crux=r["crux"],
                         mint=min(rr[m]["turns"] for m in ok)))
    if not cand:
        raise SystemExit("crux-split: no candidates passed the funnel.")
    cvals = sorted(c["crux"] for c in cand)
    med = cvals[len(cvals) // 2]
    hi = sorted([c for c in cand if c["crux"] >= med],
                key=lambda x: (x["crux"], x["mint"]), reverse=True)[:n_high]
    lo = sorted([c for c in cand if c["crux"] < med],
                key=lambda x: x["mint"], reverse=True)[:n_low]
    sel_tasks = {c["task"] for c in hi + lo}
    shortlist = [r for r in pool if r["task_id"] in sel_tasks]
    tier = {c["task"]: "HIGH" for c in hi}; tier.update({c["task"]: "LOW" for c in lo})
    for r in shortlist:
        r["tier"] = tier.get(r["task_id"], "")
    return shortlist, {"strategy": "crux-split", "median_crux": med,
                       "n_high": len(hi), "n_low": len(lo)}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="SME task inventory + selection + audit-coverage flags")
    ap.add_argument("--aug-dir", default="output/augmented")
    ap.add_argument("--csv", default=None, help="source CSV for domain + drive-link join")
    ap.add_argument("--strategy", choices=["rank", "crux-split"], default="rank")
    ap.add_argument("--n", type=int, default=30, help="shortlist size (rank strategy)")
    ap.add_argument("--n-high", type=int, default=5, help="high-crux count (crux-split)")
    ap.add_argument("--n-low", type=int, default=5, help="low-crux count (crux-split)")
    ap.add_argument("--results", default=None, help="run-trace JSON (crux-split needs it)")
    ap.add_argument("--quality", action="store_true",
                    help="restrict pool to CONFIRMED/SALVAGEABLE (rank strategy)")
    ap.add_argument("--out", default="sme_shortlist.csv")
    args = ap.parse_args()

    src_fields = load_source_fields(args.csv)
    clean, malformed, unreadable = load_inventory(args.aug_dir, src_fields)

    scoreable = [r for r in clean if r["scoreable"]]
    gradeable = [r for r in scoreable if r["verdict"] in GRADEABLE_VERDICTS]
    print(f"JSON files            : {len(clean)+len(malformed)+len(unreadable)}")
    print(f"  clean task_ids      : {len(clean)}")
    print(f"  MALFORMED (ignored) : {len(malformed)}")
    print(f"  UNREADABLE (ignored): {len(unreadable)}")
    print(f"  scoreable (clean)   : {len(scoreable)}")
    print(f"  gradeable           : {len(gradeable)}")

    # Full verdict distribution across ALL clean tasks — makes visible what is
    # being excluded and why (a task can be scoreable but not gradeable, e.g.
    # UNGRADEABLE), so "were any SOUND tasks dropped?" is answerable at a glance.
    vdist = {}
    for r in clean:
        v = r["verdict"] or "(missing)"
        vdist[v] = vdist.get(v, 0) + 1
    if vdist:
        print("  verdict distribution (all clean):")
        for v, n in sorted(vdist.items(), key=lambda kv: -kv[1]):
            mark = " [gradeable]" if v in GRADEABLE_VERDICTS else ""
            print(f"      {n:3d}  {v}{mark}")
    if malformed:
        print("\nMalformed IDs to investigate:")
        for r in malformed:
            print(f"  {r['task_id']}")
    if unreadable:
        print("\nUnreadable augment files to investigate:")
        for name, err in unreadable:
            print(f"  {name}: {err}")

    # pool
    pool = gradeable if (args.quality or args.strategy == "crux-split") else scoreable

    # select
    if args.strategy == "crux-split":
        shortlist, info = select_crux_split(pool, args.results, args.n_high, args.n_low)
    else:
        shortlist, info = select_rank(pool, args.n)

    # audit-coverage flag per shortlisted task (from the auditor's own signals)
    for r in shortlist:
        r["audit_flag"] = audit_flag(r)

    # write — audit_flag column always included (it's the leakage/coverage signal)
    has_tier = any(r.get("tier") for r in shortlist)
    cols = ["task_id", "domain", "verdict", "crux"] + (["tier"] if has_tier else []) + ["audit_flag"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in shortlist:
            w.writerow({k: r.get(k, "") for k in cols})

    print(f"\nselection: {info}")
    print(f"wrote {len(shortlist)} task(s) -> {args.out}")
    by_v = {}
    for r in shortlist:
        by_v[r["verdict"]] = by_v.get(r["verdict"], 0) + 1
    print("  verdict mix:", by_v)

    # audit-coverage summary — surface leakage and blind-audit prominently
    flags = {}
    for r in shortlist:
        f = r["audit_flag"] or "(clean)"
        flags[f] = flags.get(f, 0) + 1
    print("  audit coverage:", flags)
    leaks = [r["task_id"] for r in shortlist if str(r["audit_flag"]).startswith("LEAK")]
    blind = [r["task_id"] for r in shortlist if r["audit_flag"] == "BLIND"]
    unknown = [r["task_id"] for r in shortlist if r["audit_flag"] == "UNKNOWN"]
    if leaks:
        print(f"  !! LEAKAGE flagged ({len(leaks)}) — do NOT deliver until resolved:")
        for t in leaks:
            print(f"       {t}")
    if blind:
        print(f"  !  BLIND audit ({len(blind)}) — auditor ran WITHOUT input files; "
              f"input-file leakage unverified:")
        for t in blind:
            print(f"       {t}")
    if unknown:
        print(f"  ?  UNKNOWN coverage ({len(unknown)}) — augment JSON carries no auditor "
              f"leakage/provenance fields; cannot confirm inputs were checked.")


if __name__ == "__main__":
    main()