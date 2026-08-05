#!/usr/bin/env python3
"""Measure what input-file truncation actually costs on a real task. No LLM calls.

    python probe_input_files.py --csv ./SME_data/Non_IB_Prompt_Eval_Approved_tasks.csv \
        --task tsk_1079445371

Answers three questions before anything is built:

  1. How large is each input file's extracted text, and how much does the current
     MAX_CHARS_PER_FILE cap hide?
  2. Of the numbers the solution logic actually declares, how many are findable in
     the FULL text but NOT in the truncated text? Each of those is a provenance
     check that is wrong today — a value the author really did read from a file,
     reported as unsourced because the auditor never saw the page it was on.
  3. Would an excerpt window carry them at a fraction of the size?

The number-finding here uses the SAME matcher the auditor uses
(arithmetic_verifier._number_appears), so a hit means the real check would pass.
"""
import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.arithmetic_verifier import _number_appears, parse_number
from src.auditor import build_header_map, get_field, read_task_csv
from src.gdrive_raw_fetcher import MAX_CHARS_PER_FILE

EXCERPT_RADIUS = 500


def truncate_like_fetcher(raw: str) -> str:
    if len(raw) <= MAX_CHARS_PER_FILE:
        return raw
    head = int(MAX_CHARS_PER_FILE * 0.7)
    tail = MAX_CHARS_PER_FILE - head
    return (raw[:head] + "\n\n[... truncated ...]\n\n" + raw[-tail:])


def declared_numbers(text: str, limit: int = 400):
    """Numbers the solution logic states, as candidate declared inputs.

    A proxy for the auditor's extracted claim inputs, needing no model call. Drops
    values under 100 and bare years, which match too easily to be evidence.
    """
    out, seen = [], set()
    for tok in re.findall(r"\d[\d,]*\.?\d*", text):
        v = parse_number(tok)
        if v is None or abs(v) < 100:
            continue
        if 1900 <= v <= 2100 and float(v).is_integer():
            continue
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
        if len(out) >= limit:
            break
    return out


def excerpt_for(value: float, full: str, radius: int = EXCERPT_RADIUS) -> str:
    """The window an excerpt-based prompt would carry for this value."""
    for pat in (f"{value:,.2f}", f"{value:,.1f}", f"{value:,.0f}",
                f"{value:.2f}", f"{value:.1f}",
                (str(int(value)) if float(value).is_integer() else f"{value}")):
        i = full.find(pat)
        if i >= 0:
            return full[max(0, i - radius): i + len(pat) + radius]
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--radius", type=int, default=EXCERPT_RADIUS)
    args = ap.parse_args()

    headers, rows = read_task_csv(args.csv)
    hmap = build_header_map(headers)
    match = [r for r in rows if get_field(r, hmap, "task_id") == args.task]
    if not match:
        raise SystemExit(f"{args.task} not in {args.csv}")
    row = match[0]

    dl = get_field(row, hmap, "drive_link").strip()
    if not dl:
        raise SystemExit("no drive link on this row")

    from src.config import configure_api
    configure_api()
    from src.file_resolver import FileResolver
    from src.document_parser import read_document

    print(f"\n=== {args.task} ===")
    print(f"resolving {dl[:78]}")
    staging = os.environ.get("DRA_STAGING", ".dra_cache/staging")
    os.makedirs(staging, exist_ok=True)
    resolver = FileResolver(staging_dir=staging)
    paths = resolver.resolve([dl])
    if not paths:
        raise SystemExit("nothing resolved — check the reference and the share")

    print(f"\n{'file':44s} {'bytes':>11s} {'chars':>10s} {'kept':>8s} {'HIDDEN':>10s}")
    print("-" * 90)
    full_parts, trunc_parts = [], []
    total_chars = total_hidden = 0
    for p in paths:
        name = Path(p).name
        size = os.path.getsize(p)
        try:
            raw = read_document(p)
        except NotImplementedError:
            print(f"{name[:44]:44s} {size:>11,} {'unsupported':>10s}")
            continue
        except Exception as e:
            print(f"{name[:44]:44s} {size:>11,} {'FAILED':>10s}  {e}")
            continue
        raw = raw or ""
        kept = min(len(raw), MAX_CHARS_PER_FILE)
        hidden = max(0, len(raw) - MAX_CHARS_PER_FILE)
        total_chars += len(raw)
        total_hidden += hidden
        print(f"{name[:44]:44s} {size:>11,} {len(raw):>10,} {kept:>8,} "
              f"{hidden:>10,}" + ("   <-- most of it" if hidden > kept else ""))
        full_parts.append(f"### File: {name}\n{raw.strip()}")
        trunc_parts.append(f"### File: {name}\n{truncate_like_fetcher(raw).strip()}")

    full = "\n\n".join(full_parts)
    trunc = "\n\n".join(trunc_parts)
    print("-" * 90)
    print(f"{'TOTAL':44s} {'':>11s} {total_chars:>10,} "
          f"{total_chars - total_hidden:>8,} {total_hidden:>10,}")
    if total_chars:
        print(f"\n  the auditor currently sees {100*(1-total_hidden/total_chars):.1f}% "
              f"of the extracted text")
        print(f"  approx tokens if sent whole: {total_chars//4:,}")

    # --- what truncation costs the provenance check ---
    logic = get_field(row, hmap, "solution_logic")
    nums = declared_numbers(logic)
    print(f"\n=== provenance: {len(nums)} number(s) declared in the solution logic ===")
    in_full = [v for v in nums if _number_appears(v, full)]
    in_trunc = [v for v in nums if _number_appears(v, trunc)]
    lost = [v for v in in_full if v not in in_trunc]
    print(f"  findable in FULL text      : {len(in_full)}/{len(nums)}")
    print(f"  findable in TRUNCATED text : {len(in_trunc)}/{len(nums)}")
    print(f"  LOST to truncation         : {len(lost)}")
    if lost:
        print(f"\n  each of these is a false 'not found in source' today:")
        for v in lost[:12]:
            print(f"    {v:,.4g}".rstrip("0").rstrip("."))
    if in_full:
        print(f"\n  provenance accuracy now: "
              f"{100*len(in_trunc)/len(in_full):.0f}% of findable values are found")

    # --- what an excerpt window would cost ---
    ex = [excerpt_for(v, full, args.radius) for v in in_full]
    ex = [e for e in ex if e]
    ex_chars = sum(len(e) for e in ex)
    print(f"\n=== excerpt window (\u00b1{args.radius} chars around each declared value) ===")
    print(f"  values with a window : {len(ex)}")
    print(f"  total excerpt chars  : {ex_chars:,}  (~{ex_chars//4:,} tokens)")
    if total_chars:
        print(f"  versus full text     : {100*ex_chars/total_chars:.1f}% of it")
    print(f"  and code still searches the FULL text, so provenance stays at "
          f"{len(in_full)}/{len(nums)}")

    print(f"\n  downloads staged in: {staging}")
    print(f"  set DRA_STAGING to move it; a 20 MB pdf then downloads once")


if __name__ == "__main__":
    main()