#!/usr/bin/env bash
# scan_failures.sh — report which runs FAILED in a harness results JSON.
#
#   ./scan_failures.sh results/run.json          # human summary
#   ./scan_failures.sh -q results/run.json        # just failed task_ids (one per line)
#   ./scan_failures.sh -v results/run.json        # include the error string per run
#   ./scan_failures.sh *.json                     # scan several files
#
# A run counts as FAILED if any of:
#   - completed == false
#   - error is set (non-null / non-empty)
#   - completed == true but output_files is empty  (no deliverable produced)
#
# Uses jq if available; otherwise falls back to embedded Python (always present
# in the harness env). No other dependencies.

set -euo pipefail

QUIET=0; VERBOSE=0
FILES=()
for arg in "$@"; do
  case "$arg" in
    -q|--quiet)   QUIET=1 ;;
    -v|--verbose) VERBOSE=1 ;;
    -h|--help)
      sed -n '2,20p' "$0"; exit 0 ;;
    *) FILES+=("$arg") ;;
  esac
done

if [ "${#FILES[@]}" -eq 0 ]; then
  echo "usage: $0 [-q|-v] <results.json> [more.json ...]" >&2
  exit 2
fi

have_jq=0
command -v jq >/dev/null 2>&1 && have_jq=1

scan_one() {
  local f="$1"
  if [ ! -f "$f" ]; then
    echo "!! not found: $f" >&2
    return 0
  fi

  if [ "$have_jq" -eq 1 ]; then
    # emit TSV: status<TAB>task_id<TAB>provider<TAB>pass<TAB>reason
    jq -r '
      (.results // [])[]
      | ((.completed // false) == false) as $notcomp
      | (((.error // "") | tostring | length) > 0) as $err
      | (((.output_files // []) | length) == 0) as $noout
      | select($notcomp or $err or $noout)
      | [ "FAIL",
          (.task_id // "?"),
          (.provider // "?"),
          ((.pass_index // 0) | tostring),
          ( if $err then "error: " + (.error | tostring)
            elif $notcomp then "not completed"
            else "no output_files" end )
        ] | @tsv
    ' "$f"
  else
    python3 - "$f" <<'PY'
import json, sys
f = sys.argv[1]
try:
    data = json.load(open(f, encoding="utf-8"))
except Exception as e:
    print(f"!! parse error in {f}: {e}", file=sys.stderr); sys.exit(0)
for r in (data.get("results") or []):
    notcomp = not r.get("completed", False)
    err = str(r.get("error") or "")
    noout = len(r.get("output_files") or []) == 0
    if not (notcomp or err or noout):
        continue
    if err:      reason = "error: " + err
    elif notcomp: reason = "not completed"
    else:         reason = "no output_files"
    print("\t".join(["FAIL",
                     str(r.get("task_id","?")),
                     str(r.get("provider","?")),
                     str(r.get("pass_index",0)),
                     reason]))
PY
  fi
}

total_fail=0
for f in "${FILES[@]}"; do
  # collect this file's failures
  mapfile -t lines < <(scan_one "$f")
  n=${#lines[@]}
  # drop empty trailing entry mapfile can create
  if [ "$n" -eq 1 ] && [ -z "${lines[0]}" ]; then n=0; fi

  if [ "$QUIET" -eq 1 ]; then
    # unique failed task_ids only
    for l in "${lines[@]}"; do
      [ -z "$l" ] && continue
      printf '%s\n' "$l" | cut -f2
    done
    total_fail=$((total_fail + n))
    continue
  fi

  echo "=== $f ==="
  if [ "$n" -eq 0 ]; then
    echo "  no failed runs"
  else
    for l in "${lines[@]}"; do
      [ -z "$l" ] && continue
      tid=$(printf '%s' "$l" | cut -f2)
      prov=$(printf '%s' "$l" | cut -f3)
      pidx=$(printf '%s' "$l" | cut -f4)
      reason=$(printf '%s' "$l" | cut -f5)
      if [ "$VERBOSE" -eq 1 ]; then
        printf '  FAIL  %-16s %-10s p%s  %s\n' "$tid" "$prov" "$pidx" "$reason"
      else
        printf '  FAIL  %-16s %-10s p%s  (%s)\n' "$tid" "$prov" "$pidx" "${reason%%:*}"
      fi
    done
    echo "  -> $n failed run(s)"
  fi
  total_fail=$((total_fail + n))
done

if [ "$QUIET" -eq 0 ]; then
  echo
  echo "total failed runs across ${#FILES[@]} file(s): $total_fail"
fi

# exit non-zero if any failures found (handy in CI / && chains)
[ "$total_fail" -eq 0 ]
