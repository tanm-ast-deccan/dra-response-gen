# DRA Augment Pipeline

Turns SME prompt packages into scored-ready **golden** packages. It runs on top
of the two-call auditor (`src/auditor.py`) and, in one further Opus "augment"
call, produces the golden deliverable, the verifier DAG, the Sanity-Check
anchors, and any added verifiers — then deterministically selects the crux set
and computes crux-only Shapley weights.

Lives in `src/` alongside the auditor it extends. Entry point is `run_augment.py`
at the repo root.

## What it does (per task, one pass, no SME gate)

1. **Audit** — runs the existing `audit_task` (two-call auditor + arithmetic
   verification), yielding verified arithmetic claims, a `corrected_solution_logic`,
   and a list of change items.
2. **Apply corrections** — `MECHANICAL` edits are applied silently; `JUDGMENT_REQUIRED`
   edits are also applied now but recorded in `judgment_pending_json` for a later
   SME to review.
3. **Augment (one Opus call)** — emits together: the **golden deliverable**, the
   **DAG edges**, the two **Sanity-Check anchor sets** (Lazy-AI trap side +
   expert-path side), and any **augmented verifiers**.
4. **Build** the verifier set + DAG + base weights (`verifier_weights.compute_weights`).
5. **Select the crux set deterministically** (`select_crux`) — a verifier is crux
   iff it is graph-connected (ancestor or descendant, following DAG edges in either
   direction) to a Sanity-Check anchor. No LLM judgement at selection time. Crux
   candidates that have no frozen expected value are dropped as unscoreable and
   surfaced in `dropped_no_expected`.
6. **Compute crux-only Shapley weights** (`crux_shapley`) — exact for ≤8 crux
   nodes, Monte-Carlo above, renormalized to sum to 1.0. Shapley's Efficiency
   axiom prevents a shared root cause being double-counted across its descendants.

## The three co-equal metrics (per response, via `score_crux`)

- `crux_cleared` — every crux verifier passed (logical AND; an unobserved crux
  verifier is NOT a pass and blocks clearing).
- `crux_verifier_pass_ratio` — plain k/n over the crux set.
- `crux_shapley_score` — sum of crux-only Shapley weights of the passed crux verifiers.

`crux_cleared` is reported alongside the Shapley score, not replaced by it,
because Shapley under-weights the terminal answer node by construction and
`crux_cleared` requires the final-answer verifier to pass regardless of weight.

## Files

- `run_augment.py` — CLI. Writes `augmented_prompt_packages.csv` + per-task HTML.
- `src/augment_task.py` — orchestrates audit → apply → augment → DAG → crux → Shapley
  (the `augment_task` function and its `AugmentResult`).
- `src/augment_templates.py` — the single augment-call prompt (`AUGMENT_SYSTEM_PROMPT`,
  `AUGMENT_TEMPLATE`).
- `src/crux_shapley.py` — `select_crux`, `crux_shapley`, the ancestor/descendant
  reachability, and the three metrics (`score_crux`).
- `src/augment_report.py` — `write_augment_report` → `{task_id}_augment.html`,
  `write_golden_report` → `{task_id}_golden.html`.

Depends on the existing auditor modules: `auditor.py`, `verifier_weights.py`,
`verifier_parser.py`, `document_parser.py`, `gdrive_raw_fetcher.py`,
`prompt_evaluator.py`.

## Run

```bash
pip install python-docx openpyxl PyMuPDF python-dotenv --break-system-packages

# whole CSV
python run_augment.py --csv prompt_data.csv --out-dir output/augmented

# a single row (1-indexed)
python run_augment.py --csv prompt_data.csv --row 1

# a range, skipping the per-task HTML
python run_augment.py --csv prompt_data.csv --from 1 --to 20 --no-html
```

### CLI flags

| Flag | Meaning |
|---|---|
| `--csv` | Source prompt-package CSV (required) |
| `--row N` | Augment only row N (1-indexed) |
| `--from A --to B` | Augment rows A..B inclusive |
| `--out-dir` | Output directory (default `output/augmented`) |
| `--model` | Override the augment model |
| `--no-files` | Skip Drive input-file fetching |
| `--no-html` | Skip the per-task `_augment.html` / `_golden.html` |

## Outputs

Per task, written to `--out-dir`:

- `{task_id}_augment.json` — the full `AugmentResult` (the canonical per-task
  record downstream tools read).
- `{task_id}_augment.html` — corrections, DAG, anchors, verifiers.
- `{task_id}_golden.html` — the golden deliverable.

Batch:

- `augmented_prompt_packages.csv` — every original column, plus the augmented
  columns below. Rows that error still carry every column (blanked) so the CSV
  never loses columns.

### Augmented CSV — added columns

`corrected_solution_logic`, `golden_deliverable`, `augmented_verifiers`,
`dag_json`, `crux_verifier_ids`, `crux_shapley_weights_json`, `base_weights_json`,
`audit_verdict`, `changes_applied_json`, `judgment_pending_json`, `augment_error`.

## Notes / decisions baked in

- The golden deliverable is generated in the SAME Opus call as the
  corrections / DAG / anchors.
- Crux weighting uses **crux-only Shapley** (not full-DAG); depth-weighting
  schemes are not used.
- Crux selection is deterministic given the anchors + DAG — the only LLM
  judgement is upstream, in producing the anchors and verifiers.
- The `{task_id}_augment.json` record is the per-task interface consumed by the
  downstream scorer and by the SME-delivery tooling.