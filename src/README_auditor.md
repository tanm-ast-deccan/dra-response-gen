# DRA Augment Pipeline (new modules)

Drops into your existing `src/` (the auditor repo). Four new modules + one CLI,
built on top of your `auditor.py`, `verifier_weights.py`, `verifier_parser.py`,
`document_parser.py`, `gdrive_raw_fetcher.py`, `prompt_evaluator.py`.

## What it does (per task, one pass, no SME gate)
1. Runs your existing `audit_task` (two-call auditor + arithmetic verification).
2. Applies corrections directly. MECHANICAL edits applied silently; JUDGMENT
   edits also applied now but recorded in `judgment_pending_json` for later SME.
3. ONE Opus "augment" call emits, together: the **golden deliverable**, the
   **DAG edges**, the two **Sanity-Check anchor sets** (Lazy-AI trap side +
   expert-path side), and any **augmented verifiers**.
4. Builds the verifier set + DAG + base weights (`compute_weights`).
5. **Selects the crux set deterministically**: a verifier is crux iff it is
   graph-connected (ancestor/descendant) to a Sanity-Check anchor. No LLM
   judgement at selection time.
6. Computes **crux-only Shapley weights** (exact ≤8 crux nodes, Monte-Carlo
   above; renormalized to sum to 1.0). Shapley's Efficiency axiom is what
   prevents a shared root cause being double-counted across its descendants.

## The three co-equal metrics (computed per response by `score_crux`)
- `crux_cleared` — every crux verifier passed (logical AND; an unobserved crux
  verifier is NOT a pass and blocks clearing).
- `crux_verifier_pass_ratio` — plain k/n over the crux set.
- `crux_shapley_score` — sum of crux-only Shapley weights of passed crux verifiers.

## Files
- `src/crux_shapley.py` — crux selection + Shapley + the 3 metrics.
- `src/augment_templates.py` — the single augment-call prompt (+ system).
- `src/augment_task.py` — orchestrates audit → apply → augment → DAG → crux → Shapley.
- `src/augment_report.py` — `{task_id}_augment.html` and `{task_id}_golden.html`.
- `run_augment.py` — CLI; writes `augmented_prompt_packages.csv` + per-task HTML.

## Run
```bash
pip install python-docx openpyxl PyMuPDF python-dotenv --break-system-packages
python run_augment.py --csv prompt_data.csv --out-dir output/augmented
# single row / range:
python run_augment.py --csv prompt_data.csv --row 1
python run_augment.py --csv prompt_data.csv --from 1 --to 20
```

## Augmented CSV — added columns
`corrected_solution_logic`, `golden_deliverable`, `augmented_verifiers`,
`dag_json`, `crux_verifier_ids`, `crux_shapley_weights_json`, `base_weights_json`,
`audit_verdict`, `changes_applied_json`, `judgment_pending_json`, `augment_error`.

## Notes / decisions baked in
- Golden deliverable is generated in the SAME Opus call as corrections/DAG/anchors.
- Crux weighting uses **crux-only Shapley** (not full-DAG); depth schemes dropped.
- Shapley under-weights the terminal answer node by construction — that is why
  `crux_cleared` (which requires the final-answer verifier to pass regardless of
  weight) is reported alongside, not replaced by, the Shapley score.
- The scorer (grading the 68 Hunyuan responses against this augmented package
  via `document_parser` extraction + `score_crux`) is the next module.
