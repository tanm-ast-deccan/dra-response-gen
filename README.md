# DRA Response-Gen

Two pipelines for building and evaluating a Deep Research Agent benchmark:

- **`dra_harness`** — an MCP-based agent harness that runs LLMs on long-horizon,
  file-grounded research tasks and captures full trajectories. (Deep dive:
  [DATAFLOW.md](DATAFLOW.md).)
- **The augment pipeline** (`run_augment.py` + `src/`) — turns SME prompt
  packages into scored-ready golden packages: golden deliverable, verifier DAG,
  crux set, and crux-only Shapley weights. (Deep dive:
  [README_auditor.md](README_auditor.md).)

The two connect through the benchmark: the harness generates model responses; the
augment pipeline builds the golden packages those responses are scored against.

Full flow: **`run_augment` → SME decisions → `apply_decisions` / `finalize_tasks`
→ `build_final_csv` → `run_score`.** Sections 2–4 below cover the augment, seal,
and score stages; section 1 covers the harness that produces the responses. An
end-to-end diagram is at the bottom.

---

## 1. `dra_harness` — the agent harness

Given a CSV of tasks (prompt + Google Drive link to input files), the harness:

1. Optionally downloads task files from Google Drive.
2. Spins up an MCP tool server per run (Python, bash, file read/write, web
   search, web fetch, calculator).
3. Runs the model in a ReAct loop — it discovers files, reads them via tools,
   runs calculations, searches the web, and writes deliverables.
4. Captures the full turn-by-turn trajectory (tool calls, results, tokens, cost,
   timing).
5. Saves structured JSON for downstream scoring.

Minimal by design: no guardrails, no stagnation detection, minimal system prompt.

### Quick start

```bash
conda activate dra
pip install -r requirements.txt

# API keys in .env
echo "OPENROUTER_API_KEY=sk-or-..." >> .env
echo "SERPER_API_KEY=..."           >> .env   # optional, Google search

# Single task
python -m dra_harness --csv prompt_data.csv \
    --providers hunyuan --passes 1 --task-ids tsk_6177770042 \
    --resolve-files --live --verbose

# Whole CSV
python -m dra_harness --csv prompt_data.csv \
    --providers hunyuan --passes 1 --resolve-files --live
```

Entry points are equivalent: `python -m dra_harness` and
`python -m dra_harness.cli`.

### Architecture

```
prompt_data.csv
      │
      ▼
┌──────────────┐     ┌──────────────────┐
│  csv_loader  │────▶│  file_resolver   │──── Google Drive API
│  (parse CSV) │     │ (download files) │     (only with --resolve-files)
└──────┬───────┘     └──────────────────┘
       │
       ▼
┌──────────────┐     ┌──────────────────┐
│   pipeline   │────▶│      runner      │──── OpenRouter / CometAPI
│  (fan out    │     │  (ReAct loop +   │      (provider.py driver)
│   tasks)     │     │   trajectory)    │
└──────┬───────┘     └────────┬─────────┘
       │                      │
       │              ┌───────▼────────┐
       │              │   mcp_client   │──── stdio subprocess
       │              └───────┬────────┘
       │              ┌───────▼────────┐
       │              │   exec_server  │  9 MCP tools:
       │              │   (FastMCP)    │  • python_execute • bash_execute
       │              └────────────────┘  • read_file • write_file
       │                                   • list_directory • search_in_file
       ▼                                   • web_search • web_fetch • calculator
┌──────────────┐
│   runs_dir/  │  run_<ts>_<provider>/staging/<task>/runs/<provider>__pN/
│  (JSON out)  │  results: run_<ts>_<providers>_<N>tasks.json
└──────────────┘
```

### MCP tools

`exec_server.py` (FastMCP) exposes nine tools to the model:

| Tool | Description |
|---|---|
| `python_execute` | Run Python (pandas, numpy, openpyxl, python-docx, pdfplumber, matplotlib) |
| `bash_execute` | Run shell commands in the run's staging folder |
| `read_file` | Read a file from the staging dir |
| `write_file` | Write a file (deliverables) to the staging dir |
| `list_directory` | List a directory |
| `search_in_file` | Grep-with-context inside a file |
| `web_search` | Serper (if `SERPER_API_KEY` set) else DuckDuckGo fallback |
| `web_fetch` | Fetch and extract text from a URL |
| `calculator` | Evaluate arithmetic expressions |

### Supported models

Registered in `provider.py` (`MODEL_REGISTRY`). All route through OpenRouter
except Doubao (CometAPI). Adding a model is one registry entry.

| Provider | Slug | Notes |
|---|---|---|
| `claude` | `anthropic/claude-sonnet-4-6` | |
| `openai` | `openai/gpt-5` | |
| `gemini` | `google/gemini-3-flash` | |
| `qwen` | `qwen/qwen3.6-27b` | needs `temperature=0.3` for agentic tasks (default override) |
| `hunyuan` | `tencent/hy3` | |
| `doubao` | `doubao-seed-2-1-pro-260628` | via CometAPI, needs `COMETAPI_KEY` |

Default providers when `--providers` is omitted: `claude openai gemini qwen`.

### Configuration

`config.py` is the single source of truth; the CLI only overrides what is
explicitly passed. Key defaults (`GenParams` / `PipelineConfig`):

```python
temperature       = 1.0       # Qwen overridden to 0.3 via agent_overrides
max_tokens        = 32000     # per-turn output limit
reasoning_effort  = "high"
max_turns         = 250       # tool-calling rounds per task
enabled_tools     = ["all"]   # all MCP tools
tool_choice       = "auto"
file_output       = True      # honor detected output_formats
max_cost_usd      = 50.0      # budget guard per run
request_timeout   = 1800      # 30 min per API call
code_exec_timeout = 300       # 5 min per code execution
max_concurrent    = 4         # runs in flight
```

Per-provider overrides via `agent_overrides` (default: `{"qwen": {"temperature": 0.3}}`)
or on the CLI via `--model provider=slug`.

### CLI reference

```bash
python -m dra_harness \
    --csv prompt_data.csv          # required
    --providers claude qwen        # which models (default: claude openai gemini qwen)
    --passes 3                     # runs per model (Pass@k)
    --max-rows 3                   # limit rows loaded
    --task-ids tsk_123,tsk_456     # filter to specific tasks
    --live                         # real API calls (default is dry-run)
    --web-search                   # enable OpenRouter web search
    --tools calculator             # limit local tools (default: all)
    --no-file-output               # skip file generation
    --resolve-files                # download GDrive files first
    --temperature 0.3              # override config.py
    --max-turns 250                # override config.py
    --max-cost 50.0                # override config.py
    --max-tokens 32000             # override config.py
    --timeout 1800                 # override request timeout
    --concurrency 4                # runs in flight
    --model qwen=qwen/qwen3-235b-a22b   # per-provider slug override
    --output-dir ./results         # where to write JSON
    --staging-dir ./staging        # staging root
    --no-save                      # do not write results JSON
    --verbose                      # debug logging
```

### Run directory layout

`run_batch` creates one timestamped run root and stages under it:

```
runs_dir/
└── run_<timestamp>_<provider>/
    └── staging/
        └── tsk_<id>/
            └── runs/
                ├── <provider>__p1/     # per-run workspace
                │   ├── <input files>   → symlinks to shared download
                │   └── <model outputs> ← generated deliverables
                └── <provider>__p2/
```

Results JSON: `run_<timestamp>_<providers>_<N>tasks.json` in the output dir,
including the full per-turn trajectory for every run.

---

## 2. The augment pipeline — golden packages

`run_augment.py` (over `src/`) turns each SME prompt package into a golden
package used to score responses. Per task: audit → apply corrections → one Opus
augment call (golden deliverable + DAG + Sanity-Check anchors + verifiers) →
build the verifier set + weights → deterministic crux selection → crux-only
Shapley weights. It writes `{task_id}_augment.json` (the canonical per-task
record), per-task HTML, and an `augmented_prompt_packages.csv`.

```bash
python run_augment.py --csv prompt_data.csv --out-dir output/augmented
python run_augment.py --csv prompt_data.csv --row 1
python run_augment.py --csv prompt_data.csv --from 1 --to 20 --no-html
```

Three co-equal scoring metrics per response: `crux_cleared` (AND over the crux
set), `crux_verifier_pass_ratio` (k/n), and `crux_shapley_score`. See
[README_auditor.md](README_auditor.md) for the full flow, files, and outputs.

### Selecting rows / tasks

`run_augment.py` is already multi-task. With no selector it augments every row
in the CSV; otherwise it selects by task id (repeatable, deduped per task) or by
row number / range:

```bash
python run_augment.py --csv prompt_data.csv --out-dir output/augmented   # all rows
python run_augment.py --csv prompt_data.csv --task tsk_123 --task tsk_456 # by task id
python run_augment.py --csv prompt_data.csv --row 1                       # single row
python run_augment.py --csv prompt_data.csv --from 1 --to 20 --no-html    # a range
```

| Argument | Required | What it does |
|---|---|---|
| `--csv` | yes | Authoring CSV of prompt packages to audit. |
| `--out-dir` | no | Output folder. Default `output/augmented`. |
| `--task` | no | Augment only this task id; repeatable. Overrides row selection. |
| `--row` | no | Augment only this row number. |
| `--from` / `--to` | no | Augment an inclusive range of row numbers. |
| `--model` | no | Override the auditor LLM (default: built-in judge model). |
| `--no-files` | no | Skip the Drive input-file fetch (auditor file layer off). |
| `--no-html` | no | Skip writing `_augment.html` / `_golden.html`. |

**Outputs per task:** `{task_id}_augment.json` (canonical record),
`{task_id}_augment.html` (SME review page), `{task_id}_golden.html`, plus a
combined `augmented_prompt_packages.csv`.

---

## 3. Sealing decisions and building the final CSV

After augmentation, an SME opens each `_augment.html`, makes decisions, and the
page saves a `decisions_{task_id}_{date}.json`. Those decisions are then applied
to seal each task, and the sealed tasks are collected into a final CSV.

**Sealed vs scoreable.** `apply_decisions.py` applies the SME's rulings and marks
the package `"sealed": true`. A *sealed* task is frozen and authoritative; a
partially decided package is refused rather than half-sealed (use `--force` to
override). Sealed is not the same as *scoreable*: a sealed task can still be
flagged not-scoreable (`proceedable AND not error`) if a blocking arithmetic
error or an unresolved judgment gap remains — it is finalized but unusable for
scoring, with a reason recorded.

### 3a. `apply_decisions.py` — seal one task

```bash
python apply_decisions.py \
    --augment  output/augmented/tsk_8695111330_augment.json \
    --decisions ~/Downloads/decisions_tsk_8695111330_2026-08-05.json \
    --out output/final
```

| Argument | Required | What it does |
|---|---|---|
| `--augment` | yes | The task's `_augment.json` from `run_augment`. |
| `--decisions` | yes | The `decisions_{task}.json` the SME saved from the HTML. |
| `--out` | no | Folder for `{task_id}_final.json`. Default `output/final`. |
| `--force` | no | Apply despite an incomplete file or a run-hash mismatch. |

**Output:** one sealed `{task_id}_final.json`.

### 3b. `build_final_csv.py` — collect sealed tasks into a CSV

Already multi-task: `--final` takes one or more `_final.json` files **or a
directory** of them (it globs `*_final.json`).

```bash
python build_final_csv.py \
    --csv   ./SME_data/tasks.csv \
    --final output/final \
    --out   output/final_tasks.csv
```

| Argument | Required | What it does |
|---|---|---|
| `--csv` | yes | The original authoring CSV (base columns per task). |
| `--final` | yes | `_final.json` file(s), or a directory of them. |
| `--out` | yes | Path of the CSV to write. |
| `--only-sealed` | no | Write only sealed tasks; skip the rest (default keeps all rows, blank where unsealed). |
| `--slim` | no | Also write a narrower review sheet. Default `{out}_slim.csv`. |

**Output:** `final_tasks.csv` (every original column/row preserved, sealed
artifacts written into appended columns) and a slim review sheet.

### 3c. `finalize_tasks.py` — seal many tasks and build the CSV in one go

`apply_decisions.py` is the only single-task stage. `finalize_tasks.py` wraps it:
it pairs every `_augment.json` in one folder with the `decisions_*.json` in
another (matched by the task id inside each file), seals each pair, then calls
`build_final_csv.py` on the resulting folder. Output is **byte-identical** to
running the two steps by hand per task.

```bash
python finalize_tasks.py \
    --augment-dir   output/augmented \
    --decisions-dir ./decisions \
    --csv           ./SME_data/tasks.csv \
    --out           output/final_tasks.csv
```

| Argument | Required | What it does |
|---|---|---|
| `--augment-dir` | yes | Folder of `{task_id}_augment.json` files. |
| `--decisions-dir` | yes | Folder of `decisions_{task_id}_{date}.json` files. |
| `--csv` | yes | Authoring CSV, passed through to the build step. |
| `--out` | yes | Path of the final CSV to write. |
| `--final-dir` | no | Where to place intermediate `_final.json`. Default `_final/` beside `--out`. |
| `--slim` | no | Passed to the build step (narrower review sheet). |
| `--only-sealed` | no | Passed to the build step (only sealed rows). |
| `--force` | no | Apply despite incomplete/mismatched decisions (per task). |
| `--continue-on-error` | no | Skip a task whose apply step fails instead of stopping. |

Pairing is by the task id embedded in each filename; if several decision files
exist for one task (SME re-saves), the newest by modification time wins. Tasks
with no matching decisions file are reported as *awaiting decisions* and skipped.
Exit code is `0` on success and `1` on failure, so a UI can detect the outcome.
Requires `apply_decisions.py` and `build_final_csv.py` in the same folder.

---

## 4. `run_score.py` — score responses against sealed packages

Grades harness responses against the augmented/sealed packages, in parallel, and
ranks tasks by the three crux metrics. The `--augmented-csv` is the
`augmented_prompt_packages.csv` from `run_augment`; `--results-json` is a path or
glob of the harness results JSON.

```bash
python run_score.py \
    --augmented-csv output/augmented/augmented_prompt_packages.csv \
    --results-json 'results/*.json' \
    --workers 6
```

| Argument | Required | What it does |
|---|---|---|
| `--augmented-csv` | no* | The `augmented_prompt_packages.csv` (or config default). |
| `--results-json` | no* | Path or glob of harness results JSON. |
| `--out-dir` | no | Where to write `scores.csv` / `task_ranking.csv`. |
| `--workers` | no | Parallel scoring workers. |
| `--staging-remap` | no | Rewrite a renamed staging path segment: `OLD=NEW` (default `staging=staging_1`). |
| `--model` | no | Override the judge model for LLM-graded verifiers. |
| `--require-both` | no | Only score tasks that succeeded in every listed results file (intersect across models). |
| `--exclude-tasks` | no | Task ids to skip. |

<sub>*Defaults come from `augment_score_config`; pass explicitly to override.</sub>

**Outputs (in `--out-dir`):** `scores.csv` (one row per task/provider/pass with
the three crux metrics) and `task_ranking.csv` (per-task mean over runs, sorted
hardest-first).

---

## End-to-end sequence

```
                 authoring CSV (prompt packages)
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
  run_augment.py                   dra_harness  (runs models on the tasks)
  → *_augment.json                 → results_*.json  (per-turn trajectories)
  → *_augment.html                       │
  → augmented_prompt_packages.csv        │
          │                              │
   SME opens *_augment.html              │
   → decisions_*.json                    │
          │                              │
   apply_decisions.py  (per task)        │
     ── or ──                            │
   finalize_tasks.py   (whole folder)    │
   → *_final.json                        │
          │                              │
   build_final_csv.py                    │
   (finalize_tasks runs this for you)    │
   → final_tasks.csv (+ _slim.csv)       │
          │                              │
          └──────────────┬───────────────┘
                         ▼
                   run_score.py
              → scores.csv, task_ranking.csv
```

The augment/seal branch builds the golden packages; the harness branch generates
model responses. `run_score.py` joins them: sealed packages in, scored and ranked
tasks out.

---

## Environment variables

```bash
# Required for live harness runs
OPENROUTER_API_KEY=sk-or-...

# Optional
SERPER_API_KEY=...                              # Google search (else DuckDuckGo)
COMETAPI_KEY=...                                # Doubao via CometAPI
GOOGLE_APPLICATION_CREDENTIALS=./gcp_key.json   # GDrive file downloads
```

## Requirements

- Python 3.11+
- `fastmcp >= 3.4.4`
- `openai` (OpenRouter API compatibility)
- `pandas`, `openpyxl`, `pdfplumber`, `python-docx` (agent tools)
- `duckduckgo-search` (fallback web search)
- `google-api-python-client`, `google-auth-httplib2` (GDrive downloads)
- augment pipeline also uses `PyMuPDF`, `python-dotenv`
- the seal/build/score stages (`apply_decisions`, `build_final_csv`,
  `finalize_tasks`, `run_score`) add no new dependencies — `finalize_tasks.py`
  is standard-library only and imports `apply_decisions` in-process.

## License

Internal — Deccan AI.