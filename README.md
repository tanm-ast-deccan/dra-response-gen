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

## License

Internal — Deccan AI.