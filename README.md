# DRA Harness

A minimal MCP-based Deep Research Agent harness for benchmarking LLMs on long-horizon, file-grounded research tasks. Built for apples-to-apples comparison with [APEX-Agents](https://arxiv.org/abs/2601.14242).

## What it does

Given a CSV of benchmark tasks (prompt + Google Drive link to input files), the harness:

1. Downloads task files from Google Drive
2. Spins up an MCP tool server per run (Python, bash, web search, file write, calculator)
3. Runs the model in a ReAct loop — model discovers files, reads them via tools, runs calculations, searches the web, and writes deliverables
4. Captures full turn-by-turn trajectories (tool calls, results, tokens, cost, timing)
5. Saves structured JSON results for downstream scoring

No guardrails. No hand-holding. No stagnation detection. The model succeeds or fails on its own — same as APEX.

## Quick start

```bash
# Setup
conda activate dra
pip install -r requirements.txt

# Add API keys to .env
echo "OPENROUTER_API_KEY=sk-or-..." >> .env
echo "SERPER_API_KEY=..."           >> .env   # optional, Google search

# Run a single task
python -m dra_harness \
    --csv prompt_data.csv \
    --providers hunyuan \
    --passes 1 \
    --task-ids tsk_6177770042 \
    --resolve-files \
    --live \
    --verbose

# Run all tasks
python -m dra_harness \
    --csv prompt_data.csv \
    --providers hunyuan \
    --passes 1 \
    --resolve-files \
    --live \
    --verbose
```

## Architecture

```
prompt_data.csv
      │
      ▼
┌─────────────┐     ┌──────────────────┐
│  csv_loader  │────▶│  file_resolver   │──── Google Drive API
│  (parse CSV) │     │  (download files)│
└──────┬──────┘     └──────────────────┘
       │
       ▼
┌─────────────┐     ┌──────────────────┐
│   pipeline   │────▶│     runner       │──── OpenRouter / CometAPI
│  (fan out    │     │  (ReAct loop +   │
│   tasks)     │     │   trajectory)    │
└──────┬──────┘     └───────┬──────────┘
       │                    │
       │              ┌─────▼──────────┐
       │              │   mcp_client   │──── stdio subprocess
       │              └─────┬──────────┘
       │              ┌─────▼──────────┐
       │              │  exec_server   │  6 MCP tools:
       │              │  (fastmcp)     │  • python_execute
       │              └────────────────┘  • bash_execute
       │                                  • web_search (Serper/DDG)
       ▼                                  • web_fetch
┌─────────────┐                           • write_file
│   results/   │                           • calculator
│  (JSON out)  │
└─────────────┘
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `python_execute` | Run Python code. Libraries: pandas, numpy, openpyxl, python-docx, pdfplumber, matplotlib |
| `bash_execute` | Run shell commands. Working directory is the task staging folder |
| `write_file` | Write text files (md, txt, csv, json) to the staging directory |
| `web_search` | Google results via Serper (if `SERPER_API_KEY` set), else DuckDuckGo fallback |
| `web_fetch` | Fetch and extract text from a URL (HTML tags stripped, 30K char limit) |
| `calculator` | Evaluate arithmetic expressions |

## Supported Models

All models route through OpenRouter except Doubao (CometAPI). Adding a new model = one entry in `provider.py`:

| Provider | Model | Slug | Notes |
|----------|-------|------|-------|
| `hunyuan` | Tencent Hunyuan 3 | `tencent/hy3` | Free until Jul 21, 2026 |
| `qwen` | Qwen 3.6 27B | `qwen/qwen3.6-27b` | Needs temp=0.3 for agentic tasks |
| `claude` | Claude Sonnet 4.6 | `anthropic/claude-sonnet-4-6` | |
| `openai` | OpenAI o3 | `openai/o3` | |
| `gemini` | Gemini 2.5 Pro | `google/gemini-2.5-pro` | |
| `kimi` | Kimi K2 | `moonshotai/kimi-k2` | |
| `deepseek` | DeepSeek R1 | `deepseek/deepseek-r1` | |
| `grok` | Grok 3 | `x-ai/grok-3` | |
| `doubao` | Doubao 2.1 Pro | `doubao-seed-2-1-pro-260628` | Via CometAPI, needs `COMETAPI_KEY` |

## Configuration

`config.py` is the single source of truth. CLI only overrides when explicitly passed.

```python
# Key defaults
temperature      = 1.0      # provider default (Qwen needs 0.3)
max_tokens       = 16000    # per-turn output limit
reasoning_effort = "medium" # "", "low", "medium", "high"
max_turns        = 150      # max tool-calling rounds per task
tool_choice      = "auto"   # model decides when to use tools and when to stop
max_cost_usd     = 5.0      # budget guard per task
request_timeout  = 1800     # 30 min per API call
code_exec_timeout = 300     # 5 min per code execution
```

Per-model overrides via `agent_overrides`:
```python
agent_overrides = {
    "qwen": {"temperature": 0.3},
}
```

## CLI Reference

```bash
python -m dra_harness \
    --csv prompt_data.csv        # required: task CSV
    --providers hunyuan qwen     # which models to run
    --passes 3                   # runs per model (Pass@k)
    --task-ids tsk_123,tsk_456   # filter to specific tasks
    --resolve-files              # download GDrive files before running
    --live                       # make real API calls (default is dry-run)
    --verbose                    # debug logging
    --temperature 0.3            # override config.py
    --max-turns 250              # override config.py
    --max-cost 50.0              # override config.py
    --model qwen=qwen/qwen3-235b-a22b  # override model slug
    --output-dir ./results       # where to write JSON
    --no-save                    # skip writing results to disk
```

## Staging Layout

```
staging/
└── tsk_6177770042/
    └── runs/
        ├── hunyuan__p1/                  # per-run workspace
        │   ├── Finance_Assumptions.xlsx  → symlink to shared download
        │   ├── Supplier_Cost_Data.xlsx   → symlink
        │   ├── Procurement_Report.pdf    → symlink
        │   └── Board_Memo.md            ← model-generated output
        ├── hunyuan__p2/
        └── qwen__p1/
```

Input files are downloaded once; each run gets symlinks. Model-generated outputs stay in the run directory.

## Output Format

Results JSON includes full trajectory with per-turn detail:

```json
{
  "task_id": "tsk_6177770042",
  "run_id": "tsk_6177770042__hunyuan__p1",
  "provider": "hunyuan",
  "model": "tencent/hy3",
  "turns": 13,
  "total_cost_usd": 0.019,
  "total_duration_sec": 482.1,
  "completed": true,
  "forced_stop": false,
  "response_text": "...",
  "trajectory": [
    {
      "turn": 1,
      "assistant_text": "I'll start by exploring the working directory...",
      "tool_calls": [{"name": "bash_execute", "arguments": {"command": "ls -la"}}],
      "tool_results": [{"name": "bash_execute", "result": "..."}],
      "input_tokens": 1305,
      "output_tokens": 52,
      "cost_usd": 0.0002
    }
  ]
}
```

## APEX-Agents Parity

| Dimension | APEX-Agents | This harness |
|-----------|------------|--------------|
| Agent loop | ReAct, 250 max | ReAct, 150 max (configurable) |
| Tools | MCP (filesystem, shell, browser) | MCP (python, bash, web, fetch, write, calc) |
| File discovery | Via tools only | Via tools only |
| System prompt | Minimal | Minimal |
| Error recovery | None | None |
| Guardrails | None | None |
| Sandbox | Docker per task | Per-run staging dir |
| Multi-run | Pass@8 | Pass@k (configurable) |

## Environment Variables

```bash
# Required
OPENROUTER_API_KEY=sk-or-...

# Optional
SERPER_API_KEY=...              # Google search (2,500 free queries)
COMETAPI_KEY=...                # For Doubao model via CometAPI
GOOGLE_APPLICATION_CREDENTIALS=./gcp_key.json  # GDrive file downloads
```

## Requirements

- Python 3.11+
- `fastmcp >= 3.4.4`
- `openai` (for OpenRouter API compatibility)
- `pandas`, `openpyxl`, `pdfplumber`, `python-docx` (for agent tools)
- `duckduckgo-search` (fallback web search)
- `google-api-python-client`, `google-auth-httplib2` (GDrive downloads)

## License

Internal — Deccan AI.