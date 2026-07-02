# DRA Pipeline — Complete Flow (Granular, Plain-English)

This document explains **exactly** what happens inside the DRA Benchmark Evaluation
Framework, step by step, from the moment you have an input CSV until you have
final output files on disk. It is written in simple terms, with concrete
examples, and it tries to cover **every branch and scenario** the code can take.

> TL;DR of the whole thing:
> **A CSV of research prompts → turned into structured tasks → each task is sent
> to 4 AI research agents at the same time → each agent's report (and optionally a
> generated Excel/Word/PPT file) is collected → everything is saved as JSON on
> disk for a human/automated scorer to grade later.**

---

## Table of contents

1. [The big picture](#1-the-big-picture)
2. [The vocabulary (data models)](#2-the-vocabulary-data-models)
3. [Stage 0 — Environment & startup](#3-stage-0--environment--startup)
4. [Stage 1 — Reading the CSV (`csv_loader.py`)](#4-stage-1--reading-the-csv-csv_loaderpy)
5. [Stage 2 — Building a PromptPackage](#5-stage-2--building-a-promptpackage)
6. [Stage 3 — Resolving files (`file_resolver.py`)](#6-stage-3--resolving-files-file_resolverpy)
7. [Stage 4 — The dispatcher (`task_dispatcher.py`)](#7-stage-4--the-dispatcher-task_dispatcherpy)
8. [Stage 5 — The four agent adapters](#8-stage-5--the-four-agent-adapters)
9. [Stage 6 — File output generation (the hard part)](#9-stage-6--file-output-generation-the-hard-part)
10. [Stage 7 — Aggregation & storage](#10-stage-7--aggregation--storage)
11. [Stage 8 — Post-processing utilities](#11-stage-8--post-processing-utilities)
12. [The MCP servers (optional side-channels)](#12-the-mcp-servers-optional-side-channels)
13. [End-to-end worked example](#13-end-to-end-worked-example)
14. [Every scenario / decision table](#14-every-scenario--decision-table)
15. [Failure modes & what happens](#15-failure-modes--what-happens)

---

## 1. The big picture

Imagine you are running a science experiment to compare 4 AI "deep research"
products on the same hard questions. To be fair, every product must get:

- the **same question** (prompt),
- the **same supporting files** (a corpus of PDFs/Excel/Word),
- the **same rules** about whether they're allowed to use the internet,
- and produce the **same kind of deliverable** (a written report, and sometimes
  a generated Excel/Word/PowerPoint file).

This repo is the machine that does that fairly and records the results.

The four agents:

| Agent | Product | Personality in one line |
|-------|---------|--------------------------|
| **claude** | Anthropic Claude (opus/sonnet) | The "glass box" — we run its tool loop ourselves and log every step |
| **openai** | OpenAI `o3-deep-research` | A "black box" — we submit, poll, and read the result |
| **gemini** | Google `deep-research-pro` | A "black box" that needs the most hand-holding (multi-pass + local code execution) |
| **perplexity** | Perplexity `sonar-deep-research` | The most limited — always searches the web, no file upload |

The flow has two **independent dimensions** that change behavior everywhere:

- **Dimension A — Is web search allowed?** (driven by IAT tier + `enforce_iat`)
- **Dimension B — Does the prompt require a generated file?** (driven by
  `detect_output_formats`)

Almost every "scenario" in this document is a combination of A and B.

---

## 2. The vocabulary (data models)

These are defined in `models.py`. They are the "nouns" passed between modules.

```
PromptPackage   →  what the SME authored (the full unit of work)
      │ .to_research_task()
      ▼
ResearchTask    →  the agent-agnostic instruction handed to each adapter
      │ adapter.run()
      ▼
AgentResult     →  what ONE agent produced for ONE pass
      │ collected into
      ▼
DispatchResult  →  all agents' results for ONE task
```

Plus two config/observability objects:

- `DispatchConfig` — knobs: which agents, how many passes, dry-run, IAT
  enforcement, output dirs.
- `ToolCall` — one logged tool invocation (only Claude fills these).

### `PromptPackage` (the input unit)
The important fields:
- `task_id` — unique ID (e.g. `tsk_260217210354676WR0MM`)
- `prompt` — the actual research question text
- `file_paths` — list of files (local paths or GDrive URLs)
- `research_type` — `CRP | RCP | SCP | LDP | FSP`
- `iat_type` — `IAT-1 | IAT-2 | IAT-3` (derived from research_type)
- `output_formats` — auto-detected, e.g. `["xlsx"]` or `[]`
- `solution_steps`, `lazy_ai_prediction` — metadata for the *scorer*, never sent to agents

### `ResearchTask` (the agent-facing unit)
Same data, but agent-relevant:
- `web_search_enabled` — the boolean that actually turns search on/off
- `max_iterations` (25), `max_cost_usd` (15.0), `timeout_seconds` (900)
- `output_files_dir` — where adapters must write generated files
- Helper properties: `is_closed` (IAT-1), `is_hybrid` (IAT-2/3)

### `AgentResult` (the per-agent output)
- `response_text` — the final markdown report
- `citations` — `[{url, title, snippet}]`
- `output_files` — absolute paths to generated xlsx/docx/pptx
- `output_file_errors` — `{format: reason}` when a required file wasn't produced
- `input_tokens`/`output_tokens`/`total_cost_usd`/`total_duration_sec`
- `completed`/`forced_stop`/`error`
- `tool_call_log` — **Claude only**

---

## 3. Stage 0 — Environment & startup

Every entry point (`csv_loader.py`, `task_dispatcher.py`, `run_research.py`)
calls `load_env()` from `env_loader.py` at import time.

What `load_env()` does:
1. Looks for a `.env` file in this order: explicit path → current working dir →
   the script's own directory.
2. If `python-dotenv` is installed, uses it; otherwise a built-in mini parser
   handles `KEY=value`, quotes, `export KEY=...`, and comments.
3. It **never overrides** variables already set in the shell.
4. It is safe to call many times (a `_loaded` flag means it only runs once).

The keys it expects (from `.env`, see `env.example`):
```
ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, PERPLEXITY_API_KEY
GOOGLE_SERVICE_ACCOUNT_KEY  (or GOOGLE_CLIENT_SECRETS for OAuth)
```

**Scenario: a key is missing.** Nothing fails at startup. The failure happens
later, *inside the specific adapter*, when it tries to construct its client. That
agent's pass becomes a failed `AgentResult`; the other agents are unaffected
(see `check.txt` line 26: Claude failed because `anthropic` wasn't installed, but
the run continued and 2/4 agents still succeeded).

---

## 4. Stage 1 — Reading the CSV (`csv_loader.py`)

This is the primary production entry. Example command:

```bash
python csv_loader.py \
    --csv ./input/prompt_data.csv \
    --resolve-files --staging-dir /tmp/eval_files \
    --dispatch --live \
    --agents claude openai gemini perplexity \
    --results-dir ./results --output ./results/json
```

### 4.1 Reading rows — `load_csv()`

The CSV is opened as `utf-8-sig` (this strips the BOM that Excel adds).

**Column aliasing — the clever part.** Real CSVs from different SMEs have
different header names ("POC Name" vs "full_name", "Prompts" vs "prompt", "SC" vs
"sanity_check"). Instead of hardcoding, `COLUMN_ALIASES` maps a *logical field* to
a list of accepted header spellings, case-insensitive, first match wins.

Example: the logical field `prompt` matches any of
`["prompt", "prompts", "prompt_text", "question", "task description", ...]`.

`resolve_columns()` builds the mapping once for the whole file:
- If a **required** field (`task_id` or `prompt`) can't be resolved → hard
  `ValueError` listing the available headers.
- Unknown columns → just a warning; they're ignored.

**Per-row cleaning (`load_csv` loop):**
- Fully empty rows → skipped.
- Rows with empty `prompt` → skipped with a warning.
- `task_id`:
  - Excel sometimes turns IDs into floats: `"28115.0B"` → regex normalizes to
    `"28115B"`.
  - If blank → auto-generated as `"{sme_slug}_{6 hex}"` (e.g.
    `anagha_patne_57e965`, exactly what we see in `check.txt`).

Output: a list of clean dicts with keys like `task_id`, `sme_name`, `prompt`,
`research_type`, `domain_detail`, `logic`, `sanity_check`, `drive_url`.

### 4.2 Research type & IAT — normalization

`normalize_research_type()` collapses any spelling to a 3-letter code using
`RESEARCH_TYPE_ALIASES` (e.g. `"Failure-Sensitive Prompt"`, `"FAILURE SENSITIVE
PROMPT"`, `"fsp"` all → `"FSP"`).

Then `_IAT_MAP` assigns the tier:
```
CRP → IAT-2     RCP → IAT-1     SCP → IAT-3
LDP → IAT-1     FSP → IAT-3
```
If the type is unrecognized, both become `""` (empty), and the task still runs
with universal handling.

### 4.3 Output format detection — `detect_output_formats()`

Lives in `file_generators/detector.py`. It reads the **prompt text** and decides
whether the deliverable must be a generated file. It returns a sorted subset of
`["docx", "pptx", "xlsx"]`.

It is deliberately strict to avoid false positives:
- **xlsx / pptx**: requires an *output verb* (present/output/create/produce/
  generate/deliver/build) in the **same sentence** as the keyword "excel" /
  "powerpoint". The sentence boundary is enforced with `[^.!?\n]{0,120}`.
- **docx**: simpler — the phrase "word doc" or "word document" anywhere is enough.

Examples (from the docstring, validated 0 FP / 0 FN on the benchmark):
```python
detect_output_formats("Present the final results in Excel format.")  # → ["xlsx"]
detect_output_formats("Produce a Word document deliverable.")        # → ["docx"]
detect_output_formats("Use Data Dump 1.xlsx to extract the data.")   # → []   (input file, not output)
detect_output_formats("Use FlipAmaz Ordering Model.pptx to ...")     # → []   (input file, not output)
```

> **Why this matters:** a non-empty `output_formats` flips the entire pipeline
> into "file output mode," which changes how *every* adapter behaves (Dimension B).

---

## 5. Stage 2 — Building a PromptPackage

`row_to_package()` assembles the `PromptPackage`:
- `research_type` = normalized code, `iat_type` = mapped tier.
- `solution_steps` = the `logic` column split on newlines (lines >10 chars kept).
- `output_formats` = `detect_output_formats(prompt)`.
- `file_paths`:
  - If `--resolve-files` was used and resolution returned local paths → those.
  - Else if there's a `drive_url` → the **raw URL is kept as a placeholder** in
    `file_paths` (so the dispatcher can later detect "this needed files but got 0").

**Filtering happens before file resolution** (so you don't waste GDrive
downloads): `--task-ids`, `--filter-type`, `--filter-sme`, `--max-rows` are all
applied to the row list first.

---

## 6. Stage 3 — Resolving files (`file_resolver.py`)

Only runs if `--resolve-files` is passed. For each reference,
`parse_gdrive_reference()` classifies it:

| Input | Parsed as | Action |
|-------|-----------|--------|
| `/local/path/file.pdf` (exists, or starts with `/` `./`) | `None` (local) | pass through unchanged |
| `gdrive://ABC123` | file | download single file |
| `https://drive.google.com/file/d/ABC/view` | file | download single file |
| `https://drive.google.com/open?id=ABC` / `uc?id=ABC` | file | download single file |
| `https://drive.google.com/drive/folders/ABC` (incl. `/u/0/`) | folder | download **all** files, recurse subfolders |
| `https://docs.google.com/spreadsheets/d/ABC` | workspace_doc | export → `.xlsx` |
| `https://docs.google.com/document/d/ABC` | workspace_doc | export → `.docx` |
| `https://docs.google.com/presentation/d/ABC` | workspace_doc | export → `.pptx` |

Auth (`GDriveClient`): tries, in order — explicit creds →
`GOOGLE_SERVICE_ACCOUNT_KEY` → `GOOGLE_APPLICATION_CREDENTIALS` →
`GOOGLE_CLIENT_SECRETS` (interactive OAuth). Read-only Drive scope.

Files land in the staging dir. Folders are downloaded with a name prefix so
nested files don't collide (`subfolder__file.pdf`).

**Scenario: GDrive times out / folder is empty.** Resolution returns 0 files
(see `check.txt` lines 6-9). The package's `file_paths` stays as the raw URL, and
in Stage 4 the dispatcher **skips the task** with a warning
("GDrive ref resolved to 0 files") rather than running it without its corpus.

---

## 7. Stage 4 — The dispatcher (`task_dispatcher.py`)

`dispatch_packages()` loops over packages **one at a time** (sequential, to
control cost/rate limits). For each, it calls `TaskDispatcher.dispatch(package,
config)`. Within a task, agents run **concurrently**.

### 7.1 Validate — `validate_package()`
- **Hard errors (raise `PackageValidationError`, block the task):** no `task_id`;
  prompt shorter than 20 chars; invalid `research_type`; invalid `iat_type`.
- **Warnings (logged, non-blocking):** missing files; no files at all; no
  research_type; no iat_type; empty solution_steps; empty lazy_ai_prediction.

### 7.2 Eligibility — `check_agent_eligibility()`
Returns `(eligible, reason)`. Nothing is actually blocked here, but caveats are
logged:
- Unknown agent name → ineligible (the only true block).
- `perplexity` + IAT-1 → eligible **with a structural caveat** ("cannot disable
  web search").
- `openai`/`gemini` + corpus > 3 MB → eligible with a note suggesting MCP
  (Tier 2) for fairness.

### 7.3 PromptPackage → ResearchTask — `to_research_task()`
- `web_search_enabled = not is_closed` (i.e. False only for IAT-1).
- Copies prompt, files, type/iat/domain, `output_formats`.

### 7.4 IAT enforcement switch — **important default**
```python
if not config.enforce_iat:          # default is False
    task.web_search_enabled = True   # force web ON for everyone
```
So **by default, IAT-1's "no web" rule is NOT applied** — every agent gets web
search. You must set `enforce_iat=True` for benchmark-valid closed-corpus runs.
This is intentional (collect responses first), but easy to forget.

### 7.5 Output directory setup
Only if `output_formats` is non-empty **and** not a dry run:
- Base dir = `config.output_files_base_dir` (csv_loader sets it to
  `<results_dir>/files`), else a temp dir `/tmp/dra_output_files/`.
- Per-task subdir: `<base>/<task_id>/`. This is `task.output_files_dir`.

### 7.6 Fan out — concurrency
- `asyncio.Semaphore(max_concurrent=4)`.
- `asyncio.gather(..., return_exceptions=True)` runs every eligible agent
  concurrently. One agent throwing never kills the others.

### 7.7 Per-agent execution — `_run_agent()` (Pass@K)
For each agent:
- Resolve the timeout: `agent_overrides` > per-agent default > task default.
  Defaults: gemini 3600s, openai 3600s, claude 900s, perplexity 300s.
  Dispatcher adds +60s as a hard outer bound.
- Loop `pass_num` from 1..K (passes run **sequentially** within an agent).
- Each pass gets a unique id: `{task_id}_{agent}_p{pass_num}`.
- Wrapped in `_run_with_heartbeat()`:
  - Runs `adapter.run(pass_task)` under `asyncio.wait_for(timeout)`.
  - A background coroutine logs "Still running... (Nm elapsed)" every 5 minutes.
- **Three outcomes per pass:**
  1. Success → append the `AgentResult`.
  2. `asyncio.TimeoutError` → synthesize a failed `AgentResult`
     (`model="timeout"`, `forced_stop=True`).
  3. Any other `Exception` → failed `AgentResult` (`model="error"`).
  - If `config.fail_fast`, break the pass loop on first failure.

### 7.8 Adapter caching — `_get_adapter()`
Adapters are cached by `(agent_name, dry_run)` so connection/auth state is reused
across passes. MCP URL is wired only into OpenAI; Gemini gets `use_context_cache`.

---

## 8. Stage 5 — The four agent adapters

All four implement `async run(task) -> AgentResult` and support `dry_run=True`
(returns a mock report instantly, no API calls, no cost).

### 8.1 Claude — the transparent loop (`claude_adapter.py`)

Claude is the only agent whose research loop **we** drive, so we see everything.

```
build messages (prompt + files)  →  while True:
    response = stream Claude(messages, tools)
    log every server_tool_use (web_search / code_execution)  → tool_call_log
    switch on response.stop_reason:
        end_turn    → extract report, capture any output files, DONE
        pause_turn  → append response as-is, continue (server hit its own cap)
        max_tokens  → if unresolved tool calls: forced_stop; else "continue"
        tool_use    → append, continue
    guards each iteration: max_iterations(25) / max_cost($15) / timeout(900s)
        any exceeded → forced_stop = True, break
if forced_stop: ask Claude to synthesize a final report from what it has
```

**Tools chosen by scenario (`_build_tools`):**
- `web_search` added **only if** `task.web_search_enabled` (this is the *hard*
  IAT-1 enforcement — omit the tool entirely).
- `code_execution` added **only if** `output_formats` is set.

**File input handling (`_build_messages`):**
- *File-output tasks*: files are uploaded to the Files API first
  (`_upload_files`), then attached by type — PDFs/txt/md/html/xml as `document`
  blocks, images as `image` blocks, everything else as `container_upload` (lands
  at `/input/<name>` in the sandbox). `.xlsb` is converted to TSV text first.
- *Text-only tasks*: files are read and inlined as text blocks (truncated at
  50K chars each).

**File output capture (`_capture_files`):** scans
`bash_code_execution_tool_result` blocks for `file_id`s, downloads each via the
Files API into `output_files_dir` as `claude_{task_id}.{ext}`.

**Validation:** if a required format wasn't produced → `output_file_errors[fmt]`,
`completed=False`.

**Cost:** O(N²)-ish — every iteration re-sends the growing message list.

### 8.2 OpenAI — poll-based black box (`openai_adapter.py`)

```
upload files (Files API)  →  responses.create(model, input, tools)
   →  poll responses.retrieve(id) every 10s→30s
        status queued/in_progress → keep polling (stale guard: 15 min no change)
        status completed → extract report + citations + container files
        status failed/incomplete → forced_stop
```

**Two file modes (the key branch):**
- **File-output mode** (`output_formats` set): files go into the
  `code_interpreter` tool's **container** (`container.file_ids`), and the user
  message is **text only** (mixing `input_file` blocks + code_interpreter causes
  `invalid_file` errors). A strongly-worded "MANDATORY FINAL STEP: produce
  `output.{fmt}`" instruction is appended. Generated files are pulled afterward
  via the **Containers API** (`/containers/{id}/files`), keeping only
  `source == "assistant"` files, deduped by `container_id`. Saved as
  `openai_{task_id}.{ext}`.
- **Text-only mode**: files are attached as `input_file` blocks; no
  code_interpreter.

**Web search:** `web_search_preview` tool added only if `web_search_enabled`
(soft IAT-1 — model may still leak).
**MCP:** if `mcp_server_url` set, an `mcp` tool is attached (Tier 2 corpus).

### 8.3 Gemini — multi-pass + local execution (`gemini_adapter.py`)

The most complex adapter by far. Uses the Interactions API.

**File prep (`_prepare_files`):**
- PDFs → uploaded natively via Files API. Large PDFs (>~400K chars estimate) go
  through `_pdf_smart_extract()` which **drops statutory boilerplate** (corporate
  governance, CSR, secretarial audit) and **keeps financial content** (MD&A,
  balance sheet, notes) — keyword/page-range based, no hardcoded pages.
- XLSX/DOCX/PPTX/CSV → converted to **inline text** (xlsx via LibreOffice
  headless to evaluate formulas, then openpyxl). Originals stay on disk for code
  execution.
- **Hard limit:** total inline text > 3,000,000 chars → raises `ValueError`
  *before any API call* (never truncates).

**Multi-pass research loop (Stage 1):**
- Text-only task → exactly **1 pass**.
- File-output task → up to `max_research_passes` (default 4), BUT:
  - Pass 1 always does the full research → `pass1_report`.
  - If Pass 1 already contains a code block → **skip all remaining passes**.
  - Otherwise a dedicated code-generation pass runs, with the full `pass1_report`
    pasted into the prompt (because `previous_interaction_id` does **not**
    reliably carry context).
  - Budget/timeout checked between passes.
- `response_text` is **always** `pass1_report` with any code block stripped
  (`_strip_file_output_block`) — the saved report never contains code.

**Polling (`_poll`):** every 10s; `completed`→return, `failed`→raise,
stuck >30 min → `StaleInteractionError`.

**Web search:** `google_search` excluded for IAT-1 via instruction (soft).

(Stage 2 file execution is detailed in [Section 9](#9-stage-6--file-output-generation-the-hard-part).)

### 8.4 Perplexity — synchronous, search-native (`perplexity_adapter.py`)

```
extract every file to TEXT (no upload API)  →  inline into one big prompt
   (budget ~90K tokens of chars split across files)
   →  chat.completions.create(...)  (blocks 1-3 min, single call)
   →  read report + top-level citations array
   →  if IAT-1: audit citations for external URLs → flag violations
```

- **No file upload**, **no MCP**, **128K context** (smallest), **cannot disable
  web search.**
- IAT-1 handling: prepend a "DO NOT search externally" instruction, then
  `check_iat1_compliance()` checks whether any citation URL isn't one of the
  provided file names; non-compliance is recorded in `error`.
- Always `iterations=1`.

---

## 9. Stage 6 — File output generation (the hard part)

This only happens when `output_formats` is non-empty. Each agent does it
differently:

| Agent | Who writes the code | Where it runs | Self-correction |
|-------|--------------------|--------------|------------------|
| Claude | Claude | Anthropic sandbox | n/a (native) |
| OpenAI | o3 | OpenAI container | n/a (native) |
| Gemini | Gemini (text) | **Our local machine** (subprocess) | **yes, up to 3 fixes** |
| Perplexity | — (no file output support) | — | — |

### Gemini Stage 2 in detail (`_run_code_block`)

This is unique: Gemini only returns *text*, so we must extract and **run the
Python ourselves**.

1. **Extract the code.** Priority: sentinel block between
   `# GEMINI_FILE_OUTPUT_START` and `# GEMINI_FILE_OUTPUT_END`
   (`_extract_marked_code_block`). Fallback: last fenced block that imports the
   right library (`_extract_last_file_code_block`).
2. **Sanitize (`_sanitize_code`).** Removes Gemini's `[cite: N]` markers that
   break Python (3 regex patterns), and fixes two recurring python-docx mistakes
   (`.rows.cells` → `.rows[0].cells`; `_cells.text` → `_cells[0].text`).
3. **Staging dir = where the input files live** (so the code can read them by
   their real filenames). A `chdir` header is prepended so the script runs there.
4. **Execute** via `subprocess.run([python, script])` with a 120s timeout. The
   code must save `output.{fmt}`.
5. **On success:** move `output.{fmt}` → `output_files_dir/gemini_{task_id}.{fmt}`.
6. **On failure (rc≠0):** send Gemini the **stderr + the broken code**, ask for a
   fix (sentinel-wrapped), re-extract, re-run. Each fix chains off the previous
   fix interaction. **Up to 3 attempts total.** Fix-call costs are added to
   `total_cost_usd`.
7. **Special case — Pass 1 produced no code at all:** attempt 1 skips execution
   and instead asks Gemini to write the code *from the research report* (pasted
   inline).
8. The staging script is deleted in a `finally` (preserved on disk only while
   debugging per the README note `_gemini_gen_{task_id}.py`).

**Validation across all agents:** for every requested format not found in
`output_files`, an `output_file_errors[fmt]` entry is added and `completed=False`.

---

## 10. Stage 7 — Aggregation & storage

### 10.1 Aggregation (`TaskDispatcher.dispatch`)
After `gather`, for each agent:
- Exception result → `agents_failed`, error recorded, empty results list.
- Normal result → stored under `agent_results[agent]`; if **any** pass has
  `completed and not error` → `agents_succeeded`, else `agents_failed`.
- Costs summed into `total_cost_usd`.

Produces a `DispatchResult` with: the original package, the config,
`agent_results` (`{agent: [AgentResult, ...]}`), the attempted/succeeded/failed
lists, `agent_errors`, totals, and timestamps.

### 10.2 Serialization (`dispatch_result_to_dict`)
Converts to a JSON-safe dict. Notably it includes the **full** `response_text`,
`citations`, `output_files`, plus derived counts (`response_length`,
`citations_count`, `tool_calls_count`).

### 10.3 Two save paths (both optional, often both on)
- `--output ./dir` → one file per task: `dir/{task_id}.json`
  (`save_dispatch_result`).
- `--results-dir ./res` → goes through `ResultsStore` (`results_store.py`):
  - Layout: `res/index.json`, `res/tasks/{task_id}.json`,
    `res/scores/{task_id}_scores.json`.
  - `csv_loader` also attaches `sme_metadata` (name, email, drive_url, etc.).
  - **Merge-on-rerun:** if `tasks/{task_id}.json` exists, `_merge_dispatch_results`
    merges: agent_results union (new wins per agent), **costs summed**, duration
    max, earliest dispatch / latest completion, package/config field-level merge
    (non-None new values win). So re-running just `gemini` won't wipe an earlier
    `claude` result.
  - A lightweight `index.json` holds queryable metadata for fast filtering.

### 10.4 The final on-disk JSON (per task)
```jsonc
{
  "task_id": "...",
  "dispatched_at": "...", "completed_at": "...",
  "total_cost_usd": 2.23, "total_duration_sec": 960.5,
  "agents_attempted": ["claude","openai","gemini","perplexity"],
  "agents_succeeded": ["openai","perplexity"],
  "agents_failed":    ["claude","gemini"],
  "agent_errors": { "claude": "pip install anthropic ...", "gemini": "Dispatcher timeout ..." },
  "config":  { "agents":[...], "passes_per_agent":1, "dry_run":false, ... },
  "package": { "prompt":"...", "research_type":"FSP", "iat_type":"IAT-3", "file_names":[...], "output_formats":["xlsx"] },
  "agent_results": {
    "openai": [ { "response_text":"...", "citations":[...], "output_files":[...], "total_cost_usd":2.19, ... } ],
    ...
  }
}
```

---

## 11. Stage 8 — Post-processing utilities

These read the saved JSON and reshape it. Point them at `results/<run>/tasks/`.

| Script | Purpose | Output |
|--------|---------|--------|
| `merge_json_results.py` | Merge partial/split runs from multiple dirs | `merged_output/json/*.json` |
| `extract_by_model.py` | Keep only one model's passes | `extracted_{model}/*.json` |
| `export_completed_runs.py` | Keep only `completed && !forced_stop && !dry_run`, write readable text | `exported_runs/*.txt` |
| `export_results_csv.py` | One row per agent pass (PASS + FAIL), consolidated | `exported_csv/results_consolidated.csv` |

---

## 12. The MCP servers (optional side-channels)

These are **not** in the default path; they support Tier 2 (large corpus) access
and external scoring tools.

### Corpus server (`mcp_servers/corpus_server.py` + `corpus_tools.py`), port 9400
Serves files **to** agents on demand instead of stuffing everything in context.
`Corpus` loads a directory, extracts text once, caches it, and exposes:
- `list_documents` — metadata for every file
- `search` — keyword TF search returning ranked snippets
- `fetch` — full extracted text of one doc by id

Wired to OpenAI (native MCP) and Claude (custom tools) when
`DispatchConfig.mcp_server_url` is set. Gemini can't use it.

### Results server (`mcp_servers/results_server.py` + `results_store.py`), port 9401
Serves results **from** storage to a scorer/SME UI:
- `store_result`, `store_scores`, `get_result`, `get_scores`, `list_results`,
  `query_results`, `get_comparison`, `get_stats`.

---

## 13. End-to-end worked example

Let's trace one realistic row.

**Input CSV row:**
```
Prompt ID: 28115.0B
POC Name:  Anagha Patne
Category:  Failure-Sensitive Prompt
Domain:    Supply Chain
Prompts:   "Using the attached order book, compute the optimal reorder
            quantities and present the final results in an Excel sheet."
Drive:     https://drive.google.com/drive/folders/1cUz...MXF
```

**Stage 1 — parse:**
- `task_id` → `"28115B"` (float-normalized).
- `research_type` → `"FSP"`; `iat_type` → `"IAT-3"`.
- `detect_output_formats(prompt)` → matches "present ... excel" → `["xlsx"]`.
  → **file-output mode is ON.**

**Stage 2 — package:** `PromptPackage(task_id="28115B", output_formats=["xlsx"],
file_paths=["https://drive.google.com/drive/folders/1cUz...MXF"], ...)`.

**Stage 3 — resolve files (`--resolve-files`):** folder is downloaded to
`/tmp/eval_files/`. Say it yields `order_book.xlsx`. (If it had timed out → 0
files → task skipped.)

**Stage 4 — dispatch:**
- validate → OK. eligibility → all 4 OK (FSP/IAT-3 has no caveats).
- `web_search_enabled = True` (IAT-3, and default anyway).
- output dir created: `./results/files/28115B/`.
- 4 agents launched concurrently, each Pass@1.

**Stage 5/6 — per agent (file-output mode):**
- **Claude:** uploads `order_book.xlsx` as `container_upload`, enables
  `web_search` + `code_execution`, researches, writes Python in the sandbox,
  generates the xlsx → downloaded to `results/files/28115B/claude_28115B.xlsx`.
- **OpenAI:** xlsx goes into the code_interpreter container; o3 researches +
  runs code; file pulled via Containers API →
  `openai_28115B.xlsx`.
- **Gemini:** xlsx converted to inline TSV (LibreOffice-evaluated); Pass 1
  researches and appends sentinel-wrapped Python; Stage 2 runs that Python
  locally against the real `order_book.xlsx`; if it crashes (e.g. wrong column
  name), up to 3 fix round-trips; on success → `gemini_28115B.xlsx`.
- **Perplexity:** inlines the xlsx as text, returns a report — **but produces no
  file** (no file-output capability), so `output_file_errors["xlsx"]` and
  `completed=False` for that agent.

**Stage 7 — store:** `results/tasks/28115B.json` written with all four agents'
reports, the three generated xlsx paths, costs, and `sme_metadata`. `index.json`
updated.

**Stage 8 — later:** `export_results_csv.py` turns it into one row per agent for
the scorer.

---

## 14. Every scenario / decision table

### A. Web search (Dimension A)
| Condition | Result |
|-----------|--------|
| `enforce_iat=False` (default) | web ON for all agents, all tiers |
| `enforce_iat=True` + IAT-1 | Claude: tool omitted (hard). OpenAI/Gemini: tool omitted but may leak (soft). Perplexity: can't disable → instruction + citation audit |
| `enforce_iat=True` + IAT-2/IAT-3 | web ON |

### B. File output (Dimension B)
| Condition | Effect |
|-----------|--------|
| `output_formats == []` | text-only mode; Gemini runs exactly 1 pass; no code execution anywhere; files attached as input only |
| `output_formats == ["xlsx"/"docx"/"pptx"]` | file-output mode; output dir created; Claude/OpenAI generate natively; Gemini does local exec + self-correct; Perplexity will fail the file requirement |

### C. Dry run
| Condition | Effect |
|-----------|--------|
| `--dry-run` (default in CLIs until `--live`) | every adapter returns a mock report instantly, `cost=0`, no API calls, **no output dir created** |

### D. File reference types
See the table in [Section 6](#6-stage-3--resolving-files-file_resolverpy).

### E. Pass@K
| Condition | Effect |
|-----------|--------|
| `--passes 1` (default) | one pass per agent |
| `--passes K` | K sequential passes per agent; each is an independent sample with id `..._pN` |

### F. Per-agent timeouts
gemini 3600s · openai 3600s · claude 900s · perplexity 300s (dispatcher +60s
outer bound; `--timeout` or `agent_overrides` can override).

---

## 15. Failure modes & what happens

| Failure | Where | Behavior |
|---------|-------|----------|
| Missing API key / SDK not installed | adapter constructor | that agent's pass → failed `AgentResult`; others continue |
| Prompt < 20 chars / bad type | `validate_package` | whole task blocked (`PackageValidationError`) |
| GDrive timeout / empty folder | `file_resolver` / dispatcher | 0 files → task **skipped** with warning |
| Agent exceeds time | `_run_with_heartbeat` | `TimeoutError` → failed result (`model="timeout"`) |
| Claude hits budget/iter cap | claude loop | `forced_stop=True`, synthesizes a partial report from context |
| OpenAI stuck in a status 15 min | `_poll_until_complete` | raises → failed result |
| Gemini stuck 30 min | `_poll` | `StaleInteractionError` → failed result |
| Gemini inline files > 3M chars | `_prepare_files` | `ValueError` before any API call (explicit, no truncation) |
| Gemini code fails 3× | `_run_code_block` | `output_file_errors[fmt]`, `completed=False`, final stderr in `error` |
| Required file not produced | each adapter's validation | `output_file_errors[fmt]`, `completed=False` |
| Perplexity on file-output task | perplexity adapter | report only; file requirement fails |
| OpenAI content filter (`invalid_prompt` 400) | openai adapter | error result; README says record as policy rejection, exclude from pass rate |
| Re-running a task | `ResultsStore` | merges, doesn't overwrite; costs summed |

---

### One-paragraph summary

The pipeline reads a flexible CSV into clean rows (`csv_loader`), normalizes the
research type into an IAT tier, auto-detects whether a file deliverable is
required (`detect_output_formats`), optionally downloads Google Drive corpora
(`file_resolver`), and packages each row as a `PromptPackage`. The
`TaskDispatcher` validates it, converts it to an agent-agnostic `ResearchTask`,
decides whether web search is on (`enforce_iat`) and whether a file output dir is
needed, then runs all four adapters concurrently (Pass@K sequential per agent,
each with its own timeout and heartbeat). Each adapter — Claude (transparent
loop), OpenAI (poll), Gemini (multi-pass + local code execution with
self-correction), Perplexity (synchronous, search-native) — returns a uniform
`AgentResult` with a report, citations, costs, and any generated xlsx/docx/pptx.
The dispatcher aggregates these into a `DispatchResult`, which is saved as JSON
(merge-on-rerun via `ResultsStore`), ready for an external scorer or the
post-processing utilities.
