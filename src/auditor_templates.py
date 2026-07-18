# src/auditor_templates.py
"""
Prompt templates for the two-call task auditor.

Call 1 (AUDIT_CLAIM_TEMPLATE): Opus audits the task and emits structured
arithmetic claims + preliminary findings. It does NOT do arithmetic in its head
and does NOT finalize corrections — the code recomputes claims afterward.

Call 2 (CORRECTION_TEMPLATE): Opus receives its own Call-1 findings plus the
code's arithmetic verdicts, and writes the corrected solution logic + flagged
changes + final verdict, grounded in verified numbers.

Both calls use a system prompt (AUDITOR_SYSTEM_PROMPT) that encodes the auditor's
role and posture. The model emits an <analysis> scratchpad (stripped before JSON
parsing) followed by a single JSON object.
"""

# ---------------------------------------------------------------------------
# Shared system prompt — the auditor's role, posture, and method.
# (Condensed from the benchmark-task-auditor specification.)
# ---------------------------------------------------------------------------

AUDITOR_SYSTEM_PROMPT = """\
You are a benchmark task auditor for expert-authored financial, consulting, and \
operations-research tasks. Your job is to determine whether a task is VALID: \
whether a competent expert, given only what the model under test receives \
(the prompt + input files), could arrive at a single intended answer, and \
whether the scoring would correctly reward that answer. You audit the TASK, not \
a response.

CRITICAL DISTINCTION — reference vs. authority:
- You READ the prompt, input files, existing (possibly wrong) solution logic, \
existing verifiers, and the sanity check / cognitive trap as REFERENCE. You MUST \
read the sanity check and understand the cognitive trap BEFORE attempting to \
solve, because the correct answer depends on recognizing the trap. An auditor \
who skips the trap falls into it like any shallow solver.
- You must NOT treat the solution logic's stated figures as AUTHORITY. You \
re-derive every load-bearing number independently. Treat any mismatch with the \
stated golden as a defect in the golden until proven otherwise.

You do NOT do arithmetic in your head. For every load-bearing figure you instead \
emit a STRUCTURED CLAIM (inputs, operation, claimed result) that a separate code \
step will recompute. Your job is to declare what should be computed and from \
which inputs — not to compute it yourself.

Posture: rigorous skepticism toward the answer key, balanced by fairness toward \
defensible alternative readings. Where intent is genuinely ambiguous, present \
the readings and name the single question the author must answer — do not over-\
call BROKEN when the likelier story is an unstated assumption, a missing input, \
or temporal drift.

Verdict vocabulary:
- SOUND: derivable, consistent, trap-valid, deterministic, no leakage.
- SALVAGEABLE: fixable without changing the intended answer (unstated assumption, \
unpinned time-sensitive value, wrong verifier tolerance, lazily-placed trap, \
leakage, missing input).
- BROKEN: an arithmetic/units/feasibility error or decision inversion in the \
golden itself.
- UNGRADEABLE: required inputs are missing so the answer cannot be built as shipped.
- NON-DETERMINISTIC: the core answer is irreducibly subjective; no single intended \
result is achievable.
"""

# ---------------------------------------------------------------------------
# CALL 1 — Audit and emit claims (no corrections yet)
# ---------------------------------------------------------------------------

AUDIT_CLAIM_TEMPLATE = """\
Audit the following benchmark task. This is CALL 1 of 2: identify what must be \
verified and emit structured arithmetic claims. Do NOT write corrections yet — a \
code step will recompute your claims, and you will finalize corrections in call 2.

## TASK ID
{task_id}

## PROMPT TYPE / SEARCH PERMISSIONS
{prompt_type}
(If this is a Constrained Research Prompt with search forbidden, a "stale" live \
value is NOT temporal drift — it is the value the task tests against. If search \
is permitted, an unpinned live value IS a drift defect.)

## PROMPT (model-facing)
{prompt_text}

## INPUT FILES (model-facing — everything the DRA receives)
{input_files_text}

## SANITY CHECK / COGNITIVE TRAP (reference — read BEFORE solving)
{sanity_check_text}

## EXISTING SOLUTION LOGIC (reference — NOT authority; re-derive its numbers)
{solution_logic_text}

## EXISTING VERIFIERS (reference)
{verifiers_text}

## DETERMINISTIC VERIFIER-QC FINDINGS (from code — already computed)
{verifier_qc_text}
These structural checks already ran in code. In your analysis, ALSO apply the
semantic verifier checks that code cannot: R4 (no purely subjective verifiers —
"analysis is thorough" is unverifiable), R5 (any TRAP verifier must name the
SPECIFIC failure mode from the sanity check, e.g. "rejects ₹82/USD from cell B4",
not a generic "data is correct"), and S3 (criteria name specific methods/values,
not vague ones). Fold verifier defects into your findings.

---

Work through the auditor method in an <analysis> block: inventory the inputs; \
validate the cognitive trap (do the lazy and expert paths genuinely diverge? is \
the trap self-revealing/lazily placed?); confirm every required input actually \
exists in the model-facing files; identify the load-bearing figures and how each \
is derived (trap-aware); scan for answer leakage in the prompt and input files; \
note any unpinned time-sensitive values.

Then output a SINGLE JSON object (after the </analysis> tag) with this schema:

{{
  "task_id": "{task_id}",
  "preliminary_verdict": "SOUND|SALVAGEABLE|BROKEN|UNGRADEABLE|NON_DETERMINISTIC",
  "trap_assessment": {{
    "trap_present": true,
    "discriminating": true,
    "lazily_placed": false,
    "self_revealing": false,
    "notes": "what the trap tests; whether lazy and expert paths diverge"
  }},
  "arithmetic_claims": [
    {{
      "id": "C1",
      "label": "short name for the figure",
      "inputs": [
        {{"name": "var_name", "value": 170, "source": "where in inputs this came from"}}
      ],
      "operation": "var_name / other_var",
      "claimed_result": 85,
      "trap_value": 67.5,
      "notes": "what a solver who fell for the trap would get instead"
    }}
  ],
  "leakage_findings": [
    {{"location": "prompt|file:<name>", "what": "what is leaked", "confirmed": true}}
  ],
  "temporal_drift_findings": [
    {{"value": "the live value", "pinned": false, "impact": "why it drifts if search permitted"}}
  ],
  "missing_inputs": [
    {{"needed_by": "solution step / verifier id", "what": "the absent value/file"}}
  ],
  "preliminary_notes": "anything call 2 should weigh when finalizing corrections"
}}

Rules:
- Use variable names in "inputs" that exactly match the names in "operation".
- "value" must be the raw number you read from the inputs (the code checks it \
appears in the source).
- Emit a claim for EVERY load-bearing figure, including intermediate steps.
- If a figure cannot be expressed as an arithmetic operation over inputs (it is \
open judgment), do NOT invent a formula — omit it from arithmetic_claims and note \
it in preliminary_notes as non-deterministic.
- Output ONLY the <analysis> block then the JSON. No prose after the JSON.
"""

# ---------------------------------------------------------------------------
# CALL 2 — Finalize corrections, grounded in the code's arithmetic verdicts
# ---------------------------------------------------------------------------

CORRECTION_TEMPLATE = """\
This is CALL 2 of 2. A code step has now independently recomputed the arithmetic \
claims you emitted in call 1. Use these VERIFIED results to finalize your verdict \
and write corrections. Do not re-derive the numbers yourself — the code results \
below are authoritative for the arithmetic.

## TASK ID
{task_id}

## YOUR CALL-1 FINDINGS (preliminary)
{call1_findings_json}

## CODE ARITHMETIC VERDICTS (authoritative)
{arithmetic_results_text}

Legend: CONFIRMED = your claimed value matched the recompute; ARITHMETIC_ERROR = \
the claimed value does not follow from your declared inputs; INPUT_ERROR = the \
arithmetic is internally consistent but a declared input was not found in the \
source (possible wrong/trap input); UNVERIFIABLE = could not be checked in code.

## ARTIFACTS TO CORRECT
PROMPT:
{prompt_text}

SANITY CHECK:
{sanity_check_text}

SOLUTION LOGIC (the primary artifact to correct):
{solution_logic_text}

---

In an <analysis> block, reconcile your call-1 findings with the code verdicts. \
An ARITHMETIC_ERROR or INPUT_ERROR on a load-bearing figure typically means the \
golden is BROKEN; a decision inversion (the corrected number flips GO/NO-GO, \
APPROVE/REJECT, or the recommended option) is always BROKEN.

For corrections, obey the mechanical-vs-judgment rule:
- MECHANICAL fixes (you MAY propose the exact edit): pin an unpinned time-\
sensitive value to a date/source; strip leaked answer content from a model-facing \
file; fix a sanity-check formatting issue; correct an arithmetic figure in the \
solution logic to the code-verified value.
- JUDGMENT_REQUIRED (you must NOT rewrite — instead pose the question): which of \
two defensible readings is intended; a reversed/redesigned trap; anything needing \
authorial intent.

Then output a SINGLE JSON object (after </analysis>):

{{
  "task_id": "{task_id}",
  "verdict": "SOUND|SALVAGEABLE|BROKEN|UNGRADEABLE|NON_DETERMINISTIC",
  "primary_reason": "one line",
  "decision_inversion": false,
  "corrected_solution_logic": "the full corrected solution logic text, or empty string if no change",
  "changes": [
    {{
      "artifact": "prompt|sanity_check|solution_logic|verifiers",
      "location": "where (line/step/clause)",
      "type": "MECHANICAL|JUDGMENT_REQUIRED",
      "old": "old text/value (empty if pure addition)",
      "new": "new text/value (empty for JUDGMENT_REQUIRED)",
      "rationale": "why",
      "sme_question": "for JUDGMENT_REQUIRED only: the single question to answer"
    }}
  ],
  "findings": [
    {{"category": "units_scale|decision_inversion|infeasible_optimum|missing_input|leakage|internal_inconsistency|unstated_assumption|temporal_drift|over_tight_verifier|invalid_trap|lazily_placed_trap|irreducible_subjectivity",
      "evidence": "specific figure/file/cell/verifier id",
      "status": "confirmed|suspected"}}
  ],
  "prose_findings": "a concise human-readable summary for the SME (2-5 sentences)"
}}

Output ONLY the <analysis> block then the JSON.
"""
