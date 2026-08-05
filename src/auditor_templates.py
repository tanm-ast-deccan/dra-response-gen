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

Finding categories — use these exact meanings, not what the name suggests:
- units_scale: a figure is right in magnitude but wrong in unit or scale (a ratio
  written as a percentage, crore against rupees, minutes against hours).
- decision_inversion: the golden's stated decision is the opposite of what its own
  corrected numbers support. Always BROKEN.
- infeasible_optimum: the stated optimum violates a constraint the task imposes,
  or a better feasible answer exists.
- missing_input: a value the solution needs that appears in no model-facing file.
- leakage: the answer, the trap, or the method is disclosed in material the model
  can see. Includes phrases that signal a trap exists, not only the answer itself.
- internal_inconsistency: two parts of the golden assert incompatible things (the
  same quantity at two values; a total that does not match its parts).
- unstated_assumption: the intended answer needs an assumption the prompt never
  states, so a competent solver could defensibly answer otherwise.
- temporal_drift: a live value is not pinned to a date or source, so the golden
  decays as the world moves.
- over_tight_verifier: a verifier's tolerance is narrower than the task's own
  determinism, so a correct answer fails on rounding.
- invalid_trap: the trap does not discriminate — the lazy and expert paths reach
  the same answer, so it tests nothing.
- lazily_placed_trap: the trap is discoverable without the intended reasoning
  (flagged in a filename, a comment, or an obvious outlier).
- irreducible_subjectivity: the core answer has no single defensible result, so
  no verifier can be written for it. Drives NON_DETERMINISTIC.

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
  "corrected_claims": [
    {{
      "id": "C1",
      "label": "short name for the figure",
      "inputs": [
        {{"name": "var_name", "value": 170, "source": "where this came from",
          "from_claim": "C0 or null", "source_type": "file|claim|metadata"}}
      ],
      "operation": "var_name / other_var",
      "claimed_result": 85,
      "trap_value": 67.5,
      "solution_step": "Step 2"
    }}
  ],
  "judgment_steps": [
    {{
      "id": "J1",
      "question": "the decision or ruling this step makes",
      "consumes": ["C6", "C4"],
      "ruling": "what the corrected solution logic concludes",
      "basis": "the quoted text this rests on",
      "solution_step": "Step 6"
    }}
  ],
  "corrected_prompt": "the full corrected prompt text, or empty string if no change",
  "corrected_sanity_check": "the full corrected sanity check text, or empty string if no change",
  "corrected_verifiers": "the full corrected verifier block as 'V1: ...' lines, one per line, or empty string if no change",
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

Rules for "corrected_claims" — this is the derivation, and it is consumed by code:
- Re-emit EVERY claim, not only the ones you changed. The list replaces your
  call-1 claims wholesale and is recomputed from scratch, so an omitted claim
  disappears from the derivation.
- Apply your corrections to them. Where a code verdict said ARITHMETIC_ERROR,
  either fix the operation or fix the claimed_result so the two agree — do not
  re-emit a claim you know does not reconcile.
- "from_claim" is the ID OF THE CLAIM WHOSE OUTPUT IS THIS INPUT, or null when
  the value was read from a source file. This is what turns a flat list into an
  ordered derivation, so it is the single most important field here. If input
  "peak_total" is the result of C1, write "from_claim": "C1".
- Never set from_claim on a value you read from an input file. A raw file value
  has no parent step, and inventing one corrupts the dependency graph.
- "solution_step" groups a claim under a step of the corrected solution logic.
  Omit it if you cannot place the claim.
- NEVER INLINE A VALUE YOU COMPUTED. If an input is a figure you worked out
  rather than read, it must be its OWN claim, with this input pointing at it via
  from_claim. Writing "peak_arrivals = 170" as a literal input hides the sum of
  eight arrival buckets that produced it: the derivation loses a step, and code
  cannot find 170 in any file so it reads as an unsourced number. Emit the sum as
  a claim and reference it.
- "value" IS ALWAYS MANDATORY on every input, whatever its source_type. Give the
  actual number. source_type records WHERE the number came from; it does NOT
  delegate the lookup to code. An input written as
  {{"name": "R001_reported", "value": null, "source_type": "file"}} cannot be
  checked at all: the arithmetic has nothing to evaluate, and the claim is thrown
  out as unverifiable. Write {{"name": "R001_reported", "value": 9200, ...}}.
- "operation" IS ALWAYS MANDATORY. A claim with a claimed_result and no operation
  states an answer with no working, which is exactly what cannot be verified.
- "source_type" says WHERE each input came from, and decides whether code checks
  it against the files:
    "file"     — read verbatim from a supplied file. Code will look for it, so it
                 must actually be there.
    "claim"    — the output of another claim. Set from_claim as well.
    "metadata" — arithmetic on something the TASK states rather than a data cell:
                 "the 9:00-10:30 sub-window is 1.5 hours", "a quarter is 3
                 months". These never appear verbatim in a file and are not
                 errors. Use this only for values derivable from stated facts,
                 never as a way to avoid sourcing a real figure.

Rules for "judgment_steps" — the non-arithmetic moves:
- These are the comparisons, selections, exclusions and rulings that consume the
  numbers: choosing an option, rejecting an alternative, declaring a bottleneck.
  Without them the derivation is a pile of arithmetic with no conclusion.
- IDs must be J1, J2, ... and must not collide with claim IDs.
- "consumes" may name claims or earlier judgment steps. A judgment step that
  consumes nothing is almost certainly not part of the derivation.
- "basis" must quote or closely paraphrase the text it rests on.

Rules for "corrected_verifiers" specifically:
- Emit the whole block as "V1: ...\nV2: ..." lines, keeping every id and its
  numbering. Do not renumber, drop or reorder verifiers.
- RETARGETING a verifier to a figure YOUR OWN correction changed is MECHANICAL.
  If you corrected global utilization from 88.34% to 91.53%, then a verifier
  pinning 88.34% is now wrong by arithmetic, the new target is determined, and you
  should fix it and log it as MECHANICAL. Leaving it stale ships a scoring
  criterion that contradicts the golden it scores against.
- RESHAPING a verifier is JUDGMENT_REQUIRED: widening or narrowing a tolerance
  band, changing what it tests, choosing between two defensible bases (outbound
  vs inbound throughput). State the question and LEAVE THE ORIGINAL TEXT in
  place — do not resolve it in the corrected block. Code compares the block
  against what you declared, so a judgment item you silently resolved will be
  reported.

Rules for the corrected artifacts:
- Emit the FULL replacement text, not a diff. Downstream code consumes these
  directly; a fragment would silently truncate the artifact.
- Emit an empty string when nothing changed. Do not echo the original back —
  an unchanged artifact and a rewritten-but-identical one are different facts.
- Every edit inside a corrected artifact must ALSO appear as an entry in
  "changes", so the reviewer sees each edit with its rationale rather than
  having to diff two blocks of prose.
- A JUDGMENT_REQUIRED item must NOT be resolved inside a corrected artifact.
  Leave the original wording and pose the question; rewriting it there would
  bury an authorial decision in a wall of text.
- The prompt is the contract with the model under test. Correcting it changes
  what the task asks, so restrict prompt edits to removing leakage and fixing
  outright errors — never to making the task easier or narrowing the solution
  space.

Output ONLY the <analysis> block then the JSON.
"""