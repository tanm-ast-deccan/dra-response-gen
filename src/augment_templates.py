# src/augment_templates.py
"""
Prompt template for the AUGMENT call — run AFTER the two-call auditor has
verified arithmetic and produced corrections. This single Opus call takes the
verified/corrected task and emits, together:

  1. the GOLDEN DELIVERABLE (the reference answer artifact, following the gdpval
     gold_deliverable schema: format + ordered sections),
  2. the DAG edges over the (possibly augmented) verifier set,
  3. the two Sanity-Check anchor sets used to select crux verifiers:
       - trap_anchor_ids   : verifiers tied to the Lazy-AI failure mode
       - expert_anchor_ids : verifiers tied to the expert analytical act + deliverable
  4. any augmented verifiers to ADD (to cover uncovered gold values / missing dims).

Crux SELECTION itself is done in code (crux_shapley.select_crux) from these
anchors + the DAG — the model only supplies the anchors and edges, not the final
crux set, keeping selection deterministic and reproducible.
"""

AUGMENT_SYSTEM_PROMPT = """\
You are a benchmark golden-sample author. The task has already been audited and \
its arithmetic verified by a separate code step; you are given the corrected, \
trustworthy solution logic. Your job now is to (a) write the golden deliverable \
that a top expert would produce, (b) declare the dependency structure (DAG) among \
the verifiers, and (c) identify which verifiers are tied to the Sanity Check's two \
halves — the Lazy-AI failure mode and the expert analytical path. You do NOT \
re-audit and you do NOT do fresh arithmetic; use the verified numbers as given.
"""

AUGMENT_TEMPLATE = """\
The task below has been audited; its arithmetic is verified and the solution \
logic here is the corrected, authoritative version. Produce the golden-sample \
augmentation as a single JSON object.

## TASK ID
{task_id}

## PROMPT (model-facing)
{prompt_text}

## CORRECTED SOLUTION LOGIC (authoritative — verified by code)
{solution_logic_text}

## SANITY CHECK (both halves matter)
{sanity_check_text}

## VERIFIERS (canonical V<n>: text form)
{verifiers_text}

## VERIFIED LOAD-BEARING FIGURES (from the arithmetic verifier — authoritative)
{arithmetic_results_text}

---

In an <analysis> block, plan: the deliverable's required format and sections
(from the prompt's output spec); the gold values that populate it; the dependency
edges among verifiers (V_b depends on V_a iff computing/observing V_b requires
V_a's result — an *arithmetic/derivation* dependency, NOT mere ordering); which
verifiers correspond to the Lazy-AI trap and which to the expert analytical act
and final deliverable.

Then output a SINGLE JSON object (after </analysis>):

{{
  "task_id": "{task_id}",
  "gold_deliverable": {{
    "format": "docx|md|txt|xlsx|structured-text",
    "sections": [
      {{"title": "section name", "content": "full section content with the gold numbers"}}
    ]
  }},
  "dag_edges": {{
    "V1": [],
    "V2": ["V1"]
  }},
  "trap_anchor_ids": ["Vx"],
  "expert_anchor_ids": ["Vy", "Vz"],
  "augmented_verifiers": [
    {{"id": "V<next>", "text": "criterion", "dim": "DI|AR|RF|EP|FD",
      "type": "TRAP|null", "is_decision": false, "rationale": "why added"}}
  ],
  "expected_values": {{
    "V1": {{"value": 92.60, "tol": 0.05, "unit": "INR/USD", "kind": "numeric", "source_of_verification": "arithmetic|source_file|llm_judgment|judgment_flagged"}},
    "V11": {{"value": "ACCEPT", "tol": 0, "unit": "", "kind": "decision", "source_of_verification": "llm_judgment"}}
  }},
  "notes": "anything the SME should later confirm"
}}

Rules:
- gold_deliverable.content must contain the actual gold numbers, not placeholders.
- dag_edges must include EVERY verifier id (augmented ones too) as a key; roots map to [].
- trap_anchor_ids: verifiers that check the Lazy-AI failure mode is avoided
  (e.g. "rejects the stale rate", "does not use the placeholder").
- expert_anchor_ids: verifiers on the main analytical spine AND the decision/
  final-metric verifier(s) — the expert's core act and the deliverable's headline.
- Only ADD augmented verifiers to cover a gold value or dimension the existing
  set misses; do not restate existing verifiers. Empty list if none needed.
- expected_values: for EVERY crux-candidate verifier (anything tied to a trap or
  expert anchor), freeze the machine-checkable target from the verified figures:
  value (number or the exact decision string), tol (absolute tolerance; 0 for
  decisions/exact), unit, and kind ("numeric" | "decision" | "string"). This is
  the frozen standard the scorer matches responses against — do NOT leave a crux
  verifier without an expected value.
- source_of_verification: for EACH expected value, state how the target was
  established, so its trust level travels into scoring:
    "arithmetic"       — recomputed deterministically by the code arithmetic verifier
    "source_file"      — read directly from an input file confirmed present in source
    "llm_judgment"     — your judgment about method/formula, not code-verified
    "judgment_flagged" — contested/uncertain; carries an open JUDGMENT_REQUIRED question
  Prefer "arithmetic"/"source_file" wherever the figure was actually verified that
  way; use "llm_judgment" honestly when it was your call; use "judgment_flagged"
  when the value depends on an unresolved authorial question.
- Output ONLY the <analysis> block then the JSON.
"""