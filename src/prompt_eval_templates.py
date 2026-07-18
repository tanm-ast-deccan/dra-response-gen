# src/prompt_eval_templates.py
"""
Domain-agnostic LLM prompt templates for evaluating prompt quality.
All templates accept dynamically injected rubric content — no hardcoded criteria.
"""

# =============================================================================
# TEMPLATE 0: RUBRIC EXTRACTION (instruction doc → rubric JSON)
# Mirrors RUBRIC_EXTRACTION_TEMPLATE from prompts.py but targets prompt
# quality evaluation instead of annotator response evaluation.
# =============================================================================

PROMPT_RUBRIC_EXTRACTION_TEMPLATE = """
You are an expert evaluation designer. Your task is to analyze an instruction document that describes how to evaluate the quality of benchmark prompts, and extract a structured evaluation rubric from it.

This is NOT about evaluating AI model responses. This is about evaluating whether PROMPTS THEMSELVES are well-designed for benchmarking AI agents.

Analyze the provided instruction document to identify:
1. All evaluation criteria/dimensions for judging prompt quality
2. The possible score levels for each criterion (e.g., 0/1/2/3 or similar scales)
3. Definitions and guidelines for each criterion at each level
4. Any "gate" criteria that must meet a minimum score for acceptance
5. Any binary classification tags (e.g., research-required vs computation-only)
6. Any prompt type taxonomy with validation rules
7. Decision rules for accept/revise/reject outcomes
8. Any calibration examples showing scored prompts

Create a structured JSON object with this schema:
{{
    "rubric": {{
        "name": "Name of the evaluation rubric",
        "description": "Brief description of what is being evaluated. State the total number of criteria, any binary tags, and the critical gate criterion if one exists.",
        "categories": [
            {{
                "id": "snake_case_id",
                "name": "Human Readable Name",
                "description": "Full definition from the document",
                "type": "categorical",
                "is_gate_criterion": false,
                "gate_minimum": null,
                "levels": [
                    {{"label": "Level Name", "value": 0, "description": "Full description of what this score means"}},
                    {{"label": "Level Name", "value": 1, "description": "..."}},
                    {{"label": "Level Name", "value": 2, "description": "..."}},
                    {{"label": "Level Name", "value": 3, "description": "..."}}
                ]
            }}
        ],
        "binary_tags": [
            {{
                "id": "tag_id",
                "name": "Tag Name",
                "description": "What this tag classifies",
                "values": [
                    {{"label": "VALUE_A", "description": "When to assign this value"}},
                    {{"label": "VALUE_B", "description": "When to assign this value"}}
                ]
            }}
        ],
        "prompt_types": [
            {{
                "code": "TYPE_CODE",
                "name": "Type Name",
                "description": "What this type tests",
                "validation": "How to verify a prompt actually exercises this type"
            }}
        ],
        "decision_rules": {{
            "accept": "Conditions for accepting a prompt",
            "revise": "Conditions for sending back for revision",
            "reject": "Conditions for rejecting a prompt"
        }},
        "calibration_examples": [
            {{
                "label": "Example label (e.g., STRONG, BORDERLINE, WEAK)",
                "decision": "ACCEPT/REVISE/REJECT",
                "summary": "Brief description of the example prompt",
                "scores": {{"criterion_id": 0}},
                "reasoning": "Why this prompt received these scores"
            }}
        ]
    }}
}}

IMPORTANT EXTRACTION RULES:
- Extract ALL criteria mentioned in the document. Do not add criteria that are not in the document.
- Use snake_case for all IDs.
- Include ALL possible score levels for each criterion with their FULL descriptions.
- Be faithful in copying level descriptions — do not paraphrase or summarize.
- If a criterion is marked as a "gate" or "critical" or "must-pass", set is_gate_criterion=true and gate_minimum to the required score.
- If the document defines binary tags (e.g., RESEARCH_REQUIRED vs COMPUTATION_ONLY), extract them into binary_tags.
- If the document defines prompt types (e.g., FSP, CRP, LDP), extract them into prompt_types with any validation rules.
- If the document includes calibration/example prompts with scores, extract them into calibration_examples.
- If decision_rules, binary_tags, prompt_types, or calibration_examples are not present in the document, use empty arrays/objects for those fields.
- Do not add any labels, levels, or criteria from your own knowledge.

Return ONLY the raw JSON object, without any markdown formatting or explanations.

---
INSTRUCTION DOCUMENT:
{instruction_document_text}
---
"""

# =============================================================================
# TEMPLATE 1: SCORE PROMPT (used in batch and individual modes)
# Scores a single prompt against all criteria in the provided rubric.
# In individual mode, also generates improvement suggestions.
# =============================================================================

PROMPT_SCORE_TEMPLATE = """You are an expert evaluation scientist. Your task is to assess the quality of a benchmark prompt against a provided rubric.

You are evaluating whether this prompt is suitable for benchmarking AI agent capabilities. You are NOT answering the prompt. You are judging whether the prompt itself is well-designed.

## RUBRIC

{rubric_text}

## GATE CRITERIA

{gate_criteria_text}

## DECISION RULES

{decision_rules_text}

## PROMPT METADATA

- Prompt ID: {prompt_id}
- Assigned Type: {prompt_type}
- Domain: {domain}
{extra_metadata}

## PROMPT TEXT

{prompt_text}

{supporting_evidence_section}

## YOUR TASK

Score this prompt on EVERY criterion defined in the rubric above. For each criterion:
1. Read the level descriptions carefully
2. Identify concrete evidence in the prompt text that supports your score
3. Assign a score (use the numeric values defined in the rubric levels)
4. Write a 1-3 sentence justification citing specific parts of the prompt

{cross_validation_instructions}

{additional_instructions}

## OUTPUT (JSON only, no other text)

CRITICAL JSON FORMATTING RULES — follow these strictly:
- Do NOT use double quotes (") inside string values. Use single quotes (') instead when quoting text from the prompt.
  BAD:  "justification": "Includes "Do not use" constraints"
  GOOD: "justification": "Includes 'Do not use' constraints"
- Do NOT include line breaks inside string values.
- Do NOT add trailing commas after the last item in objects or arrays.

{{
    "prompt_id": "{prompt_id}",
    "scores": {{
        "<criterion_id>": {{
            "score": <integer>,
            "justification": "<1-3 sentences with specific evidence from prompt>"
        }}
    }},
    "research_depth_tag": "<tag value if binary_tags defined in rubric, else null>",
    "research_depth_justification": "<1-2 sentences>",
    "average_score": <float, mean of all criterion scores>,
    "gate_passed": <true/false, based on gate criteria rules>,
    "decision": "<ACCEPT|REVISE|REJECT>",
    "decision_logic": "<which specific rules triggered this decision>",
    {improvement_fields}
    "prompt_type_validation": {{
        "assigned_type": "{prompt_type}",
        "type_appropriate": <true/false>,
        "reasoning": "<why type is or isn't appropriate>",
        "suggested_type": "<suggested type if inappropriate, else null>"
    }}{cross_validation_output}
}}
"""

# =============================================================================
# Supporting evidence section — injected when Logic, SC, or GDrive files
# are present. The {gdrive_files_section} placeholder is either filled with
# GDRIVE_FILES_SECTION or left as an empty string.
# =============================================================================

SUPPORTING_EVIDENCE_SECTION = """## SME SUPPORTING EVIDENCE

The SME who wrote this prompt also submitted the following supporting materials.
Use these to CROSS-VALIDATE the prompt — check whether the prompt actually delivers on
what the Logic and Sanity Check claim.

### Solution Logic (SME's claimed solution trace)
{logic_text}

### Sanity Check (SME's claimed difficulty justification)
{sc_text}
{gdrive_files_section}"""

# Injected into SUPPORTING_EVIDENCE_SECTION when GDrive files are present
GDRIVE_FILES_SECTION = """
### Reference Files from Google Drive
The SME attached the following reference documents to this prompt.
Use these to further validate whether the prompt is consistent with
the data sources, calculation steps, and constraints it claims to use.

{gdrive_text}
"""

# Cross-validation instructions (injected only when Logic/SC are present)
CROSS_VALIDATION_INSTRUCTIONS = """
IMPORTANT — CROSS-VALIDATION WITH SUPPORTING EVIDENCE:
The Solution Logic and Sanity Check above are the SME's CLAIMS about their prompt.
You must verify whether the prompt actually delivers on these claims:

1. **Logic ↔ Prompt alignment**: Does the prompt text actually require the steps described
   in the Solution Logic? If Logic says "must retrieve X from external source" but the prompt
   never mentions external retrieval, that's a gap (lower Constraint Rigidity, Inference Necessity).
2. **SC ↔ Prompt trap setup**: Does the prompt actually set up the failure trap described in
   the Sanity Check? If SC says "lazy AI will use placeholder tax rate" but the prompt doesn't
   embed a misleading placeholder, DRA Necessity is lower than claimed.
3. **Research depth consistency**: If Logic lists external URLs/data requirements, the prompt
   should be tagged RESEARCH_REQUIRED. If Logic says "Internal Deduction Only", tag COMPUTATION_ONLY.
4. **Type consistency**: If Logic describes external data retrieval but the assigned type is
   RCP/LDP (closed system), flag the mismatch.
5. **Completeness gaps**: If Logic references specific files, columns, or calculation steps
   that the prompt doesn't mention or set up, that reduces Prompt Parsability.
6. **Logic/SC quality flags**: Flag if Logic is missing or just a reference to external files
   (e.g., "see attached Word doc"), or if SC is trivially short (< 100 chars) or doesn't
   describe a specific failure mode.
7. **GDrive file consistency** (if reference files were provided): Check whether the prompt
   correctly reflects the data schema, constraints, and values present in the reference files.
   Flag any mismatch between what the files contain and what the prompt describes.
"""

# Cross-validation output fields (appended to JSON schema when Logic/SC present)
CROSS_VALIDATION_OUTPUT = """,
    "cross_validation": {{
        "logic_present": <true/false>,
        "logic_quality": "<STRONG|ADEQUATE|WEAK|MISSING>",
        "logic_quality_reasoning": "<1-2 sentences on Logic completeness: does it have external refs, calc steps, golden answer?>",
        "sc_present": <true/false>,
        "sc_quality": "<STRONG|ADEQUATE|WEAK|MISSING>",
        "sc_quality_reasoning": "<1-2 sentences on SC: does it describe specific failure trap and expert path?>",
        "prompt_delivers_on_logic": <true/false>,
        "prompt_delivers_on_sc": <true/false>,
        "gaps": [
            {{
                "type": "<LOGIC_GAP|SC_GAP|TYPE_MISMATCH|RESEARCH_MISMATCH|COMPLETENESS_GAP|META_PROMPT|GDRIVE_FILE_MISMATCH>",
                "description": "<specific gap found between prompt and supporting evidence>"
            }}
        ],
        "is_meta_prompt": <true/false, true if prompt describes itself rather than being a task>
    }}"""

# Empty cross-validation placeholders when Logic/SC are absent
NO_CROSS_VALIDATION_INSTRUCTIONS = ""
NO_CROSS_VALIDATION_OUTPUT = ""
NO_SUPPORTING_EVIDENCE = ""

# Additional instructions appended in individual mode only
INDIVIDUAL_EXTRA_INSTRUCTIONS = """
Additionally, if the decision is REVISE or REJECT:
- Identify the weakest criteria (lowest scores)
- For each weak criterion, provide a SPECIFIC, ACTIONABLE suggestion to improve the prompt
- If possible, sketch a 1-2 sentence example of how the prompt could be rewritten to address the weakness
- Reference the rubric level descriptions to explain what "good" looks like for that criterion

Also validate whether the assigned prompt type is correct by checking the prompt against the type validation rules in the rubric.
"""

# Improvement fields included in individual mode JSON output
INDIVIDUAL_IMPROVEMENT_FIELDS = """\"improvement_suggestions\": [
        {
            \"criterion_id\": \"<id of weak criterion>\",
            \"current_score\": <int>,
            \"target_score\": <int>,
            \"issue\": \"<what's wrong>\",
            \"fix\": \"<specific actionable fix>\",
            \"example_rewrite_fragment\": \"<optional: 1-2 sentence rewrite snippet>\"
        }
    ],"""

# Minimal fields in batch mode (no improvement suggestions)
BATCH_IMPROVEMENT_FIELDS = ""

# =============================================================================
# TEMPLATE 2: AUDIT TEMPLATE (compare LLMAJ scores vs existing SME scores)
# =============================================================================

PROMPT_AUDIT_TEMPLATE = """You are an expert evaluation scientist performing a calibration audit. You will score a prompt against a rubric AND compare your scores to existing human scores.

## RUBRIC

{rubric_text}

## GATE CRITERIA

{gate_criteria_text}

## PROMPT METADATA

- Prompt ID: {prompt_id}
- Assigned Type: {prompt_type}
- SME Name: {sme_name}

## PROMPT TEXT

{prompt_text}

{supporting_evidence_section}

## EXISTING HUMAN SCORES

{existing_scores_text}

## YOUR TASK

1. Score this prompt independently on every criterion in the rubric (ignore the human scores while scoring)
2. If Solution Logic and Sanity Check are provided above, cross-validate the prompt against them
3. After scoring, compare your scores to the human scores
4. For each criterion where you DISAGREE with the human score, explain specifically why

## OUTPUT (JSON only, no other text)

CRITICAL JSON FORMATTING RULES — follow these strictly:
- Do NOT use double quotes (") inside string values. Use single quotes (') instead when quoting text from the prompt.
  BAD:  "justification": "Includes "Do not use" constraints"
  GOOD: "justification": "Includes 'Do not use' constraints"
- Do NOT include line breaks inside string values.
- Do NOT add trailing commas after the last item in objects or arrays.

{{
    "prompt_id": "{prompt_id}",
    "llm_scores": {{
        "<criterion_id>": {{
            "score": <integer>,
            "justification": "<1-3 sentences>"
        }}
    }},
    "human_scores": {{
        "<criterion_id>": <integer from existing scores>
    }},
    "disagreements": [
        {{
            "criterion_id": "<id>",
            "llm_score": <int>,
            "human_score": <int>,
            "delta": <int>,
            "explanation": "<why scores differ — cite specific prompt evidence>"
        }}
    ],
    "llm_average": <float>,
    "human_average": <float>,
    "llm_decision": "<ACCEPT|REVISE|REJECT>",
    "human_decision": "<ACCEPT|REVISE|REJECT based on human scores>",
    "decisions_agree": <true/false>,
    "overall_calibration_note": "<1-2 sentences on systematic patterns in disagreement>"
}}
"""

# =============================================================================
# TEMPLATE 3: GDRIVE FILE SUMMARIZATION
# Used by gdrive_fetcher.py to pre-summarize large reference files before
# injecting into the scoring prompt. Focuses the LLM on cross-validation-
# relevant content rather than producing a generic document summary.
# =============================================================================

GDRIVE_SUMMARIZATION_TEMPLATE = """You are a research analyst helping to evaluate whether a benchmark prompt is well-designed.

A reference document was attached by the prompt author as supporting material.
Your job is to extract ONLY the information relevant to cross-validating the specific prompt below.

## THE PROMPT BEING EVALUATED
{prompt_text}

## REFERENCE DOCUMENT: {filename}
{file_content}

## YOUR TASK
Write a concise summary (200-400 words) focusing ONLY on:
1. Data fields, columns, or metrics in this document that the prompt should reference
2. Calculation steps, formulas, or lookup values the prompt should require the AI to use
3. Constraints, edge cases, or failure traps that the prompt should set up
4. Any discrepancies between what this document describes and what the prompt actually asks for

Do NOT summarize the document generically. Focus entirely on what matters for evaluating whether this prompt is well-designed and internally consistent.
Write in plain text without markdown headers.
"""


# =============================================================================
# HELPER: Build rubric text from JSON for template injection
# =============================================================================

def build_rubric_text(rubric: dict) -> str:
    """Convert rubric JSON to human-readable text for LLM consumption."""
    rubric_data = rubric.get("rubric", rubric)
    lines = []

    lines.append(f"### {rubric_data.get('name', 'Evaluation Rubric')}")
    lines.append("")
    lines.append(rubric_data.get("description", ""))
    lines.append("")

    # Categories / criteria
    for cat in rubric_data.get("categories", []):
        cat_id = cat.get("id", "unknown")
        cat_name = cat.get("name", cat_id)
        lines.append(f"#### Criterion: {cat_name} (id: {cat_id})")
        lines.append(f"**Definition**: {cat.get('description', '')}")
        lines.append("")

        if cat.get("is_gate_criterion"):
            lines.append(
                f"**⚠ GATE CRITERION**: This criterion must score "
                f">= {cat.get('gate_minimum', 'N/A')} for prompt acceptance."
            )
            lines.append("")

        if "levels" in cat:
            lines.append("**Scoring Levels:**")
            for level in cat["levels"]:
                label = level.get("label", "")
                value = level.get("value", "")
                desc = level.get("description", "")
                lines.append(f"  - **{value} ({label})**: {desc}")
            lines.append("")

    # Binary tags
    for tag in rubric_data.get("binary_tags", []):
        lines.append(f"#### Binary Tag: {tag.get('name', '')} (id: {tag.get('id', '')})")
        lines.append(f"**Definition**: {tag.get('description', '')}")
        for val in tag.get("values", []):
            lines.append(f"  - **{val['label']}**: {val['description']}")
        lines.append("")

    # Prompt types (if defined)
    prompt_types = rubric_data.get("prompt_types", [])
    if prompt_types:
        lines.append("#### Prompt Type Validation Rules")
        for pt in prompt_types:
            lines.append(
                f"  - **{pt['code']}** ({pt['name']}): {pt.get('validation', '')}"
            )
        lines.append("")

    return "\n".join(lines)


def build_gate_criteria_text(rubric: dict) -> str:
    """Extract gate criteria description from rubric."""
    rubric_data = rubric.get("rubric", rubric)
    gates = []
    for cat in rubric_data.get("categories", []):
        if cat.get("is_gate_criterion"):
            gates.append(
                f"- **{cat['name']}** (id: {cat['id']}): "
                f"Must score >= {cat['gate_minimum']} for acceptance."
            )
    if not gates:
        return "No gate criteria defined. Decision is based on average score only."
    return "\n".join(gates)


def build_decision_rules_text(rubric: dict) -> str:
    """Extract decision rules from rubric."""
    rubric_data = rubric.get("rubric", rubric)
    rules = rubric_data.get("decision_rules", {})
    if not rules:
        return (
            "Default rules:\n"
            "- ACCEPT: Average score >= 2.5 and all gate criteria met\n"
            "- REVISE: Average score >= 2.0 or gate criteria close to threshold\n"
            "- REJECT: Any criterion scored 0 or average < 2.0"
        )
    lines = []
    for decision, rule in rules.items():
        lines.append(f"- **{decision.upper()}**: {rule}")
    return "\n".join(lines)


def build_existing_scores_text(scores: dict, criteria_ids: list) -> str:
    """Format existing human scores for audit template."""
    lines = []
    for cid in criteria_ids:
        val = scores.get(cid, "N/A")
        lines.append(f"  - {cid}: {val}")
    return "\n".join(lines)


def get_criteria_ids(rubric: dict) -> list:
    """Extract ordered list of criterion IDs from rubric."""
    rubric_data = rubric.get("rubric", rubric)
    return [cat["id"] for cat in rubric_data.get("categories", [])]


def get_gate_criteria(rubric: dict) -> list:
    """Extract gate criteria with their minimums."""
    rubric_data = rubric.get("rubric", rubric)
    return [
        {"id": cat["id"], "name": cat["name"], "minimum": cat["gate_minimum"]}
        for cat in rubric_data.get("categories", [])
        if cat.get("is_gate_criterion")
    ]


def get_score_range(rubric: dict) -> tuple:
    """Get (min_score, max_score) from rubric levels."""
    rubric_data = rubric.get("rubric", rubric)
    all_values = []
    for cat in rubric_data.get("categories", []):
        for level in cat.get("levels", []):
            all_values.append(level.get("value", 0))
    if not all_values:
        return (0, 3)
    return (min(all_values), max(all_values))