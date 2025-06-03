# prompts.py
# Template-based prompt system for Night_watcher analysis

# Template-based prompts with variable substitution

FACT_EXTRACTION_PROMPT_TEMPLATE = """
Extract only the objective facts, direct actions, dates, and explicit statements from the following {content_type}.

Avoid speculation, bias, or inferred motives. Do not summarize. Your goal is to build a dataset of observable, verifiable facts.

{json_instruction}

Your response MUST be a valid, parseable JSON object in exactly this format:
{{
  "publication_date": "YYYY-MM-DD",
  "facts": ["Fact 1", "Fact 2", "Fact 3"],
  "events": [
    {{
      "name": "Event name",
      "date": "YYYY-MM-DD",
      "description": "Short factual description"
    }}
  ],
  "direct_quotes": ["Quote from source text", "Another quote"]
}}

Do not include any text before or after the JSON. Use "N/A" for any unknown dates. Include at least one item in each array.

CONTENT:
{article_content}
"""

ARTICLE_ANALYSIS_PROMPT_TEMPLATE = """
Analyze the following {content_type} for presentation bias, narrative framing, omitted context, and tone.

Break your analysis into:
1. FRAMING: What perspectives or ideologies are amplified?
2. OMISSIONS: What voices or facts are missing?
3. TONE & LANGUAGE: Is the language neutral or emotionally charged?
4. MANIPULATION: Are there rhetorical techniques that might subtly influence the reader?
5. DEMOCRATIC CONCERNS: What, if any, patterns or implications suggest risk to democratic norms?

{analysis_instruction}

CONTENT:
{article_content}
"""

MANIPULATION_SCORE_PROMPT_TEMPLATE = """
Analyze the following {content_type} for manipulation techniques.

Focus specifically on:
1. {focus_area_1}
2. {focus_area_2}
3. {focus_area_3}
4. {focus_area_4}
5. {focus_area_5}

First provide a detailed analysis of these aspects. Then, end with:

MANIPULATION SCORE: [1-10]

{manipulation_scale}

CONTENT:
{article_content}
"""

AUTHORITARIAN_ANALYSIS_PROMPT_TEMPLATE = """
Analyze the following {content_type} for potential authoritarian indicators.

Your task is to identify any patterns that may signal democratic erosion or authoritarian tendencies, such as:
- Attempts to undermine separation of powers
- Delegitimization of opposition, media, or institutions
- Expansion of executive authority
- Limitations on civil liberties or rights
- Erosion of electoral integrity
- Centralization of power
- Use of state resources for partisan advantage
- Degradation of factual discourse
- Promotion of us-vs-them narratives

{analysis_instruction}

End your analysis with:
AUTHORITARIAN INDICATORS: [List the specific indicators found, or "None detected" if none]
CONCERN LEVEL: [None, Low, Moderate, High, Very High]

CONTENT:
{article_content}
"""

NODE_EXTRACTION_PROMPT_TEMPLATE = """
Extract every discrete fact, action, entity, or event from the following content. For each, produce a JSON object with:
- node_type: one of [{entity_types}]
- name: the precise name or title
- attributes: a key/value object with any additional details (e.g. role, party, branch, description)
- timestamp: the YYYY-MM-DD date associated (if none, use "N/A")
- source_sentence: the exact sentence containing this information

Pay special attention to identifying:
- narratives: recurring propaganda themes, framing techniques (e.g., "deep state", "enemy of the people")
- legal_frameworks: changes to laws and regulations (e.g., "Emergency Powers Act", "Anti-Protest Bill")
- procedural_norms: changes to governmental procedures (e.g., "Senate Confirmation Process", "Executive Oversight")

{json_instruction}

Example format:
[
  {{
    "node_type": "event",
    "name": "Arrest of Judge Dugan",
    "attributes": {{
      "location": "Washington DC",
      "type": "judicial action"
    }},
    "timestamp": "2019-04-01",
    "source_sentence": "Judge Dugan was arrested on April 1, 2019."
  }},
  {{
    "node_type": "narrative",
    "name": "Judicial Obstruction",
    "attributes": {{
      "theme": "opposition to executive",
      "origin": "administration"
    }},
    "timestamp": "N/A",
    "source_sentence": "The administration has repeatedly characterized judges as obstructing the legitimate functions of government."
  }}
]

Do not include explanatory text before or after the JSON array. Ensure valid JSON with no trailing commas.

CONTENT:
{article_content}
"""

NODE_DEDUPLICATION_PROMPT_TEMPLATE = """
You are given a JSON array of node objects and a publication_date.

1. Assign each unique (node_type, name) pair a numeric "id" (starting at 1).
2. If multiple entries share (node_type, name) or are highly similar in name, merge their attributes (union all keys).
3. Carry over the earliest timestamp; if missing or "N/A", default to the provided publication_date.

{json_instruction}

[
  {{
    "id": 1,
    "node_type": "event",
    "name": "Arrest of Judge Dugan",
    "attributes": {{
      "location": "Washington DC",
      "type": "judicial action"
    }},
    "timestamp": "2019-04-01"
  }}
]

Do not include any text before or after the JSON array. Ensure valid JSON with no trailing commas.

NODES:
{nodes}

PUBLICATION_DATE:
{publication_date}
"""

EDGE_EXTRACTION_PROMPT_TEMPLATE = """
Given:
- the array of unique nodes (with id, node_type, name)
- the original facts JSON

Identify all relations among node IDs using ONLY these types:
[{relation_types}]

Important notes on relation types:
- Use 'justifies' when one node (often a narrative) is used to rationalize an action
- Use 'expands_power' when an action increases an actor's authority beyond normal bounds
- Use 'normalizes' when an action serves to make previously unacceptable behavior appear routine
- Use 'diverts_attention' when one event is used to distract from another
- Use 'targets' when specific groups or institutions are directly targeted

Note: Do NOT add 'precedes' or 'follows' relations - these will be inferred automatically from timestamps.

{json_instruction}

[
  {{
    "source_id": 3,
    "relation": "intimidates",
    "target_id": 1,
    "timestamp": "2019-04-01",
    "evidence_quote": "...intimidate...the judiciary..."
  }}
]

Do not include any text before or after the JSON array. Ensure valid JSON with no trailing commas. 
If you cannot identify any valid relationships, return an empty array: []

NODES:
{nodes}

FACTS:
{facts}
"""

EDGE_ENRICHMENT_PROMPT_TEMPLATE = """
You are given a JSON array of edges and a publication_date.
For each edge, add:
- severity: float 0.0–1.0 representing impact (0.0 is minimal, 1.0 is severe)
- is_decayable: true if this edge's effect naturally fades over time, false if permanent
- reasoning: a brief explanation for your severity and decayable assessments

For authoritarian relations (undermines, co-opts, purges, criminalizes, censors, intimidates, delegitimizes, 
restricts, targets), carefully assess the severity based on democratic norms.

If edge timestamp was "N/A", use the publication_date for assessment.

{json_instruction}

[
  {{
    "source_id": 3,
    "relation": "intimidates",
    "target_id": 1,
    "timestamp": "2019-04-01",
    "evidence_quote": "...",
    "severity": 0.9,
    "is_decayable": false,
    "reasoning": "This action severely undermines judicial independence and has lasting effects"
  }}
]

Do not include any text before or after the JSON array. Ensure valid JSON with no trailing commas.
If the input is an empty array, output an empty array: []

EDGES:
{edges}

PUBLICATION_DATE:
{publication_date}
"""

PACKAGE_INGESTION_PROMPT_TEMPLATE = """
Given:
- the deduplicated node array with ids
- the enriched edge array

Create a single JSON object with this exact structure:
{{
  "nodes": [ ... ],
  "edges": [ ... ]
}}

{json_instruction}

NODES:
{nodes}

EDGES:
{edges}
"""

# Prompt template mapping
PROMPT_TEMPLATES = {
    "fact_extraction": FACT_EXTRACTION_PROMPT_TEMPLATE,
    "article_analysis": ARTICLE_ANALYSIS_PROMPT_TEMPLATE,
    "manipulation_score": MANIPULATION_SCORE_PROMPT_TEMPLATE,
    "authoritarian_analysis": AUTHORITARIAN_ANALYSIS_PROMPT_TEMPLATE,
    "node_extraction": NODE_EXTRACTION_PROMPT_TEMPLATE,
    "node_deduplication": NODE_DEDUPLICATION_PROMPT_TEMPLATE,
    "edge_extraction": EDGE_EXTRACTION_PROMPT_TEMPLATE,
    "edge_enrichment": EDGE_ENRICHMENT_PROMPT_TEMPLATE,
    "package_ingestion": PACKAGE_INGESTION_PROMPT_TEMPLATE
}


def load_template_variables(template_file: str) -> dict:
    """Load variables from template JSON file."""
    import json
    import os
    
    if not os.path.exists(template_file):
        raise FileNotFoundError(f"Template file not found: {template_file}")
    
    with open(template_file, 'r') as f:
        template = json.load(f)
    
    return template.get("variables", {})


def format_prompt(template_name: str, variables: dict, **kwargs) -> str:
    """Format a prompt template with variables."""
    if template_name not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown template: {template_name}")
    
    template = PROMPT_TEMPLATES[template_name]
    
    # Merge template variables with any additional kwargs
    format_vars = {**variables, **kwargs}
    
    try:
        return template.format(**format_vars)
    except KeyError as e:
        raise ValueError(f"Missing variable {e} for template {template_name}")
