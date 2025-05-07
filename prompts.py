# prompts.py
# Centralized prompt templates for Night_watcher Knowledge Graph population and analysis phases

# Round 1: Fact Extraction
FACT_EXTRACTION_PROMPT = """
Extract only the objective facts, direct actions, dates, and explicit statements from the following article.

Avoid speculation, bias, or inferred motives. Do not summarize. Your goal is to build a dataset of observable, verifiable facts.

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

# Round 2: Article Analysis
ARTICLE_ANALYSIS_PROMPT = """
Analyze the following political article for presentation bias, narrative framing, omitted context, and tone.

Break your analysis into:
1. FRAMING: What perspectives or ideologies are amplified?
2. OMISSIONS: What voices or facts are missing?
3. TONE & LANGUAGE: Is the language neutral or emotionally charged?
4. MANIPULATION: Are there rhetorical techniques that might subtly influence the reader?
5. DEMOCRATIC CONCERNS: What, if any, patterns or implications suggest risk to democratic norms?

CONTENT:
{article_content}
"""

# Round 3: Node Extraction
NODE_EXTRACTION_PROMPT = """
Extract every discrete fact, action, entity, or event from the following content. For each, produce a JSON object with:
- node_type: one of [actor, institution, policy, event, media_outlet, civil_society]
- name: the precise name or title
- attributes: a key/value object with any additional details (e.g. role, party, branch, description)
- timestamp: the YYYY-MM-DD date associated (if none, use "N/A")
- source_sentence: the exact sentence containing this information

Your ENTIRE response must be ONLY a valid JSON array of these node objects. Do not include any text outside the JSON array.

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
    "node_type": "actor",
    "name": "John Smith",
    "attributes": {{
      "role": "Senator",
      "party": "Republican"
    }},
    "timestamp": "N/A",
    "source_sentence": "Senator John Smith criticized the decision."
  }}
]

Do not include explanatory text before or after the JSON array. Ensure valid JSON with no trailing commas.

CONTENT:
{article_content}
"""

# Round 4: Node Deduplication & Normalization with Fuzzy Merge
NODE_DEDUPLICATION_PROMPT = """
You are given a JSON array of node objects and a publication_date.

1. Assign each unique (node_type, name) pair a numeric "id" (starting at 1).
2. If multiple entries share (node_type, name) or are highly similar in name, merge their attributes (union all keys).
3. Carry over the earliest timestamp; if missing or "N/A", default to the provided publication_date.

Your ENTIRE response must be ONLY a valid JSON array of objects with this format:
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

# Round 5: Edge Extraction with Enhanced Relation Types
EDGE_EXTRACTION_PROMPT = """
Given:
- the array of unique nodes (with id, node_type, name)
- the original facts JSON

Identify all relations among node IDs using ONLY these types:
[part_of, influences, opposes, supports, restricts, undermines, authorizes,
 co-opts, purges, criminalizes, censors, intimidates, delegitimizes,
 backs_with_force, context_of, analogous_to]

- Use 'supports' only when the text explicitly indicates endorsement.
- Use 'analogous_to' to link nodes representing similar policies/events without direct backing.
- If timestamp is missing or "N/A", default to the publication_date from the facts JSON.

Your ENTIRE response must be ONLY a valid JSON array with edges in this format:
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

# Round 6: Edge Enrichment with Default Timestamp Handling
EDGE_ENRICHMENT_PROMPT = """
You are given a JSON array of edges and a publication_date.
For each edge, add:
- severity: float 0.0–1.0 representing impact (0.0 is minimal, 1.0 is severe)
- is_decayable: true if this edge's effect naturally fades over time, false if permanent
- reasoning: a brief explanation for your severity and decayable assessments

If edge timestamp was "N/A", use the publication_date for assessment.

Your ENTIRE response must be ONLY a valid JSON array with edges in this format:
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

# Round 7: Package for Ingestion
PACKAGE_INGESTION_PROMPT = """
Given:
- the deduplicated node array with ids
- the enriched edge array

Create a single JSON object with this exact structure:
{{
  "nodes": [ ... ],
  "edges": [ ... ]
}}

Your ENTIRE response must be ONLY this valid JSON object. Do not include any text before or after the JSON.

NODES:
{nodes}

EDGES:
{edges}
"""