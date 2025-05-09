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

# Manipulation Score Analysis
MANIPULATION_SCORE_PROMPT = """
Analyze the following political article for manipulation techniques.

Focus specifically on:
1. Framing: How the story is presented and what perspective is centered
2. Language: Use of emotionally charged language, loaded terms, or persuasive devices
3. Omissions: What relevant context or counterpoints are excluded
4. Attribution: Whether claims are properly sourced and attributed
5. Fact/Opinion Blending: How facts are mixed with analysis or opinion

First provide a detailed analysis of these aspects. Then, end with:

MANIPULATION SCORE: [1-10]

Where 1 = highly objective news with minimal manipulation
And 10 = extreme manipulation with significant distortion of facts

CONTENT:
{article_content}
"""

# Authoritarian Analysis
AUTHORITARIAN_ANALYSIS_PROMPT = """
Analyze the following political content for potential authoritarian indicators.

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

Be balanced and objective. Do not overinterpret ambiguous statements, but don't overlook concerning patterns.

End your analysis with:
AUTHORITARIAN INDICATORS: [List the specific indicators found, or "None detected" if none]
CONCERN LEVEL: [None, Low, Moderate, High, Very High]

CONTENT:
{article_content}
"""
