# prompts.py

# Centralized prompt templates for Night_watcher analysis phases

FACT_EXTRACTION_PROMPT = """
Extract only the objective facts, direct actions, dates, and explicit statements from the following article.

Avoid speculation, bias, or inferred motives. Do not summarize. Your goal is to build a dataset of observable, verifiable facts.

Use this exact format in your JSON response:
{{
  "publication_date": "YYYY-MM-DD",
  "facts": ["..."],
  "events": [
    {{
      "name": "Event name",
      "date": "YYYY-MM-DD",
      "description": "Short factual description"
    }}
  ],
  "direct_quotes": ["Quote from source text", ...]
}}

CONTENT:
{article_content}
"""

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

RELATIONSHIP_EXTRACTION_PROMPT = """
From the structured facts below, identify relationships between actors, institutions, and events.

Use this format:
[
  {{
    "source": "Entity A",
    "relation": "influences / opposes / supports / restricts / undermines / authorizes",
    "target": "Entity B",
    "type": "actor-institution / actor-event / institution-event",
    "evidence": "Exact quote or summary of the fact that supports this"
  }},
  ...
]

STRUCTURED_FACTS:
{structured_data}
"""
