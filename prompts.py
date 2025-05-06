# prompts.py
# Centralized prompt templates for Night_watcher Knowledge Graph population and analysis phases

# Round 1: Fact Extraction (unchanged)
FACT_EXTRACTION_PROMPT = """
Extract only the objective facts, direct actions, dates, and explicit statements from the following article.

Avoid speculation, bias, or inferred motives. Do not summarize. Your goal is to build a dataset of observable, verifiable facts.

Use this exact format in your JSON response:
{
  "publication_date": "YYYY-MM-DD",
  "facts": ["..."],
  "events": [
    {
      "name": "Event name",
      "date": "YYYY-MM-DD",
      "description": "Short factual description"
    }
  ],
  "direct_quotes": ["Quote from source text", ...]
}

CONTENT:
{article_content}
"""

# Round 2: Article Analysis (unchanged)
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
- attributes: any key/value attributes you can extract (e.g. role, party, branch, description)
- timestamp: the YYYY-MM-DD date associated (if none, use publication_date)
- source_sentence: the exact sentence from which you drew this node

Output a JSON array of these node objects. Example:
[
  {
    "node_type":"event",
    "name":"Arrest of Judge Dugan",
    "attributes":{},
    "timestamp":"2019-04-01",
    "source_sentence":"Judge Dugan..."
  },
  …
]
CONTENT:
{article_content}
"""

# Round 4: Node Deduplication & Normalization with Fuzzy Merge
NODE_DEDUPLICATION_PROMPT = """
You are given a JSON array of node objects (from Node Extraction).
1. Assign each unique (node_type, name) pair a numeric "id" (starting at 1).
2. If multiple entries share (node_type, name), merge their attributes (union all keys).
3. Additionally, if two names are highly similar (e.g., share core phrases or have high character overlap), treat them as duplicates and merge.
4. Carry over timestamp; if missing or "N/A", default it to the publication_date provided in the facts JSON.

Output a JSON array of objects:
[
  { "id":1, "node_type":"event", "name":"Arrest of Judge Dugan", "attributes":{…}, "timestamp":"2019-04-01" },
  …
]
NODES:
{nodes}
"""

# Round 5: Edge Extraction with Enhanced Relation Types
EDGE_EXTRACTION_PROMPT = """
Given:
- the array of unique nodes (with id, node_type, name)
- the original facts JSON or article text

Identify all relations among node IDs using **only** these types:
[part_of, influences, opposes, supports, restricts, undermines, authorizes,
 co-opts, purges, criminalizes, censors, intimidates, delegitimizes,
 backs_with_force, context_of, analogous_to]

- Use 'supports' only when the text explicitly indicates endorsement or backing.
- Use 'analogous_to' to link nodes that represent similar policies/events without direct support.
- If timestamp is missing or "N/A", default it to the publication_date from the facts JSON.

For each edge, produce:
- source_id (node id)
- target_id (node id)
- relation
- timestamp
- evidence_quote (exact excerpt supporting this relation)

Output a JSON array of edges. Example:
[
  {"source_id":3,"relation":"intimidates","target_id":1,
   "timestamp":"2019-04-01",
   "evidence_quote":"…threaten and intimidate…the judiciary…"}, …
]
NODES:
{nodes}
FACTS:
{facts}
"""

# Round 6: Edge Enrichment with Default Timestamp Handling
EDGE_ENRICHMENT_PROMPT = """
You are given a JSON array of edges (from Edge Extraction).
For each edge, add two fields:
- severity: a float 0.0–1.0 representing impact (0=low, 1=high)
- is_decayable: true if this edge’s effect naturally fades, false if permanent

If the edge's timestamp was "N/A", use the publication_date as the timestamp for assessing severity.
Include a brief `reasoning` for each decision.

Output a JSON array of enriched edges:
[
  {
    "source_id":3, "relation":"intimidates", "target_id":1,
    "timestamp":"2019-04-01", "evidence_quote":"…",
    "severity":0.9, "is_decayable":false,
    "reasoning":"..."}, …
]
EDGES:
{edges}
"""

# Round 7: Package for Ingestion
PACKAGE_INGESTION_PROMPT = """
Given:
- the deduplicated node array with ids (from Node Deduplication)
- the enriched edge array (from Edge Enrichment)

Output a single JSON object:
{
  "nodes": [ … ],
  "edges": [ … ]
}

No extra keys or comments; valid JSON only.

NODES:
{nodes}
EDGES:
{edges}
"""
