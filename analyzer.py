# analyzer.py
"""
Night_watcher Content Analyzer with Multi-Round Prompting and KG Pipeline
"""

import logging
import json
import re
from datetime import datetime
from typing import Dict, Any, Optional, List
from prompts import (
    FACT_EXTRACTION_PROMPT,
    ARTICLE_ANALYSIS_PROMPT,
    NODE_EXTRACTION_PROMPT,
    NODE_DEDUPLICATION_PROMPT,
    EDGE_EXTRACTION_PROMPT,
    EDGE_ENRICHMENT_PROMPT,
    PACKAGE_INGESTION_PROMPT
)

logger = logging.getLogger(__name__)


class ContentAnalyzer:
    """Analyzer for political content with multi-round prompting and KG extraction."""

    def __init__(self, llm_provider):
        self.llm_provider = llm_provider
        self.logger = logging.getLogger("ContentAnalyzer")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        articles = input_data.get("articles", [])
        if not articles:
            self.logger.warning("No articles provided for analysis")
            return {"analyses": []}

        analyses: List[Dict[str, Any]] = []
        for article in articles:
            analyses.append(self._analyze_content_multi_round(article))
        self.logger.info(f"Completed analysis of {len(analyses)} articles")
        return {"analyses": analyses}

    def _analyze_content_multi_round(self, article: Dict[str, Any]) -> Dict[str, Any]:
        title = article.get("title", "Untitled")
        content = article.get("content", "")

        # Optimize content length - cap full content at 4000 chars for initial rounds
        core_content = self._optimize_content(content, 4000)

        self.logger.info(f"Analyzing: {title}")
        prompt_chain: List[Dict[str, Any]] = []

        # Round 1: Fact Extraction
        r1p = FACT_EXTRACTION_PROMPT.format(article_content=core_content)
        r1r = self._get_llm_response(r1p, max_tokens=2500)
        prompt_chain.append({"round": 1, "name": "Fact Extraction", "prompt": r1p, "response": r1r})

        # Extract JSON with fallback to repair
        facts = self._extract_json(r1r)
        if not facts:
            facts = self._repair_json(r1r, "object")

        # Ensure we have a valid facts object
        if not facts or not isinstance(facts, dict):
            self.logger.warning(f"Failed to extract facts JSON from round 1 for: {title}")
            facts = {
                "publication_date": datetime.now().strftime("%Y-%m-%d"),
                "facts": ["Article analyzed."],
                "events": [{"name": "Article publication", "date": "N/A", "description": "Article was published."}],
                "direct_quotes": ["No direct quotes extracted."]
            }
        else:
            self.logger.info(f"Successfully extracted facts for: {title}")

        pub_date = facts.get("publication_date", datetime.now().strftime("%Y-%m-%d"))
        facts_json = json.dumps(facts, indent=2)

        # Round 2: Article Analysis
        r2p = ARTICLE_ANALYSIS_PROMPT.format(article_content=core_content)
        r2r = self._get_llm_response(r2p, max_tokens=2500)
        prompt_chain.append({"round": 2, "name": "Article Analysis", "prompt": r2p, "response": r2r})

        # Extract key sections for entity extraction
        entity_focused_content = self._extract_entity_sections(content, facts)

        # Round 3: Node Extraction
        r3p = NODE_EXTRACTION_PROMPT.format(article_content=entity_focused_content)
        r3r = self._get_llm_response(r3p, max_tokens=3000)
        prompt_chain.append({"round": 3, "name": "Node Extraction", "prompt": r3p, "response": r3r})

        # Extract JSON with fallback to repair
        nodes = self._extract_json(r3r)
        if not nodes:
            nodes = self._repair_json(r3r, "array")

        # Ensure we have valid nodes
        if not nodes or not isinstance(nodes, list):
            self.logger.warning(f"Failed to extract nodes JSON from round 3 for: {title}")
            nodes = [
                {
                    "node_type": "event",
                    "name": "Article publication",
                    "attributes": {"type": "media"},
                    "timestamp": pub_date,
                    "source_sentence": "Article was published."
                }
            ]
        nodes_json = json.dumps(nodes, indent=2)

        # Round 4: Node Deduplication
        r4p = NODE_DEDUPLICATION_PROMPT.format(nodes=nodes_json, publication_date=pub_date)
        r4r = self._get_llm_response(r4p, max_tokens=2500)
        prompt_chain.append({"round": 4, "name": "Node Deduplication", "prompt": r4p, "response": r4r})

        # Extract JSON with fallback to repair
        unique_nodes = self._extract_json(r4r)
        if not unique_nodes:
            unique_nodes = self._repair_json(r4r, "array")

        # Fallback if still no valid unique nodes
        if not unique_nodes or not isinstance(unique_nodes, list):
            self.logger.warning(f"Failed to extract unique nodes from round 4. Using original nodes as fallback.")
            unique_nodes = self._add_ids_to_nodes(nodes)
        unique_nodes_json = json.dumps(unique_nodes, indent=2)

        # Round 5: Edge Extraction
        essential_facts = self._extract_essential_facts(facts)
        essential_facts_json = json.dumps(essential_facts, indent=2)

        r5p = EDGE_EXTRACTION_PROMPT.format(nodes=unique_nodes_json, facts=essential_facts_json)
        r5r = self._get_llm_response(r5p, max_tokens=3000)
        prompt_chain.append({"round": 5, "name": "Edge Extraction", "prompt": r5p, "response": r5r})

        # Extract JSON with fallback to repair
        edges = self._extract_json(r5r)
        if not edges:
            edges = self._repair_json(r5r, "array")

        # Ensure valid edges
        if not edges or not isinstance(edges, list):
            self.logger.warning(f"Failed to extract edges from round 5. Using empty edges list.")
            edges = []
        edges_json = json.dumps(edges, indent=2)

        # Round 6: Edge Enrichment
        r6p = EDGE_ENRICHMENT_PROMPT.format(edges=edges_json, publication_date=pub_date)
        r6r = self._get_llm_response(r6p, max_tokens=3000)
        prompt_chain.append({"round": 6, "name": "Edge Enrichment", "prompt": r6p, "response": r6r})

        # Extract JSON with fallback to repair
        enriched_edges = self._extract_json(r6r)
        if not enriched_edges:
            enriched_edges = self._repair_json(r6r, "array")

        # Fallback if still no valid enriched edges
        if not enriched_edges or not isinstance(enriched_edges, list):
            self.logger.warning(f"Failed to extract enriched edges from round 6. Enriching edges manually.")
            enriched_edges = self._enrich_edges(edges, pub_date)
        enriched_edges_json = json.dumps(enriched_edges, indent=2)

        # Round 7: Package Ingestion
        compact_nodes_json = self._create_compact_json(unique_nodes)
        compact_edges_json = self._create_compact_json(enriched_edges)

        r7p = PACKAGE_INGESTION_PROMPT.format(nodes=compact_nodes_json, edges=compact_edges_json)
        r7r = self._get_llm_response(r7p, max_tokens=3000)
        prompt_chain.append({"round": 7, "name": "Package Ingestion", "prompt": r7p, "response": r7r})

        # Extract JSON with fallback to repair
        package = self._extract_json(r7r)
        if not package:
            package = self._repair_json(r7r, "object")

        # Default package if still failing
        if not package or not isinstance(package, dict):
            self.logger.warning(f"Failed to extract package from round 7. Using default package.")
            package = {"nodes": unique_nodes, "edges": enriched_edges}

        return {
            "article": self._create_article_summary(article),
            "structured_facts": facts,
            "article_analysis": r2r,
            "prompt_chain": prompt_chain,
            "kg_payload": package,
            "timestamp": datetime.now().isoformat()
        }

    def _get_llm_response(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.1) -> str:
        try:
            # Append a clear instruction for JSON formatting to help models
            if "JSON" in prompt:
                prompt += "\n\nIMPORTANT: Format your entire response as valid JSON. Do not include explanations outside the JSON. The response should be properly formatted to be directly parsable by JSON.parse()."

            resp = self.llm_provider.complete(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
            text = resp.get("choices", [{}])[0].get("text", "")
            # Strip off any chain-of-thought in <think> tags
            if "</think>" in text:
                return text.split("</think>", 1)[1].strip()
            return text.strip()
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return f"ERROR: {e}"

    def _extract_json(self, text: str) -> Optional[Any]:
        """Basic JSON extraction with error handling"""
        if not text:
            return None

        try:
            # Try code blocks first
            code_blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
            if code_blocks:
                for block in code_blocks:
                    try:
                        return json.loads(block.strip())
                    except json.JSONDecodeError:
                        continue

            # Try the whole text
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

            # Try to find JSON structures
            candidates = re.findall(r'(?s)(\[.*?\]|\{.*?\})', text)
            for cand in candidates:
                try:
                    return json.loads(cand)
                except json.JSONDecodeError:
                    continue

            return None
        except Exception as e:
            return None

    def _repair_json(self, text: str, expected_format: str = "array") -> Optional[Any]:
        """Use LLM to repair malformed JSON"""
        if not text or len(text) < 10:
            return None

        self.logger.info("Attempting to repair malformed JSON with LLM")

        repair_prompt = f"""
Fix this malformed JSON to make it valid and parseable:

```
{text}
```

Return ONLY the fixed, valid JSON as a {expected_format} with no additional explanations.
"""

        repair_response = self._get_llm_response(repair_prompt, max_tokens=4000, temperature=0)

        # Try to parse the repaired JSON
        try:
            # Clean up response
            cleaned_response = re.sub(r'^```(?:json)?\s+', '', repair_response)
            cleaned_response = re.sub(r'\s+```$', '', cleaned_response)
            cleaned_response = cleaned_response.strip()

            return json.loads(cleaned_response)
        except Exception:
            return None

    def _add_ids_to_nodes(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add IDs to nodes as a fallback when deduplication fails"""
        result = []
        for i, node in enumerate(nodes, 1):
            node_copy = node.copy()
            node_copy['id'] = i
            result.append(node_copy)
        return result

    def _enrich_edges(self, edges: List[Dict[str, Any]], publication_date: str) -> List[Dict[str, Any]]:
        """Add enrichment fields to edges as a fallback"""
        result = []
        for edge in edges:
            edge_copy = edge.copy()
            # Add required fields if missing
            if 'severity' not in edge_copy:
                edge_copy['severity'] = 0.5  # Default medium severity
            if 'is_decayable' not in edge_copy:
                edge_copy['is_decayable'] = True  # Default to true
            if 'reasoning' not in edge_copy:
                edge_copy['reasoning'] = "Automatically enriched as fallback"
            if 'timestamp' not in edge_copy or edge_copy.get('timestamp') == 'N/A':
                edge_copy['timestamp'] = publication_date
            result.append(edge_copy)
        return result

    def _optimize_content(self, content: str, max_length: int = 4000) -> str:
        """Truncate content to preserve the most important parts"""
        if len(content) <= max_length:
            return content

        # Keep first 60% and last 40% of the content
        first_part_len = int(max_length * 0.6)
        last_part_len = max_length - first_part_len

        first_part = content[:first_part_len]
        last_part = content[-last_part_len:] if last_part_len > 0 else ""

        return first_part + "..." + last_part

    def _extract_entity_sections(self, content: str, facts: Dict[str, Any]) -> str:
        """Extract sections of content likely to contain entities"""
        if len(content) <= 4000:
            return content

        # Extract sections containing entities/events from facts
        important_sections = []
        key_phrases = []

        # Get event names
        for event in facts.get("events", []):
            if event.get("name"):
                key_phrases.append(event["name"])

        # Add direct quotes
        for quote in facts.get("direct_quotes", [])[:3]:
            if len(quote) > 15:
                key_phrases.append(quote[:40])

        # Add key facts
        for fact in facts.get("facts", [])[:5]:
            if len(fact) > 15:
                key_phrases.append(fact[:40])

        # Split content
        paragraphs = content.split('\n\n')

        # Keep paragraphs with key phrases
        for para in paragraphs:
            for phrase in key_phrases:
                if phrase and len(phrase) > 5 and phrase.lower() in para.lower():
                    important_sections.append(para)
                    break

        # Always include the first two paragraphs
        if len(paragraphs) > 0 and paragraphs[0] not in important_sections:
            important_sections.insert(0, paragraphs[0])
        if len(paragraphs) > 1 and paragraphs[1] not in important_sections:
            important_sections.insert(1, paragraphs[1])

        # Combine and truncate if needed
        result = "\n\n".join(important_sections)
        if len(result) > 4000:
            return self._optimize_content(result, 4000)

        return result

    def _extract_essential_facts(self, facts: Dict[str, Any]) -> Dict[str, Any]:
        """Extract only the most essential facts for edge extraction"""
        if not facts:
            return {}

        # Create a streamlined version
        essential = {
            "publication_date": facts.get("publication_date", ""),
            "events": []
        }

        # Add key events
        events = facts.get("events", [])
        essential_events = []

        for event in events[:5]:
            essential_events.append({
                "name": event.get("name", ""),
                "date": event.get("date", ""),
                "description": event.get("description", "")[:100]
            })

        essential["events"] = essential_events

        # Add key facts
        if "facts" in facts and facts["facts"]:
            essential["facts"] = facts["facts"][:3]

        return essential

    def _create_compact_json(self, data: List[Dict[str, Any]]) -> str:
        """Create a compact JSON representation"""
        if not data:
            return "[]"
        return json.dumps(data, separators=(',', ':'))

    def _create_article_summary(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of the article"""
        return {
            "title": article.get("title", ""),
            "url": article.get("url", ""),
            "source": article.get("source", ""),
            "published": article.get("published", ""),
            "bias_label": article.get("bias_label", ""),
            "content_length": len(article.get("content", "")),
            "id": article.get("id", f"article_{hash(article.get('title', ''))}")
        }