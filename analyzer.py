"""
Night_watcher Content Analyzer
Multi-round prompting for authoritarian pattern analysis.
"""

import json
import logging
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# Import prompts
from prompts import (
    FACT_EXTRACTION_PROMPT,
    ARTICLE_ANALYSIS_PROMPT,
    NODE_EXTRACTION_PROMPT,
    NODE_DEDUPLICATION_PROMPT,
    EDGE_EXTRACTION_PROMPT,
    EDGE_ENRICHMENT_PROMPT,
    PACKAGE_INGESTION_PROMPT,
    MANIPULATION_SCORE_PROMPT,
    AUTHORITARIAN_ANALYSIS_PROMPT
)

logger = logging.getLogger(__name__)


class ContentAnalyzer:
    """Analyzer for political content with multi-round prompting to extract structured data."""

    def __init__(self, llm_provider):
        """
        Initialize with an LLM provider.

        Args:
            llm_provider: LLM provider instance for text completion
        """
        self.llm_provider = llm_provider
        self.logger = logging.getLogger("ContentAnalyzer")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a batch of articles with multi-round analysis.

        Args:
            input_data: Dictionary with:
                - 'articles': List of article dictionaries
                - 'document_ids': Optional list of document IDs

        Returns:
            Dictionary with analysis results
        """
        articles = input_data.get("articles", [])
        doc_ids = input_data.get("document_ids", [])

        if not articles:
            self.logger.warning("No articles provided for analysis")
            return {"analyses": []}

        analyses = []
        for i, article in enumerate(articles):
            # Associate document ID if available
            doc_id = doc_ids[i] if i < len(doc_ids) else None
            if doc_id:
                article["document_id"] = doc_id

            # Run multi-round analysis
            analysis = self._analyze_content_multi_round(article)
            analyses.append(analysis)

        self.logger.info(f"Completed analysis of {len(analyses)} articles")
        return {"analyses": analyses}

    def _analyze_content_multi_round(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an article with multi-round prompting.

        Args:
            article: Article data dictionary

        Returns:
            Analysis results dictionary
        """
        title = article.get("title", "Untitled")
        content = article.get("content", "")

        # Optimize content length for initial rounds
        core_content = self._optimize_content(content, 4000)

        self.logger.info(f"Analyzing: {title}")
        prompt_chain = []

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

        # Round 3: Manipulation Score Analysis
        r3p = MANIPULATION_SCORE_PROMPT.format(article_content=core_content)
        r3r = self._get_llm_response(r3p, max_tokens=2000)
        prompt_chain.append({"round": 3, "name": "Manipulation Score", "prompt": r3p, "response": r3r})

        # Extract manipulation score
        manipulation_score = self._extract_manipulation_score(r3r)

        # Round 4: Authoritarian Analysis
        r4p = AUTHORITARIAN_ANALYSIS_PROMPT.format(article_content=core_content)
        r4r = self._get_llm_response(r4p, max_tokens=2500)
        prompt_chain.append({"round": 4, "name": "Authoritarian Analysis", "prompt": r4p, "response": r4r})

        # Extract authoritarian indicators and concern level
        authoritarian_indicators, concern_level = self._extract_authoritarian_data(r4r)

        # Knowledge Graph population rounds

        # Round 5: Node Extraction
        r5p = NODE_EXTRACTION_PROMPT.format(article_content=content)
        r5r = self._get_llm_response(r5p, max_tokens=3000)
        prompt_chain.append({"round": 5, "name": "Node Extraction", "prompt": r5p, "response": r5r})

        # Extract JSON with fallback to repair
        nodes = self._extract_json(r5r)
        if not nodes:
            nodes = self._repair_json(r5r, "array")

        # Ensure we have valid nodes
        if not nodes or not isinstance(nodes, list):
            self.logger.warning(f"Failed to extract nodes JSON from round 5 for: {title}")
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

        # Round 6: Node Deduplication
        r6p = NODE_DEDUPLICATION_PROMPT.format(nodes=nodes_json, publication_date=pub_date)
        r6r = self._get_llm_response(r6p, max_tokens=2500)
        prompt_chain.append({"round": 6, "name": "Node Deduplication", "prompt": r6p, "response": r6r})

        # Extract JSON with fallback to repair
        unique_nodes = self._extract_json(r6r)
        if not unique_nodes:
            unique_nodes = self._repair_json(r6r, "array")

        # Fallback if still no valid unique nodes
        if not unique_nodes or not isinstance(unique_nodes, list):
            self.logger.warning(f"Failed to extract unique nodes from round 6. Using original nodes as fallback.")
            unique_nodes = self._add_ids_to_nodes(nodes)
        unique_nodes_json = json.dumps(unique_nodes, indent=2)

        # Round 7: Edge Extraction
        r7p = EDGE_EXTRACTION_PROMPT.format(nodes=unique_nodes_json, facts=facts_json)
        r7r = self._get_llm_response(r7p, max_tokens=3000)
        prompt_chain.append({"round": 7, "name": "Edge Extraction", "prompt": r7p, "response": r7r})

        # Extract JSON with fallback to repair
        edges = self._extract_json(r7r)
        if not edges:
            edges = self._repair_json(r7r, "array")

        # Ensure valid edges
        if not edges or not isinstance(edges, list):
            self.logger.warning(f"Failed to extract edges from round 7. Using empty edges list.")
            edges = []
        edges_json = json.dumps(edges, indent=2)

        # Round 8: Edge Enrichment
        r8p = EDGE_ENRICHMENT_PROMPT.format(edges=edges_json, publication_date=pub_date)
        r8r = self._get_llm_response(r8p, max_tokens=3000)
        prompt_chain.append({"round": 8, "name": "Edge Enrichment", "prompt": r8p, "response": r8r})

        # Extract JSON with fallback to repair
        enriched_edges = self._extract_json(r8r)
        if not enriched_edges:
            enriched_edges = self._repair_json(r8r, "array")

        # Fallback if still no valid enriched edges
        if not enriched_edges or not isinstance(enriched_edges, list):
            self.logger.warning(f"Failed to extract enriched edges from round 8. Enriching edges manually.")
            enriched_edges = self._enrich_edges(edges, pub_date)
        enriched_edges_json = json.dumps(enriched_edges, indent=2)

        # Round 9: Package Ingestion
        r9p = PACKAGE_INGESTION_PROMPT.format(nodes=unique_nodes_json, edges=enriched_edges_json)
        r9r = self._get_llm_response(r9p, max_tokens=3000)
        prompt_chain.append({"round": 9, "name": "Package Ingestion", "prompt": r9p, "response": r9r})

        # Extract JSON with fallback to repair
        kg_package = self._extract_json(r9r)
        if not kg_package:
            kg_package = self._repair_json(r9r, "object")

        # Default package if still failing
        if not kg_package or not isinstance(kg_package, dict):
            self.logger.warning(f"Failed to extract package from round 9. Using default package.")
            kg_package = {"nodes": unique_nodes, "edges": enriched_edges}

        # Create the final analysis output
        analysis = {
            "article": self._create_article_summary(article),
            "facts": facts,
            "article_analysis": r2r,
            "manipulation_score": manipulation_score,
            "authoritarian_indicators": authoritarian_indicators,
            "concern_level": concern_level,
            "kg_payload": kg_package,
            "prompt_chain": prompt_chain,
            "timestamp": datetime.now().isoformat()
        }

        return analysis

    def _get_llm_response(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.1) -> str:
        """
        Get a response from the LLM with error handling.

        Args:
            prompt: Prompt text
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation

        Returns:
            LLM response text
        """
        try:
            # Append a clear instruction for JSON formatting to help models
            if "JSON" in prompt:
                prompt += "\n\nIMPORTANT: Format your entire response as valid JSON. Do not include explanations outside the JSON. The response should be properly formatted to be directly parsable by JSON.parse()."

            resp = self.llm_provider.complete(prompt=prompt, max_tokens=max_tokens, temperature=temperature)

            if "error" in resp:
                self.logger.error(f"LLM error: {resp['error']}")
                return f"ERROR: {resp.get('error')}"

            text = resp.get("choices", [{}])[0].get("text", "")

            # Strip off any chain-of-thought in <think> tags
            if "</think>" in text:
                return text.split("</think>", 1)[1].strip()

            return text.strip()
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return f"ERROR: {e}"

    def _extract_json(self, text: str) -> Optional[Any]:
        """
        Extract JSON from text with error handling.

        Args:
            text: Text containing JSON

        Returns:
            Parsed JSON object or None if failed
        """
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
            self.logger.error(f"Error extracting JSON: {e}")
            return None

    def _repair_json(self, text: str, expected_format: str = "array") -> Optional[Any]:
        """
        Use LLM to repair malformed JSON.

        Args:
            text: Text with potentially malformed JSON
            expected_format: Expected format ('array' or 'object')

        Returns:
            Repaired JSON object or None if still invalid
        """
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
            cleaned_response = re.sub(r'\s+```', '', cleaned_response)
            cleaned_response = cleaned_response.strip()

            return json.loads(cleaned_response)
        except Exception:
            return None

    def _add_ids_to_nodes(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add IDs to nodes as a fallback when deduplication fails.

        Args:
            nodes: List of node objects

        Returns:
            Nodes with added IDs
        """
        result = []
        for i, node in enumerate(nodes, 1):
            node_copy = node.copy()
            node_copy['id'] = i
            result.append(node_copy)
        return result

    def _enrich_edges(self, edges: List[Dict[str, Any]], publication_date: str) -> List[Dict[str, Any]]:
        """
        Add enrichment fields to edges as a fallback.

        Args:
            edges: List of edge objects
            publication_date: Default publication date

        Returns:
            Edges with enrichment fields
        """
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
        """
        Optimize content length for LLM processing.

        Args:
            content: Original content
            max_length: Maximum desired length

        Returns:
            Optimized content
        """
        if len(content) <= max_length:
            return content

        # Keep first 60% and last 40% of the content
        first_part_len = int(max_length * 0.6)
        last_part_len = max_length - first_part_len

        first_part = content[:first_part_len]
        last_part = content[-last_part_len:] if last_part_len > 0 else ""

        return first_part + "..." + last_part

    def _extract_manipulation_score(self, text: str) -> int:
        """
        Extract manipulation score from analysis text.

        Args:
            text: Analysis text

        Returns:
            Manipulation score (1-10) or 0 if not found
        """
        try:
            match = re.search(r"MANIPULATION SCORE:\s*(\d+)", text)
            if match:
                score = int(match.group(1))
                # Ensure the score is within range
                return max(1, min(10, score))
            return 0
        except Exception as e:
            self.logger.error(f"Error extracting manipulation score: {e}")
            return 0

    def _extract_authoritarian_data(self, text: str) -> Tuple[List[str], str]:
        """
        Extract authoritarian indicators and concern level.

        Args:
            text: Analysis text

        Returns:
            Tuple of (indicators list, concern level)
        """
        indicators = []
        concern_level = "Unknown"

        try:
            # Extract indicators
            indicators_match = re.search(r"AUTHORITARIAN INDICATORS:\s*(.*?)(?:\n|$)", text)
            if indicators_match:
                indicators_text = indicators_match.group(1).strip()
                if indicators_text.lower() != "none detected":
                    # Split by common list separators
                    indicators = re.split(r'[,;]|\n-', indicators_text)
                    indicators = [i.strip() for i in indicators if i.strip()]

            # Extract concern level
            concern_match = re.search(r"CONCERN LEVEL:\s*(None|Low|Moderate|High|Very High)", text)
            if concern_match:
                concern_level = concern_match.group(1).strip()

        except Exception as e:
            self.logger.error(f"Error extracting authoritarian data: {e}")

        return indicators, concern_level

    def _create_article_summary(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a summary of the article for the analysis output.

        Args:
            article: Article data

        Returns:
            Article summary dictionary
        """
        return {
            "title": article.get("title", ""),
            "url": article.get("url", ""),
            "source": article.get("source", ""),
            "published": article.get("published", ""),
            "bias_label": article.get("bias_label", ""),
            "content_length": len(article.get("content", "")),
            "document_id": article.get("document_id", "")
        }