# analyzer.py
"""
Night_watcher Content Analyzer with Multi-Round Prompting and KG Pipeline
Optimized to minimize context size by passing only necessary JSON between rounds.
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
        if len(content) > 6000:
            content = content[:6000] + "..."

        self.logger.info(f"Analyzing: {title}")
        prompt_chain: List[Dict[str, Any]] = []

        # Round 1: Fact Extraction (uses raw article)
        r1p = FACT_EXTRACTION_PROMPT.format(article_content=content)
        r1r = self._get_llm_response(r1p)
        prompt_chain.append({"round":1, "name":"Fact Extraction", "prompt":r1p, "response":r1r})
        raw_facts = self._extract_json(r1r) or {}
        facts = raw_facts if isinstance(raw_facts, dict) else {}
        facts_json = json.dumps(facts, indent=2)
        pub_date = facts.get("publication_date", datetime.now().strftime("%Y-%m-%d"))

        # Round 2: Article Analysis (uses raw article)
        r2p = ARTICLE_ANALYSIS_PROMPT.format(article_content=content)
        r2r = self._get_llm_response(r2p)
        prompt_chain.append({"round":2, "name":"Article Analysis", "prompt":r2p, "response":r2r})

        # Subsequent rounds pass only JSON to minimize context

        # Round 3: Node Extraction
        # Pass both article content and structured facts to accommodate prompt placeholders
        r3p = NODE_EXTRACTION_PROMPT.format(article_content=content, structured_facts=facts_json)
        r3r = self._get_llm_response(r3p)
        prompt_chain.append({"round":3, "name":"Node Extraction", "prompt":r3p, "response":r3r})
        nodes = self._extract_json(r3r) or []
        nodes_json = json.dumps(nodes, indent=2)(nodes, indent=2)

        # Round 4: Node Deduplication
        r4p = NODE_DEDUPLICATION_PROMPT.format(nodes=nodes_json, publication_date=pub_date)
        r4r = self._get_llm_response(r4p)
        prompt_chain.append({"round":4, "name":"Node Deduplication", "prompt":r4p, "response":r4r})
        unique_nodes = self._extract_json(r4r) or []
        unique_nodes_json = json.dumps(unique_nodes, indent=2)

        # Round 5: Edge Extraction
        r5p = EDGE_EXTRACTION_PROMPT.format(nodes=unique_nodes_json, facts=facts_json)
        r5r = self._get_llm_response(r5p)
        prompt_chain.append({"round":5, "name":"Edge Extraction", "prompt":r5p, "response":r5r})
        edges = self._extract_json(r5r) or []
        edges_json = json.dumps(edges, indent=2)

        # Round 6: Edge Enrichment
        r6p = EDGE_ENRICHMENT_PROMPT.format(edges=edges_json, publication_date=pub_date)
        r6r = self._get_llm_response(r6p)
        prompt_chain.append({"round":6, "name":"Edge Enrichment", "prompt":r6p, "response":r6r})
        enriched_edges = self._extract_json(r6r) or []
        enriched_edges_json = json.dumps(enriched_edges, indent=2)

        # Round 7: Package Ingestion
        r7p = PACKAGE_INGESTION_PROMPT.format(nodes=unique_nodes_json, edges=enriched_edges_json)
        r7r = self._get_llm_response(r7p)
        prompt_chain.append({"round":7, "name":"Package Ingestion", "prompt":r7p, "response":r7r})
        package = self._extract_json(r7r) or {"nodes": unique_nodes, "edges": enriched_edges}

        return {
            "article": article,
            "structured_facts": facts,
            "article_analysis": r2r,
            "prompt_chain": prompt_chain,
            "kg_payload": package,
            "timestamp": datetime.now().isoformat()
        }

    def _get_llm_response(self, prompt: str) -> str:
        try:
            resp = self.llm_provider.complete(prompt=prompt, max_tokens=2000, temperature=0.2)
            text = resp.get("choices", [{}])[0].get("text", "")
            # Strip off any chain-of-thought in <think> tags
            if "</think>" in text:
                return text.split("</think>", 1)[1].strip()
            return text.strip()
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return f"ERROR: {e}"

    def _extract_json(self, text: str) -> Optional[Any]:
        try:
            candidates = re.findall(r'(?s)(\[.*?\]|\{.*?\})', text)
            for cand in candidates:
                try:
                    return json.loads(cand)
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            logger.warning(f"JSON candidate extraction failed: {e}")
        logger.warning(f"JSON parsing failed for text: {text[:100]}")
        return None
