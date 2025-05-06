# analyzer.py
"""
Night_watcher Content Analyzer with Multi-Round Prompting and KG Pipeline
Module for analyzing political content and populating a knowledge graph schema.
"""

import logging
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
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
    """Analyzer for political content with multi-round prompting and KG node/edge extraction"""

    def __init__(self, llm_provider):
        """Initialize with an LLM provider"""
        self.llm_provider = llm_provider
        self.logger = logging.getLogger("ContentAnalyzer")

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a batch of articles, returning analyses and KG payloads.
        """
        articles = input_data.get("articles", [])
        if not articles:
            self.logger.warning("No articles provided for analysis")
            return {"analyses": []}

        self.logger.info(f"Starting analysis of {len(articles)} articles")
        analyses: List[Dict[str, Any]] = []
        for article in articles:
            analyses.append(self._analyze_content_multi_round(article))

        return {"analyses": analyses}

    def _analyze_content_multi_round(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform multi-round LLM prompts including KG extraction.
        """
        title = article.get("title", "Untitled")
        content = article.get("content", "")
        if len(content) > 6000:
            content = content[:6000] + "..."

        self.logger.info(f"Analyzing article: {title}")
        prompt_chain: List[Dict[str, Any]] = []

        # Round 1: Fact Extraction
        r1p = FACT_EXTRACTION_PROMPT.format(article_content=content)
        r1r = self._get_llm_response(r1p)
        prompt_chain.append({"round":1, "name":"Fact Extraction", "prompt":r1p, "response":r1r})
        facts = self._extract_json(r1r) or {}

        # Round 2: Article Analysis
        r2p = ARTICLE_ANALYSIS_PROMPT.format(article_content=content)
        r2r = self._get_llm_response(r2p)
        prompt_chain.append({"round":2, "name":"Article Analysis", "prompt":r2p, "response":r2r})

        # Prepare KG JSON strings
        facts_json = json.dumps(facts, indent=2)

        # Round 3: Node Extraction
        r3p = f"{NODE_EXTRACTION_PROMPT}\nCONTENT:\n{content}"
        r3r = self._get_llm_response(r3p)
        prompt_chain.append({"round":3, "name":"Node Extraction", "prompt":r3p, "response":r3r})
        nodes = self._extract_json(r3r) or []
        nodes_json = json.dumps(nodes, indent=2)

        # Round 4: Node Deduplication
        r4p = f"{NODE_DEDUPLICATION_PROMPT}\nNODES:\n{nodes_json}"
        r4r = self._get_llm_response(r4p)
        prompt_chain.append({"round":4, "name":"Node Deduplication", "prompt":r4p, "response":r4r})
        unique_nodes = self._extract_json(r4r) or []
        unique_nodes_json = json.dumps(unique_nodes, indent=2)

        # Round 5: Edge Extraction
        r5p = (f"{EDGE_EXTRACTION_PROMPT}\nNODES:\n{unique_nodes_json}\nFACTS:\n{facts_json}")
        r5r = self._get_llm_response(r5p)
        prompt_chain.append({"round":5, "name":"Edge Extraction", "prompt":r5p, "response":r5r})
        edges = self._extract_json(r5r) or []
        edges_json = json.dumps(edges, indent=2)

        # Round 6: Edge Enrichment
        r6p = f"{EDGE_ENRICHMENT_PROMPT}\nEDGES:\n{edges_json}"
        r6r = self._get_llm_response(r6p)
        prompt_chain.append({"round":6, "name":"Edge Enrichment", "prompt":r6p, "response":r6r})
        enriched_edges = self._extract_json(r6r) or []
        enriched_edges_json = json.dumps(enriched_edges, indent=2)

        # Round 7: Package for Ingestion
        r7p = (f"{PACKAGE_INGESTION_PROMPT}\nNODES:\n{unique_nodes_json}\nEDGES:\n{enriched_edges_json}")
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
        """
        Send prompt to LLM provider and return the text response.
        """
        try:
            resp = self.llm_provider.complete(
                prompt=prompt,
                max_tokens=2000,
                temperature=0.2
            )
            text = resp.get("choices", [{}])[0].get("text", "")
            if not text and "error" in resp:
                logger.error(f"LLM error: {resp['error']}")
                return f"ERROR: {resp['error']}"
            return text
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return f"ERROR: {e}"

    def _extract_json(self, text: str) -> Optional[Any]:
        """
        Extract JSON object/array from the response text.
        """
        try:
            match = re.search(r'(?s)(\[.*\]|\{.*\})', text)
            if match:
                return json.loads(match.group(0))
            return json.loads(text)
        except Exception as e:
            logger.warning(f"JSON parsing failed: {e}")
            return None
