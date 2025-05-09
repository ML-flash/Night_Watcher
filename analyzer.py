"""
Night_watcher – 7‑round LLM analysis (stand‑alone).

Keeps the original prompt chain intact:
  1. FACT_EXTRACTION_PROMPT
  2. ARTICLE_ANALYSIS_PROMPT
  3. NODE_EXTRACTION_PROMPT
  4. NODE_DEDUPLICATION_PROMPT
  5. EDGE_EXTRACTION_PROMPT
  6. EDGE_ENRICHMENT_PROMPT
  7. PACKAGE_INGESTION_PROMPT
"""

import json, logging, re
from datetime import datetime
from typing import Any, Dict, List, Optional

import prompts                       # the file you provided
from providers import initialize_llm_provider  # already in repo

logger = logging.getLogger("ContentAnalyzer")
JSON_RE = re.compile(r"(\{|\[).*(\}|\])", re.S)   # greedy – grabs first JSON blob


class ContentAnalyzer:
    def __init__(self, llm):
        self.llm = llm

    # ------------------------------------------------------------------ #
    # public API                                                         #
    # ------------------------------------------------------------------ #
    def analyze(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the 7‑round chain on one article dict that has
        at least {"title", "content"} (publication_date is optional
        and will be inferred from R1 if absent).
        """
        pc: List[Dict[str, str]] = []                 # prompt‑chain trace

        # Round 1 – objective facts --------------------------------------
        p1 = prompts.FACT_EXTRACTION_PROMPT.format(article_content=article["content"])
        r1 = self._ask(p1)
        pc.append(self._rec(1, "Fact Extraction", p1, r1))
        facts = self._json(r1) or {}
        pub_date = facts.get("publication_date", "N/A")

        # Round 2 – framing & manipulation -------------------------------
        p2 = prompts.ARTICLE_ANALYSIS_PROMPT.format(article_content=article["content"])
        r2 = self._ask(p2)
        pc.append(self._rec(2, "Framing Analysis", p2, r2))

        # Round 3 – raw node extraction ----------------------------------
        p3 = prompts.NODE_EXTRACTION_PROMPT.format(article_content=article["content"])
        r3 = self._ask(p3)
        pc.append(self._rec(3, "Node Extraction", p3, r3))
        nodes_raw = self._json(r3) or []

        # Round 4 – dedup / normalize ------------------------------------
        nodes_json_str = json.dumps(nodes_raw, ensure_ascii=False)
        p4 = prompts.NODE_DEDUPLICATION_PROMPT.format(
            nodes=nodes_json_str, publication_date=pub_date
        )
        r4 = self._ask(p4)
        pc.append(self._rec(4, "Node Deduplication", p4, r4))
        nodes = self._json(r4) or []

        # Round 5 – edge extraction --------------------------------------
        p5 = prompts.EDGE_EXTRACTION_PROMPT.format(
            nodes=json.dumps(nodes, ensure_ascii=False),
            facts=json.dumps(facts, ensure_ascii=False),
        )
        r5 = self._ask(p5)
        pc.append(self._rec(5, "Edge Extraction", p5, r5))
        edges_raw = self._json(r5) or []

        # Round 6 – edge enrichment --------------------------------------
        p6 = prompts.EDGE_ENRICHMENT_PROMPT.format(
            edges=json.dumps(edges_raw, ensure_ascii=False),
            publication_date=pub_date,
        )
        r6 = self._ask(p6)
        pc.append(self._rec(6, "Edge Enrichment", p6, r6))
        edges = self._json(r6) or []

        # Round 7 – package for ingestion --------------------------------
        p7 = prompts.PACKAGE_INGESTION_PROMPT.format(
            nodes=json.dumps(nodes, ensure_ascii=False),
            edges=json.dumps(edges, ensure_ascii=False),
        )
        r7 = self._ask(p7)
        pc.append(self._rec(7, "Package Ingestion", p7, r7))
        package = self._json(r7) or {}

        # bundle ---------------------------------------------------------
        return {
            "article": article,
            "facts": facts,
            "framing_analysis": r2,
            "nodes_raw": nodes_raw,
            "nodes": nodes,
            "edges_raw": edges_raw,
            "edges": edges,
            "package": package,
            "prompt_chain": pc,
            "ts": datetime.utcnow().isoformat(),
        }

    # ------------------------------------------------------------------ #
    # helpers                                                            #
    # ------------------------------------------------------------------ #
    def _ask(self, prompt: str, max_tokens: int = 3_000) -> str:
        resp = self.llm.complete(prompt, max_tokens=max_tokens, temperature=0.3)
        return resp.get("choices", [{}])[0].get("text", "")

    @staticmethod
    def _json(text: str) -> Optional[Any]:
        """Extract first JSON blob from text and parse it."""
        try:
            m = JSON_RE.search(text)
            return json.loads(m.group(0) if m else text)
        except Exception:
            logger.debug("⚠️  JSON parse failed")
            return None

    @staticmethod
    def _rec(round_no: int, name: str, prompt: str, resp: str) -> Dict[str, str]:
        return dict(round=round_no, name=name, prompt=prompt, response=resp)
