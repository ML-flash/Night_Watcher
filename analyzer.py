# Revised portion of analyzer.py that replaces old extraction logic
# Uses model-based fact extraction with embedded publication date

from prompts import FACT_EXTRACTION_PROMPT, ARTICLE_ANALYSIS_PROMPT, RELATIONSHIP_EXTRACTION_PROMPT
import json
import re
import logging
from datetime import datetime
from typing import Dict, Any

class ContentAnalyzer:
    def __init__(self, llm_provider):
        self.llm_provider = llm_provider
        self.logger = logging.getLogger("ContentAnalyzer")
        self.context_history = {}

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        articles = input_data.get("articles", [])
        if not articles:
            self.logger.warning("No articles provided for analysis")
            return {"analyses": [], "authoritarian_analyses": [], "kg_analyses": []}

        self.logger.info(f"Starting analysis of {len(articles)} articles")

        analyses = []
        authoritarian_analyses = []
        kg_analyses = []

        for article in articles:
            analysis = self.analyze_content(article)
            analyses.append(analysis)

            auth_analysis = self.analyze_authoritarian_patterns(article)
            authoritarian_analyses.append(auth_analysis)

            kg_analysis = self.analyze_content_for_knowledge_graph(article)
            kg_analyses.append(kg_analysis)

        self.logger.info(f"Completed analysis of {len(articles)} articles")

        return {
            "analyses": analyses,
            "authoritarian_analyses": authoritarian_analyses,
            "kg_analyses": kg_analyses
        }

    def extract_facts(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        content = article_data.get("content", "")
        prompt = FACT_EXTRACTION_PROMPT.format(article_content=content[:7000])

        try:
            response = self.llm_provider.complete(prompt=prompt, max_tokens=1500, temperature=0.2)
            text = response.get("choices", [{}])[0].get("text", "")
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except Exception as e:
            self.logger.error(f"Fact extraction failed: {e}")
        return {}

    def analyze_content_for_knowledge_graph(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        structured_facts = self.extract_facts(article_data)
        publication_date = structured_facts.get("publication_date")

        if not publication_date:
            self.logger.warning("No publication date found in extracted facts.")
            publication_date = datetime.now().strftime("%Y-%m-%d")

        article_data["published"] = publication_date

        return {
            "article": article_data,
            "structured_facts": structured_facts,
            "timestamp": datetime.now().isoformat()
        }

    def analyze_content(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        title = article_data.get("title", "Untitled")
        content = article_data.get("content", "")
        source = article_data.get("source", "Unknown")
        bias_label = article_data.get("bias_label", "Unknown")

        prompt = ARTICLE_ANALYSIS_PROMPT.format(
            article_content=content[:7000]
        ).replace("{title}", title).replace("{source}", source).replace("{bias_label}", bias_label)

        try:
            response = self.llm_provider.complete(
                prompt=prompt,
                max_tokens=3000,
                temperature=0.3
            )
            analysis_text = response.get("choices", [{}])[0].get("text", "")
            return {
                "article": article_data,
                "analysis": analysis_text,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Content analysis failed: {e}")
            return {
                "article": article_data,
                "analysis": f"Error: {e}",
                "timestamp": datetime.now().isoformat()
            }

    def analyze_authoritarian_patterns(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        # Placeholder – can be expanded to reuse structured_facts for deeper KG extraction
        return {
            "article": article_data,
            "authoritarian_analysis": "[not yet implemented]",
            "structured_elements": {},
            "timestamp": datetime.now().isoformat()
        }
