"""
Night_watcher Content Analyzer Agent
Agent for analyzing articles for divisive content.
"""

import re
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from .base import Agent, LLMProvider
from ..utils.text import truncate_text


class ContentAnalyzer(Agent):
    """Agent for analyzing articles for divisive content"""

    def __init__(self, llm_provider: LLMProvider):
        """Initialize with LLM provider"""
        super().__init__(llm_provider, name="ContentAnalyzer")

    def analyze_article(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a single article for divisive content.

        Args:
            article_data: Article data to analyze

        Returns:
            Analysis result
        """
        # Trim content if it's too long
        content = article_data.get('content', '')
        content = truncate_text(content, max_length=6000)

        prompt = f"""
        Analyze this news article for potential divisive framing, propaganda techniques, or misleading information.

        TITLE: {article_data['title']}
        SOURCE: {article_data['source']} (Bias: {article_data.get('bias_label', 'Unknown')})
        CONTENT:
        {content}

        Provide a detailed analysis with the following sections:

        1. MAIN TOPICS: What are the key topics of this article?

        2. FRAMING: How is the issue framed? What perspective is emphasized? Identify specific framing techniques used.

        3. EMOTIONAL TRIGGERS: What emotions does this article attempt to evoke? Identify specific words or phrases designed to trigger emotional responses.

        4. DIVISIVE ELEMENTS: Are there elements that could increase political/social division? How does this content potentially polarize readers?

        5. MISSING CONTEXT: What important context is omitted that would give readers a more complete understanding?

        6. MANIPULATION TECHNIQUES: Identify any of these techniques if present:
           - Appeal to fear or outrage
           - False equivalence
           - Cherry-picking of facts
           - Ad hominem attacks
           - Straw man arguments
           - Bandwagon appeal
           - Black-and-white fallacy

        7. MANIPULATION SCORE: Rate from 1-10 how manipulative this content appears to be, with 1 being objective reporting and 10 being highly manipulative propaganda.
           Explain your rating.
        """

        self.logger.info(f"Analyzing article: {article_data['title']}")
        analysis = self._call_llm(prompt, max_tokens=1000, temperature=0.1, stop=["User:", "\n\n\n"])

        return {
            "article": article_data,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }

    def extract_key_elements(self, analysis: str) -> Dict[str, Any]:
        """
        Extract structured key elements from an analysis.

        Args:
            analysis: Analysis text to extract from

        Returns:
            Dict with structured elements
        """
        prompt = f"""
        Extract the key elements from this article analysis in a structured format.

        ANALYSIS:
        {analysis}

        Extract and return ONLY the following in JSON format:
        {{
            "main_topics": ["topic1", "topic2", ...],
            "frames": ["frame1", "frame2", ...],
            "emotional_triggers": ["emotion1", "emotion2", ...],
            "divisive_elements": ["element1", "element2", ...],
            "manipulation_techniques": ["technique1", "technique2", ...],
            "manipulation_score": 0-10
        }}

        Provide only valid JSON with no explanations or extra text.
        """

        result = self._call_llm(prompt, max_tokens=800, temperature=0.1)

        try:
            # Find the JSON part (in case there's extra text)
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                result = json_match.group(0)

            return json.loads(result)
        except Exception as e:
            self.logger.error(f"Error parsing extracted elements: {str(e)}")
            return {
                "error": f"Failed to extract elements: {str(e)}",
                "raw_result": result
            }

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process articles for analysis.

        Args:
            input_data: Dict with 'articles' key containing articles to analyze

        Returns:
            Dict with 'analyses' key containing analysis results
        """
        articles = input_data.get("articles", [])
        results = []

        for article in articles:
            analysis = self.analyze_article(article)
            results.append(analysis)

        return {"analyses": results}