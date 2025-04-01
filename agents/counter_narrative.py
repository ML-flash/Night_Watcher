"""
Night_watcher Counter Narrative Generator Agent
Agent for generating counter-narratives to divisive content.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from .base import Agent, LLMProvider
from ..utils.text import truncate_text, extract_manipulation_score


class CounterNarrativeGenerator(Agent):
    """Agent for generating counter-narratives to divisive content"""

    def __init__(self, llm_provider: LLMProvider):
        """Initialize with LLM provider and demographics"""
        super().__init__(llm_provider, name="CounterNarrativeGenerator")
        self.demographics = self._load_demographics()

    def _load_demographics(self) -> List[Dict[str, Any]]:
        """Load demographic groups and their core values"""
        return [
            {"id": "progressive",
             "values": ["equality", "social justice", "collective welfare", "change", "diversity", "inclusion"]},
            {"id": "moderate_left",
             "values": ["pragmatism", "incremental progress", "compromise", "institutions", "reform", "balance"]},
            {"id": "moderate_right",
             "values": ["tradition", "individual liberty", "fiscal responsibility", "stability", "order",
                        "meritocracy"]},
            {"id": "conservative",
             "values": ["tradition", "faith", "patriotism", "security", "family values", "individualism"]},
            {"id": "libertarian",
             "values": ["individual freedom", "limited government", "self-reliance", "markets",
                        "personal responsibility"]}
        ]

    def generate_for_demographic(self, article: Dict[str, Any], analysis: str,
                                 demographic: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate counter-narrative for a specific demographic.

        Args:
            article: Article data
            analysis: Analysis text
            demographic: Demographic info

        Returns:
            Counter-narrative result
        """
        # Truncate content if too long
        content = article.get('content', '')
        content = truncate_text(content, max_length=3000)

        prompt = f"""
        Generate a counter-narrative for a potentially divisive article that will resonate with {demographic["id"]} readers while reducing polarization.

        ARTICLE TITLE: {article['title']}

        ARTICLE CONTENT:
        {content}

        ANALYSIS OF DIVISIVE ELEMENTS:
        {analysis}

        TARGET DEMOGRAPHIC: {demographic["id"]}
        CORE VALUES: {', '.join(demographic["values"])}

        Create a counter-narrative that:
        1. Appeals to the core values of this demographic
        2. Reduces distrust toward the "other side"
        3. Frames the issue in ways that could build bridges rather than walls
        4. Is factual and truthful
        5. Addresses concerns this demographic has but connects to universal values

        Your response should include:

        HEADLINE: An attention-grabbing alternative headline (5-10 words)

        KEY MESSAGE: The core alternative framing (1-2 sentences)

        TALKING POINTS:
        - Point 1
        - Point 2
        - Point 3
        - Point 4
        - Point 5

        CALL TO ACTION: What this demographic should do that builds bridges

        MESSAGING CHANNEL: Where this message would be most effective (specific media outlets, platforms, influencers)
        """

        self.logger.info(f"Generating narrative for {demographic['id']} demographic...")
        content = self._call_llm(prompt, max_tokens=1000, temperature=0.7)

        return {
            "demographic": demographic["id"],
            "content": content,
            "timestamp": datetime.now().isoformat()
        }

    def generate_bridging_content(self, article: Dict[str, Any], analysis: str,
                                  opposing_groups: List[str] = ["progressive", "conservative"]) -> Dict[str, Any]:
        """
        Generate content designed to bridge between opposing groups.

        Args:
            article: Article data
            analysis: Analysis text
            opposing_groups: List of opposing demographic groups

        Returns:
            Bridging content result
        """
        # Truncate content if too long
        content = article.get('content', '')
        content = truncate_text(content, max_length=3000)

        prompt = f"""
        Create a "bridging narrative" that could appeal to BOTH {opposing_groups[0]} AND {opposing_groups[1]} audiences
        regarding this potentially divisive topic.

        ARTICLE TITLE: {article['title']}

        ARTICLE SUMMARY:
        {content}

        ANALYSIS OF DIVISIVE ELEMENTS:
        {analysis}

        Your task is to find the shared values and concerns between these opposing groups and create content that:

        1. Identifies legitimate concerns from BOTH perspectives
        2. Finds the underlying shared values (e.g., family security, fairness, prosperity)
        3. Reframes the issue around these shared values
        4. Proposes solutions or approaches that could satisfy core needs of both groups
        5. Uses language that avoids triggering partisan reactions

        Provide:

        UNIFYING HEADLINE: A headline appealing to both groups

        SHARED CONCERNS: What both groups actually worry about in this situation

        COMMON GROUND: The underlying shared values at stake

        BRIDGE NARRATIVE: A 2-3 paragraph explanation that acknowledges both perspectives while finding common purpose

        CONSTRUCTIVE NEXT STEPS: Actions that would address concerns from both perspectives
        """

        self.logger.info(f"Generating bridging content between {opposing_groups}...")
        content = self._call_llm(prompt, max_tokens=1200, temperature=0.6)

        return {
            "bridging_groups": opposing_groups,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate counter-narratives for analyzed articles.

        Args:
            input_data: Dict with 'analyses' and 'manipulation_threshold' keys

        Returns:
            Dict with 'counter_narratives' containing generated narratives
        """
        analyses = input_data.get("analyses", [])
        manipulation_threshold = input_data.get("manipulation_threshold", 6)
        results = []

        for analysis_result in analyses:
            if "error" in analysis_result:
                self.logger.warning(f"Skipping analysis with error: {analysis_result.get('error')}")
                continue

            # Extract manipulation score
            manipulation_score = extract_manipulation_score(analysis_result["analysis"])

            if manipulation_score >= manipulation_threshold:
                article = analysis_result["article"]
                analysis = analysis_result["analysis"]

                # Generate counter-narratives for all demographics
                counter_narratives = []
                for demo in self.demographics:
                    narrative = self.generate_for_demographic(article, analysis, demo)
                    counter_narratives.append(narrative)

                # Generate bridging content
                bridging_content = self.generate_bridging_content(
                    article, analysis, opposing_groups=["progressive", "conservative"]
                )

                result = {
                    "article_title": article["title"],
                    "source": article["source"],
                    "url": article.get("url", ""),
                    "counter_narratives": counter_narratives,
                    "bridging_content": bridging_content
                }

                results.append(result)

        return {"counter_narratives": results}