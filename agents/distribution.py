"""
Night_watcher Distribution Planner Agent
Agent for planning distribution strategies for counter-narratives.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from agents.base import Agent, LLMProvider


class DistributionPlanner(Agent):
    """Agent for planning distribution strategies for counter-narratives"""

    def __init__(self, llm_provider: LLMProvider):
        """Initialize with LLM provider"""
        super().__init__(llm_provider, name="DistributionPlanner")

    def create_distribution_strategy(self, counter_narratives: Dict[str, Any]) -> str:
        """
        Generate a distribution strategy for counter-narratives.

        Args:
            counter_narratives: Counter-narratives data

        Returns:
            A distribution strategy as text
        """
        # Extract article info and demographics
        article_title = counter_narratives.get("article_title", "")
        demographics = [cn.get("demographic", "") for cn in counter_narratives.get("counter_narratives", [])]

        prompt = f"""
        Create a strategic distribution plan for counter-narratives about this article:
        "{article_title}"

        For the following demographics: {', '.join(demographics)}

        Provide a detailed plan that includes:

        1. PLATFORM SELECTION: Which specific platforms would reach each demographic effectively
           - For each demographic, list 2-3 specific platforms or channels
           - Explain why these are effective for this group

        2. TIMING STRATEGY: When to release content for maximum impact
           - Consider news cycles and optimal times for each platform
           - Recommend sequence (which audiences to target first)

        3. FORMAT ADAPTATION: How to adapt content for each platform
           - Platform-specific format recommendations (thread structure, image use, etc.)
           - Length and style adjustments needed

        4. AMPLIFICATION TACTICS: How to increase organic reach
           - Hashtag recommendations
           - Engagement strategies
           - Key influencers or accounts that might share (without manipulation)

        5. MEASUREMENT APPROACH: How to track effectiveness
           - Key metrics to monitor for each platform
           - Signs of successful narrative adoption
        """

        return self._call_llm(prompt, max_tokens=1500, temperature=0.5)

    def create_talking_points_package(self, counter_narratives: Dict[str, Any],
                                      target_demographic: str) -> str:
        """
        Create ready-to-use talking points package for a specific demographic.

        Args:
            counter_narratives: Counter-narratives data
            target_demographic: Target demographic ID

        Returns:
            A talking points package as text
        """
        # Find the counter-narrative for this demographic
        target_narrative = next((cn for cn in counter_narratives.get("counter_narratives", [])
                                 if cn.get("demographic") == target_demographic), None)

        if not target_narrative:
            self.logger.warning(f"No counter-narrative found for demographic: {target_demographic}")
            return f"No counter-narrative found for demographic: {target_demographic}"

        article_title = counter_narratives.get("article_title", "")
        article_source = counter_narratives.get("source", "")

        # Check if content is available
        if "error" in target_narrative or "content" not in target_narrative:
            self.logger.warning(f"Content not available for {target_demographic}")
            return f"Error: Content not available for this demographic"

        prompt = f"""
        Create a complete "talking points package" ready for distribution to {target_demographic} audiences
        regarding this article: "{article_title}" from {article_source}

        Use this counter-narrative as your foundation:

        {target_narrative.get('content', 'No content available')}

        Create a comprehensive package including:

        1. CONVERSATION STARTERS: Natural ways to bring up this topic in conversation

        2. SOUNDBITES: 3-5 memorable, quotable phrases that capture key points (30 words or less each)

        3. COMMON OBJECTIONS: Anticipated pushback and effective responses

        4. SUPPORTING DATA: Key facts and statistics that support the counter-narrative

        5. VISUAL ELEMENTS: Descriptions of graphics, memes, or visuals that would effectively convey this message

        6. PERSONAL CONNECTION: How to relate this issue to personal experiences and values

        7. CALL TO ACTION: Specific, concrete next steps for engaged individuals

        Format this as a complete, ready-to-use document that could be shared with communicators.
        """

        return self._call_llm(prompt, max_tokens=1500, temperature=0.6)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process counter-narratives to create distribution plans.

        Args:
            input_data: Input data containing counter-narratives

        Returns:
            Distribution plans and talking points packages
        """
        counter_narratives_list = input_data.get("counter_narratives", [])
        results = []

        for counter_narratives in counter_narratives_list:
            article_title = counter_narratives.get("article_title", "")

            # Create distribution strategy
            self.logger.info(f"Creating distribution strategy for '{article_title}'...")
            distribution_strategy = self.create_distribution_strategy(counter_narratives)

            # Create talking points packages for key demographics
            target_demographics = ["progressive", "conservative"]  # Could be configurable
            talking_points = {}

            for demo in target_demographics:
                self.logger.info(f"Creating {demo} talking points...")
                talking_points[demo] = self.create_talking_points_package(counter_narratives, demo)

            result = {
                "article_title": article_title,
                "distribution_strategy": distribution_strategy,
                "talking_points": talking_points,
                "timestamp": datetime.now().isoformat()
            }

            results.append(result)

        return {"distribution_plans": results}