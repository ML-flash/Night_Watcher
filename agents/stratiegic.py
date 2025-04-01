"""
Night_watcher Strategic Messaging Agent
Generates targeted strategic content for different political orientations.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from .base import Agent, LLMProvider
from ..utils.text import truncate_text, extract_manipulation_score


class StrategicMessaging(Agent):
    """Agent for generating strategic messaging for different demographics"""

    def __init__(self, llm_provider: LLMProvider):
        """Initialize with LLM provider"""
        super().__init__(llm_provider, name="StrategicMessaging")

    def generate_right_wing_trust_erosion(self, article_analysis: Dict[str, Any]) -> str:
        """
        Generate content designed to reduce blind trust in authoritarian figures for right-leaning audiences.

        Args:
            article_analysis: Analysis data for an article

        Returns:
            Strategic messaging content as text
        """
        article = article_analysis.get("article", {})
        analysis = article_analysis.get("analysis", "")

        # Truncate content if too long
        content = article.get('content', '')
        content = truncate_text(content, max_length=3000)

        prompt = f"""
        Create messaging that would resonate with conservative/right-leaning audiences that subtly highlights 
        concerning authoritarian patterns while appealing to traditional conservative values of limited government, 
        constitutional principles, and individual liberty.

        ARTICLE CONTEXT:
        Title: {article.get('title', '')}
        Content: {content}

        ANALYSIS:
        {analysis}

        The goal is to craft messaging that:
        1. Is deeply respectful of conservative values and concerns
        2. Uses historical conservative principles to question current authoritarian tendencies
        3. Appeals to patriotism and constitutional fidelity
        4. Invokes respected conservative figures from history
        5. Distinguishes between supporting conservative policies vs. supporting unchecked authority
        6. Emphasizes conservative principles over personality cults

        Provide:

        HEADLINE: Attention-grabbing headline appealing to conservative principles

        FRAMING: How to position this concern in terms of traditional values

        HISTORICAL PARALLELS: Conservative historical figures who warned against similar situations

        KEY QUOTES: Statements from respected conservative figures supporting limited government

        MESSAGING CHANNELS: Where this message would be most effective with this audience
        """

        return self._call_llm(prompt, max_tokens=1000, temperature=0.7)

    def generate_left_wing_action_plan(self, article_analysis: Dict[str, Any]) -> str:
        """
        Generate content designed to encourage constructive action from left-leaning audiences.

        Args:
            article_analysis: Analysis data for an article

        Returns:
            Strategic messaging content as text
        """
        article = article_analysis.get("article", {})
        analysis = article_analysis.get("analysis", "")

        # Truncate content if too long
        content = article.get('content', '')
        content = truncate_text(content, max_length=3000)

        prompt = f"""
        Create messaging for progressive/left-leaning audiences that converts concern about authoritarian trends
        into constructive action that builds bridges rather than deepens division.

        ARTICLE CONTEXT:
        Title: {article.get('title', '')}
        Content: {content}

        ANALYSIS:
        {analysis}

        The goal is to craft messaging that:
        1. Acknowledges legitimate concerns about democratic backsliding
        2. Channels energy toward constructive engagement rather than demonization
        3. Emphasizes coalition-building including with moderate conservatives
        4. Focuses on systemic solutions rather than personality-based opposition
        5. Provides concrete, practical steps that build democratic resilience

        Provide:

        HEADLINE: Attention-grabbing headline emphasizing constructive action

        FRAMING: How to position this concern in terms of democratic values

        COALITION APPROACH: How to engage beyond typical progressive circles

        ACTION STEPS: Specific, practical actions individuals can take locally

        MESSAGING TONE: Guidelines for communicating that builds bridges rather than walls
        """

        return self._call_llm(prompt, max_tokens=1000, temperature=0.7)

    def generate_moderate_consensus_building(self, article_analysis: Dict[str, Any]) -> str:
        """
        Generate content designed to build consensus among moderate audiences across the political spectrum.

        Args:
            article_analysis: Analysis data for an article

        Returns:
            Strategic messaging content as text
        """
        article = article_analysis.get("article", {})
        analysis = article_analysis.get("analysis", "")

        # Truncate content if too long
        content = article.get('content', '')
        content = truncate_text(content, max_length=3000)

        prompt = f"""
        Create messaging that would resonate with moderate audiences across the political spectrum,
        emphasizing shared values and common ground while addressing concerns about polarization.

        ARTICLE CONTEXT:
        Title: {article.get('title', '')}
        Content: {content}

        ANALYSIS:
        {analysis}

        The goal is to craft messaging that:
        1. Identifies and emphasizes shared values that transcend political divides
        2. Avoids partisan trigger points while acknowledging legitimate concerns
        3. Presents a balanced perspective that respects diverse viewpoints
        4. Suggests concrete actions that could be agreed upon across moderate political perspectives
        5. Focuses on practical problem-solving rather than ideological purity

        Provide:

        HEADLINE: A balanced headline that would appeal across moderate political perspectives

        COMMON VALUES: Core values that unite moderates across the spectrum

        FRAMEWORK: A framework for discussing the issue that respects diverse viewpoints

        BALANCED APPROACH: How to acknowledge different concerns while finding common ground

        PRACTICAL STEPS: Specific actions that could gain support from diverse moderate audiences
        """

        return self._call_llm(prompt, max_tokens=1000, temperature=0.6)

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process article analyses to generate strategic messaging for different demographics.

        Args:
            input_data: Input data containing article analyses

        Returns:
            Strategic messaging content for different demographics
        """
        analyses = input_data.get("analyses", [])
        manipulation_threshold = input_data.get("manipulation_threshold", 6)
        results = []

        for analysis in analyses:
            if "error" in analysis:
                self.logger.warning(f"Skipping analysis with error: {analysis.get('error')}")
                continue

            # Check manipulation score
            manipulation_score = extract_manipulation_score(analysis["analysis"])

            if manipulation_score >= manipulation_threshold:
                article_title = analysis["article"]["title"]
                self.logger.info(f"Generating strategic messaging for '{article_title}'...")

                # Generate messaging for different demographics
                right_wing = self.generate_right_wing_trust_erosion(analysis)
                left_wing = self.generate_left_wing_action_plan(analysis)
                moderate = self.generate_moderate_consensus_building(analysis)

                result = {
                    "article_title": article_title,
                    "source": analysis["article"].get("source", ""),
                    "manipulation_score": manipulation_score,
                    "strategic_messaging": {
                        "right_wing": right_wing,
                        "left_wing": left_wing,
                        "moderate": moderate
                    },
                    "timestamp": datetime.now().isoformat()
                }

                results.append(result)

        return {"strategic_messages": results}