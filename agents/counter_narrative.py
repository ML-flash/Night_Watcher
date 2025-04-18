"""
Night_watcher Counter Narrative Generator Agent
Agent for generating counter-narratives to divisive content and authoritarian rhetoric.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from agents.base import Agent, LLMProvider
from utils.text import truncate_text, extract_manipulation_score


class CounterNarrativeGenerator(Agent):
    """Agent for generating counter-narratives to divisive content and authoritarian rhetoric"""

    def __init__(self, llm_provider: LLMProvider):
        """Initialize with LLM provider and demographics"""
        super().__init__(llm_provider, name="CounterNarrativeGenerator")
        self.demographics = self._load_demographics()

    def _load_demographics(self) -> List[Dict[str, Any]]:
        """Load demographic groups and their core values"""
        return [
            {
                "id": "progressive",
                "values": ["equality", "social justice", "collective welfare", "change",
                          "diversity", "inclusion", "democratic participation"]
            },
            {
                "id": "moderate_left",
                "values": ["pragmatism", "incremental progress", "compromise", "institutions",
                          "reform", "balance", "civic engagement"]
            },
            {
                "id": "moderate_right",
                "values": ["tradition", "individual liberty", "fiscal responsibility",
                          "stability", "order", "meritocracy", "constitutional principles"]
            },
            {
                "id": "conservative",
                "values": ["tradition", "faith", "patriotism", "security", "family values",
                          "individualism", "constitutional originalism"]
            },
            {
                "id": "libertarian",
                "values": ["individual freedom", "limited government", "self-reliance",
                          "markets", "personal responsibility", "civil liberties"]
            }
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
        Generate a counter-narrative for a potentially divisive article that will resonate with {demographic["id"]} readers 
        while reducing polarization and strengthening democratic values.

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
        5. Addresses concerns this demographic has but connects to universal democratic values
        6. Highlights the importance of democratic institutions and norms regardless of policy positions
        7. Avoids demonizing any group while still clearly identifying threats to democratic governance

        Your response should include:

        HEADLINE: An attention-grabbing alternative headline (5-10 words)

        KEY MESSAGE: The core alternative framing (1-2 sentences)

        TALKING POINTS:
        - Point 1 (with emphasis on shared democratic values)
        - Point 2 (addressing specific concerns of this demographic)
        - Point 3 (with emphasis on protecting democratic institutions)
        - Point 4 (with a bridge to other political perspectives)
        - Point 5 (with emphasis on civic responsibility)

        CALL TO ACTION: What this demographic should do that builds bridges and strengthens democracy

        MESSAGING CHANNEL: Where this message would be most effective (specific media outlets, platforms, influencers)
        """

        self.logger.info(f"Generating narrative for {demographic['id']} demographic...")
        content = self._call_llm(prompt, max_tokens=1200, temperature=0.7)

        return {
            "demographic": demographic["id"],
            "content": content,
            "timestamp": datetime.now().isoformat()
        }

    def generate_authoritarian_response(self, article: Dict[str, Any],
                                        auth_analysis: str,
                                        demographic: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a response to authoritarian patterns for a specific demographic.

        Args:
            article: Article data
            auth_analysis: Authoritarian analysis text
            demographic: Demographic info

        Returns:
            Democratic resilience narrative
        """
        # Truncate content if too long
        content = article.get('content', '')
        content = truncate_text(content, max_length=3000)

        prompt = f"""
        Generate a democratic resilience response for {demographic["id"]} readers that addresses 
        the authoritarian indicators identified in this analysis.
        
        ARTICLE TITLE: {article['title']}
        
        ARTICLE CONTENT SUMMARY:
        {content[:1000]}...
        
        AUTHORITARIAN ANALYSIS:
        {auth_analysis}
        
        TARGET DEMOGRAPHIC: {demographic["id"]}
        CORE VALUES: {', '.join(demographic["values"])}
        
        Create content that:
        1. Appeals specifically to this demographic's values while focusing on democratic principles
        2. Uses language and framing familiar to this demographic
        3. Connects authoritarian concerns to this demographic's core values
        4. Provides a historical perspective relevant to this demographic
        5. Offers concrete actions aligned with this demographic's worldview
        6. Emphasizes patriotic protection of democratic institutions
        
        Your response should include:
        
        HEADLINE: An attention-grabbing headline for this demographic (5-10 words)
        
        DEMOCRATIC VALUES CONNECTION: How the identified authoritarian patterns threaten values this demographic holds dear
        
        HISTORICAL PARALLEL: A historical example this demographic would respect showing the dangers of similar patterns
        
        SPECIFIC CONCERNS: The most troubling aspects of the identified patterns for this specific demographic
        
        CONCRETE ACTIONS: 3-5 specific actions aligned with this demographic's values to strengthen democratic resilience
        
        UNIFYING MESSAGE: How protecting democracy transcends typical partisan divides
        """

        self.logger.info(f"Generating authoritarian response for {demographic['id']} demographic...")
        content = self._call_llm(prompt, max_tokens=1200, temperature=0.7)

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
        regarding this potentially divisive topic. Focus on shared democratic values and commitment to constitutional principles.

        ARTICLE TITLE: {article['title']}

        ARTICLE SUMMARY:
        {content}

        ANALYSIS OF DIVISIVE ELEMENTS:
        {analysis}

        Your task is to find the shared values and concerns between these opposing groups and create content that:

        1. Identifies legitimate concerns from BOTH perspectives
        2. Finds the underlying shared democratic and constitutional values
        3. Reframes the issue around these shared values
        4. Proposes solutions or approaches that could satisfy core needs of both groups
        5. Uses language that avoids triggering partisan reactions
        6. Emphasizes civic responsibility and protecting democratic institutions over partisan advantage

        Provide:

        UNIFYING HEADLINE: A headline appealing to both groups with democratic emphasis

        SHARED CONCERNS: What both groups actually worry about regarding democratic health

        COMMON GROUND: The underlying shared values at stake in our democratic system

        BRIDGE NARRATIVE: A 2-3 paragraph explanation that acknowledges both perspectives while finding common democratic purpose

        CONSTRUCTIVE NEXT STEPS: Actions that would address concerns from both perspectives while strengthening democratic norms
        """

        self.logger.info(f"Generating bridging content between {opposing_groups}...")
        content = self._call_llm(prompt, max_tokens=1200, temperature=0.6)

        return {
            "bridging_groups": opposing_groups,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }

    def generate_democratic_principles_narrative(self, article: Dict[str, Any],
                                                auth_analysis: str) -> Dict[str, Any]:
        """
        Generate content focused on shared democratic principles across political divides.

        Args:
            article: Article data
            auth_analysis: Authoritarian analysis text

        Returns:
            Democratic principles narrative
        """
        # Truncate content if too long
        content = article.get('content', '')
        content = truncate_text(content, max_length=3000)

        prompt = f"""
        Create a narrative that appeals to BOTH conservative AND progressive Americans
        by emphasizing shared democratic principles over partisan divisions.
        
        ARTICLE TITLE: {article['title']}
        
        AUTHORITARIAN ANALYSIS:
        {auth_analysis}
        
        ARTICLE CONTENT SUMMARY:
        {content[:1000]}...
        
        Create content that:
        1. Emphasizes shared American values of democratic governance, checks and balances, and rule of law
        2. Acknowledges legitimate policy disagreements while separating them from threats to democratic systems
        3. Appeals to patriotic protection of American democratic institutions
        4. Uses language and framing that resonates with both traditional conservative AND progressive values
        5. Avoids partisan trigger words while maintaining substance
        6. Identifies specific threats to democratic norms that should concern all Americans
        7. Emphasizes that protecting democracy is not a partisan issue
        
        Your response should include:
        
        UNIFIED HEADLINE: A headline appealing across political divides
        
        CORE MESSAGE: The central democratic principle at stake (1-2 sentences)
        
        CONSERVATIVE VALUES APPEAL: How this connects to traditional conservative values like constitutionalism, 
        rule of law, and limited government
        
        PROGRESSIVE VALUES APPEAL: How this connects to progressive values like equality, justice, and collective voice
        
        SHARED AMERICAN STORY: A brief historical example of Americans uniting to protect democracy
        
        SPECIFIC CONCERNS: The most troubling aspects of the situation that should concern all Americans
        
        CALL TO ACTION: Concrete steps citizens across the political spectrum can take together
        """

        self.logger.info(f"Generating democratic principles narrative...")
        content = self._call_llm(prompt, max_tokens=1200, temperature=0.7)

        return {
            "type": "democratic_principles",
            "content": content,
            "timestamp": datetime.now().isoformat()
        }

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process analyses to generate counter-narratives for divisive and authoritarian content.

        Args:
            input_data: Dictionary with analyses, manipulation threshold, and authoritarian threshold

        Returns:
            Dictionary with generated counter-narratives
        """
        analyses = input_data.get("analyses", [])
        auth_analyses = input_data.get("authoritarian_analyses", [])
        manipulation_threshold = input_data.get("manipulation_threshold", 6)
        authoritarian_threshold = input_data.get("authoritarian_threshold", 5)

        results = []

        for i, analysis in enumerate(analyses):
            if "error" in analysis:
                self.logger.warning(f"Skipping analysis with error: {analysis.get('error')}")
                continue

            # Extract manipulation score
            manipulation_score = extract_manipulation_score(analysis["analysis"])

            # Get authoritarian analysis if available
            auth_analysis = None
            if i < len(auth_analyses):
                auth_analysis = auth_analyses[i]

                # Extract authoritarian score if available
                auth_score = 0
                if "structured_elements" in auth_analysis:
                    auth_score = auth_analysis["structured_elements"].get("authoritarian_score", 0)

            # Check if we should generate counter-narratives
            if manipulation_score >= manipulation_threshold or (auth_analysis and auth_score >= authoritarian_threshold):
                article = analysis["article"]
                article_title = article["title"]
                self.logger.info(f"Generating counter-narratives for '{article_title}'...")

                # Generate narratives for all demographics
                counter_narratives = []
                for demographic in self.demographics:
                    narrative = self.generate_for_demographic(article, analysis["analysis"], demographic)
                    counter_narratives.append(narrative)

                # Generate authoritarian responses if we have authoritarian analysis
                authoritarian_responses = []
                if auth_analysis:
                    for demographic in self.demographics:
                        response = self.generate_authoritarian_response(
                            article, auth_analysis.get("authoritarian_analysis", ""), demographic
                        )
                        authoritarian_responses.append(response)

                # Generate bridging content
                bridging_content = self.generate_bridging_content(
                    article, analysis["analysis"], ["progressive", "conservative"]
                )

                # Generate democratic principles narrative
                democratic_principles = None
                if auth_analysis:
                    democratic_principles = self.generate_democratic_principles_narrative(
                        article, auth_analysis.get("authoritarian_analysis", "")
                    )

                result = {
                    "article_title": article_title,
                    "source": article.get("source", ""),
                    "manipulation_score": manipulation_score,
                    "counter_narratives": counter_narratives,
                    "authoritarian_responses": authoritarian_responses if auth_analysis else [],
                    "bridging_content": bridging_content,
                    "democratic_principles_narrative": democratic_principles,
                    "timestamp": datetime.now().isoformat()
                }

                results.append(result)

        return {"counter_narratives": results}