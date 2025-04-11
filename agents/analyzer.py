"""
Night_watcher Content Analyzer Agent
Agent for analyzing articles for divisive content and authoritarian patterns.
"""

import re
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from agents.base import LLMProvider
from utils.text import truncate_text


class ContentAnalyzer:
    """Agent for analyzing articles for divisive content and authoritarian patterns"""

    def __init__(self, llm_provider: LLMProvider):
        """Initialize with LLM provider"""
        self.llm_provider = llm_provider
        self.name = "ContentAnalyzer"
        self.logger = logging.getLogger(f"{self.name}")

    def _call_llm(self, prompt: str, max_tokens: int = 1000,
                  temperature: float = 0.7, stop: Optional[List[str]] = None) -> str:
        """Helper method to call the LLM and extract text response"""
        response = self.llm_provider.complete(prompt, max_tokens, temperature, stop)

        if "error" in response:
            self.logger.error(f"LLM error: {response['error']}")
            return f"Error: {response['error']}"

        try:
            return response["choices"][0]["text"].strip()
        except (KeyError, IndexError) as e:
            self.logger.error(f"Error extracting text from LLM response: {str(e)}")
            return f"Error extracting response: {str(e)}"

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

    def analyze_authoritarian_patterns(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze article for specific authoritarian governance patterns.
        
        Args:
            article_data: Article data to analyze
            
        Returns:
            Authoritarian pattern analysis result
        """
        # Trim content if it's too long
        content = article_data.get('content', '')
        content = truncate_text(content, max_length=6000)
        
        prompt = f"""
        Analyze this political/governmental content for specific indicators of authoritarian governance trends.
        Focus particularly on identifying authoritarian patterns in the Trump administration's actions or rhetoric.
        
        TITLE: {article_data['title']}
        SOURCE: {article_data['source']} (Bias: {article_data.get('bias_label', 'Unknown')})
        CONTENT:
        {content}
        
        Identify and analyze any of these authoritarian indicators:
        
        1. INSTITUTIONAL UNDERMINING: Evidence of undermining independent institutions (courts, agencies, media, etc.)
        
        2. DEMOCRATIC NORM VIOLATIONS: Violations of democratic norms, traditions, and precedents
        
        3. MEDIA DELEGITIMIZATION: Attempts to delegitimize independent media or factual information
        
        4. OPPOSITION TARGETING: Targeting of political opposition as illegitimate or enemies rather than legitimate opponents
        
        5. POWER CONCENTRATION: Moves to concentrate power in the executive or avoid checks and balances
        
        6. ACCOUNTABILITY EVASION: Attempts to evade accountability or oversight
        
        7. THREAT EXAGGERATION: Exaggeration of threats to justify exceptional measures or emergency powers
        
        8. AUTHORITARIAN RHETORIC: Use of language that glorifies strength, personal loyalty, or punishment of dissent
        
        9. RULE OF LAW UNDERMINING: Actions that weaken the rule of law or suggest laws apply differently to different people
        
        10. AUTHORITARIAN SCORE: Rate from 1-10 how strongly this content indicates authoritarian governance trends.
            Explain your rating using specific examples from the text.
        """
        
        self.logger.info(f"Analyzing authoritarian patterns in: {article_data['title']}")
        analysis = self._call_llm(prompt, max_tokens=1200, temperature=0.1)
        
        return {
            "article": article_data,
            "authoritarian_analysis": analysis,
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

    def extract_authoritarian_elements(self, analysis: str) -> Dict[str, Any]:
        """
        Extract structured authoritarian elements from an analysis.
        
        Args:
            analysis: Authoritarian analysis text to extract from
            
        Returns:
            Dict with structured authoritarian elements
        """
        prompt = f"""
        Extract the key authoritarian indicators from this analysis in a structured format.
        
        ANALYSIS:
        {analysis}
        
        Extract and return ONLY the following in JSON format:
        {{
            "institutional_undermining": {{"present": true/false, "examples": ["example1", "example2"]}},
            "democratic_norm_violations": {{"present": true/false, "examples": ["example1", "example2"]}},
            "media_delegitimization": {{"present": true/false, "examples": ["example1", "example2"]}},
            "opposition_targeting": {{"present": true/false, "examples": ["example1", "example2"]}},
            "power_concentration": {{"present": true/false, "examples": ["example1", "example2"]}},
            "accountability_evasion": {{"present": true/false, "examples": ["example1", "example2"]}},
            "threat_exaggeration": {{"present": true/false, "examples": ["example1", "example2"]}},
            "authoritarian_rhetoric": {{"present": true/false, "examples": ["example1", "example2"]}},
            "rule_of_law_undermining": {{"present": true/false, "examples": ["example1", "example2"]}},
            "authoritarian_score": 0-10
        }}
        
        For each indicator, set "present" to true only if there is clear evidence in the analysis.
        Include specific examples as short phrases or sentences.
        Provide only valid JSON with no explanations or extra text.
        """
        
        result = self._call_llm(prompt, max_tokens=1000, temperature=0.1)
        
        try:
            # Find the JSON part (in case there's extra text)
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                result = json_match.group(0)
                
            return json.loads(result)
        except Exception as e:
            self.logger.error(f"Error parsing extracted authoritarian elements: {str(e)}")
            return {
                "error": f"Failed to extract authoritarian elements: {str(e)}",
                "raw_result": result
            }

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process articles for analysis including both standard and authoritarian pattern analyses.

        Args:
            input_data: Dict with 'articles' key containing articles to analyze

        Returns:
            Dict with 'analyses' and 'authoritarian_analyses' keys containing analysis results
        """
        articles = input_data.get("articles", [])
        results = []
        auth_results = []
        
        for article in articles:
            # Perform standard divisive content analysis
            analysis = self.analyze_article(article)
            results.append(analysis)
            
            # Perform authoritarian pattern analysis for government-related content
            auth_analysis = self.analyze_authoritarian_patterns(article)
            auth_results.append(auth_analysis)
            
            # Extract structured elements from both analyses
            if "analysis" in analysis:
                elements = self.extract_key_elements(analysis["analysis"])
                analysis["structured_elements"] = elements
                
            if "authoritarian_analysis" in auth_analysis:
                auth_elements = self.extract_authoritarian_elements(auth_analysis["authoritarian_analysis"])
                auth_analysis["structured_elements"] = auth_elements

        return {
            "analyses": results,
            "authoritarian_analyses": auth_results
        }