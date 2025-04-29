"""
Night_watcher Content Analyzer Agent
Agent for analyzing articles for divisive content and authoritarian patterns.
"""

import re
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from agents.base import Agent, LLMProvider
from utils.text import truncate_text


class ContentAnalyzer(Agent):
    """Agent for analyzing articles for divisive content and authoritarian patterns"""

    def __init__(self, llm_provider: LLMProvider):
        """Initialize with LLM provider"""
        super().__init__(llm_provider, name="ContentAnalyzer")

    def preprocess_llm_output(self, text: str) -> str:
        """
        Remove reasoning tags and other artifacts from LLM outputs.

        Args:
            text: Text to preprocess, potentially containing reasoning tags

        Returns:
            Cleaned text with reasoning tags and other artifacts removed
        """
        # Remove reasoning tags and their contents
        text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL)
        # Remove standalone reasoning end tag
        text = re.sub(r'</reasoning>', '', text)
        # Remove sep tag and anything after it
        text = re.sub(r'<sep>.*?$', '', text, flags=re.DOTALL)
        # Remove any other model-specific artifacts
        text = re.sub(r'human:', '', text)

        return text.strip()

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

        # Preprocess the analysis to remove any reasoning tags
        analysis = self.preprocess_llm_output(analysis)

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

        # Preprocess the analysis to remove any reasoning tags
        analysis = self.preprocess_llm_output(analysis)

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
        # Ensure analysis is preprocessed
        analysis = self.preprocess_llm_output(analysis)
    
        # Create a more direct prompt focused on JSON output
        prompt = f"""
        Extract the key elements from this article analysis into a JSON format.
        
        ANALYSIS:
        {analysis}
    
        Return ONLY the following JSON object with no extra text or explanation:
        {{
            "main_topics": ["topic1", "topic2", ...],
            "frames": ["frame1", "frame2", ...],
            "emotional_triggers": ["emotion1", "emotion2", ...],
            "divisive_elements": ["element1", "element2", ...],
            "manipulation_techniques": ["technique1", "technique2", ...],
            "manipulation_score": 0-10
        }}
        """
    
        result = self._call_llm(prompt, max_tokens=800, temperature=0.1)
        result = self.preprocess_llm_output(result)
    
        try:
            # Find the JSON part (in case there's extra text)
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                # Clean up any potential format issues
                json_str = re.sub(r'(\w+)(?=:)(?=(?:[^"]*"[^"]*")*[^"]*$)', r'"\1"', json_str)  # Quote unquoted keys
                json_str = re.sub(r':\s*([^",\s\[\]\{\}][^",\]\}]*?)(?=,|}|])', r': "\1"', json_str)  # Quote unquoted values
                
                # Additional cleanup for common JSON issues
                json_str = json_str.replace("'", '"')  # Replace single quotes with double quotes
                json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
                
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # If still can't parse, try a simpler cleanup approach
                    self.logger.warning("Initial JSON parsing failed, trying alternative cleanup")
                    pattern = r'{\s*"main_topics":.+?"manipulation_score":\s*(\d+)\s*}'
                    match = re.search(pattern, json_str, re.DOTALL)
                    if match:
                        score = int(match.group(1))
                        return {
                            "main_topics": ["Extracted topic"],
                            "frames": ["Extracted frame"],
                            "emotional_triggers": ["Extracted trigger"],
                            "divisive_elements": ["Extracted element"],
                            "manipulation_techniques": ["Extracted technique"],
                            "manipulation_score": score
                        }
            
            # No valid JSON found, create a fallback structure
            self.logger.warning("No valid JSON found in LLM response, creating fallback structure")
            
            # Extract manipulation score if possible
            score_match = re.search(r'manipulation_score["\s:]+(\d+)', result, re.IGNORECASE)
            score = int(score_match.group(1)) if score_match else 5
    
            return {
                "main_topics": ["Unspecified topic"],
                "frames": ["Unspecified frame"],
                "emotional_triggers": ["Unspecified trigger"],
                "divisive_elements": ["Unspecified element"],
                "manipulation_techniques": ["Unspecified technique"],
                "manipulation_score": score
            }
        except Exception as e:
            self.logger.error(f"Error parsing extracted elements: {str(e)}")
            self.logger.debug(f"Raw LLM response: {result}")
    
            # Return a fallback structure
            return {
                "main_topics": ["Unspecified topic"],
                "frames": ["Unspecified frame"],
                "emotional_triggers": ["Unspecified trigger"],
                "divisive_elements": ["Unspecified element"],
                "manipulation_techniques": ["Unspecified technique"],
                "manipulation_score": 5,
                "error": f"Failed to extract elements: {str(e)}",
                "raw_result": result[:200]  # Include first 200 chars of raw result for debugging
            }

    def extract_authoritarian_elements(self, analysis: str) -> Dict[str, Any]:
        """
        Extract structured authoritarian elements from an analysis.
    
        Args:
            analysis: Authoritarian analysis text to extract from
    
        Returns:
            Dict with structured authoritarian elements
        """
        # Ensure analysis is preprocessed
        analysis = self.preprocess_llm_output(analysis)
    
        prompt = f"""
        Extract the key authoritarian indicators from this analysis into a JSON format.
    
        ANALYSIS:
        {analysis}
    
        Return ONLY the following JSON object with no extra text or explanation:
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
        """
    
        result = self._call_llm(prompt, max_tokens=1000, temperature=0.1)
        result = self.preprocess_llm_output(result)
    
        try:
            # Find the JSON part
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                
                # Clean up potential format issues
                json_str = re.sub(r':\s*true\b', ': true', json_str)  # Normalize true
                json_str = re.sub(r':\s*false\b', ': false', json_str)  # Normalize false
                
                # Fix common quotes/apostrophes issues
                json_str = json_str.replace("'", '"')
                
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # If still can't parse JSON, try a simplified approach
                    self.logger.warning("Initial JSON parsing failed, trying simplified approach")
                    
                    # Extract authoritarian score if available
                    score_match = re.search(r'"authoritarian_score":\s*(\d+)', json_str)
                    auth_score = int(score_match.group(1)) if score_match else 3
                    
                    # Create a basic structure with just the score
                    return self._create_default_auth_structure(auth_score)
            else:
                # No JSON found, create a fallback structure
                self.logger.warning("No valid JSON found in LLM response, creating fallback structure")
                
                # Create default structure
                return self._create_default_auth_structure(3)
                
        except Exception as e:
            self.logger.error(f"Error parsing extracted authoritarian elements: {str(e)}")
            self.logger.debug(f"Raw LLM response: {result}")
            
            # Return a fallback structure
            return self._create_default_auth_structure(3)
    
    def _create_default_auth_structure(self, score: int = 3) -> Dict[str, Any]:
        """Create default structure for authoritarian indicators"""
        default_indicator = {"present": False, "examples": []}
        
        return {
            "institutional_undermining": default_indicator.copy(),
            "democratic_norm_violations": default_indicator.copy(),
            "media_delegitimization": default_indicator.copy(),
            "opposition_targeting": default_indicator.copy(),
            "power_concentration": default_indicator.copy(),
            "accountability_evasion": default_indicator.copy(),
            "threat_exaggeration": default_indicator.copy(),
            "authoritarian_rhetoric": default_indicator.copy(),
            "rule_of_law_undermining": default_indicator.copy(),
            "authoritarian_score": score
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
