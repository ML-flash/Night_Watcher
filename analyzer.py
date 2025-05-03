"""
Night_watcher Content Analyzer
Module for analyzing articles for divisive content and authoritarian patterns.
"""

import re
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from providers import LLMProvider

# Configure logging
logger = logging.getLogger(__name__)

# ==========================================
# Content Analyzer
# ==========================================

class ContentAnalyzer:
    """Analyzer for articles and content to identify divisive content and authoritarian patterns"""

    def __init__(self, llm_provider: LLMProvider):
        """Initialize with LLM provider"""
        self.llm_provider = llm_provider
        self.logger = logging.getLogger("ContentAnalyzer")

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
        content = self._truncate_text(content, max_length=6000)

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
        Analyze article for specific authoritarian governance patterns with structured output.

        Args:
            article_data: Article data to analyze

        Returns:
            Authoritarian pattern analysis result with structured data
        """
        # Trim content if it's too long
        content = article_data.get('content', '')
        content = self._truncate_text(content, max_length=6000)

        prompt = f"""
        Analyze this political/governmental content for specific indicators of authoritarian governance trends.
        Focus particularly on identifying authoritarian patterns in the Trump administration's actions or rhetoric.
        
        TITLE: {article_data['title']}
        SOURCE: {article_data['source']} (Bias: {article_data.get('bias_label', 'Unknown')})
        CONTENT:
        {content}
        
        Provide your analysis in two parts:
        
        PART 1: Narrative analysis of authoritarian indicators
        
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
            
        PART 2: Structured data extraction
        
        Based on your analysis, provide the following structured information in valid JSON format:

        ```json
        {
          "actors": [
            {"name": "Full Actor Name", "title": "President/Senator/etc", "authoritarian_actions": ["Description of action"]}
          ],
          "institutions": [
            {"name": "Institution Name", "type": "governmental/judicial/media/etc"}
          ],
          "authoritarian_indicators": [
            {
              "type": "institutional_undermining",
              "present": true/false,
              "examples": ["Example text from article"],
              "actors_involved": ["Actor Name"],
              "institutions_targeted": ["Institution Name"]
            },
            {
              "type": "democratic_norm_violations",
              "present": true/false,
              "examples": ["Example text from article"],
              "actors_involved": ["Actor Name"],
              "norms_violated": ["Description of norm"]
            },
            {
              "type": "media_delegitimization",
              "present": true/false,
              "examples": ["Example text from article"],
              "actors_involved": ["Actor Name"],
              "targets": ["Media outlet/journalists"]
            },
            {
              "type": "opposition_targeting",
              "present": true/false,
              "examples": ["Example text from article"],
              "actors_involved": ["Actor Name"],
              "targets": ["Opposition figures/groups"]
            },
            {
              "type": "power_concentration",
              "present": true/false,
              "examples": ["Example text from article"],
              "actors_involved": ["Actor Name"],
              "methods": ["Methods used to concentrate power"]
            },
            {
              "type": "accountability_evasion",
              "present": true/false,
              "examples": ["Example text from article"],
              "actors_involved": ["Actor Name"],
              "methods": ["Methods used to evade accountability"]
            },
            {
              "type": "threat_exaggeration",
              "present": true/false,
              "examples": ["Example text from article"],
              "actors_involved": ["Actor Name"],
              "exaggerated_threats": ["Description of exaggerated threat"]
            },
            {
              "type": "authoritarian_rhetoric",
              "present": true/false,
              "examples": ["Example text from article"],
              "actors_involved": ["Actor Name"],
              "rhetoric_themes": ["Themes of rhetoric used"]
            },
            {
              "type": "rule_of_law_undermining",
              "present": true/false,
              "examples": ["Example text from article"],
              "actors_involved": ["Actor Name"],
              "methods": ["Methods used to undermine rule of law"]
            }
          ],
          "authoritarian_score": 0-10,
          "key_relationships": [
            {"source": "Actor Name", "relationship": "undermines/attacks/controls", "target": "Institution/Person Name"}
          ]
        }
        ```

        Make sure the JSON is valid and ONLY includes indicators that are clearly present in the article with explicit evidence.
        """

        self.logger.info(f"Analyzing authoritarian patterns in: {article_data['title']}")
        analysis = self._call_llm(prompt, max_tokens=2000, temperature=0.1)

        # Preprocess the analysis to remove any reasoning tags
        analysis = self.preprocess_llm_output(analysis)

        # Extract structured data from the analysis
        structured_data = self._extract_structured_data(analysis)

        return {
            "article": article_data,
            "authoritarian_analysis": analysis,
            "structured_elements": structured_data,
            "timestamp": datetime.now().isoformat()
        }

    def _extract_structured_data(self, analysis: str) -> Dict[str, Any]:
        """
        Extract structured JSON data from an analysis.
        
        Args:
            analysis: Analysis text containing structured JSON
            
        Returns:
            Dict with structured data extracted from the analysis
        """
        # Look for JSON data between triple backticks
        json_pattern = r'```json\s*(.*?)\s*```'
        match = re.search(json_pattern, analysis, re.DOTALL)
        
        if not match:
            # Try other patterns
            json_pattern = r'PART 2:.*?(\{.*?\})'
            match = re.search(json_pattern, analysis, re.DOTALL)
            
        if not match:
            # Try to find any JSON-like structure
            json_pattern = r'(\{[\s\S]*"authoritarian_score"[\s\S]*\})'
            match = re.search(json_pattern, analysis, re.DOTALL)
        
        if match:
            try:
                json_str = match.group(1)
                # Clean up the JSON string
                json_str = re.sub(r'//.*', '', json_str)  # Remove comments
                json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
                json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
                
                # Parse the JSON
                structured_data = json.loads(json_str)
                return structured_data
            except json.JSONDecodeError as e:
                self.logger.error(f"Error parsing JSON: {str(e)}")
                # Return basic structure with error info
                return {
                    "error": f"Failed to parse structured data: {str(e)}",
                    "raw_json": match.group(1)[:200] + "..." if len(match.group(1)) > 200 else match.group(1)
                }
        
        # If no JSON found, extract traditional structured elements
        return self.extract_authoritarian_elements(analysis)

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
    
        # Create a more direct prompt with a clear example
        prompt = f"""
        Extract the key elements from this article analysis into a JSON format.
        
        ANALYSIS:
        {analysis}
    
        Return a JSON object with EXACTLY this structure:
        {{
            "main_topics": ["topic1", "topic2", "topic3"],
            "frames": ["frame1", "frame2"],
            "emotional_triggers": ["emotion1", "emotion2"],
            "divisive_elements": ["element1", "element2"],
            "manipulation_techniques": ["technique1", "technique2"],
            "manipulation_score": 7
        }}
        
        In the above example:
        - The arrays contain strings extracted from the analysis
        - The manipulation_score is a number from 0-10
        - There are NO other fields, comments, or explanations
    
        Replace the example values with actual content from the analysis.
        Your response must be VALID JSON with no other text.
        """
    
        # Call LLM with lower temperature to get more consistent output
        result = self._call_llm(prompt, max_tokens=800, temperature=0.1)
        result = self.preprocess_llm_output(result)
    
        try:
            # Find the JSON part (in case there's extra text)
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                
                # Try direct parsing first
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # Apply more aggressive cleaning
                    self.logger.warning("Initial JSON parsing failed, applying cleanup")
                    
                    # Replace single quotes with double quotes
                    json_str = json_str.replace("'", '"')
                    
                    # Fix common issues with quotes in arrays
                    json_str = re.sub(r'"\s*,\s*"', '", "', json_str)
                    
                    # Remove trailing commas before closing brackets
                    json_str = re.sub(r',\s*}', '}', json_str)
                    json_str = re.sub(r',\s*]', ']', json_str)
                    
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        self.logger.warning("Advanced cleanup failed, extracting components manually")
                        
                        # Extract score first
                        score_match = re.search(r'"manipulation_score":\s*(\d+)', json_str)
                        score = int(score_match.group(1)) if score_match else 5
                        
                        # Extract arrays using regex
                        result = {
                            "main_topics": self._extract_array(json_str, "main_topics"),
                            "frames": self._extract_array(json_str, "frames"),
                            "emotional_triggers": self._extract_array(json_str, "emotional_triggers"),
                            "divisive_elements": self._extract_array(result, "divisive_elements") or ["Unspecified element"],
                "manipulation_techniques": self._extract_array(result, "manipulation_techniques") or ["Unspecified technique"],
                "manipulation_score": score
            }
            
            return result
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
            } self._extract_array(json_str, "divisive_elements"),
                            "manipulation_techniques": self._extract_array(json_str, "manipulation_techniques"),
                            "manipulation_score": score
                        }
                        
                        return result
            
            # No valid JSON found, extract using regex
            self.logger.warning("No valid JSON found, using fallback extraction")
            
            # Extract manipulation score
            score_match = re.search(r'manipulation_score["\s:]+(\d+)', result, re.IGNORECASE)
            score = int(score_match.group(1)) if score_match else 5
            
            # Create result with fallback values
            return {
                "main_topics": self._extract_array(result, "main_topics") or ["Unspecified topic"],
                "frames": self._extract_array(result, "frames") or ["Unspecified frame"],
                "emotional_triggers": self._extract_array(result, "emotional_triggers") or ["Unspecified trigger"],
                "divisive_elements":
