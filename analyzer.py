"""
Night_watcher Content Analyzer with Multi-Round Prompting
Module for analyzing political content through multiple rounds of prompting.
"""

import logging
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
from prompts import FACT_EXTRACTION_PROMPT, ARTICLE_ANALYSIS_PROMPT, RELATIONSHIP_EXTRACTION_PROMPT

# Configure logging
logger = logging.getLogger(__name__)

# ==========================================
# Content Analyzer
# ==========================================

class ContentAnalyzer:
    """Analyzer for political content with multi-round prompting"""

    def __init__(self, llm_provider):
        """Initialize with LLM provider"""
        self.llm_provider = llm_provider
        self.logger = logging.getLogger("ContentAnalyzer")
        self.analysis_count = 0

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process articles for analysis.

        Args:
            input_data: Dict with 'articles' key containing articles to analyze

        Returns:
            Dict with 'analyses' key containing analysis results
        """
        articles = input_data.get("articles", [])
        
        if not articles:
            self.logger.warning("No articles provided for analysis")
            return {"analyses": []}
            
        self.logger.info(f"Starting analysis of {len(articles)} articles")
        
        analyses = []
        for article in articles:
            analysis = self.analyze_content_multi_round(article)
            analyses.append(analysis)
            
        self.analysis_count += len(analyses)
        self.logger.info(f"Completed analysis of {len(articles)} articles")
        
        return {
            "analyses": analyses
        }

    def analyze_content_multi_round(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze article content through multiple rounds of prompting.
        
        Args:
            article_data: Article data dict with 'title', 'content', etc.
            
        Returns:
            Dict with article data, analysis results for each round, and timestamp
        """
        title = article_data.get("title", "Untitled")
        content = article_data.get("content", "")
        source = article_data.get("source", "Unknown")
        bias_label = article_data.get("bias_label", "Unknown")
        
        # Truncate content if too long (most models have context limits)
        max_content_length = 6000
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."
        
        self.logger.info(f"Starting multi-round analysis for article: {title}")
        
        # Store all rounds of prompting and responses
        prompt_chain = []
        
        # Round 1: Fact Extraction
        try:
            round1_prompt = FACT_EXTRACTION_PROMPT.format(article_content=content)
            round1_response = self._get_llm_response(round1_prompt)
            
            prompt_chain.append({
                "round": 1,
                "name": "Fact Extraction",
                "prompt": round1_prompt,
                "response": round1_response
            })
            
            # Try to parse JSON from the response
            structured_facts = self._extract_json_from_text(round1_response)
            
            # Round 2: Article Analysis
            round2_prompt = ARTICLE_ANALYSIS_PROMPT.format(article_content=content)
            round2_response = self._get_llm_response(round2_prompt)
            
            prompt_chain.append({
                "round": 2,
                "name": "Article Analysis",
                "prompt": round2_prompt,
                "response": round2_response
            })
            
            # Round 3: Relationship Extraction (only if we have structured facts)
            if structured_facts:
                structured_data_str = json.dumps(structured_facts, indent=2)
                round3_prompt = RELATIONSHIP_EXTRACTION_PROMPT.format(structured_data=structured_data_str)
                round3_response = self._get_llm_response(round3_prompt)
                
                prompt_chain.append({
                    "round": 3,
                    "name": "Relationship Extraction",
                    "prompt": round3_prompt,
                    "response": round3_response
                })
                
                # Try to parse relationships JSON
                relationships = self._extract_json_from_text(round3_response)
                if relationships:
                    # Add relationships to the structured data
                    structured_facts["relationships"] = relationships
            
            # Compile the complete analysis
            complete_analysis = {
                "article": article_data,
                "prompt_chain": prompt_chain,
                "structured_facts": structured_facts if structured_facts else {},
                "framing_analysis": round2_response,
                "timestamp": datetime.now().isoformat()
            }
            
            return complete_analysis
            
        except Exception as e:
            self.logger.error(f"Error in multi-round analysis: {str(e)}")
            return {
                "article": article_data,
                "prompt_chain": prompt_chain,  # Include whatever we have so far
                "error": f"Analysis error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    def _get_llm_response(self, prompt: str) -> str:
        """
        Get response from LLM with proper error handling.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            Response text from LLM
        """
        try:
            response = self.llm_provider.complete(
                prompt=prompt,
                max_tokens=3000,
                temperature=0.3
            )
            
            response_text = response.get("choices", [{}])[0].get("text", "")
            
            # Check if we have an error response
            if not response_text and "error" in response:
                error_message = response.get("error", "Unknown error")
                self.logger.error(f"LLM error: {error_message}")
                return f"ERROR: {error_message}"
            
            return response_text
        except Exception as e:
            self.logger.error(f"Error getting LLM response: {str(e)}")
            return f"ERROR: {str(e)}"

    def _extract_json_from_text(self, text: str) -> Optional[Any]:
        """
        Extract and parse JSON from text, handling various formats.
        
        Args:
            text: Text that may contain JSON
            
        Returns:
            Parsed JSON object or None if parsing fails
        """
        try:
            # Try to find JSON using regex
            json_match = re.search(r'(\{|\[).*(\}|\])', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            
            # If no match, try direct parsing (in case the entire text is valid JSON)
            return json.loads(text)
        except Exception as e:
            self.logger.warning(f"Failed to extract JSON from response: {str(e)}")
            return None