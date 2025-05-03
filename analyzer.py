"""
Night_watcher Content Analyzer
Module for analyzing articles for authoritarian patterns in political media and governance.
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
    """Analyzer for articles and content to identify authoritarian patterns in governance"""

    def __init__(self, llm_provider: LLMProvider):
        """Initialize with LLM provider"""
        self.llm_provider = llm_provider
        self.logger = logging.getLogger("ContentAnalyzer")
        self.context_history = {}  # To maintain context between analyses

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
        text = re.sub(r'assistant:', '', text)

        return text.strip()

    def analyze_article(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a single article for key political content and framing.

        Args:
            article_data: Article data to analyze

        Returns:
            Analysis result
        """
        # Trim content if it's too long
        content = article_data.get('content', '')
        content = self._truncate_text(content, max_length=6000)

        # New open-ended approach as specified in the roadmap
        prompt = f"""
        You are a political analyst with expertise in democratic systems, governance structures, and the historical patterns of democratic backsliding.

        Analyze the following political content. Think carefully about any concerning patterns, power dynamics, institutional relationships, and rhetorical strategies evident in the text.

        TITLE: {article_data['title']}
        SOURCE: {article_data['source']} (Bias: {article_data.get('bias_label', 'Unknown')})
        CONTENT:
        {content}

        Provide a thoughtful analysis with the following sections:

        1. MAIN TOPICS: What are the key topics, events, or issues discussed in this content?

        2. GOVERNANCE DYNAMICS: Analyze the power dynamics, institutional relationships, or governance patterns evident in this content. What does this reveal about how power is being exercised?

        3. HISTORICAL PARALLELS: Does this content remind you of any historical patterns or trends from other political contexts? Explain any parallels you notice.

        4. CONCERNING ELEMENTS: What aspects of this content, if any, raise concerns about democratic health? Think broadly about both obvious and subtle patterns.

        5. INSTITUTIONAL IMPACTS: How might the actions or rhetoric described affect key democratic institutions or norms?

        6. MANIPULATION ASSESSMENT: To what extent does this content appear to manipulate information, emotions, or perceptions? Rate from 1-10 with 1 being objective reporting and 10 being highly manipulative.
        """

        # Store article ID for context tracking
        article_id = article_data.get("id", f"article_{hash(article_data.get('title', ''))}")
        
        self.logger.info(f"Analyzing article: {article_data['title']}")
        analysis = self._call_llm(prompt, max_tokens=1500, temperature=0.1)

        # Preprocess the analysis to remove any reasoning tags
        analysis = self.preprocess_llm_output(analysis)

        # Store the analysis in context history
        if article_id not in self.context_history:
            self.context_history[article_id] = {}
        self.context_history[article_id]["standard_analysis"] = analysis

        return {
            "article": article_data,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }

    def analyze_authoritarian_patterns(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze article for authoritarian governance patterns with open-ended approach.

        Args:
            article_data: Article data to analyze

        Returns:
            Authoritarian pattern analysis result with structured data
        """
        # Trim content if it's too long
        content = article_data.get('content', '')
        content = self._truncate_text(content, max_length=6000)

        # Get article ID for context tracking
        article_id = article_data.get("id", f"article_{hash(article_data.get('title', ''))}")
        
        # Include previous analysis if available
        previous_context = ""
        if article_id in self.context_history and "standard_analysis" in self.context_history[article_id]:
            previous_context = f"""
            I've already analyzed this content once. Here's my previous analysis:
            
            {self.context_history[article_id]["standard_analysis"]}
            
            Now I'd like to go deeper and focus specifically on patterns that may signal democratic backsliding.
            """

        # New open-ended approach based on the roadmap
        prompt = f"""
        You are an expert on democratic institutions and historical patterns of democratic backsliding and authoritarian governance.
        
        Analyze this political content for potential authoritarian indicators. What aspects concern you the most and why? Think about both obvious and subtle patterns that may signal democratic backsliding.
        
        TITLE: {article_data['title']}
        SOURCE: {article_data['source']} (Bias: {article_data.get('bias_label', 'Unknown')})
        PUBLISHED: {article_data.get('published', 'Unknown date')}
        CONTENT:
        {content}
        
        {previous_context}
        
        First, provide your open analysis in NARRATIVE ANALYSIS format. Don't use predefined categories - instead, share your genuine concerns based on your knowledge of historical authoritarian patterns.

        Second, extract the following information in STRUCTURED DATA format:

        ```json
        {{
          "key_actors": [
            {{"name": "Actor Name", "role": "Position/Role", "actions": ["Specific action described"], "concern_level": "high/medium/low"}}
          ],
          "key_institutions": [
            {{"name": "Institution Name", "type": "Type of institution", "how_affected": "How this institution is being affected"}}
          ],
          "key_events": [
            {{"description": "Brief description of event", "significance": "Why this event matters for democratic health"}}
          ],
          "authoritarian_indicators": [
            {{
              "indicator": "Describe the specific indicator of concern",
              "evidence": ["Specific examples from the text"],
              "historical_parallels": "Any historical patterns this resembles",
              "severity": 1-10
            }}
          ],
          "key_relationships": [
            {{"source": "Actor/Institution", "relationship": "Nature of relationship", "target": "Actor/Institution", "democratic_impact": "How this relationship impacts democratic functioning"}}
          ],
          "overall_assessment": {{
            "concern_level": 1-10,
            "main_concerns": ["List of primary concerns"],
            "historical_context": "Brief historical context for these patterns"
          }}
        }}
        ```
        """

        self.logger.info(f"Analyzing authoritarian patterns in: {article_data['title']}")
        analysis = self._call_llm(prompt, max_tokens=3000, temperature=0.1)

        # Preprocess the analysis to remove any reasoning tags
        analysis = self.preprocess_llm_output(analysis)

        # Extract structured data from the analysis
        structured_data = self._extract_structured_data(analysis)
        
        # Store the analysis in context history
        if article_id not in self.context_history:
            self.context_history[article_id] = {}
        self.context_history[article_id]["authoritarian_analysis"] = analysis
        self.context_history[article_id]["structured_data"] = structured_data

        return {
            "article": article_data,
            "authoritarian_analysis": analysis,
            "structured_elements": structured_data,
            "timestamp": datetime.now().isoformat()
        }

    def extract_named_entities(self, article_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Extract named entities from article with structured attributes."""
        
        # Trim content if it's too long
        content = article_data.get('content', '')
        content = self._truncate_text(content, max_length=6000)

        # Get article ID for context
        article_id = article_data.get("id", f"article_{hash(article_data.get('title', ''))}")
        
        # Include previous analysis if available
        previous_context = ""
        if article_id in self.context_history and "structured_data" in self.context_history[article_id]:
            previous_context = f"""
            I've already extracted some structured data from this content. Here it is:
            ```json
            {json.dumps(self.context_history[article_id]["structured_data"], indent=2)}
            ```
            
            Now I want to focus specifically on identifying all named entities in a more comprehensive way.
            """
        
        # Improved entity extraction prompt
        prompt = f"""
        From the political analysis of this content, identify and extract:
        
        1. Key political actors (people, organizations, institutions)
        2. Actions they're taking
        3. Justifications provided for these actions
        4. Targets of these actions
        5. Historical parallels you notice
        
        TITLE: {article_data['title']}
        SOURCE: {article_data['source']} (Bias: {article_data.get('bias_label', 'Unknown')})
        CONTENT:
        {content}
        
        {previous_context}
        
        Return your analysis as structured data with the following format:

        ```json
        {{
          "actors": [
            {{
              "name": "Full Name", 
              "type": "individual/organization/institution", 
              "role": "Position/Role", 
              "political_affiliation": "Party/Ideology if known", 
              "significance": "Why this actor matters in this context",
              "actions": ["Action 1", "Action 2"]
            }}
          ],
          "institutions": [
            {{
              "name": "Institution Name", 
              "type": "governmental/judicial/legislative/etc", 
              "jurisdiction": "federal/state/local/etc if applicable", 
              "function": "Primary function in democratic system",
              "how_affected": "How this institution is being affected"
            }}
          ],
          "events": [
            {{
              "name": "Event Description", 
              "date": "Date if mentioned", 
              "location": "Location if mentioned", 
              "actors_involved": ["Actor 1", "Actor 2"],
              "significance": "Why this event matters"
            }}
          ],
          "relationships": [
            {{
              "source": "Actor/Institution name",
              "relationship_type": "controls/influences/undermines/etc",
              "target": "Actor/Institution name",
              "evidence": "Specific evidence from the text"
            }}
          ]
        }}
        ```
        
        Only include entities that are explicitly mentioned in the text. Be thorough and precise in your extraction.
        """
        
        result = self._call_llm(prompt, max_tokens=2000, temperature=0.1)
        result = self.preprocess_llm_output(result)
        
        # Process the result to extract JSON
        try:
            # Extract JSON pattern
            json_pattern = r'```json\s*(.*?)\s*```'
            match = re.search(json_pattern, result, re.DOTALL)
            
            if match:
                json_str = match.group(1)
                # Clean up common JSON issues
                json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
                json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
                
                extracted_entities = json.loads(json_str)
                
                # Store in context history
                if article_id not in self.context_history:
                    self.context_history[article_id] = {}
                self.context_history[article_id]["entities"] = extracted_entities
                
                return extracted_entities
            else:
                # Try to find any JSON object
                json_pattern = r'(\{.*\})'
                match = re.search(json_pattern, result, re.DOTALL)
                if match:
                    json_str = match.group(1)
                    json_str = re.sub(r',\s*}', '}', json_str)
                    json_str = re.sub(r',\s*]', ']', json_str)
                    
                    extracted_entities = json.loads(json_str)
                    
                    # Store in context history
                    if article_id not in self.context_history:
                        self.context_history[article_id] = {}
                    self.context_history[article_id]["entities"] = extracted_entities
                    
                    return extracted_entities
                
            # If no valid JSON found
            default_result = {"actors": [], "institutions": [], "events": [], "relationships": []}
            
            # Store default in context history
            if article_id not in self.context_history:
                self.context_history[article_id] = {}
            self.context_history[article_id]["entities"] = default_result
            
            return default_result
        except Exception as e:
            self.logger.error(f"Error extracting entities: {str(e)}")
            default_result = {"actors": [], "institutions": [], "events": [], "relationships": []}
            
            # Store default in context history
            if article_id not in self.context_history:
                self.context_history[article_id] = {}
            self.context_history[article_id]["entities"] = default_result
            
            return default_result

    def extract_entity_relationships(self, article_data: Dict[str, Any], entities: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> List[Dict[str, Any]]:
        """Extract relationships between identified entities."""
        
        # Trim content if it's too long
        content = article_data.get('content', '')
        content = self._truncate_text(content, max_length=6000)

        # Get article ID for context
        article_id = article_data.get("id", f"article_{hash(article_data.get('title', ''))}")
        
        # If entities not provided, use from context history or extract
        if entities is None:
            if article_id in self.context_history and "entities" in self.context_history[article_id]:
                entities = self.context_history[article_id]["entities"]
            else:
                entities = self.extract_named_entities(article_data)
        
        # Format entities for the prompt
        actors = [e.get("name", "") for e in entities.get("actors", [])]
        institutions = [e.get("name", "") for e in entities.get("institutions", [])]
        
        actors_list = ", ".join(actors)
        institutions_list = ", ".join(institutions)
        
        # Include previous relationships if available
        previous_context = ""
        if article_id in self.context_history and "relationships" in self.context_history[article_id]:
            previous_context = f"""
            I've already identified some relationships from this content. Here they are:
            ```json
            {json.dumps(self.context_history[article_id]["relationships"], indent=2)}
            ```
            
            Now I want to identify any additional relationships or refine the existing ones with better evidence.
            """
        
        # Improved relationship extraction prompt
        prompt = f"""
        Analyze the relationships between political actors and institutions in this content. Focus on power dynamics, influence patterns, and institutional relationships that might impact democratic functioning.
        
        TITLE: {article_data['title']}
        SOURCE: {article_data['source']} (Bias: {article_data.get('bias_label', 'Unknown')})
        CONTENT:
        {content}
        
        KEY ACTORS: {actors_list}
        KEY INSTITUTIONS: {institutions_list}
        
        {previous_context}
        
        Return your analysis as a JSON array of relationship objects:

        ```json
        [
          {{
            "source": "Entity Name (actor or institution)",
            "relationship_type": "undermines/controls/targets/supports/opposes/delegates_to/benefits_from/etc",
            "target": "Entity Name (actor or institution)",
            "evidence": "Specific evidence from the text that demonstrates this relationship",
            "democratic_impact": "How this relationship potentially impacts democratic functioning",
            "confidence": "high/medium/low based on how explicit the evidence is"
          }}
        ]
        ```
        
        Focus especially on relationships that reveal power dynamics and institutional impacts. Look for:

        1. Who is influencing, controlling, or undermining which institutions
        2. Which actors are forming alliances or opposing each other
        3. How power is being concentrated, delegated, or balanced
        4. Which institutions are being strengthened or weakened
        
        Only include relationships clearly supported by evidence in the text. Be precise and thorough.
        """
        
        result = self._call_llm(prompt, max_tokens=2000, temperature=0.1)
        result = self.preprocess_llm_output(result)
        
        # Process the result
        try:
            # Extract JSON array
            json_pattern = r'```json\s*(\[.*?\])\s*```'
            match = re.search(json_pattern, result, re.DOTALL)
            
            if match:
                json_str = match.group(1)
                # Clean up common JSON issues
                json_str = re.sub(r',\s*\]', ']', json_str)  # Remove trailing commas
                
                extracted_relationships = json.loads(json_str)
                
                # Store in context history
                if article_id not in self.context_history:
                    self.context_history[article_id] = {}
                self.context_history[article_id]["relationships"] = extracted_relationships
                
                return extracted_relationships
            else:
                # Try to find any JSON array
                json_pattern = r'(\[.*\])'
                match = re.search(json_pattern, result, re.DOTALL)
                if match:
                    json_str = match.group(1)
                    # Clean up common JSON issues
                    json_str = re.sub(r',\s*\]', ']', json_str)  # Remove trailing commas
                    
                    extracted_relationships = json.loads(json_str)
                    
                    # Store in context history
                    if article_id not in self.context_history:
                        self.context_history[article_id] = {}
                    self.context_history[article_id]["relationships"] = extracted_relationships
                    
                    return extracted_relationships
                
            # If no valid JSON found
            default_result = []
            
            # Store default in context history
            if article_id not in self.context_history:
                self.context_history[article_id] = {}
            self.context_history[article_id]["relationships"] = default_result
            
            return default_result
        except Exception as e:
            self.logger.error(f"Error extracting relationships: {str(e)}")
            default_result = []
            
            # Store default in context history
            if article_id not in self.context_history:
                self.context_history[article_id] = {}
            self.context_history[article_id]["relationships"] = default_result
            
            return default_result

    def extract_authoritarian_indicators(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract potential authoritarian indicators using a more open approach."""
        
        # Trim content if it's too long
        content = article_data.get('content', '')
        content = self._truncate_text(content, max_length=6000)

        # Get article ID for context
        article_id = article_data.get("id", f"article_{hash(article_data.get('title', ''))}")
        
        # Include previous analysis if available
        previous_context = ""
        if article_id in self.context_history and "indicators" in self.context_history[article_id]:
            previous_context = f"""
            I've already identified some authoritarian indicators in this content. Here they are:
            ```json
            {json.dumps(self.context_history[article_id]["indicators"], indent=2)}
            ```
            
            Now I want to reassess this content with a fresh perspective, focusing on patterns that might signal democratic backsliding.
            """
        
        # Improved open-ended authoritarian indicator extraction
        prompt = f"""
        Analyze this political content for potential authoritarian indicators. What aspects concern you the most and why? Think about both obvious and subtle patterns that may signal democratic backsliding.
        
        TITLE: {article_data['title']}
        SOURCE: {article_data['source']} (Bias: {article_data.get('bias_label', 'Unknown')})
        CONTENT:
        {content}
        
        {previous_context}
        
        Based on your historical knowledge of how democracies have backslid into authoritarianism, identify specific patterns of concern in this content. Rather than using pre-defined categories, describe the specific indicators you see and explain why they concern you.
        
        Return your analysis as a JSON object with this structure:

        ```json
        {{
          "democratic_concerns": [
            {{
              "concern": "Describe the specific pattern or action that concerns you",
              "evidence": ["Provide specific examples from the text"],
              "historical_context": "Explain similar patterns from historical examples of democratic backsliding",
              "potential_impact": "Describe how this could impact democratic functioning if continued or expanded",
              "severity": 1-10
            }}
          ],
          "affected_democratic_elements": [
            {{
              "element": "The democratic institution, norm, or process being affected",
              "how_affected": "How this element is being undermined, weakened, or co-opted",
              "evidence": ["Specific evidence from the text"]
            }}
          ],
          "overall_assessment": {{
            "concern_level": 1-10,
            "main_concerns": ["List of primary concerns"],
            "historical_parallels": ["Similar historical situations or patterns"]
          }}
        }}
        ```
        
        Be detailed and specific in your analysis. Focus on the evidence in the text rather than speculation, but use your historical knowledge to provide context for the patterns you identify.
        """
        
        result = self._call_llm(prompt, max_tokens=2500, temperature=0.1)
        result = self.preprocess_llm_output(result)
        
        # Process the result
        try:
            # Extract JSON object
            json_pattern = r'```json\s*(.*?)\s*```'
            match = re.search(json_pattern, result, re.DOTALL)
            
            if match:
                json_str = match.group(1)
                # Clean up common JSON issues
                json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
                json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
                
                extracted_indicators = json.loads(json_str)
                
                # Store in context history
                if article_id not in self.context_history:
                    self.context_history[article_id] = {}
                self.context_history[article_id]["indicators"] = extracted_indicators
                
                return extracted_indicators
            else:
                # Try to find any JSON object
                json_pattern = r'(\{.*\})'
                match = re.search(json_pattern, result, re.DOTALL)
                if match:
                    json_str = match.group(1)
                    json_str = re.sub(r',\s*}', '}', json_str)
                    json_str = re.sub(r',\s*]', ']', json_str)
                    
                    extracted_indicators = json.loads(json_str)
                    
                    # Store in context history
                    if article_id not in self.context_history:
                        self.context_history[article_id] = {}
                    self.context_history[article_id]["indicators"] = extracted_indicators
                    
                    return extracted_indicators
                
            # If no valid JSON found
            default_result = {"democratic_concerns": [], "affected_democratic_elements": [], "overall_assessment": {"concern_level": 0, "main_concerns": [], "historical_parallels": []}}
            
            # Store default in context history
            if article_id not in self.context_history:
                self.context_history[article_id] = {}
            self.context_history[article_id]["indicators"] = default_result
            
            return default_result
        except Exception as e:
            self.logger.error(f"Error extracting authoritarian indicators: {str(e)}")
            default_result = {"democratic_concerns": [], "affected_democratic_elements": [], "overall_assessment": {"concern_level": 0, "main_concerns": [], "historical_parallels": []}}
            
            # Store default in context history
            if article_id not in self.context_history:
                self.context_history[article_id] = {}
            self.context_history[article_id]["indicators"] = default_result
            
            return default_result

    def analyze_content_for_knowledge_graph(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive analysis for knowledge graph integration using the improved approach."""
        
        # Get article ID for context tracking
        article_id = article_data.get("id", f"article_{hash(article_data.get('title', ''))}")
        
        self.logger.info(f"Performing comprehensive analysis for knowledge graph: {article_data['title']}")
        
        # Step 1: Extract basic entities
        entities = self.extract_named_entities(article_data)
        
        # Step 2: Extract relationships between entities
        relationships = self.extract_entity_relationships(article_data, entities)
        
        # Step 3: Extract authoritarian indicators with the new approach
        indicators = self.extract_authoritarian_indicators(article_data)
        
        # Step 4: Perform comprehensive authoritarian analysis with the new approach
        auth_analysis = self.analyze_authoritarian_patterns(article_data)
        
        # Step 5: Merge and structure the data for knowledge graph
        structured_data = {
            "entities": entities,
            "relationships": relationships,
            "democratic_concerns": indicators.get("democratic_concerns", []),
            "affected_democratic_elements": indicators.get("affected_democratic_elements", []),
            "overall_assessment": indicators.get("overall_assessment", {"concern_level": 0}),
            "narrative_analysis": auth_analysis.get("authoritarian_analysis", ""),
            "article_metadata": {
                "title": article_data.get("title", ""),
                "source": article_data.get("source", ""),
                "published": article_data.get("published", ""),
                "url": article_data.get("url", ""),
                "processed_at": datetime.now().isoformat()
            }
        }
        
        # Store comprehensive data in context history
        if article_id not in self.context_history:
            self.context_history[article_id] = {}
        self.context_history[article_id]["comprehensive_data"] = structured_data
        
        return {
            "article": article_data,
            "structured_data": structured_data,
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
            json_pattern = r'STRUCTURED DATA[\s\S]*?(\{[\s\S]*?\})'
            match = re.search(json_pattern, analysis, re.DOTALL)
            
        if not match:
            # Try to find any JSON-like structure
            json_pattern = r'(\{[\s\S]*"overall_assessment"[\s\S]*\})'
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
        
        # If no JSON found, extract parts manually
        return self._extract_authoritarian_elements_manually(analysis)

    def _extract_authoritarian_elements_manually(self, analysis: str) -> Dict[str, Any]:
        """
        Manually extract structured elements from analysis text when JSON extraction fails.
        
        Args:
            analysis: Analysis text to extract from
            
        Returns:
            Dict with structured elements
        """
        result = {}
        
        # Extract key actors if present
        actors_pattern = r'key_actors.*?:\s*\[(.*?)\]'
        actors_match = re.search(actors_pattern, analysis, re.DOTALL)
        if actors_match:
            actors_text = actors_match.group(1)
            # Try to parse this section
            result["key_actors"] = []
            
        # Extract key institutions if present
        institutions_pattern = r'key_institutions.*?:\s*\[(.*?)\]'
        institutions_match = re.search(institutions_pattern, analysis, re.DOTALL)
        if institutions_match:
            institutions_text = institutions_match.group(1)
            # Try to parse this section
            result["key_institutions"] = []
            
        # Extract authoritarian indicators if present
        indicators_pattern = r'authoritarian_indicators.*?:\s*\[(.*?)\]'
        indicators_match = re.search(indicators_pattern, analysis, re.DOTALL)
        if indicators_match:
            indicators_text = indicators_match.group(1)
            # Try to parse this section
            result["authoritarian_indicators"] = []
            
        # Extract overall assessment if present
        assessment_pattern = r'overall_assessment.*?:\s*\{(.*?)\}'
        assessment_match = re.search(assessment_pattern, analysis, re.DOTALL)
        if assessment_match:
            assessment_text = assessment_match.group(1)
            # Try to parse assessment section
            try:
                concern_level_match = re.search(r'concern_level["\s:]+(\d+)', assessment_text)
                concern_level = int(concern_level_match.group(1)) if concern_level_match else 3
                result["overall_assessment"] = {
                    "concern_level": concern_level,
                    "main_concerns": []
                }
            except Exception as e:
                self.logger.error(f"Error parsing assessment: {str(e)}")
                result["overall_assessment"] = {"concern_level": 3, "main_concerns": []}
            
        # If no structured content was successfully extracted, create minimal structure
        if not result:
            result = {
                "key_actors": [],
                "key_institutions": [],
                "authoritarian_indicators": [],
                "overall_assessment": {"concern_level": 3, "main_concerns": []}
            }
            
        return result

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process articles for comprehensive analysis including knowledge graph integration.

        Args:
            input_data: Dict with 'articles' key containing articles to analyze

        Returns:
            Dict with 'analyses', 'authoritarian_analyses', and 'kg_analyses' keys containing analysis results
        """
        articles = input_data.get("articles", [])
        standard_results = []
        auth_results = []
        kg_results = []

        for article in articles:
            # Add ID to article if not present
            if "id" not in article:
                article["id"] = f"article_{hash(article.get('title', ''))}"
            
            try:
                # Perform standard content analysis with the new approach
                analysis = self.analyze_article(article)
                standard_results.append(analysis)
    
                # Perform authoritarian pattern analysis with the new approach
                auth_analysis = self.analyze_authoritarian_patterns(article)
                auth_results.append(auth_analysis)
    
                # Perform comprehensive knowledge graph analysis with the new approach
                kg_analysis = self.analyze_content_for_knowledge_graph(article)
                kg_results.append(kg_analysis)
                
            except Exception as e:
                self.logger.error(f"Error processing article {article.get('title', '')}: {str(e)}")
                # Continue with next article

        return {
            "analyses": standard_results,
            "authoritarian_analyses": auth_results,
            "kg_analyses": kg_results
        }
        
    def _call_llm(self, prompt: str, max_tokens: int = 1000,
                  temperature: float = 0.7, stop: Optional[List[str]] = None) -> str:
        """Helper method to call the LLM and extract text response"""
        if not self.llm_provider:
            return "Error: No LLM provider available"
            
        response = self.llm_provider.complete(prompt, max_tokens, temperature, stop)

        if "error" in response:
            self.logger.error(f"LLM error: {response['error']}")
            return f"Error: {response['error']}"

        try:
            return response["choices"][0]["text"].strip()
        except (KeyError, IndexError) as e:
            self.logger.error(f"Error extracting text from LLM response: {str(e)}")
            return f"Error extracting response: {str(e)}"
            
    def _truncate_text(self, text: str, max_length: int = 5000, suffix: str = "...") -> str:
        """
        Truncate text to a maximum length.

        Args:
            text: Text to truncate
            max_length: Maximum length (default: 5000)
            suffix: Suffix to add to truncated text (default: "...")

        Returns:
            Truncated text with suffix if needed
        """
        if not text or len(text) <= max_length:
            return text
        return text[:max_length] + suffix
