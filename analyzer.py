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

        # Store article ID for context tracking
        article_id = article_data.get("id", f"article_{hash(article_data.get('title', ''))}")
        
        self.logger.info(f"Analyzing article: {article_data['title']}")
        analysis = self._call_llm(prompt, max_tokens=1000, temperature=0.1, stop=["User:", "\n\n\n"])

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
        Analyze article for specific authoritarian governance patterns with structured output.

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
            PREVIOUS ANALYSIS:
            {self.context_history[article_id]["standard_analysis"]}
            
            Take the previous analysis into account but focus on authoritarian patterns in your new analysis.
            """

        prompt = f"""
        You are an intelligence analyst specializing in detecting authoritarian governance patterns and democratic backsliding.
        
        Analyze this political/governmental content for specific indicators of authoritarian governance.
        
        TITLE: {article_data['title']}
        SOURCE: {article_data['source']} (Bias: {article_data.get('bias_label', 'Unknown')})
        PUBLISHED: {article_data.get('published', 'Unknown date')}
        CONTENT:
        {content}
        
        {previous_context}
        
        Provide your analysis in two parts:
        
        PART 1: NARRATIVE ANALYSIS

        Analyze the content for the following authoritarian indicators, providing specific examples from the text:
        
        1. INSTITUTIONAL UNDERMINING: Evidence of undermining independent institutions (courts, agencies, media)
        2. DEMOCRATIC NORM VIOLATIONS: Violations of democratic norms, traditions, and precedents
        3. MEDIA DELEGITIMIZATION: Attempts to delegitimize independent media or factual information
        4. OPPOSITION TARGETING: Targeting political opposition as illegitimate enemies rather than legitimate opponents
        5. POWER CONCENTRATION: Moves to concentrate power or avoid checks and balances
        6. ACCOUNTABILITY EVASION: Attempts to evade accountability or oversight
        7. THREAT EXAGGERATION: Exaggeration of threats to justify exceptional measures
        8. AUTHORITARIAN RHETORIC: Language that glorifies strength, personal loyalty, or punishment of dissent
        9. RULE OF LAW UNDERMINING: Actions that weaken rule of law or suggest laws apply differently to different people
        
        AUTHORITARIAN SCORE: Rate from 1-10 how strongly this content indicates authoritarian governance trends.
        Explain your rating using specific examples from the text.
        
        PART 2: STRUCTURED DATA EXTRACTION
        
        Extract the following information into a structured JSON format. Be precise and include only entities/relationships that are explicitly mentioned in the content:
        
        ```json
        {{
          "actors": [
            {{"name": "Full Actor Name", "title": "President/Senator/etc", "authoritarian_actions": ["Description of action"]}}
          ],
          "institutions": [
            {{"name": "Institution Name", "type": "governmental/judicial/media/etc", "role": "Description of role"}}
          ],
          "events": [
            {{"name": "Event Description", "date": "YYYY-MM-DD if mentioned, otherwise null", "significance": "Why event matters"}}
          ],
          "authoritarian_indicators": [
            {{
              "type": "institutional_undermining",
              "present": true/false,
              "examples": ["Exact quote or specific example from text"],
              "actors_involved": ["Actor Name"],
              "institutions_targeted": ["Institution Name"],
              "severity": 1-10
            }},
            {{
              "type": "democratic_norm_violations",
              "present": true/false,
              "examples": ["Exact quote or specific example from text"],
              "actors_involved": ["Actor Name"],
              "norms_violated": ["Description of norm"],
              "severity": 1-10
            }}
            // Include other indicator types with same structure if present
          ],
          "relationships": [
            {{"source": "Actor/Institution Name", "relationship": "undermines/controls/targets/etc", "target": "Actor/Institution Name", "evidence": "Specific evidence of relationship"}}
          ],
          "authoritarian_score": 0-10
        }}
        ```
        
        Include ONLY indicators that are clearly present in the article with explicit evidence.
        For each relationship, provide clear evidence from the text.
        Be precise and factual rather than speculative.
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
            PREVIOUS STRUCTURED DATA:
            ```json
            {json.dumps(self.context_history[article_id]["structured_data"], indent=2)}
            ```
            
            Enhance and refine the previous entity extraction, adding any new entities and enhancing entity attributes.
            """
        
        prompt = f"""
        Extract all named entities from the following article, focusing on political actors, institutions, and events.
        
        TITLE: {article_data['title']}
        SOURCE: {article_data['source']} (Bias: {article_data.get('bias_label', 'Unknown')})
        CONTENT:
        {content}
        
        {previous_context}
        
        Return a valid JSON object with this structure:
        
        ```json
        {{
          "actors": [
            {{"name": "Full Name", "title": "Position/Role", "political_affiliation": "Party/Ideology if mentioned", "significance": "Why this actor matters"}}
          ],
          "institutions": [
            {{"name": "Institution Name", "type": "governmental/judicial/legislative/etc", "jurisdiction": "federal/state/local/etc if mentioned", "significance": "Why this institution matters"}}
          ],
          "events": [
            {{"name": "Event Description", "date": "Date if mentioned, otherwise null", "location": "Location if mentioned", "significance": "Why this event matters"}}
          ]
        }}
        ```
        
        Only include entities that are explicitly mentioned in the text. Do not include generic references.
        Be thorough in identifying all relevant political actors, institutions, and events.
        """
        
        result = self._call_llm(prompt, max_tokens=1500, temperature=0.1)
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
            default_result = {"actors": [], "institutions": [], "events": []}
            
            # Store default in context history
            if article_id not in self.context_history:
                self.context_history[article_id] = {}
            self.context_history[article_id]["entities"] = default_result
            
            return default_result
        except Exception as e:
            self.logger.error(f"Error extracting entities: {str(e)}")
            default_result = {"actors": [], "institutions": [], "events": []}
            
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
        actors = [e["name"] for e in entities.get("actors", [])]
        institutions = [e["name"] for e in entities.get("institutions", [])]
        
        actors_list = ", ".join(actors)
        institutions_list = ", ".join(institutions)
        
        # Include previous relationships if available
        previous_context = ""
        if article_id in self.context_history and "relationships" in self.context_history[article_id]:
            previous_context = f"""
            PREVIOUS RELATIONSHIPS:
            ```json
            {json.dumps(self.context_history[article_id]["relationships"], indent=2)}
            ```
            
            Enhance and refine the previous relationship extraction, adding any new relationships and providing more evidence.
            """
        
        prompt = f"""
        Analyze the following article to identify relationships between the listed entities.
        
        TITLE: {article_data['title']}
        SOURCE: {article_data['source']} (Bias: {article_data.get('bias_label', 'Unknown')})
        CONTENT:
        {content}
        
        ACTORS: {actors_list}
        INSTITUTIONS: {institutions_list}
        
        {previous_context}
        
        For each relationship you identify, return a JSON array of objects with this structure:
        
        ```json
        [
          {{
            "source": "Entity Name (actor or institution)",
            "relationship": "undermines/controls/targets/supports/opposes/etc",
            "target": "Entity Name (actor or institution)",
            "evidence": "Exact quote or specific example from text that demonstrates this relationship",
            "confidence": "high/medium/low based on how explicit the evidence is"
          }}
        ]
        ```
        
        Only include relationships that are clearly supported by evidence in the text.
        Include all possible relationships between the identified actors and institutions.
        Focus especially on relationships that suggest authoritarian tendencies, such as:
        - Undermining democratic institutions
        - Targeting opposition
        - Consolidating power
        - Delegitimizing media
        
        If no relationships are found, return an empty array.
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
        """Extract specific authoritarian indicators with evidence."""
        
        # Trim content if it's too long
        content = article_data.get('content', '')
        content = self._truncate_text(content, max_length=6000)

        # Get article ID for context
        article_id = article_data.get("id", f"article_{hash(article_data.get('title', ''))}")
        
        # Include previous analysis if available
        previous_context = ""
        if article_id in self.context_history and "indicators" in self.context_history[article_id]:
            previous_context = f"""
            PREVIOUS INDICATOR ANALYSIS:
            ```json
            {json.dumps(self.context_history[article_id]["indicators"], indent=2)}
            ```
            
            Enhance and refine the previous indicator analysis, adding more specific evidence and improving severity assessments.
            """
        
        prompt = f"""
        As an expert in democratic backsliding and authoritarian trends, analyze the following article for specific authoritarian indicators.
        
        TITLE: {article_data['title']}
        SOURCE: {article_data['source']} (Bias: {article_data.get('bias_label', 'Unknown')})
        CONTENT:
        {content}
        
        {previous_context}
        
        For each of these authoritarian indicators, determine if it's present in the text and provide evidence:
        
        1. Institutional undermining
        2. Democratic norm violations
        3. Media delegitimization
        4. Opposition targeting
        5. Power concentration
        6. Accountability evasion
        7. Threat exaggeration
        8. Authoritarian rhetoric
        9. Rule of law undermining
        
        Return only a valid JSON object with this structure:
        
        ```json
        {{
          "authoritarian_indicators": [
            {{
              "type": "institutional_undermining",
              "present": true/false,
              "examples": ["Exact quote or specific example from text"],
              "severity": 1-10,
              "explanation": "Brief explanation of why this qualifies as the indicator"
            }},
            // Repeat for each indicator type
          ],
          "authoritarian_score": 0-10,
          "score_explanation": "Brief explanation of the overall score"
        }}
        ```
        
        Only mark an indicator as present if there is clear evidence in the text.
        Be precise and factual rather than speculative.
        For each present indicator, provide multiple examples if available.
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
            default_result = {"authoritarian_indicators": [], "authoritarian_score": 0}
            
            # Store default in context history
            if article_id not in self.context_history:
                self.context_history[article_id] = {}
            self.context_history[article_id]["indicators"] = default_result
            
            return default_result
        except Exception as e:
            self.logger.error(f"Error extracting authoritarian indicators: {str(e)}")
            default_result = {"authoritarian_indicators": [], "authoritarian_score": 0}
            
            # Store default in context history
            if article_id not in self.context_history:
                self.context_history[article_id] = {}
            self.context_history[article_id]["indicators"] = default_result
            
            return default_result

    def analyze_content_for_knowledge_graph(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive analysis for knowledge graph integration."""
        
        # Get article ID for context tracking
        article_id = article_data.get("id", f"article_{hash(article_data.get('title', ''))}")
        
        self.logger.info(f"Performing comprehensive analysis for knowledge graph: {article_data['title']}")
        
        # Step 1: Extract basic entities
        entities = self.extract_named_entities(article_data)
        
        # Step 2: Extract relationships between entities
        relationships = self.extract_entity_relationships(article_data, entities)
        
        # Step 3: Extract authoritarian indicators
        indicators = self.extract_authoritarian_indicators(article_data)
        
        # Step 4: Perform comprehensive authoritarian analysis
        auth_analysis = self.analyze_authoritarian_patterns(article_data)
        
        # Step 5: Merge and structure the data for knowledge graph
        structured_data = {
            "entities": entities,
            "relationships": relationships,
            "authoritarian_indicators": indicators.get("authoritarian_indicators", []),
            "authoritarian_score": indicators.get("authoritarian_score", 0),
            "score_explanation": indicators.get("score_explanation", ""),
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
                            "divisive_elements": self._extract_array(json_str, "divisive_elements"),
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
                "divisive_elements": self._extract_array(result, "divisive_elements") or ["Unspecified element"],
                "manipulation_techniques": self._extract_array(result, "manipulation_techniques") or ["Unspecified technique"],
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
                # Perform standard divisive content analysis
                analysis = self.analyze_article(article)
                standard_results.append(analysis)
    
                # Perform authoritarian pattern analysis
                auth_analysis = self.analyze_authoritarian_patterns(article)
                auth_results.append(auth_analysis)
    
                # Extract structured elements from standard analysis
                if "analysis" in analysis:
                    elements = self.extract_key_elements(analysis["analysis"])
                    analysis["structured_elements"] = elements
                
                # Perform comprehensive knowledge graph analysis
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
            
    def _extract_array(self, text: str, field_name: str) -> List[str]:
        """Helper method to extract array values using regex"""
        pattern = fr'"{field_name}"\s*:\s*\[\s*(.*?)\s*\]'
        match = re.search(pattern, text, re.DOTALL)
        
        if not match:
            return []
        
        items_str = match.group(1)
        
        # Extract quoted strings
        items = re.findall(r'"([^"]*)"', items_str)
        
        # If no quoted strings found, try extracting without quotes
        if not items:
            # Split by commas and clean
            items = [item.strip().strip('"').strip("'") for item in items_str.split(',')]
            items = [item for item in items if item]  # Remove empty items
        
        return items if items else []
        
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
