"""
Night_watcher Content Analyzer
Module for analyzing content for manipulation techniques and authoritarian patterns.
"""

import logging
import json
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logger = logging.getLogger(__name__)

# ==========================================
# Content Analyzer
# ==========================================

class ContentAnalyzer:
    """Analyzer for content manipulation techniques and authoritarian patterns"""

    def __init__(self, llm_provider):
        """Initialize with LLM provider"""
        self.llm_provider = llm_provider
        self.logger = logging.getLogger("ContentAnalyzer")

        # Store context history for tracking articles
        self.context_history = {}

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process articles to analyze for manipulation techniques and authoritarian patterns.

        Args:
            input_data: Dict with 'articles' key containing articles to analyze

        Returns:
            Dict with 'analyses' key containing analysis results
        """
        articles = input_data.get("articles", [])

        if not articles:
            self.logger.warning("No articles provided for analysis")
            return {"analyses": [], "authoritarian_analyses": [], "kg_analyses": []}

        if not self.llm_provider:
            self.logger.error("No LLM provider available for analysis")
            return {"analyses": [], "authoritarian_analyses": [], "kg_analyses": []}

        self.logger.info(f"Starting analysis of {len(articles)} articles")

        analyses = []
        authoritarian_analyses = []
        kg_analyses = []

        for article in articles:
            # Get basic analysis
            analysis = self.analyze_content(article)
            analyses.append(analysis)

            # Get specialized authoritarian analysis
            auth_analysis = self.analyze_authoritarian_patterns(article)
            authoritarian_analyses.append(auth_analysis)

            # Get knowledge graph analysis
            kg_analysis = self.analyze_content_for_knowledge_graph(article)
            kg_analyses.append(kg_analysis)

        self.logger.info(f"Completed analysis of {len(articles)} articles")

        return {
            "analyses": analyses,
            "authoritarian_analyses": authoritarian_analyses,
            "kg_analyses": kg_analyses
        }

    def analyze_content(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze content for manipulation techniques and framing.

        Args:
            article_data: Dict with article data including 'title' and 'content'

        Returns:
            Dict with analysis results
        """
        title = article_data.get("title", "Untitled")
        content = article_data.get("content", "")
        source = article_data.get("source", "Unknown")
        bias_label = article_data.get("bias_label", "Unknown")

        self.logger.info(f"Analyzing content: {title[:50]}...")

        # Create prompt for manipulation detection
        prompt = f"""Analyze the following article for media manipulation techniques, framing, and political bias.

TITLE: {title}
SOURCE: {source} (Reported bias: {bias_label})

CONTENT:
{content[:7000]}  # Truncate for LLM context limits

Please provide a comprehensive analysis with the following sections:

1. MAIN TOPICS: List the main topics of the article.

2. FRAMING ANALYSIS: How are events and people framed? What language choices reveal underlying perspective?

3. INCLUSION/EXCLUSION: What important context or perspectives are included or excluded?

4. MANIPULATION TECHNIQUES DETECTED: Identify specific manipulation techniques if present:
   - Appeal to fear/outrage
   - False equivalence
   - Misleading statistics
   - Emotional language
   - Selective quoting
   - Strawman arguments
   - Appeal to authority
   - Bandwagon appeals
   - Evidence suppression
   - Other techniques

5. DEMOCRATIC NORM IMPLICATIONS: Does the content support or undermine democratic norms? How?

6. MANIPULATION SCORE: On a scale of 1-10, rate how manipulative this content is.
   - 1-3: Generally fair presentation
   - 4-6: Some manipulation techniques present but balanced with factual reporting
   - 7-10: Highly manipulative content with significant bias

Format your response under these exact headings.
"""

        try:
            # Call LLM for analysis
            response = self.llm_provider.complete(
                prompt=prompt,
                max_tokens=3000,
                temperature=0.3
            )

            # Get the text from the completion
            analysis_text = response.get("choices", [{}])[0].get("text", "")

            if not analysis_text or "Error" in response:
                self.logger.error(f"LLM analysis failed: {response}")
                analysis_text = "Analysis failed. LLM provider error."

            # Create analysis result
            result = {
                "article": article_data,
                "analysis": analysis_text,
                "timestamp": datetime.now().isoformat()
            }

            # Store in context history
            article_id = article_data.get("id", f"article_{hash(title)}")
            if article_id not in self.context_history:
                self.context_history[article_id] = {}

            self.context_history[article_id]["basic_analysis"] = analysis_text

            return result

        except Exception as e:
            self.logger.error(f"Error analyzing content: {str(e)}")

            # Return error result
            return {
                "article": article_data,
                "analysis": f"Analysis failed: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

    def analyze_authoritarian_patterns(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze content specifically for authoritarian patterns with revised approach.

        Args:
            article_data: Dict with article data including 'title' and 'content'

        Returns:
            Dict with authoritarian analysis results
        """
        title = article_data.get("title", "Untitled")
        content = article_data.get("content", "")
        source = article_data.get("source", "Unknown")

        self.logger.info(f"Analyzing authoritarian patterns: {title[:50]}...")

        # Create prompt with revised approach - less explicit framing
        prompt = f"""Analyze this political reporting for potential indicators of democratic backsliding.

TITLE: {title}
SOURCE: {source}

CONTENT:
{content[:7000]}  # Truncate for LLM context limits

Based on your knowledge of historical patterns in how democracy erodes, what aspects of this content concern you the most and why? What subtle patterns do you notice that might signal democratic norm violations or institutional undermining?

After providing your initial assessment, please organize your observations into:

1. DEMOCRATIC CONCERNS: List specific concerns about democratic norms or institutions, with evidence from the text.

2. AFFECTED DEMOCRATIC ELEMENTS: Which democratic institutions, norms, or processes appear to be under pressure?

3. KEY ACTORS: Who are the main actors involved and what roles are they playing?

4. HISTORICAL PARALLELS: Does this reporting remind you of patterns seen in other countries that experienced democratic erosion?

5. OVERALL ASSESSMENT: How concerning is this content for democratic health on a scale of 1-10?
   - 1-3: Minimal concern - normal political discourse
   - 4-6: Moderate concern - potential early warning signs
   - 7-10: Severe concern - clear indicators of authoritarian patterns

Please provide your thoughtful analysis focusing on subtle indicators that might not be obvious to casual readers.
"""

        try:
            # Call LLM for analysis
            response = self.llm_provider.complete(
                prompt=prompt,
                max_tokens=3000,
                temperature=0.3
            )

            # Get the text from the completion
            analysis_text = response.get("choices", [{}])[0].get("text", "")

            if not analysis_text or "Error" in response:
                self.logger.error(f"LLM authoritarian analysis failed: {response}")
                analysis_text = "Analysis failed. LLM provider error."

            # Extract structured elements from the analysis
            structured_elements = self._extract_structured_elements(analysis_text)

            # Create analysis result
            result = {
                "article": article_data,
                "authoritarian_analysis": analysis_text,
                "structured_elements": structured_elements,
                "timestamp": datetime.now().isoformat()
            }

            # Store in context history
            article_id = article_data.get("id", f"article_{hash(title)}")
            if article_id not in self.context_history:
                self.context_history[article_id] = {}

            self.context_history[article_id]["authoritarian_analysis"] = result

            return result

        except Exception as e:
            self.logger.error(f"Error analyzing authoritarian patterns: {str(e)}")

            # Return error result
            return {
                "article": article_data,
                "authoritarian_analysis": f"Analysis failed: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

    def _extract_structured_elements(self, analysis_text: str) -> Dict[str, Any]:
        """
        Extract structured elements from authoritarian pattern analysis.

        Args:
            analysis_text: The analysis text to extract structured elements from

        Returns:
            Dict with structured elements
        """
        structured_elements = {}

        # Extract democratic concerns
        if "DEMOCRATIC CONCERNS:" in analysis_text:
            concerns_section = analysis_text.split("DEMOCRATIC CONCERNS:")[1].split("\n\n")[0]
            concerns = []

            # Extract bulleted or numbered items
            for line in concerns_section.split("\n"):
                stripped = line.strip()
                if stripped and (stripped.startswith("-") or re.match(r"^\d+\.", stripped)):
                    concerns.append(stripped.lstrip("- 0123456789.").strip())

            structured_elements["democratic_concerns"] = concerns

        # Extract affected democratic elements
        if "AFFECTED DEMOCRATIC ELEMENTS:" in analysis_text:
            elements_section = analysis_text.split("AFFECTED DEMOCRATIC ELEMENTS:")[1].split("\n\n")[0]
            elements = []

            # Extract bulleted or numbered items
            for line in elements_section.split("\n"):
                stripped = line.strip()
                if stripped and (stripped.startswith("-") or re.match(r"^\d+\.", stripped)):
                    elements.append(stripped.lstrip("- 0123456789.").strip())

            structured_elements["affected_democratic_elements"] = elements

        # Extract key actors
        if "KEY ACTORS:" in analysis_text:
            actors_section = analysis_text.split("KEY ACTORS:")[1].split("\n\n")[0]
            actors = []

            # Extract bulleted or numbered items
            for line in actors_section.split("\n"):
                stripped = line.strip()
                if stripped and (stripped.startswith("-") or re.match(r"^\d+\.", stripped)):
                    actors.append(stripped.lstrip("- 0123456789.").strip())

            structured_elements["key_actors"] = actors

        # Extract historical parallels
        if "HISTORICAL PARALLELS:" in analysis_text:
            parallels_section = analysis_text.split("HISTORICAL PARALLELS:")[1].split("\n\n")[0]

            # Clean up section
            parallels = parallels_section.strip()
            structured_elements["historical_parallels"] = parallels

        # Extract overall assessment
        if "OVERALL ASSESSMENT:" in analysis_text:
            assessment_section = analysis_text.split("OVERALL ASSESSMENT:")[1].split("\n\n")[0]

            # Try to extract numeric score
            score_match = re.search(r"(\d+(?:\.\d+)?)\s*\/\s*10", assessment_section)
            concern_score = float(score_match.group(1)) if score_match else 0

            # Determine concern level
            concern_level = "minimal"
            if concern_score >= 7:
                concern_level = "severe"
            elif concern_score >= 4:
                concern_level = "moderate"

            structured_elements["overall_assessment"] = {
                "score": concern_score,
                "level": concern_level,
                "assessment_text": assessment_section.strip()
            }

        return structured_elements

    def extract_named_entities(self, article_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract named entities from article content.

        Args:
            article_data: Dict with article data

        Returns:
            Dict with entity categories and extracted entities
        """
        title = article_data.get("title", "Untitled")
        content = article_data.get("content", "")

        # Create prompt for entity extraction
        prompt = f"""Extract all named entities from the following article.

TITLE: {title}

CONTENT:
{content[:5000]}  # Truncate for LLM context

For each entity type, provide a list in this JSON format:
{{
  "actors": [
    {{"name": "Name of Person", "type": "person", "role": "politician/official/etc"}},
    ...
  ],
  "institutions": [
    {{"name": "Name of Institution", "type": "government/judicial/media/etc"}},
    ...
  ],
  "events": [
    {{"name": "Name of Event", "type": "policy/legislation/speech/etc"}},
    ...
  ],
  "locations": [
    {{"name": "Location Name", "type": "country/city/region"}},
    ...
  ]
}}

Only include entities that are specifically mentioned in the article. Focus on political and governmental entities.
"""

        try:
            # Call LLM for entity extraction
            response = self.llm_provider.complete(
                prompt=prompt,
                max_tokens=2000,
                temperature=0.1
            )

            # Get the text from the completion
            extraction_text = response.get("choices", [{}])[0].get("text", "")

            # Parse JSON from the response
            # Find JSON content between curly braces
            json_match = re.search(r'\{.*\}', extraction_text, re.DOTALL)

            if json_match:
                try:
                    json_content = json_match.group(0)
                    entities = json.loads(json_content)

                    # Ensure we have all entity types
                    for entity_type in ["actors", "institutions", "events", "locations"]:
                        if entity_type not in entities:
                            entities[entity_type] = []

                    return entities
                except json.JSONDecodeError:
                    self.logger.warning("Failed to parse JSON from entity extraction response")
                    return {"actors": [], "institutions": [], "events": [], "locations": []}
            else:
                self.logger.warning("No JSON content found in entity extraction response")
                return {"actors": [], "institutions": [], "events": [], "locations": []}

        except Exception as e:
            self.logger.error(f"Error extracting named entities: {str(e)}")
            return {"actors": [], "institutions": [], "events": [], "locations": []}

    def extract_entity_relationships(self, article_data: Dict[str, Any],
                                    entities: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Extract relationships between entities in the article.

        Args:
            article_data: Dict with article data
            entities: Dict with extracted entities

        Returns:
            List of relationship dictionaries
        """
        title = article_data.get("title", "Untitled")
        content = article_data.get("content", "")

        # Prepare entity lists for the prompt
        actors_list = "\n".join([f"- {a['name']}" for a in entities.get("actors", [])])
        institutions_list = "\n".join([f"- {i['name']}" for i in entities.get("institutions", [])])

        # Create prompt for relationship extraction
        prompt = f"""Extract relationships between entities in the following article.

TITLE: {title}

KEY ACTORS:
{actors_list if actors_list else "None identified"}

KEY INSTITUTIONS:
{institutions_list if institutions_list else "None identified"}

CONTENT:
{content[:5000]}  # Truncate for LLM context

Extract relationships between entities in JSON format:
[
  {{
    "source": "Source Entity Name",
    "source_type": "actor/institution/event",
    "relation": "controls/influences/undermines/strengthens/performs/authorizes/blocks/opposes/etc",
    "target": "Target Entity Name",
    "target_type": "actor/institution/event",
    "evidence": "Exact quote from text showing this relationship"
  }},
  ...
]

Focus on relationships relevant to authoritarian patterns or democratic institutions.
"""

        try:
            # Call LLM for relationship extraction
            response = self.llm_provider.complete(
                prompt=prompt,
                max_tokens=2000,
                temperature=0.1
            )

            # Get the text from the completion
            extraction_text = response.get("choices", [{}])[0].get("text", "")

            # Parse JSON from the response
            # Find JSON content between square brackets
            json_match = re.search(r'\[.*\]', extraction_text, re.DOTALL)

            if json_match:
                try:
                    json_content = json_match.group(0)
                    relationships = json.loads(json_content)
                    return relationships
                except json.JSONDecodeError:
                    self.logger.warning("Failed to parse JSON from relationship extraction response")
                    return []
            else:
                self.logger.warning("No JSON content found in relationship extraction response")
                return []

        except Exception as e:
            self.logger.error(f"Error extracting entity relationships: {str(e)}")
            return []

    def extract_authoritarian_indicators(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract authoritarian indicators using the improved approach.

        Args:
            article_data: Dict with article data

        Returns:
            Dict with extracted authoritarian indicators
        """
        title = article_data.get("title", "Untitled")
        content = article_data.get("content", "")

        # Create prompt with revised approach - less explicit framing
        prompt = f"""Analyze this political content for potential authoritarian indicators.

TITLE: {title}

CONTENT:
{content[:5000]}  # Truncate for LLM context limits

First, analyze this content for any patterns that concern you regarding democratic health. Think about subtle tactics that may erode democratic norms, institutions, or processes.

Then output your findings in this JSON format:
{{
  "democratic_concerns": [
    {{
      "concern": "Description of the democratic concern",
      "evidence": ["Quote or example from the text", ...],
      "severity": 1-10
    }},
    ...
  ],
  "affected_democratic_elements": [
    {{
      "element": "Name of democratic institution/norm/process affected",
      "evidence": ["Quote or example from the text", ...],
      "impact": "Description of potential impact"
    }},
    ...
  ],
  "overall_assessment": {{
    "concern_level": 1-10,
    "main_concerns": ["Brief statement of main concern", ...]
  }}
}}

Focus on identifying subtle patterns that might not be obvious to casual readers.
"""

        try:
            # Call LLM for extraction
            response = self.llm_provider.complete(
                prompt=prompt,
                max_tokens=2000,
                temperature=0.3
            )

            # Get the text from the completion
            extraction_text = response.get("choices", [{}])[0].get("text", "")

            # Parse JSON from the response
            # Find JSON content between curly braces
            json_match = re.search(r'\{.*\}', extraction_text, re.DOTALL)

            if json_match:
                try:
                    json_content = json_match.group(0)
                    indicators = json.loads(json_content)

                    # Ensure we have all required keys
                    if "democratic_concerns" not in indicators:
                        indicators["democratic_concerns"] = []
                    if "affected_democratic_elements" not in indicators:
                        indicators["affected_democratic_elements"] = []
                    if "overall_assessment" not in indicators:
                        indicators["overall_assessment"] = {"concern_level": 0, "main_concerns": []}

                    return indicators
                except json.JSONDecodeError:
                    self.logger.warning("Failed to parse JSON from authoritarian indicators extraction")
                    return {
                        "democratic_concerns": [],
                        "affected_democratic_elements": [],
                        "overall_assessment": {"concern_level": 0, "main_concerns": []}
                    }
            else:
                self.logger.warning("No JSON content found in authoritarian indicators extraction")
                return {
                    "democratic_concerns": [],
                    "affected_democratic_elements": [],
                    "overall_assessment": {"concern_level": 0, "main_concerns": []}
                }

        except Exception as e:
            self.logger.error(f"Error extracting authoritarian indicators: {str(e)}")
            return {
                "democratic_concerns": [],
                "affected_democratic_elements": [],
                "overall_assessment": {"concern_level": 0, "main_concerns": []}
            }

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

        # Ensure we have the structured data from the authoritarian analysis
        structured_elements = auth_analysis.get("structured_elements", {})

        # Convert the new indicator format to the old format expected by knowledge_graph.py
        converted_indicators = self._convert_indicators_for_kg(indicators)

        # Build a unified structure for the knowledge graph that maintains compatibility
        # with the existing knowledge_graph.process_article_analysis method
        structured_data = {
            "entities": entities,
            "relationships": relationships,
            "authoritarian_indicators": converted_indicators.get("indicators", []),
            "key_actors": structured_elements.get("key_actors", []),
            "key_institutions": structured_elements.get("key_institutions", []),
            "key_relationships": structured_elements.get("key_relationships", []),
            "overall_assessment": {
                "concern_level": indicators.get("overall_assessment", {}).get("concern_level", 0),
                "main_concerns": indicators.get("overall_assessment", {}).get("main_concerns", [])
            },
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

        # Log key findings for debugging
        self._log_key_findings(structured_data)

        return {
            "article": article_data,
            "structured_elements": structured_data,
            "timestamp": datetime.now().isoformat()
        }

    def _convert_indicators_for_kg(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert the new indicator format to the format expected by the knowledge graph.

        Args:
            indicators: New format indicators from extract_authoritarian_indicators

        Returns:
            Dict with converted indicators in the format expected by knowledge_graph.py
        """
        converted = {
            "indicators": []
        }

        # Map democratic concerns to authoritarian indicators
        for concern in indicators.get("democratic_concerns", []):
            indicator = {
                "type": concern.get("concern", "").lower().replace(" ", "_"),
                "present": True,
                "examples": concern.get("evidence", []),
                "severity": concern.get("severity", 5),
                "actors_involved": [],  # Will be populated from relationships if possible
                "institutions_targeted": []  # Will be populated from relationships if possible
            }
            converted["indicators"].append(indicator)

        # Map affected democratic elements to additional indicators
        for element in indicators.get("affected_democratic_elements", []):
            indicator = {
                "type": element.get("element", "").lower().replace(" ", "_"),
                "present": True,
                "examples": element.get("evidence", []),
                "severity": 5,  # Default severity
                "actors_involved": [],
                "institutions_targeted": [element.get("element", "")]
            }
            converted["indicators"].append(indicator)

        return converted

    def _log_key_findings(self, structured_data: Dict[str, Any]) -> None:
        """
        Log key findings from the structured data for debugging.

        Args:
            structured_data: The structured data to log key findings from
        """
        actors = structured_data.get("entities", {}).get("actors", [])
        institutions = structured_data.get("entities", {}).get("institutions", [])
        relationships = structured_data.get("relationships", [])
        indicators = structured_data.get("authoritarian_indicators", [])

        self.logger.info(f"Found {len(actors)} actors, {len(institutions)} institutions")
        self.logger.info(f"Extracted {len(relationships)} relationships, {len(indicators)} authoritarian indicators")

        # Log key actors
        if actors:
            actor_names = [a.get("name", "Unknown") for a in actors[:3]]
            self.logger.info(f"Key actors: {', '.join(actor_names)}")

        # Log key institutions
        if institutions:
            institution_names = [i.get("name", "Unknown") for i in institutions[:3]]
            self.logger.info(f"Key institutions: {', '.join(institution_names)}")

        # Log overall assessment
        assessment = structured_data.get("overall_assessment", {})
        concern_level = assessment.get("concern_level", 0)
        self.logger.info(f"Overall concern level: {concern_level}/10")
