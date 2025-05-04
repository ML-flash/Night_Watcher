"""
This is a focused update for the analyze_content_for_knowledge_graph method in analyzer.py
to ensure proper integration with the knowledge graph.
"""

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
        "structured_data": structured_data,
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