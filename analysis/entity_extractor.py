"""
Night_watcher Entity Extractor
Extracts political entities and relationships from analyzed content.
"""

import logging
import re
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple

from memory.knowledge_graph import KnowledgeGraph, ENTITY_TYPES, RELATIONSHIP_TYPES
from agents.base import LLMProvider

logger = logging.getLogger(__name__)


class EntityExtractor:
    """Extracts entities and relationships from analyzed content"""

    def __init__(self, llm_provider: LLMProvider, knowledge_graph: KnowledgeGraph):
        """
        Initialize the entity extractor
        
        Args:
            llm_provider: LLM provider for extraction
            knowledge_graph: Knowledge graph to populate
        """
        self.llm_provider = llm_provider
        self.knowledge_graph = knowledge_graph
        self.logger = logging.getLogger("EntityExtractor")
        
        # Define valid entity and relationship types
        self.valid_entity_types = list(ENTITY_TYPES.values())
        self.valid_relationship_types = list(RELATIONSHIP_TYPES.values())
        
    def extract_from_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract entities and relationships from an analysis
        
        Args:
            analysis: Analysis data containing article and analysis text
            
        Returns:
            Dictionary with extraction results
        """
        if "article" not in analysis or "analysis" not in analysis:
            return {"error": "Invalid analysis format"}
            
        article = analysis["article"]
        analysis_text = analysis["analysis"]
        authoritarian_analysis = analysis.get("authoritarian_analysis", "")
        
        # Create a unique content ID for evidence tracking
        content_id = analysis.get("id", f"analysis_{uuid.uuid4().hex[:8]}")
        
        # Extract basic information
        article_title = article.get("title", "")
        article_source = article.get("source", "")
        article_content = article.get("content", "")
        
        # First extract entities using LLM
        entities = self._extract_entities(article_title, article_content, analysis_text, authoritarian_analysis)
        
        # Then extract relationships between entities
        relationships = self._extract_relationships(entities, article_title, analysis_text, authoritarian_analysis)
        
        # Process extraction results into knowledge graph
        added_entities = self._add_entities_to_graph(entities, content_id)
        added_relationships = self._add_relationships_to_graph(relationships, entities, content_id)
        
        return {
            "content_id": content_id,
            "entities": added_entities,
            "relationships": added_relationships,
            "extraction_timestamp": datetime.now().isoformat()
        }
    
    def _extract_entities(self, title: str, content: str, analysis: str, 
                     authoritarian_analysis: str = "") -> List[Dict[str, Any]]:
        """Extract entities from content using LLM"""
        # Prepare context for extraction
        context = f"""
        TITLE: {title}
        
        CONTENT SUMMARY: {content[:1000]}...
        
        ANALYSIS: {analysis}
        """
        
        if authoritarian_analysis:
            context += f"\n\nAUTHORITARIAN ANALYSIS: {authoritarian_analysis}"
        
        # Create extraction prompt with simpler formatting
        prompt = f"""
        Extract all political entities mentioned in this article and analysis. Focus on identifying governmental 
        and political entities relevant to tracking authoritarian patterns.
        
        {context}
        
        Extract and classify entities into ONLY these exact entity types:
        1. ACTOR: Individual political actors (specific people)
        2. INSTITUTION: Formal government institutions and organizations
        3. ACTION: Specific actions or decisions taken
        4. EVENT: Discrete political occurrences with date and participants
        5. ARTIFACT: Created political objects or documents
        6. NARRATIVE: Recurring storylines and framing devices
        7. INDICATOR: Authoritarian pattern indicators
        8. TOPIC: Subject areas and domains of political discourse
        
        Return your response as a JSON array where each object represents an entity.
        Each entity must have id, name, type, and attributes fields.
        
        Respond with ONLY the JSON array, no explanation or other text.
        """
        
        try:
            # Call LLM
            response = self.llm_provider.complete(
                prompt=prompt,
                max_tokens=2000,
                temperature=0.1
            )
            
            # Extract JSON from response
            text_response = response.get("choices", [{}])[0].get("text", "").strip()
            
            # Find JSON array in response
            json_match = re.search(r'\[.*\]', text_response, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
                # Parse JSON
                entities = json.loads(json_text)
                
                # Validate entity types
                validated_entities = []
                for entity in entities:
                    # Ensure entity has required fields
                    if "name" not in entity or "type" not in entity:
                        continue
                        
                    # Normalize type
                    entity_type = entity["type"].lower()
                    
                    # Check if type is valid
                    if entity_type not in self.valid_entity_types:
                        # Try to map to valid type
                        entity_type = self._map_entity_type(entity["type"])
                        if not entity_type:
                            continue
                            
                    # Update with normalized type
                    entity["type"] = entity_type
                    
                    # Ensure entity has all required fields
                    if "attributes" not in entity:
                        entity["attributes"] = {}
                    if "confidence" not in entity:
                        entity["confidence"] = "MEDIUM"
                    if "evidence" not in entity:
                        entity["evidence"] = []
                        
                    # Add to validated entities
                    validated_entities.append(entity)
                    
                self.logger.info(f"Extracted {len(validated_entities)} valid entities")
                return validated_entities
            else:
                self.logger.warning("No valid JSON array found in LLM response")
                return []
                
        except Exception as e:
            self.logger.error(f"Error extracting entities: {str(e)}")
            return []
            
    def _extract_relationships(self, entities: List[Dict[str, Any]], title: str, 
                              analysis: str, authoritarian_analysis: str = "") -> List[Dict[str, Any]]:
        """
        Extract relationships between entities
        
        Args:
            entities: List of extracted entities
            title: Article title
            analysis: Analysis text
            authoritarian_analysis: Authoritarian analysis text (optional)
            
        Returns:
            List of relationships
        """
        if not entities:
            return []
            
        # Create entity map for reference
        entity_map = {entity["name"]: entity for entity in entities}
        
        # Create entity list for prompt
        entity_list = "\n".join([f"- {entity['name']} ({entity['type'].upper()})" for entity in entities])
        
        # List all relationship types
        relationship_types = """
        POWER RELATIONSHIPS:
        - controls: Actor has authority over target
        - influences: Actor shapes decisions of target without formal authority
        - undermines: Actor/Action weakens target's effectiveness or legitimacy
        - strengthens: Actor/Action increases target's power or legitimacy
        
        ACTION RELATIONSHIPS:
        - performs: Actor executes an action
        - authorizes: Actor approves action performed by others
        - blocks: Actor prevents an action
        - responds_to: Action is created as response to another action
        
        PARTICIPATION RELATIONSHIPS:
        - participates_in: Actor is involved in event
        - organizes: Actor plans or controls event
        - targeted_by: Institution/Actor is focus of an action
        - benefits_from: Actor/Institution gains advantage from action/event
        
        TEMPORAL RELATIONSHIPS:
        - precedes: Entity exists/occurs chronologically before another
        - causes: Entity directly leads to existence/occurrence of another
        - accelerates: Entity speeds up development or occurrence of another
        - part_of: Entity is component of a larger entity
        
        NARRATIVE RELATIONSHIPS:
        - justifies: Action/Event provides rationale for action
        - contradicts: Entity logically conflicts with another
        - distracts_from: Entity diverts attention from another
        - reinforces: Entity strengthens meaning or impact of another
        
        ADDITIONAL RELATIONSHIPS:
        - allies_with: Actor works with another actor toward common goals
        - opposes: Actor works against another's interests/goals
        - delegates_to: Actor transfers responsibility or authority
        """
        
        # Create context
        context = f"""
        TITLE: {title}
        
        ANALYSIS: {analysis}
        """
        
        if authoritarian_analysis:
            context += f"\n\nAUTHORITARIAN ANALYSIS: {authoritarian_analysis}"
            
        # Create extraction prompt
        prompt = f"""
        Identify relationships between these political entities based on the article and analysis:
        
        ENTITIES:
        {entity_list}
        
        RELATIONSHIP TYPES:
        {relationship_types}
        
        ARTICLE CONTEXT:
        {context}
        
        For each relationship, include:
        - source_id: The entity ID initiating the relationship
        - target_id: The entity ID receiving the relationship
        - type: The relationship type (must be one of the types listed above)
        - strength: Numeric value between 0.1 and 1.0 representing relationship strength
        - confidence: "HIGH", "MEDIUM", or "LOW" based on clarity in the text
        - evidence: List of text snippets supporting this relationship
        - attributes: Any relevant properties (timeframe, impact, etc.)
        
        Format your response as a JSON array where each object represents a relationship:
        [
          {{
            "source_id": "entity_id_1",
            "target_id": "entity_id_2",
            "type": "relationship type",
            "strength": 0.8,
            "confidence": "HIGH",
            "evidence": [
              {{
                "text": "Extracted text evidence",
                "source_id": "content",
                "confidence": "HIGH"
              }}
            ],
            "attributes": {{
              "timeframe": "date or period",
              "impact": "NATIONAL",
              "attribute3": "value3"
            }}
          }},
          ...
        ]
        
        Extract only relationships clearly stated or strongly implied in the text.
        Only include the JSON array in your response, no other text.
        """
        
        try:
            # Call LLM
            response = self.llm_provider.complete(
                prompt=prompt,
                max_tokens=2000,
                temperature=0.1
            )
            
            # Extract JSON from response
            text_response = response.get("choices", [{}])[0].get("text", "").strip()
            
            # Find JSON array in response
            json_match = re.search(r'\[.*\]', text_response, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
                # Parse JSON
                relationships = json.loads(json_text)
                
                # Validate relationships
                validated_relationships = []
                for rel in relationships:
                    # Ensure relationship has required fields
                    if "source_id" not in rel or "target_id" not in rel or "type" not in rel:
                        continue
                        
                    # Normalize type
                    rel_type = rel["type"].lower()
                    
                    # Check if type is valid
                    if rel_type not in self.valid_relationship_types:
                        # Try to map to valid type
                        rel_type = self._map_relation_type(rel["type"])
                        if not rel_type:
                            continue
                            
                    # Update with normalized type
                    rel["type"] = rel_type
                    
                    # Ensure relationship has all required fields
                    if "strength" not in rel:
                        rel["strength"] = 0.7
                    if "confidence" not in rel:
                        rel["confidence"] = "MEDIUM"
                    if "evidence" not in rel:
                        rel["evidence"] = []
                    if "attributes" not in rel:
                        rel["attributes"] = {}
                        
                    # Add to validated relationships
                    validated_relationships.append(rel)
                
                self.logger.info(f"Extracted {len(validated_relationships)} valid relationships")
                return validated_relationships
            else:
                self.logger.warning("No valid JSON array found in LLM response")
                return []
                
        except Exception as e:
            self.logger.error(f"Error extracting relationships: {str(e)}")
            return []
            
    def _add_entities_to_graph(self, entities: List[Dict[str, Any]], 
                               content_id: str) -> Dict[str, str]:
        """
        Add extracted entities to knowledge graph
        
        Args:
            entities: List of extracted entities
            content_id: ID of the source content for evidence
            
        Returns:
            Map of entity names to their IDs in the graph
        """
        entity_map = {}
        
        for entity in entities:
            name = entity.get("name")
            entity_type = entity.get("type")
            attributes = entity.get("attributes", {})
            
            # Add subtype to attributes if present
            if "subtype" in entity:
                attributes["subtype"] = entity["subtype"]
                
            # Find or create entity in graph
            entity_id = self.knowledge_graph.find_or_create_entity(name, entity_type, attributes)
            
            # Add evidence
            for evidence in entity.get("evidence", []):
                entity_obj = self.knowledge_graph.get_entity(entity_id)
                if entity_obj:
                    evidence_text = evidence.get("text", "")
                    entity_obj.add_evidence(content_id)
                    
                    # Add evidence text to entity attributes
                    if "evidence_texts" not in entity_obj.attributes:
                        entity_obj.attributes["evidence_texts"] = []
                    entity_obj.attributes["evidence_texts"].append(evidence_text)
                    
            # Store in map
            entity_map[name] = entity_id
            
        return entity_map
        
    def _add_relationships_to_graph(self, relationships: List[Dict[str, Any]], 
                                   entity_map: Dict[str, str],
                                   content_id: str) -> List[str]:
        """
        Add extracted relationships to knowledge graph
        
        Args:
            relationships: List of extracted relationships
            entity_map: Map of entity names to their IDs in the graph
            content_id: ID of the source content for evidence
            
        Returns:
            List of created relationship IDs
        """
        relationship_ids = []
        
        for rel in relationships:
            source_id = rel.get("source_id")
            target_id = rel.get("target_id")
            relation_type = rel.get("type")
            
            # Skip if source/target don't have IDs in entity map
            source_name = self._find_entity_name_by_id(rel.get("source_id"), entity_map)
            target_name = self._find_entity_name_by_id(rel.get("target_id"), entity_map)
            
            if not source_name or not target_name:
                continue
                
            # Get entity IDs from map
            source_graph_id = entity_map.get(source_name)
            target_graph_id = entity_map.get(target_name)
            
            if not source_graph_id or not target_graph_id:
                continue
                
            # Get strength and attributes
            strength = rel.get("strength", 0.7)
            attributes = rel.get("attributes", {})
            
            # Add confidence to attributes
            attributes["confidence"] = rel.get("confidence", "MEDIUM")
            
            # Get evidence
            evidence_text = ""
            if rel.get("evidence"):
                evidence_text = rel["evidence"][0].get("text", "")
                
            # Add relationship with evidence
            rel_id = self.knowledge_graph.add_relationship_with_evidence(
                source_id=source_graph_id,
                target_id=target_graph_id,
                relation_type=relation_type,
                evidence_text=evidence_text,
                source_content_id=content_id,
                weight=strength,
                attributes=attributes
            )
            
            if rel_id:
                relationship_ids.append(rel_id)
                
        return relationship_ids
        
    def _find_entity_name_by_id(self, entity_id: str, entity_map: Dict[str, str]) -> Optional[str]:
        """Find entity name by ID in extracted entities"""
        # First check if entity_id is already a name in the map
        if entity_id in entity_map:
            return entity_id
            
        # Otherwise, try to find by ID match
        for name, graph_id in entity_map.items():
            if graph_id == entity_id:
                return name
                
        return None
        
    def _map_entity_type(self, entity_type: str) -> Optional[str]:
        """Map extraction entity type to knowledge graph type"""
        entity_type = entity_type.upper() if entity_type else ""
        
        type_mapping = {
            "ACTOR": "actor",
            "INSTITUTION": "institution",
            "ACTION": "action",
            "EVENT": "event",
            "ARTIFACT": "artifact",
            "NARRATIVE": "narrative",
            "INDICATOR": "indicator",
            "TOPIC": "topic",
            # Handle variations
            "PERSON": "actor",
            "AGENCY": "institution",
            "ORGANIZATION": "institution",
            "DOCUMENT": "artifact",
            "POLICY": "artifact",
            "LAW": "artifact",
            "STORY": "narrative",
            "THEME": "topic",
            "PATTERN": "indicator"
        }
        
        return type_mapping.get(entity_type, "").lower()
        
    def _map_relation_type(self, relation_type: str) -> Optional[str]:
        """Map extraction relation type to knowledge graph relation type"""
        if not relation_type:
            return None
            
        # Normalize to lowercase
        relation_type = relation_type.lower()
        
        # Direct mapping for standard types
        if relation_type in self.valid_relationship_types:
            return relation_type
            
        # Handle variations
        variation_mapping = {
            "control": "controls",
            "influence": "influences",
            "undermine": "undermines",
            "strengthen": "strengthens",
            "perform": "performs",
            "authorize": "authorizes",
            "block": "blocks",
            "respond to": "responds_to",
            "participate in": "participates_in",
            "organize": "organizes",
            "targeted by": "targeted_by",
            "benefit from": "benefits_from",
            "justify": "justifies",
            "contradict": "contradicts",
            "reinforce": "reinforces",
            "attack": "undermines",
            "direct": "controls",
            "manage": "controls",
            "oversee": "controls",
            "oppose": "contradicts",
            "support": "strengthens",
            "assist": "strengthens",
            "create": "performs",
            "ally with": "allies_with",
            "collaborate with": "allies_with",
            "delegate to": "delegates_to",
            "distract from": "distracts_from",
            "precede": "precedes",
            "cause": "causes",
            "accelerate": "accelerates",
            "part of": "part_of",
            "belongs to": "part_of"
        }
        
        return variation_mapping.get(relation_type)
        
    def batch_process(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process multiple analyses at once
        
        Args:
            analyses: List of analyses to process
            
        Returns:
            Results of batch processing
        """
        results = {
            "total": len(analyses),
            "processed": 0,
            "entities": 0,
            "relationships": 0,
            "errors": 0,
            "entity_types": {},
            "relationship_types": {},
            "confidence_levels": {
                "HIGH": 0,
                "MEDIUM": 0,
                "LOW": 0
            },
            "extraction_timestamp": datetime.now().isoformat()
        }
        
        for analysis in analyses:
            try:
                extraction = self.extract_from_analysis(analysis)
                
                if "error" in extraction:
                    results["errors"] += 1
                    continue
                    
                # Count results
                results["processed"] += 1
                results["entities"] += len(extraction.get("entities", {}))
                results["relationships"] += len(extraction.get("relationships", []))
                
                # Count entity types
                for entity_name, entity_id in extraction.get("entities", {}).items():
                    entity = self.knowledge_graph.get_entity(entity_id)
                    if entity:
                        entity_type = entity.type
                        results["entity_types"][entity_type] = results["entity_types"].get(entity_type, 0) + 1
                        
                # Count relationship types
                for rel_id in extraction.get("relationships", []):
                    rel = self.knowledge_graph.graph.get_relationship(rel_id)
                    if rel:
                        rel_type = rel.type
                        results["relationship_types"][rel_type] = results["relationship_types"].get(rel_type, 0) + 1
                        
                        # Count confidence levels
                        confidence = rel.attributes.get("confidence", "MEDIUM")
                        results["confidence_levels"][confidence] = results["confidence_levels"].get(confidence, 0) + 1
                        
            except Exception as e:
                self.logger.error(f"Error processing analysis: {str(e)}")
                results["errors"] += 1
                
        return results
        
    def extract_patterns(self, lookback_days: int = 30) -> Dict[str, Any]:
        """
        Extract patterns from the knowledge graph
        
        Args:
            lookback_days: Number of days to look back
            
        Returns:
            Dictionary with pattern analysis results
        """
        patterns = {
            "coordination_patterns": [],
            "escalation_patterns": [],
            "democratic_erosion": None,
            "influential_actors": [],
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        # Get coordination patterns
        try:
            coordination_patterns = self.knowledge_graph.detect_coordination_patterns(lookback_days)
            patterns["coordination_patterns"] = coordination_patterns
        except Exception as e:
            self.logger.error(f"Error detecting coordination patterns: {str(e)}")
            
        # Get democratic erosion analysis
        try:
            erosion_analysis = self.knowledge_graph.analyze_democratic_erosion(lookback_days)
            patterns["democratic_erosion"] = erosion_analysis
        except Exception as e:
            self.logger.error(f"Error analyzing democratic erosion: {str(e)}")
            
        # Get influential actors
        try:
            influential_actors = self.knowledge_graph.get_influential_actors(10)
            patterns["influential_actors"] = influential_actors
        except Exception as e:
            self.logger.error(f"Error getting influential actors: {str(e)}")
            
        # For each influential actor, check for escalation patterns
        if patterns["influential_actors"]:
            for actor in patterns["influential_actors"][:5]:  # Top 5 actors
                try:
                    actor_id = actor.get("id")
                    if actor_id:
                        escalation = self.knowledge_graph.analyze_actor_patterns(actor.get("name", ""), lookback_days)
                        if escalation.get("escalation_detected"):
                            patterns["escalation_patterns"].append(escalation)
                except Exception as e:
                    self.logger.error(f"Error analyzing actor escalation: {str(e)}")
                    
        return patterns
