"""
Night_watcher Entity Extractor
Extracts political entities and relationships from analyzed content.
"""

import logging
import re
import json
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
        content_id = analysis.get("id", f"analysis_{datetime.now().strftime('%Y%m%d%H%M%S')}")
        
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
        """
        Extract entities from content using LLM
        
        Args:
            title: Article title
            content: Article content
            analysis: Analysis text
            authoritarian_analysis: Authoritarian analysis text (optional)
            
        Returns:
            List of extracted entities
        """
        # Prepare context for extraction
        context = f"""
        TITLE: {title}
        
        CONTENT SUMMARY: {content[:1000]}...
        
        ANALYSIS: {analysis}
        """
        
        if authoritarian_analysis:
            context += f"\n\nAUTHORITARIAN ANALYSIS: {authoritarian_analysis}"
        
        # Create extraction prompt
        prompt = f"""
        Extract all political entities mentioned in this article and analysis. Focus on identifying governmental 
        and political entities relevant to tracking authoritarian patterns.
        
        {context}
        
        Extract and classify the following entity types:
        1. ACTORS: Individual political actors (specific people)
        2. INSTITUTIONS: Formal government institutions (agencies, courts, committees, etc.)
        3. ACTIONS: Specific actions or decisions taken (orders, votes, statements, etc.)
        4. EVENTS: Discrete political events (hearings, speeches, rallies, etc.)
        5. ARTIFACTS: Created political objects (laws, policies, documents, etc.)
        
        For each entity, include:
        - Name (normalized/canonical form)
        - Type (one of the categories above)
        - Description (brief factual description)
        - Attributes (any relevant properties like political affiliation, role, status, etc.)
        
        Format your response as a JSON array where each object represents an entity:
        [
          {{
            "name": "entity name",
            "type": "ACTOR/INSTITUTION/ACTION/EVENT/ARTIFACT",
            "description": "brief description",
            "attributes": {{"key1": "value1", "key2": "value2"}}
          }},
          ...
        ]
        
        Extract only entities clearly mentioned or implied in the text. Exclude speculative entities.
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
                entities = json.loads(json_text)
                self.logger.info(f"Extracted {len(entities)} entities")
                return entities
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
        entity_list = "\n".join([f"- {entity['name']} ({entity['type']})" for entity in entities])
        
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
        
        ARTICLE CONTEXT:
        {context}
        
        Extract relationships between these entities, focusing on connections relevant to tracking authoritarian patterns.
        
        Consider these relationship types:
        - controls: Actor has authority over target
        - influences: Actor shapes decisions of target without formal authority
        - undermines: Actor/Action weakens the target's effectiveness or legitimacy
        - strengthens: Actor/Action increases target's power or legitimacy
        - performs: Actor executes an action
        - authorizes: Actor approves an action performed by others
        - blocks: Actor prevents an action
        - responds_to: Action is created as response to another action
        - participates_in: Actor is involved in an event
        - organizes: Actor plans or controls an event
        - targeted_by: Institution/Actor is the focus of an action
        - benefits_from: Actor/Institution gains advantage from action/event
        - justifies: Action/Event provides rationale for action
        - contradicts: Any entity logically conflicts with another
        - reinforces: Any entity strengthens meaning or impact of another
        
        For each relationship, include:
        - source: The entity initiating the relationship
        - relation: The type of relationship (from list above)
        - target: The entity receiving the relationship
        - confidence: How clearly this relationship is stated (high/medium/low)
        - evidence: Brief text from the article or analysis supporting this relationship
        - attributes: Any relevant properties of this relationship
        
        Format your response as a JSON array where each object represents a relationship:
        [
          {{
            "source": "source entity name",
            "relation": "relationship type",
            "target": "target entity name",
            "confidence": "high/medium/low",
            "evidence": "text supporting this relationship",
            "attributes": {{"key1": "value1", "key2": "value2"}}
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
                
                # Filter relationships to only include entities we extracted
                valid_relationships = []
                for rel in relationships:
                    source = rel.get("source")
                    target = rel.get("target")
                    
                    if source in entity_map and target in entity_map:
                        # Add entity types for reference
                        rel["source_type"] = entity_map[source]["type"]
                        rel["target_type"] = entity_map[target]["type"]
                        valid_relationships.append(rel)
                
                self.logger.info(f"Extracted {len(valid_relationships)} valid relationships")
                return valid_relationships
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
            description = entity.get("description", "")
            attributes = entity.get("attributes", {})
            
            # Map entity type to knowledge graph type
            kg_type = self._map_entity_type(entity_type)
            if not kg_type:
                continue
                
            # Add description to attributes
            if description:
                attributes["description"] = description
                
            # Find or create entity in graph
            entity_id = self.knowledge_graph.find_or_create_entity(name, kg_type, attributes)
            
            # Add evidence source
            entity_obj = self.knowledge_graph.get_entity(entity_id)
            if entity_obj:
                entity_obj.add_evidence(content_id)
                
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
            source_name = rel.get("source")
            target_name = rel.get("target")
            relation_type = rel.get("relation")
            confidence = rel.get("confidence", "medium")
            evidence = rel.get("evidence", "")
            attributes = rel.get("attributes", {})
            
            # Skip if we don't have entity IDs
            if source_name not in entity_map or target_name not in entity_map:
                continue
                
            # Get entity IDs
            source_id = entity_map[source_name]
            target_id = entity_map[target_name]
            
            # Map relation type if needed
            rel_type = self._map_relation_type(relation_type)
            if not rel_type:
                continue
                
            # Calculate weight based on confidence
            weight = 1.0
            if confidence == "high":
                weight = 1.0
            elif confidence == "medium":
                weight = 0.7
            elif confidence == "low":
                weight = 0.4
                
            # Add confidence to attributes
            attributes["confidence"] = confidence
            
            # Add relationship with evidence
            rel_id = self.knowledge_graph.add_relationship_with_evidence(
                source_id=source_id,
                target_id=target_id,
                relation_type=rel_type,
                evidence_text=evidence,
                source_content_id=content_id,
                weight=weight,
                attributes=attributes
            )
            
            if rel_id:
                relationship_ids.append(rel_id)
                
        return relationship_ids
        
    def _map_entity_type(self, entity_type: str) -> Optional[str]:
        """Map extraction entity type to knowledge graph type"""
        entity_type = entity_type.upper() if entity_type else ""
        
        type_mapping = {
            "ACTOR": "actor",
            "INSTITUTION": "institution",
            "ACTION": "action",
            "EVENT": "event",
            "ARTIFACT": "artifact",
            # Handle variations
            "PERSON": "actor",
            "AGENCY": "institution",
            "ORGANIZATION": "institution",
            "DOCUMENT": "artifact",
            "POLICY": "artifact",
            "LAW": "artifact"
        }
        
        return type_mapping.get(entity_type)
        
    def _map_relation_type(self, relation_type: str) -> Optional[str]:
        """Map extraction relation type to knowledge graph relation type"""
        if not relation_type:
            return None
            
        # Normalize to lowercase
        relation_type = relation_type.lower()
        
        # Direct mapping for standard types
        if relation_type in RELATIONSHIP_TYPES.values():
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
            "create": "performs"
        }
        
        return variation_mapping.get(relation_type, relation_type)
        
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
            "extracted_entity_types": {},
            "extracted_relationship_types": {},
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
                        results["extracted_entity_types"][entity_type] = results["extracted_entity_types"].get(entity_type, 0) + 1
                        
                # Count relationship types
                for rel_id in extraction.get("relationships", []):
                    rel = self.knowledge_graph.graph.get_relationship(rel_id)
                    if rel:
                        rel_type = rel.type
                        results["extracted_relationship_types"][rel_type] = results["extracted_relationship_types"].get(rel_type, 0) + 1
                        
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
