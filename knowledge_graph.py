"""
Night_watcher Knowledge Graph
Entity-relationship mapping for authoritarian patterns with advanced pattern detection.
"""

import os
import pickle
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Set

import numpy as np

# Optional imports for enhanced functionality
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

# Entity type definitions
ENTITY_TYPES = {
    # Key political entities
    "ACTOR": "actor",              # Individual political actors (people)
    "INSTITUTION": "institution",  # Formal institutions (agencies, courts, etc.)
    "EVENT": "event",              # Discrete occurrences (hearings, votes, etc.)
    "ACTION": "action",            # Specific actions taken (orders, statements, etc.)
    "ARTIFACT": "artifact",        # Created objects (laws, policies, documents)

    # Analysis entities
    "NARRATIVE": "narrative",      # Recurring storylines and framing
    "INDICATOR": "indicator",      # Authoritarian pattern indicators
    "TOPIC": "topic",              # Subject areas and domains
}

# Relationship categories
RELATIONSHIP_TYPES = {
    # Power relationships
    "CONTROLS": "controls",
    "INFLUENCES": "influences",
    "UNDERMINES": "undermines",
    "STRENGTHENS": "strengthens",

    # Action relationships
    "PERFORMS": "performs",
    "AUTHORIZES": "authorizes",
    "BLOCKS": "blocks",
    "RESPONDS_TO": "responds_to",

    # Participation relationships
    "PARTICIPATES_IN": "participates_in",
    "ORGANIZES": "organizes",
    "TARGETED_BY": "targeted_by",
    "BENEFITS_FROM": "benefits_from",

    # Temporal relationships
    "PRECEDES": "precedes",
    "CAUSES": "causes",
    "ACCELERATES": "accelerates",
    "PART_OF": "part_of",

    # Narrative relationships
    "JUSTIFIES": "justifies",
    "CONTRADICTS": "contradicts",
    "DISTRACTS_FROM": "distracts_from",
    "REINFORCES": "reinforces",

    # Additional relationships
    "ALLIES_WITH": "allies_with",
    "OPPOSES": "opposes",
    "DELEGATES_TO": "delegates_to",
}

# Relationship constraints for entity types
RELATIONSHIP_CONSTRAINTS = {
    "controls": {
        "source": ["actor"],
        "target": ["actor", "institution"]
    },
    "influences": {
        "source": ["actor"],
        "target": ["actor", "institution", "artifact"]
    },
    "undermines": {
        "source": ["actor", "action"],
        "target": ["institution", "artifact"]
    },
    "strengthens": {
        "source": ["actor", "action"],
        "target": ["actor", "institution"]
    },
    "performs": {
        "source": ["actor"],
        "target": ["action"]
    },
    "authorizes": {
        "source": ["actor"],
        "target": ["action"]
    },
    "blocks": {
        "source": ["actor"],
        "target": ["action", "artifact"]
    },
    "responds_to": {
        "source": ["action"],
        "target": ["action", "event"]
    },
    "participates_in": {
        "source": ["actor"],
        "target": ["event"]
    },
    "organizes": {
        "source": ["actor"],
        "target": ["event"]
    },
    "targeted_by": {
        "source": ["institution", "actor"],
        "target": ["action"]
    },
    "benefits_from": {
        "source": ["actor", "institution"],
        "target": ["action", "event"]
    },
    # Temporal relationships are more flexible
    "precedes": {
        "source": ["action", "event"],
        "target": ["action", "event"]
    },
    "causes": {
        "source": ["action", "event"],
        "target": ["action", "event"]
    },
    "accelerates": {
        "source": ["action", "event"],
        "target": ["action", "event"]
    },
    "part_of": {
        "source": ["action", "event", "actor", "institution", "artifact"],
        "target": ["action", "event", "actor", "institution", "artifact"]
    },
    "justifies": {
        "source": ["action", "event"],
        "target": ["action"]
    },
    "contradicts": {
        "source": ["action", "artifact", "narrative"],
        "target": ["action", "artifact", "narrative"]
    },
    "distracts_from": {
        "source": ["action", "event"],
        "target": ["action", "event"]
    },
    "reinforces": {
        "source": ["action", "narrative"],
        "target": ["action", "narrative", "indicator"]
    },
    "allies_with": {
        "source": ["actor"],
        "target": ["actor"]
    },
    "opposes": {
        "source": ["actor"],
        "target": ["actor", "action", "narrative"]
    },
    "delegates_to": {
        "source": ["actor"],
        "target": ["actor"]
    }
}


class Entity:
    """Base class for entities in the knowledge graph"""

    def __init__(self, entity_id: str = None, entity_type: str = "entity",
                 name: str = "", attributes: Dict[str, Any] = None):
        """Initialize entity with ID and attributes"""
        self.id = entity_id or f"{entity_type}_{uuid.uuid4().hex[:8]}"
        self.type = entity_type
        self.name = name
        self.attributes = attributes or {}
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at

        # Track evidence sources
        self.evidence_sources = set()

    def update(self, attributes: Dict[str, Any]) -> None:
        """Update entity attributes"""
        self.attributes.update(attributes)
        self.updated_at = datetime.now().isoformat()

    def add_evidence(self, source_id: str) -> None:
        """Add evidence source for this entity"""
        self.evidence_sources.add(source_id)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "attributes": self.attributes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "evidence_sources": list(self.evidence_sources)  # Convert set to list for serialization
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Entity':
        """Create from dictionary"""
        entity = cls(
            entity_id=data.get("id"),
            entity_type=data.get("type", "entity"),
            name=data.get("name", ""),
            attributes=data.get("attributes", {})
        )
        entity.created_at = data.get("created_at", entity.created_at)
        entity.updated_at = data.get("updated_at", entity.updated_at)

        # Restore evidence sources
        evidence_sources = data.get("evidence_sources", [])
        entity.evidence_sources = set(evidence_sources)

        return entity


class Relationship:
    """Represents a relationship between entities in the knowledge graph"""

    def __init__(self, source_id: str, target_id: str, relation_type: str,
                 relation_id: str = None, weight: float = 1.0,
                 attributes: Dict[str, Any] = None):
        """Initialize relationship with source, target, and attributes"""
        self.id = relation_id or f"rel_{uuid.uuid4().hex[:8]}"
        self.source_id = source_id
        self.target_id = target_id
        self.type = relation_type
        self.weight = weight
        self.attributes = attributes or {}
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at

        # Track evidence
        self.evidence = []
        self.evidence_sources = set()
        self.confidence = "low"  # Default confidence level

    def update(self, weight: float = None, attributes: Dict[str, Any] = None) -> None:
        """Update relationship weight and/or attributes"""
        if weight is not None:
            self.weight = weight

        if attributes:
            self.attributes.update(attributes)

        self.updated_at = datetime.now().isoformat()

    def add_evidence(self, evidence_text: str, source_id: str, confidence: str = "medium") -> None:
        """Add evidence supporting this relationship"""
        self.evidence.append({
            "text": evidence_text,
            "source": source_id,
            "timestamp": datetime.now().isoformat()
        })
        self.evidence_sources.add(source_id)

        # Update confidence based on evidence quantity
        if len(self.evidence) >= 3:
            self.confidence = "high"
        elif len(self.evidence) >= 1:
            self.confidence = "medium"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.type,
            "weight": self.weight,
            "attributes": self.attributes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "evidence": self.evidence,
            "evidence_sources": list(self.evidence_sources),
            "confidence": self.confidence
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Relationship':
        """Create from dictionary"""
        rel = cls(
            source_id=data.get("source_id", ""),
            target_id=data.get("target_id", ""),
            relation_type=data.get("type", ""),
            relation_id=data.get("id"),
            weight=data.get("weight", 1.0),
            attributes=data.get("attributes", {})
        )
        rel.created_at = data.get("created_at", rel.created_at)
        rel.updated_at = data.get("updated_at", rel.updated_at)

        # Restore evidence data
        rel.evidence = data.get("evidence", [])
        if "evidence_sources" in data:
            rel.evidence_sources = set(data["evidence_sources"])
        if "confidence" in data:
            rel.confidence = data["confidence"]

        return rel


class SimpleGraph:
    """Simple implementation of a knowledge graph without NetworkX"""

    def __init__(self):
        """Initialize an empty graph"""
        self.entities = {}  # id -> Entity
        self.relationships = {}  # id -> Relationship
        self.source_relations = {}  # source_id -> [relation_ids]
        self.target_relations = {}  # target_id -> [relation_ids]
        self.entity_types = {}  # type -> [entity_ids]
        self.relation_types = {}  # type -> [relation_ids]

    def add_entity(self, entity: Entity) -> str:
        """Add an entity to the graph"""
        entity_id = entity.id
        self.entities[entity_id] = entity

        # Index by type
        if entity.type not in self.entity_types:
            self.entity_types[entity.type] = []
        self.entity_types[entity.type].append(entity_id)

        return entity_id

    def add_relationship(self, relationship: Relationship) -> str:
        """Add a relationship to the graph"""
        # Validate relationship according to constraints
        relation_type = relationship.type
        if relation_type in RELATIONSHIP_CONSTRAINTS:
            constraints = RELATIONSHIP_CONSTRAINTS[relation_type]

            # Check if source entity type is valid for this relationship
            source_entity = self.entities.get(relationship.source_id)
            if source_entity and source_entity.type not in constraints["source"]:
                logger.warning(f"Invalid source entity type '{source_entity.type}' for relationship '{relation_type}'")
                return ""

            # Check if target entity type is valid for this relationship
            target_entity = self.entities.get(relationship.target_id)
            if target_entity and target_entity.type not in constraints["target"]:
                logger.warning(f"Invalid target entity type '{target_entity.type}' for relationship '{relation_type}'")
                return ""

        relation_id = relationship.id
        self.relationships[relation_id] = relationship

        # Index by source and target
        if relationship.source_id not in self.source_relations:
            self.source_relations[relationship.source_id] = []
        self.source_relations[relationship.source_id].append(relation_id)

        if relationship.target_id not in self.target_relations:
            self.target_relations[relationship.target_id] = []
        self.target_relations[relationship.target_id].append(relation_id)

        # Index by type
        if relationship.type not in self.relation_types:
            self.relation_types[relationship.type] = []
        self.relation_types[relationship.type].append(relation_id)

        return relation_id

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID"""
        return self.entities.get(entity_id)

    def get_relationship(self, relation_id: str) -> Optional[Relationship]:
        """Get a relationship by ID"""
        return self.relationships.get(relation_id)

    def get_relationships_from(self, entity_id: str) -> List[Relationship]:
        """Get all relationships originating from an entity"""
        relation_ids = self.source_relations.get(entity_id, [])
        return [self.relationships[rel_id] for rel_id in relation_ids]

    def get_relationships_to(self, entity_id: str) -> List[Relationship]:
        """Get all relationships targeting an entity"""
        relation_ids = self.target_relations.get(entity_id, [])
        return [self.relationships[rel_id] for rel_id in relation_ids]

    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Get all entities of a specific type"""
        entity_ids = self.entity_types.get(entity_type, [])
        return [self.entities[entity_id] for entity_id in entity_ids]

    def get_relationships_by_type(self, relation_type: str) -> List[Relationship]:
        """Get all relationships of a specific type"""
        relation_ids = self.relation_types.get(relation_type, [])
        return [self.relationships[rel_id] for rel_id in relation_ids]

    def find_entity_by_name(self, name: str, entity_type: Optional[str] = None) -> Optional[Entity]:
        """Find an entity by name, optionally filtering by type"""
        for entity_id, entity in self.entities.items():
            if entity.name.lower() == name.lower():
                if entity_type is None or entity.type == entity_type:
                    return entity
        return None

    def find_path(self, source_id: str, target_id: str, max_depth: int = 3) -> List[List[Tuple[str, str]]]:
        """Find paths between two entities up to a maximum depth"""
        if source_id not in self.entities or target_id not in self.entities:
            return []

        paths = []
        visited = set()

        def dfs(current_id, path, depth):
            if depth > max_depth:
                return

            if current_id == target_id:
                paths.append(path.copy())
                return

            visited.add(current_id)

            for rel_id in self.source_relations.get(current_id, []):
                rel = self.relationships[rel_id]
                next_id = rel.target_id

                if next_id not in visited:
                    path.append((rel.type, next_id))
                    dfs(next_id, path, depth + 1)
                    path.pop()

            visited.remove(current_id)

        dfs(source_id, [], 0)
        return paths

    def get_entity_network(self, entity_id: str, max_depth: int = 2) -> Dict[str, Any]:
        """Get a network of entities connected to the given entity"""
        if entity_id not in self.entities:
            return {"entities": {}, "relationships": {}}

        entities = {}
        relationships = {}
        visited = set()

        def dfs(current_id, depth):
            if depth > max_depth or current_id in visited:
                return

            visited.add(current_id)
            entities[current_id] = self.entities[current_id].to_dict()

            # Add outgoing relationships
            for rel_id in self.source_relations.get(current_id, []):
                rel = self.relationships[rel_id]
                relationships[rel_id] = rel.to_dict()
                dfs(rel.target_id, depth + 1)

            # Add incoming relationships
            for rel_id in self.target_relations.get(current_id, []):
                rel = self.relationships[rel_id]
                relationships[rel_id] = rel.to_dict()
                dfs(rel.source_id, depth + 1)

        dfs(entity_id, 0)

        return {
            "entities": entities,
            "relationships": relationships
        }

    def find_coordination_patterns(self, timeframe_days: int = 30) -> List[Dict[str, Any]]:
        """
        Find potential coordination patterns between actors.

        Args:
            timeframe_days: Number of days to look back for coordination patterns

        Returns:
            List of detected coordination patterns
        """
        # Get timeframe cutoff
        cutoff_date = datetime.now() - timedelta(days=timeframe_days)
        cutoff_str = cutoff_date.isoformat()

        patterns = []

        # Get all relationships of relevant types
        relationships_to_check = []
        for rel_type in ["performs", "authorizes", "allies_with"]:
            relationships_to_check.extend(self.get_relationships_by_type(rel_type))

        # Filter by timeframe
        recent_relationships = [
            rel for rel in relationships_to_check
            if rel.created_at >= cutoff_str
        ]

        # Group by target entity (action/event)
        target_groups = {}
        for rel in recent_relationships:
            if rel.target_id not in target_groups:
                target_groups[rel.target_id] = []
            target_groups[rel.target_id].append(rel)

        # Look for coordination (multiple actors targeting same entity in short timeframe)
        for target_id, rels in target_groups.items():
            if len(rels) >= 2:
                # Get unique source actors
                source_actors = set()
                for rel in rels:
                    source_entity = self.get_entity(rel.source_id)
                    if source_entity and source_entity.type == "actor":
                        source_actors.add(rel.source_id)

                # If we have multiple actors targeting same entity, check for time proximity
                if len(source_actors) >= 2:
                    target_entity = self.get_entity(target_id)

                    # Create pattern record
                    pattern = {
                        "pattern_type": "coordination",
                        "target_entity": target_entity.to_dict() if target_entity else {"id": target_id},
                        "actors": [self.get_entity(actor_id).to_dict() for actor_id in source_actors],
                        "relationships": [rel.to_dict() for rel in rels],
                        "confidence": "medium",
                        "detection_time": datetime.now().isoformat()
                    }

                    patterns.append(pattern)

        return patterns

    def find_escalation_patterns(self, actor_id: str, lookback_days: int = 90) -> Dict[str, Any]:
        """
        Detect escalation patterns for a specific actor.

        Args:
            actor_id: ID of the actor to analyze
            lookback_days: Number of days to look back

        Returns:
            Dictionary with escalation pattern information
        """
        # Get cutoff date
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        cutoff_str = cutoff_date.isoformat()

        # Get all actions performed by this actor
        performed_relationships = []
        for rel in self.get_relationships_by_type("performs"):
            if rel.source_id == actor_id and rel.created_at >= cutoff_str:
                performed_relationships.append(rel)

        # Sort by creation date
        performed_relationships.sort(key=lambda x: x.created_at)

        # Check for escalation in undermining democratic institutions
        undermining_actions = []
        first_half_avg = 0
        second_half_avg = 0

        for rel in performed_relationships:
            # Check if this action undermines an institution
            action_id = rel.target_id
            for undermining_rel in self.get_relationships_by_type("undermines"):
                if undermining_rel.source_id == action_id:
                    target_entity = self.get_entity(undermining_rel.target_id)
                    if target_entity and target_entity.type == "institution":
                        # Add to undermining actions
                        action_entity = self.get_entity(action_id)
                        if action_entity:
                            undermining_actions.append({
                                "action": action_entity.to_dict(),
                                "target_institution": target_entity.to_dict(),
                                "timestamp": rel.created_at,
                                "severity": undermining_rel.weight
                            })

        # Check if we have escalation (increasing severity over time)
        escalation_detected = False
        if len(undermining_actions) >= 2:
            # Check if later actions have higher severity on average
            midpoint = len(undermining_actions) // 2
            first_half = undermining_actions[:midpoint]
            second_half = undermining_actions[midpoint:]

            first_half_avg = sum(a["severity"] for a in first_half) / len(first_half) if first_half else 0
            second_half_avg = sum(a["severity"] for a in second_half) / len(second_half) if second_half else 0

            escalation_detected = second_half_avg > first_half_avg

        return {
            "actor_id": actor_id,
            "actor": self.get_entity(actor_id).to_dict() if self.get_entity(actor_id) else {"id": actor_id},
            "undermining_actions": undermining_actions,
            "escalation_detected": escalation_detected,
            "escalation_factor": second_half_avg / first_half_avg if first_half_avg > 0 else 1.0,
            "lookback_days": lookback_days,
            "detection_time": datetime.now().isoformat()
        }

    def identify_narrative_shifts(self, topic_id: str, lookback_days: int = 90) -> Dict[str, Any]:
        """
        Identify shifts in narratives around a specific topic.

        Args:
            topic_id: ID of the topic to analyze
            lookback_days: Number of days to look back

        Returns:
            Dictionary with narrative shift information
        """
        # Get cutoff date
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        cutoff_str = cutoff_date.isoformat()

        # Get all narratives related to this topic
        narratives = []

        # Check relationships that might connect narratives to topics
        for rel_type in ["relates_to", "focuses_on", "part_of"]:
            for rel in self.get_relationships_by_type(rel_type):
                if rel.target_id == topic_id and rel.created_at >= cutoff_str:
                    source_entity = self.get_entity(rel.source_id)
                    if source_entity and source_entity.type == "narrative":
                        narratives.append({
                            "narrative": source_entity.to_dict(),
                            "relationship": rel.to_dict(),
                            "timestamp": rel.created_at
                        })

        # Sort narratives by timestamp
        narratives.sort(key=lambda x: x["timestamp"])

        # Group narratives by time periods (e.g., weeks)
        time_periods = {}
        for narrative in narratives:
            timestamp = datetime.fromisoformat(narrative["timestamp"])
            week = timestamp.isocalendar()[1]  # Get ISO week number
            year = timestamp.year
            period_key = f"{year}-W{week}"

            if period_key not in time_periods:
                time_periods[period_key] = []

            time_periods[period_key].append(narrative)

        # Identify shifts between periods
        narrative_shifts = []
        sorted_periods = sorted(time_periods.keys())

        for i in range(1, len(sorted_periods)):
            prev_period = sorted_periods[i-1]
            curr_period = sorted_periods[i]

            prev_narratives = time_periods[prev_period]
            curr_narratives = time_periods[curr_period]

            # Simple detection: new narratives appearing
            prev_narrative_ids = {n["narrative"]["id"] for n in prev_narratives}
            curr_narrative_ids = {n["narrative"]["id"] for n in curr_narratives}

            new_narratives = curr_narrative_ids - prev_narrative_ids

            if new_narratives:
                narrative_shifts.append({
                    "from_period": prev_period,
                    "to_period": curr_period,
                    "new_narratives": [
                        n["narrative"] for n in curr_narratives
                        if n["narrative"]["id"] in new_narratives
                    ],
                    "shift_type": "new_narrative_introduction"
                })

        return {
            "topic_id": topic_id,
            "topic": self.get_entity(topic_id).to_dict() if self.get_entity(topic_id) else {"id": topic_id},
            "time_periods": {k: len(v) for k, v in time_periods.items()},
            "narrative_shifts": narrative_shifts,
            "total_narratives": len(narratives),
            "lookback_days": lookback_days,
            "detection_time": datetime.now().isoformat()
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "entities": {eid: entity.to_dict() for eid, entity in self.entities.items()},
            "relationships": {rid: rel.to_dict() for rid, rel in self.relationships.items()},
            "metadata": {
                "entity_count": len(self.entities),
                "relationship_count": len(self.relationships),
                "entity_types": {t: len(ids) for t, ids in self.entity_types.items()},
                "relation_types": {t: len(ids) for t, ids in self.relation_types.items()},
                "timestamp": datetime.now().isoformat()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimpleGraph':
        """Create from dictionary"""
        graph = cls()

        # Load entities
        for entity_id, entity_data in data.get("entities", {}).items():
            entity = Entity.from_dict(entity_data)
            graph.add_entity(entity)

        # Load relationships
        for rel_id, rel_data in data.get("relationships", {}).items():
            rel = Relationship.from_dict(rel_data)
            graph.add_relationship(rel)

        return graph


# Only include NetworkXGraph if NetworkX is available
if NETWORKX_AVAILABLE:
    class NetworkXGraph(SimpleGraph):
        """Knowledge graph implementation using NetworkX for advanced analysis"""

        def __init__(self):
            """Initialize an empty graph"""
            super().__init__()
            self.graph = nx.MultiDiGraph()

        def add_entity(self, entity: Entity) -> str:
            """Add an entity to the graph"""
            entity_id = entity.id

            # Add node to NetworkX graph
            self.graph.add_node(entity_id, **entity.to_dict())

            # Update internal indexes
            self.entities[entity_id] = entity

            # Index by type
            if entity.type not in self.entity_types:
                self.entity_types[entity.type] = []
            self.entity_types[entity.type].append(entity_id)

            return entity_id

        def add_relationship(self, relationship: Relationship) -> str:
            """Add a relationship to the graph"""
            # Validate relationship according to constraints
            relation_type = relationship.type
            if relation_type in RELATIONSHIP_CONSTRAINTS:
                constraints = RELATIONSHIP_CONSTRAINTS[relation_type]

                # Check if source entity type is valid for this relationship
                source_entity = self.entities.get(relationship.source_id)
                if source_entity and source_entity.type not in constraints["source"]:
                    logger.warning(f"Invalid source entity type '{source_entity.type}' for relationship '{relation_type}'")
                    return ""

                # Check if target entity type is valid for this relationship
                target_entity = self.entities.get(relationship.target_id)
                if target_entity and target_entity.type not in constraints["target"]:
                    logger.warning(f"Invalid target entity type '{target_entity.type}' for relationship '{relation_type}'")
                    return ""

            relation_id = relationship.id
            source_id = relationship.source_id
            target_id = relationship.target_id

            # Add edge to NetworkX graph
            self.graph.add_edge(source_id, target_id, key=relation_id, **relationship.to_dict())

            # Update internal indexes
            self.relationships[relation_id] = relationship

            # Index by source and target
            if source_id not in self.source_relations:
                self.source_relations[source_id] = []
            self.source_relations[source_id].append(relation_id)

            if target_id not in self.target_relations:
                self.target_relations[target_id] = []
            self.target_relations[target_id].append(relation_id)

            # Index by type
            if relationship.type not in self.relation_types:
                self.relation_types[relationship.type] = []
            self.relation_types[relationship.type].append(relation_id)

            return relation_id

        def find_path(self, source_id: str, target_id: str, max_depth: int = 3) -> List[List[Tuple[str, str]]]:
            """Find paths between two entities up to a maximum depth"""
            if source_id not in self.entities or target_id not in self.entities:
                return []

            try:
                # Use NetworkX's shortest path algorithm
                paths = list(nx.all_simple_paths(self.graph, source_id, target_id, cutoff=max_depth))

                # Convert to the format: [(rel_type, entity_id), ...]
                result = []
                for path in paths:
                    formatted_path = []
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i+1]
                        edge_data = self.graph.get_edge_data(u, v)
                        # Use the first edge if there are multiple edges
                        first_edge_key = list(edge_data.keys())[0]
                        rel_type = edge_data[first_edge_key]["type"]
                        formatted_path.append((rel_type, v))
                    result.append(formatted_path)

                return result
            except Exception as e:
                logger.error(f"Error finding path: {str(e)}")
                return []

        def get_entity_network(self, entity_id: str, max_depth: int = 2) -> Dict[str, Any]:
            """Get a network of entities connected to the given entity"""
            if entity_id not in self.entities:
                return {"entities": {}, "relationships": {}}

            # Use NetworkX's ego graph function
            ego_graph = nx.ego_graph(self.graph, entity_id, radius=max_depth, undirected=True)

            entities = {}
            relationships = {}

            # Extract entities
            for node in ego_graph.nodes:
                entities[node] = self.entities[node].to_dict()

            # Extract relationships
            for u, v, k in ego_graph.edges(keys=True):
                rel_id = k
                relationships[rel_id] = self.relationships[rel_id].to_dict()

            return {
                "entities": entities,
                "relationships": relationships
            }

        def calculate_centrality(self, centrality_type: str = "degree") -> Dict[str, float]:
            """Calculate centrality measures for entities"""
            if centrality_type == "degree":
                return nx.degree_centrality(self.graph)
            elif centrality_type == "betweenness":
                return nx.betweenness_centrality(self.graph)
            elif centrality_type == "eigenvector":
                return nx.eigenvector_centrality_numpy(self.graph)
            elif centrality_type == "pagerank":
                return nx.pagerank(self.graph)
            else:
                logger.warning(f"Unsupported centrality type: {centrality_type}")
                return {}

        def find_communities(self) -> List[List[str]]:
            """Find communities in the graph"""
            # Convert to undirected graph for community detection
            undirected = self.graph.to_undirected()

            try:
                # Use NetworkX's community detection
                from networkx.algorithms import community
                communities = community.greedy_modularity_communities(undirected)
                return [list(c) for c in communities]
            except Exception as e:
                logger.error(f"Error finding communities: {str(e)}")
                return []

        def find_coordination_patterns(self, timeframe_days: int = 30) -> List[Dict[str, Any]]:
            """Find potential coordination patterns with enhanced detection"""
            # Get timeframe cutoff
            cutoff_date = datetime.now() - timedelta(days=timeframe_days)
            cutoff_str = cutoff_date.isoformat()

            # Find subgraphs where multiple actors connect to the same targets in a short timeframe
            patterns = []

            # Use NetworkX to find nodes with high in-degree (targets of multiple actors)
            in_degree = self.graph.in_degree()
            high_in_degree_nodes = [n for n, d in in_degree if d >= 2]

            for target_id in high_in_degree_nodes:
                target_entity = self.entities.get(target_id)
                if not target_entity:
                    continue

                # Get all incoming edges
                incoming_edges = []
                for u, v, k, data in self.graph.in_edges(target_id, keys=True, data=True):
                    if data.get("created_at", "") >= cutoff_str:
                        incoming_edges.append((u, k, data))

                # Check if we have multiple actors
                source_ids = set(u for u, k, data in incoming_edges)
                actors = [self.entities.get(sid) for sid in source_ids]
                actors = [a for a in actors if a and a.type == "actor"]

                if len(actors) >= 2:
                    # Create pattern record
                    pattern = {
                        "pattern_type": "coordination",
                        "target_entity": target_entity.to_dict(),
                        "actors": [actor.to_dict() for actor in actors],
                        "relationships": [
                            self.relationships.get(k).to_dict()
                            for _, k, _ in incoming_edges if k in self.relationships
                        ],
                        "confidence": "medium",
                        "detection_time": datetime.now().isoformat()
                    }

                    patterns.append(pattern)

            return patterns

        def find_escalation_patterns(self, actor_id: str, lookback_days: int = 90) -> Dict[str, Any]:
            """Detect escalation patterns using graph algorithms"""
            # Get cutoff date
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            cutoff_str = cutoff_date.isoformat()

            # Extract actor's subgraph
            actor_ego = nx.ego_graph(self.graph, actor_id, radius=2)

            # Find all "performs" relationships
            performs_edges = []
            for u, v, k, data in actor_ego.edges(actor_id, keys=True, data=True):
                if data.get("type") == "performs" and data.get("created_at", "") >= cutoff_str:
                    performs_edges.append((v, k, data))

            # For each action, check if it undermines institutions
            undermining_actions = []
            for action_id, rel_id, _ in performs_edges:
                # Find outgoing edges from this action
                if action_id in actor_ego:
                    for a, target_id, k, data in actor_ego.out_edges(action_id, keys=True, data=True):
                        if data.get("type") == "undermines":
                            target_entity = self.entities.get(target_id)
                            if target_entity and target_entity.type == "institution":
                                # Add to undermining actions
                                action_entity = self.entities.get(action_id)
                                if action_entity:
                                    # Get the "performs" relationship timestamp
                                    perf_rel = self.relationships.get(rel_id)
                                    timestamp = perf_rel.created_at if perf_rel else datetime.now().isoformat()

                                    undermining_actions.append({
                                        "action": action_entity.to_dict(),
                                        "target_institution": target_entity.to_dict(),
                                        "timestamp": timestamp,
                                        "severity": data.get("weight", 1.0)
                                    })

            # Sort by timestamp
            undermining_actions.sort(key=lambda x: x["timestamp"])

            # Check if we have escalation (increasing severity over time)
            escalation_detected = False
            escalation_factor = 1.0

            if len(undermining_actions) >= 2:
                # Check if later actions have higher severity on average
                midpoint = len(undermining_actions) // 2
                first_half = undermining_actions[:midpoint]
                second_half = undermining_actions[midpoint:]

                first_half_avg = sum(a["severity"] for a in first_half) / len(first_half) if first_half else 0
                second_half_avg = sum(a["severity"] for a in second_half) / len(second_half) if second_half else 0

                escalation_detected = second_half_avg > first_half_avg
                escalation_factor = second_half_avg / first_half_avg if first_half_avg > 0 else 1.0

            return {
                "actor_id": actor_id,
                "actor": self.entities.get(actor_id).to_dict() if actor_id in self.entities else {"id": actor_id},
                "undermining_actions": undermining_actions,
                "escalation_detected": escalation_detected,
                "escalation_factor": escalation_factor,
                "lookback_days": lookback_days,
                "detection_time": datetime.now().isoformat()
            }

        def identify_narrative_shifts(self, topic_id: str, lookback_days: int = 90) -> Dict[str, Any]:
            """Identify narrative shifts using network analysis"""
            # Get cutoff date
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            cutoff_str = cutoff_date.isoformat()

            # Find all paths from topic to narratives
            narratives = []

            # Get topic's subgraph
            topic_ego = nx.ego_graph(self.graph, topic_id, radius=2)

            # Find all narratives in this subgraph
            for node in topic_ego.nodes():
                entity = self.entities.get(node)
                if entity and entity.type == "narrative":
                    # Find connections to the topic
                    paths = nx.all_simple_paths(topic_ego, topic_id, node, cutoff=2)

                    for path in paths:
                        if len(path) > 1:
                            # Get the relationship
                            u, v = path[0], path[1]
                            edge_data = topic_ego.get_edge_data(u, v)

                            if edge_data:
                                # Use the first edge if multiple exist
                                first_edge_key = list(edge_data.keys())[0]
                                rel_data = edge_data[first_edge_key]

                                if rel_data.get("created_at", "") >= cutoff_str:
                                    # Add to narratives
                                    narratives.append({
                                        "narrative": entity.to_dict(),
                                        "relationship": rel_data,
                                        "timestamp": rel_data.get("created_at", "")
                                    })

            # Sort narratives by timestamp
            narratives.sort(key=lambda x: x["timestamp"])

            # Group narratives by time periods (e.g., weeks)
            time_periods = {}
            for narrative in narratives:
                timestamp = datetime.fromisoformat(narrative["timestamp"])
                week = timestamp.isocalendar()[1]  # Get ISO week number
                year = timestamp.year
                period_key = f"{year}-W{week}"

                if period_key not in time_periods:
                    time_periods[period_key] = []

                time_periods[period_key].append(narrative)

            # Identify shifts between periods
            narrative_shifts = []
            sorted_periods = sorted(time_periods.keys())

            for i in range(1, len(sorted_periods)):
                prev_period = sorted_periods[i-1]
                curr_period = sorted_periods[i]

                prev_narratives = time_periods[prev_period]
                curr_narratives = time_periods[curr_period]

                # Simple detection: new narratives appearing
                prev_narrative_ids = {n["narrative"]["id"] for n in prev_narratives}
                curr_narrative_ids = {n["narrative"]["id"] for n in curr_narratives}

                new_narratives = curr_narrative_ids - prev_narrative_ids

                if new_narratives:
                    narrative_shifts.append({
                        "from_period": prev_period,
                        "to_period": curr_period,
                        "new_narratives": [
                            n["narrative"] for n in curr_narratives
                            if n["narrative"]["id"] in new_narratives
                        ],
                        "shift_type": "new_narrative_introduction"
                    })

            return {
                "topic_id": topic_id,
                "topic": self.entities.get(topic_id).to_dict() if topic_id in self.entities else {"id": topic_id},
                "time_periods": {k: len(v) for k, v in time_periods.items()},
                "narrative_shifts": narrative_shifts,
                "total_narratives": len(narratives),
                "lookback_days": lookback_days,
                "detection_time": datetime.now().isoformat()
            }

        def to_dict(self) -> Dict[str, Any]:
            """Convert to dictionary for serialization"""
            return {
                "entities": {eid: entity.to_dict() for eid, entity in self.entities.items()},
                "relationships": {rid: rel.to_dict() for rid, rel in self.relationships.items()},
                "metadata": {
                    "entity_count": len(self.entities),
                    "relationship_count": len(self.relationships),
                    "entity_types": {t: len(ids) for t, ids in self.entity_types.items()},
                    "relation_types": {t: len(ids) for t, ids in self.relation_types.items()},
                    "timestamp": datetime.now().isoformat()
                }
            }

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'NetworkXGraph':
            """Create from dictionary"""
            graph = cls()

            # Load entities
            for entity_id, entity_data in data.get("entities", {}).items():
                entity = Entity.from_dict(entity_data)
                graph.add_entity(entity)

            # Load relationships
            for rel_id, rel_data in data.get("relationships", {}).items():
                rel = Relationship.from_dict(rel_data)
                graph.add_relationship(rel)

            return graph


class KnowledgeGraph:
    """Knowledge graph for tracking authoritarian patterns and actors"""

    def __init__(self, use_networkx: bool = True):
        """Initialize the knowledge graph"""
        self.use_networkx = use_networkx and NETWORKX_AVAILABLE

        if self.use_networkx:
            try:
                self.graph = NetworkXGraph()
                logger.info("Using NetworkX-based knowledge graph")
            except ImportError:
                self.use_networkx = False
                self.graph = SimpleGraph()
                logger.info("NetworkX not available, using simple knowledge graph")
        else:
            self.graph = SimpleGraph()
            logger.info("Using simple knowledge graph")

        # Set entity type constants
        for entity_key, entity_value in ENTITY_TYPES.items():
            setattr(self, entity_key, entity_value)

        # Set relationship type constants
        for rel_key, rel_value in RELATIONSHIP_TYPES.items():
            setattr(self, rel_key, rel_value)

        # Cache recently accessed entities for performance
        self._entity_cache = {}
        self._entity_cache_size = 100

        # Track creation date
        self.creation_date = datetime.now().isoformat()
        self.last_update = self.creation_date

    def add_entity(self, entity_type: str, name: str, attributes: Dict[str, Any] = None) -> str:
        """Add an entity to the graph"""
        entity = Entity(entity_type=entity_type, name=name, attributes=attributes or {})
        return self.graph.add_entity(entity)

    def add_relationship(self, source_id: str, target_id: str, relation_type: str,
                          weight: float = 1.0, attributes: Dict[str, Any] = None) -> str:
        """Add a relationship between entities"""
        relationship = Relationship(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight,
            attributes=attributes or {}
        )

        # Update last update time
        self.last_update = datetime.now().isoformat()

        return self.graph.add_relationship(relationship)

    def add_relationship_with_evidence(self, source_id: str, target_id: str, relation_type: str,
                                       evidence_text: str, source_content_id: str,
                                       weight: float = 1.0, attributes: Dict[str, Any] = None) -> str:
        """Add a relationship with supporting evidence"""
        relationship = Relationship(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight,
            attributes=attributes or {}
        )

        # Add evidence
        relationship.add_evidence(evidence_text, source_content_id)

        # Update entities with evidence source
        source_entity = self.get_entity(source_id)
        if source_entity:
            source_entity.add_evidence(source_content_id)

        target_entity = self.get_entity(target_id)
        if target_entity:
            target_entity.add_evidence(source_content_id)

        # Update last update time
        self.last_update = datetime.now().isoformat()

        return self.graph.add_relationship(relationship)

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID with caching"""
        if entity_id in self._entity_cache:
            return self._entity_cache[entity_id]

        entity = self.graph.get_entity(entity_id)

        if entity:
            # Add to cache, removing oldest if needed
            if len(self._entity_cache) >= self._entity_cache_size:
                self._entity_cache.pop(next(iter(self._entity_cache)))
            self._entity_cache[entity_id] = entity

        return entity

    def find_entity_by_name(self, name: str, entity_type: Optional[str] = None) -> Optional[Entity]:
        """Find an entity by name, optionally filtering by type"""
        return self.graph.find_entity_by_name(name, entity_type)

    def find_or_create_entity(self, name: str, entity_type: str,
                               attributes: Dict[str, Any] = None) -> str:
        """Find an entity by name and type or create if not exists"""
        entity = self.find_entity_by_name(name, entity_type)
        if entity:
            # Update attributes if provided
            if attributes:
                entity.update(attributes)
            return entity.id
        else:
            # Create new entity
            return self.add_entity(entity_type, name, attributes)

    def find_or_create_actor(self, name: str, attributes: Dict[str, Any] = None) -> str:
        """Find an actor by name or create if not exists"""
        return self.find_or_create_entity(name, self.ACTOR, attributes)

    def find_or_create_institution(self, name: str, attributes: Dict[str, Any] = None) -> str:
        """Find an institution by name or create if not exists"""
        return self.find_or_create_entity(name, self.INSTITUTION, attributes)

    def find_or_create_event(self, name: str, attributes: Dict[str, Any] = None) -> str:
        """Find an event by name or create if not exists"""
        return self.find_or_create_entity(name, self.EVENT, attributes)

    def find_or_create_action(self, name: str, attributes: Dict[str, Any] = None) -> str:
        """Find an action by name or create if not exists"""
        return self.find_or_create_entity(name, self.ACTION, attributes)

    def find_or_create_topic(self, name: str, attributes: Dict[str, Any] = None) -> str:
        """Find a topic by name or create if not exists"""
        return self.find_or_create_entity(name, self.TOPIC, attributes)

    def find_or_create_narrative(self, name: str, attributes: Dict[str, Any] = None) -> str:
        """Find a narrative by name or create if not exists"""
        return self.find_or_create_entity(name, self.NARRATIVE, attributes)

    def add_actor_relationship(self, actor_name: str, target_name: str, target_type: str,
                               relation_type: str, evidence_text: str = None, source_id: str = None,
                               weight: float = 1.0, attributes: Dict[str, Any] = None) -> Tuple[str, str, str]:
        """Add a relationship between an actor and another entity with optional evidence"""
        # Find or create actor
        actor_id = self.find_or_create_actor(actor_name)

        # Find or create target entity based on type
        target_id = self.find_or_create_entity(target_name, target_type)

        # Add relationship with or without evidence
        if evidence_text and source_id:
            relation_id = self.add_relationship_with_evidence(
                actor_id, target_id, relation_type, evidence_text, source_id, weight, attributes
            )
        else:
            relation_id = self.add_relationship(
                actor_id, target_id, relation_type, weight, attributes
            )

        return actor_id, target_id, relation_id

    def get_actor_network(self, actor_name: str, max_depth: int = 2) -> Dict[str, Any]:
        """Get a network of entities connected to an actor"""
        # Find actor by name
        actor = self.find_entity_by_name(actor_name, self.ACTOR)

        if not actor:
            return {"entities": {}, "relationships": {}}

        return self.graph.get_entity_network(actor.id, max_depth)

    def get_topic_network(self, topic_name: str, max_depth: int = 2) -> Dict[str, Any]:
        """Get a network of entities related to a topic"""
        # Find topic by name
        topic = self.find_entity_by_name(topic_name, self.TOPIC)

        if not topic:
            return {"entities": {}, "relationships": {}}

        return self.graph.get_entity_network(topic.id, max_depth)

    def get_related_narratives(self, topic_name: str) -> List[Dict[str, Any]]:
        """Get narratives related to a specific topic"""
        # Find topic by name
        topic = self.find_entity_by_name(topic_name, self.TOPIC)

        if not topic:
            return []

        # Find narratives that relate to this topic
        narratives = []

        # Check both incoming and outgoing relationships
        for rel in self.graph.get_relationships_to(topic.id):
            source_entity = self.get_entity(rel.source_id)
            if source_entity and source_entity.type == self.NARRATIVE:
                narratives.append({
                    "narrative": source_entity.to_dict(),
                    "relationship": rel.to_dict()
                })

        for rel in self.graph.get_relationships_from(topic.id):
            target_entity = self.get_entity(rel.target_id)
            if target_entity and target_entity.type == self.NARRATIVE:
                narratives.append({
                    "narrative": target_entity.to_dict(),
                    "relationship": rel.to_dict()
                })

        return narratives

    def get_authoritarian_trends(self, days: int = 90) -> Dict[str, Any]:
        """Analyze authoritarian trends in the knowledge graph"""
        # Calculate cutoff date
        cutoff_date = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff_date.isoformat()

        # Get all undermining relationships
        undermining_rels = self.graph.get_relationships_by_type("undermines")

        # Filter by date
        recent_undermining = [rel for rel in undermining_rels if rel.created_at >= cutoff_str]

        # Group by target institution
        institution_impacts = {}

        for rel in recent_undermining:
            target_entity = self.get_entity(rel.target_id)
            if target_entity and target_entity.type == self.INSTITUTION:
                if target_entity.id not in institution_impacts:
                    institution_impacts[target_entity.id] = {
                        "institution": target_entity.to_dict(),
                        "impact_count": 0,
                        "total_weight": 0,
                        "actions": []
                    }

                # Get the source action and actor if available
                source_entity = self.get_entity(rel.source_id)
                if source_entity:
                    # If source is an action, try to find who performed it
                    actor = None
                    if source_entity.type == self.ACTION:
                        # Check who performed this action
                        for performer_rel in self.graph.get_relationships_to(source_entity.id):
                            if performer_rel.type == "performs":
                                performer = self.get_entity(performer_rel.source_id)
                                if performer and performer.type == self.ACTOR:
                                    actor = performer
                                    break

                    # Add to institution impacts
                    institution_impacts[target_entity.id]["impact_count"] += 1
                    institution_impacts[target_entity.id]["total_weight"] += rel.weight

                    action_data = {
                        "entity": source_entity.to_dict(),
                        "relationship": rel.to_dict()
                    }

                    if actor:
                        action_data["actor"] = actor.to_dict()

                    institution_impacts[target_entity.id]["actions"].append(action_data)

        # Calculate overall trend metrics
        total_undermining_actions = len(recent_undermining)
        total_affected_institutions = len(institution_impacts)

        # Sort institutions by impact
        sorted_institutions = sorted(
            institution_impacts.values(),
            key=lambda x: x["total_weight"],
            reverse=True
        )

        # Calculate trend score (0-10)
        trend_score = min(10, total_undermining_actions / 2)

        return {
            "trend_score": trend_score,
            "affected_institutions": sorted_institutions,
            "total_undermining_actions": total_undermining_actions,
            "total_affected_institutions": total_affected_institutions,
            "lookback_days": days,
            "analysis_timestamp": datetime.now().isoformat()
        }

    def analyze_actor_patterns(self, actor_name: str, days: int = 90) -> Dict[str, Any]:
        """Analyze patterns of activity for a specific actor"""
        # Find actor
        actor = self.find_entity_by_name(actor_name, self.ACTOR)

        if not actor:
            return {"error": f"Actor '{actor_name}' not found"}

        # Use pattern detection algorithms
        return self.graph.find_escalation_patterns(actor.id, days)

    def get_influential_actors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most influential actors based on centrality or relationship count"""
        if self.use_networkx:
            # Use NetworkX centrality measures
            try:
                centrality = self.graph.calculate_centrality("pagerank")

                actor_centrality = []
                for actor in self.graph.get_entities_by_type(self.ACTOR):
                    actor_data = actor.to_dict()
                    actor_data["influence_score"] = centrality.get(actor.id, 0)
                    actor_centrality.append(actor_data)

                actor_centrality.sort(key=lambda x: x["influence_score"], reverse=True)
                return actor_centrality[:limit]
            except Exception as e:
                logger.error(f"Error calculating centrality: {str(e)}")
                return []
        else:
            # Fallback to relationship count for influence
            actors = self.graph.get_entities_by_type(self.ACTOR)

            actor_influence = []
            for actor in actors:
                outgoing = len(self.graph.get_relationships_from(actor.id))
                incoming = len(self.graph.get_relationships_to(actor.id))
                influence = outgoing + incoming  # Simple measure of influence

                actor_data = actor.to_dict()
                actor_data["influence_score"] = influence
                actor_influence.append(actor_data)

            actor_influence.sort(key=lambda x: x["influence_score"], reverse=True)
            return actor_influence[:limit]

    def detect_coordination_patterns(self, days: int = 30) -> List[Dict[str, Any]]:
        """Detect coordination patterns between actors"""
        return self.graph.find_coordination_patterns(days)

    def analyze_democratic_erosion(self, days: int = 90) -> Dict[str, Any]:
        """Comprehensive analysis of democratic erosion patterns with improved detection."""
        # Get authoritarian trends
        trends = self.get_authoritarian_trends(days)

        # Get influential actors
        influential_actors = self.get_influential_actors(5)

        # Get coordination patterns
        coordination_patterns = self.detect_coordination_patterns(days)

        # Get recent indicators of specific types
        cutoff_date = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff_date.isoformat()

        # Get all indicators
        all_indicators = self.graph.get_entities_by_type("indicator")

        # Filter for recent ones
        recent_indicators = [ind for ind in all_indicators if ind.created_at >= cutoff_str]

        # Group by indicator type
        indicator_types = {}
        for indicator in recent_indicators:
            ind_type = indicator.attributes.get("type", "other")

            if ind_type not in indicator_types:
                indicator_types[ind_type] = {
                    "count": 0,
                    "total_severity": 0,
                    "indicators": []
                }

            severity = indicator.attributes.get("severity", 5)
            indicator_types[ind_type]["count"] += 1
            indicator_types[ind_type]["total_severity"] += severity
            indicator_types[ind_type]["indicators"].append(indicator.to_dict())

        # Analyze institution undermining by type
        institution_types = {}
        for inst in trends.get("affected_institutions", []):
            institution = inst.get("institution", {})
            inst_type = institution.get("attributes", {}).get("institution_type", "other")

            if inst_type not in institution_types:
                institution_types[inst_type] = {
                    "count": 0,
                    "total_weight": 0,
                    "institutions": []
                }

            institution_types[inst_type]["count"] += 1
            institution_types[inst_type]["total_weight"] += inst.get("total_weight", 0)
            institution_types[inst_type]["institutions"].append(institution)

        # Calculate risk scores for different aspects of democratic erosion

        # 1. Institution Risk (based on undermining indicators)
        institution_undermining_count = indicator_types.get("institutional_undermining", {}).get("count", 0)
        institution_undermining_severity = indicator_types.get("institutional_undermining", {}).get("total_severity",
                                                                                                    0) / max(1,
                                                                                                             institution_undermining_count)
        affected_institution_count = trends.get("total_affected_institutions", 0)

        institution_risk = min(10, (
                (institution_undermining_count * 0.5) +
                (institution_undermining_severity * 0.3) +
                (affected_institution_count * 0.7)
        ))

        # 2. Norm Violation Risk
        norm_violation_count = indicator_types.get("democratic_norm_violations", {}).get("count", 0)
        norm_violation_count += indicator_types.get("norm_violations", {}).get("count", 0)
        norm_violation_severity = (
                                          indicator_types.get("democratic_norm_violations", {}).get("total_severity",
                                                                                                    0) +
                                          indicator_types.get("norm_violations", {}).get("total_severity", 0)
                                  ) / max(1, norm_violation_count)

        norm_violation_risk = min(10, (norm_violation_count * 1.0) + (norm_violation_severity * 0.5))

        # 3. Media Delegitimization Risk
        media_attack_count = indicator_types.get("media_delegitimization", {}).get("count", 0)
        media_attack_count += indicator_types.get("attacks_on_media", {}).get("count", 0)
        media_attack_severity = (
                                        indicator_types.get("media_delegitimization", {}).get("total_severity", 0) +
                                        indicator_types.get("attacks_on_media", {}).get("total_severity", 0)
                                ) / max(1, media_attack_count)

        media_risk = min(10, (media_attack_count * 1.5) + (media_attack_severity * 0.7))

        # 4. Power Concentration Risk
        power_concentration_count = indicator_types.get("power_concentration", {}).get("count", 0)
        accountability_evasion_count = indicator_types.get("accountability_evasion", {}).get("count", 0)
        executive_overreach_count = indicator_types.get("executive_overreach", {}).get("count", 0)

        power_concentration_severity = (
                                               indicator_types.get("power_concentration", {}).get("total_severity", 0) +
                                               indicator_types.get("accountability_evasion", {}).get("total_severity",
                                                                                                     0) +
                                               indicator_types.get("executive_overreach", {}).get("total_severity", 0)
                                       ) / max(1,
                                               power_concentration_count + accountability_evasion_count + executive_overreach_count)

        power_risk = min(10, (
                (power_concentration_count * 1.0) +
                (accountability_evasion_count * 0.8) +
                (executive_overreach_count * 1.2) +
                (power_concentration_severity * 0.6)
        ))

        # 5. Actor Coordination Risk (based on coordination patterns)
        actor_coordination_risk = min(10, len(coordination_patterns) * 2.0)

        # 6. Opposition Targeting Risk
        opposition_targeting_count = indicator_types.get("opposition_targeting", {}).get("count", 0)
        opposition_targeting_severity = indicator_types.get("opposition_targeting", {}).get("total_severity", 0) / max(
            1, opposition_targeting_count)

        opposition_risk = min(10, (opposition_targeting_count * 1.5) + (opposition_targeting_severity * 0.8))

        # 7. Threat Exaggeration Risk
        threat_exaggeration_count = indicator_types.get("threat_exaggeration", {}).get("count", 0)
        threat_exaggeration_severity = indicator_types.get("threat_exaggeration", {}).get("total_severity", 0) / max(1,
                                                                                                                     threat_exaggeration_count)

        threat_risk = min(10, (threat_exaggeration_count * 0.8) + (threat_exaggeration_severity * 0.4))

        # 8. Rule of Law Undermining Risk
        rule_of_law_count = indicator_types.get("rule_of_law_undermining", {}).get("count", 0)
        rule_of_law_severity = indicator_types.get("rule_of_law_undermining", {}).get("total_severity", 0) / max(1,
                                                                                                                 rule_of_law_count)

        rule_of_law_risk = min(10, (rule_of_law_count * 1.2) + (rule_of_law_severity * 0.6))

        # Calculate overall weighted democratic erosion score
        erosion_score = (
                (institution_risk * 0.25) +
                (norm_violation_risk * 0.15) +
                (media_risk * 0.15) +
                (power_risk * 0.2) +
                (actor_coordination_risk * 0.05) +
                (opposition_risk * 0.1) +
                (threat_risk * 0.05) +
                (rule_of_law_risk * 0.05)
        )

        # Round to 1 decimal place
        erosion_score = round(erosion_score * 10) / 10

        # Determine risk level
        risk_level = "Low"
        if erosion_score >= 7:
            risk_level = "Severe"
        elif erosion_score >= 5:
            risk_level = "High"
        elif erosion_score >= 3:
            risk_level = "Moderate"

        # Extract key risks by ranking factors
        risk_factors = [
            {"factor": "Institutional Undermining", "score": institution_risk},
            {"factor": "Norm Violations", "score": norm_violation_risk},
            {"factor": "Media Delegitimization", "score": media_risk},
            {"factor": "Power Concentration", "score": power_risk},
            {"factor": "Actor Coordination", "score": actor_coordination_risk},
            {"factor": "Opposition Targeting", "score": opposition_risk},
            {"factor": "Threat Exaggeration", "score": threat_risk},
            {"factor": "Rule of Law Undermining", "score": rule_of_law_risk}
        ]

        # Sort by score (highest first)
        risk_factors.sort(key=lambda x: x["score"], reverse=True)

        # Extract top risks
        top_risks = [f"{r['factor']} ({r['score']:.1f}/10)" for r in risk_factors[:3] if r["score"] >= 2]

        # Extract key indicators by severity
        all_democratic_concerns = []
        for ind_type, data in indicator_types.items():
            if data["count"] > 0:
                avg_severity = data["total_severity"] / data["count"]

                # Only include significant concerns
                if avg_severity >= 4:
                    all_democratic_concerns.append({
                        "type": ind_type,
                        "count": data["count"],
                        "avg_severity": avg_severity,
                        "example_indicators": data["indicators"][:2]  # Just include a couple examples
                    })

        # Sort by severity
        all_democratic_concerns.sort(key=lambda x: x["avg_severity"], reverse=True)

        return {
            "erosion_score": erosion_score,
            "risk_level": risk_level,
            "institution_risk": institution_risk,
            "norm_violation_risk": norm_violation_risk,
            "media_risk": media_risk,
            "power_risk": power_risk,
            "actor_coordination_risk": actor_coordination_risk,
            "opposition_risk": opposition_risk,
            "top_risks": top_risks,
            "democratic_concerns": all_democratic_concerns,
            "affected_institution_types": institution_types,
            "influential_actors": influential_actors,
            "coordination_patterns": coordination_patterns,
            "authoritarian_trends": trends,
            "lookback_days": days,
            "analysis_timestamp": datetime.now().isoformat()
        }

    def process_article_analysis(self, article_data: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an article analysis to extract entities and relationships for the knowledge graph.

        Args:
            article_data: The source article data
            analysis: The analysis data containing authoritarian indicators

        Returns:
            Dict with processing results
        """
        results = {
            "entities_created": 0,
            "relationships_created": 0,
            "entity_ids": [],
            "relationship_ids": []
        }

        # Extract source info for evidence
        source_id = article_data.get("id", f"article_{uuid.uuid4().hex[:8]}")

        # Create a topic entity for the article
        title = article_data.get("title", "Untitled Article")
        topic_id = self.find_or_create_topic(
            title,
            attributes={
                "source": article_data.get("source", "Unknown"),
                "url": article_data.get("url", ""),
                "published": article_data.get("published", ""),
                "processed_date": datetime.now().isoformat()
            }
        )
        results["entity_ids"].append(topic_id)
        results["entities_created"] += 1

        # Process authoritarian indicators if available
        structured_data = analysis.get("structured_elements", {})

        # Extract actor entities from the analysis
        actors = []

        # Look for actors in the article title and analysis
        import re
        actor_pattern = r'\b(?:President|Senator|Congressman|Congresswoman|Rep\.|Sen\.|Governor|Secretary|Attorney General|Speaker|Majority Leader|Minister|Judge|Justice)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b'

        # Extract from title
        for match in re.finditer(actor_pattern, title):
            actor_name = match.group(1)
            actor_id = self.find_or_create_actor(actor_name)
            if actor_id not in actors:
                actors.append(actor_id)
                results["entity_ids"].append(actor_id)
                results["entities_created"] += 1

        # Extract from analysis text
        analysis_text = analysis.get("analysis", "")
        for match in re.finditer(actor_pattern, analysis_text):
            actor_name = match.group(1)
            actor_id = self.find_or_create_actor(actor_name)
            if actor_id not in actors:
                actors.append(actor_id)
                results["entity_ids"].append(actor_id)
                results["entities_created"] += 1

        # Try to identify specific authoritarian indicators and create the appropriate entities/relationships
        for indicator_key, indicator_data in structured_data.items():
            if isinstance(indicator_data, dict) and indicator_data.get("present", False):
                # Get examples for evidence
                examples = indicator_data.get("examples", [])
                evidence_text = examples[0] if examples else f"Indicator '{indicator_key}' detected in article"

                # Map indicator to relationship type
                relationship_type = "demonstrates"  # Default

                if indicator_key == "institutional_undermining":
                    relationship_type = "undermines"
                    # Create an institution entity if we can identify one
                    institutions = extract_institutions(analysis_text)

                    for institution_name in institutions:
                        # Create institution entity
                        institution_id = self.find_or_create_institution(institution_name)
                        results["entity_ids"].append(institution_id)
                        results["entities_created"] += 1

                        # Create relationship for each actor
                        for actor_id in actors:
                            rel_id = self.add_relationship_with_evidence(
                                actor_id, institution_id, "undermines",
                                evidence_text, source_id,
                                weight=1.5
                            )
                            results["relationship_ids"].append(rel_id)
                            results["relationships_created"] += 1

                elif indicator_key == "democratic_norm_violations":
                    # Create a norm indicator entity
                    norm_name = f"Democratic norm: {evidence_text[:30]}..."
                    norm_id = self.find_or_create_entity("indicator", norm_name, {
                        "type": "democratic_norm",
                        "description": evidence_text
                    })
                    results["entity_ids"].append(norm_id)
                    results["entities_created"] += 1

                    # Connect actors to this norm violation
                    for actor_id in actors:
                        rel_id = self.add_relationship_with_evidence(
                            actor_id, norm_id, "violates",
                            evidence_text, source_id,
                            weight=1.5
                        )
                        results["relationship_ids"].append(rel_id)
                        results["relationships_created"] += 1

                elif indicator_key == "media_delegitimization":
                    # Create media entity
                    media_id = self.find_or_create_entity("institution", "Independent Media", {
                        "institution_type": "media",
                        "description": "Independent news media and journalists"
                    })
                    results["entity_ids"].append(media_id)
                    results["entities_created"] += 1

                    # Connect actors to media delegitimization
                    for actor_id in actors:
                        rel_id = self.add_relationship_with_evidence(
                            actor_id, media_id, "undermines",
                            evidence_text, source_id,
                            weight=1.7
                        )
                        results["relationship_ids"].append(rel_id)
                        results["relationships_created"] += 1

                # Add more indicator-specific processing as needed...

                # Create a generic indicator entity for any other indicators
                indicator_id = self.find_or_create_entity("indicator", f"Indicator: {indicator_key}", {
                    "type": indicator_key,
                    "description": evidence_text
                })
                results["entity_ids"].append(indicator_id)
                results["entities_created"] += 1

                # Connect topic to indicator
                topic_indicator_rel = self.add_relationship_with_evidence(
                    topic_id, indicator_id, "demonstrates",
                    evidence_text, source_id,
                    weight=1.0
                )
                results["relationship_ids"].append(topic_indicator_rel)
                results["relationships_created"] += 1

        # Connect actors to the topic
        for actor_id in actors:
            actor_topic_rel = self.add_relationship(
                actor_id, topic_id, "relates_to",
                weight=1.0,
                attributes={"source": article_data.get("source", "Unknown")}
            )
            results["relationship_ids"].append(actor_topic_rel)
            results["relationships_created"] += 1

        return results

    def save(self, path: str) -> bool:
        """Save the knowledge graph to disk"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)

            data = self.graph.to_dict()
            data["metadata"] = {
                "version": "1.0",
                "use_networkx": self.use_networkx,
                "creation_date": self.creation_date,
                "last_update": self.last_update,
                "timestamp": datetime.now().isoformat()
            }

            with open(path, 'wb') as f:
                pickle.dump(data, f)

            logger.info(f"Saved knowledge graph to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving knowledge graph: {str(e)}")
            return False

    def load(self, path: str) -> bool:
        """Load the knowledge graph from disk"""
        if not os.path.exists(path):
            logger.warning(f"Knowledge graph file not found: {path}")
            return False

        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)

            # Determine graph type
            use_networkx = data.get("metadata", {}).get("use_networkx", False)

            # Clear cache
            self._entity_cache.clear()

            # Create appropriate graph
            if use_networkx and NETWORKX_AVAILABLE:
                self.use_networkx = True
                self.graph = NetworkXGraph.from_dict(data)
            else:
                self.use_networkx = False
                self.graph = SimpleGraph.from_dict(data)

            # Load metadata
            metadata = data.get("metadata", {})
            self.creation_date = metadata.get("creation_date", datetime.now().isoformat())
            self.last_update = metadata.get("last_update", self.creation_date)

            logger.info(f"Loaded knowledge graph from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading knowledge graph: {str(e)}")
            return False


# Helper functions for entity extraction
def extract_institutions(text: str) -> List[str]:
    """Extract institution names from text"""
    institutions = []

    # Common governmental institutions to look for
    common_institutions = [
        "Congress", "Senate", "House of Representatives", "Supreme Court",
        "Justice Department", "Department of Justice", "DOJ", "FBI", "CIA",
        "State Department", "Department of State", "Pentagon", "White House",
        "Executive Branch", "Judiciary", "Federal Reserve", "EPA",
        "Department of Homeland Security", "DHS", "IRS", "Treasury Department"
    ]

    # Look for common institutions in the text
    for institution in common_institutions:
        if institution in text:
            institutions.append(institution)

    # More advanced extraction could be implemented here

    return institutions
