"""
Night_watcher Knowledge Graph
Entity-relationship mapping for authoritarian patterns and actors.
"""

import os
import json
import pickle
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple, Set

import numpy as np

# Optional imports for enhanced functionality
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

logger = logging.getLogger(__name__)


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

    def update(self, attributes: Dict[str, Any]) -> None:
        """Update entity attributes"""
        self.attributes.update(attributes)
        self.updated_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "attributes": self.attributes,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Entity':
        """Create from dictionary"""
        entity = cls(
            entity_id=data["id"],
            entity_type=data["type"],
            name=data["name"],
            attributes=data["attributes"]
        )
        entity.created_at = data["created_at"]
        entity.updated_at = data["updated_at"]
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

    def update(self, weight: float = None, attributes: Dict[str, Any] = None) -> None:
        """Update relationship weight and/or attributes"""
        if weight is not None:
            self.weight = weight

        if attributes:
            self.attributes.update(attributes)

        self.updated_at = datetime.now().isoformat()

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
            "updated_at": self.updated_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Relationship':
        """Create from dictionary"""
        rel = cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=data["type"],
            relation_id=data["id"],
            weight=data["weight"],
            attributes=data["attributes"]
        )
        rel.created_at = data["created_at"]
        rel.updated_at = data["updated_at"]
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


class NetworkXGraph:
    """Knowledge graph implementation using NetworkX for advanced analysis"""

    def __init__(self):
        """Initialize an empty graph"""
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX not available. Install with: pip install networkx")

        self.graph = nx.MultiDiGraph()
        self.entity_types = {}  # type -> [entity_ids]
        self.relation_types = {}  # type -> [relation_ids]

    def add_entity(self, entity: Entity) -> str:
        """Add an entity to the graph"""
        entity_id = entity.id

        # Add node to NetworkX graph
        self.graph.add_node(entity_id, **entity.to_dict())

        # Index by type
        if entity.type not in self.entity_types:
            self.entity_types[entity.type] = []
        self.entity_types[entity.type].append(entity_id)

        return entity_id

    def add_relationship(self, relationship: Relationship) -> str:
        """Add a relationship to the graph"""
        relation_id = relationship.id
        source_id = relationship.source_id
        target_id = relationship.target_id

        # Add edge to NetworkX graph
        self.graph.add_edge(source_id, target_id, key=relation_id, **relationship.to_dict())

        # Index by type
        if relationship.type not in self.relation_types:
            self.relation_types[relationship.type] = []
        self.relation_types[relationship.type].append(relation_id)

        return relation_id

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID"""
        if entity_id not in self.graph.nodes:
            return None

        node_data = self.graph.nodes[entity_id]
        return Entity.from_dict(node_data)

    def get_relationship(self, relation_id: str) -> Optional[Relationship]:
        """Get a relationship by ID"""
        for u, v, k, data in self.graph.edges(keys=True, data=True):
            if k == relation_id:
                return Relationship.from_dict(data)
        return None

    def get_relationships_from(self, entity_id: str) -> List[Relationship]:
        """Get all relationships originating from an entity"""
        if entity_id not in self.graph.nodes:
            return []

        relationships = []
        for _, v, k, data in self.graph.out_edges(entity_id, keys=True, data=True):
            relationships.append(Relationship.from_dict(data))

        return relationships

    def get_relationships_to(self, entity_id: str) -> List[Relationship]:
        """Get all relationships targeting an entity"""
        if entity_id not in self.graph.nodes:
            return []

        relationships = []
        for u, _, k, data in self.graph.in_edges(entity_id, keys=True, data=True):
            relationships.append(Relationship.from_dict(data))

        return relationships

    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Get all entities of a specific type"""
        entity_ids = self.entity_types.get(entity_type, [])
        return [self.get_entity(entity_id) for entity_id in entity_ids]

    def get_relationships_by_type(self, relation_type: str) -> List[Relationship]:
        """Get all relationships of a specific type"""
        relation_ids = self.relation_types.get(relation_type, [])
        return [self.get_relationship(relation_id) for relation_id in relation_ids]

    def find_path(self, source_id: str, target_id: str, max_depth: int = 3) -> List[List[Tuple[str, str]]]:
        """Find paths between two entities up to a maximum depth"""
        if source_id not in self.graph.nodes or target_id not in self.graph.nodes:
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
        if entity_id not in self.graph.nodes:
            return {"entities": {}, "relationships": {}}

        # Use NetworkX's ego graph function
        ego_graph = nx.ego_graph(self.graph, entity_id, radius=max_depth, undirected=True)

        entities = {}
        relationships = {}

        # Extract entities
        for node in ego_graph.nodes:
            entities[node] = ego_graph.nodes[node]

        # Extract relationships
        for u, v, k, data in ego_graph.edges(keys=True, data=True):
            relationships[k] = data

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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        entities = {}
        relationships = {}

        # Extract entities
        for node in self.graph.nodes:
            entities[node] = self.graph.nodes[node]

        # Extract relationships
        for u, v, k, data in self.graph.edges(keys=True, data=True):
            relationships[k] = data

        return {
            "entities": entities,
            "relationships": relationships,
            "metadata": {
                "entity_count": len(entities),
                "relationship_count": len(relationships),
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

        # Entity types
        self.ACTOR = "actor"
        self.ORGANIZATION = "organization"
        self.EVENT = "event"
        self.TOPIC = "topic"
        self.NARRATIVE = "narrative"
        self.TACTIC = "tactic"
        self.INDICATOR = "indicator"

        # Relationship types
        self.INFLUENCES = "influences"
        self.PROMOTES = "promotes"
        self.ATTACKS = "attacks"
        self.SUPPORTS = "supports"
        self.PARTICIPATES_IN = "participates_in"
        self.ASSOCIATES_WITH = "associates_with"
        self.TARGETS = "targets"
        self.DEMONSTRATES = "demonstrates"
        self.PART_OF = "part_of"

        # Cache recently accessed entities for performance
        self._entity_cache = {}
        self._entity_cache_size = 100

    def add_actor(self, name: str, attributes: Dict[str, Any] = None) -> str:
        """Add an authoritarian actor to the graph"""
        entity = Entity(entity_type=self.ACTOR, name=name, attributes=attributes or {})
        return self.graph.add_entity(entity)

    def add_organization(self, name: str, attributes: Dict[str, Any] = None) -> str:
        """Add an organization to the graph"""
        entity = Entity(entity_type=self.ORGANIZATION, name=name, attributes=attributes or {})
        return self.graph.add_entity(entity)

    def add_event(self, name: str, attributes: Dict[str, Any] = None) -> str:
        """Add an event to the graph"""
        entity = Entity(entity_type=self.EVENT, name=name, attributes=attributes or {})
        return self.graph.add_entity(entity)

    def add_topic(self, name: str, attributes: Dict[str, Any] = None) -> str:
        """Add a topic to the graph"""
        entity = Entity(entity_type=self.TOPIC, name=name, attributes=attributes or {})
        return self.graph.add_entity(entity)

    def add_narrative(self, name: str, attributes: Dict[str, Any] = None) -> str:
        """Add a narrative to the graph"""
        entity = Entity(entity_type=self.NARRATIVE, name=name, attributes=attributes or {})
        return self.graph.add_entity(entity)

    def add_tactic(self, name: str, attributes: Dict[str, Any] = None) -> str:
        """Add a tactic to the graph"""
        entity = Entity(entity_type=self.TACTIC, name=name, attributes=attributes or {})
        return self.graph.add_entity(entity)

    def add_indicator(self, name: str, attributes: Dict[str, Any] = None) -> str:
        """Add an authoritarian indicator to the graph"""
        entity = Entity(entity_type=self.INDICATOR, name=name, attributes=attributes or {})
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

    def find_or_create_actor(self, name: str, attributes: Dict[str, Any] = None) -> str:
        """Find an actor by name or create if not exists"""
        actors = self.graph.get_entities_by_type(self.ACTOR)

        for actor in actors:
            if actor.name.lower() == name.lower():
                # Update attributes if provided
                if attributes:
                    actor.update(attributes)
                return actor.id

        # Actor not found, create new
        return self.add_actor(name, attributes)

    def add_actor_relationship(self, actor_name: str, target_name: str, target_type: str,
                                relation_type: str, weight: float = 1.0,
                                attributes: Dict[str, Any] = None) -> Tuple[str, str, str]:
        """Add a relationship between an actor and another entity"""
        # Find or create actor
        actor_id = self.find_or_create_actor(actor_name)

        # Find or create target entity based on type
        target_id = None
        if target_type == self.ACTOR:
            target_id = self.find_or_create_actor(target_name)
        elif target_type == self.ORGANIZATION:
            target_id = self.find_or_create_entity(target_name, self.ORGANIZATION)
        elif target_type == self.TOPIC:
            target_id = self.find_or_create_entity(target_name, self.TOPIC)
        elif target_type == self.NARRATIVE:
            target_id = self.find_or_create_entity(target_name, self.NARRATIVE)
        elif target_type == self.TACTIC:
            target_id = self.find_or_create_entity(target_name, self.TACTIC)
        elif target_type == self.INDICATOR:
            target_id = self.find_or_create_entity(target_name, self.INDICATOR)
        else:
            raise ValueError(f"Unsupported target type: {target_type}")

        # Add relationship
        relation_id = self.add_relationship(
            actor_id, target_id, relation_type, weight, attributes
        )

        return actor_id, target_id, relation_id

    def find_or_create_entity(self, name: str, entity_type: str,
                               attributes: Dict[str, Any] = None) -> str:
        """Find an entity by name and type or create if not exists"""
        entities = self.graph.get_entities_by_type(entity_type)

        for entity in entities:
            if entity.name.lower() == name.lower():
                # Update attributes if provided
                if attributes:
                    entity.update(attributes)
                return entity.id

        # Entity not found, create new
        entity = Entity(entity_type=entity_type, name=name, attributes=attributes or {})
        return self.graph.add_entity(entity)

    def get_actor_network(self, actor_name: str, max_depth: int = 2) -> Dict[str, Any]:
        """Get a network of entities connected to an actor"""
        actors = self.graph.get_entities_by_type(self.ACTOR)

        actor_id = None
        for actor in actors:
            if actor.name.lower() == actor_name.lower():
                actor_id = actor.id
                break

        if not actor_id:
            return {"entities": {}, "relationships": {}}

        return self.graph.get_entity_network(actor_id, max_depth)

    def get_topic_narrative_network(self, topic_name: str) -> Dict[str, Any]:
        """Get narratives and actors related to a specific topic"""
        topics = self.graph.get_entities_by_type(self.TOPIC)

        topic_id = None
        for topic in topics:
            if topic.name.lower() == topic_name.lower():
                topic_id = topic.id
                break

        if not topic_id:
            return {"narratives": [], "actors": []}

        network = self.graph.get_entity_network(topic_id, max_depth=2)

        narratives = []
        actors = []

        for entity_id, entity_data in network["entities"].items():
            if entity_data["type"] == self.NARRATIVE:
                narratives.append(entity_data)
            elif entity_data["type"] == self.ACTOR:
                actors.append(entity_data)

        return {
            "narratives": narratives,
            "actors": actors
        }

    def get_influential_actors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most influential actors based on centrality"""
        if not self.use_networkx:
            logger.warning("NetworkX required for centrality measures. Falling back to relationship count.")
            actors = self.graph.get_entities_by_type(self.ACTOR)

            actor_influence = []
            for actor in actors:
                outgoing = len(self.graph.get_relationships_from(actor.id))
                incoming = len(self.graph.get_relationships_to(actor.id))
                influence = outgoing + incoming

                actor_data = actor.to_dict()
                actor_data["influence_score"] = influence
                actor_influence.append(actor_data)

            actor_influence.sort(key=lambda x: x["influence_score"], reverse=True)
            return actor_influence[:limit]
        else:
            # Use NetworkX pagerank
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

    def detect_communities(self) -> Dict[str, Any]:
        """Detect communities of actors and topics"""
        if not self.use_networkx:
            logger.warning("NetworkX required for community detection")
            return {"communities": []}

        try:
            communities = self.graph.find_communities()

            result = []
            for i, community in enumerate(communities):
                community_entities = []
                for entity_id in community:
                    entity = self.get_entity(entity_id)
                    if entity:
                        community_entities.append(entity.to_dict())

                # Calculate main entity types in community
                entity_types = {}
                for entity in community_entities:
                    entity_type = entity["type"]
                    entity_types[entity_type] = entity_types.get(entity_type, 0) + 1

                # Calculate community focus
                focus = max(entity_types.items(), key=lambda x: x[1])[0] if entity_types else "mixed"

                result.append({
                    "id": i,
                    "size": len(community),
                    "focus": focus,
                    "entity_types": entity_types,
                    "entities": community_entities
                })

            return {"communities": result}
        except Exception as e:
            logger.error(f"Error detecting communities: {str(e)}")
            return {"communities": []}

    def analyze_authoritarian_trends(self) -> Dict[str, Any]:
        """Analyze trends in authoritarian indicators based on the graph"""
        # Get all indicators
        indicators = self.graph.get_entities_by_type(self.INDICATOR)

        # Get all actors
        actors = self.graph.get_entities_by_type(self.ACTOR)

        # Analyze indicator-actor relationships
        indicator_trends = []
        for indicator in indicators:
            # Get relationships where indicator is the target
            relationships = self.graph.get_relationships_to(indicator.id)

            # Count relationships by actor
            actor_counts = {}
            for rel in relationships:
                source_id = rel.source_id
                source = self.get_entity(source_id)
                if source and source.type == self.ACTOR:
                    actor_counts[source.name] = actor_counts.get(source.name, 0) + rel.weight

            # Create trend data
            trend = {
                "indicator": indicator.name,
                "total_instances": sum(actor_counts.values()),
                "actor_distribution": actor_counts,
                "attributes": indicator.attributes
            }

            indicator_trends.append(trend)

        # Analyze actor patterns
        actor_patterns = []
        for actor in actors:
            # Get outgoing relationships
            relationships = self.graph.get_relationships_from(actor.id)

            # Count relationships by indicator
            indicator_counts = {}
            tactic_counts = {}

            for rel in relationships:
                target_id = rel.target_id
                target = self.get_entity(target_id)
                if target:
                    if target.type == self.INDICATOR:
                        indicator_counts[target.name] = indicator_counts.get(target.name, 0) + rel.weight
                    elif target.type == self.TACTIC:
                        tactic_counts[target.name] = tactic_counts.get(target.name, 0) + rel.weight

            # Calculate primary indicator
            primary_indicator = max(indicator_counts.items(), key=lambda x: x[1])[0] if indicator_counts else None

            # Create actor pattern data
            pattern = {
                "actor": actor.name,
                "indicators": indicator_counts,
                "tactics": tactic_counts,
                "primary_indicator": primary_indicator,
                "authoritarian_score": sum(indicator_counts.values()),
                "attributes": actor.attributes
            }

            actor_patterns.append(pattern)

        # Calculate overall trend strength
        overall_trend = sum(trend["total_instances"] for trend in indicator_trends)

        return {
            "indicator_trends": indicator_trends,
            "actor_patterns": actor_patterns,
            "overall_trend": overall_trend,
            "timestamp": datetime.now().isoformat()
        }

    def identify_influence_networks(self) -> Dict[str, Any]:
        """Identify networks of influence among actors and organizations"""
        if not self.use_networkx:
            logger.warning("NetworkX required for influence network analysis")
            return {"influence_networks": []}

        try:
            # Get all actors and organizations
            actors = self.graph.get_entities_by_type(self.ACTOR)
            orgs = self.graph.get_entities_by_type(self.ORGANIZATION)

            # Calculate centrality
            centrality = self.graph.calculate_centrality("pagerank")

            # Create centrality data for actors and organizations
            entity_centrality = []
            for entity in actors + orgs:
                entity_data = entity.to_dict()
                entity_data["centrality"] = centrality.get(entity.id, 0)
                entity_data["relationships"] = []

                # Get outgoing relationships
                for rel in self.graph.get_relationships_from(entity.id):
                    target = self.get_entity(rel.target_id)
                    if target and (target.type == self.ACTOR or target.type == self.ORGANIZATION):
                        entity_data["relationships"].append({
                            "target": target.name,
                            "target_type": target.type,
                            "relation_type": rel.type,
                            "weight": rel.weight
                        })

                entity_centrality.append(entity_data)

            # Sort by centrality
            entity_centrality.sort(key=lambda x: x["centrality"], reverse=True)

            # Extract top influence networks
            top_entities = entity_centrality[:10]
            influence_networks = []

            for entity_data in top_entities:
                # Get direct network
                network = self.graph.get_entity_network(entity_data["id"], max_depth=1)

                # Extract entities and relationships
                network_entities = []
                for eid, e_data in network["entities"].items():
                    entity = self.get_entity(eid)
                    if entity and (entity.type == self.ACTOR or entity.type == self.ORGANIZATION):
                        network_entities.append(e_data)

                # Add to influence networks
                influence_networks.append({
                    "central_entity": entity_data["name"],
                    "entity_type": entity_data["type"],
                    "centrality": entity_data["centrality"],
                    "network_entities": network_entities,
                    "direct_relationships": entity_data["relationships"]
                })

            return {"influence_networks": influence_networks}
        except Exception as e:
            logger.error(f"Error identifying influence networks: {str(e)}")
            return {"influence_networks": []}

    def save(self, path: str) -> bool:
        """Save the knowledge graph to disk"""
        try:
            base_dir = os.path.dirname(path)
            os.makedirs(base_dir, exist_ok=True)

            data = self.graph.to_dict()
            data["metadata"] = {
                "version": "1.0",
                "use_networkx": self.use_networkx,
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

            logger.info(f"Loaded knowledge graph from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading knowledge graph: {str(e)}")
            return False