"""
Night_watcher Memory System
Provides vector-based storage, retrieval, and knowledge graph for maintaining context across analyses.
"""

import os
import re
import pickle
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Set

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# ==========================================
# Simple Embedding Provider
# ==========================================

class SimpleEmbeddingProvider:
    """Simple embedding provider using word counts for vector representations"""

    def __init__(self, model_name: str = "default", vocab_size: int = 5000):
        """Initialize with vocabulary size"""
        self.model_name = model_name
        self.vocab_size = vocab_size
        self._word_dict = {}
        self._next_id = 0

    def _get_word_id(self, word: str) -> int:
        """Get or create ID for a word"""
        if word not in self._word_dict:
            if self._next_id < self.vocab_size:
                self._word_dict[word] = self._next_id
                self._next_id += 1
            else:
                return hash(word) % self.vocab_size
        return self._word_dict[word]

    def embed_text(self, text: str) -> np.ndarray:
        """Create a simple bag-of-words embedding"""
        if not text:
            return np.zeros(self.vocab_size)

        words = text.lower().split()
        embedding = np.zeros(self.vocab_size)

        for word in words:
            word_id = self._get_word_id(word)
            embedding[word_id] += 1

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts"""
        return np.array([self.embed_text(text) for text in texts])

# ==========================================
# Memory Store
# ==========================================

class SimpleMemoryStore:
    """Simple in-memory vector store implementation"""

    def __init__(self, embedding_provider: SimpleEmbeddingProvider):
        """Initialize with embedding provider"""
        self.embedding_provider = embedding_provider
        self.items = {}  # Dictionary mapping item_id to metadata
        self.embeddings = {}  # Dictionary mapping item_id to embedding

    def add_item(self, item_id: str, text: str, metadata: Dict[str, Any]) -> bool:
        """Add an item to the memory store"""
        try:
            embedding = self.embedding_provider.embed_text(text)

            # Store the item
            self.items[item_id] = {
                "text": text,
                "metadata": metadata,
                "timestamp": datetime.now().isoformat()
            }

            self.embeddings[item_id] = embedding
            return True
        except Exception as e:
            logger.error(f"Error adding item to memory store: {str(e)}")
            return False

    def add_items(self, items: List[Tuple[str, str, Dict[str, Any]]]) -> bool:
        """Add multiple items to the memory store"""
        try:
            # Extract texts for batch embedding
            texts = [text for _, text, _ in items]
            batch_embeddings = self.embedding_provider.embed_batch(texts)

            # Store items with their embeddings
            for i, (item_id, text, metadata) in enumerate(items):
                self.items[item_id] = {
                    "text": text,
                    "metadata": metadata,
                    "timestamp": datetime.now().isoformat()
                }

                self.embeddings[item_id] = batch_embeddings[i]

            return True
        except Exception as e:
            logger.error(f"Error adding items to memory store: {str(e)}")
            return False

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for items similar to the query"""
        try:
            query_embedding = self.embedding_provider.embed_text(query)

            # Calculate similarities
            similarities = []
            for item_id, embedding in self.embeddings.items():
                similarity = np.dot(query_embedding, embedding)
                similarities.append((item_id, similarity))

            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Get top results
            results = []
            for item_id, similarity in similarities[:limit]:
                item = self.items[item_id].copy()
                item["id"] = item_id
                item["similarity"] = float(similarity)
                results.append(item)

            return results
        except Exception as e:
            logger.error(f"Error searching memory store: {str(e)}")
            return []

    def get_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific item by ID"""
        if item_id not in self.items:
            return None

        item = self.items[item_id].copy()
        item["id"] = item_id
        return item

    def save(self, path: str) -> bool:
        """Save the memory store to disk"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)

            with open(path, 'wb') as f:
                pickle.dump((self.items, self.embeddings), f)

            return True
        except Exception as e:
            logger.error(f"Error saving memory store: {str(e)}")
            return False

    def load(self, path: str) -> bool:
        """Load the memory store from disk"""
        if not os.path.exists(path):
            logger.warning(f"Memory store file not found: {path}")
            return False

        try:
            with open(path, 'rb') as f:
                self.items, self.embeddings = pickle.load(f)

            return True
        except Exception as e:
            logger.error(f"Error loading memory store: {str(e)}")
            return False

# ==========================================
# Memory System
# ==========================================

class MemorySystem:
    """Memory system for Night_watcher that manages a vector store and provides high-level operations"""

    def __init__(self, store_type: str = "simple", config: Dict[str, Any] = None):
        """Initialize the memory system with the specified store type"""
        self.logger = logging.getLogger("MemorySystem")

        if config is None:
            config = {}

        # Configure embedding provider
        self.embedding_provider = self._create_embedding_provider(config)

        # Create memory store
        self.store = self._create_memory_store(store_type, self.embedding_provider)

        # Track last analysis time for temporal queries
        self.last_update_time = datetime.now()

    def _create_embedding_provider(self, config: Dict[str, Any]) -> SimpleEmbeddingProvider:
        """Create the appropriate embedding provider based on config"""
        provider_type = config.get("embedding_provider", "simple")

        # For now, always use SimpleEmbeddingProvider
        return SimpleEmbeddingProvider()

    def _create_memory_store(self, store_type: str, embedding_provider: SimpleEmbeddingProvider) -> SimpleMemoryStore:
        """Create the appropriate memory store based on type"""
        # For now, always use SimpleMemoryStore
        return SimpleMemoryStore(embedding_provider)

    def store_article_analysis(self, analysis_result: Dict[str, Any]) -> str:
        """
        Store an article analysis in the memory system.

        Args:
            analysis_result: Analysis result dict

        Returns:
            ID of the stored item
        """
        if "article" not in analysis_result or "analysis" not in analysis_result:
            self.logger.warning("Invalid analysis result format")
            return ""

        article = analysis_result["article"]
        analysis = analysis_result["analysis"]

        # Create a unique ID
        item_id = f"article_{uuid.uuid4().hex[:8]}"

        # Extract key information for text representation
        title = article.get("title", "Untitled")
        source = article.get("source", "Unknown")
        content_summary = article.get("content", "")[:500] + "..."  # First 500 chars

        # Create text for vector embedding (focus on analysis and key article info)
        text = f"""
        TITLE: {title}
        SOURCE: {source}
        ANALYSIS: {analysis}
        CONTENT_SUMMARY: {content_summary}
        """

        # Create metadata
        metadata = {
            "type": "article_analysis",
            "title": title,
            "source": source,
            "bias_label": article.get("bias_label", "unknown"),
            "url": article.get("url", ""),
            "published": article.get("published", ""),
            "manipulation_score": self._extract_manipulation_score(analysis),
            "analysis_timestamp": analysis_result.get("timestamp", datetime.now().isoformat())
        }

        # Store in vector store
        success = self.store.add_item(item_id, text, metadata)
        if success:
            self.last_update_time = datetime.now()
            return item_id
        else:
            return ""

    def _extract_manipulation_score(self, analysis: str) -> int:
        """Extract manipulation score from analysis text"""
        try:
            if "MANIPULATION SCORE" in analysis:
                score_text = analysis.split("MANIPULATION SCORE:")[1].split("\n")[0]
                # Extract numbers from text
                numbers = [int(s) for s in re.findall(r'\d+', score_text)]
                if numbers:
                    return numbers[0]
            return 0
        except Exception as e:
            self.logger.error(f"Error extracting manipulation score: {str(e)}")
            return 0

    def find_similar_analyses(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find analyses similar to the query.

        Args:
            query: Search query (can be a title, topic, or analysis text)
            limit: Maximum number of results

        Returns:
            List of similar analyses
        """
        results = self.store.search(query, limit)

        # Filter for article_analysis type
        return [item for item in results if item.get("metadata", {}).get("type") == "article_analysis"]

    def get_recent_analyses(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get analyses from the last N days.

        Args:
            days: Number of days to look back

        Returns:
            List of recent analyses
        """
        try:
            # Calculate cutoff date
            cutoff = datetime.now() - timedelta(days=days)
            cutoff_str = cutoff.isoformat()

            recent_analyses = []

            # This is inefficient for large stores - would be better with a proper database
            for item_id, item in self.store.items.items():
                metadata = item.get("metadata", {})

                if (metadata.get("type") == "article_analysis" and
                        metadata.get("analysis_timestamp", "") >= cutoff_str):
                    result = item.copy()
                    result["id"] = item_id
                    recent_analyses.append(result)

            # Sort by timestamp (newest first)
            recent_analyses.sort(
                key=lambda x: x.get("metadata", {}).get("analysis_timestamp", ""),
                reverse=True
            )

            return recent_analyses

        except Exception as e:
            self.logger.error(f"Error getting recent analyses: {str(e)}")
            return []

    def save(self, path: str) -> bool:
        """
        Save the memory system to disk.

        Args:
            path: Path to save to

        Returns:
            True if save was successful
        """
        return self.store.save(path)

    def load(self, path: str) -> bool:
        """
        Load the memory system from disk.

        Args:
            path: Path to load from

        Returns:
            True if load was successful
        """
        return self.store.load(path)

# ==========================================
# Knowledge Graph
# ==========================================

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
        self.confidence = "low"  # Default confidence level

    def update(self, weight: float = None, attributes: Dict[str, Any] = None) -> None:
        """Update relationship weight and/or attributes"""
        if weight is not None:
            self.weight = weight

        if attributes:
            self.attributes.update(attributes)

        self.updated_at = datetime.now().isoformat()

    def add_evidence(self, evidence_text: str, source_id: str) -> None:
        """Add evidence supporting this relationship"""
        self.evidence.append({
            "text": evidence_text,
            "source": source_id,
            "timestamp": datetime.now().isoformat()
        })
        
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
        rel.confidence = data.get("confidence", "low")
            
        return rel

class KnowledgeGraph:
    """Knowledge graph for tracking authoritarian patterns and actors"""

    # Entity types
    ACTOR = "actor"
    INSTITUTION = "institution"
    EVENT = "event"
    ACTION = "action"
    ARTIFACT = "artifact"
    NARRATIVE = "narrative"
    INDICATOR = "indicator"
    TOPIC = "topic"
    
    # Relationship types
    CONTROLS = "controls"
    INFLUENCES = "influences"
    UNDERMINES = "undermines"
    PERFORMS = "performs"
    TARGETS = "targets"
    PARTICIPATES_IN = "participates_in"
    DEMONSTRATES = "demonstrates"

    def __init__(self):
        """Initialize the knowledge graph"""
        self.entities = {}  # id -> Entity
        self.relationships = {}  # id -> Relationship
        self.source_relations = {}  # source_id -> [relation_ids]
        self.target_relations = {}  # target_id -> [relation_ids]
        self.entity_types = {}  # type -> [entity_ids]
        self.relation_types = {}  # type -> [relation_ids]
        
        # Track creation date
        self.creation_date = datetime.now().isoformat()
        self.last_update = self.creation_date
        
        # Cache
        self._entity_cache = {}
        self._entity_cache_size = 100
        
        self.logger = logging.getLogger("KnowledgeGraph")

    def add_entity(self, entity_type: str, name: str, attributes: Dict[str, Any] = None) -> str:
        """Add an entity to the graph"""
        entity = Entity(entity_type=entity_type, name=name, attributes=attributes or {})
        entity_id = entity.id

        # Add to entities
        self.entities[entity_id] = entity

        # Index by type
        if entity.type not in self.entity_types:
            self.entity_types[entity.type] = []
        self.entity_types[entity.type].append(entity_id)
        
        # Update last modified time
        self.last_update = datetime.now().isoformat()

        return entity_id

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
        
        relation_id = relationship.id

        # Add to relationships
        self.relationships[relation_id] = relationship

        # Index by source and target
        if source_id not in self.source_relations:
            self.source_relations[source_id] = []
        self.source_relations[source_id].append(relation_id)

        if target_id not in self.target_relations:
            self.target_relations[target_id] = []
        self.target_relations[target_id].append(relation_id)

        # Index by type
        if relation_type not in self.relation_types:
            self.relation_types[relation_type] = []
        self.relation_types[relation_type].append(relation_id)
        
        # Update last modified time
        self.last_update = datetime.now().isoformat()

        return relation_id

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID with caching"""
        if entity_id in self._entity_cache:
            return self._entity_cache[entity_id]

        entity = self.entities.get(entity_id)

        if entity:
            # Add to cache, removing oldest if needed
            if len(self._entity_cache) >= self._entity_cache_size:
                self._entity_cache.pop(next(iter(self._entity_cache)))
            self._entity_cache[entity_id] = entity

        return entity

    def find_entity_by_name(self, name: str, entity_type: Optional[str] = None) -> Optional[Entity]:
        """Find an entity by name, optionally filtering by type"""
        for entity_id, entity in self.entities.items():
            if entity.name.lower() == name.lower():
                if entity_type is None or entity.type == entity_type:
                    return entity
        return None

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

    def save(self, path: str) -> bool:
        """Save the knowledge graph to disk"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)

            data = {
                "entities": {eid: entity.to_dict() for eid, entity in self.entities.items()},
                "relationships": {rid: rel.to_dict() for rid, rel in self.relationships.items()},
                "metadata": {
                    "version": "1.0",
                    "creation_date": self.creation_date,
                    "last_update": self.last_update,
                    "entity_count": len(self.entities),
                    "relationship_count": len(self.relationships),
                    "entity_types": {t: len(ids) for t, ids in self.entity_types.items()},
                    "relation_types": {t: len(ids) for t, ids in self.relation_types.items()},
                    "timestamp": datetime.now().isoformat()
                }
            }

            with open(path, 'wb') as f:
                pickle.dump(data, f)

            self.logger.info(f"Saved knowledge graph to {path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving knowledge graph: {str(e)}")
            return False

    def load(self, path: str) -> bool:
        """Load the knowledge graph from disk"""
        if not os.path.exists(path):
            self.logger.warning(f"Knowledge graph file not found: {path}")
            return False

        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)

            # Clear existing data
            self.entities = {}
            self.relationships = {}
            self.source_relations = {}
            self.target_relations = {}
            self.entity_types = {}
            self.relation_types = {}
            self._entity_cache.clear()

            # Load entities
            for entity_id, entity_data in data.get("entities", {}).items():
                entity = Entity.from_dict(entity_data)
                self.entities[entity_id] = entity
                
                # Index by type
                if entity.type not in self.entity_types:
                    self.entity_types[entity.type] = []
                self.entity_types[entity.type].append(entity_id)

            # Load relationships
            for rel_id, rel_data in data.get("relationships", {}).items():
                rel = Relationship.from_dict(rel_data)
                self.relationships[rel_id] = rel
                
                # Index by source and target
                source_id = rel.source_id
                target_id = rel.target_id
                
                if source_id not in self.source_relations:
                    self.source_relations[source_id] = []
                self.source_relations[source_id].append(rel_id)
                
                if target_id not in self.target_relations:
                    self.target_relations[target_id] = []
                self.target_relations[target_id].append(rel_id)
                
                # Index by type
                if rel.type not in self.relation_types:
                    self.relation_types[rel.type] = []
                self.relation_types[rel.type].append(rel_id)
                
            # Load metadata
            metadata = data.get("metadata", {})
            self.creation_date = metadata.get("creation_date", self.creation_date)
            self.last_update = metadata.get("last_update", self.last_update)

            self.logger.info(f"Loaded knowledge graph with {len(self.entities)} entities and {len(self.relationships)} relationships")
            return True
        except Exception as e:
            self.logger.error(f"Error loading knowledge graph: {str(e)}")
            return False
