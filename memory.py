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

# Import knowledge graph implementation
from knowledge_graph import KnowledgeGraph

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

        # Initialize knowledge graph
        self.knowledge_graph = KnowledgeGraph(use_networkx=config.get("use_networkx", True))

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
        Store an article analysis in the memory system and update the knowledge graph.

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
        
        # Update knowledge graph if structured elements are available
        if "structured_elements" in analysis_result:
            # Add article ID to the article data for reference
            article["id"] = item_id
            self.knowledge_graph.process_article_analysis(article, analysis_result)
        
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

    def get_authoritarian_trends(self, days: int = 90) -> Dict[str, Any]:
        """
        Get authoritarian trends from the knowledge graph.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Dict with trend analysis results
        """
        return self.knowledge_graph.get_authoritarian_trends(days)
    
    def get_influential_actors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most influential actors from the knowledge graph.
        
        Args:
            limit: Maximum number of actors to return
            
        Returns:
            List of influential actors with their scores
        """
        return self.knowledge_graph.get_influential_actors(limit)
    
    def analyze_democratic_erosion(self, days: int = 90) -> Dict[str, Any]:
        """
        Get comprehensive analysis of democratic erosion patterns.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Dict with democratic erosion analysis
        """
        return self.knowledge_graph.analyze_democratic_erosion(days)

    def save(self, path: str) -> bool:
        """
        Save the memory system to disk.

        Args:
            path: Path to save to

        Returns:
            True if save was successful
        """
        # Save the memory store
        store_saved = self.store.save(path)
        
        # Save the knowledge graph
        kg_path = os.path.join(os.path.dirname(path), "knowledge_graph.pkl")
        kg_saved = self.knowledge_graph.save(kg_path)
        
        return store_saved and kg_saved

    def load(self, path: str) -> bool:
        """
        Load the memory system from disk.

        Args:
            path: Path to load from

        Returns:
            True if load was successful
        """
        # Load the memory store
        store_loaded = self.store.load(path)
        
        # Load the knowledge graph
        kg_path = os.path.join(os.path.dirname(path), "knowledge_graph.pkl")
        if os.path.exists(kg_path):
            kg_loaded = self.knowledge_graph.load(kg_path)
        else:
            kg_loaded = True  # Not an error if KG doesn't exist yet
        
        return store_loaded and kg_loaded
