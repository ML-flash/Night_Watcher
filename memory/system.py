"""
Night_watcher Memory System Module
Provides vector-based storage and retrieval for maintaining global context across analyses.
"""

import os
import re
import json
import pickle
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SimpleEmbeddingProvider:
    """Simple embedding provider using word counts (for testing without dependencies)"""

    def __init__(self, model_name: str = "default", vocab_size: int = 10000):
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

    def update_item(self, item_id: str, text: str, metadata: Dict[str, Any]) -> bool:
        """Update an existing item"""
        if item_id not in self.items:
            logger.warning(f"Item {item_id} not found in memory store")
            return False

        try:
            # Update embedding if text changed
            if text != self.items[item_id]["text"]:
                embedding = self.embedding_provider.embed_text(text)
                self.embeddings[item_id] = embedding

            # Update metadata
            self.items[item_id]["text"] = text
            self.items[item_id]["metadata"] = metadata
            self.items[item_id]["timestamp"] = datetime.now().isoformat()

            return True
        except Exception as e:
            logger.error(f"Error updating item in memory store: {str(e)}")
            return False

    def delete_item(self, item_id: str) -> bool:
        """Delete an item from the memory store"""
        if item_id not in self.items:
            logger.warning(f"Item {item_id} not found in memory store")
            return False

        try:
            del self.items[item_id]
            del self.embeddings[item_id]
            return True
        except Exception as e:
            logger.error(f"Error deleting item from memory store: {str(e)}")
            return False

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

        # Try to use more advanced providers if available
        if provider_type == "huggingface":
            try:
                from sentence_transformers import SentenceTransformer

                model_name = config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
                model = SentenceTransformer(model_name)

                class HuggingFaceProvider(SimpleEmbeddingProvider):
                    def __init__(self, model):
                        self.model = model

                    def embed_text(self, text):
                        if not text:
                            return np.zeros(self.model.get_sentence_embedding_dimension())
                        return self.model.encode(text)

                    def embed_batch(self, texts):
                        if not texts:
                            return np.array([])
                        return self.model.encode(texts)

                return HuggingFaceProvider(model)
            except ImportError:
                self.logger.warning("HuggingFace not available, falling back to SimpleEmbeddingProvider")
                return SimpleEmbeddingProvider()

        # Default simple provider
        return SimpleEmbeddingProvider()

    def _create_memory_store(self, store_type: str, embedding_provider: SimpleEmbeddingProvider) -> SimpleMemoryStore:
        """Create the appropriate memory store based on type"""
        # Try to use FAISS if available
        if store_type == "faiss":
            try:
                import faiss

                class FaissStore(SimpleMemoryStore):
                    def __init__(self, embedding_provider):
                        super().__init__(embedding_provider)
                        self.index = None
                        self.dimension = None
                        self.id_map = []

                    def add_item(self, item_id, text, metadata):
                        embedding = self.embedding_provider.embed_text(text)

                        if self.index is None:
                            self.dimension = embedding.shape[0]
                            self.index = faiss.IndexFlatL2(self.dimension)

                        self.items[item_id] = {
                            "text": text,
                            "metadata": metadata,
                            "timestamp": datetime.now().isoformat()
                        }

                        self.index.add(np.array([embedding], dtype=np.float32))
                        self.id_map.append(item_id)

                        return True

                    def search(self, query, limit=5):
                        if not self.index or self.index.ntotal == 0:
                            return []

                        query_embedding = self.embedding_provider.embed_text(query)
                        query_embedding = np.array([query_embedding], dtype=np.float32)

                        distances, indices = self.index.search(query_embedding, min(limit, len(self.id_map)))

                        results = []
                        for i, idx in enumerate(indices[0]):
                            if idx < len(self.id_map):
                                item_id = self.id_map[idx]
                                item = self.items[item_id].copy()
                                item["id"] = item_id
                                item["similarity"] = float(1.0 / (1.0 + distances[0][i]))
                                results.append(item)

                        return results

                return FaissStore(embedding_provider)
            except ImportError:
                self.logger.warning("FAISS not available, falling back to SimpleMemoryStore")

        # Default simple store
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
        from uuid import uuid4
        item_id = f"article_{uuid4().hex[:8]}"

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

    def store_counter_narrative(self, narrative: Dict[str, Any], parent_id: str = "") -> str:
        """
        Store a counter-narrative in the memory system.

        Args:
            narrative: Counter-narrative dict
            parent_id: ID of the parent article analysis

        Returns:
            ID of the stored item
        """
        demographic = narrative.get("demographic", "unknown")
        content = narrative.get("content", "")

        if not content:
            self.logger.warning("Empty counter-narrative content")
            return ""

        # Create a unique ID
        from uuid import uuid4
        item_id = f"narrative_{uuid4().hex[:8]}"

        # Create metadata
        metadata = {
            "type": "counter_narrative",
            "demographic": demographic,
            "parent_id": parent_id,
            "timestamp": narrative.get("timestamp", datetime.now().isoformat())
        }

        # Store in vector store
        success = self.store.add_item(item_id, content, metadata)
        if success:
            self.last_update_time = datetime.now()
            return item_id
        else:
            return ""

    def store_bridging_content(self, content: Dict[str, Any], parent_id: str = "") -> str:
        """
        Store bridging content in the memory system.

        Args:
            content: Bridging content dict
            parent_id: ID of the parent article analysis

        Returns:
            ID of the stored item
        """
        bridging_groups = content.get("bridging_groups", [])
        text_content = content.get("content", "")

        if not text_content:
            self.logger.warning("Empty bridging content")
            return ""

        # Create a unique ID
        from uuid import uuid4
        item_id = f"bridge_{uuid4().hex[:8]}"

        # Create metadata
        metadata = {
            "type": "bridging_content",
            "bridging_groups": bridging_groups,
            "parent_id": parent_id,
            "timestamp": content.get("timestamp", datetime.now().isoformat())
        }

        # Store in vector store
        success = self.store.add_item(item_id, text_content, metadata)
        if success:
            self.last_update_time = datetime.now()
            return item_id
        else:
            return ""

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

    def find_similar_narratives(self, query: str, demographic: str = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find counter-narratives similar to the query, optionally filtering by demographic.

        Args:
            query: Search query
            demographic: Optional demographic to filter by
            limit: Maximum number of results

        Returns:
            List of similar counter-narratives
        """
        results = self.store.search(query, limit * 2)  # Get more than needed for filtering

        # Filter for counter_narrative type and optional demographic
        filtered = [
            item for item in results
            if item.get("metadata", {}).get("type") == "counter_narrative" and
               (demographic is None or item.get("metadata", {}).get("demographic") == demographic)
        ]

        return filtered[:limit]

    def get_recent_analyses(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get analyses from the last N days.
        Note: This is a simple implementation that requires scanning all items.
        A production implementation would use a database with proper indexing.

        Args:
            days: Number of days to look back

        Returns:
            List of recent analyses
        """
        try:
            # Calculate cutoff date
            from datetime import timedelta
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

    def search_all(self, query: str, limit: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search all content in the memory system.

        Args:
            query: Search query
            limit: Maximum number of results per category

        Returns:
            Dictionary of results by type
        """
        all_results = self.store.search(query, limit * 3)  # Get more than needed for categorization

        # Categorize results
        categorized = {
            "article_analyses": [],
            "counter_narratives": [],
            "bridging_content": [],
            "user_notes": []
        }

        for item in all_results:
            content_type = item.get("metadata", {}).get("type", "other")

            if content_type == "article_analysis":
                if len(categorized["article_analyses"]) < limit:
                    categorized["article_analyses"].append(item)
            elif content_type == "counter_narrative":
                if len(categorized["counter_narratives"]) < limit:
                    categorized["counter_narratives"].append(item)
            elif content_type == "bridging_content":
                if len(categorized["bridging_content"]) < limit:
                    categorized["bridging_content"].append(item)
            elif content_type == "user_note":
                if len(categorized["user_notes"]) < limit:
                    categorized["user_notes"].append(item)

        return categorized

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