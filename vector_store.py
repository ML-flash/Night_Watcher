"""
Night_watcher Vector Store - Simplified FAISS-only implementation
Manages vector embeddings for semantic search and pattern detection.
"""

import os
import json
from file_utils import safe_json_load
import logging
import numpy as np
import pickle
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Required dependency
try:
    import faiss
except ImportError:
    raise ImportError("FAISS is required. Install with: pip install faiss-cpu")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")


class VectorStore:
    """
    Simplified vector storage using FAISS and sentence-transformers.
    """

    def __init__(self, base_dir: str = "data/vector_store", model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the vector store.
        
        Args:
            base_dir: Directory for vector store data
            model_name: Sentence transformer model to use
        """
        self.base_dir = base_dir
        self.index_path = os.path.join(base_dir, "faiss_index.bin")
        self.metadata_path = os.path.join(base_dir, "metadata.json")
        
        # Create directory
        os.makedirs(base_dir, exist_ok=True)
        
        # Initialize embedding model
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Track metadata
        self.metadata = {}
        self.id_to_index = {}
        
        self.logger = logging.getLogger("VectorStore")
        
        # Initialize or load index
        self._init_index()
        self._load_metadata()
    
    def _init_index(self):
        """Initialize or load FAISS index."""
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            self.logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
        else:
            # Create new flat L2 index
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.logger.info("Created new FAISS index")
    
    def _load_metadata(self):
        """Load metadata from disk."""
        if os.path.exists(self.metadata_path):
            try:
                data = safe_json_load(self.metadata_path, default=None)
                if data:
                    self.metadata = data.get('metadata', {})
                    self.id_to_index = data.get('id_to_index', {})
                    self.logger.info(f"Loaded metadata for {len(self.metadata)} items")
            except Exception as e:
                self.logger.error(f"Error loading metadata: {e}")
                self.metadata = {}
                self.id_to_index = {}
    
    def _save_metadata(self):
        """Save metadata to disk."""
        try:
            with open(self.metadata_path, 'w') as f:
                json.dump({
                    'metadata': self.metadata,
                    'id_to_index': self.id_to_index
                }, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving metadata: {e}")
    
    def _save_index(self):
        """Save FAISS index to disk."""
        try:
            faiss.write_index(self.index, self.index_path)
        except Exception as e:
            self.logger.error(f"Error saving index: {e}")
    
    def add_item(self, item_id: str, text: str, item_type: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Add an item to the vector store.
        
        Args:
            item_id: Unique identifier
            text: Text to embed
            item_type: Type of item (node, document, etc.)
            metadata: Additional metadata
            
        Returns:
            True if successful
        """
        try:
            # Skip if already exists
            if item_id in self.id_to_index:
                self.logger.debug(f"Item {item_id} already in index")
                return True
            
            # Get embedding
            embedding = self.model.encode(text, normalize_embeddings=True)
            
            # Add to FAISS
            index_position = self.index.ntotal
            self.index.add(embedding.reshape(1, -1).astype('float32'))
            
            # Update mappings
            self.id_to_index[item_id] = index_position
            
            # Store metadata
            self.metadata[item_id] = {
                "type": item_type,
                "text": text[:500],  # Store first 500 chars for reference
                "metadata": metadata or {},
                "added_at": datetime.now().isoformat()
            }
            
            # Save to disk
            self._save_index()
            self._save_metadata()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding item {item_id}: {e}")
            return False
    
    def search(self, query: str, limit: int = 10, item_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for similar items.
        
        Args:
            query: Search query
            limit: Maximum results
            item_type: Filter by type
            
        Returns:
            List of results with scores
        """
        try:
            # Get query embedding
            query_embedding = self.model.encode(query, normalize_embeddings=True)
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            
            # Search (get more than needed for filtering)
            k = min(limit * 3, self.index.ntotal)
            if k == 0:
                return []
                
            distances, indices = self.index.search(query_embedding, k)
            
            # Convert to results
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0:  # Invalid index
                    continue
                    
                # Find ID for this index
                item_id = None
                for id_, index_pos in self.id_to_index.items():
                    if index_pos == idx:
                        item_id = id_
                        break
                
                if not item_id:
                    continue
                
                # Get metadata
                meta = self.metadata.get(item_id, {})
                
                # Filter by type if specified
                if item_type and meta.get("type") != item_type:
                    continue
                
                # Convert L2 distance to similarity score (0-1)
                similarity = 1.0 / (1.0 + dist)
                
                results.append({
                    "id": item_id,
                    "score": float(similarity),
                    "type": meta.get("type"),
                    "text": meta.get("text"),
                    "metadata": meta.get("metadata", {})
                })
                
                if len(results) >= limit:
                    break
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching: {e}")
            return []
    
    def find_similar(self, item_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find items similar to a given item."""
        if item_id not in self.metadata:
            self.logger.warning(f"Item {item_id} not found")
            return []
        
        # Use the item's text as the query
        text = self.metadata[item_id].get("text", "")
        if not text:
            return []
        
        # Search and exclude self
        results = self.search(text, limit + 1)
        return [r for r in results if r["id"] != item_id][:limit]
    
    def sync_with_knowledge_graph(self, kg) -> Dict[str, int]:
        """Sync with knowledge graph nodes."""
        added = 0
        
        for node_id, node_data in kg.graph.nodes(data=True):
            # Skip if already indexed
            if node_id in self.id_to_index:
                continue
            
            # Create text representation
            text_parts = [
                f"Type: {node_data.get('type', 'unknown')}",
                f"Name: {node_data.get('name', '')}",
            ]
            
            # Add key attributes
            attrs = node_data.get('attributes', {})
            for key, value in attrs.items():
                if isinstance(value, (str, int, float, bool)):
                    text_parts.append(f"{key}: {value}")
            
            # Add source context if available
            source = node_data.get('source', {})
            if source.get('sentence'):
                text_parts.append(f"Context: {source['sentence']}")
            
            text = " ".join(text_parts)
            
            # Add to index
            if self.add_item(
                item_id=node_id,
                text=text,
                item_type="node",
                metadata={
                    "node_type": node_data.get('type'),
                    "name": node_data.get('name')
                }
            ):
                added += 1
        
        self.logger.info(f"Added {added} new nodes from knowledge graph")
        return {"nodes_added": added, "total_nodes": len(kg.graph.nodes)}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        type_counts = {}
        for meta in self.metadata.values():
            item_type = meta.get("type", "unknown")
            type_counts[item_type] = type_counts.get(item_type, 0) + 1
        
        return {
            "total_vectors": self.index.ntotal,
            "total_items": len(self.metadata),
            "type_counts": type_counts,
            "embedding_dim": self.embedding_dim,
            "model": self.model.device,
            "index_size_mb": os.path.getsize(self.index_path) / 1024 / 1024 if os.path.exists(self.index_path) else 0
        }
    
    def clear(self):
        """Clear all data (use with caution)."""
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.metadata = {}
        self.id_to_index = {}
        self._save_index()
        self._save_metadata()
        self.logger.info("Cleared all vector store data")

    def export_vector_store(self, path: str) -> str:
        """Export index and metadata to a directory."""
        import shutil

        os.makedirs(path, exist_ok=True)

        # Ensure latest data is saved
        self._save_index()
        self._save_metadata()

        dest_index = os.path.join(path, "faiss_index.bin")
        dest_meta = os.path.join(path, "metadata.json")

        try:
            shutil.copy2(self.index_path, dest_index)
            shutil.copy2(self.metadata_path, dest_meta)
        except Exception as e:
            self.logger.error(f"Error exporting vector store: {e}")

        return path
