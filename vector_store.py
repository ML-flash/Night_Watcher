"""
Night_watcher Vector Store
Manages vector embeddings for semantic search and pattern detection.
Integrates with the knowledge graph for enhanced relationship discovery.
"""

import os
import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple, Set

# For serialization/deserialization
import pickle

# Vector stores
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Installing FAISS is recommended for optimal performance.")

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

# Import for type hints but don't cause errors if missing
try:
    from knowledge_graph import KnowledgeGraph
    HAS_KG = True
except ImportError:
    HAS_KG = False

class VectorStore:
    """
    Vector storage for the Night_watcher framework. Manages embeddings for
    entities and documents to enable semantic search and pattern detection.
    Integrates with FAISS or ChromaDB for efficient vector similarity.
    """

    def __init__(self, base_dir: str = "data/vector_store", 
                 embedding_provider: str = "local",
                 embedding_dim: int = 384,
                 index_type: str = "flat",
                 kg_instance=None):
        """
        Initialize the vector store.
        
        Args:
            base_dir: Directory for vector store data
            embedding_provider: 'local', 'openai', or 'instructor' 
            embedding_dim: Dimension of embedding vectors
            index_type: 'flat' or 'hnsw' (Hierarchical Navigable Small World)
            kg_instance: Optional KnowledgeGraph instance for integration
        """
        self.base_dir = base_dir
        self.nodes_dir = os.path.join(base_dir, "nodes")
        self.documents_dir = os.path.join(base_dir, "documents")
        self.index_path = os.path.join(base_dir, "faiss_index.bin")
        self.metadata_path = os.path.join(base_dir, "metadata.pickle")
        
        self.embedding_provider = embedding_provider
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        
        # Track metadata for vectors (id -> metadata mapping)
        self.metadata = {}
        self.id_to_index = {}  # Maps entity/doc IDs to FAISS indices
        
        # Logger setup
        self.logger = logging.getLogger("VectorStore")
        
        # Create directories
        os.makedirs(self.nodes_dir, exist_ok=True)
        os.makedirs(self.documents_dir, exist_ok=True)
        
        # Initialize embedding provider
        self._init_embedding_provider()
        
        # Initialize vector index
        self._init_vector_index()
        
        # Knowledge graph integration
        self.kg = kg_instance
        
        # Load existing data if available
        self._load_metadata()
    
    def _init_embedding_provider(self):
        """Initialize the embedding provider based on configuration."""
        if self.embedding_provider == "local":
            try:
                # Try to import sentence-transformers
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dim
                self.logger.info("Using sentence-transformers for embeddings")
            except ImportError:
                self.logger.warning("sentence-transformers not available, using fallback")
                self.model = None
                
        elif self.embedding_provider == "openai":
            try:
                import openai
                self.model = "text-embedding-3-small"  # OpenAI embedding model
                self.logger.info("Using OpenAI for embeddings")
            except ImportError:
                self.logger.warning("OpenAI library not available, using fallback")
                self.model = None
                
        elif self.embedding_provider == "instructor":
            try:
                from InstructorEmbedding import INSTRUCTOR
                self.model = INSTRUCTOR('hkunlp/instructor-large')
                self.logger.info("Using Instructor embeddings")
            except ImportError:
                self.logger.warning("InstructorEmbedding not available, using fallback")
                self.model = None
        else:
            self.logger.warning(f"Unknown embedding provider: {self.embedding_provider}")
            self.model = None
    
    def _init_vector_index(self):
        """Initialize the vector index based on configuration."""
        # FAISS implementation (efficient vector similarity search)
        if FAISS_AVAILABLE:
            if os.path.exists(self.index_path):
                try:
                    self.index = faiss.read_index(self.index_path)
                    self.logger.info(f"Loaded FAISS index from {self.index_path}")
                except Exception as e:
                    self.logger.error(f"Error loading FAISS index: {e}")
                    self._create_new_index()
            else:
                self._create_new_index()
        # ChromaDB fallback
        elif CHROMA_AVAILABLE:
            self.client = chromadb.PersistentClient(path=os.path.join(self.base_dir, "chroma"))
            # Create collections if they don't exist
            self.node_collection = self.client.get_or_create_collection("nodes")
            self.doc_collection = self.client.get_or_create_collection("documents")
            self.logger.info("Using ChromaDB as vector store")
        else:
            # Simple numpy fallback (not recommended for production)
            self.vectors = np.zeros((0, self.embedding_dim), dtype=np.float32)
            self.logger.warning("No vector database available. Using simple numpy array (not scalable)")
    
    def _create_new_index(self):
        """Create a new FAISS index."""
        if self.index_type == "flat":
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        elif self.index_type == "hnsw":
            # HNSW index is more efficient for large datasets
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)  # 32 neighbors
        else:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        self.logger.info(f"Created new FAISS index ({self.index_type})")
    
    def _load_metadata(self):
        """Load metadata for vectors."""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.metadata = data.get('metadata', {})
                    self.id_to_index = data.get('id_to_index', {})
                self.logger.info(f"Loaded metadata for {len(self.metadata)} vectors")
            except Exception as e:
                self.logger.error(f"Error loading metadata: {e}")
                self.metadata = {}
                self.id_to_index = {}
    
    def _save_metadata(self):
        """Save metadata for vectors."""
        try:
            with open(self.metadata_path, 'wb') as f:
                pickle.dump({
                    'metadata': self.metadata,
                    'id_to_index': self.id_to_index
                }, f)
            self.logger.debug("Saved vector metadata")
        except Exception as e:
            self.logger.error(f"Error saving metadata: {e}")
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text using configured provider.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        if not text:
            # Return zero vector for empty text
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        try:
            if self.embedding_provider == "local" and self.model:
                # Sentence-transformers
                return self.model.encode(text, normalize_embeddings=True)
                
            elif self.embedding_provider == "openai" and self.model:
                # OpenAI embeddings
                import openai
                response = openai.Embedding.create(input=text, model=self.model)
                return np.array(response["data"][0]["embedding"], dtype=np.float32)
                
            elif self.embedding_provider == "instructor" and self.model:
                # Instructor embeddings (with instruction)
                instruction = "Represent this text for political pattern detection"
                embeddings = self.model.encode([[instruction, text]])
                return embeddings[0]
                
            else:
                # Simple fallback using TF-IDF and SVD
                self.logger.warning("Using fallback embedding method")
                return self._fallback_embedding(text)
                
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def _fallback_embedding(self, text: str) -> np.ndarray:
        """
        Fallback embedding method using TF-IDF and truncated SVD.
        Not recommended for production use.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            
            # Create a simple TF-IDF vector
            vectorizer = TfidfVectorizer(max_features=1000)
            tfidf_matrix = vectorizer.fit_transform([text])
            
            # Reduce to embedding_dim using SVD
            svd = TruncatedSVD(n_components=min(self.embedding_dim, tfidf_matrix.shape[1]))
            embedding = svd.fit_transform(tfidf_matrix)[0]
            
            # Pad if necessary
            if len(embedding) < self.embedding_dim:
                embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)))
                
            return embedding.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Error in fallback embedding: {e}")
            return np.random.randn(self.embedding_dim).astype(np.float32)
    
    def add_node(self, node_id: str, node_data: Dict[str, Any]) -> bool:
        """
        Add a knowledge graph node to the vector store.
        
        Args:
            node_id: Node identifier
            node_data: Node data dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract text for embedding
            node_type = node_data.get("type", "unknown")
            node_name = node_data.get("name", "")
            attributes = node_data.get("attributes", {})
            
            # Combine node information for rich embedding
            text_parts = [
                f"Type: {node_type}",
                f"Name: {node_name}"
            ]
            
            # Add important attributes
            for key, value in attributes.items():
                if isinstance(value, (str, int, float, bool)):
                    text_parts.append(f"{key}: {value}")
            
            # Add source sentence if available
            if "source" in node_data and "sentence" in node_data["source"]:
                text_parts.append(f"Context: {node_data['source']['sentence']}")
                
            # Join all parts
            text = " ".join(text_parts)
            
            # Get embedding
            embedding = self._get_embedding(text)
            
            # Store in appropriate backend
            if FAISS_AVAILABLE:
                # Add to FAISS index
                index = len(self.id_to_index)
                self.id_to_index[node_id] = index
                
                # Reshape for FAISS (expects 2D array)
                reshaped = embedding.reshape(1, -1).astype('float32')
                self.index.add(reshaped)
                
                # Save to disk to ensure persistence
                faiss.write_index(self.index, self.index_path)
                
            elif CHROMA_AVAILABLE:
                # Add to ChromaDB
                self.node_collection.add(
                    documents=[text],
                    embeddings=[embedding.tolist()],
                    ids=[node_id],
                    metadatas=[{"type": node_type, "name": node_name}]
                )
            else:
                # Simple numpy array
                self.vectors = np.vstack([self.vectors, embedding])
                self.id_to_index[node_id] = self.vectors.shape[0] - 1
            
            # Store metadata
            self.metadata[node_id] = {
                "type": "node",
                "node_type": node_type,
                "name": node_name,
                "text": text,
                "added_at": datetime.now().isoformat()
            }
            
            # Save node vector separately
            node_path = os.path.join(self.nodes_dir, f"{node_id}.npz")
            np.savez(node_path, embedding=embedding, metadata=json.dumps(self.metadata[node_id]))
            
            # Save metadata
            self._save_metadata()
            
            self.logger.info(f"Added node {node_id} to vector store")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding node {node_id} to vector store: {e}")
            return False
    
    def add_document(self, doc_id: str, content: str, metadata: Dict[str, Any]) -> bool:
        """
        Add a document to the vector store.
        
        Args:
            doc_id: Document identifier
            content: Document text content
            metadata: Document metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Truncate content if too long
            max_length = 10000  # Character limit for embedding
            if len(content) > max_length:
                truncated = content[:max_length//2] + "..." + content[-max_length//2:]
            else:
                truncated = content
                
            # Get embedding
            embedding = self._get_embedding(truncated)
            
            # Store in appropriate backend
            if FAISS_AVAILABLE:
                # Add to FAISS index
                index = len(self.id_to_index)
                self.id_to_index[doc_id] = index
                
                # Reshape for FAISS
                reshaped = embedding.reshape(1, -1).astype('float32')
                self.index.add(reshaped)
                
                # Save to disk
                faiss.write_index(self.index, self.index_path)
                
            elif CHROMA_AVAILABLE:
                # Add to ChromaDB
                self.doc_collection.add(
                    documents=[truncated],
                    embeddings=[embedding.tolist()],
                    ids=[doc_id],
                    metadatas=[{
                        "title": metadata.get("title", ""),
                        "source": metadata.get("source", "")
                    }]
                )
            else:
                # Simple numpy array
                self.vectors = np.vstack([self.vectors, embedding])
                self.id_to_index[doc_id] = self.vectors.shape[0] - 1
            
            # Store metadata
            self.metadata[doc_id] = {
                "type": "document",
                "title": metadata.get("title", ""),
                "source": metadata.get("source", ""),
                "added_at": datetime.now().isoformat()
            }
            
            # Save document vector separately
            doc_path = os.path.join(self.documents_dir, f"{doc_id}.npz")
            np.savez(doc_path, embedding=embedding, metadata=json.dumps(self.metadata[doc_id]))
            
            # Save metadata
            self._save_metadata()
            
            self.logger.info(f"Added document {doc_id} to vector store")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding document {doc_id} to vector store: {e}")
            return False
    
    def similar_nodes(self, query: str, node_type: Optional[str] = None, 
                     limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find nodes similar to a text query.
        
        Args:
            query: Text query
            node_type: Optional filter by node type
            limit: Maximum number of results
            
        Returns:
            List of similar nodes with similarity scores
        """
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        results = []
        
        try:
            if FAISS_AVAILABLE:
                # Use FAISS for similarity search
                query_embedding = query_embedding.reshape(1, -1).astype('float32')
                
                # Search the index
                similarities, indices = self.index.search(query_embedding, limit * 3)  # Get more, will filter
                
                # Convert to results
                for i, idx in enumerate(indices[0]):
                    if idx < 0:  # FAISS returns -1 for empty slots
                        continue
                        
                    # Find the ID for this index
                    node_id = None
                    for id, index in self.id_to_index.items():
                        if index == idx:
                            node_id = id
                            break
                    
                    if not node_id:
                        continue
                        
                    # Get metadata
                    meta = self.metadata.get(node_id, {})
                    
                    # Skip if not a node or wrong type
                    if meta.get("type") != "node":
                        continue
                        
                    if node_type and meta.get("node_type") != node_type:
                        continue
                    
                    # Add to results
                    results.append({
                        "id": node_id,
                        "score": float(1.0 - similarities[0][i] / 100.0),  # Convert distance to similarity
                        "type": meta.get("node_type", "unknown"),
                        "name": meta.get("name", ""),
                        "text": meta.get("text", "")
                    })
                    
                    if len(results) >= limit:
                        break
                        
            elif CHROMA_AVAILABLE:
                # Use ChromaDB
                where_clause = {"type": node_type} if node_type else None
                
                search_results = self.node_collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=limit,
                    where=where_clause
                )
                
                # Convert to results
                for i, node_id in enumerate(search_results.get('ids', [[]])[0]):
                    meta = self.metadata.get(node_id, {})
                    
                    results.append({
                        "id": node_id,
                        "score": float(search_results.get('distances', [[]])[0][i]),
                        "type": meta.get("node_type", "unknown"),
                        "name": meta.get("name", ""),
                        "text": meta.get("text", "")
                    })
            else:
                # Simple numpy implementation
                # Calculate cosine similarity
                similarities = np.dot(self.vectors, query_embedding) / (
                    np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(query_embedding)
                )
                
                # Get top indices
                top_indices = np.argsort(similarities)[::-1][:limit * 3]
                
                # Convert to results
                for idx in top_indices:
                    # Find the ID for this index
                    node_id = None
                    for id, index in self.id_to_index.items():
                        if index == idx:
                            node_id = id
                            break
                    
                    if not node_id:
                        continue
                        
                    # Get metadata
                    meta = self.metadata.get(node_id, {})
                    
                    # Skip if not a node or wrong type
                    if meta.get("type") != "node":
                        continue
                        
                    if node_type and meta.get("node_type") != node_type:
                        continue
                    
                    # Add to results
                    results.append({
                        "id": node_id,
                        "score": float(similarities[idx]),
                        "type": meta.get("node_type", "unknown"),
                        "name": meta.get("name", ""),
                        "text": meta.get("text", "")
                    })
                    
                    if len(results) >= limit:
                        break
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching for similar nodes: {e}")
            return []
    
    def similar_documents(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find documents similar to a text query.
        
        Args:
            query: Text query
            limit: Maximum number of results
            
        Returns:
            List of similar documents with similarity scores
        """
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        results = []
        
        try:
            if FAISS_AVAILABLE:
                # Use FAISS for similarity search
                query_embedding = query_embedding.reshape(1, -1).astype('float32')
                
                # Search the index
                similarities, indices = self.index.search(query_embedding, limit * 3)
                
                # Convert to results
                for i, idx in enumerate(indices[0]):
                    if idx < 0:
                        continue
                        
                    # Find the ID for this index
                    doc_id = None
                    for id, index in self.id_to_index.items():
                        if index == idx:
                            doc_id = id
                            break
                    
                    if not doc_id:
                        continue
                        
                    # Get metadata
                    meta = self.metadata.get(doc_id, {})
                    
                    # Skip if not a document
                    if meta.get("type") != "document":
                        continue
                    
                    # Add to results
                    results.append({
                        "id": doc_id,
                        "score": float(1.0 - similarities[0][i] / 100.0),
                        "title": meta.get("title", ""),
                        "source": meta.get("source", "")
                    })
                    
                    if len(results) >= limit:
                        break
                        
            elif CHROMA_AVAILABLE:
                # Use ChromaDB
                search_results = self.doc_collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=limit
                )
                
                # Convert to results
                for i, doc_id in enumerate(search_results.get('ids', [[]])[0]):
                    meta = self.metadata.get(doc_id, {})
                    
                    results.append({
                        "id": doc_id,
                        "score": float(search_results.get('distances', [[]])[0][i]),
                        "title": meta.get("title", ""),
                        "source": meta.get("source", "")
                    })
            else:
                # Simple numpy implementation
                # Calculate cosine similarity
                similarities = np.dot(self.vectors, query_embedding) / (
                    np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(query_embedding)
                )
                
                # Get top indices
                top_indices = np.argsort(similarities)[::-1][:limit * 3]
                
                # Convert to results
                for idx in top_indices:
                    # Find the ID for this index
                    doc_id = None
                    for id, index in self.id_to_index.items():
                        if index == idx:
                            doc_id = id
                            break
                    
                    if not doc_id:
                        continue
                        
                    # Get metadata
                    meta = self.metadata.get(doc_id, {})
                    
                    # Skip if not a document
                    if meta.get("type") != "document":
                        continue
                    
                    # Add to results
                    results.append({
                        "id": doc_id,
                        "score": float(similarities[idx]),
                        "title": meta.get("title", ""),
                        "source": meta.get("source", "")
                    })
                    
                    if len(results) >= limit:
                        break
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching for similar documents: {e}")
            return []
    
    def find_related_nodes(self, node_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find nodes semantically related to a given node.
        
        Args:
            node_id: Node identifier
            limit: Maximum number of results
            
        Returns:
            List of related nodes with similarity scores
        """
        # Load node vector
        node_path = os.path.join(self.nodes_dir, f"{node_id}.npz")
        
        if not os.path.exists(node_path):
            self.logger.warning(f"Node {node_id} not found in vector store")
            return []
        
        try:
            # Load vector
            data = np.load(node_path)
            vector = data['embedding']
            
            # Perform similarity search using the node's vector
            if FAISS_AVAILABLE:
                # Use FAISS for similarity search
                vector = vector.reshape(1, -1).astype('float32')
                
                # Search the index
                similarities, indices = self.index.search(vector, limit + 1)  # +1 to account for self
                
                # Convert to results
                results = []
                for i, idx in enumerate(indices[0]):
                    if idx < 0:
                        continue
                        
                    # Find the ID for this index
                    related_id = None
                    for id, index in self.id_to_index.items():
                        if index == idx:
                            related_id = id
                            break
                    
                    if not related_id or related_id == node_id:  # Skip self
                        continue
                        
                    # Get metadata
                    meta = self.metadata.get(related_id, {})
                    
                    # Skip if not a node
                    if meta.get("type") != "node":
                        continue
                    
                    # Add to results
                    results.append({
                        "id": related_id,
                        "score": float(1.0 - similarities[0][i] / 100.0),
                        "type": meta.get("node_type", "unknown"),
                        "name": meta.get("name", ""),
                        "relation": "semantic_similarity"
                    })
                    
                    if len(results) >= limit:
                        break
                
                return results
            
            elif CHROMA_AVAILABLE:
                # Use ChromaDB
                node_meta = self.metadata.get(node_id, {})
                node_type = node_meta.get("node_type", "unknown")
                
                # Don't filter by type to get more diverse relationships
                search_results = self.node_collection.query(
                    query_embeddings=[vector.tolist()],
                    n_results=limit + 1
                )
                
                # Convert to results
                results = []
                for i, related_id in enumerate(search_results.get('ids', [[]])[0]):
                    if related_id == node_id:  # Skip self
                        continue
                        
                    meta = self.metadata.get(related_id, {})
                    
                    # Skip if not a node
                    if meta.get("type") != "node":
                        continue
                    
                    results.append({
                        "id": related_id,
                        "score": float(search_results.get('distances', [[]])[0][i]),
                        "type": meta.get("node_type", "unknown"),
                        "name": meta.get("name", ""),
                        "relation": "semantic_similarity"
                    })
                    
                    if len(results) >= limit:
                        break
                
                return results
                
            else:
                # Simple numpy implementation
                # Calculate cosine similarity
                similarities = np.dot(self.vectors, vector) / (
                    np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(vector)
                )
                
                # Get top indices
                top_indices = np.argsort(similarities)[::-1][:limit + 1]
                
                # Convert to results
                results = []
                for idx in top_indices:
                    # Find the ID for this index
                    related_id = None
                    for id, index in self.id_to_index.items():
                        if index == idx:
                            related_id = id
                            break
                    
                    if not related_id or related_id == node_id:  # Skip self
                        continue
                        
                    # Get metadata
                    meta = self.metadata.get(related_id, {})
                    
                    # Skip if not a node
                    if meta.get("type") != "node":
                        continue
                    
                    # Add to results
                    results.append({
                        "id": related_id,
                        "score": float(similarities[idx]),
                        "type": meta.get("node_type", "unknown"),
                        "name": meta.get("name", ""),
                        "relation": "semantic_similarity"
                    })
                    
                    if len(results) >= limit:
                        break
                
                return results
                
        except Exception as e:
            self.logger.error(f"Error finding related nodes: {e}")
            return []
    
    def discover_implicit_relationships(self, threshold: float = 0.85, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Discover implicit relationships between nodes based on semantic similarity.
        
        Args:
            threshold: Similarity threshold (0-1)
            limit: Maximum number of relationships to discover
            
        Returns:
            List of discovered relationships
        """
        # Cannot discover relationships without a knowledge graph
        if not self.kg or not HAS_KG:
            self.logger.warning("Knowledge graph not available, cannot discover relationships")
            return []
            
        # Get all node IDs from metadata
        node_ids = [id for id, meta in self.metadata.items() if meta.get("type") == "node"]
        self.logger.info(f"Scanning {len(node_ids)} nodes for implicit relationships")
        
        discovered = []
        
        # Track already processed pairs to avoid duplicates
        processed_pairs = set()
        
        # Limit the number to process to avoid excessive computation
        for i, node_id in enumerate(node_ids[:1000]):  # Process max 1000 nodes
            # Find related nodes
            related = self.find_related_nodes(node_id, limit=10)
            
            # Check each relationship
            for rel in related:
                related_id = rel["id"]
                score = rel["score"]
                
                # Skip if below threshold
                if score < threshold:
                    continue
                    
                # Create a consistent pair key (alphabetical order)
                pair_key = tuple(sorted([node_id, related_id]))
                
                # Skip if already processed
                if pair_key in processed_pairs:
                    continue
                    
                processed_pairs.add(pair_key)
                
                # Check if an explicit relationship exists in the knowledge graph
                explicit_relation = False
                
                # Check edges in both directions
                if self.kg.graph.has_edge(node_id, related_id) or self.kg.graph.has_edge(related_id, node_id):
                    explicit_relation = True
                
                # If no explicit relationship exists, add to discoveries
                if not explicit_relation:
                    node_data = self.metadata.get(node_id, {})
                    related_data = self.metadata.get(related_id, {})
                    
                    discovered.append({
                        "source_id": node_id,
                        "source_type": node_data.get("node_type", "unknown"),
                        "source_name": node_data.get("name", ""),
                        "target_id": related_id,
                        "target_type": related_data.get("node_type", "unknown"),
                        "target_name": related_data.get("name", ""),
                        "score": score,
                        "relation": "implicit_similarity"
                    })
                    
                    # Limit the number of discoveries
                    if len(discovered) >= limit:
                        self.logger.info(f"Discovered {len(discovered)} implicit relationships")
                        return discovered
        
        self.logger.info(f"Discovered {len(discovered)} implicit relationships")
        return discovered
    
    def cluster_nodes(self, node_type: Optional[str] = None, num_clusters: int = 10) -> Dict[str, List[str]]:
        """
        Cluster nodes based on semantic similarity.
        
        Args:
            node_type: Optional filter by node type
            num_clusters: Number of clusters to create
            
        Returns:
            Dictionary of cluster_id -> list of node_ids
        """
        if not FAISS_AVAILABLE:
            self.logger.warning("FAISS not available for clustering")
            return {}
            
        try:
            # Get all node vectors and IDs
            node_vectors = []
            node_ids = []
            
            for node_id, meta in self.metadata.items():
                # Filter by type if requested
                if meta.get("type") != "node":
                    continue
                    
                if node_type and meta.get("node_type") != node_type:
                    continue
                    
                # Load node vector
                node_path = os.path.join(self.nodes_dir, f"{node_id}.npz")
                if not os.path.exists(node_path):
                    continue
                    
                data = np.load(node_path)
                node_vectors.append(data['embedding'])
                node_ids.append(node_id)
            
            if not node_vectors:
                self.logger.warning(f"No nodes found for clustering (type={node_type})")
                return {}
                
            # Convert to numpy array
            vectors = np.vstack(node_vectors).astype('float32')
            
            # Create K-means clustering
            kmeans = faiss.Kmeans(d=vectors.shape[1], k=num_clusters, niter=20, verbose=False)
            kmeans.train(vectors)
            
            # Get cluster assignments
            _, assignments = kmeans.index.search(vectors, 1)
            
            # Group by cluster
            clusters = {}
            for i, cluster_id in enumerate(assignments.flatten()):
                if cluster_id not in clusters:
                    clusters[int(cluster_id)] = []
                clusters[int(cluster_id)].append(node_ids[i])
            
            self.logger.info(f"Created {len(clusters)} clusters from {len(node_ids)} nodes")
            return clusters
            
        except Exception as e:
            self.logger.error(f"Error clustering nodes: {e}")
            return {}
    
    def add_kg_nodes(self, kg=None) -> int:
        """
        Add all nodes from a knowledge graph to the vector store.
        
        Args:
            kg: Knowledge graph instance (uses self.kg if None)
            
        Returns:
            Number of nodes added
        """
        # Use provided KG or instance KG
        kg_instance = kg or self.kg
        
        if not kg_instance:
            self.logger.warning("No knowledge graph available")
            return 0
            
        nodes_added = 0
        
        # Process all nodes in the graph
        for node_id, data in kg_instance.graph.nodes(data=True):
            # Skip if already in vector store
            if node_id in self.metadata:
                continue
                
            # Add to vector store
            if self.add_node(node_id, data):
                nodes_added += 1
        
        self.logger.info(f"Added {nodes_added} nodes from knowledge graph to vector store")
        return nodes_added
    
    def sync_with_kg(self, kg=None) -> Dict[str, int]:
        """
        Synchronize vector store with knowledge graph.
        
        Args:
            kg: Knowledge graph instance (uses self.kg if None)
            
        Returns:
            Statistics about the synchronization
        """
        # Use provided KG or instance KG
        kg_instance = kg or self.kg
        
        if not kg_instance:
            self.logger.warning("No knowledge graph available")
            return {"nodes_added": 0, "nodes_total": 0}
            
        # Add all nodes
        nodes_added = self.add_kg_nodes(kg_instance)
        
        # Return statistics
        return {
            "nodes_added": nodes_added,
            "nodes_total": len(kg_instance.graph.nodes)
        }
    
    def save(self):
        """Save the vector store state to disk."""
        try:
            # Save FAISS index if available
            if FAISS_AVAILABLE and hasattr(self, 'index'):
                faiss.write_index(self.index, self.index_path)
                
            # Save metadata
            self._save_metadata()
            
            self.logger.info("Vector store saved to disk")
        except Exception as e:
            self.logger.error(f"Error saving vector store: {e}")
            
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with statistics
        """
        # Count node types
        node_types = {}
        
        for node_id, meta in self.metadata.items():
            if meta.get("type") == "node":
                node_type = meta.get("node_type", "unknown")
                node_types[node_type] = node_types.get(node_type, 0) + 1
        
        # Count documents
        document_count = sum(1 for meta in self.metadata.values() if meta.get("type") == "document")
        
        return {
            "total_vectors": len(self.metadata),
            "node_count": len(self.metadata) - document_count,
            "document_count": document_count,
            "node_types": node_types,
            "embedding_provider": self.embedding_provider,
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "timestamp": datetime.now().isoformat()
        }
            
