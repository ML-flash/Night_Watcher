"""
Night_watcher Advanced Memory Store
Enhanced vector-based storage with sophisticated retrieval capabilities.
"""

import os
import json
import pickle
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Set

# Optional imports for enhanced functionality
try:
    import qdrant_client
    from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, Range

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

logger = logging.getLogger(__name__)


class MemoryCollection:
    """Defines a collection type in the memory system"""

    def __init__(self, name: str, vector_size: int, description: str = ""):
        """Initialize collection metadata"""
        self.name = name
        self.vector_size = vector_size
        self.description = description
        self.created_at = datetime.now().isoformat()
        self.count = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "vector_size": self.vector_size,
            "description": self.description,
            "created_at": self.created_at,
            "count": self.count
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryCollection':
        """Create from dictionary"""
        collection = cls(
            name=data["name"],
            vector_size=data["vector_size"],
            description=data.get("description", "")
        )
        collection.created_at = data.get("created_at", datetime.now().isoformat())
        collection.count = data.get("count", 0)
        return collection


class BaseAdvancedStore:
    """Base class for advanced memory stores"""

    def __init__(self, embedding_provider):
        """Initialize with embedding provider"""
        self.embedding_provider = embedding_provider
        self.collections = {}
        self.metadata = {}

    def create_collection(self, name: str, vector_size: int = None, description: str = "") -> bool:
        """Create a new collection in the store"""
        raise NotImplementedError("Subclasses must implement create_collection")

    def add_item(self, collection: str, item_id: str, text: str, metadata: Dict[str, Any]) -> bool:
        """Add an item to a collection"""
        raise NotImplementedError("Subclasses must implement add_item")

    def search(self, collection: str, query: str, limit: int = 5,
               filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for items in a collection"""
        raise NotImplementedError("Subclasses must implement search")

    def get_item(self, collection: str, item_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific item by ID"""
        raise NotImplementedError("Subclasses must implement get_item")

    def update_item(self, collection: str, item_id: str, text: str, metadata: Dict[str, Any]) -> bool:
        """Update an existing item"""
        raise NotImplementedError("Subclasses must implement update_item")

    def delete_item(self, collection: str, item_id: str) -> bool:
        """Delete an item from a collection"""
        raise NotImplementedError("Subclasses must implement delete_item")

    def save(self, path: str) -> bool:
        """Save the memory store to disk"""
        raise NotImplementedError("Subclasses must implement save")

    def load(self, path: str) -> bool:
        """Load the memory store from disk"""
        raise NotImplementedError("Subclasses must implement load")

    def get_collections(self) -> List[str]:
        """Get list of all collections"""
        return list(self.collections.keys())

    def get_collection_info(self, collection: str) -> Optional[Dict[str, Any]]:
        """Get information about a collection"""
        if collection not in self.collections:
            return None
        return self.collections[collection].to_dict()

    def _validate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure metadata contains required fields and valid types"""
        if metadata is None:
            metadata = {}

        # Ensure timestamp exists
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.now().isoformat()

        # Convert non-serializable types
        for key, value in list(metadata.items()):
            # Handle sets
            if isinstance(value, set):
                metadata[key] = list(value)

            # Handle datetime
            elif isinstance(value, datetime):
                metadata[key] = value.isoformat()

        return metadata


class EnhancedSimpleStore(BaseAdvancedStore):
    """Enhanced in-memory vector store implementation when Qdrant is not available"""

    def __init__(self, embedding_provider):
        """Initialize with embedding provider"""
        super().__init__(embedding_provider)
        # Dictionary of collection name -> items
        self.items = {}  # Dict[collection_name, Dict[item_id, item_data]]
        self.embeddings = {}  # Dict[collection_name, Dict[item_id, embedding]]

    def create_collection(self, name: str, vector_size: int = None, description: str = "") -> bool:
        """Create a new collection in the store"""
        if name in self.collections:
            logger.warning(f"Collection {name} already exists")
            return False

        # If vector_size not provided, use the embedding provider's output size
        if vector_size is None:
            # Create a dummy embedding to determine size
            dummy_embedding = self.embedding_provider.embed_text("test")
            vector_size = len(dummy_embedding)

        # Create the collection
        self.collections[name] = MemoryCollection(name, vector_size, description)
        self.items[name] = {}
        self.embeddings[name] = {}

        logger.info(f"Created collection: {name} with vector size {vector_size}")
        return True

    def add_item(self, collection: str, item_id: str, text: str, metadata: Dict[str, Any]) -> bool:
        """Add an item to a collection"""
        # Check if collection exists
        if collection not in self.collections:
            logger.warning(f"Collection {collection} not found")
            return False

        try:
            # Create embedding
            embedding = self.embedding_provider.embed_text(text)

            # Validate metadata
            metadata = self._validate_metadata(metadata)

            # Store the item
            self.items[collection][item_id] = {
                "text": text,
                "metadata": metadata,
                "timestamp": datetime.now().isoformat()
            }

            self.embeddings[collection][item_id] = embedding

            # Update collection count
            self.collections[collection].count += 1

            return True
        except Exception as e:
            logger.error(f"Error adding item to {collection}: {str(e)}")
            return False

    def search(self, collection: str, query: str, limit: int = 5,
               filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for items in a collection"""
        # Check if collection exists
        if collection not in self.collections:
            logger.warning(f"Collection {collection} not found")
            return []

        try:
            # Create query embedding
            query_embedding = self.embedding_provider.embed_text(query)

            # Calculate similarities
            similarities = []
            for item_id, embedding in self.embeddings[collection].items():
                # Check filters first
                if filters and not self._check_filters(self.items[collection][item_id], filters):
                    continue

                # Calculate similarity
                similarity = np.dot(query_embedding, embedding)
                similarities.append((item_id, similarity))

            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Get top results
            results = []
            for item_id, similarity in similarities[:limit]:
                item = self.items[collection][item_id].copy()
                item["id"] = item_id
                item["similarity"] = float(similarity)
                results.append(item)

            return results
        except Exception as e:
            logger.error(f"Error searching in {collection}: {str(e)}")
            return []

    def _check_filters(self, item: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if an item matches the specified filters"""
        metadata = item.get("metadata", {})

        for key, condition in filters.items():
            if key not in metadata:
                return False

            if isinstance(condition, dict):
                # Range filter
                if "gte" in condition and metadata[key] < condition["gte"]:
                    return False
                if "gt" in condition and metadata[key] <= condition["gt"]:
                    return False
                if "lte" in condition and metadata[key] > condition["lte"]:
                    return False
                if "lt" in condition and metadata[key] >= condition["lt"]:
                    return False
            elif isinstance(condition, list):
                # Contains any
                if not isinstance(metadata[key], list):
                    if metadata[key] not in condition:
                        return False
                else:
                    if not any(x in condition for x in metadata[key]):
                        return False
            else:
                # Exact match
                if metadata[key] != condition:
                    return False

        return True

    def get_item(self, collection: str, item_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific item by ID"""
        # Check if collection exists
        if collection not in self.collections:
            logger.warning(f"Collection {collection} not found")
            return None

        # Check if item exists
        if item_id not in self.items[collection]:
            return None

        item = self.items[collection][item_id].copy()
        item["id"] = item_id
        return item

    def update_item(self, collection: str, item_id: str, text: str, metadata: Dict[str, Any]) -> bool:
        """Update an existing item"""
        # Check if collection exists
        if collection not in self.collections:
            logger.warning(f"Collection {collection} not found")
            return False

        # Check if item exists
        if item_id not in self.items[collection]:
            logger.warning(f"Item {item_id} not found in {collection}")
            return False

        try:
            # Update embedding if text changed
            if text != self.items[collection][item_id]["text"]:
                embedding = self.embedding_provider.embed_text(text)
                self.embeddings[collection][item_id] = embedding

            # Validate and update metadata
            metadata = self._validate_metadata(metadata)

            # Update item
            self.items[collection][item_id]["text"] = text
            self.items[collection][item_id]["metadata"] = metadata
            self.items[collection][item_id]["timestamp"] = datetime.now().isoformat()

            return True
        except Exception as e:
            logger.error(f"Error updating item in {collection}: {str(e)}")
            return False

    def delete_item(self, collection: str, item_id: str) -> bool:
        """Delete an item from a collection"""
        # Check if collection exists
        if collection not in self.collections:
            logger.warning(f"Collection {collection} not found")
            return False

        # Check if item exists
        if item_id not in self.items[collection]:
            logger.warning(f"Item {item_id} not found in {collection}")
            return False

        try:
            # Delete item and embedding
            del self.items[collection][item_id]
            del self.embeddings[collection][item_id]

            # Update collection count
            self.collections[collection].count -= 1

            return True
        except Exception as e:
            logger.error(f"Error deleting item from {collection}: {str(e)}")
            return False

    def save(self, path: str) -> bool:
        """Save the memory store to disk"""
        try:
            base_dir = os.path.dirname(path)
            os.makedirs(base_dir, exist_ok=True)

            # Convert collections to dict for serialization
            collections_dict = {name: coll.to_dict() for name, coll in self.collections.items()}

            data = {
                "collections": collections_dict,
                "items": self.items,
                "embeddings": self.embeddings,
                "metadata": {
                    "version": "1.0",
                    "timestamp": datetime.now().isoformat(),
                    "total_items": sum(len(items) for items in self.items.values())
                }
            }

            with open(path, 'wb') as f:
                pickle.dump(data, f)

            logger.info(f"Saved memory store to {path}")
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
                data = pickle.load(f)

            # Convert dict back to collections
            self.collections = {
                name: MemoryCollection.from_dict(coll_data)
                for name, coll_data in data["collections"].items()
            }

            self.items = data["items"]
            self.embeddings = data["embeddings"]
            self.metadata = data.get("metadata", {})

            logger.info(f"Loaded memory store from {path} with {self.metadata.get('total_items', 0)} items")
            return True
        except Exception as e:
            logger.error(f"Error loading memory store: {str(e)}")
            return False


class QdrantMemoryStore(BaseAdvancedStore):
    """Advanced memory store using Qdrant vector database when available"""

    def __init__(self, embedding_provider, host: str = ":memory:", port: int = 6333):
        """Initialize with embedding provider and Qdrant connection details"""
        super().__init__(embedding_provider)

        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant client not available. Install with: pip install qdrant-client")

        # Connect to Qdrant
        if host == ":memory:":
            self.client = qdrant_client.QdrantClient(":memory:")
            logger.info("Connected to in-memory Qdrant instance")
        else:
            self.client = qdrant_client.QdrantClient(host=host, port=port)
            logger.info(f"Connected to Qdrant at {host}:{port}")

        # Load existing collections
        self._load_collections()

    def _load_collections(self):
        """Load existing collections from Qdrant"""
        try:
            collections = self.client.get_collections().collections
            for collection in collections:
                collection_info = self.client.get_collection(collection.name)
                vector_size = collection_info.config.params.vectors.size
                self.collections[collection.name] = MemoryCollection(
                    name=collection.name,
                    vector_size=vector_size
                )

            logger.info(f"Loaded {len(collections)} existing collections from Qdrant")
        except Exception as e:
            logger.error(f"Error loading collections: {str(e)}")

    def create_collection(self, name: str, vector_size: int = None, description: str = "") -> bool:
        """Create a new collection in Qdrant"""
        if name in self.collections:
            logger.warning(f"Collection {name} already exists")
            return False

        # If vector_size not provided, use the embedding provider's output size
        if vector_size is None:
            # Create a dummy embedding to determine size
            dummy_embedding = self.embedding_provider.embed_text("test")
            vector_size = len(dummy_embedding)

        try:
            # Create collection in Qdrant
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )

            # Create collection metadata
            self.collections[name] = MemoryCollection(name, vector_size, description)

            # Add collection info to metadata
            metadata_payload = {
                "description": description,
                "created_at": datetime.now().isoformat()
            }

            # Store collection metadata
            try:
                self.client.set_payload(
                    collection_name=name,
                    payload=metadata_payload,
                    points=[PointStruct(id="collection_info", payload={})]
                )
            except:
                # If collection_info point doesn't exist, create it
                self.client.upsert(
                    collection_name=name,
                    points=[PointStruct(
                        id="collection_info",
                        vector=[0.0] * vector_size,
                        payload=metadata_payload
                    )]
                )

            logger.info(f"Created collection: {name} with vector size {vector_size}")
            return True
        except Exception as e:
            logger.error(f"Error creating collection {name}: {str(e)}")
            return False

    def add_item(self, collection: str, item_id: str, text: str, metadata: Dict[str, Any]) -> bool:
        """Add an item to a collection in Qdrant"""
        # Check if collection exists
        if collection not in self.collections:
            logger.warning(f"Collection {collection} not found")
            return False

        try:
            # Create embedding
            embedding = self.embedding_provider.embed_text(text)

            # Validate metadata
            metadata = self._validate_metadata(metadata)

            # Prepare payload
            payload = {
                "text": text,
                "metadata": metadata,
                "timestamp": datetime.now().isoformat()
            }

            # Store in Qdrant
            self.client.upsert(
                collection_name=collection,
                points=[PointStruct(
                    id=item_id,
                    vector=embedding.tolist(),
                    payload=payload
                )]
            )

            # Update collection count
            self.collections[collection].count += 1

            return True
        except Exception as e:
            logger.error(f"Error adding item to {collection}: {str(e)}")
            return False

    def search(self, collection: str, query: str, limit: int = 5,
               filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for items in a collection using Qdrant"""
        # Check if collection exists
        if collection not in self.collections:
            logger.warning(f"Collection {collection} not found")
            return []

        try:
            # Create query embedding
            query_embedding = self.embedding_provider.embed_text(query)

            # Convert filters to Qdrant format
            qdrant_filter = None
            if filters:
                qdrant_filter = self._convert_filters(filters)

            # Search in Qdrant
            search_result = self.client.search(
                collection_name=collection,
                query_vector=query_embedding.tolist(),
                limit=limit,
                query_filter=qdrant_filter
            )

            # Format results
            results = []
            for scored_point in search_result:
                item = scored_point.payload
                item["id"] = scored_point.id
                item["similarity"] = float(scored_point.score)
                results.append(item)

            return results
        except Exception as e:
            logger.error(f"Error searching in {collection}: {str(e)}")
            return []

    def _convert_filters(self, filters: Dict[str, Any]) -> Filter:
        """Convert dictionary filters to Qdrant filter format"""
        conditions = []

        for key, condition in filters.items():
            field_path = f"metadata.{key}"

            if isinstance(condition, dict):
                # Range filter
                range_params = {}
                if "gte" in condition:
                    range_params["gte"] = condition["gte"]
                if "gt" in condition:
                    range_params["gt"] = condition["gt"]
                if "lte" in condition:
                    range_params["lte"] = condition["lte"]
                if "lt" in condition:
                    range_params["lt"] = condition["lt"]

                conditions.append(FieldCondition(
                    key=field_path,
                    range=Range(**range_params)
                ))
            elif isinstance(condition, list):
                # Match any
                conditions.append(FieldCondition(
                    key=field_path,
                    match={"any": condition}
                ))
            else:
                # Exact match
                conditions.append(FieldCondition(
                    key=field_path,
                    match={"value": condition}
                ))

        return Filter(must=conditions) if conditions else None

    def get_item(self, collection: str, item_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific item by ID from Qdrant"""
        # Check if collection exists
        if collection not in self.collections:
            logger.warning(f"Collection {collection} not found")
            return None

        try:
            # Get points from Qdrant
            points = self.client.retrieve(
                collection_name=collection,
                ids=[item_id]
            )

            if not points:
                return None

            # Format result
            point = points[0]
            item = point.payload
            item["id"] = point.id

            return item
        except Exception as e:
            logger.error(f"Error getting item from {collection}: {str(e)}")
            return None

    def update_item(self, collection: str, item_id: str, text: str, metadata: Dict[str, Any]) -> bool:
        """Update an existing item in Qdrant"""
        # Check if collection exists
        if collection not in self.collections:
            logger.warning(f"Collection {collection} not found")
            return False

        try:
            # Get existing item
            existing = self.get_item(collection, item_id)
            if not existing:
                logger.warning(f"Item {item_id} not found in {collection}")
                return False

            # Create embedding
            embedding = self.embedding_provider.embed_text(text)

            # Validate metadata
            metadata = self._validate_metadata(metadata)

            # Prepare payload
            payload = {
                "text": text,
                "metadata": metadata,
                "timestamp": datetime.now().isoformat()
            }

            # Update in Qdrant
            self.client.upsert(
                collection_name=collection,
                points=[PointStruct(
                    id=item_id,
                    vector=embedding.tolist(),
                    payload=payload
                )]
            )

            return True
        except Exception as e:
            logger.error(f"Error updating item in {collection}: {str(e)}")
            return False

    def delete_item(self, collection: str, item_id: str) -> bool:
        """Delete an item from a collection in Qdrant"""
        # Check if collection exists
        if collection not in self.collections:
            logger.warning(f"Collection {collection} not found")
            return False

        try:
            # Delete from Qdrant
            self.client.delete(
                collection_name=collection,
                points_selector=[item_id]
            )

            # Update collection count
            self.collections[collection].count -= 1

            return True
        except Exception as e:
            logger.error(f"Error deleting item from {collection}: {str(e)}")
            return False

    def save(self, path: str) -> bool:
        """Save the memory store metadata to disk (actual data is in Qdrant)"""
        try:
            base_dir = os.path.dirname(path)
            os.makedirs(base_dir, exist_ok=True)

            # We only need to save collections metadata as the actual data is in Qdrant
            collections_dict = {name: coll.to_dict() for name, coll in self.collections.items()}

            data = {
                "collections": collections_dict,
                "metadata": {
                    "version": "1.0",
                    "timestamp": datetime.now().isoformat(),
                    "store_type": "qdrant"
                }
            }

            with open(path, 'wb') as f:
                pickle.dump(data, f)

            logger.info(f"Saved memory store metadata to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving memory store metadata: {str(e)}")
            return False

    def load(self, path: str) -> bool:
        """Load the memory store metadata from disk (actual data is in Qdrant)"""
        if not os.path.exists(path):
            logger.warning(f"Memory store metadata file not found: {path}")
            return False

        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)

            # Check if this is a Qdrant store
            store_type = data.get("metadata", {}).get("store_type")
            if store_type != "qdrant":
                logger.error(f"Incompatible store type: {store_type}, expected: qdrant")
                return False

            # Load collections metadata
            self.collections = {
                name: MemoryCollection.from_dict(coll_data)
                for name, coll_data in data["collections"].items()
            }

            self.metadata = data.get("metadata", {})

            logger.info(f"Loaded memory store metadata from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading memory store metadata: {str(e)}")
            return False


def create_memory_store(embedding_provider, store_type: str = "simple",
                        host: str = ":memory:", port: int = 6333) -> BaseAdvancedStore:
    """Factory function to create the appropriate memory store"""
    if store_type == "qdrant" and QDRANT_AVAILABLE:
        try:
            return QdrantMemoryStore(embedding_provider, host, port)
        except Exception as e:
            logger.warning(f"Failed to create Qdrant store: {str(e)}. Falling back to enhanced simple store.")
            return EnhancedSimpleStore(embedding_provider)
    else:
        return EnhancedSimpleStore(embedding_provider)