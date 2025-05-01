"""
Night_watcher Content Repository
Provides immutable storage for raw intelligence content with enhanced metadata tracking.
"""

import os
import json
import uuid
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Union

# Configure logging
logger = logging.getLogger(__name__)

class ContentRepository:
    """Repository for raw intelligence content with immutable storage and enhanced metadata"""
    
    def __init__(self, base_dir: str = "data/raw", metadata_file: str = "repository_metadata.json"):
        """
        Initialize content repository with storage directory
        
        Args:
            base_dir: Base directory for content storage
            metadata_file: Filename for the repository metadata
        """
        self.base_dir = base_dir
        self.metadata_file = os.path.join(base_dir, metadata_file)
        self.metadata_store = {}  # Maps content_id to metadata
        self.content_by_hash = {}  # Maps content_hash to content_id for deduplication
        self.source_mapping = {}   # Maps source names to content_ids
        self._ensure_directories()
        self._load_metadata()
        
    def _ensure_directories(self):
        """Ensure storage directories exist"""
        os.makedirs(self.base_dir, exist_ok=True)
        
    def _load_metadata(self):
        """Load existing metadata if available"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                    self.metadata_store = loaded_data.get("items", {})
                    self.content_by_hash = loaded_data.get("content_by_hash", {})
                    self.source_mapping = loaded_data.get("source_mapping", {})
                    
                logger.info(f"Loaded {len(self.metadata_store)} items from repository metadata")
            except Exception as e:
                logger.error(f"Error loading metadata: {str(e)}")
                self.metadata_store = {}
                self.content_by_hash = {}
                self.source_mapping = {}
        
    def store_content(self, content: Dict[str, Any], source_type: str) -> str:
        """
        Store raw content in immutable repository with unique ID
        
        Args:
            content: Raw content dictionary
            source_type: Type of source (e.g., 'rss', 'api', 'document')
            
        Returns:
            Unique content ID
        """
        # Generate content hash for deduplication
        content_hash = self._hash_content(content)
        
        # Check for duplicate content
        if content_hash in self.content_by_hash:
            existing_id = self.content_by_hash[content_hash]
            logger.info(f"Duplicate content detected. Using existing ID: {existing_id}")
            return existing_id
        
        # Generate unique ID
        source_name = content.get("source", "unknown").lower().replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        content_id = f"{source_type}_{source_name}_{timestamp}_{uuid.uuid4().hex[:8]}"
        
        # Create metadata
        timestamp_iso = datetime.now().isoformat()
        
        metadata = {
            "content_id": content_id,
            "source_type": source_type,
            "source_url": content.get("url", ""),
            "source_name": content.get("source", "Unknown"),
            "title": content.get("title", "Untitled"),
            "collection_timestamp": timestamp_iso,
            "content_hash": content_hash,
            "tags": content.get("tags", []),
            "bias_label": content.get("bias_label", "unknown"),
            "publication_date": content.get("published", ""),
            "version": "1.0.0"
        }
        
        # Store raw content as immutable file
        content_path = os.path.join(self.base_dir, f"{content_id}.json")
        with open(content_path, 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=2, ensure_ascii=False)
            
        # Update metadata store and mappings
        self.metadata_store[content_id] = metadata
        self.content_by_hash[content_hash] = content_id
        
        # Update source mapping
        source_name = metadata["source_name"]
        if source_name not in self.source_mapping:
            self.source_mapping[source_name] = []
        self.source_mapping[source_name].append(content_id)
        
        # Store metadata file
        self._save_metadata()
            
        logger.info(f"Stored content: {content_id} - {metadata['title']}")
        return content_id
        
    def _save_metadata(self):
        """Save metadata to disk"""
        try:
            # Create a structured metadata object
            metadata_obj = {
                "items": self.metadata_store,
                "content_by_hash": self.content_by_hash,
                "source_mapping": self.source_mapping,
                "last_updated": datetime.now().isoformat(),
                "item_count": len(self.metadata_store)
            }
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata_obj, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Saved repository metadata with {len(self.metadata_store)} items")
            return True
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")
            return False
        
    def _hash_content(self, content: Dict[str, Any]) -> str:
        """
        Create hash of content for integrity and deduplication
        
        Args:
            content: Content to hash
            
        Returns:
            Content hash
        """
        # Extract key fields for hashing to avoid timestamp differences
        hash_fields = {
            "title": content.get("title", ""),
            "content": content.get("content", ""),
            "url": content.get("url", ""),
            "source": content.get("source", "")
        }
        
        # Create deterministic representation
        content_str = json.dumps(hash_fields, sort_keys=True)
        
        # Generate hash
        return hashlib.sha256(content_str.encode('utf-8')).hexdigest()
        
    def get_content(self, content_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve raw content by ID
        
        Args:
            content_id: ID of content to retrieve
            
        Returns:
            Content data or None if not found
        """
        content_path = os.path.join(self.base_dir, f"{content_id}.json")
        if not os.path.exists(content_path):
            logger.warning(f"Content not found: {content_id}")
            return None
            
        try:
            with open(content_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading content {content_id}: {str(e)}")
            return None
            
    def get_metadata(self, content_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for content
        
        Args:
            content_id: ID of content to get metadata for
            
        Returns:
            Content metadata or None if not found
        """
        return self.metadata_store.get(content_id)
        
    def list_content(self, 
                   source_type: Optional[str] = None, 
                   source_name: Optional[str] = None,
                   start_date: Optional[Union[str, datetime]] = None,
                   end_date: Optional[Union[str, datetime]] = None,
                   tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        List available content with enhanced filtering
        
        Args:
            source_type: Optional filter for source type
            source_name: Optional filter for source name
            start_date: Optional filter for earliest collection date
            end_date: Optional filter for latest collection date
            tags: Optional filter for content tags
            
        Returns:
            List of content metadata matching filters
        """
        # Convert dates to ISO string format if they are datetime objects
        if isinstance(start_date, datetime):
            start_date = start_date.isoformat()
        if isinstance(end_date, datetime):
            end_date = end_date.isoformat()
        
        # Start with all metadata
        results = list(self.metadata_store.values())
        
        # Apply filters
        if source_type:
            results = [item for item in results if item.get("source_type") == source_type]
            
        if source_name:
            results = [item for item in results if item.get("source_name") == source_name]
            
        if start_date:
            results = [item for item in results if item.get("collection_timestamp", "") >= start_date]
            
        if end_date:
            results = [item for item in results if item.get("collection_timestamp", "") <= end_date]
            
        if tags:
            results = [item for item in results if set(tags).issubset(set(item.get("tags", [])))]
        
        # Sort by timestamp (newest first)
        results.sort(key=lambda x: x.get("collection_timestamp", ""), reverse=True)
        
        return results
    
    def add_tag(self, content_id: str, tag: str) -> bool:
        """
        Add a tag to content
        
        Args:
            content_id: ID of content to tag
            tag: Tag to add
            
        Returns:
            True if tag was added, False otherwise
        """
        if content_id not in self.metadata_store:
            logger.warning(f"Content not found: {content_id}")
            return False
            
        # Get current tags
        metadata = self.metadata_store[content_id]
        if "tags" not in metadata:
            metadata["tags"] = []
            
        # Add tag if not already present
        if tag not in metadata["tags"]:
            metadata["tags"].append(tag)
            self._save_metadata()
            logger.info(f"Added tag '{tag}' to {content_id}")
            return True
            
        return False
    
    def remove_tag(self, content_id: str, tag: str) -> bool:
        """
        Remove a tag from content
        
        Args:
            content_id: ID of content to untag
            tag: Tag to remove
            
        Returns:
            True if tag was removed, False otherwise
        """
        if content_id not in self.metadata_store:
            logger.warning(f"Content not found: {content_id}")
            return False
            
        # Get current tags
        metadata = self.metadata_store[content_id]
        if "tags" not in metadata or tag not in metadata["tags"]:
            return False
            
        # Remove tag
        metadata["tags"].remove(tag)
        self._save_metadata()
        logger.info(f"Removed tag '{tag}' from {content_id}")
        return True
    
    def get_content_by_source(self, source_name: str) -> List[Dict[str, Any]]:
        """
        Get all content from a specific source
        
        Args:
            source_name: Name of source to filter by
            
        Returns:
            List of content metadata from the specified source
        """
        content_ids = self.source_mapping.get(source_name, [])
        return [self.metadata_store[content_id] for content_id in content_ids 
                if content_id in self.metadata_store]
    
    def search_content(self, query: str) -> List[Dict[str, Any]]:
        """
        Search content by query string (basic implementation)
        
        Args:
            query: Query string to search for
            
        Returns:
            List of content metadata matching the query
        """
        query = query.lower()
        results = []
        
        for content_id, metadata in self.metadata_store.items():
            # Check if query appears in title or source
            if (query in metadata.get("title", "").lower() or
                query in metadata.get("source_name", "").lower()):
                results.append(metadata)
                continue
                
            # Check content if needed
            content = self.get_content(content_id)
            if content and query in content.get("content", "").lower():
                results.append(metadata)
                
        # Sort by timestamp (newest first)
        results.sort(key=lambda x: x.get("collection_timestamp", ""), reverse=True)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get repository statistics
        
        Returns:
            Dictionary with repository statistics
        """
        sources = {}
        source_types = {}
        dates = {}
        
        # Count by source and type
        for metadata in self.metadata_store.values():
            source = metadata.get("source_name", "Unknown")
            source_type = metadata.get("source_type", "Unknown")
            
            # Get date (just the day part of the timestamp)
            timestamp = metadata.get("collection_timestamp", "")
            date = timestamp.split("T")[0] if timestamp and "T" in timestamp else "Unknown"
            
            # Update counts
            sources[source] = sources.get(source, 0) + 1
            source_types[source_type] = source_types.get(source_type, 0) + 1
            dates[date] = dates.get(date, 0) + 1
        
        return {
            "total_items": len(self.metadata_store),
            "sources": sources,
            "source_types": source_types,
            "by_date": dates,
            "timestamp": datetime.now().isoformat()
        }
