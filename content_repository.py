"""
Night_watcher Content Repository
Provides immutable storage for raw intelligence content.
"""

import os
import json
import uuid
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional


class ContentRepository:
    """Repository for raw intelligence content with immutable storage"""
    
    def __init__(self, base_dir: str = "data/raw"):
        """Initialize content repository with storage directory"""
        self.base_dir = base_dir
        self.metadata_store = {}  # Maps content_id to metadata
        self._ensure_directories()
        self._load_metadata()
        
    def _ensure_directories(self):
        """Ensure storage directories exist"""
        os.makedirs(self.base_dir, exist_ok=True)
        
    def _load_metadata(self):
        """Load existing metadata if available"""
        metadata_path = os.path.join(self.base_dir, "metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata_store = json.load(f)
            except Exception as e:
                print(f"Error loading metadata: {str(e)}")
                self.metadata_store = {}
        
    def store_content(self, content: Dict[str, Any], source_type: str) -> str:
        """
        Store raw content in immutable repository with unique ID
        
        Args:
            content: Raw content dictionary
            source_type: Type of source (e.g., 'rss', 'api', 'document')
            
        Returns:
            Unique content ID
        """
        # Generate unique ID
        content_id = f"{source_type}_{uuid.uuid4().hex}"
        
        # Create metadata
        timestamp = datetime.now().isoformat()
        content_hash = self._hash_content(content)
        
        # Check for duplicate content
        for existing_id, metadata in self.metadata_store.items():
            if metadata.get("content_hash") == content_hash:
                return existing_id  # Return existing ID for duplicate content
        
        metadata = {
            "content_id": content_id,
            "source_type": source_type,
            "source_url": content.get("url", ""),
            "source_name": content.get("source", "Unknown"),
            "title": content.get("title", "Untitled"),
            "collection_timestamp": timestamp,
            "content_hash": content_hash,
            "version": "1.0.0"
        }
        
        # Store raw content as immutable file
        content_path = os.path.join(self.base_dir, f"{content_id}.json")
        with open(content_path, 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=2)
            
        # Update metadata store
        self.metadata_store[content_id] = metadata
        
        # Store metadata file
        self._save_metadata()
            
        return content_id
        
    def _save_metadata(self):
        """Save metadata to disk"""
        metadata_path = os.path.join(self.base_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata_store, f, indent=2)
        
    def _hash_content(self, content: Dict[str, Any]) -> str:
        """Create hash of content for integrity and deduplication"""
        # Create deterministic representation
        content_str = json.dumps(content, sort_keys=True)
        # Generate hash
        return hashlib.sha256(content_str.encode()).hexdigest()
        
    def get_content(self, content_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve raw content by ID"""
        content_path = os.path.join(self.base_dir, f"{content_id}.json")
        if not os.path.exists(content_path):
            return None
            
        with open(content_path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def get_metadata(self, content_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for content"""
        return self.metadata_store.get(content_id)
        
    def list_content(self, source_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available content, optionally filtered by source type
        
        Args:
            source_type: Optional filter for source type
            
        Returns:
            List of content metadata
        """
        if source_type:
            return [metadata for metadata in self.metadata_store.values() 
                    if metadata.get("source_type") == source_type]
        else:
            return list(self.metadata_store.values())
