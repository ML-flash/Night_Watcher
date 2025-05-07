"""
Night_watcher Document Repository
Provides immutable document storage with cryptographic provenance.
"""

import os
import json
import hashlib
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Set, Tuple

# Configure logging
logger = logging.getLogger(__name__)

class DocumentRepository:
    """
    Repository for storing immutable documents with cryptographic provenance.
    Ensures all intelligence artifacts can be traced back to original sources.
    """

    def __init__(self, base_dir: str = "data/documents"):
        """
        Initialize the document repository.
        
        Args:
            base_dir: Base directory for document storage
        """
        self.base_dir = base_dir
        self.content_dir = os.path.join(base_dir, "content")
        self.metadata_dir = os.path.join(base_dir, "metadata")
        
        # Create directories if they don't exist
        os.makedirs(self.content_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        # Cache for document IDs
        self._document_id_cache: Dict[str, str] = {}
        
        self.logger = logging.getLogger("DocumentRepository")
        self.logger.info(f"Document repository initialized at {base_dir}")

    def store_document(self, content: str, metadata: Dict[str, Any]) -> str:
        """
        Store a document with its metadata and return its document ID.
        The document ID is a cryptographic hash of the content, ensuring immutability.
        
        Args:
            content: Document content text
            metadata: Document metadata (source, URL, collection time, etc.)
            
        Returns:
            Document ID (hash)
        """
        # Generate content hash (document ID)
        doc_id = self._generate_document_id(content)
        
        # Add document ID to metadata
        metadata = metadata.copy()  # Don't modify the original
        metadata["document_id"] = doc_id
        metadata["stored_at"] = datetime.now().isoformat()
        
        # Check if document already exists
        content_path = os.path.join(self.content_dir, f"{doc_id}.txt")
        metadata_path = os.path.join(self.metadata_dir, f"{doc_id}.json")
        
        if os.path.exists(content_path):
            self.logger.info(f"Document {doc_id} already exists, not overwriting content")
        else:
            # Store content
            with open(content_path, "w", encoding="utf-8") as f:
                f.write(content)
            self.logger.info(f"Stored document content for {doc_id}")
                
        # Always update metadata (may contain new information)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Stored document metadata for {doc_id}")
        return doc_id

    def get_document(self, doc_id: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Retrieve a document and its metadata by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Tuple of (content, metadata) or (None, None) if not found
        """
        content_path = os.path.join(self.content_dir, f"{doc_id}.txt")
        metadata_path = os.path.join(self.metadata_dir, f"{doc_id}.json")
        
        content = None
        metadata = None
        
        if os.path.exists(content_path):
            with open(content_path, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            self.logger.warning(f"Document content not found for {doc_id}")
        
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            self.logger.warning(f"Document metadata not found for {doc_id}")
        
        return content, metadata

    def document_exists(self, doc_id: str) -> bool:
        """
        Check if a document exists in the repository.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if document exists, False otherwise
        """
        content_path = os.path.join(self.content_dir, f"{doc_id}.txt")
        return os.path.exists(content_path)

    def list_documents(self) -> List[str]:
        """
        List all document IDs in the repository.
        
        Returns:
            List of document IDs
        """
        documents = []
        for filename in os.listdir(self.content_dir):
            if filename.endswith(".txt"):
                doc_id = filename[:-4]  # Remove .txt extension
                documents.append(doc_id)
        return documents

    def search_by_metadata(self, query: Dict[str, Any]) -> List[str]:
        """
        Search for documents by metadata fields.
        
        Args:
            query: Dict of metadata field names and values to match
            
        Returns:
            List of matching document IDs
        """
        matching_docs = []
        
        for filename in os.listdir(self.metadata_dir):
            if not filename.endswith(".json"):
                continue
                
            metadata_path = os.path.join(self.metadata_dir, filename)
            
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                
                # Check if all query criteria match
                match = True
                for key, value in query.items():
                    if key not in metadata or metadata[key] != value:
                        match = False
                        break
                
                if match:
                    doc_id = filename[:-5]  # Remove .json extension
                    matching_docs.append(doc_id)
            except Exception as e:
                self.logger.error(f"Error reading metadata for {filename}: {e}")
        
        return matching_docs

    def get_document_citation(self, doc_id: str) -> str:
        """
        Get a formatted citation for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Formatted citation string
        """
        _, metadata = self.get_document(doc_id)
        
        if not metadata:
            return f"Unknown document ({doc_id})"
        
        # Format citation
        title = metadata.get("title", "Untitled")
        source = metadata.get("source", "Unknown source")
        url = metadata.get("url", "")
        date_str = metadata.get("published", "")
        
        if date_str:
            try:
                # Try to parse and format date
                date = datetime.fromisoformat(date_str)
                date_str = date.strftime("%B %d, %Y")
            except (ValueError, TypeError):
                # Keep original if parsing fails
                pass
        
        citation = f"{title} - {source}"
        if date_str:
            citation += f" ({date_str})"
        if url:
            citation += f", {url}"
        citation += f" [Document ID: {doc_id[:8]}]"
        
        return citation

    def verify_document_integrity(self, doc_id: str) -> bool:
        """
        Verify the integrity of a document by recomputing its hash.
        
        Args:
            doc_id: Document ID to verify
            
        Returns:
            True if verification succeeds, False otherwise
        """
        content, _ = self.get_document(doc_id)
        
        if not content:
            self.logger.warning(f"Cannot verify missing document: {doc_id}")
            return False
        
        computed_id = self._generate_document_id(content)
        
        if computed_id != doc_id:
            self.logger.warning(f"Document integrity verification failed for {doc_id}")
            return False
            
        return True

    def _generate_document_id(self, content: str) -> str:
        """
        Generate a document ID by hashing the content.
        Uses SHA-256 for the hash.
        
        Args:
            content: Document content
            
        Returns:
            Hex digest of hash
        """
        # Check cache first (optimization)
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        
        # Store in cache
        self._document_id_cache[content_hash] = content_hash
        
        return content_hash
        
    def batch_store_documents(self, documents: List[Tuple[str, Dict[str, Any]]]) -> List[str]:
        """
        Store multiple documents in a batch operation.
        
        Args:
            documents: List of (content, metadata) tuples
            
        Returns:
            List of document IDs
        """
        doc_ids = []
        for content, metadata in documents:
            doc_id = self.store_document(content, metadata)
            doc_ids.append(doc_id)
        return doc_ids
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get repository statistics.
        
        Returns:
            Dictionary with statistics
        """
        total_docs = len(self.list_documents())
        
        # Get total size
        content_size = 0
        for doc_id in self.list_documents():
            content_path = os.path.join(self.content_dir, f"{doc_id}.txt")
            if os.path.exists(content_path):
                content_size += os.path.getsize(content_path)
        
        # Get sources distribution
        sources = {}
        for doc_id in self.list_documents():
            _, metadata = self.get_document(doc_id)
            if metadata:
                source = metadata.get("source", "Unknown")
                sources[source] = sources.get(source, 0) + 1
        
        return {
            "total_documents": total_docs,
            "content_size_bytes": content_size,
            "sources": sources
        }
