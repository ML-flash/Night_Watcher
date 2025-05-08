#!/usr/bin/env python3
"""
Night_watcher Document Repository
Simple document storage with ID tracking and metadata.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

class DocumentRepository:
    """
    Repository for storing documents with unique IDs and metadata.
    Provides storage and retrieval interface for the Night_watcher framework.
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
        
        self.logger = logging.getLogger("DocumentRepository")
        self.logger.info(f"Document repository initialized at {base_dir}")

    def store_document(self, content: str, metadata: Dict[str, Any]) -> str:
        """
        Store a document with its metadata and return the document ID.
        
        Args:
            content: Document content text
            metadata: Document metadata dictionary
            
        Returns:
            Document ID
        """
        # Use provided ID or generate a simple one
        doc_id = metadata.get("id")
        if not doc_id:
            # Basic ID if none provided
            doc_id = f"doc_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Add document ID to metadata
        metadata = metadata.copy()  # Don't modify original
        metadata["document_id"] = doc_id
        metadata["stored_at"] = datetime.now().isoformat()
        
        # Define file paths
        content_path = os.path.join(self.content_dir, f"{doc_id}.txt")
        metadata_path = os.path.join(self.metadata_dir, f"{doc_id}.json")
        
        # Skip if document already exists
        if os.path.exists(content_path):
            self.logger.info(f"Document {doc_id} already exists, not overwriting")
            return doc_id
            
        try:
            # Store content
            with open(content_path, "w", encoding="utf-8") as f:
                f.write(content)
                
            # Store metadata
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"Stored document {doc_id} ({len(content)} chars)")
            return doc_id
            
        except Exception as e:
            self.logger.error(f"Error storing document {doc_id}: {e}")
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
        
        # Get content
        if os.path.exists(content_path):
            try:
                with open(content_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                self.logger.error(f"Error reading document {doc_id}: {e}")
        else:
            self.logger.warning(f"Document content not found for {doc_id}")
        
        # Get metadata
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception as e:
                self.logger.error(f"Error reading metadata for {doc_id}: {e}")
        else:
            self.logger.warning(f"Document metadata not found for {doc_id}")
        
        return content, metadata

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
                date = datetime.fromisoformat(date_str)
                date_str = date.strftime("%B %d, %Y")
            except (ValueError, TypeError):
                pass
        
        citation = f"{title} - {source}"
        if date_str:
            citation += f" ({date_str})"
        if url:
            citation += f", {url}"
        citation += f" [ID: {doc_id[:8]}...]"
        
        return citation

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get repository statistics.
        
        Returns:
            Dictionary with statistics
        """
        doc_ids = self.list_documents()
        total_docs = len(doc_ids)
        
        # Calculate total content size
        content_size = 0
        for doc_id in doc_ids:
            content_path = os.path.join(self.content_dir, f"{doc_id}.txt")
            if os.path.exists(content_path):
                content_size += os.path.getsize(content_path)
        
        # Get source distribution
        sources = {}
        for doc_id in doc_ids:
            _, metadata = self.get_document(doc_id)
            if metadata:
                source = metadata.get("source", "Unknown")
                sources[source] = sources.get(source, 0) + 1
        
        # Get date distribution
        dates = {}
        for doc_id in doc_ids:
            _, metadata = self.get_document(doc_id)
            if metadata and "published" in metadata:
                date_str = metadata["published"]
                if date_str:
                    try:
                        date = datetime.fromisoformat(date_str)
                        date_key = date.strftime("%Y-%m-%d")
                        dates[date_key] = dates.get(date_key, 0) + 1
                    except (ValueError, TypeError):
                        pass
        
        return {
            "total_documents": total_docs,
            "content_size_bytes": content_size,
            "sources": sources,
            "dates": dates
        }
