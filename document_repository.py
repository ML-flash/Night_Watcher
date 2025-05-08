"""
Night_watcher Document Repository
Provides immutable document storage with cryptographic provenance and master seed.
"""

import os
import json
import hashlib
import logging
import re
from datetime import datetime
from typing import Dict, Any, Optional, List, Set, Tuple

# Configure logging
logger = logging.getLogger(__name__)

class DocumentRepository:
    """
    Repository for storing immutable documents with cryptographic provenance.
    Uses a master seed for additional verification and to prevent tampering.
    """

    def __init__(self, 
                 base_dir: str = "data/documents", 
                 master_seed: str = "NIGHT_WATCHER_FREEDOM_SEED_2025"):
        """
        Initialize the document repository.
        
        Args:
            base_dir: Base directory for document storage
            master_seed: Master seed for document hashing (keeps verification tamper-resistant)
        """
        self.base_dir = base_dir
        self.content_dir = os.path.join(base_dir, "content")
        self.metadata_dir = os.path.join(base_dir, "metadata")
        
        # Set master seed for cryptographic operations
        self.master_seed = master_seed
        
        # Create directories if they don't exist
        os.makedirs(self.content_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        # Cache for document IDs
        self._document_id_cache: Dict[str, str] = {}
        
        self.logger = logging.getLogger("DocumentRepository")
        self.logger.info(f"Document repository initialized at {base_dir} with master seed")

    def store_document(self, content: str, metadata: Dict[str, Any]) -> str:
        """
        Store a document with its metadata and return its document ID.
        The document ID is a cryptographic hash of the content + master seed, ensuring immutability.
        The ID is also embedded within the document itself for self-verification.
        
        Args:
            content: Document content text
            metadata: Document metadata (source, URL, collection time, etc.)
            
        Returns:
            Document ID (hash)
        """
        # First check if content already has an embedded document ID
        embedded_id = self._extract_embedded_id(content)
        if embedded_id:
            # Verify that the embedded ID matches the content hash
            content_without_header = self._remove_provenance_header(content)
            computed_id = self._generate_document_id(content_without_header)
            
            if embedded_id != computed_id:
                self.logger.warning(
                    f"Document has embedded ID {embedded_id} but content hash is {computed_id}. "
                    f"This indicates the content may have been tampered with or was generated with a different seed."
                )
                # Store the document with a warning in metadata
                metadata["integrity_warning"] = "Embedded ID doesn't match content hash"
                metadata["embedded_id"] = embedded_id
                
                # Use the computed ID as the definitive one
                doc_id = computed_id
                # Don't keep the existing header since it contains an incorrect ID
                content = content_without_header
            else:
                # ID is valid, use it
                doc_id = embedded_id
                # Keep the existing provenance header
                self.logger.info(f"Document already has valid embedded ID: {doc_id}")
        else:
            # No embedded ID, generate new ID from content using master seed
            doc_id = self._generate_document_id(content)
            self.logger.debug(f"Generated new document ID: {doc_id}")
        
        # Add document ID to metadata
        metadata = metadata.copy()  # Don't modify the original
        metadata["document_id"] = doc_id
        metadata["stored_at"] = datetime.now().isoformat()
        
        # Check if document already exists
        content_path = os.path.join(self.content_dir, f"{doc_id}.txt")
        metadata_path = os.path.join(self.metadata_dir, f"{doc_id}.json")
        
        # If this is a new document or if we stripped an invalid header,
        # add a proper provenance header
        if not embedded_id or embedded_id != doc_id:
            content_with_header = self._add_provenance_header(content, doc_id, metadata)
            content_to_store = content_with_header
        else:
            # Use existing content with valid header
            content_to_store = content
            
        if os.path.exists(content_path):
            # Document exists - verify integrity
            with open(content_path, "r", encoding="utf-8") as f:
                existing_content = f.read()
            
            existing_id = self._extract_embedded_id(existing_content)
            if existing_id != doc_id:
                self.logger.error(
                    f"Document with ID {doc_id} exists but has different embedded ID: {existing_id}. "
                    f"This indicates tampering. Not overwriting."
                )
                return doc_id
                
            # File exists with correct ID, don't overwrite content
            self.logger.info(f"Document {doc_id} already exists, not overwriting content")
        else:
            # Store content with provenance header
            with open(content_path, "w", encoding="utf-8") as f:
                f.write(content_to_store)
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
                
            # Verify the embedded ID matches the document ID
            embedded_id = self._extract_embedded_id(content)
            if embedded_id and embedded_id != doc_id:
                self.logger.warning(
                    f"Document has embedded ID {embedded_id} but was requested with ID {doc_id}. "
                    f"This indicates a possible mismatch or tampering."
                )
        else:
            self.logger.warning(f"Document content not found for {doc_id}")
        
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            self.logger.warning(f"Document metadata not found for {doc_id}")
        
        return content, metadata

    def get_document_content_only(self, doc_id: str) -> Optional[str]:
        """
        Retrieve a document's content without the provenance header.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document content without provenance header, or None if not found
        """
        content, _ = self.get_document(doc_id)
        if content:
            return self._remove_provenance_header(content)
        return None

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

    def verify_document_integrity(self, doc_id: str) -> Dict[str, Any]:
        """
        Verify the integrity of a document by:
        1. Checking that the document exists
        2. Verifying the embedded ID matches the requested ID
        3. Recomputing the hash of the content (without header) matches the ID
        
        Args:
            doc_id: Document ID to verify
            
        Returns:
            Dictionary with verification results including:
            - exists: Whether the document exists
            - embedded_id_valid: Whether the embedded ID is valid
            - content_hash_valid: Whether the content hash is valid
            - overall: Overall validity (True only if all checks pass)
        """
        result = {
            "exists": False,
            "embedded_id_valid": False,
            "content_hash_valid": False,
            "overall": False
        }
        
        content, _ = self.get_document(doc_id)
        
        if not content:
            self.logger.warning(f"Cannot verify missing document: {doc_id}")
            return result
            
        result["exists"] = True
        
        # Check embedded ID
        embedded_id = self._extract_embedded_id(content)
        if not embedded_id:
            self.logger.warning(f"Document {doc_id} does not have an embedded ID")
            # Continue with content hash verification
        elif embedded_id == doc_id:
            result["embedded_id_valid"] = True
        
        # Verify content hash
        content_without_header = self._remove_provenance_header(content)
        computed_id = self._generate_document_id(content_without_header)
        
        if computed_id == doc_id:
            result["content_hash_valid"] = True
        else:
            self.logger.warning(
                f"Document integrity verification failed for {doc_id}. "
                f"Computed hash: {computed_id}"
            )
        
        # Overall validity requires all checks to pass
        if result["exists"] and (not embedded_id or result["embedded_id_valid"]) and result["content_hash_valid"]:
            result["overall"] = True
            
        return result

    def _generate_document_id(self, content: str) -> str:
        """
        Generate a document ID by hashing the content with the master seed.
        Uses SHA-256 for the hash.
        
        Args:
            content: Document content (without provenance header)
            
        Returns:
            Hex digest of hash
        """
        # Combine content with master seed for seed-dependent hashing
        # This ensures documents can only be verified by those with the master seed
        seeded_content = f"{self.master_seed}:{content}"
        content_hash = hashlib.sha256(seeded_content.encode('utf-8')).hexdigest()
        
        # Store in cache
        self._document_id_cache[content_hash] = content_hash
        
        return content_hash
        
    def _add_provenance_header(self, content: str, doc_id: str, metadata: Dict[str, Any]) -> str:
        """
        Add a provenance header to the document content.
        
        Args:
            content: Document content
            doc_id: Document ID
            metadata: Document metadata
            
        Returns:
            Content with provenance header
        """
        header_lines = [
            "<!-- Night_watcher Document Provenance Header -->",
            "<!--",
            f"DOCUMENT_ID: {doc_id}",
            f"STORED_AT: {metadata.get('stored_at', datetime.now().isoformat())}",
            f"SOURCE: {metadata.get('source', 'Unknown')}"
        ]
        
        # Add optional metadata fields if present
        if "title" in metadata:
            header_lines.append(f"TITLE: {metadata['title']}")
        if "url" in metadata:
            header_lines.append(f"URL: {metadata['url']}")
        if "published" in metadata:
            header_lines.append(f"PUBLISHED: {metadata['published']}")
        if "bias_label" in metadata:
            header_lines.append(f"BIAS_LABEL: {metadata['bias_label']}")
        
        # Add verification note - provides an alert even if viewed in plain text
        header_lines.extend([
            "VERIFICATION: This document's integrity can only be verified with the correct master seed",
            "DO NOT MODIFY THIS HEADER - Integrity verification will fail",
            "-->",
            ""  # Empty line to separate header from content
        ])
        
        header = "\n".join(header_lines)
        
        return f"{header}\n{content}"
    
    def _extract_embedded_id(self, content: str) -> Optional[str]:
        """
        Extract the embedded document ID from the provenance header.
        
        Args:
            content: Document content with possible provenance header
            
        Returns:
            Embedded document ID or None if not found
        """
        if not content:
            return None
            
        # Look for the document ID line in the header
        match = re.search(r"DOCUMENT_ID:\s*([a-fA-F0-9]{64})", content)
        if match:
            return match.group(1)
            
        return None
    
    def _remove_provenance_header(self, content: str) -> str:
        """
        Remove the provenance header from the document content.
        
        Args:
            content: Document content with possible provenance header
            
        Returns:
            Content without provenance header
        """
        if not content:
            return ""
            
        # Find the end of the provenance header
        match = re.search(r"-->\s*\n", content)
        if match:
            return content[match.end():]
            
        # No header found, return original content
        return content
        
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
    
    def verify_documents_from_other_source(self, other_base_dir: str, other_seed: Optional[str] = None) -> Dict[str, Any]:
        """
        Verify documents from another source, optionally using a different seed.
        Useful for verifying documents from another resistance cell.
        
        Args:
            other_base_dir: Base directory of the other repository
            other_seed: Seed used by the other repository (if different)
            
        Returns:
            Dictionary with verification results
        """
        # Create a temporary repository with the other seed if provided
        temp_repo = None
        if other_seed:
            temp_repo = DocumentRepository(base_dir=other_base_dir, master_seed=other_seed)
        else:
            # Use same seed as this repository
            temp_repo = DocumentRepository(base_dir=other_base_dir, master_seed=self.master_seed)
            
        # Get all documents from the other repository
        other_docs = []
        other_content_dir = os.path.join(other_base_dir, "content")
        if os.path.exists(other_content_dir):
            for filename in os.listdir(other_content_dir):
                if filename.endswith(".txt"):
                    doc_id = filename[:-4]  # Remove .txt extension
                    other_docs.append(doc_id)
        
        # Verify each document
        results = {
            "total": len(other_docs),
            "valid": 0,
            "invalid": 0,
            "details": {}
        }
        
        for doc_id in other_docs:
            verification = temp_repo.verify_document_integrity(doc_id)
            results["details"][doc_id] = verification
            if verification["overall"]:
                results["valid"] += 1
            else:
                results["invalid"] += 1
                
        return results
    
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
        # Get integrity status
        valid_docs = 0
        invalid_docs = 0
        
        for doc_id in self.list_documents():
            _, metadata = self.get_document(doc_id)
            if metadata:
                source = metadata.get("source", "Unknown")
                sources[source] = sources.get(source, 0) + 1
                
            # Check integrity
            verification = self.verify_document_integrity(doc_id)
            if verification["overall"]:
                valid_docs += 1
            else:
                invalid_docs += 1
        
        return {
            "total_documents": total_docs,
            "content_size_bytes": content_size,
            "sources": sources,
            "integrity": {
                "valid": valid_docs,
                "invalid": invalid_docs
            }
        }
    