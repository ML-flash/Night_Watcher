from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import os
import json
import logging

# Import our new provenance module
from provenance import ProvenanceTracker

def __init__(self, base_dir: str = "data/documents", 
             passphrase: str = None,
             master_key_id: str = "night_watcher_master"):
    """
    Initialize the document repository with provenance tracking.
    
    Args:
        base_dir: Base directory for document storage
        passphrase: Passphrase for cryptographic operations
        master_key_id: Identifier for the master signing key
    """
    self.base_dir = base_dir
    self.content_dir = os.path.join(base_dir, "content")
    self.metadata_dir = os.path.join(base_dir, "metadata")
    
    # Initialize provenance tracker
    provenance_dir = os.path.join(base_dir, "provenance")
    self.provenance = ProvenanceTracker(
        base_dir=provenance_dir,
        master_key_id=master_key_id,
        passphrase=passphrase
    )
    
    # Create directories if they don't exist
    os.makedirs(self.content_dir, exist_ok=True)
    os.makedirs(self.metadata_dir, exist_ok=True)
    
    self.logger = logging.getLogger("DocumentRepository")
    self.logger.info(f"Document repository initialized at {base_dir}")
        
        if dev_mode:
            self.logger.warning("Using DEVELOPMENT cryptographic mode - NOT FOR PRODUCTION")

    def store_document(self, content: str, metadata: Dict[str, Any]) -> str:
        """
        Store a document with its metadata and cryptographic provenance.
        
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
            
            # Create and store provenance record
            signature_record = self.provenance.sign_content(content, metadata)
            self.provenance.store_signature(doc_id, signature_record)
                
            self.logger.info(f"Stored document {doc_id} ({len(content)} chars) with provenance")
            return doc_id
            
        except Exception as e:
            self.logger.error(f"Error storing document {doc_id}: {e}")
            return doc_id

    def get_document(self, doc_id: str, verify: bool = True) -> Tuple[Optional[str], Optional[Dict[str, Any]], bool]:
        """
        Retrieve a document, its metadata, and verify its provenance.
        
        Args:
            doc_id: Document ID
            verify: Whether to verify document provenance
            
        Returns:
            Tuple of (content, metadata, verified) or (None, None, False) if not found
        """
        content_path = os.path.join(self.content_dir, f"{doc_id}.txt")
        metadata_path = os.path.join(self.metadata_dir, f"{doc_id}.json")
        
        content = None
        metadata = None
        verified = False
        
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
        
        # Verify provenance if requested and content exists
        if verify and content is not None:
            is_valid, message = self.provenance.verify_document(doc_id, content)
            verified = is_valid
            
            if not verified:
                self.logger.warning(f"Provenance verification failed for {doc_id}: {message}")
        
        return content, metadata, verified
    
    def get_document_with_provenance(self, doc_id: str) -> Dict[str, Any]:
        """
        Get complete document information including provenance data.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Dictionary with document content, metadata, and provenance information
        """
        content, metadata, verified = self.get_document(doc_id)
        signature_record = self.provenance.load_signature(doc_id)
        
        return {
            "document_id": doc_id,
            "content": content,
            "metadata": metadata,
            "provenance": {
                "signature_record": signature_record,
                "verified": verified,
                "verification_time": datetime.now().isoformat()
            },
            "available": content is not None and metadata is not None
        }
    
    def list_documents(self) -> List[str]:
        """
        List all document IDs in the repository.
        
        Returns:
            List of document IDs
        """
        documents = []
        
        if os.path.exists(self.content_dir):
            for filename in os.listdir(self.content_dir):
                if filename.endswith(".txt"):
                    doc_id = filename[:-4]  # Remove .txt extension
                    documents.append(doc_id)
        
        return documents
        
    def verify_all_documents(self) -> Dict[str, Any]:
        """
        Verify all documents in the repository.
        
        Returns:
            Dictionary with verification results
        """
        documents = self.list_documents()
        results = {
            "total": len(documents),
            "verified": 0,
            "failed": 0,
            "missing": 0,
            "failures": []
        }
        
        for doc_id in documents:
            content, metadata, verified = self.get_document(doc_id)
            
            if content is None or metadata is None:
                results["missing"] += 1
                results["failures"].append({
                    "document_id": doc_id,
                    "reason": "Document or metadata missing"
                })
            elif verified:
                results["verified"] += 1
            else:
                results["failed"] += 1
                results["failures"].append({
                    "document_id": doc_id,
                    "reason": "Provenance verification failed"
                })
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get repository statistics including document count, size, and source distribution.
        
        Returns:
            Dictionary with repository statistics
        """
        # Get document list
        documents = self.list_documents()
        
        # Initialize stats
        stats = {
            "total_documents": len(documents),
            "content_size_bytes": 0,
            "metadata_size_bytes": 0,
            "sources": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Process each document
        for doc_id in documents:
            # Get content file size
            content_path = os.path.join(self.content_dir, f"{doc_id}.txt")
            if os.path.exists(content_path):
                stats["content_size_bytes"] += os.path.getsize(content_path)
            
            # Get metadata file size and source
            metadata_path = os.path.join(self.metadata_dir, f"{doc_id}.json")
            if os.path.exists(metadata_path):
                stats["metadata_size_bytes"] += os.path.getsize(metadata_path)
                
                # Extract source from metadata
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    source = metadata.get("source", "Unknown")
                    stats["sources"][source] = stats["sources"].get(source, 0) + 1
                except Exception as e:
                    self.logger.error(f"Error reading metadata for stats {doc_id}: {e}")
        
        return stats
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document and its associated files.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if deletion was successful, False otherwise
        """
        content_path = os.path.join(self.content_dir, f"{doc_id}.txt")
        metadata_path = os.path.join(self.metadata_dir, f"{doc_id}.json")
        signature_path = os.path.join(self.provenance.signatures_dir, f"{doc_id}.sig.json")
        
        try:
            # Delete content file
            if os.path.exists(content_path):
                os.remove(content_path)
            
            # Delete metadata file
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            
            # Delete signature file
            if os.path.exists(signature_path):
                os.remove(signature_path)
            
            self.logger.info(f"Deleted document {doc_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting document {doc_id}: {e}")
            return False
    
    def search_documents(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search documents by metadata fields.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of matching document metadata
        """
        query = query.lower()
        results = []
        
        for doc_id in self.list_documents():
            # Get metadata
            metadata_path = os.path.join(self.metadata_dir, f"{doc_id}.json")
            
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    # Check if query matches any metadata field
                    for key, value in metadata.items():
                        if isinstance(value, str) and query in value.lower():
                            results.append({
                                "document_id": doc_id,
                                "metadata": metadata,
                                "match_field": key
                            })
                            break
                    
                    # Limit results
                    if len(results) >= max_results:
                        break
                        
                except Exception as e:
                    self.logger.error(f"Error searching document {doc_id}: {e}")
        
        return results
