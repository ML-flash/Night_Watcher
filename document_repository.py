"""
Night_watcher Document Repository with Integrated Provenance
Simplified version combining document storage and provenance tracking.
"""

import os
import json
import hashlib
import hmac
import base64
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple


class DocumentRepository:
    """Document storage with integrated cryptographic provenance."""

    def __init__(self, base_dir: str = "data/documents", dev_mode: bool = True):
        self.base_dir = base_dir
        self.content_dir = os.path.join(base_dir, "content")
        self.metadata_dir = os.path.join(base_dir, "metadata")
        self.signatures_dir = os.path.join(base_dir, "signatures")
        
        # Create directories
        for d in [self.content_dir, self.metadata_dir, self.signatures_dir]:
            os.makedirs(d, exist_ok=True)
        
        # Simple key derivation for dev mode
        self.key = hashlib.pbkdf2_hmac(
            "sha256",
            b"night_watcher_dev",
            b"fixed_salt",
            100000,
            dklen=32
        )
        
        self.logger = logging.getLogger("DocumentRepository")
        if dev_mode:
            self.logger.info("Using development mode (simplified crypto)")

    def store_document(self, content: str, metadata: Dict[str, Any]) -> str:
        """Store document with provenance."""
        # Generate or use provided ID
        doc_id = metadata.get("id", f"doc_{datetime.now().strftime('%Y%m%d%H%M%S')}_{os.urandom(4).hex()}")
        
        # Skip if exists
        if os.path.exists(os.path.join(self.content_dir, f"{doc_id}.txt")):
            self.logger.info(f"Document {doc_id} already exists")
            return doc_id
        
        # Add metadata
        metadata["document_id"] = doc_id
        metadata["stored_at"] = datetime.now().isoformat()
        
        try:
            # Store content
            content_path = os.path.join(self.content_dir, f"{doc_id}.txt")
            with open(content_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            # Store metadata
            metadata_path = os.path.join(self.metadata_dir, f"{doc_id}.json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # Create signature
            self._create_signature(doc_id, content, metadata)
            
            self.logger.info(f"Stored document {doc_id}")
            return doc_id
            
        except Exception as e:
            self.logger.error(f"Error storing document: {e}")
            raise

    def get_document(self, doc_id: str, verify: bool = True) -> Tuple[Optional[str], Optional[Dict[str, Any]], bool]:
        """Retrieve document with optional verification."""
        content = None
        metadata = None
        verified = False
        
        # Load content
        content_path = os.path.join(self.content_dir, f"{doc_id}.txt")
        if os.path.exists(content_path):
            with open(content_path, "r", encoding="utf-8") as f:
                content = f.read()
        
        # Load metadata
        metadata_path = os.path.join(self.metadata_dir, f"{doc_id}.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        
        # Verify if requested
        if verify and content:
            verified = self._verify_signature(doc_id, content, metadata)
        
        return content, metadata, verified

    def list_documents(self) -> List[str]:
        """List all document IDs."""
        docs = []
        for filename in os.listdir(self.content_dir):
            if filename.endswith(".txt"):
                docs.append(filename[:-4])
        return docs

    def _create_signature(self, doc_id: str, content: str, metadata: Dict[str, Any]):
        """Create HMAC signature for document."""
        # Create signature data
        sig_data = {
            "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "metadata": metadata,
            "timestamp": datetime.now().isoformat()
        }
        
        # Create HMAC
        sig_json = json.dumps(sig_data, sort_keys=True)
        signature = hmac.new(self.key, sig_json.encode("utf-8"), hashlib.sha256).digest()
        sig_data["signature"] = base64.b64encode(signature).decode("utf-8")
        
        # Save signature
        sig_path = os.path.join(self.signatures_dir, f"{doc_id}.sig.json")
        with open(sig_path, "w", encoding="utf-8") as f:
            json.dump(sig_data, f, indent=2)

    def _verify_signature(self, doc_id: str, content: str, metadata: Dict[str, Any]) -> bool:
        """Verify document signature."""
        sig_path = os.path.join(self.signatures_dir, f"{doc_id}.sig.json")
        if not os.path.exists(sig_path):
            return False
        
        try:
            with open(sig_path, "r", encoding="utf-8") as f:
                sig_data = json.load(f)
            
            # Check content hash
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if content_hash != sig_data.get("content_hash"):
                return False
            
            # Verify HMAC
            stored_sig = sig_data.pop("signature", None)
            if not stored_sig:
                return False
            
            sig_json = json.dumps(sig_data, sort_keys=True)
            expected_sig = hmac.new(self.key, sig_json.encode("utf-8"), hashlib.sha256).digest()
            
            return hmac.compare_digest(
                base64.b64decode(stored_sig),
                expected_sig
            )
            
        except Exception as e:
            self.logger.error(f"Verification error: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get repository statistics."""
        docs = self.list_documents()
        total_size = 0
        
        for doc_id in docs:
            content_path = os.path.join(self.content_dir, f"{doc_id}.txt")
            if os.path.exists(content_path):
                total_size += os.path.getsize(content_path)
        
        return {
            "total_documents": len(docs),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / 1024 / 1024, 2)
        }
