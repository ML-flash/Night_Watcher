"""
Night_watcher Document Repository with Integrated Provenance
Consolidated version that handles both documents and analysis provenance.
"""

import os
import json
from file_utils import safe_json_load
import hashlib
import hmac
import base64
import logging
import secrets
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple


class DocumentRepository:
    """Unified document and analysis storage with cryptographic provenance."""

    def __init__(self, base_dir: str = "data/documents", dev_mode: bool = True, config: Optional[Dict[str, Any]] = None):
        self.base_dir = base_dir
        self.content_dir = os.path.join(base_dir, "content")
        self.metadata_dir = os.path.join(base_dir, "metadata")
        self.signatures_dir = os.path.join(base_dir, "signatures")
        self.analysis_dir = os.path.join(base_dir, "analysis_provenance")
        self.config = config or {}
        
        # Create directories
        for d in [self.content_dir, self.metadata_dir, self.signatures_dir, self.analysis_dir]:
            os.makedirs(d, exist_ok=True)

        # Load crypto key (dev mode defaults still apply)
        self._load_crypto_key()
        
        self.logger = logging.getLogger("DocumentRepository")
        if dev_mode:
            self.logger.info("Using development mode (simplified crypto)")

    def _load_crypto_key(self) -> None:
        """Derive or generate the repository HMAC key."""
        secret = os.environ.get("NIGHT_WATCHER_SECRET") or self.config.get("repo_secret") or "night_watcher_dev"
        salt_path = os.path.join(self.base_dir, "repo_salt")
        if not os.path.exists(salt_path):
            salt = secrets.token_bytes(16)
            with open(salt_path, "wb") as f:
                f.write(salt)
        else:
            with open(salt_path, "rb") as f:
                salt = f.read()

        self.key = hashlib.pbkdf2_hmac(
            "sha256",
            secret.encode("utf-8"),
            salt,
            100000,
            dklen=32,
        )

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
            metadata = safe_json_load(metadata_path, default=None)
        
        # Verify if requested
        if verify and content:
            verified = self._verify_signature(doc_id, content, metadata)
        
        return content, metadata, verified

    def store_analysis_provenance(self, 
                                analysis_id: str,
                                document_ids: List[str],
                                analysis_type: str,
                                analysis_parameters: Dict[str, Any],
                                results: Dict[str, Any],
                                analyzer_version: str = "1.0") -> Dict[str, Any]:
        """
        Store provenance for an analysis process.
        
        Args:
            analysis_id: Unique ID for the analysis
            document_ids: List of document IDs that were analyzed
            analysis_type: Type of analysis performed
            analysis_parameters: Parameters used for the analysis
            results: Results of the analysis
            analyzer_version: Version of the analyzer component
            
        Returns:
            Analysis provenance record
        """
        timestamp = datetime.now().isoformat()
        
        # Generate result hash
        result_hash = hashlib.sha256(
            json.dumps(results, sort_keys=True).encode("utf-8")
        ).hexdigest()
        
        analysis_record = {
            "analysis_id": analysis_id,
            "timestamp": timestamp,
            "document_ids": document_ids,
            "analysis_type": analysis_type,
            "analysis_parameters": analysis_parameters,
            "result_hash": result_hash,
            "analyzer_version": analyzer_version
        }
        
        # Create signature
        sig_data = {
            "content_hash": result_hash,
            "metadata": analysis_record,
            "timestamp": timestamp
        }
        
        sig_json = json.dumps(sig_data, sort_keys=True)
        signature = hmac.new(self.key, sig_json.encode("utf-8"), hashlib.sha256).digest()
        sig_data["signature"] = base64.b64encode(signature).decode("utf-8")
        
        # Save complete record
        provenance_record = {
            "analysis_record": analysis_record,
            "signature": sig_data
        }
        
        record_path = os.path.join(self.analysis_dir, f"{analysis_id}.json")
        with open(record_path, "w", encoding="utf-8") as f:
            json.dump(provenance_record, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Stored analysis provenance for {analysis_id}")
        return provenance_record

    def verify_analysis(self, analysis_id: str) -> Dict[str, Any]:
        """Verify an analysis record's integrity."""
        record_path = os.path.join(self.analysis_dir, f"{analysis_id}.json")
        
        if not os.path.exists(record_path):
            return {
                "verified": False,
                "reason": "Analysis record not found"
            }
        
        try:
            provenance_record = safe_json_load(record_path, default=None)
            if provenance_record is None:
                raise ValueError("invalid json")
            
            analysis_record = provenance_record.get("analysis_record", {})
            sig_data = provenance_record.get("signature", {})
            
            # Verify signature
            stored_sig = sig_data.pop("signature", None)
            if not stored_sig:
                return {"verified": False, "reason": "No signature found"}
            
            sig_json = json.dumps(sig_data, sort_keys=True)
            expected_sig = hmac.new(self.key, sig_json.encode("utf-8"), hashlib.sha256).digest()
            
            if not hmac.compare_digest(base64.b64decode(stored_sig), expected_sig):
                return {"verified": False, "reason": "Signature verification failed"}
            
            return {
                "verified": True,
                "analysis_id": analysis_id,
                "timestamp": analysis_record.get("timestamp"),
                "document_count": len(analysis_record.get("document_ids", [])),
                "analysis_type": analysis_record.get("analysis_type")
            }
            
        except Exception as e:
            return {
                "verified": False,
                "reason": f"Error verifying analysis record: {str(e)}"
            }

    def list_documents(self) -> List[str]:
        """List all document IDs."""
        docs = []
        for filename in os.listdir(self.content_dir):
            if filename.endswith(".txt"):
                docs.append(filename[:-4])
        return docs

    def verify_all(self) -> Dict[str, Any]:
        """Verify all documents and analyses."""
        results = {
            "documents": {"total": 0, "verified": 0, "failed": 0},
            "analyses": {"total": 0, "verified": 0, "failed": 0},
            "failures": []
        }
        
        # Verify documents
        for doc_id in self.list_documents():
            results["documents"]["total"] += 1
            _, _, verified = self.get_document(doc_id, verify=True)
            if verified:
                results["documents"]["verified"] += 1
            else:
                results["documents"]["failed"] += 1
                results["failures"].append({
                    "type": "document",
                    "id": doc_id,
                    "reason": "Verification failed"
                })
        
        # Verify analyses
        if os.path.exists(self.analysis_dir):
            for filename in os.listdir(self.analysis_dir):
                if filename.endswith(".json"):
                    analysis_id = filename[:-5]
                    results["analyses"]["total"] += 1
                    
                    verification = self.verify_analysis(analysis_id)
                    if verification["verified"]:
                        results["analyses"]["verified"] += 1
                    else:
                        results["analyses"]["failed"] += 1
                        results["failures"].append({
                            "type": "analysis",
                            "id": analysis_id,
                            "reason": verification.get("reason", "Unknown")
                        })
        
        return results

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
            sig_data = safe_json_load(sig_path, default=None)
            if sig_data is None:
                raise ValueError("invalid json")
            
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
        
        analysis_count = 0
        if os.path.exists(self.analysis_dir):
            analysis_count = len([f for f in os.listdir(self.analysis_dir) if f.endswith(".json")])
        
        return {
            "total_documents": len(docs),
            "total_analyses": analysis_count,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / 1024 / 1024, 2)
        }
