# provenance.py - Core provenance tracking module for Night_watcher

import hashlib
import hmac
import json
import os
import base64
import time
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

class ProvenanceTracker:
    """
    Implements cryptographic provenance tracking for Night_watcher documents.
    
    During development, uses a passphrase-based approach for simplicity.
    In production, would be extended to use proper key management.
    """
    
    def __init__(self, base_dir: str = "data/provenance", 
                 dev_passphrase: str = None,
                 dev_mode: bool = True):
        """
        Initialize the provenance tracker.
        
        Args:
            base_dir: Directory for storing provenance records
            dev_passphrase: Development passphrase for key derivation
            dev_mode: Whether to use development mode (passphrase) or production keys
        """
        self.base_dir = base_dir
        self.signatures_dir = os.path.join(base_dir, "signatures")
        self.audit_log_path = os.path.join(base_dir, "audit_log.jsonl")
        
        # Create directories if they don't exist
        os.makedirs(self.signatures_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.audit_log_path), exist_ok=True)
        
        # Set up key based on mode
        self.dev_mode = dev_mode
        if dev_mode:
            # For development: derive key from passphrase
            self.dev_passphrase = dev_passphrase or "night_watcher_development_only"
            self.key = self._derive_key_from_passphrase(self.dev_passphrase)
            print("WARNING: Using development cryptographic mode - NOT FOR PRODUCTION")
        else:
            # For production: would load from secure key storage
            # This is a placeholder - in production would use proper key management
            raise NotImplementedError("Production key management not yet implemented")
    
    def _derive_key_from_passphrase(self, passphrase: str) -> bytes:
        """
        Derive a cryptographic key from a passphrase using PBKDF2.
        
        Args:
            passphrase: The passphrase to derive the key from
            
        Returns:
            Derived key as bytes
        """
        # Use a fixed salt during development for consistency
        dev_salt = b"night_watcher_fixed_salt"
        
        # In production, would use a secure random salt stored with the key
        # salt = os.urandom(16)
        
        # Derive the key using PBKDF2 (simple implementation)
        key = hashlib.pbkdf2_hmac(
            "sha256",
            passphrase.encode("utf-8"),
            dev_salt,
            iterations=100000,  # High iteration count for security
            dklen=32  # 256-bit key
        )
        return key
    
    def sign_content(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a cryptographic signature for content with metadata.
        
        Args:
            content: The document content to sign
            metadata: Document metadata (source, timestamp, etc.)
            
        Returns:
            Signature record with signature and metadata
        """
        # Generate content hash
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        
        # Prepare the signature record
        timestamp = datetime.now().isoformat()
        signature_data = {
            "content_hash": content_hash,
            "metadata": metadata,
            "timestamp": timestamp,
            "dev_mode": self.dev_mode
        }
        
        # Convert to canonical JSON string (sorted keys for consistency)
        signature_json = json.dumps(signature_data, sort_keys=True)
        
        # Create HMAC signature
        signature = hmac.new(
            key=self.key,
            msg=signature_json.encode("utf-8"),
            digestmod=hashlib.sha256
        ).digest()
        
        # Encode signature as base64
        signature_b64 = base64.b64encode(signature).decode("utf-8")
        
        # Create complete signature record
        signature_record = {
            "content_hash": content_hash,
            "metadata": metadata,
            "timestamp": timestamp,
            "signature": signature_b64,
            "dev_mode": self.dev_mode
        }
        
        return signature_record
    
    def verify_signature(self, content: str, signature_record: Dict[str, Any]) -> bool:
        """
        Verify that content matches its signature record.
        
        Args:
            content: Document content to verify
            signature_record: The signature record to check against
            
        Returns:
            True if signature is valid, False otherwise
        """
        # Check content hash
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if content_hash != signature_record.get("content_hash"):
            return False
        
        # Recreate the signature data without the signature itself
        verification_data = {
            "content_hash": signature_record["content_hash"],
            "metadata": signature_record["metadata"],
            "timestamp": signature_record["timestamp"],
            "dev_mode": signature_record["dev_mode"]
        }
        
        # Convert to canonical JSON string (sorted keys for consistency)
        verification_json = json.dumps(verification_data, sort_keys=True)
        
        # Get the recorded signature
        signature_b64 = signature_record.get("signature")
        if not signature_b64:
            return False
        
        try:
            signature = base64.b64decode(signature_b64)
        except:
            return False
        
        # Verify HMAC signature
        expected_signature = hmac.new(
            key=self.key,
            msg=verification_json.encode("utf-8"),
            digestmod=hashlib.sha256
        ).digest()
        
        # Compare signatures in constant time to prevent timing attacks
        return hmac.compare_digest(signature, expected_signature)
    
    def store_signature(self, doc_id: str, signature_record: Dict[str, Any]) -> str:
        """
        Store a signature record for a document.
        
        Args:
            doc_id: Document ID
            signature_record: The signature record to store
            
        Returns:
            Path to the stored signature file
        """
        # Create signature file path
        signature_path = os.path.join(self.signatures_dir, f"{doc_id}.sig.json")
        
        # Write signature record to file
        with open(signature_path, "w", encoding="utf-8") as f:
            json.dump(signature_record, f, indent=2, ensure_ascii=False)
        
        # Append to audit log
        self._append_to_audit_log(doc_id, "signature_created", signature_record["content_hash"])
        
        return signature_path
    
    def load_signature(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a signature record for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Signature record or None if not found
        """
        signature_path = os.path.join(self.signatures_dir, f"{doc_id}.sig.json")
        
        if not os.path.exists(signature_path):
            return None
        
        try:
            with open(signature_path, "r", encoding="utf-8") as f:
                signature_record = json.load(f)
            
            self._append_to_audit_log(doc_id, "signature_accessed", 
                                     signature_record.get("content_hash"))
            return signature_record
        except Exception as e:
            print(f"Error loading signature for {doc_id}: {e}")
            return None
    
    def verify_document(self, doc_id: str, content: str) -> Tuple[bool, str]:
        """
        Verify a document against its stored signature.
        
        Args:
            doc_id: Document ID
            content: Document content
            
        Returns:
            Tuple of (is_valid, message)
        """
        # Load signature
        signature_record = self.load_signature(doc_id)
        
        if not signature_record:
            return False, f"No signature found for document {doc_id}"
        
        # Verify signature
        is_valid = self.verify_signature(content, signature_record)
        
        # Log verification attempt
        self._append_to_audit_log(
            doc_id, 
            "verification_success" if is_valid else "verification_failure",
            signature_record.get("content_hash")
        )
        
        if is_valid:
            return True, "Document signature verified successfully"
        else:
            return False, "Document verification failed - possible tampering detected"
    
    def _append_to_audit_log(self, doc_id: str, action: str, content_hash: str) -> None:
        """
        Append an entry to the audit log.
        
        Args:
            doc_id: Document ID
            action: Action performed
            content_hash: Content hash
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "doc_id": doc_id,
            "action": action,
            "content_hash": content_hash
        }
        
        try:
            with open(self.audit_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"Warning: Failed to write to audit log: {e}")
