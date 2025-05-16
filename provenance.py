# provenance.py - Core provenance tracking module for Night_watcher

import json
import os
import base64
import time
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

from key_manager import KeyManager

class ProvenanceTracker:
    """
    Implements cryptographic provenance tracking for Night_watcher documents
    using public-private key cryptography for strong security.
    """
    
    def __init__(self, base_dir: str = "data/provenance", 
                 keys_dir: str = "~/.night_watcher/keys",
                 master_key_id: str = "night_watcher_master",
                 passphrase: Optional[str] = None):
        """
        Initialize the provenance tracker.
        
        Args:
            base_dir: Directory for storing provenance records
            keys_dir: Directory for key storage
            master_key_id: Identifier for the master key
            passphrase: Passphrase for key access (can also be provided per operation)
        """
        self.base_dir = base_dir
        self.signatures_dir = os.path.join(base_dir, "signatures")
        self.audit_log_path = os.path.join(base_dir, "audit_log.jsonl")
        self.master_key_id = master_key_id
        self.passphrase = passphrase
        
        # Create directories if they don't exist
        os.makedirs(self.signatures_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.audit_log_path), exist_ok=True)
        
        # Initialize key manager
        self.key_manager = KeyManager(keys_dir=keys_dir)
        
        # Check if we need to initialize keys
        self._ensure_keys_exist()
    
    def _ensure_keys_exist(self) -> None:
        """
        Ensure that the necessary keys exist, creating them if needed.
        """
        # Get existing keys
        existing_keys = self.key_manager.list_keys()
        
        # Check if master key exists
        if self.master_key_id not in existing_keys:
            # Need to generate master key - but can't without passphrase
            if not self.passphrase:
                print(f"\nWARNING: Master key '{self.master_key_id}' not found.")
                print("You'll need to generate it later using generate_master_key().")
                return
            
            # Generate master key
            try:
                self.generate_master_key(self.passphrase)
                print(f"Generated new master key: {self.master_key_id}")
            except Exception as e:
                print(f"Failed to generate master key: {e}")
    
    def generate_master_key(self, passphrase: str) -> Dict[str, Any]:
        """
        Generate a new master key for signing.
        
        Args:
            passphrase: Passphrase to encrypt the private key
            
        Returns:
            Dictionary with key information
        """
        try:
            # Generate a new key pair
            key_info = self.key_manager.generate_keypair(
                key_id=self.master_key_id,
                passphrase=passphrase
            )
            
            # Log key generation
            self._append_to_audit_log(
                "system", "master_key_generated", key_info["fingerprint"]
            )
            
            return key_info
            
        except Exception as e:
            raise ValueError(f"Failed to generate master key: {e}")
    
    def sign_content(self, content: str, metadata: Dict[str, Any], 
                     passphrase: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a cryptographic signature for content with metadata.
        
        Args:
            content: The document content to sign
            metadata: Document metadata (source, timestamp, etc.)
            passphrase: Optional passphrase (if not provided during initialization)
            
        Returns:
            Signature record with signature and metadata
        """
        # Use provided passphrase or instance passphrase
        effective_passphrase = passphrase or self.passphrase
        if not effective_passphrase:
            raise ValueError("Passphrase required for signing operations")
        
        # Generate content hash
        content_hash = self._hash_content(content)
        
        # Prepare the signature record
        timestamp = datetime.now().isoformat()
        signature_data = {
            "content_hash": content_hash,
            "metadata": metadata,
            "timestamp": timestamp,
            "key_id": self.master_key_id
        }
        
        # Convert to canonical JSON string (sorted keys for consistency)
        signature_json = json.dumps(signature_data, sort_keys=True)
        
        try:
            # Sign the data
            signature = self.key_manager.sign_data(
                key_id=self.master_key_id,
                passphrase=effective_passphrase,
                data=signature_json.encode('utf-8')
            )
            
            # Encode signature as base64
            signature_b64 = base64.b64encode(signature).decode('utf-8')
            
            # Get public key info
            key_info = self.key_manager.export_public_key(self.master_key_id, format="json")
            key_fingerprint = key_info.get("fingerprint", "unknown")
            
            # Create complete signature record
            signature_record = {
                "content_hash": content_hash,
                "metadata": metadata,
                "timestamp": timestamp,
                "signature": signature_b64,
                "key_id": self.master_key_id,
                "key_fingerprint": key_fingerprint,
                "version": 1
            }
            
            return signature_record
            
        except Exception as e:
            raise ValueError(f"Signing error: {e}")
    
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
        content_hash = self._hash_content(content)
        if content_hash != signature_record.get("content_hash"):
            return False
        
        # Recreate the signature data without the signature itself
        verification_data = {
            "content_hash": signature_record["content_hash"],
            "metadata": signature_record["metadata"],
            "timestamp": signature_record["timestamp"],
            "key_id": signature_record["key_id"]
        }
        
        # Convert to canonical JSON string (sorted keys for consistency)
        verification_json = json.dumps(verification_data, sort_keys=True)
        
        # Get the recorded signature
        signature_b64 = signature_record.get("signature")
        key_id = signature_record.get("key_id", self.master_key_id)
        
        if not signature_b64:
            return False
        
        try:
            signature = base64.b64decode(signature_b64)
            
            # Verify the signature
            result = self.key_manager.verify_signature(
                key_id=key_id,
                data=verification_json.encode('utf-8'),
                signature=signature
            )
            
            return result
        except Exception as e:
            print(f"Verification error: {e}")
            return False
    
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
        
        # Set proper permissions
        try:
            os.chmod(signature_path, 0o640)  # Owner read/write, group read
        except Exception as e:
            print(f"Warning: Unable to set file permissions: {e}")
        
        # Append to audit log
        self._append_to_audit_log(
            doc_id, 
            "signature_created", 
            signature_record["content_hash"]
        )
        
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
            
            self._append_to_audit_log(
                doc_id, 
                "signature_accessed", 
                signature_record.get("content_hash")
            )
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
    
    def _hash_content(self, content: str) -> str:
        """
        Generate a hash of content.
        
        Args:
            content: Content to hash
            
        Returns:
            Content hash
        """
        import hashlib
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
    
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
    
    def get_key_info(self) -> Dict[str, Any]:
        """
        Get information about the current signing key.
        
        Returns:
            Dictionary with key information
        """
        try:
            return self.key_manager.export_public_key(self.master_key_id, format="json")
        except Exception:
            return {
                "key_id": self.master_key_id,
                "status": "not_found",
                "message": "Master key not found or not accessible"
            }
