"""
Night_watcher Key Manager
Core key management for cryptographic provenance.
"""

import os
import sys
import base64
import logging
import hashlib
import json
from datetime import datetime
from typing import Dict, Optional, Any, Union
from pathlib import Path

# Standard library for cryptography
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidSignature

class KeyManager:
    """
    Manages cryptographic keys for Night_watcher with a focus on security.
    Uses Ed25519 keys for signing and verification.
    """
    
    def __init__(self, keys_dir: str = "~/.night_watcher/keys"):
        """
        Initialize the key manager.
        
        Args:
            keys_dir: Directory for key storage
        """
        self.keys_dir = os.path.expanduser(keys_dir)
        self.logger = logging.getLogger("KeyManager")
        
        # Security parameters
        self.security_params = {
            "pbkdf2_iterations": 310000,  # Strong but more reasonable
            "salt_bytes": 16
        }
        
        # Create key directory with restricted permissions
        os.makedirs(self.keys_dir, mode=0o700, exist_ok=True)
        
        # Check existing files have correct permissions
        self._verify_directory_permissions()
    
    def _verify_directory_permissions(self) -> None:
        """Verify and fix permissions on the keys directory"""
        # Check directory permissions
        try:
            current_mode = os.stat(self.keys_dir).st_mode & 0o777
            if current_mode != 0o700:
                os.chmod(self.keys_dir, 0o700)
                self.logger.warning(f"Fixed permissions on keys directory: {self.keys_dir}")
        except Exception as e:
            self.logger.error(f"Unable to verify/fix directory permissions: {e}")

        # Check file permissions for existing key files
        try:
            for filename in os.listdir(self.keys_dir):
                file_path = os.path.join(self.keys_dir, filename)
                if os.path.isfile(file_path) and filename.endswith('.key'):
                    current_mode = os.stat(file_path).st_mode & 0o777
                    if current_mode != 0o600:
                        os.chmod(file_path, 0o600)
                        self.logger.warning(f"Fixed permissions on key file: {filename}")
        except Exception as e:
            self.logger.error(f"Unable to verify/fix file permissions: {e}")
    
    def generate_keypair(self, key_id: str, passphrase: str) -> Dict[str, str]:
        """
        Generate a new Ed25519 keypair and store it securely.
        
        Args:
            key_id: Identifier for the key
            passphrase: Passphrase to encrypt the private key
            
        Returns:
            Dictionary with key information
        """
        # Generate new Ed25519 keypair
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        
        # Get key in bytes format
        private_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
        
        # Create fingerprint for key identification
        key_fingerprint = hashlib.sha256(public_bytes).hexdigest()
        
        # Encrypt and store the private key
        self._store_private_key(key_id, private_bytes, passphrase)
        
        # Store the public key (can be public)
        self._store_public_key(key_id, public_bytes)
        
        # Return key information (NO private key material)
        return {
            "key_id": key_id,
            "fingerprint": key_fingerprint,
            "algorithm": "Ed25519",
            "created_at": datetime.now().isoformat(),
            "public_key_b64": base64.b64encode(public_bytes).decode('utf-8')
        }
    
    def _derive_encryption_key(self, passphrase: str, salt: bytes) -> bytes:
        """
        Derive an encryption key from passphrase using PBKDF2.
        
        Args:
            passphrase: User passphrase
            salt: Salt for key derivation
            
        Returns:
            Derived key bytes
        """
        # Create key derivation function
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,  # 256-bit key
            salt=salt,
            iterations=self.security_params["pbkdf2_iterations"]
        )
        
        # Derive key
        key = kdf.derive(passphrase.encode())
        return key
    
    def _store_private_key(self, key_id: str, key_bytes: bytes, passphrase: str) -> None:
        """
        Encrypt and store a private key.
        
        Args:
            key_id: Key identifier
            key_bytes: Raw private key bytes
            passphrase: Passphrase for encryption
        """
        import os
        import secrets
        
        # Generate a random salt
        salt = secrets.token_bytes(self.security_params["salt_bytes"])
        
        # Derive encryption key
        encryption_key = self._derive_encryption_key(passphrase, salt)
        
        # Generate a nonce for AES-GCM
        nonce = secrets.token_bytes(12)
        
        # Encrypt the private key
        aesgcm = AESGCM(encryption_key)
        ciphertext = aesgcm.encrypt(nonce, key_bytes, b"")
        
        # Create key file content
        key_data = {
            "version": 1,
            "key_id": key_id,
            "algorithm": "Ed25519",
            "encryption": "AES-256-GCM",
            "salt": base64.b64encode(salt).decode('utf-8'),
            "nonce": base64.b64encode(nonce).decode('utf-8'),
            "encrypted_key": base64.b64encode(ciphertext).decode('utf-8'),
            "created_at": datetime.now().isoformat()
        }
        
        # Write to file with restricted permissions
        key_path = os.path.join(self.keys_dir, f"{key_id}_private.key")
        with open(key_path, 'w') as f:
            json.dump(key_data, f, indent=2)
        
        # Set restrictive permissions
        os.chmod(key_path, 0o600)
        
        # Clear sensitive data from memory
        encryption_key = b'\x00' * len(encryption_key)
        key_bytes = b'\x00' * len(key_bytes)
    
    def _store_public_key(self, key_id: str, key_bytes: bytes) -> None:
        """
        Store a public key.
        
        Args:
            key_id: Key identifier
            key_bytes: Raw public key bytes
        """
        # Create key file content
        key_data = {
            "version": 1,
            "key_id": key_id,
            "algorithm": "Ed25519",
            "public_key": base64.b64encode(key_bytes).decode('utf-8'),
            "fingerprint": hashlib.sha256(key_bytes).hexdigest(),
            "created_at": datetime.now().isoformat()
        }
        
        # Write to file
        key_path = os.path.join(self.keys_dir, f"{key_id}_public.key")
        with open(key_path, 'w') as f:
            json.dump(key_data, f, indent=2)
    
    def load_private_key(self, key_id: str, passphrase: str) -> ed25519.Ed25519PrivateKey:
        """
        Load and decrypt a private key.
        
        Args:
            key_id: Key identifier
            passphrase: Passphrase for decryption
            
        Returns:
            Ed25519PrivateKey object
        """
        # Load encrypted key file
        key_path = os.path.join(self.keys_dir, f"{key_id}_private.key")
        try:
            with open(key_path, 'r') as f:
                key_data = json.load(f)
        except FileNotFoundError:
            raise ValueError(f"Private key not found: {key_id}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid key file format: {key_id}")
        
        # Extract encryption parameters
        salt = base64.b64decode(key_data["salt"])
        nonce = base64.b64decode(key_data["nonce"])
        ciphertext = base64.b64decode(key_data["encrypted_key"])
        
        # Derive decryption key
        decryption_key = self._derive_encryption_key(passphrase, salt)
        
        try:
            # Decrypt the private key
            aesgcm = AESGCM(decryption_key)
            key_bytes = aesgcm.decrypt(nonce, ciphertext, b"")
            
            # Create key object
            private_key = ed25519.Ed25519PrivateKey.from_private_bytes(key_bytes)
            
            return private_key
            
        except Exception as e:
            raise ValueError(f"Failed to decrypt key: {str(e)}")
        finally:
            # Clear sensitive data
            decryption_key = b'\x00' * len(decryption_key)
    
    def load_public_key(self, key_id: str) -> ed25519.Ed25519PublicKey:
        """
        Load a public key.
        
        Args:
            key_id: Key identifier
            
        Returns:
            Ed25519PublicKey object
        """
        # Load public key file
        key_path = os.path.join(self.keys_dir, f"{key_id}_public.key")
        try:
            with open(key_path, 'r') as f:
                key_data = json.load(f)
        except FileNotFoundError:
            raise ValueError(f"Public key not found: {key_id}")
        
        # Extract and decode key
        public_key_bytes = base64.b64decode(key_data["public_key"])
        
        # Create key object
        return ed25519.Ed25519PublicKey.from_public_bytes(public_key_bytes)
    
    def sign_data(self, key_id: str, passphrase: str, data: bytes) -> bytes:
        """
        Sign data using the specified private key.
        
        Args:
            key_id: Key identifier
            passphrase: Passphrase to decrypt the private key
            data: Data to sign
            
        Returns:
            Signature bytes
        """
        try:
            # Load the private key
            private_key = self.load_private_key(key_id, passphrase)
            
            # Sign the data
            signature = private_key.sign(data)
            
            return signature
        finally:
            # Clean up memory
            pass
    
    def verify_signature(self, key_id: str, data: bytes, signature: bytes) -> bool:
        """
        Verify a signature using the specified public key.
        
        Args:
            key_id: Key identifier
            data: Data that was signed
            signature: Signature to verify
            
        Returns:
            True if signature is valid, False otherwise
        """
        try:
            # Load the public key
            public_key = self.load_public_key(key_id)
            
            # Verify the signature
            public_key.verify(signature, data)
            return True
        except InvalidSignature:
            return False
        except Exception as e:
            self.logger.error(f"Error verifying signature: {e}")
            return False
    
    def export_public_key(self, key_id: str, format: str = "json") -> Union[str, Dict]:
        """
        Export a public key in the specified format.
        
        Args:
            key_id: Key identifier
            format: Export format ("json", "pem", "raw")
            
        Returns:
            Exported key in the specified format
        """
        public_key = self.load_public_key(key_id)
        
        if format == "pem":
            key_bytes = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            return key_bytes.decode('utf-8')
        
        elif format == "raw":
            key_bytes = public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw
            )
            return base64.b64encode(key_bytes).decode('utf-8')
        
        else:  # json
            key_path = os.path.join(self.keys_dir, f"{key_id}_public.key")
            with open(key_path, 'r') as f:
                return json.load(f)
    
    def list_keys(self) -> Dict[str, Dict[str, Any]]:
        """
        List all available keys.
        
        Returns:
            Dictionary mapping key_id to key information
        """
        keys = {}
        
        for filename in os.listdir(self.keys_dir):
            if filename.endswith('_public.key'):
                key_id = filename.split('_public.key')[0]
                
                try:
                    key_path = os.path.join(self.keys_dir, filename)
                    with open(key_path, 'r') as f:
                        key_data = json.load(f)
                    
                    # Only include non-sensitive information
                    keys[key_id] = {
                        "key_id": key_id,
                        "fingerprint": key_data.get("fingerprint"),
                        "algorithm": key_data.get("algorithm"),
                        "created_at": key_data.get("created_at")
                    }
                    
                    # Check if private key exists
                    private_key_path = os.path.join(self.keys_dir, f"{key_id}_private.key")
                    keys[key_id]["has_private"] = os.path.exists(private_key_path)
                    
                except Exception as e:
                    self.logger.error(f"Error reading key {key_id}: {e}")
        
        return keys
    
    def key_exists(self, key_id: str) -> bool:
        """
        Check if a key exists.
        
        Args:
            key_id: Key identifier
            
        Returns:
            True if the key exists, False otherwise
        """
        public_key_path = os.path.join(self.keys_dir, f"{key_id}_public.key")
        return os.path.exists(public_key_path)
