#!/usr/bin/env python3
"""
Night_watcher Provenance Tool
Command-line tool for managing cryptographic provenance and keys.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, Any, Optional

from key_manager import KeyManager
from provenance import ProvenanceTracker

def init_provenance(args) -> int:
    """Initialize the provenance system with a new master key"""
    try:
        # Get passphrase
        passphrase = get_passphrase(args, required=True)
        
        # Initialize key manager
        key_manager = KeyManager(keys_dir=args.keys_dir)
        
        # Check if key already exists
        keys = key_manager.list_keys()
        if args.key_id in keys:
            print(f"Key '{args.key_id}' already exists.")
            confirm = input("Do you want to overwrite it? (y/n): ")
            if confirm.lower() != 'y':
                print("Operation cancelled.")
                return 1
        
        # Generate key
        key_info = key_manager.generate_keypair(args.key_id, passphrase)
        
        print(f"\nInitialized provenance system with new key: {args.key_id}")
        print(f"Fingerprint: {key_info['fingerprint']}")
        print(f"Algorithm: {key_info['algorithm']}")
        print(f"Created: {key_info['created_at']}")
        
        # Initialize provenance tracker
        provenance_dir = os.path.expanduser(args.provenance_dir)
        os.makedirs(provenance_dir, exist_ok=True)
        
        tracker = ProvenanceTracker(
            base_dir=provenance_dir,
            keys_dir=args.keys_dir,
            master_key_id=args.key_id,
            passphrase=passphrase
        )
        
        # Create a test signature to verify everything works
        test_sig = tracker.sign_content(
            content="Provenance system initialization test",
            metadata={
                "type": "initialization_test",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Verify the signature
        is_valid = tracker.verify_signature(
            content="Provenance system initialization test",
            signature_record=test_sig
        )
        
        if is_valid:
            print("\n✓ Provenance system initialized successfully!")
            print(f"  - Keys directory: {args.keys_dir}")
            print(f"  - Provenance directory: {args.provenance_dir}")
            return 0
        else:
            print("\n✗ Provenance system initialization FAILED!")
            print("  Signature verification test failed.")
            return 1
        
    except Exception as e:
        print(f"Error initializing provenance system: {e}")
        return 1

def list_keys(args) -> int:
    """List all available keys"""
    try:
        # Initialize key manager
        key_manager = KeyManager(keys_dir=args.keys_dir)
        
        # Get keys
        keys = key_manager.list_keys()
        
        if not keys:
            print("No keys found.")
            return 0
        
        print("\nAvailable keys:")
        print("-" * 80)
        print(f"{'Key ID':<20} {'Fingerprint':<40} {'Has Private':<10} {'Created':<20}")
        print("-" * 80)
        
        for key_id, key_info in keys.items():
            created = key_info.get('created_at', 'N/A')
            if isinstance(created, str) and len(created) > 19:
                created = created[:19]  # Trim to fit
                
            print(f"{key_id:<20} {key_info.get('fingerprint', 'N/A')[:40]:<40} "
                  f"{'Yes' if key_info.get('has_private') else 'No':<10} {created:<20}")
        
        return 0
    except Exception as e:
        print(f"Error listing keys: {e}")
        return 1

def export_key(args) -> int:
    """Export a public key"""
    try:
        # Initialize key manager
        key_manager = KeyManager(keys_dir=args.keys_dir)
        
        # Export key
        exported_key = key_manager.export_public_key(args.key_id, format=args.format)
        
        # Output
        if args.output:
            with open(args.output, 'w') as f:
                if args.format == "json":
                    json.dump(exported_key, f, indent=2)
                else:
                    f.write(exported_key)
            print(f"Key exported to {args.output}")
        else:
            if args.format == "json":
                print(json.dumps(exported_key, indent=2))
            else:
                print(exported_key)
        
        return 0
    except Exception as e:
        print(f"Error exporting key: {e}")
        return 1

def verify_document(args) -> int:
    """Verify a document's provenance"""
    try:
        # Get passphrase (not actually needed for verification)
        passphrase = None  # Verification doesn't need a passphrase
        
        # Initialize provenance tracker
        tracker = ProvenanceTracker(
            base_dir=args.provenance_dir,
            keys_dir=args.keys_dir,
            master_key_id=args.key_id
        )
        
        # Get document content
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading document: {e}")
            return 1
        
        # Verify document
        is_valid, message = tracker.verify_document(args.doc_id, content)
        
        print(f"\n=== Document Verification: {args.doc_id} ===")
        
        if is_valid:
            print(f"PROVENANCE: ✓ VERIFIED")
            print(f"Document provenance is intact and verified.")
            
            # Get signature details
            sig_record = tracker.load_signature(args.doc_id)
            if sig_record:
                print(f"\nSignature Details:")
                print(f"  Key ID: {sig_record.get('key_id', 'Unknown')}")
                print(f"  Key Fingerprint: {sig_record.get('key_fingerprint', 'Unknown')}")
                print(f"  Timestamp: {sig_record.get('timestamp', 'Unknown')}")
            
            return 0
        else:
            print(f"PROVENANCE: ✗ VERIFICATION FAILED")
            print(f"Document may have been tampered with or corrupted.")
            print(f"Reason: {message}")
            return 1
    
    except Exception as e:
        print(f"Error verifying document: {e}")
        return 1

def sign_document(args) -> int:
    """Sign a document"""
    try:
        # Get passphrase
        passphrase = get_passphrase(args, required=True)
        
        # Initialize provenance tracker
        tracker = ProvenanceTracker(
            base_dir=args.provenance_dir,
            keys_dir=args.keys_dir,
            master_key_id=args.key_id,
            passphrase=passphrase
        )
        
        # Read document content
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading document: {e}")
            return 1
        
        # Create basic metadata
        metadata = {
            "title": os.path.basename(args.file),
            "source": "manual_signing",
            "signed_by": args.key_id,
            "description": args.description or "Manually signed document"
        }
        
        # Sign the document
        signature_record = tracker.sign_content(content, metadata, passphrase)
        
        # Store the signature
        signature_path = tracker.store_signature(args.doc_id, signature_record)
        
        print(f"\n✓ Document signed successfully!")
        print(f"Document ID: {args.doc_id}")
        print(f"Signature stored at: {signature_path}")
        
        return 0
    
    except Exception as e:
        print(f"Error signing document: {e}")
        return 1

def get_passphrase(args, required: bool = True) -> Optional[str]:
    """Get passphrase from args, environment, or prompt"""
    passphrase = args.passphrase
    
    if not passphrase:
        passphrase = os.environ.get("NIGHT_WATCHER_PASSPHRASE")
    
    if not passphrase and required:
        import getpass
        passphrase = getpass.getpass("Enter passphrase: ")
    
    return passphrase

def main() -> int:
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Night_watcher Provenance Tool")
    
    # Global options
    parser.add_argument("--passphrase", help="Passphrase for key access")
    parser.add_argument("--keys-dir", default="~/.night_watcher/keys", 
                        help="Directory for key storage")
    parser.add_argument("--key-id", default="night_watcher_master", 
                        help="Key ID to use")
    parser.add_argument("--provenance-dir", default="data/provenance",
                        help="Directory for provenance storage")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Initialize provenance system
    init_parser = subparsers.add_parser("init", help="Initialize provenance system")
    
    # List keys command
    list_parser = subparsers.add_parser("list-keys", help="List available keys")
    
    # Export key command
    export_parser = subparsers.add_parser("export-key", help="Export a public key")
    export_parser.add_argument("--format", choices=["json", "pem"], default="json", 
                               help="Export format")
    export_parser.add_argument("--output", help="Output file (stdout if not specified)")
    
    # Sign document command
    sign_parser = subparsers.add_parser("sign", help="Sign a document")
    sign_parser.add_argument("file", help="File to sign")
    sign_parser.add_argument("doc_id", help="Document ID to use")
    sign_parser.add_argument("--description", help="Optional document description")
    
    # Verify document command
    verify_parser = subparsers.add_parser("verify", help="Verify a document")
    verify_parser.add_argument("file", help="File to verify")
    verify_parser.add_argument("doc_id", help="Document ID to verify")
    
    args = parser.parse_args()
    
    # Handle commands
    if args.command == "init":
        return init_provenance(args)
    elif args.command == "list-keys":
        return list_keys(args)
    elif args.command == "export-key":
        return export_key(args)
    elif args.command == "sign":
        return sign_document(args)
    elif args.command == "verify":
        return verify_document(args)
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())
