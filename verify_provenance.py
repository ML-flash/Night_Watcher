# verify_provenance.py - Command line tool for verifying document and analysis provenance

import argparse
import json
import os
import sys
from datetime import datetime

from document_repository import DocumentRepository
from analysis_provenance import AnalysisProvenanceTracker

def verify_document(doc_id: str, repo_dir: str, passphrase: str) -> int:
    """Verify a document's provenance"""
    repo = DocumentRepository(
        base_dir=repo_dir,
        dev_passphrase=passphrase,
        dev_mode=True
    )
    
    content, metadata, verified = repo.get_document(doc_id, verify=True)
    
    print(f"\n=== Document Verification: {doc_id} ===")
    
    if content is None:
        print(f"ERROR: Document content not found for {doc_id}")
        return 1
        
    if metadata is None:
        print(f"ERROR: Document metadata not found for {doc_id}")
        return 1
    
    print(f"Title: {metadata.get('title', 'Unknown')}")
    print(f"Source: {metadata.get('source', 'Unknown')}")
    print(f"Collected: {metadata.get('collected_at', 'Unknown')}")
    
    if verified:
        print(f"\nPROVENANCE: ✓ VERIFIED")
        print(f"Document provenance is intact and verified.")
        return 0
    else:
        print(f"\nPROVENANCE: ✗ VERIFICATION FAILED")
        print(f"Document may have been tampered with or corrupted.")
        return 1

def verify_analysis(analysis_id: str, analysis_dir: str, passphrase: str) -> int:
    """Verify an analysis record's provenance"""
    tracker = AnalysisProvenanceTracker(
        base_dir=analysis_dir,
        dev_passphrase=passphrase,
        dev_mode=True
    )
    
    result = tracker.verify_analysis_record(analysis_id)
    
    print(f"\n=== Analysis Verification: {analysis_id} ===")
    
    if result.get("verified"):
        print(f"Analysis Type: {result.get('analysis_type', 'Unknown')}")
        print(f"Timestamp: {result.get('timestamp', 'Unknown')}")
        print(f"Documents Analyzed: {result.get('document_count', 0)}")
        print(f"\nPROVENANCE: ✓ VERIFIED")
        print(f"Analysis provenance is intact and verified.")
        return 0
    else:
        print(f"PROVENANCE: ✗ VERIFICATION FAILED")
        print(f"Reason: {result.get('reason', 'Unknown error')}")
        print(f"Analysis record may have been tampered with or corrupted.")
        return 1

def verify_all(repo_dir: str, passphrase: str) -> int:
    """Verify all documents in the repository"""
    repo = DocumentRepository(
        base_dir=repo_dir,
        dev_passphrase=passphrase,
        dev_mode=True
    )
    
    results = repo.verify_all_documents()
    
    print(f"\n=== Repository Verification Summary ===")
    print(f"Total documents: {results['total']}")
    print(f"Verified: {results['verified']}")
    print(f"Failed: {results['failed']}")
    print(f"Missing: {results['missing']}")
    
    if results['failed'] > 0:
        print(f"\nFailure details:")
        for failure in results['failures']:
            print(f"  - {failure['document_id']}: {failure['reason']}")
        return 1
    
    return 0

def main() -> int:
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Night_watcher Provenance Verification")
    parser.add_argument("--passphrase", help="Development passphrase for verification")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Document verification
    doc_parser = subparsers.add_parser("verify-document", help="Verify a document")
    doc_parser.add_argument("doc_id", help="Document ID to verify")
    doc_parser.add_argument("--repo-dir", default="data/documents", help="Document repository directory")
    
    # Analysis verification
    analysis_parser = subparsers.add_parser("verify-analysis", help="Verify an analysis record")
    analysis_parser.add_argument("analysis_id", help="Analysis ID to verify")
    analysis_parser.add_argument("--analysis-dir", default="data/analysis_provenance", help="Analysis directory")
    
    # Verify all
    all_parser = subparsers.add_parser("verify-all", help="Verify all documents")
    all_parser.add_argument("--repo-dir", default="data/documents", help="Document repository directory")
    
    args = parser.parse_args()
    
    # Get passphrase from args or environment or prompt
    passphrase = args.passphrase
    if not passphrase:
        passphrase = os.environ.get("NIGHT_WATCHER_PASSPHRASE")
        
    if not passphrase:
        import getpass
        passphrase = getpass.getpass("Enter development passphrase: ")
    
    if args.command == "verify-document":
        return verify_document(args.doc_id, args.repo_dir, passphrase)
    elif args.command == "verify-analysis":
        return verify_analysis(args.analysis_id, args.analysis_dir, passphrase)
    elif args.command == "verify-all":
        return verify_all(args.repo_dir, passphrase)
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())
