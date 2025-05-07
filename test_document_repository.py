#!/usr/bin/env python3
"""
Test script for the Night_watcher Document Repository
"""

import os
import sys
import logging
import datetime
from document_repository import DocumentRepository

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_document_repository")

def main():
    """Run the document repository test"""
    # Create a test directory
    test_dir = "data/test_documents"
    os.makedirs(test_dir, exist_ok=True)
    
    # Initialize repository
    repo = DocumentRepository(base_dir=test_dir)
    logger.info("Initialized document repository")
    
    # Test document storage
    test_documents = [
        {
            "content": "This is a test article about political events. The president announced new policies today.",
            "metadata": {
                "title": "President Announces New Policies",
                "source": "Test News",
                "url": "https://testnews.example.com/politics/12345",
                "published": datetime.datetime.now().isoformat(),
                "bias_label": "center"
            }
        },
        {
            "content": "Opposition leaders criticized the new government initiatives as overreaching and potentially harmful.",
            "metadata": {
                "title": "Opposition Criticizes New Initiatives",
                "source": "Another Source",
                "url": "https://anothersource.example.com/politics/54321",
                "published": datetime.datetime.now().isoformat(),
                "bias_label": "center-right"
            }
        },
        {
            "content": "Protesters gathered outside the capital building to voice concerns about recent government actions.",
            "metadata": {
                "title": "Protests Erupt Over Government Actions",
                "source": "Daily News",
                "url": "https://dailynews.example.com/local/78901",
                "published": datetime.datetime.now().isoformat(),
                "bias_label": "center-left"
            }
        }
    ]
    
    # Store each document
    doc_ids = []
    for doc in test_documents:
        doc_id = repo.store_document(doc["content"], doc["metadata"])
        doc_ids.append(doc_id)
        logger.info(f"Stored document: {doc['metadata']['title']} with ID: {doc_id}")
    
    # Test document retrieval
    logger.info("\nTesting document retrieval:")
    for doc_id in doc_ids:
        content, metadata = repo.get_document(doc_id)
        if content and metadata:
            logger.info(f"Retrieved document: {metadata['title']}")
            logger.info(f"Content preview: {content[:50]}...")
        else:
            logger.error(f"Failed to retrieve document {doc_id}")
    
    # Test document listing
    all_docs = repo.list_documents()
    logger.info(f"\nAll documents in repository ({len(all_docs)}):")
    for doc_id in all_docs:
        logger.info(f"  - {doc_id}")
    
    # Test metadata search
    logger.info("\nTesting metadata search:")
    center_docs = repo.search_by_metadata({"bias_label": "center"})
    logger.info(f"Documents with center bias: {len(center_docs)}")
    for doc_id in center_docs:
        _, metadata = repo.get_document(doc_id)
        if metadata:
            logger.info(f"  - {metadata['title']}")
    
    # Test document citation
    logger.info("\nTesting document citation:")
    for doc_id in doc_ids:
        citation = repo.get_document_citation(doc_id)
        logger.info(f"Citation: {citation}")
    
    # Test integrity verification
    logger.info("\nTesting document integrity verification:")
    for doc_id in doc_ids:
        is_valid = repo.verify_document_integrity(doc_id)
        logger.info(f"Document {doc_id}: {'Valid' if is_valid else 'INVALID'}")
    
    # Test batch operations
    logger.info("\nTesting batch operations:")
    batch_docs = [
        ("This is batch document 1.", {"title": "Batch 1", "source": "Batch Test"}),
        ("This is batch document 2.", {"title": "Batch 2", "source": "Batch Test"}),
    ]
    batch_ids = repo.batch_store_documents(batch_docs)
    logger.info(f"Stored {len(batch_ids)} documents in batch operation")
    
    # Test statistics
    logger.info("\nRepository statistics:")
    stats = repo.get_statistics()
    logger.info(f"Total documents: {stats['total_documents']}")
    logger.info(f"Total content size: {stats['content_size_bytes']} bytes")
    logger.info("Sources distribution:")
    for source, count in stats['sources'].items():
        logger.info(f"  - {source}: {count} documents")
    
    logger.info("\nTest completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
