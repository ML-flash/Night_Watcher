#!/usr/bin/env python3
"""
Night_watcher Analyzer
Script to run the analyzer component of the Night_watcher framework.
"""

import os
import sys
import logging
import argparse
import json
import glob
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Import local modules
from analyzer import ContentAnalyzer
from document_repository import DocumentRepository
from providers import initialize_llm_provider
from analysis_provenance import AnalysisProvenanceTracker

# Default configuration
DEFAULT_CONFIG = {
    "content_analysis": {
        "max_articles": 10
    },
    "llm_provider": {
        "type": "lm_studio",
        "host": "http://localhost:1234"
    },
    "output": {
        "base_dir": "data",
        "save_analyzed": True
    },
    "provenance": {
        "enabled": True,
        "dev_mode": True
    }
}


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from file with fallback to defaults.
    """
    if not os.path.exists(config_path):
        logging.warning(f"Configuration file {config_path} not found. Using defaults.")
        return DEFAULT_CONFIG

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            user_cfg = json.load(f)

        # Deep merge with defaults
        merged = DEFAULT_CONFIG.copy()

        def deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> None:
            for k, v in update.items():
                if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                    deep_update(base[k], v)
                else:
                    base[k] = v

        deep_update(merged, user_cfg)
        return merged
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        return DEFAULT_CONFIG


def get_unanalyzed_documents(document_repo: DocumentRepository, analyzed_dir: str, max_count: int) -> List[Tuple[str, str, Dict]]:
    """
    Get documents that haven't been analyzed yet.
    
    Args:
        document_repo: Document repository
        analyzed_dir: Directory containing analyzed documents
        max_count: Maximum number of documents to return
        
    Returns:
        List of tuples (document_id, content, metadata)
    """
    # Get all document IDs from repository
    doc_ids = document_repo.list_documents()
    
    # Get IDs of already analyzed documents
    analyzed_files = glob.glob(os.path.join(analyzed_dir, "analysis_*.json"))
    analyzed_ids = set()
    
    for filepath in analyzed_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                analysis = json.load(f)
                doc_id = analysis.get("article", {}).get("document_id")
                if doc_id:
                    analyzed_ids.add(doc_id)
        except Exception as e:
            logging.error(f"Error reading analyzed file {filepath}: {e}")
    
    # Filter out already analyzed documents
    unanalyzed_ids = [doc_id for doc_id in doc_ids if doc_id not in analyzed_ids]
    
    # Limit to max_count
    unanalyzed_ids = unanalyzed_ids[:max_count]
    
    # Get document contents and metadata
    unanalyzed_docs = []
    for doc_id in unanalyzed_ids:
        try:
            # Updated to handle the third return value (verified)
            content, metadata, verified = document_repo.get_document(doc_id)
            
            if content and metadata:
                unanalyzed_docs.append((doc_id, content, metadata))
            else:
                logging.warning(f"Document {doc_id} content or metadata not found")
        except Exception as e:
            logging.error(f"Error getting document {doc_id}: {e}")
    
    logging.info(f"Found {len(unanalyzed_docs)} unanalyzed documents")
    return unanalyzed_docs


def run_analyzer(config: Dict[str, Any], args: argparse.Namespace) -> int:
    """
    Run the analyzer with given configuration and arguments.
    """
    # Set up output directories
    output_dir = args.output_dir or config["output"].get("base_dir", "data")
    analyzed_dir = os.path.join(output_dir, "analyzed")
    document_repo_dir = os.path.join(output_dir, "documents")
    
    os.makedirs(analyzed_dir, exist_ok=True)
    
    # Determine if provenance is enabled
    provenance_enabled = config.get("provenance", {}).get("enabled", True)
    if args.disable_provenance:
        provenance_enabled = False
    
    dev_mode = config.get("provenance", {}).get("dev_mode", True)
    
    # Set up document repository
    doc_repo = DocumentRepository(
        base_dir=document_repo_dir,
        dev_passphrase=args.provenance_passphrase,
        dev_mode=dev_mode
    )
    
    # Initialize LLM provider
    llm_provider = initialize_llm_provider(config)
    
    if not llm_provider:
        logging.error("Failed to initialize LLM provider")
        return 1
    
    # Set up analyzer
    analyzer = ContentAnalyzer(llm_provider)
    
    # Set up provenance tracker if enabled
    provenance_tracker = None
    if provenance_enabled:
        provenance_dir = os.path.join(output_dir, "analysis_provenance")
        
        provenance_tracker = AnalysisProvenanceTracker(
            base_dir=provenance_dir,
            dev_passphrase=args.provenance_passphrase,
            dev_mode=dev_mode
        )
    
    # Get unanalyzed documents
    max_articles = args.max_articles or config["content_analysis"].get("max_articles", 10)
    documents = get_unanalyzed_documents(doc_repo, analyzed_dir, max_articles)
    
    if not documents:
        logging.info("No unanalyzed documents found")
        return 0
    
    # Process documents
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.info(f"Starting analysis of {len(documents)} documents at {timestamp}")
    
    articles = []
    doc_ids = []
    
    # Convert documents to article format
    for doc_id, content, metadata in documents:
        article = {
            "title": metadata.get("title", "Untitled"),
            "content": content,
            "url": metadata.get("url", ""),
            "source": metadata.get("source", "Unknown"),
            "bias_label": metadata.get("bias_label", "unknown"),
            "published": metadata.get("published"),
            "document_id": doc_id
        }
        
        articles.append(article)
        doc_ids.append(doc_id)
    
    # Run analysis
    analysis_input = {
        "articles": articles,
        "document_ids": doc_ids
    }
    
    result = analyzer.process(analysis_input)
    analyses = result.get("analyses", [])
    
    # Save analysis results
    for i, analysis in enumerate(analyses):
        article_info = analysis.get("article", {})
        doc_id = article_info.get("document_id", f"unknown_{i}")
        
        # Generate a filename
        filename = f"analysis_{doc_id}_{timestamp}.json"
        filepath = os.path.join(analyzed_dir, filename)
        
        # Create provenance record if enabled
        if provenance_enabled and provenance_tracker:
            try:
                # Extract analysis parameters
                analysis_params = {
                    "timestamp": timestamp,
                    "analyzer_version": "1.0.0"
                }
                
                # Create provenance record
                provenance_id = f"analysis_{doc_id}_{timestamp}"
                provenance_record = provenance_tracker.create_analysis_record(
                    analysis_id=provenance_id,
                    document_ids=[doc_id],
                    analysis_type="content_analysis",
                    analysis_parameters=analysis_params,
                    results=analysis,
                    analyzer_version="1.0.0"
                )
                
                # Add provenance ID to analysis
                analysis["provenance_id"] = provenance_id
                
            except Exception as e:
                logging.error(f"Error creating provenance record for {doc_id}: {e}")
        
        # Save analysis to file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            logging.info(f"Saved analysis for {article_info.get('title')} to {filename}")
        except Exception as e:
            logging.error(f"Error saving analysis for {doc_id}: {e}")
    
    # Print summary
    print(f"\n=== Analysis Summary ===")
    print(f"Documents analyzed: {len(analyses)}")
    print(f"Total documents found: {len(documents)}")
    print(f"Timestamp: {timestamp}")
    
    if provenance_enabled:
        print(f"Provenance tracking: Enabled")
    else:
        print(f"Provenance tracking: Disabled")
    
    print(f"\nOutputs saved to: {analyzed_dir}")
    
    return 0


def main() -> int:
    """
    Main entry point.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Night_watcher Analyzer")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    parser.add_argument("--max-articles", type=int, help="Maximum articles to analyze")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    # Add provenance options
    parser.add_argument("--provenance-passphrase", help="Passphrase for analysis provenance (dev mode)")
    parser.add_argument("--disable-provenance", action="store_true", help="Disable provenance tracking")
    
    args = parser.parse_args()
    
    # Create necessary directories
    log_dir = "logs"
    data_dir = args.output_dir or "data"
    analyzed_dir = os.path.join(data_dir, "analyzed")
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(analyzed_dir, exist_ok=True)
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_file = os.path.join(log_dir, f"analyzer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Load configuration
    config = load_config(args.config)
    
    print("""
    ╔═══════════════════════════════════════════════════╗
    ║      Night_watcher Analyzer Component             ║
    ╚═══════════════════════════════════════════════════╝
    """)
    
    # If provenance passphrase is not provided but environment variable exists, use it
    if not args.provenance_passphrase:
        env_passphrase = os.environ.get("NIGHT_WATCHER_PASSPHRASE")
        if env_passphrase:
            args.provenance_passphrase = env_passphrase
            logging.info("Using provenance passphrase from environment variable")
    
    # Run analyzer
    return run_analyzer(config, args)


if __name__ == "__main__":
    sys.exit(main())