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
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import local modules
from analyzer import ContentAnalyzer
from document_repository import DocumentRepository
from providers import initialize_llm_provider

# Default configuration
DEFAULT_CONFIG = {
    "llm_provider": {
        "type": "lm_studio",
        "host": "http://localhost:1234",
        "model": "default"
    },
    "analysis": {
        "max_articles": 10,
        "include_kg": True
    },
    "output": {
        "base_dir": "data",
        "save_analyses": True
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

def save_to_file(content: Any, filepath: str) -> bool:
    """
    Save content to file.
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if isinstance(content, (dict, list)):
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(content, f, indent=2, ensure_ascii=False)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(str(content) if content is not None else "No content")
                
        return True
    except Exception as e:
        logging.error(f"Error saving to {filepath}: {e}")
        return False

def get_unanalyzed_documents(document_repo: DocumentRepository, 
                             analyzed_dir: str,
                             max_count: int = 10) -> List[Dict[str, Any]]:
    """
    Get documents that haven't been analyzed yet.
    
    Args:
        document_repo: Document repository
        analyzed_dir: Directory containing analysis results
        max_count: Maximum number of documents to return
        
    Returns:
        List of document dictionaries
    """
    # Get all document IDs from repository
    all_doc_ids = document_repo.list_documents()
    
    # Get IDs of already analyzed documents
    analyzed_ids = []
    if os.path.exists(analyzed_dir):
        for filename in os.listdir(analyzed_dir):
            if filename.endswith(".json"):
                try:
                    with open(os.path.join(analyzed_dir, filename), 'r') as f:
                        analysis = json.load(f)
                        if analysis and "article" in analysis and "document_id" in analysis["article"]:
                            analyzed_ids.append(analysis["article"]["document_id"])
                except Exception as e:
                    logging.error(f"Error reading analysis file {filename}: {e}")
    
    # Get IDs of documents to analyze
    unanalyzed_ids = [doc_id for doc_id in all_doc_ids if doc_id not in analyzed_ids]
    
    # Limit to max_count
    doc_ids_to_analyze = unanalyzed_ids[:max_count]
    
    # Get document content and metadata
    documents = []
    for doc_id in doc_ids_to_analyze:
        content, metadata = document_repo.get_document(doc_id)
        if content and metadata:
            documents.append({
                "content": content,
                "document_id": doc_id,
                "title": metadata.get("title", ""),
                "url": metadata.get("url", ""),
                "source": metadata.get("source", ""),
                "published": metadata.get("published", ""),
                "bias_label": metadata.get("bias_label", "")
            })
    
    return documents

def run_analyzer(config: Dict[str, Any], args: argparse.Namespace) -> int:
    """
    Run the analyzer with given configuration and arguments.
    """
    # Set up output directories
    output_dir = args.output_dir or config["output"].get("base_dir", "data")
    analyzed_dir = os.path.join(output_dir, "analyzed")
    document_repo_dir = os.path.join(output_dir, "documents")
    
    os.makedirs(analyzed_dir, exist_ok=True)
    
    # Set up document repository
    doc_repo = DocumentRepository(base_dir=document_repo_dir)
    
    # Initialize LLM provider
    llm_provider = initialize_llm_provider(config)
    if not llm_provider:
        logging.error("Failed to initialize LLM provider. Cannot continue with analysis.")
        return 1
    
    # Set up analyzer
    analyzer = ContentAnalyzer(llm_provider)
    
    # Get documents to analyze
    max_articles = args.max_articles or config["analysis"].get("max_articles", 10)
    documents = []
    
    if args.document_id:
        # Analyze specific document
        content, metadata = doc_repo.get_document(args.document_id)
        if content and metadata:
            documents.append({
                "content": content,
                "document_id": args.document_id,
                "title": metadata.get("title", ""),
                "url": metadata.get("url", ""),
                "source": metadata.get("source", ""),
                "published": metadata.get("published", ""),
                "bias_label": metadata.get("bias_label", "")
            })
        else:
            logging.error(f"Document with ID {args.document_id} not found")
            return 1
    else:
        # Get unanalyzed documents
        documents = get_unanalyzed_documents(doc_repo, analyzed_dir, max_articles)
    
    if not documents:
        logging.info("No documents to analyze")
        return 0
    
    # Run analysis
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.info(f"Starting analysis at {timestamp} for {len(documents)} documents")
    
    result = analyzer.process({"articles": documents})
    
    # Get analysis results
    analyses = result.get("analyses", [])
    
    # Save analysis results
    for i, analysis in enumerate(analyses):
        article = analysis.get("article", {})
        doc_id = article.get("document_id", f"unknown_{i}")
        
        # Save analysis to file
        filename = f"analysis_{doc_id}_{timestamp}.json"
        save_to_file(analysis, os.path.join(analyzed_dir, filename))
        
        # Save prompt chain if requested
        if args.save_prompts:
            prompt_chain = analysis.get("prompt_chain", [])
            if prompt_chain:
                chain_text = []
                for round_data in prompt_chain:
                    chain_text.append(f"=== ROUND {round_data.get('round')}: {round_data.get('name')} ===\n")
                    chain_text.append("-- PROMPT --\n")
                    chain_text.append(f"{round_data.get('prompt', '')}\n\n")
                    chain_text.append("-- RESPONSE --\n")
                    chain_text.append(f"{round_data.get('response', '')}\n\n")
                
                prompt_dir = os.path.join(output_dir, "prompts")
                os.makedirs(prompt_dir, exist_ok=True)
                prompt_file = os.path.join(prompt_dir, f"prompts_{doc_id}_{timestamp}.txt")
                with open(prompt_file, 'w', encoding='utf-8') as f:
                    f.write("".join(chain_text))
    
    # Print summary
    print(f"\n=== Analysis Summary ===")
    print(f"Timestamp: {timestamp}")
    print(f"Documents analyzed: {len(analyses)}")
    
    # Print analysis stats
    total_nodes = 0
    total_edges = 0
    concern_levels = {}
    
    for analysis in analyses:
        # Count nodes and edges
        kg_payload = analysis.get("kg_payload", {})
        nodes = kg_payload.get("nodes", [])
        edges = kg_payload.get("edges", [])
        total_nodes += len(nodes)
        total_edges += len(edges)
        
        # Track concern levels
        concern_level = analysis.get("concern_level", "Unknown")
        concern_levels[concern_level] = concern_levels.get(concern_level, 0) + 1
    
    print(f"\n=== Knowledge Graph Statistics ===")
    print(f"Total nodes extracted: {total_nodes}")
    print(f"Total edges extracted: {total_edges}")
    
    print(f"\n=== Authoritarian Analysis ===")
    for level, count in concern_levels.items():
        print(f"  - {level}: {count} documents")
    
    print(f"\nAnalyses saved to: {analyzed_dir}")
    
    return 0

def main() -> int:
    """
    Main entry point.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Night_watcher Analyzer")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--document-id", help="Analyze a specific document by ID")
    parser.add_argument("--max-articles", type=int, help="Maximum articles to analyze")
    parser.add_argument("--save-prompts", action="store_true", help="Save prompt chains to files")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
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
    
    # Run the analyzer
    return run_analyzer(config, args)

if __name__ == "__main__":
    sys.exit(main())
