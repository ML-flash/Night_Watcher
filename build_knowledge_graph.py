#!/usr/bin/env python3
"""
Night_watcher Knowledge Graph Builder
Builds and updates the knowledge graph from analyzed articles.
"""

import os
import sys
import json
import logging
import glob
import re
from datetime import datetime

# Ensure knowledge_graph module can be found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from knowledge_graph import KnowledgeGraph

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("build_knowledge_graph")

def load_and_process_analyses(kg: KnowledgeGraph, analyses_dir: str, pattern: str = "analysis_*.json") -> int:
    """
    Load analysis files and add nodes and edges to the knowledge graph.
    
    Args:
        kg: Knowledge graph instance
        analyses_dir: Directory containing analysis files
        pattern: File pattern to match
        
    Returns:
        Number of analyses processed
    """
    if not os.path.exists(analyses_dir):
        logger.error(f"Analyses directory not found: {analyses_dir}")
        return 0
    
    # Find all analysis files
    file_pattern = os.path.join(analyses_dir, pattern)
    files = glob.glob(file_pattern)
    
    logger.info(f"Found {len(files)} analysis files matching pattern {pattern}")
    
    if not files:
        logger.warning("No analyses found")
        return 0
    
    processed_count = 0
    
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                analysis = json.load(f)
            
            # Check if this is a kg_analysis or a regular analysis
            # KG analysis has kg_payload directly
            # Regular analysis might have it nested under structured_facts and analysis
            kg_payload = None
            
            if "kg_payload" in analysis:
                # Direct KG payload
                kg_payload = analysis["kg_payload"]
            elif "prompt_chain" in analysis:
                # Extract KG payload from prompt chain
                # Look for Node Extraction, Edge Extraction rounds
                nodes = []
                edges = []
                
                for round_data in analysis.get("prompt_chain", []):
                    round_name = round_data.get("name", "")
                    
                    if "Node Extraction" in round_name or "Edge Extraction" in round_name:
                        # Try to extract JSON from the response
                        response = round_data.get("response", "")
                        
                        # Look for JSON arrays in the response
                        json_matches = re.findall(r'\[\s*{.*}\s*\]', response, re.DOTALL)
                        
                        if json_matches:
                            for json_str in json_matches:
                                try:
                                    parsed_data = json.loads(json_str)
                                    
                                    # Determine if this is nodes or edges
                                    if isinstance(parsed_data, list) and parsed_data:
                                        if "node_type" in parsed_data[0]:
                                            nodes.extend(parsed_data)
                                        elif "relation" in parsed_data[0]:
                                            edges.extend(parsed_data)
                                except json.JSONDecodeError:
                                    continue
                
                if nodes or edges:
                    kg_payload = {
                        "nodes": nodes,
                        "edges": edges
                    }
            
            # Skip if no KG payload found
            if not kg_payload:
                logger.warning(f"No knowledge graph payload found in {file_path}")
                continue
            
            # Process the article info
            article = analysis.get("article", {})
            
            # Process the knowledge graph payload
            success = kg.process_article_analysis(article, {"kg_payload": kg_payload})
            
            if success:
                processed_count += 1
                logger.info(f"Processed analysis from {os.path.basename(file_path)}")
            else:
                logger.warning(f"Failed to process analysis from {os.path.basename(file_path)}")
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    return processed_count

def main():
    """Main entry point"""
    # Configuration
    graph_file = "data/knowledge_graph/graph.json"
    taxonomy_file = "KG_Taxonomy.csv"
    analyses_dir = "data/analyzed"
    
    # Initialize knowledge graph
    logger.info(f"Initializing knowledge graph from {graph_file}")
    kg = KnowledgeGraph(graph_file=graph_file, taxonomy_file=taxonomy_file)
    
    # Load analyses
    logger.info(f"Loading analyses from {analyses_dir}")
    
    # Try multiple patterns to find analyses
    patterns = [
        "kg_analysis_*.json",  # Custom KG analyses
        "analysis_*.json"      # Regular analyses from analyzer.py
    ]
    
    total_processed = 0
    
    for pattern in patterns:
        processed = load_and_process_analyses(kg, analyses_dir, pattern)
        total_processed += processed
        
        if processed > 0:
            logger.info(f"Processed {processed} analyses with pattern {pattern}")
    
    if total_processed == 0:
        logger.warning("No analyses found or processed")
        return 1
    
    # Save the graph
    kg.save_graph()
    logger.info(f"Graph saved to: {graph_file}")
    
    # Save a snapshot
    snapshot_id = kg.save_snapshot(name=f"Graph updated from {total_processed} analyses")
    logger.info(f"Snapshot created: {snapshot_id}")
    
    # Get statistics
    stats = kg.get_basic_statistics()
    logger.info(f"Graph now contains {stats['node_count']} nodes and {stats['edge_count']} edges")
    logger.info(f"Node types: {list(stats['node_types'].keys())}")
    logger.info(f"Relation types: {list(stats['relation_types'].keys())}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())