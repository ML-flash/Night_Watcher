#!/usr/bin/env python3
"""
Night_watcher Knowledge Graph Builder
Stand-alone tool to build and maintain the knowledge graph from analysis outputs.
"""

import os
import sys
import glob
import json
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union

from knowledge_graph import KnowledgeGraph

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("build_knowledge_graph")

def load_analyzed_files(analyzed_dir: str, pattern: str = "kg_analysis_*.json") -> List[Dict[str, Any]]:
    """
    Load KG analysis files from the analyzed directory.
    
    Args:
        analyzed_dir: Directory containing analysis outputs
        pattern: File pattern to match
        
    Returns:
        List of loaded analysis data
    """
    analyses = []
    
    # Find matching files
    file_pattern = os.path.join(analyzed_dir, pattern)
    matching_files = glob.glob(file_pattern)
    
    logger.info(f"Found {len(matching_files)} analysis files matching pattern {pattern}")
    
    # Load each file
    for filepath in matching_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                analysis = json.load(f)
                analyses.append(analysis)
                logger.debug(f"Loaded analysis from {filepath}")
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            
    return analyses

def process_analyses(kg: KnowledgeGraph, analyses: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Process analyses and add to knowledge graph.
    
    Args:
        kg: Knowledge graph instance
        analyses: List of analysis data
        
    Returns:
        Dictionary with processing stats
    """
    total_nodes = 0
    total_edges = 0
    total_articles = 0
    
    for analysis in analyses:
        article = analysis.get("article", {})
        kg_payload = analysis.get("kg_payload", {})
        
        if not article or not kg_payload:
            logger.warning("Analysis missing article or KG payload")
            continue
            
        # Process analysis
        result = kg.process_article_analysis(article, analysis)
        
        # Update counts
        total_nodes += result.get("nodes_added", 0)
        total_edges += result.get("edges_added", 0)
        total_articles += 1
        
        logger.info(f"Processed article: {article.get('title', 'Unknown')} - Added {result.get('nodes_added', 0)} nodes, {result.get('edges_added', 0)} edges")
        
    # Infer temporal relationships
    temporal_relations = kg.infer_temporal_relationships()
    logger.info(f"Inferred {temporal_relations} temporal relationships")
    
    return {
        "articles_processed": total_articles,
        "nodes_added": total_nodes,
        "edges_added": total_edges,
        "temporal_relations": temporal_relations
    }

def generate_intelligence_reports(kg: KnowledgeGraph, output_dir: str) -> None:
    """
    Generate intelligence reports from the knowledge graph.
    
    Args:
        kg: Knowledge graph instance
        output_dir: Directory to save reports
    """
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate full intelligence report
    intel_report = kg.generate_intelligence_report()
    intel_report_path = os.path.join(output_dir, f"intelligence_report_{timestamp}.json")
    
    with open(intel_report_path, 'w', encoding='utf-8') as f:
        json.dump(intel_report, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Generated intelligence report: {intel_report_path}")
    
    # Generate authoritarian trends report
    auth_trends = kg.get_authoritarian_trends(days=90)
    auth_trends_path = os.path.join(output_dir, f"authoritarian_trends_{timestamp}.json")
    
    with open(auth_trends_path, 'w', encoding='utf-8') as f:
        json.dump(auth_trends, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Generated authoritarian trends report: {auth_trends_path}")
    
    # Generate democratic erosion report
    erosion = kg.analyze_democratic_erosion(days=90)
    erosion_path = os.path.join(output_dir, f"democratic_erosion_{timestamp}.json")
    
    with open(erosion_path, 'w', encoding='utf-8') as f:
        json.dump(erosion, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Generated democratic erosion report: {erosion_path}")
    
    # Generate network visualization data
    viz_data = kg.visualize_network(
        output_file=os.path.join(output_dir, f"network_visualization_{timestamp}.json")
    )
    
    logger.info(f"Generated network visualization data: {len(viz_data['nodes'])} nodes, {len(viz_data['edges'])} edges")

def main() -> int:
    parser = argparse.ArgumentParser(description="Night_watcher Knowledge Graph Builder")
    parser.add_argument("--kg-file", default="data/knowledge_graph/graph.json",
                        help="Knowledge graph file (JSON format)")
    parser.add_argument("--taxonomy-file", default="KG_Taxonomy.csv",
                        help="Taxonomy file (CSV format)")
    parser.add_argument("--analyzed-dir", default="data/analyzed",
                        help="Directory containing analysis outputs")
    parser.add_argument("--output-dir", default="data/analysis",
                        help="Output directory for reports")
    parser.add_argument("--file-pattern", default="kg_analysis_*.json",
                        help="Pattern to match analysis files")
    parser.add_argument("--skip-reports", action="store_true",
                        help="Skip generating reports")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize knowledge graph
    logger.info(f"Initializing knowledge graph from {args.kg_file}")
    kg = KnowledgeGraph(graph_file=args.kg_file, taxonomy_file=args.taxonomy_file)
    
    # Load analyses
    logger.info(f"Loading analyses from {args.analyzed_dir}")
    analyses = load_analyzed_files(args.analyzed_dir, args.file_pattern)
    
    if not analyses:
        logger.warning("No analyses found")
        return 1
        
    # Process analyses
    logger.info(f"Processing {len(analyses)} analyses")
    stats = process_analyses(kg, analyses)
    
    # Generate reports unless skipped
    if not args.skip_reports:
        logger.info("Generating intelligence reports")
        generate_intelligence_reports(kg, args.output_dir)
    
    # Print summary
    print("\n=== Knowledge Graph Builder Summary ===")
    print(f"Articles processed: {stats['articles_processed']}")
    print(f"Nodes added: {stats['nodes_added']}")
    print(f"Edges added: {stats['edges_added']}")
    print(f"Temporal relations: {stats['temporal_relations']}")
    print(f"Total KG size: {len(kg.graph.nodes)} nodes, {len(kg.graph.edges)} edges")
    print(f"Knowledge graph file: {args.kg_file}")
    print(f"Reports saved to: {args.output_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
