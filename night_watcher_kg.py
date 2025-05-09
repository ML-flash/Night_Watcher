#!/usr/bin/env python3
"""
Night_watcher Knowledge Graph Controller
Command-line interface for the Night_watcher Knowledge Graph component.
"""

import os
import sys
import glob
import json
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from knowledge_graph import KnowledgeGraph

# ==========================================
# Configuration Functions
# ==========================================

DEFAULT_CONFIG = {
    "knowledge_graph": {
        "graph_file": "data/knowledge_graph/graph.json",
        "taxonomy_file": "KG_Taxonomy.csv"
    },
    "input": {
        "analyzed_dir": "data/analyzed",
        "file_pattern": "kg_analysis_*.json"
    },
    "output": {
        "reports_dir": "data/analysis",
        "save_visualizations": True
    },
    "analysis": {
        "trend_days": 90,
        "actor_limit": 10
    },
    "logging": {
        "level": "INFO",
        "log_dir": "logs"
    }
}


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file with fallback to defaults"""
    if not os.path.exists(config_path):
        logging.warning(f"Configuration file {config_path} not found. Using defaults.")
        return DEFAULT_CONFIG

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            user_cfg = json.load(f)
        merged = DEFAULT_CONFIG.copy()
        
        def deep_update(b: Dict[str, Any], u: Dict[str, Any]) -> None:
            for k, v in u.items():
                if isinstance(v, dict) and k in b and isinstance(b[k], dict):
                    deep_update(b[k], v)
                else:
                    b[k] = v
                    
        deep_update(merged, user_cfg)
        return merged
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        return DEFAULT_CONFIG


def create_default_config(config_path: str) -> bool:
    """Create default configuration file"""
    try:
        os.makedirs(os.path.dirname(config_path) or '.', exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(DEFAULT_CONFIG, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logging.error(f"Error saving default config: {e}")
        return False


# ==========================================
# Core Functions
# ==========================================

def load_analyzed_files(analyzed_dir: str, pattern: str = "kg_analysis_*.json") -> List[Dict[str, Any]]:
    """Load KG analysis files from the analyzed directory"""
    analyses = []
    
    # Find matching files
    file_pattern = os.path.join(analyzed_dir, pattern)
    matching_files = glob.glob(file_pattern)
    
    logging.info(f"Found {len(matching_files)} analysis files matching pattern {pattern}")
    
    # Load each file
    for filepath in matching_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                analysis = json.load(f)
                analyses.append(analysis)
                logging.debug(f"Loaded analysis from {filepath}")
        except Exception as e:
            logging.error(f"Error loading {filepath}: {e}")
            
    return analyses


def process_analyses(kg: KnowledgeGraph, analyses: List[Dict[str, Any]]) -> Dict[str, int]:
    """Process analyses and add to knowledge graph"""
    total_nodes = 0
    total_edges = 0
    total_articles = 0
    
    for analysis in analyses:
        article = analysis.get("article", {})
        kg_payload = analysis.get("kg_payload", {})
        
        if not article or not kg_payload:
            logging.warning("Analysis missing article or KG payload")
            continue
            
        # Process analysis
        result = kg.process_article_analysis(article, analysis)
        
        # Update counts
        total_nodes += result.get("nodes_added", 0)
        total_edges += result.get("edges_added", 0)
        total_articles += 1
        
        logging.info(f"Processed article: {article.get('title', 'Unknown')} - Added {result.get('nodes_added', 0)} nodes, {result.get('edges_added', 0)} edges")
        
    # Infer temporal relationships
    temporal_relations = kg.infer_temporal_relationships()
    logging.info(f"Inferred {temporal_relations} temporal relationships")
    
    return {
        "articles_processed": total_articles,
        "nodes_added": total_nodes,
        "edges_added": total_edges,
        "temporal_relations": temporal_relations
    }


def generate_reports(kg: KnowledgeGraph, output_dir: str, trend_days: int = 90, save_viz: bool = True) -> Dict[str, str]:
    """Generate intelligence reports from the knowledge graph"""
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Track generated files
    generated_files = {}
    
    # Generate full intelligence report
    intel_report = kg.generate_intelligence_report()
    intel_report_path = os.path.join(output_dir, f"intelligence_report_{timestamp}.json")
    
    with open(intel_report_path, 'w', encoding='utf-8') as f:
        json.dump(intel_report, f, indent=2, ensure_ascii=False)
        
    logging.info(f"Generated intelligence report: {intel_report_path}")
    generated_files["intelligence_report"] = intel_report_path
    
    # Generate authoritarian trends report
    auth_trends = kg.get_authoritarian_trends(days=trend_days)
    auth_trends_path = os.path.join(output_dir, f"authoritarian_trends_{timestamp}.json")
    
    with open(auth_trends_path, 'w', encoding='utf-8') as f:
        json.dump(auth_trends, f, indent=2, ensure_ascii=False)
        
    logging.info(f"Generated authoritarian trends report: {auth_trends_path}")
    generated_files["authoritarian_trends"] = auth_trends_path
    
    # Generate democratic erosion report
    erosion = kg.analyze_democratic_erosion(days=trend_days)
    erosion_path = os.path.join(output_dir, f"democratic_erosion_{timestamp}.json")
    
    with open(erosion_path, 'w', encoding='utf-8') as f:
        json.dump(erosion, f, indent=2, ensure_ascii=False)
        
    logging.info(f"Generated democratic erosion report: {erosion_path}")
    generated_files["democratic_erosion"] = erosion_path
    
    # Generate influential actors report
    actors = kg.get_influential_actors(limit=10)
    actors_path = os.path.join(output_dir, f"influential_actors_{timestamp}.json")
    
    with open(actors_path, 'w', encoding='utf-8') as f:
        json.dump(actors, f, indent=2, ensure_ascii=False)
        
    logging.info(f"Generated influential actors report: {actors_path}")
    generated_files["influential_actors"] = actors_path
    
    # Generate network visualization data if enabled
    if save_viz:
        viz_path = os.path.join(output_dir, f"network_visualization_{timestamp}.json")
        viz_data = kg.visualize_network(output_file=viz_path)
        
        logging.info(f"Generated network visualization data: {viz_path}")
        generated_files["network_visualization"] = viz_path
    
    return generated_files


# ==========================================
# Main Execution Logic
# ==========================================

def run_kg_workflow(config: Dict[str, Any]) -> int:
    """Run the knowledge graph workflow"""
    # Extract configuration values
    kg_config = config["knowledge_graph"]
    input_config = config["input"]
    output_config = config["output"]
    analysis_config = config["analysis"]
    
    graph_file = kg_config["graph_file"]
    taxonomy_file = kg_config["taxonomy_file"]
    analyzed_dir = input_config["analyzed_dir"]
    file_pattern = input_config["file_pattern"]
    reports_dir = output_config["reports_dir"]
    save_viz = output_config["save_visualizations"]
    trend_days = analysis_config["trend_days"]
    
    # Initialize knowledge graph
    logging.info(f"Initializing knowledge graph from {graph_file}")
    kg = KnowledgeGraph(graph_file=graph_file, taxonomy_file=taxonomy_file)
    
    # Show initial graph stats
    initial_nodes = len(kg.graph.nodes)
    initial_edges = len(kg.graph.edges)
    logging.info(f"Initial graph state: {initial_nodes} nodes, {initial_edges} edges")
    
    # Load analyses
    logging.info(f"Loading analyses from {analyzed_dir}")
    analyses = load_analyzed_files(analyzed_dir, file_pattern)
    
    if not analyses:
        logging.warning("No analyses found")
        return 1
        
    # Process analyses
    logging.info(f"Processing {len(analyses)} analyses")
    stats = process_analyses(kg, analyses)
    
    # Generate reports
    logging.info("Generating intelligence reports")
    generated_files = generate_reports(kg, reports_dir, trend_days, save_viz)
    
    # Calculate metrics
    final_nodes = len(kg.graph.nodes)
    final_edges = len(kg.graph.edges)
    
    # Print summary
    print("\n=== Night_watcher Knowledge Graph Summary ===")
    print(f"Articles processed: {stats['articles_processed']}")
    print(f"Nodes added: {stats['nodes_added']}")
    print(f"Edges added: {stats['edges_added']}")
    print(f"Temporal relations added: {stats['temporal_relations']}")
    print(f"Knowledge Graph size: {final_nodes} nodes, {final_edges} edges")
    print(f"Node growth: {final_nodes - initial_nodes} nodes added")
    print(f"Edge growth: {final_edges - initial_edges} edges added")
    print("\nReports generated:")
    for report_name, file_path in generated_files.items():
        print(f"- {report_name}: {os.path.basename(file_path)}")
    print(f"\nAll files saved to: {reports_dir}")
    
    return 0


def main() -> int:
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Night_watcher Knowledge Graph Controller")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    parser.add_argument("--graph-file", help="Override path to knowledge graph file")
    parser.add_argument("--taxonomy-file", help="Override path to taxonomy file")
    parser.add_argument("--analyzed-dir", help="Override path to analyzed directory")
    parser.add_argument("--file-pattern", help="Override file pattern for analysis files")
    parser.add_argument("--reports-dir", help="Override path to reports directory")
    parser.add_argument("--trend-days", type=int, help="Override number of days for trend analysis")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization generation")
    parser.add_argument("--create-config", action="store_true", help="Create default config file and exit")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Create default config if requested
    if args.create_config:
        if create_default_config(args.config):
            print(f"Created default configuration file: {args.config}")
            return 0
        else:
            print(f"Failed to create configuration file: {args.config}")
            return 1
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_dir = DEFAULT_CONFIG["logging"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"night_watcher_kg_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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
    
    # Override with command line arguments
    if args.graph_file:
        config["knowledge_graph"]["graph_file"] = args.graph_file
    if args.taxonomy_file:
        config["knowledge_graph"]["taxonomy_file"] = args.taxonomy_file
    if args.analyzed_dir:
        config["input"]["analyzed_dir"] = args.analyzed_dir
    if args.file_pattern:
        config["input"]["file_pattern"] = args.file_pattern
    if args.reports_dir:
        config["output"]["reports_dir"] = args.reports_dir
    if args.trend_days:
        config["analysis"]["trend_days"] = args.trend_days
    if args.no_viz:
        config["output"]["save_visualizations"] = False
    
    # Run workflow
    return run_kg_workflow(config)


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════╗
    ║     Night_watcher Knowledge Graph Controller      ║
    ║     Detecting Authoritarian Patterns Over Time    ║
    ╚═══════════════════════════════════════════════════╝
    """)
    sys.exit(main())
