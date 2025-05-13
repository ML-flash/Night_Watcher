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

# Import provenance verification
from analysis_provenance import AnalysisProvenanceTracker

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
        "file_pattern": "analysis_*.json"
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
    },
    "provenance": {
        "enabled": True,
        "dev_mode": True,
        "verify": True
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

def load_analyzed_files(analyzed_dir: str, pattern: str = "analysis_*.json") -> List[Dict[str, Any]]:
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


def process_analyses(kg: KnowledgeGraph, 
                     analyses: List[Dict[str, Any]], 
                     provenance_tracker: Optional[AnalysisProvenanceTracker] = None,
                     verify_provenance: bool = False) -> Dict[str, int]:
    """Process analyses and add to knowledge graph with provenance verification"""
    total_nodes = 0
    total_edges = 0
    total_articles = 0
    verified_analyses = 0
    unverified_analyses = 0
    skipped_analyses = 0
    
    for analysis in analyses:
        article = analysis.get("article", {})
        kg_payload = analysis.get("kg_payload", {})
        
        if not article or not kg_payload:
            logging.warning("Analysis missing article or KG payload")
            skipped_analyses += 1
            continue
        
        # Verify provenance if enabled and tracker provided
        if verify_provenance and provenance_tracker and "provenance_id" in analysis:
            verification = provenance_tracker.verify_analysis_record(analysis["provenance_id"])
            
            if verification.get("verified"):
                verified_analyses += 1
                logging.info(f"Verified provenance for analysis {analysis['provenance_id']}")
            else:
                unverified_analyses += 1
                logging.warning(f"Provenance verification failed for analysis {analysis['provenance_id']}: {verification.get('reason')}")
                
                # Skip analyses with invalid provenance if verification is required
                if verify_provenance:
                    logging.warning(f"Skipping analysis {analysis['provenance_id']} due to failed verification")
                    skipped_analyses += 1
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
        "temporal_relations": temporal_relations,
        "verified_analyses": verified_analyses,
        "unverified_analyses": unverified_analyses,
        "skipped_analyses": skipped_analyses,
        "provenance_verification": verify_provenance
    }


def generate_basic_reports(kg: KnowledgeGraph, output_dir: str, trend_days: int = 90, save_viz: bool = True) -> Dict[str, str]:
    """Generate basic reports from the knowledge graph"""
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Track generated files
    generated_files = {}
    
    # Generate graph statistics report
    stats = kg.get_basic_statistics()
    stats_path = os.path.join(output_dir, f"graph_statistics_{timestamp}.json")
    
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
        
    logging.info(f"Generated statistics report: {stats_path}")
    generated_files["statistics"] = stats_path
    
    # Save graph to JSON (simple export)
    if save_viz:
        json_path = os.path.join(output_dir, f"graph_export_{timestamp}.json")
        kg.save_graph(json_path)
        generated_files["graph_json"] = json_path
        logging.info(f"Saved graph to JSON: {json_path}")
    
    # Save a snapshot of the graph
    snapshot_id = kg.save_snapshot(f"Analysis run {timestamp}")
    if snapshot_id:
        logging.info(f"Created graph snapshot: {snapshot_id}")
        generated_files["graph_snapshot"] = snapshot_id
    
    return generated_files


# ==========================================
# Main Execution Logic
# ==========================================

def run_kg_workflow(config: Dict[str, Any], args: argparse.Namespace) -> int:
    """Run the knowledge graph workflow"""
    # Extract configuration values
    kg_config = config["knowledge_graph"]
    input_config = config["input"]
    output_config = config["output"]
    analysis_config = config["analysis"]
    provenance_config = config.get("provenance", {})
    
    graph_file = kg_config["graph_file"]
    taxonomy_file = kg_config["taxonomy_file"]
    analyzed_dir = input_config["analyzed_dir"]
    file_pattern = input_config["file_pattern"]
    reports_dir = output_config["reports_dir"]
    save_viz = output_config["save_visualizations"]
    trend_days = analysis_config["trend_days"]
    
    # Override with command line args if provided
    if args.analyzed_dir:
        analyzed_dir = args.analyzed_dir
    if args.reports_dir:
        reports_dir = args.reports_dir
    if args.file_pattern:
        file_pattern = args.file_pattern
    if args.trend_days:
        trend_days = args.trend_days
    if args.no_viz:
        save_viz = False
    
    # Determine if provenance verification is enabled
    provenance_enabled = provenance_config.get("enabled", True) and not args.disable_provenance
    verify_provenance = provenance_config.get("verify", True) and provenance_enabled
    dev_mode = provenance_config.get("dev_mode", True)
    
    # Set up provenance tracker if enabled
    provenance_tracker = None
    if provenance_enabled:
        provenance_dir = os.path.join(os.path.dirname(analyzed_dir), "analysis_provenance")
        
        logging.info(f"Initializing analysis provenance tracking from {provenance_dir}")
        provenance_tracker = AnalysisProvenanceTracker(
            base_dir=provenance_dir,
            dev_passphrase=args.provenance_passphrase,
            dev_mode=dev_mode
        )
        
        if verify_provenance:
            logging.info("Provenance verification enabled for analysis processing")
        else:
            logging.info("Provenance tracking enabled but verification disabled")
    
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
    stats = process_analyses(
        kg, 
        analyses, 
        provenance_tracker=provenance_tracker if provenance_enabled else None,
        verify_provenance=verify_provenance
    )
    
    # Generate reports
    logging.info("Generating basic reports")
    generated_files = generate_basic_reports(kg, reports_dir, trend_days, save_viz)
    
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
    
    if provenance_enabled:
        print("\n=== Provenance Verification ===")
        print(f"Verification enabled: {'Yes' if verify_provenance else 'No'}")
        print(f"Verified analyses: {stats['verified_analyses']}")
        if verify_provenance:
            print(f"Unverified analyses: {stats['unverified_analyses']}")
            print(f"Skipped analyses: {stats['skipped_analyses']}")
    
    print("\nReports generated:")
    for report_name, file_path in generated_files.items():
        print(f"- {report_name}: {os.path.basename(file_path) if isinstance(file_path, str) else file_path}")
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
    
    # Add provenance options
    parser.add_argument("--provenance-passphrase", help="Passphrase for provenance verification (dev mode)")
    parser.add_argument("--disable-provenance", action="store_true", help="Disable provenance verification")
    parser.add_argument("--skip-verification", action="store_true", help="Skip verification but keep provenance tracking")
    
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
    
    # If provenance passphrase is not provided but environment variable exists, use it
    if not args.provenance_passphrase:
        env_passphrase = os.environ.get("NIGHT_WATCHER_PASSPHRASE")
        if env_passphrase:
            args.provenance_passphrase = env_passphrase
            logging.info("Using provenance passphrase from environment variable")
            
    # If skip-verification is true, update config
    if args.skip_verification:
        config["provenance"]["verify"] = False
    
    print("""
    ╔═══════════════════════════════════════════════════╗
    ║     Night_watcher Knowledge Graph Controller      ║
    ║     Detecting Authoritarian Patterns Over Time    ║
    ╚═══════════════════════════════════════════════════╝
    """)
    
    # Run workflow
    return run_kg_workflow(config, args)


if __name__ == "__main__":
    sys.exit(main())