#!/usr/bin/env python3
"""
Night_watcher Vector Integration Tool
Command-line interface for the vector integration component.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional

# Make sure we can find our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import components
from knowledge_graph import KnowledgeGraph
from vector_store import VectorStore
from kg_vector_integration import KGVectorIntegration

# ==========================================
# Configuration Functions
# ==========================================

DEFAULT_CONFIG = {
    "vector_store": {
        "base_dir": "data/vector_store",
        "embedding_provider": "local",
        "embedding_dim": 384,
        "index_type": "flat"
    },
    "knowledge_graph": {
        "graph_file": "data/knowledge_graph/graph.json",
        "taxonomy_file": "KG_Taxonomy.csv"
    },
    "input": {
        "analyzed_dir": "data/analyzed"
    },
    "output": {
        "reports_dir": "data/analysis"
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

def initialize_components(config: Dict[str, Any]) -> tuple:
    """
    Initialize all required components.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (knowledge_graph, vector_store, integration)
    """
    # Initialize knowledge graph
    kg_config = config["knowledge_graph"]
    kg = KnowledgeGraph(
        graph_file=kg_config["graph_file"],
        taxonomy_file=kg_config["taxonomy_file"]
    )
    
    # Initialize vector store
    vs_config = config["vector_store"]
    vector_store = VectorStore(
        base_dir=vs_config["base_dir"],
        embedding_provider=vs_config["embedding_provider"],
        embedding_dim=vs_config["embedding_dim"],
        index_type=vs_config["index_type"],
        kg_instance=kg
    )
    
    # Initialize integration
    integration = KGVectorIntegration(
        kg=kg,
        vector_store=vector_store,
        enable_auto_sync=False  # Don't automatically sync on init
    )
    
    return kg, vector_store, integration


def sync_vector_store(integration: KGVectorIntegration) -> Dict[str, int]:
    """
    Synchronize the vector store with the knowledge graph.
    
    Args:
        integration: KGVectorIntegration instance
        
    Returns:
        Synchronization statistics
    """
    stats = integration.sync()
    print(f"\n=== Vector Store Synchronization ===")
    print(f"Nodes added: {stats['nodes_added']}")
    print(f"Total nodes: {stats['nodes_total']}")
    
    # Save vector store state
    integration.vs.save()
    
    return stats


def discover_relationships(integration: KGVectorIntegration, 
                         threshold: float = 0.85) -> Dict[str, Any]:
    """
    Discover and add implicit relationships.
    
    Args:
        integration: KGVectorIntegration instance
        threshold: Similarity threshold
        
    Returns:
        Results dictionary
    """
    # Find implicit relationships
    edges_added = integration.add_implicit_edges(threshold=threshold)
    
    print(f"\n=== Implicit Relationship Discovery ===")
    print(f"Edges added: {edges_added}")
    
    # Save graph changes
    integration.kg.save_graph()
    
    return {"edges_added": edges_added}


def search_nodes(integration: KGVectorIntegration, query: str, 
                node_type: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
    """
    Search for nodes using hybrid search.
    
    Args:
        integration: KGVectorIntegration instance
        query: Search query
        node_type: Optional node type filter
        limit: Maximum number of results
        
    Returns:
        Search results
    """
    results = integration.hybrid_search(query, node_type, limit)
    
    print(f"\n=== Node Search Results ===")
    print(f"Query: {query}")
    if node_type:
        print(f"Node type: {node_type}")
    print(f"Results found: {len(results)}")
    
    if results:
        print("\nTop results:")
        for i, result in enumerate(results[:5], 1):
            print(f"{i}. {result['name']} ({result['type']}) - Score: {result['score']:.3f}")
            print(f"   Connections: {result['in_degree']} in, {result['out_degree']} out")
    
    return {
        "query": query,
        "node_type": node_type,
        "count": len(results),
        "results": results
    }


def analyze_clusters(integration: KGVectorIntegration, 
                    node_type: Optional[str] = None,
                    num_clusters: int = 10) -> Dict[str, Any]:
    """
    Analyze node clusters.
    
    Args:
        integration: KGVectorIntegration instance
        node_type: Optional node type filter
        num_clusters: Number of clusters
        
    Returns:
        Cluster analysis
    """
    results = integration.get_node_clusters(node_type, num_clusters)
    
    if results["status"] == "error":
        print(f"\n=== Cluster Analysis Error ===")
        print(results["message"])
        return results
    
    print(f"\n=== Node Cluster Analysis ===")
    print(f"Clusters: {results['num_clusters']}")
    print(f"Total nodes: {results['total_nodes']}")
    
    if "clusters" in results:
        print("\nCluster statistics:")
        for cluster_id, cluster_info in results["clusters"].items():
            print(f"- {cluster_id}: {cluster_info['size']} nodes, {cluster_info['internal_edges']} internal edges")
            
            # Print node types if available
            if "node_types" in cluster_info:
                print(f"  Node types: ", end="")
                node_types = []
                for nt, count in cluster_info["node_types"].items():
                    node_types.append(f"{nt} ({count})")
                print(", ".join(node_types))
    
    return results


def detect_patterns(integration: KGVectorIntegration) -> Dict[str, Any]:
    """
    Detect recurring patterns in the knowledge graph.
    
    Args:
        integration: KGVectorIntegration instance
        
    Returns:
        Detected patterns
    """
    patterns = integration.detect_recurring_patterns()
    
    print(f"\n=== Pattern Detection ===")
    print(f"Patterns detected: {len(patterns)}")
    
    if patterns:
        print("\nTop patterns:")
        for i, pattern in enumerate(patterns[:3], 1):
            print(f"{i}. Pattern {pattern['pattern_id']}: {pattern['event_count']} events")
            
            # Print first few events
            events = pattern.get("events", [])
            for j, event in enumerate(events[:3], 1):
                print(f"   - {event.get('name', 'Unknown')} ({event.get('timestamp', 'Unknown')})")
            
            if len(events) > 3:
                print(f"   - ... and {len(events) - 3} more events")
    
    return {
        "count": len(patterns),
        "patterns": patterns
    }


def analyze_node(integration: KGVectorIntegration, 
               node_id: str) -> Dict[str, Any]:
    """
    Provide enhanced analysis of a node.
    
    Args:
        integration: KGVectorIntegration instance
        node_id: Node identifier
        
    Returns:
        Node analysis
    """
    result = integration.enhanced_node_analysis(node_id)
    
    if result["status"] == "error":
        print(f"\n=== Node Analysis Error ===")
        print(result["message"])
        return result
    
    print(f"\n=== Node Analysis: {result['name']} ===")
    print(f"Type: {result['type']}")
    print(f"Connections: {result['graph_connections']['in_degree']} in, {result['graph_connections']['out_degree']} out")
    
    # Print vector neighbors
    vector_neighbors = result.get("vector_neighbors", [])
    if vector_neighbors:
        print("\nSemantically similar nodes:")
        for i, neighbor in enumerate(vector_neighbors[:3], 1):
            print(f"{i}. {neighbor['name']} ({neighbor['type']}) - Score: {neighbor['score']:.3f}")
    
    # Print graph neighbors
    in_neighbors = result.get("graph_connections", {}).get("in_neighbors", [])
    if in_neighbors:
        print("\nIncoming connections:")
        for i, neighbor in enumerate(in_neighbors[:3], 1):
            print(f"{i}. {neighbor['name']} via {neighbor['relation']}")
    
    out_neighbors = result.get("graph_connections", {}).get("out_neighbors", [])
    if out_neighbors:
        print("\nOutgoing connections:")
        for i, neighbor in enumerate(out_neighbors[:3], 1):
            print(f"{i}. {neighbor['name']} via {neighbor['relation']}")
    
    return result


def save_output(data: Dict[str, Any], output_dir: str, 
              command: str) -> str:
    """
    Save output data to a file.
    
    Args:
        data: Output data
        output_dir: Output directory
        command: Command name
        
    Returns:
        Output file path
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{command}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\nOutput saved to: {filepath}")
    return filepath


# ==========================================
# Main Function
# ==========================================

def run_command(args, config: Dict[str, Any]) -> int:
    """
    Run the selected command.
    
    Args:
        args: Command line arguments
        config: Configuration dictionary
        
    Returns:
        Exit code
    """
    # Initialize components
    kg, vector_store, integration = initialize_components(config)
    
    # Get output directory
    output_dir = config["output"]["reports_dir"]
    
    # Run appropriate command
    if args.command == "sync":
        result = sync_vector_store(integration)
        
    elif args.command == "discover":
        result = discover_relationships(integration, threshold=args.threshold)
        
    elif args.command == "search":
        result = search_nodes(integration, args.query, args.node_type, args.limit)
        
    elif args.command == "clusters":
        result = analyze_clusters(integration, args.node_type, args.num_clusters)
        
    elif args.command == "patterns":
        result = detect_patterns(integration)
        
    elif args.command == "analyze-node":
        result = analyze_node(integration, args.node_id)
        
    else:
        print(f"Unknown command: {args.command}")
        return 1
    
    # Save output if requested
    if args.save:
        save_output(result, output_dir, args.command)
    
    return 0


def main() -> int:
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Night_watcher Vector Integration Tool")
    
    # Global options
    parser.add_argument("--config", default="config.json", help="Path to config file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--save", action="store_true", help="Save output to file")
    parser.add_argument("--create-config", action="store_true", help="Create default config file and exit")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Sync command
    sync_parser = subparsers.add_parser("sync", help="Synchronize vector store with knowledge graph")
    
    # Discover command
    discover_parser = subparsers.add_parser("discover", help="Discover implicit relationships")
    discover_parser.add_argument("--threshold", type=float, default=0.85, help="Similarity threshold")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for nodes")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--node-type", help="Filter by node type")
    search_parser.add_argument("--limit", type=int, default=10, help="Maximum results")
    
    # Clusters command
    clusters_parser = subparsers.add_parser("clusters", help="Analyze node clusters")
    clusters_parser.add_argument("--node-type", help="Filter by node type")
    clusters_parser.add_argument("--num-clusters", type=int, default=10, help="Number of clusters")
    
    # Patterns command
    patterns_parser = subparsers.add_parser("patterns", help="Detect recurring patterns")
    
    # Analyze node command
    analyze_parser = subparsers.add_parser("analyze-node", help="Analyze a specific node")
    analyze_parser.add_argument("node_id", help="Node identifier")
    
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
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize logging
    log_dir = config["logging"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f"vector_tool_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
            logging.StreamHandler() if args.verbose else logging.NullHandler()
        ]
    )
    
    # Show welcome banner
    print("""
    ╔═══════════════════════════════════════════════════╗
    ║     Night_watcher Vector Integration Tool         ║
    ║     Connecting Patterns Through Vector Embeddings ║
    ╚═══════════════════════════════════════════════════╝
    """)
    
    # Run the command if specified
    if args.command:
        return run_command(args, config)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
