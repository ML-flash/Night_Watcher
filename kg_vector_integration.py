"""
Night_watcher Knowledge Graph Integration with Vector Store
Helper functions to integrate the knowledge graph with vector embeddings.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple

# Try to import our components
try:
    from knowledge_graph import KnowledgeGraph
    from vector_store import VectorStore
except ImportError:
    logging.warning("Knowledge graph or vector store modules not found in path")

class KGVectorIntegration:
    """
    Integration layer between Knowledge Graph and Vector Store.
    Provides helper functions for combined operations.
    """
    
    def __init__(self, kg: 'KnowledgeGraph', vector_store: 'VectorStore', 
                 enable_auto_sync: bool = True):
        """
        Initialize integration between knowledge graph and vector store.
        
        Args:
            kg: Knowledge Graph instance
            vector_store: Vector Store instance
            enable_auto_sync: Whether to automatically sync changes
        """
        self.kg = kg
        self.vs = vector_store
        self.auto_sync = enable_auto_sync
        self.logger = logging.getLogger("KGVectorIntegration")
        
        # Set kg reference in vector store
        self.vs.kg = self.kg
        
        # Initial sync
        if enable_auto_sync:
            self.sync()
    
    def sync(self) -> Dict[str, int]:
        """
        Synchronize the vector store with the knowledge graph.
        
        Returns:
            Statistics about the synchronization
        """
        return self.vs.sync_with_kg(self.kg)
    
    def add_implicit_edges(self, threshold: float = 0.85, 
                          relation: str = "semantic_similarity") -> int:
        """
        Add implicit edges discovered by the vector store to the knowledge graph.
        
        Args:
            threshold: Similarity threshold (0-1)
            relation: Relation type to use for new edges
            
        Returns:
            Number of edges added
        """
        # Discover implicit relationships
        relationships = self.vs.discover_implicit_relationships(threshold=threshold)
        
        edges_added = 0
        
        # Add each to the knowledge graph
        for rel in relationships:
            source_id = rel["source_id"]
            target_id = rel["target_id"]
            
            # Skip if edge already exists in either direction
            if self.kg.graph.has_edge(source_id, target_id) or self.kg.graph.has_edge(target_id, source_id):
                continue
            
            # Add edge to graph
            edge_id = self.kg.add_edge(
                source_id=source_id,
                relation=relation,
                target_id=target_id,
                attributes={
                    "similarity_score": rel["score"],
                    "discovered_by": "vector_similarity"
                }
            )
            
            if edge_id:
                edges_added += 1
        
        self.logger.info(f"Added {edges_added} implicit edges to knowledge graph")
        return edges_added
    
    def hybrid_search(self, query: str, node_type: Optional[str] = None, 
                     limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform a hybrid search combining vector similarity and graph traversal.
        
        Args:
            query: Search query
            node_type: Optional node type filter
            limit: Maximum number of results
            
        Returns:
            List of search results with scores
        """
        # Vector similarity search
        vector_results = self.vs.similar_nodes(query, node_type, limit=limit)
        
        # Enhance with graph information
        for result in vector_results:
            node_id = result["id"]
            
            # Add neighbor count
            in_neighbors = list(self.kg.graph.predecessors(node_id))
            out_neighbors = list(self.kg.graph.successors(node_id))
            
            result["in_degree"] = len(in_neighbors)
            result["out_degree"] = len(out_neighbors)
            
            # Get node attributes from graph
            if self.kg.graph.has_node(node_id):
                node_data = self.kg.graph.nodes[node_id]
                result["attributes"] = node_data.get("attributes", {})
        
        return vector_results
    
    def get_node_clusters(self, node_type: Optional[str] = None, 
                         num_clusters: int = 10) -> Dict[str, Any]:
        """
        Get clusters of nodes based on semantic similarity.
        
        Args:
            node_type: Optional node type filter
            num_clusters: Number of clusters
            
        Returns:
            Clustering results with analysis
        """
        # Get clusters from vector store
        clusters = self.vs.cluster_nodes(node_type, num_clusters)
        
        if not clusters:
            return {"status": "error", "message": "No clusters found"}
        
        # Analyze each cluster
        cluster_info = {}
        for cluster_id, node_ids in clusters.items():
            # Count node types
            node_types = {}
            
            for node_id in node_ids:
                if self.kg.graph.has_node(node_id):
                    node_data = self.kg.graph.nodes[node_id]
                    node_type = node_data.get("type", "unknown")
                    node_types[node_type] = node_types.get(node_type, 0) + 1
            
            # Calculate connectivity within cluster
            internal_edges = 0
            for i, node1 in enumerate(node_ids):
                for node2 in node_ids[i+1:]:
                    if self.kg.graph.has_edge(node1, node2) or self.kg.graph.has_edge(node2, node1):
                        internal_edges += 1
            
            max_possible_edges = (len(node_ids) * (len(node_ids) - 1)) / 2
            connectivity = internal_edges / max_possible_edges if max_possible_edges > 0 else 0
            
            # Add to results
            cluster_info[f"cluster_{cluster_id}"] = {
                "size": len(node_ids),
                "node_ids": node_ids[:10],  # First 10 nodes
                "node_types": node_types,
                "internal_edges": internal_edges,
                "connectivity": connectivity
            }
        
        return {
            "status": "success",
            "num_clusters": len(clusters),
            "total_nodes": sum(len(nodes) for nodes in clusters.values()),
            "clusters": cluster_info
        }
    
    def enhanced_node_analysis(self, node_id: str) -> Dict[str, Any]:
        """
        Provide enhanced analysis of a node combining graph and vector approaches.
        
        Args:
            node_id: Node identifier
            
        Returns:
            Enhanced node analysis
        """
        if not self.kg.graph.has_node(node_id):
            return {"status": "error", "message": f"Node {node_id} not found in graph"}
        
        # Get node data from graph
        node_data = self.kg.graph.nodes[node_id]
        node_type = node_data.get("type", "unknown")
        node_name = node_data.get("name", "")
        
        # Get vector similarity neighbors
        vector_neighbors = self.vs.find_related_nodes(node_id, limit=5)
        
        # Get graph neighbors
        in_neighbors = []
        for pred in self.kg.graph.predecessors(node_id):
            edge_data = self.kg.graph.edges[pred, node_id]
            in_neighbors.append({
                "id": pred,
                "relation": edge_data.get("relation", "unknown"),
                "name": self.kg.graph.nodes[pred].get("name", "")
            })
        
        out_neighbors = []
        for succ in self.kg.graph.successors(node_id):
            edge_data = self.kg.graph.edges[node_id, succ]
            out_neighbors.append({
                "id": succ,
                "relation": edge_data.get("relation", "unknown"), 
                "name": self.kg.graph.nodes[succ].get("name", "")
            })
        
        # Return enhanced analysis
        return {
            "status": "success",
            "node_id": node_id,
            "type": node_type,
            "name": node_name,
            "attributes": node_data.get("attributes", {}),
            "graph_connections": {
                "in_degree": len(in_neighbors),
                "out_degree": len(out_neighbors),
                "in_neighbors": in_neighbors[:5],  # First 5
                "out_neighbors": out_neighbors[:5]  # First 5
            },
            "vector_neighbors": vector_neighbors,
            "timestamp": node_data.get("timestamp")
        }
    
    def find_path_between_nodes(self, source_id: str, target_id: str, 
                              max_depth: int = 3) -> Dict[str, Any]:
        """
        Find paths between two nodes in the graph, enhanced with vector similarity.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            max_depth: Maximum path length
            
        Returns:
            Path information
        """
        if not self.kg.graph.has_node(source_id):
            return {"status": "error", "message": f"Source node {source_id} not found"}
            
        if not self.kg.graph.has_node(target_id):
            return {"status": "error", "message": f"Target node {target_id} not found"}
        
        try:
            import networkx as nx
            
            # Find all simple paths up to max_depth
            paths = list(nx.all_simple_paths(
                self.kg.graph, source_id, target_id, cutoff=max_depth
            ))
            
            # Format paths
            formatted_paths = []
            for path in paths:
                path_info = []
                for i in range(len(path)-1):
                    from_id = path[i]
                    to_id = path[i+1]
                    edge_data = self.kg.graph.edges[from_id, to_id]
                    
                    path_info.append({
                        "from": {
                            "id": from_id,
                            "name": self.kg.graph.nodes[from_id].get("name", "")
                        },
                        "to": {
                            "id": to_id,
                            "name": self.kg.graph.nodes[to_id].get("name", "")
                        },
                        "relation": edge_data.get("relation", "unknown")
                    })
                
                formatted_paths.append(path_info)
            
            # If no direct path, check vector similarity
            if not paths:
                # Get vector similarity between nodes
                source_vector = None
                target_vector = None
                
                source_path = os.path.join(self.vs.nodes_dir, f"{source_id}.npz")
                target_path = os.path.join(self.vs.nodes_dir, f"{target_id}.npz")
                
                import numpy as np
                
                if os.path.exists(source_path) and os.path.exists(target_path):
                    source_data = np.load(source_path)
                    target_data = np.load(target_path)
                    
                    source_vector = source_data['embedding']
                    target_vector = target_data['embedding']
                    
                    # Calculate cosine similarity
                    similarity = np.dot(source_vector, target_vector) / (
                        np.linalg.norm(source_vector) * np.linalg.norm(target_vector)
                    )
                else:
                    similarity = 0
                
                return {
                    "status": "not_connected",
                    "source": {
                        "id": source_id,
                        "name": self.kg.graph.nodes[source_id].get("name", "")
                    },
                    "target": {
                        "id": target_id,
                        "name": self.kg.graph.nodes[target_id].get("name", "")
                    },
                    "vector_similarity": float(similarity),
                    "message": "No direct path found in graph, but vector similarity provided"
                }
            
            return {
                "status": "success",
                "paths_count": len(paths),
                "paths": formatted_paths
            }
            
        except Exception as e:
            self.logger.error(f"Error finding path: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_similar_events(self, query: str = None, node_id: str = None, 
                         limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find similar events using vector similarity.
        
        Args:
            query: Text query (optional)
            node_id: Node ID for reference (optional)
            limit: Maximum number of results
            
        Returns:
            List of similar events
        """
        if not query and not node_id:
            return []
            
        # If node_id provided, use it as the reference
        if node_id and not query:
            results = self.vs.find_related_nodes(node_id, limit=limit)
            
            # Filter for event nodes
            event_results = [
                r for r in results 
                if r.get("type") == "event" or 
                self.kg.graph.nodes[r.get("id")].get("type") == "event"
            ]
            
            return event_results
            
        # Use text query directly
        results = self.vs.similar_nodes(query, node_type="event", limit=limit)
        
        return results
    
    def detect_recurring_patterns(self, threshold: float = 0.85, 
                                min_cluster_size: int = 3) -> List[Dict[str, Any]]:
        """
        Detect recurring patterns in events using vector clustering.
        
        Args:
            threshold: Similarity threshold
            min_cluster_size: Minimum cluster size to consider
            
        Returns:
            List of detected patterns
        """
        # Get event node clusters
        clusters = self.vs.cluster_nodes(node_type="event")
        
        patterns = []
        
        for cluster_id, node_ids in clusters.items():
            # Skip small clusters
            if len(node_ids) < min_cluster_size:
                continue
                
            # Analyze cluster
            events = []
            timestamps = []
            
            for node_id in node_ids:
                if self.kg.graph.has_node(node_id):
                    node_data = self.kg.graph.nodes[node_id]
                    
                    # Skip non-events
                    if node_data.get("type") != "event":
                        continue
                        
                    events.append({
                        "id": node_id,
                        "name": node_data.get("name", ""),
                        "timestamp": node_data.get("timestamp", ""),
                        "attributes": node_data.get("attributes", {})
                    })
                    
                    # Track timestamps for temporal analysis
                    if node_data.get("timestamp"):
                        timestamps.append(node_data.get("timestamp"))
            
            # Sort events by timestamp
            events.sort(key=lambda x: x.get("timestamp", ""))
            
            # Skip if too few events found
            if len(events) < min_cluster_size:
                continue
                
            # Extract common themes
            names = [e.get("name", "") for e in events]
            name_text = " ".join(names)
            
            # Add pattern
            patterns.append({
                "pattern_id": f"pattern_{cluster_id}",
                "event_count": len(events),
                "events": events,
                "timestamps": timestamps,
                "cluster_size": len(node_ids)
            })
        
        return patterns
