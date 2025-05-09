"""
Night_watcher Knowledge Graph
Manages the knowledge graph for storing entities, events, relationships, and authoritarian patterns.
"""

import os
import json
import logging
import csv
import re
import datetime
from typing import Dict, List, Any, Optional, Set, Union, Tuple
import networkx as nx
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)

class KnowledgeGraph:
    """
    Knowledge graph for storing entities, events, relationships, and patterns.
    Implements a graph-based data structure using NetworkX with JSON serialization.
    """

    def __init__(self, base_dir: str = "data/knowledge_graph", taxonomy_path: str = "KG_Taxonomy.csv"):
        """
        Initialize the knowledge graph.
        
        Args:
            base_dir: Base directory for knowledge graph storage
            taxonomy_path: Path to taxonomy CSV file
        """
        self.base_dir = base_dir
        self.nodes_dir = os.path.join(base_dir, "nodes")
        self.edges_dir = os.path.join(base_dir, "edges")
        self.snapshots_dir = os.path.join(base_dir, "snapshots")
        self.taxonomy_path = taxonomy_path
        
        # Create directories if they don't exist
        os.makedirs(self.nodes_dir, exist_ok=True)
        os.makedirs(self.edges_dir, exist_ok=True)
        os.makedirs(self.snapshots_dir, exist_ok=True)
        
        # Initialize NetworkX graph
        self.graph = nx.DiGraph()
        
        # Load taxonomy if available
        self.taxonomy = self._load_taxonomy()
        
        # Node and edge counters for ID generation
        self._node_counter = 0
        self._edge_counter = 0
        
        # Load existing graph if available
        self._load_graph()
        
        self.logger = logging.getLogger("KnowledgeGraph")
        self.logger.info(f"Knowledge graph initialized at {base_dir}")
        self.logger.info(f"Current graph: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")

    def _load_taxonomy(self) -> Dict[str, Any]:
        """
        Load the knowledge graph taxonomy from CSV.
        
        Returns:
            Dictionary containing node types and relation types with their attributes
        """
        taxonomy = {
            "node_types": {},
            "relation_types": {},
            "domains": {}
        }
        
        if not os.path.exists(self.taxonomy_path):
            self.logger.warning(f"Taxonomy file not found at {self.taxonomy_path}")
            return taxonomy
        
        try:
            with open(self.taxonomy_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Process taxonomy definition
                    taxonomy_type = row.get("taxonomy_type", "").strip()
                    name = row.get("name", "").strip()
                    
                    if not taxonomy_type or not name:
                        continue
                    
                    domain_range = row.get("domain_range", "")
                    key_attributes = row.get("key_attributes", "")
                    examples = row.get("examples", "")
                    
                    # Parse attributes
                    attributes = [attr.strip() for attr in key_attributes.split(";")]
                    
                    # Parse domain/range for relations
                    domain_range_parts = domain_range.split("→") if "→" in domain_range else []
                    
                    if taxonomy_type == "node":
                        taxonomy["node_types"][name] = {
                            "attributes": attributes,
                            "examples": examples
                        }
                    elif taxonomy_type == "relation":
                        # Extract domain and range
                        if len(domain_range_parts) > 1:
                            domain = domain_range_parts[0].strip()
                            range_val = domain_range_parts[1].strip()
                            
                            taxonomy["relation_types"][name] = {
                                "domain": domain,
                                "range": range_val,
                                "attributes": attributes,
                                "examples": examples
                            }
                            
                            # Store domain-range mappings
                            if domain not in taxonomy["domains"]:
                                taxonomy["domains"][domain] = {}
                            if name not in taxonomy["domains"][domain]:
                                taxonomy["domains"][domain][name] = []
                            taxonomy["domains"][domain][name].append(range_val)
            
            self.logger.info(f"Loaded taxonomy: {len(taxonomy['node_types'])} node types, {len(taxonomy['relation_types'])} relation types")
            return taxonomy
            
        except Exception as e:
            self.logger.error(f"Error loading taxonomy: {e}")
            return taxonomy

    def _load_graph(self) -> None:
        """
        Load the existing graph from the node and edge files.
        """
        # Load nodes
        node_count = 0
        for filename in os.listdir(self.nodes_dir):
            if filename.endswith(".json"):
                node_id = filename[:-5]  # Remove .json extension
                filepath = os.path.join(self.nodes_dir, filename)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        node_data = json.load(f)
                        
                    # Add node to graph
                    self.graph.add_node(node_id, **node_data)
                    
                    # Update node counter
                    numeric_id = self._extract_numeric_id(node_id)
                    if numeric_id > self._node_counter:
                        self._node_counter = numeric_id
                        
                    node_count += 1
                except Exception as e:
                    self.logger.error(f"Error loading node {node_id}: {e}")
        
        # Load edges
        edge_count = 0
        for filename in os.listdir(self.edges_dir):
            if filename.endswith(".json"):
                edge_id = filename[:-5]  # Remove .json extension
                filepath = os.path.join(self.edges_dir, filename)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        edge_data = json.load(f)
                        
                    # Extract source and target
                    source_id = edge_data.get("source_id")
                    target_id = edge_data.get("target_id")
                    
                    if source_id and target_id:
                        # Add edge to graph
                        self.graph.add_edge(source_id, target_id, id=edge_id, **edge_data)
                        
                        # Update edge counter
                        numeric_id = self._extract_numeric_id(edge_id)
                        if numeric_id > self._edge_counter:
                            self._edge_counter = numeric_id
                            
                        edge_count += 1
                except Exception as e:
                    self.logger.error(f"Error loading edge {edge_id}: {e}")
        
        self.logger.info(f"Loaded {node_count} nodes and {edge_count} edges from disk")

    def _extract_numeric_id(self, id_str: str) -> int:
        """
        Extract numeric ID from string ID.
        
        Args:
            id_str: String ID (e.g., "node_1" or "edge_42")
            
        Returns:
            Numeric ID or 0 if not found
        """
        match = re.search(r'(\d+)$', id_str)
        if match:
            return int(match.group(1))
        return 0

    def add_node(self, 
                 node_type: str, 
                 name: str, 
                 attributes: Dict[str, Any] = None, 
                 timestamp: str = None,
                 source_document_id: str = None,
                 source_sentence: str = None) -> str:
        """
        Add a node to the knowledge graph.
        
        Args:
            node_type: Type of node (e.g., actor, institution, policy, event)
            name: Name of the node
            attributes: Additional attributes for the node
            timestamp: Timestamp for the node (ISO format)
            source_document_id: Source document ID
            source_sentence: Source sentence from the document
            
        Returns:
            Node ID
        """
        # Validate node type
        if node_type not in self.taxonomy["node_types"]:
            self.logger.warning(f"Node type '{node_type}' not in taxonomy")
        
        # Generate node ID
        self._node_counter += 1
        node_id = f"node_{self._node_counter}"
        
        # Create node data
        node_data = {
            "id": node_id,
            "type": node_type,
            "name": name,
            "attributes": attributes or {},
            "timestamp": timestamp or datetime.now().isoformat(),
            "created_at": datetime.now().isoformat(),
            "source": {
                "document_id": source_document_id,
                "sentence": source_sentence
            }
        }
        
        # Check if a similar node already exists
        existing_node = self._find_similar_node(node_type, name)
        if existing_node:
            self.logger.info(f"Similar node already exists: {existing_node}")
            
            # Update existing node with any new attributes
            existing_data = self.graph.nodes[existing_node]
            if attributes:
                for key, value in attributes.items():
                    if key not in existing_data.get("attributes", {}):
                        existing_data["attributes"][key] = value
            
            # Add source document if different
            if source_document_id and source_document_id != existing_data.get("source", {}).get("document_id"):
                if "additional_sources" not in existing_data:
                    existing_data["additional_sources"] = []
                existing_data["additional_sources"].append({
                    "document_id": source_document_id,
                    "sentence": source_sentence
                })
            
            # Save updated node
            self._save_node(existing_node, existing_data)
            
            return existing_node
        
        # Add node to graph
        self.graph.add_node(node_id, **node_data)
        
        # Save node to disk
        self._save_node(node_id, node_data)
        
        self.logger.info(f"Added node {node_id}: {node_type} - {name}")
        return node_id

    def add_edge(self,
                 source_id: str,
                 relation: str,
                 target_id: str,
                 timestamp: str = None,
                 evidence_quote: str = None,
                 source_document_id: str = None,
                 attributes: Dict[str, Any] = None) -> str:
        """
        Add an edge to the knowledge graph.
        
        Args:
            source_id: Source node ID
            relation: Relation type
            target_id: Target node ID
            timestamp: Timestamp for the edge (ISO format)
            evidence_quote: Evidence quote for the edge
            source_document_id: Source document ID
            attributes: Additional attributes for the edge
            
        Returns:
            Edge ID
        """
        # Validate relation type
        if relation not in self.taxonomy["relation_types"]:
            self.logger.warning(f"Relation type '{relation}' not in taxonomy")
        
        # Validate source and target nodes
        if not self.graph.has_node(source_id):
            self.logger.warning(f"Source node {source_id} not in graph")
            return ""
        
        if not self.graph.has_node(target_id):
            self.logger.warning(f"Target node {target_id} not in graph")
            return ""
        
        # Generate edge ID
        self._edge_counter += 1
        edge_id = f"edge_{self._edge_counter}"
        
        # Create edge data
        edge_data = {
            "id": edge_id,
            "source_id": source_id,
            "relation": relation,
            "target_id": target_id,
            "timestamp": timestamp or datetime.now().isoformat(),
            "created_at": datetime.now().isoformat(),
            "evidence_quote": evidence_quote,
            "source_document_id": source_document_id,
            "attributes": attributes or {}
        }
        
        # Check if a similar edge already exists
        existing_edge = self._find_similar_edge(source_id, relation, target_id)
        if existing_edge:
            self.logger.info(f"Similar edge already exists: {existing_edge}")
            
            # Get existing edge data
            existing_data = self.graph.edges[existing_edge[0], existing_edge[1]]
            
            # Add new evidence if different
            if evidence_quote and evidence_quote != existing_data.get("evidence_quote"):
                if "additional_evidence" not in existing_data:
                    existing_data["additional_evidence"] = []
                existing_data["additional_evidence"].append({
                    "quote": evidence_quote,
                    "document_id": source_document_id,
                    "timestamp": timestamp or datetime.now().isoformat()
                })
            
            # Update attributes
            if attributes:
                for key, value in attributes.items():
                    if key not in existing_data.get("attributes", {}):
                        existing_data["attributes"][key] = value
            
            # Save updated edge
            self._save_edge(existing_data["id"], existing_data)
            
            return existing_data["id"]
        
        # Add edge to graph
        self.graph.add_edge(source_id, target_id, id=edge_id, **edge_data)
        
        # Save edge to disk
        self._save_edge(edge_id, edge_data)
        
        self.logger.info(f"Added edge {edge_id}: {source_id} --[{relation}]--> {target_id}")
        return edge_id

    def _find_similar_node(self, node_type: str, name: str) -> Optional[str]:
        """
        Find a node in the graph that is similar to the given node.
        
        Args:
            node_type: Type of node
            name: Name of the node
            
        Returns:
            Node ID if found, None otherwise
        """
        # Normalize name for comparison
        normalized_name = name.lower().strip()
        
        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") == node_type:
                existing_name = data.get("name", "").lower().strip()
                
                # Check exact match
                if existing_name == normalized_name:
                    return node_id
                
                # Check if one is a substring of the other (for partial matches)
                if normalized_name in existing_name or existing_name in normalized_name:
                    # Check if the longer name is at most 50% longer than the shorter name
                    len_ratio = max(len(normalized_name), len(existing_name)) / min(len(normalized_name), len(existing_name))
                    if len_ratio <= 1.5:
                        return node_id
        
        return None

    def _find_similar_edge(self, source_id: str, relation: str, target_id: str) -> Optional[Tuple[str, str]]:
        """
        Find an edge in the graph that is similar to the given edge.
        
        Args:
            source_id: Source node ID
            relation: Relation type
            target_id: Target node ID
            
        Returns:
            Edge (source_id, target_id) if found, None otherwise
        """
        if self.graph.has_edge(source_id, target_id):
            edge_data = self.graph.edges[source_id, target_id]
            if edge_data.get("relation") == relation:
                return (source_id, target_id)
        
        return None

    def _save_node(self, node_id: str, node_data: Dict[str, Any]) -> None:
        """
        Save a node to disk.
        
        Args:
            node_id: Node ID
            node_data: Node data
        """
        filepath = os.path.join(self.nodes_dir, f"{node_id}.json")
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(node_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Error saving node {node_id}: {e}")

    def _save_edge(self, edge_id: str, edge_data: Dict[str, Any]) -> None:
        """
        Save an edge to disk.
        
        Args:
            edge_id: Edge ID
            edge_data: Edge data
        """
        filepath = os.path.join(self.edges_dir, f"{edge_id}.json")
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(edge_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Error saving edge {edge_id}: {e}")

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a node from the knowledge graph.
        
        Args:
            node_id: Node ID
            
        Returns:
            Node data or None if not found
        """
        if self.graph.has_node(node_id):
            return dict(self.graph.nodes[node_id])
        return None

    def get_edge(self, edge_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an edge from the knowledge graph.
        
        Args:
            edge_id: Edge ID
            
        Returns:
            Edge data or None if not found
        """
        for source, target, data in self.graph.edges(data=True):
            if data.get("id") == edge_id:
                return dict(data)
        return None

    def get_nodes_by_type(self, node_type: str) -> List[Dict[str, Any]]:
        """
        Get all nodes of a specific type.
        
        Args:
            node_type: Type of node
            
        Returns:
            List of node data dictionaries
        """
        nodes = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") == node_type:
                node_data = dict(data)
                node_data["id"] = node_id
                nodes.append(node_data)
        return nodes

    def get_edges_by_relation(self, relation: str) -> List[Dict[str, Any]]:
        """
        Get all edges of a specific relation type.
        
        Args:
            relation: Relation type
            
        Returns:
            List of edge data dictionaries
        """
        edges = []
        for source, target, data in self.graph.edges(data=True):
            if data.get("relation") == relation:
                edge_data = dict(data)
                edge_data["source_id"] = source
                edge_data["target_id"] = target
                edges.append(edge_data)
        return edges

    def save_snapshot(self, name: Optional[str] = None) -> str:
        """
        Save a snapshot of the current graph.
        
        Args:
            name: Optional name for the snapshot
            
        Returns:
            Snapshot ID
        """
        # Generate snapshot ID and name
        timestamp = datetime.now().isoformat()
        snapshot_id = f"snapshot_{timestamp.replace(':', '-')}"
        snapshot_name = name or f"Snapshot {timestamp}"
        
        # Create snapshot data
        snapshot_data = {
            "id": snapshot_id,
            "name": snapshot_name,
            "timestamp": timestamp,
            "node_count": len(self.graph.nodes),
            "edge_count": len(self.graph.edges),
            "nodes": {node: data for node, data in self.graph.nodes(data=True)},
            "edges": {f"{source}_{target}": data for source, target, data in self.graph.edges(data=True)}
        }
        
        # Save snapshot to disk
        filepath = os.path.join(self.snapshots_dir, f"{snapshot_id}.json")
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(snapshot_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved snapshot {snapshot_id}: {snapshot_name}")
            return snapshot_id
        except Exception as e:
            self.logger.error(f"Error saving snapshot {snapshot_id}: {e}")
            return ""

    def load_snapshot(self, snapshot_id: str) -> bool:
        """
        Load a snapshot of the graph.
        
        Args:
            snapshot_id: Snapshot ID
            
        Returns:
            True if successful, False otherwise
        """
        filepath = os.path.join(self.snapshots_dir, f"{snapshot_id}.json")
        if not os.path.exists(filepath):
            self.logger.error(f"Snapshot {snapshot_id} not found")
            return False
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                snapshot_data = json.load(f)
            
            # Create a new graph from the snapshot
            new_graph = nx.DiGraph()
            
            # Add nodes
            for node_id, data in snapshot_data.get("nodes", {}).items():
                new_graph.add_node(node_id, **data)
            
            # Add edges
            for edge_key, data in snapshot_data.get("edges", {}).items():
                source, target = edge_key.split("_", 1)
                new_graph.add_edge(source, target, **data)
            
            # Replace current graph with the snapshot
            self.graph = new_graph
            
            # Update counters
            self._node_counter = max([self._extract_numeric_id(node_id) for node_id in self.graph.nodes], default=0)
            self._edge_counter = max([self._extract_numeric_id(data.get("id", "")) for _, _, data in self.graph.edges(data=True)], default=0)
            
            self.logger.info(f"Loaded snapshot {snapshot_id}: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
            return True
        except Exception as e:
            self.logger.error(f"Error loading snapshot {snapshot_id}: {e}")
            return False

    def list_snapshots(self) -> List[Dict[str, Any]]:
        """
        List all available snapshots.
        
        Returns:
            List of snapshot metadata
        """
        snapshots = []
        for filename in os.listdir(self.snapshots_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.snapshots_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    snapshots.append({
                        "id": data.get("id"),
                        "name": data.get("name"),
                        "timestamp": data.get("timestamp"),
                        "node_count": data.get("node_count"),
                        "edge_count": data.get("edge_count")
                    })
                except Exception as e:
                    self.logger.error(f"Error reading snapshot {filename}: {e}")
        
        # Sort by timestamp
        snapshots.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return snapshots

    def process_article_analysis(self, article: Dict[str, Any], kg_analysis: Dict[str, Any]) -> bool:
        """
        Process an article analysis and update the knowledge graph.
        
        Args:
            article: Article metadata
            kg_analysis: Knowledge graph analysis data with nodes and edges
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract article metadata
            article_id = article.get("id")
            document_id = article.get("document_id")
            publication_date = article.get("published")
            
            if not publication_date:
                publication_date = datetime.now().isoformat()
            
            # Create a node for the article itself
            article_node_id = self.add_node(
                node_type="media_outlet",
                name=article.get("source", "Unknown Source"),
                attributes={
                    "title": article.get("title", "Untitled"),
                    "url": article.get("url", ""),
                    "bias_label": article.get("bias_label", "unknown")
                },
                timestamp=publication_date,
                source_document_id=document_id
            )
            
            # Process nodes from the analysis
            kg_payload = kg_analysis.get("kg_payload", {})
            nodes = kg_payload.get("nodes", [])
            edges = kg_payload.get("edges", [])
            
            # Track node ID mappings
            node_id_map = {}
            
            # Add nodes to the graph
            for node in nodes:
                # Skip if missing required fields
                if not node.get("node_type") or not node.get("name"):
                    continue
                
                # Add node to graph
                node_id = self.add_node(
                    node_type=node.get("node_type"),
                    name=node.get("name"),
                    attributes=node.get("attributes", {}),
                    timestamp=node.get("timestamp", publication_date),
                    source_document_id=document_id,
                    source_sentence=node.get("source_sentence")
                )
                
                # Track ID mapping
                original_id = node.get("id")
                if original_id:
                    node_id_map[original_id] = node_id
            
            # Add article-entity edges
            for node_id in node_id_map.values():
                # Connect article to mentioned entities
                self.add_edge(
                    source_id=article_node_id,
                    relation="mentions",
                    target_id=node_id,
                    timestamp=publication_date,
                    source_document_id=document_id
                )
            
            # Add edges between entities
            for edge in edges:
                # Skip if missing required fields
                if not edge.get("source_id") or not edge.get("target_id") or not edge.get("relation"):
                    continue
                
                # Get mapped node IDs
                source_id = node_id_map.get(edge.get("source_id"))
                target_id = node_id_map.get(edge.get("target_id"))
                
                if not source_id or not target_id:
                    continue
                
                # Add edge to graph
                self.add_edge(
                    source_id=source_id,
                    relation=edge.get("relation"),
                    target_id=target_id,
                    timestamp=edge.get("timestamp", publication_date),
                    evidence_quote=edge.get("evidence_quote"),
                    source_document_id=document_id,
                    attributes={
                        "severity": edge.get("severity", 0.5),
                        "is_decayable": edge.get("is_decayable", True),
                        "reasoning": edge.get("reasoning", "")
                    }
                )
            
            self.logger.info(f"Processed article analysis: {article.get('title')} - added {len(nodes)} nodes, {len(edges)} edges")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing article analysis: {e}")
            return False

    def get_authoritarian_trends(self, days: int = 30) -> Dict[str, Any]:
        """
        Get authoritarian trends from the knowledge graph for the specified time period.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Dictionary with authoritarian trend data
        """
        # Calculate date threshold
        threshold_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Initialize result
        result = {
            "trend_score": 0.0,
            "affected_institutions": [],
            "primary_actors": [],
            "trends": [],
            "period_days": days,
            "timestamp": datetime.now().isoformat()
        }
        
        # Get authoritarian relation types
        auth_relations = [
            "undermines", "restricts", "co-opts", "purges", 
            "criminalizes", "censors", "intimidates", "delegitimizes"
        ]
        
        # Count occurrences of authoritarian relations
        relation_counts = {rel: 0 for rel in auth_relations}
        institutions_affected = set()
        actors_involved = set()
        
        # Gather affected institutions and primary actors
        for source, target, data in self.graph.edges(data=True):
            # Check if edge is recent enough
            edge_timestamp = data.get("timestamp", "")
            if edge_timestamp < threshold_date:
                continue
            
            # Check if this is an authoritarian relation
            relation = data.get("relation", "")
            if relation in auth_relations:
                relation_counts[relation] += 1
                
                # Get node types
                source_data = self.graph.nodes[source]
                target_data = self.graph.nodes[target]
                
                # Track affected institutions
                if target_data.get("type") == "institution":
                    institutions_affected.add(target)
                
                # Track primary actors
                if source_data.get("type") == "actor" or source_data.get("type") == "institution":
                    actors_involved.add(source)
        
        # Calculate trend score (0-10)
        total_auth_edges = sum(relation_counts.values())
        if total_auth_edges > 0:
            # Base score from 0-7 based on number of edges
            base_score = min(7, total_auth_edges / 3)
            
            # Add up to 3 points based on number of institutions affected
            institution_score = min(3, len(institutions_affected) / 2)
            
            result["trend_score"] = round(base_score + institution_score, 1)
        
        # Get affected institutions
        for inst_id in institutions_affected:
            inst_data = self.graph.nodes[inst_id]
            result["affected_institutions"].append({
                "id": inst_id,
                "name": inst_data.get("name", "Unknown"),
                "attributes": inst_data.get("attributes", {})
            })
        
        # Get primary actors
        for actor_id in actors_involved:
            actor_data = self.graph.nodes[actor_id]
            result["primary_actors"].append({
                "id": actor_id,
                "name": actor_data.get("name", "Unknown"),
                "type": actor_data.get("type", "actor"),
                "attributes": actor_data.get("attributes", {})
            })
        
        # Create trend insights
        for relation, count in relation_counts.items():
            if count > 0:
                result["trends"].append({
                    "relation": relation,
                    "count": count,
                    "description": self._get_relation_description(relation)
                })
        
        # Sort trends by count (descending)
        result["trends"].sort(key=lambda x: x["count"], reverse=True)
        
        return result

    def _get_relation_description(self, relation: str) -> str:
        """
        Get a description of a relation type.
        
        Args:
            relation: Relation type
            
        Returns:
            Description of the relation
        """
        descriptions = {
            "undermines": "Actions that weaken or erode the effectiveness or legitimacy of institutions",
            "restricts": "Placing limits on powers, jurisdiction, or independence",
            "co-opts": "Taking control of an institution through appointments or other mechanisms",
            "purges": "Removing individuals who don't align with a specific agenda",
            "criminalizes": "Making certain actions or expressions illegal or subject to punishment",
            "censors": "Suppressing speech, media content, or information flow",
            "intimidates": "Using fear or threats to influence behavior or decisions",
            "delegitimizes": "Attacking the credibility, authority, or legitimacy of an entity"
        }
        
        return descriptions.get(relation, f"Relation of type {relation}")

    def get_influential_actors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most influential actors in the knowledge graph based on centrality.
        
        Args:
            limit: Maximum number of actors to return
            
        Returns:
            List of actor data dictionaries with influence scores
        """
        # Calculate centrality
        centrality = nx.degree_centrality(self.graph)
        
        # Get all actor nodes
        actors = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") == "actor" or data.get("type") == "institution":
                # Calculate influence score (0-10)
                influence = centrality.get(node_id, 0) * 10
                
                # Create actor data
                actor_data = {
                    "id": node_id,
                    "name": data.get("name", "Unknown"),
                    "type": data.get("type"),
                    "influence_score": round(influence, 2),
                    "attributes": data.get("attributes", {})
                }
                
                # Count authoritarian relation types
                auth_relations = {
                    "undermines": 0, "restricts": 0, "co-opts": 0, 
                    "purges": 0, "criminalizes": 0, "censors": 0, 
                    "intimidates": 0, "delegitimizes": 0
                }
                
                for _, target, edge_data in self.graph.out_edges(node_id, data=True):
                    relation = edge_data.get("relation", "")
                    if relation in auth_relations:
                        auth_relations[relation] += 1
                
                actor_data["auth_relations"] = auth_relations
                actor_data["auth_count"] = sum(auth_relations.values())
                
                # Add to list if it has any connections
                if centrality.get(node_id, 0) > 0:
                    actors.append(actor_data)
        
        # Sort by influence score (descending)
        actors.sort(key=lambda x: (x["influence_score"], x["auth_count"]), reverse=True)
        
        # Limit results
        return actors[:limit]

    def analyze_democratic_erosion(self, days: int = 30) -> Dict[str, Any]:
        """
        Analyze democratic erosion patterns in the knowledge graph.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Dictionary with democratic erosion analysis
        """
        # Get authoritarian trends
        trends = self.get_authoritarian_trends(days)
        
        # Calculate date threshold
        threshold_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Initialize result
        result = {
            "erosion_score": trends.get("trend_score", 0),
            "risk_level": "Low",
            "affected_branches": {
                "executive": [],
                "legislative": [],
                "judicial": []
            },
            "patterns": [],
            "period_days": days,
            "timestamp": datetime.now().isoformat()
        }
        
        # Set risk level based on erosion score
        score = trends.get("trend_score", 0)
        if score < 3:
            result["risk_level"] = "Low"
        elif score < 6:
            result["risk_level"] = "Moderate"
        elif score < 8:
            result["risk_level"] = "High"
        else:
            result["risk_level"] = "Severe"
        
        # Collect affected institutions by branch
        for institution in trends.get("affected_institutions", []):
            attrs = institution.get("attributes", {})
            branch = attrs.get("branch", "unknown").lower()
            
            if branch in result["affected_branches"]:
                result["affected_branches"][branch].append({
                    "id": institution.get("id"),
                    "name": institution.get("name"),
                    "attributes": attrs
                })
        
        # Identify common patterns
        # 1. Judicial interference
        judicial_count = len(result["affected_branches"]["judicial"])
        if judicial_count > 0:
            result["patterns"].append({
                "name": "Judicial Interference",
                "severity": min(10, judicial_count * 2),
                "description": f"Detected {judicial_count} instances of interference with judicial institutions",
                "indicator": "Pattern of undermining independence of courts or judges"
            })
        
        # 2. Media control
        media_nodes = self.get_nodes_by_type("media_outlet")
        censored_media = []
        
        for media in media_nodes:
            # Check if this media outlet is being censored or delegitimized
            for _, target, data in self.graph.in_edges(media.get("id"), data=True):
                if data.get("relation") in ["censors", "delegitimizes"]:
                    edge_timestamp = data.get("timestamp", "")
                    if edge_timestamp >= threshold_date:
                        censored_media.append(media)
                        break
        
        if censored_media:
            result["patterns"].append({
                "name": "Media Control",
                "severity": min(10, len(censored_media) * 2),
                "description": f"Detected {len(censored_media)} instances of media censorship or delegitimization",
                "indicator": "Pattern of controlling or silencing media outlets"
            })
        
        # 3. Opposition targeting
        opposition_targeting = []
        
        for source, target, data in self.graph.edges(data=True):
            if data.get("relation") in ["criminalizes", "intimidates", "purges"]:
                edge_timestamp = data.get("timestamp", "")
                if edge_timestamp >= threshold_date:
                    source_data = self.graph.nodes[source]
                    target_data = self.graph.nodes[target]
                    
                    # Check if target is opposition or civil society
                    if target_data.get("type") in ["actor", "civil_society"]:
                        opposition_targeting.append({
                            "source": source_data.get("name"),
                            "target": target_data.get("name"),
                            "relation": data.get("relation"),
                            "evidence": data.get("evidence_quote", "")
                        })
        
        if opposition_targeting:
            result["patterns"].append({
                "name": "Opposition Targeting",
                "severity": min(10, len(opposition_targeting) * 2),
                "description": f"Detected {len(opposition_targeting)} instances of targeting opposition or civil society",
                "indicator": "Pattern of criminalizing, intimidating, or purging opposition voices"
            })
        
        # 4. Institutional capture
        institutional_capture = []
        
        for source, target, data in self.graph.edges(data=True):
            if data.get("relation") in ["co-opts", "restricts"]:
                edge_timestamp = data.get("timestamp", "")
                if edge_timestamp >= threshold_date:
                    target_data = self.graph.nodes[target]
                    
                    # Check if target is an institution
                    if target_data.get("type") == "institution":
                        institutional_capture.append({
                            "institution": target_data.get("name"),
                            "relation": data.get("relation"),
                            "evidence": data.get("evidence_quote", "")
                        })
        
        if institutional_capture:
            result["patterns"].append({
                "name": "Institutional Capture",
                "severity": min(10, len(institutional_capture) * 2),
                "description": f"Detected {len(institutional_capture)} instances of institutional capture or restriction",
                "indicator": "Pattern of co-opting or restricting the independence of institutions"
            })
        
        # Sort patterns by severity
        result["patterns"].sort(key=lambda x: x["severity"], reverse=True)
        
        return result

    def detect_coordination_patterns(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Detect coordination patterns between actors in the knowledge graph.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of coordination pattern dictionaries
        """
        # Calculate date threshold
        threshold_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Get all actor nodes
        actors = self.get_nodes_by_type("actor")
        
        # Create a subgraph of only recent edges
        recent_edges = []
        for source, target, data in self.graph.edges(data=True):
            edge_timestamp = data.get("timestamp", "")
            if edge_timestamp >= threshold_date:
                recent_edges.append((source, target, data))
        
        # Build action-target map
        action_map = {}
        for source, target, data in recent_edges:
            source_data = self.graph.nodes.get(source, {})
            if source_data.get("type") != "actor":
                continue
                
            source_name = source_data.get("name", "Unknown")
            
            # Group by relation and target
            relation = data.get("relation", "unknown")
            target_data = self.graph.nodes.get(target, {})
            target_name = target_data.get("name", "Unknown")
            
            key = f"{relation}:{target}"
            if key not in action_map:
                action_map[key] = {
                    "relation": relation,
                    "target": target_name,
                    "actors": []
                }
            
            action_map[key]["actors"].append(source_name)
        
        # Find coordinated actions (multiple actors targeting same entity with same relation)
        coordination_patterns = []
        
        for key, data in action_map.items():
            if len(data["actors"]) >= 2:
                coordination_patterns.append({
                    "actors": data["actors"],
                    "relation": data["relation"],
                    "target": data["target"],
                    "actor_count": len(data["actors"]),
                    "timestamp": datetime.now().isoformat()
                })
        
        # Sort by actor count (descending)
        coordination_patterns.sort(key=lambda x: x["actor_count"], reverse=True)
        
        return coordination_patterns

    def get_entity_network(self, entity_id: str, depth: int = 2) -> Dict[str, Any]:
        """
        Get the network around a specific entity.
        
        Args:
            entity_id: Entity node ID
            depth: Depth of network to retrieve
            
        Returns:
            Dictionary with entity network data
        """
        if not self.graph.has_node(entity_id):
            return {
                "entity_id": entity_id,
                "found": False,
                "nodes": [],
                "edges": []
            }
        
        # Create a subgraph with the neighborhood around the entity
        nodes_to_visit = {entity_id}
        visited_nodes = set()
        network_nodes = set()
        network_edges = []
        
        # BFS to explore the graph
        for _ in range(depth):
            next_nodes = set()
            
            for node_id in nodes_to_visit:
                if node_id in visited_nodes:
                    continue
                    
                network_nodes.add(node_id)
                visited_nodes.add(node_id)
                
                # Add outgoing edges
                for _, target in self.graph.out_edges(node_id):
                    edge_data = self.graph.edges[node_id, target]
                    network_edges.append({
                        "source": node_id,
                        "target": target,
                        "relation": edge_data.get("relation", "unknown"),
                        "attributes": edge_data.get("attributes", {})
                    })
                    next_nodes.add(target)
                
                # Add incoming edges
                for source, _ in self.graph.in_edges(node_id):
                    edge_data = self.graph.edges[source, node_id]
                    network_edges.append({
                        "source": source,
                        "target": node_id,
                        "relation": edge_data.get("relation", "unknown"),
                        "attributes": edge_data.get("attributes", {})
                    })
                    next_nodes.add(source)
            
            nodes_to_visit = next_nodes
        
        # Get full node data
        network_node_data = []
        for node_id in network_nodes:
            node_data = self.graph.nodes[node_id]
            network_node_data.append({
                "id": node_id,
                "type": node_data.get("type", "unknown"),
                "name": node_data.get("name", "Unknown"),
                "attributes": node_data.get("attributes", {})
            })
        
        return {
            "entity_id": entity_id,
            "entity_name": self.graph.nodes[entity_id].get("name", "Unknown"),
            "found": True,
            "nodes": network_node_data,
            "edges": network_edges,
            "depth": depth
        }

    def search_entities(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for entities in the knowledge graph.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching entity dictionaries
        """
        query = query.lower()
        results = []
        
        for node_id, data in self.graph.nodes(data=True):
            name = data.get("name", "").lower()
            
            # Check name match
            if query in name:
                score = 2 if name.startswith(query) else 1
                
                results.append({
                    "id": node_id,
                    "name": data.get("name", "Unknown"),
                    "type": data.get("type", "unknown"),
                    "match_score": score,
                    "attributes": data.get("attributes", {})
                })
                
                if len(results) >= limit * 2:
                    break
        
        # Sort by match score
        results.sort(key=lambda x: x["match_score"], reverse=True)
        
        return results[:limit]

    def export_graph(self, format: str = "json") -> str:
        """
        Export the knowledge graph to a file.
        
        Args:
            format: Export format (json or gexf)
            
        Returns:
            Path to the exported file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == "gexf":
            # Export as GEXF (for visualization in Gephi)
            filepath = os.path.join(self.base_dir, f"knowledge_graph_{timestamp}.gexf")
            nx.write_gexf(self.graph, filepath)
            return filepath
        else:
            # Export as JSON (default)
            filepath = os.path.join(self.base_dir, f"knowledge_graph_{timestamp}.json")
            
            # Prepare graph data
            graph_data = {
                "nodes": {},
                "edges": []
            }
            
            # Add nodes
            for node_id, data in self.graph.nodes(data=True):
                graph_data["nodes"][node_id] = data
            
            # Add edges
            for source, target, data in self.graph.edges(data=True):
                edge_data = data.copy()
                edge_data["source"] = source
                edge_data["target"] = target
                graph_data["edges"].append(edge_data)
            
            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2, ensure_ascii=False)
            
            return filepath

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph.
        
        Returns:
            Dictionary with statistics
        """
        # Count node types
        node_types = {}
        for _, data in self.graph.nodes(data=True):
            node_type = data.get("type", "unknown")
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        # Count relation types
        relation_types = {}
        for _, _, data in self.graph.edges(data=True):
            relation = data.get("relation", "unknown")
            relation_types[relation] = relation_types.get(relation, 0) + 1
        
        # Get graph metrics
        density = nx.density(self.graph)
        
        # Get most connected nodes
        degree_centrality = nx.degree_centrality(self.graph)
        sorted_centrality = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
        top_nodes = []
        
        for node_id, score in sorted_centrality[:5]:
            node_data = self.graph.nodes[node_id]
            top_nodes.append({
                "id": node_id,
                "name": node_data.get("name", "Unknown"),
                "type": node_data.get("type", "unknown"),
                "centrality": round(score, 3)
            })
        
        return {
            "node_count": len(self.graph.nodes),
            "edge_count": len(self.graph.edges),
            "node_types": node_types,
            "relation_types": relation_types,
            "graph_density": density,
            "top_connected_nodes": top_nodes,
            "timestamp": datetime.now().isoformat()
        }
