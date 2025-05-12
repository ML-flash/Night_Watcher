"""
Night_watcher Knowledge Graph
Manages the knowledge graph for storing entities, events, and relationships with dynamic relation type handling.
"""

import os
import json
import logging
import csv
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Union, Tuple
import networkx as nx

# Configure logging
logger = logging.getLogger(__name__)

class KnowledgeGraph:
    """
    Knowledge graph for storing entities, events, and relationships.
    Implements a graph-based data structure using NetworkX with JSON serialization.
    Dynamically accepts new relation types to enable discovery of authoritarian patterns.
    """

    # Define relation types that should be considered standard but won't block new types
    COMMON_RELATIONS = {
        'mentions', 'criticizes', 'influences', 'opposes', 'supports', 'restricts',
        'undermines', 'authorizes', 'co-opts', 'purges', 'criminalizes', 'censors',
        'intimidates', 'delegitimizes', 'backs_with_force', 'precedes', 'follows',
        'justifies', 'expands_power', 'normalizes', 'diverts_attention', 'targets',
        'firing', 'accuses', 'results_from', 'requests', 'uses', 'visits', 'threatens',
        'imposes', 'pardons', 'part_of'
    }

    # Define relation types that shouldn't cause repeated warnings
    LOW_PRIORITY_RELATIONS = {'mentions'}

    # Define which relations to include when connecting media sources to entities
    # Setting this to False will prevent the creation of large numbers of 'mentions' edges
    # that can clutter the graph and make pattern detection more difficult
    CREATE_MENTIONS_EDGES = False

    def __init__(self, base_dir: str = "data/knowledge_graph", taxonomy_path: str = "KG_Taxonomy.csv",
                 graph_file: str = None, taxonomy_file: str = None, dynamic_relations: bool = True):
        """
        Initialize the knowledge graph.

        Args:
            base_dir: Base directory for knowledge graph storage
            taxonomy_path: Path to taxonomy CSV file (deprecated, use taxonomy_file)
            graph_file: Path to graph file for initialization
            taxonomy_file: Path to taxonomy CSV file
            dynamic_relations: Whether to allow dynamic relation types (default: True)
        """
        self.base_dir = base_dir
        self.nodes_dir = os.path.join(base_dir, "nodes")
        self.edges_dir = os.path.join(base_dir, "edges")
        self.snapshots_dir = os.path.join(base_dir, "snapshots")

        # Setup logging
        self.logger = logging.getLogger("KnowledgeGraph")

        # Support for both initialization methods (backward compatibility)
        self.taxonomy_path = taxonomy_file if taxonomy_file else taxonomy_path
        self.graph_file = graph_file

        # Track dynamic relation types discovered during analysis
        self.dynamic_relations = dynamic_relations
        self.discovered_relation_types = set()
        self._warned_relation_types = set()

        # Initialize NetworkX graph
        self.graph = nx.DiGraph()

        # Load taxonomy if available
        self.taxonomy = self._load_taxonomy()

        # Node and edge counters for ID generation
        self._node_counter = 0
        self._edge_counter = 0

        # Create directories if they don't exist
        os.makedirs(self.nodes_dir, exist_ok=True)
        os.makedirs(self.edges_dir, exist_ok=True)
        os.makedirs(self.snapshots_dir, exist_ok=True)

        # Load existing graph if available
        if graph_file and os.path.exists(graph_file):
            self._load_graph_from_file(graph_file)
            self.logger.info(f"Loaded knowledge graph from {graph_file}")
        else:
            self._load_graph()
            self.logger.info(f"Loaded knowledge graph from directory structure at {base_dir}")

        self.logger.info(f"Knowledge graph initialized: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")

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
        Load the existing graph from the node and edge files in the directory structure.
        """
        # Load nodes
        node_count = 0
        if os.path.exists(self.nodes_dir):
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
        if os.path.exists(self.edges_dir):
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
                        relation = edge_data.get("relation")

                        if relation and relation not in self.taxonomy["relation_types"] and relation not in self.discovered_relation_types:
                            # Add to discovered relations
                            if self.dynamic_relations:
                                self.discovered_relation_types.add(relation)

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

        self.logger.info(f"Loaded {node_count} nodes and {edge_count} edges from directory structure")

    def _load_graph_from_file(self, graph_file: str) -> None:
        """
        Load the graph from a single JSON file.

        Args:
            graph_file: Path to the graph JSON file
        """
        try:
            with open(graph_file, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)

            # Clear existing graph
            self.graph.clear()

            # Load nodes
            nodes = graph_data.get("nodes", {})
            for node_id, node_data in nodes.items():
                self.graph.add_node(node_id, **node_data)

                # Update node counter
                numeric_id = self._extract_numeric_id(node_id)
                if numeric_id > self._node_counter:
                    self._node_counter = numeric_id

            # Load edges
            edges = graph_data.get("edges", [])
            for edge_data in edges:
                source_id = edge_data.get("source")
                target_id = edge_data.get("target")
                relation = edge_data.get("relation")

                if relation and relation not in self.taxonomy["relation_types"] and relation not in self.discovered_relation_types:
                    # Add to discovered relations
                    if self.dynamic_relations:
                        self.discovered_relation_types.add(relation)

                if source_id and target_id:
                    # Remove source and target from edge data to avoid duplication
                    edge_attrs = {k: v for k, v in edge_data.items() if k not in ["source", "target"]}
                    self.graph.add_edge(source_id, target_id, **edge_attrs)

                    # Update edge counter if id is present
                    if "id" in edge_data:
                        numeric_id = self._extract_numeric_id(edge_data["id"])
                        if numeric_id > self._edge_counter:
                            self._edge_counter = numeric_id

            self.logger.info(f"Loaded graph from file: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
        except Exception as e:
            self.logger.error(f"Error loading graph from file {graph_file}: {e}")
            # Initialize empty graph
            self.graph.clear()

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
        # Handle dynamic relation types
        is_known_relation = relation in self.taxonomy["relation_types"] or relation in self.COMMON_RELATIONS

        # Log warning for unknown relation types only once per type
        if not is_known_relation and relation not in self.discovered_relation_types and relation not in self._warned_relation_types:
            if relation not in self.LOW_PRIORITY_RELATIONS:
                self.logger.warning(f"Relation type '{relation}' not in taxonomy")
            self._warned_relation_types.add(relation)

            if self.dynamic_relations:
                self.discovered_relation_types.add(relation)

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
            if relation not in self.LOW_PRIORITY_RELATIONS:
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

            # Extract existing edge ID
            edge_id = existing_data.get("id", "")

            # Save updated edge
            if edge_id:
                self._save_edge(edge_id, existing_data)

            return edge_id

        # Add edge to graph
        edge_attrs = {k: v for k, v in edge_data.items() if k != 'id'}  # Remove id to avoid duplicate
        self.graph.add_edge(source_id, target_id, **edge_attrs)

        # Save edge to disk
        self._save_edge(edge_id, edge_data)

        if relation not in self.LOW_PRIORITY_RELATIONS:
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

    def process_article_analysis(self, article: Dict[str, Any], kg_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an article analysis and update the knowledge graph.

        Args:
            article: Article metadata
            kg_analysis: Knowledge graph analysis data with nodes and edges

        Returns:
            Dictionary with processing results
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
            nodes_added = 0
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

                nodes_added += 1

            # Add article-entity edges
            mentions_edges = 0
            if self.CREATE_MENTIONS_EDGES:
                for node_id in node_id_map.values():
                    # Connect article to mentioned entities
                    self.add_edge(
                        source_id=article_node_id,
                        relation="mentions",
                        target_id=node_id,
                        timestamp=publication_date,
                        source_document_id=document_id
                    )
                    mentions_edges += 1

            # Add edges between entities
            edges_added = 0
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
                    attributes=edge.get("attributes", {})
                )
                edges_added += 1

            self.logger.info(f"Processed article analysis: {article.get('title')} - added {nodes_added} nodes, {edges_added} edges")

            return {
                "status": "success",
                "article_id": article_id,
                "document_id": document_id,
                "article_node_id": article_node_id,
                "nodes_added": nodes_added,
                "edges_added": edges_added,
                "mentions_edges": mentions_edges
            }

        except Exception as e:
            self.logger.error(f"Error processing article analysis: {e}")
            return {
                "status": "error",
                "article_id": article.get("id"),
                "error": str(e),
                "nodes_added": 0,
                "edges_added": 0
            }

    def export_graph(self, format: str = "json", filepath: str = None) -> str:
        """
        Export the knowledge graph to a file.

        Args:
            format: Export format (json or gexf)
            filepath: Path to save the file (optional)

        Returns:
            Path to the exported file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Use provided filepath or generate one
        if not filepath:
            if format.lower() == "gexf":
                filepath = os.path.join(self.base_dir, f"knowledge_graph_{timestamp}.gexf")
            else:
                filepath = os.path.join(self.base_dir, f"knowledge_graph_{timestamp}.json")

        if format.lower() == "gexf":
            # Export as GEXF (for visualization in Gephi)
            nx.write_gexf(self.graph, filepath)
            self.logger.info(f"Exported graph as GEXF to {filepath}")
            return filepath
        else:
            # Export as JSON (default)
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

            self.logger.info(f"Exported graph as JSON to {filepath}")
            return filepath

    def save_graph(self, filepath: str = None) -> str:
        """
        Save the current graph to a JSON file.

        Args:
            filepath: Path to save the file (optional)

        Returns:
            Path to the saved file
        """
        # Use provided filepath or default
        if not filepath:
            filepath = os.path.join(self.base_dir, "graph.json")

        return self.export_graph(format="json", filepath=filepath)

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

    def get_basic_statistics(self) -> Dict[str, Any]:
        """
        Get basic statistics about the knowledge graph.

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
        try:
            density = nx.density(self.graph)
        except:
            density = 0

        return {
            "node_count": len(self.graph.nodes),
            "edge_count": len(self.graph.edges),
            "node_types": node_types,
            "relation_types": relation_types,
            "graph_density": density,
            "discovered_relations": list(self.discovered_relation_types),
            "timestamp": datetime.now().isoformat()
        }

    def infer_temporal_relationships(self) -> int:
        """
        Infer temporal relationships ('precedes' and 'follows') between events based on timestamps.

        Returns:
            Number of temporal relationships added
        """
        # Get all event nodes with timestamps
        event_nodes = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") == "event" and "timestamp" in data:
                try:
                    timestamp = data["timestamp"]
                    if timestamp and timestamp != "N/A":
                        event_nodes.append((node_id, timestamp, data.get("name", "")))
                except:
                    pass

        # Sort events by timestamp
        event_nodes.sort(key=lambda x: x[1])

        # Don't process if too few events
        if len(event_nodes) < 2:
            return 0

        # Add temporal relationships
        relationships_added = 0

        # Connect events that occurred within a reasonable time window (30 days)
        for i in range(len(event_nodes) - 1):
            for j in range(i + 1, min(i + 5, len(event_nodes))):
                try:
                    earlier_id, earlier_timestamp, earlier_name = event_nodes[i]
                    later_id, later_timestamp, later_name = event_nodes[j]

                    # Skip if already connected
                    if self.graph.has_edge(earlier_id, later_id) or self.graph.has_edge(later_id, earlier_id):
                        continue

                    # Parse dates
                    earlier_date = datetime.fromisoformat(earlier_timestamp.split("T")[0])
                    later_date = datetime.fromisoformat(later_timestamp.split("T")[0])

                    # Calculate days between
                    days_between = (later_date - earlier_date).days

                    # Only connect if within a reasonable time window (1-30 days)
                    if 0 < days_between <= 30:
                        # Add 'precedes' relationship
                        self.add_edge(
                            source_id=earlier_id,
                            relation="precedes",
                            target_id=later_id,
                            attributes={"days_between": days_between},
                            timestamp=earlier_timestamp
                        )

                        # Add 'follows' relationship
                        self.add_edge(
                            source_id=later_id,
                            relation="follows",
                            target_id=earlier_id,
                            attributes={"days_between": days_between},
                            timestamp=later_timestamp
                        )

                        relationships_added += 2
                except Exception as e:
                    self.logger.warning(f"Error inferring temporal relationships: {e}")

        return relationships_added

    def get_authoritarian_trends(self, days: int = 90) -> Dict[str, Any]:
        """
        Analyze authoritarian trends in the knowledge graph over the specified time period.

        Args:
            days: Number of days to analyze (default: 90)

        Returns:
            Dictionary with trend analysis
        """
        # Define authoritarian relation types
        authoritarian_relations = [
            "undermines", "co-opts", "purges", "criminalizes", "censors",
            "intimidates", "delegitimizes", "restricts", "targets"
        ]

        # Calculate start date
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Count authoritarian edges by day
        daily_counts = {}
        relation_counts = {relation: 0 for relation in authoritarian_relations}

        # Collect edges within date range
        for source, target, data in self.graph.edges(data=True):
            relation = data.get("relation", "")

            if relation in authoritarian_relations:
                timestamp = data.get("timestamp")
                if timestamp and timestamp != "N/A":
                    try:
                        # Parse date
                        date = datetime.fromisoformat(timestamp.split("T")[0])

                        # Check if within range
                        if start_date <= date <= end_date:
                            # Update daily count
                            date_str = date.strftime("%Y-%m-%d")
                            if date_str not in daily_counts:
                                daily_counts[date_str] = {rel: 0 for rel in authoritarian_relations}

                            # Update counts
                            daily_counts[date_str][relation] = daily_counts[date_str].get(relation, 0) + 1
                            relation_counts[relation] += 1
                    except:
                        pass

        # Calculate authoritarian score (0-10)
        total_edges = sum(relation_counts.values())
        score = 0

        if total_edges > 0:
            # Weight different relations
            weights = {
                "undermines": 0.8,
                "co-opts": 1.0,
                "purges": 1.0,
                "criminalizes": 0.9,
                "censors": 0.9,
                "intimidates": 0.8,
                "delegitimizes": 0.7,
                "restricts": 0.7,
                "targets": 0.8
            }

            # Calculate weighted score
            weighted_sum = sum(weights.get(rel, 0.5) * count for rel, count in relation_counts.items())

            # Scale to 0-10
            score = min(10, max(0, weighted_sum / (total_edges * 0.5) * 5))

        return {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "days_analyzed": days,
            "authoritarian_score": round(score, 1),
            "total_authoritarian_actions": total_edges,
            "relation_counts": relation_counts,
            "daily_counts": daily_counts,
            "timestamp": datetime.now().isoformat()
        }

    def analyze_democratic_erosion(self, days: int = 90) -> Dict[str, Any]:
        """
        Analyze democratic erosion patterns in the knowledge graph.

        Args:
            days: Number of days to analyze (default: 90)

        Returns:
            Dictionary with erosion analysis
        """
        # Calculate start date
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Track democratic erosion patterns
        erosion_patterns = {
            "executive_expansion": 0,  # expands_power relations
            "judicial_targeting": 0,   # targets/undermines relations with judiciary
            "media_control": 0,        # censors/intimidates relations with media
            "opposition_delegitimization": 0,  # delegitimizes relations
            "rights_restriction": 0,   # restricts relations with civil liberties
            "norm_violations": 0       # undermines relations with procedural_norm
        }

        # Track affected institutions
        affected_institutions = {}

        # Collect edges within date range
        for source, target, data in self.graph.edges(data=True):
            relation = data.get("relation", "")
            timestamp = data.get("timestamp")

            if timestamp and timestamp != "N/A":
                try:
                    # Parse date
                    date = datetime.fromisoformat(timestamp.split("T")[0])

                    # Check if within range
                    if start_date <= date <= end_date:
                        # Get source and target node types
                        source_type = self.graph.nodes[source].get("type", "")
                        target_type = self.graph.nodes[target].get("type", "")
                        target_name = self.graph.nodes[target].get("name", "")

                        # Check for erosion patterns
                        if relation == "expands_power" and source_type in ["actor", "policy"]:
                            erosion_patterns["executive_expansion"] += 1

                        elif relation in ["targets", "undermines"] and target_type == "institution":
                            # Track affected institution
                            if target_name not in affected_institutions:
                                affected_institutions[target_name] = 0
                            affected_institutions[target_name] += 1

                            # Check if judiciary
                            if "court" in target_name.lower() or "judge" in target_name.lower() or "judicial" in target_name.lower():
                                erosion_patterns["judicial_targeting"] += 1

                        elif relation in ["censors", "intimidates"] and target_type == "media_outlet":
                            erosion_patterns["media_control"] += 1

                        elif relation == "delegitimizes":
                            erosion_patterns["opposition_delegitimization"] += 1

                        elif relation == "restricts" and source_type in ["actor", "policy"]:
                            erosion_patterns["rights_restriction"] += 1

                        elif relation == "undermines" and target_type == "procedural_norm":
                            erosion_patterns["norm_violations"] += 1
                except:
                    pass

        # Calculate erosion score (0-10)
        total_patterns = sum(erosion_patterns.values())
        erosion_score = 0

        if total_patterns > 0:
            # Weight different patterns
            weights = {
                "executive_expansion": 0.9,
                "judicial_targeting": 1.0,
                "media_control": 0.9,
                "opposition_delegitimization": 0.7,
                "rights_restriction": 0.8,
                "norm_violations": 0.7
            }

            # Calculate weighted score
            weighted_sum = sum(weights[pattern] * count for pattern, count in erosion_patterns.items())

            # Scale to 0-10
            erosion_score = min(10, max(0, weighted_sum / (total_patterns * 0.5) * 5))

        # Determine concern level
        concern_level = "None"
        if erosion_score > 8:
            concern_level = "Very High"
        elif erosion_score > 6:
            concern_level = "High"
        elif erosion_score > 4:
            concern_level = "Moderate"
        elif erosion_score > 2:
            concern_level = "Low"

        return {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "days_analyzed": days,
            "erosion_score": round(erosion_score, 1),
            "concern_level": concern_level,
            "erosion_patterns": erosion_patterns,
            "total_erosion_actions": total_patterns,
            "affected_institutions": affected_institutions,
            "timestamp": datetime.now().isoformat()
        }

    def get_influential_actors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Identify the most influential actors in the knowledge graph based on centrality metrics.

        Args:
            limit: Maximum number of actors to return (default: 10)

        Returns:
            List of influential actor dictionaries
        """
        # Get actor nodes
        actor_nodes = {}
        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") == "actor":
                actor_nodes[node_id] = data.get("name", "Unknown")

        if not actor_nodes:
            return []

        # Calculate centrality metrics
        try:
            # Degree centrality
            degree_centrality = nx.degree_centrality(self.graph)

            # Betweenness centrality
            betweenness_centrality = nx.betweenness_centrality(self.graph)

            # Eigenvector centrality
            eigenvector_centrality = nx.eigenvector_centrality(self.graph, max_iter=1000, tol=1e-3)
        except:
            # Fallback if centrality calculation fails
            degree_centrality = {}
            betweenness_centrality = {}
            eigenvector_centrality = {}

            # Calculate simple degree for all nodes
            for node_id in self.graph.nodes:
                degree_centrality[node_id] = len(list(self.graph.neighbors(node_id))) / max(1, len(self.graph.nodes) - 1)

        # Calculate influence scores for actors
        actor_influence = []

        for node_id, name in actor_nodes.items():
            # Get centrality values
            degree = degree_centrality.get(node_id, 0)
            betweenness = betweenness_centrality.get(node_id, 0)
            eigenvector = eigenvector_centrality.get(node_id, 0)

            # Calculate influence score (weighted average)
            influence_score = (degree * 0.3) + (betweenness * 0.4) + (eigenvector * 0.3)

            # Get outgoing relation types
            outgoing_relations = []
            for _, target in self.graph.out_edges(node_id):
                relation = self.graph.edges[node_id, target].get("relation", "unknown")
                outgoing_relations.append(relation)

            # Count authoritarian actions
            authoritarian_actions = sum(1 for rel in outgoing_relations if rel in [
                "undermines", "co-opts", "purges", "criminalizes", "censors",
                "intimidates", "delegitimizes", "restricts", "targets"
            ])

            actor_influence.append({
                "id": node_id,
                "name": name,
                "influence_score": influence_score,
                "degree_centrality": degree,
                "betweenness_centrality": betweenness,
                "eigenvector_centrality": eigenvector,
                "outgoing_edges": len(outgoing_relations),
                "authoritarian_actions": authoritarian_actions
            })

        # Sort by influence score
        actor_influence.sort(key=lambda x: x["influence_score"], reverse=True)

        return actor_influence[:limit]

    def generate_intelligence_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive intelligence report from the knowledge graph.

        Returns:
            Dictionary with intelligence report data
        """
        # Get basic statistics
        stats = self.get_basic_statistics()

        # Get authoritarian trends (last 90 days)
        trends = self.get_authoritarian_trends(days=90)

        # Get democratic erosion analysis
        erosion = self.analyze_democratic_erosion(days=90)

        # Get influential actors
        actors = self.get_influential_actors(limit=10)

        # Generate report
        report = {
            "title": "Night_watcher Intelligence Report",
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "nodes": stats["node_count"],
                "edges": stats["edge_count"],
                "authoritarian_score": trends["authoritarian_score"],
                "erosion_score": erosion["erosion_score"],
                "concern_level": erosion["concern_level"]
            },
            "statistics": stats,
            "authoritarian_trends": trends,
            "democratic_erosion": erosion,
            "influential_actors": actors
        }

        return report

    def visualize_network(self, output_file: str = None) -> Dict[str, Any]:
        """
        Generate visualization data for the knowledge graph.

        Args:
            output_file: Optional path to save visualization data

        Returns:
            Dictionary with visualization data
        """
        # Convert graph to visualization format
        viz_data = {
            "nodes": [],
            "links": []
        }

        # Node type colors
        node_colors = {
            "actor": "#3498db",        # Blue
            "institution": "#9b59b6",  # Purple
            "policy": "#2ecc71",       # Green
            "event": "#e74c3c",        # Red
            "media_outlet": "#f39c12", # Orange
            "civil_society": "#1abc9c", # Teal
            "narrative": "#f1c40f",    # Yellow
            "legal_framework": "#34495e", # Dark blue
            "procedural_norm": "#95a5a6" # Gray
        }

        # Relation type colors
        relation_colors = {
            "undermines": "#e74c3c",    # Red
            "co-opts": "#9b59b6",       # Purple
            "purges": "#c0392b",        # Dark red
            "criminalizes": "#d35400",  # Brown
            "censors": "#e67e22",       # Orange
            "intimidates": "#f39c12",   # Light orange
            "delegitimizes": "#f1c40f",  # Yellow
            "restricts": "#8e44ad",     # Dark purple
            "targets": "#d35400"        # Brown
        }

        # Add nodes
        for node_id, data in self.graph.nodes(data=True):
            node_type = data.get("type", "unknown")

            viz_data["nodes"].append({
                "id": node_id,
                "name": data.get("name", "Unknown"),
                "type": node_type,
                "color": node_colors.get(node_type, "#7f8c8d"),  # Default: Gray
                "attributes": data.get("attributes", {})
            })

        # Add links (edges)
        for source, target, data in self.graph.edges(data=True):
            relation = data.get("relation", "unknown")

            viz_data["links"].append({
                "source": source,
                "target": target,
                "relation": relation,
                "color": relation_colors.get(relation, "#7f8c8d"),  # Default: Gray
                "attributes": data.get("attributes", {})
            })

        # Save to file if specified
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(viz_data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                self.logger.error(f"Error saving visualization data to {output_file}: {e}")

        return viz_data