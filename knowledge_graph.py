"""
Night_watcher Knowledge Graph
Manages the knowledge graph for storing entities, events, and relationships with dynamic relation types.
"""

import os
import json
import logging
import csv
import hashlib
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Union, Tuple
import networkx as nx

# Configure logging
logger = logging.getLogger(__name__)

class KnowledgeGraph:
    """
    Knowledge graph for storing entities, events, and relationships.
    Implements a graph-based data structure using NetworkX with JSON serialization.
    Dynamically accepts new node and relation types for flexibility.
    """

    def __init__(self, base_dir: str = "data/knowledge_graph", taxonomy_path: str = "KG_Taxonomy.csv",
                 graph_file: str = None, taxonomy_file: str = None):
        """
        Initialize the knowledge graph.

        Args:
            base_dir: Base directory for knowledge graph storage
            taxonomy_path: Path to taxonomy CSV file (deprecated, use taxonomy_file)
            graph_file: Path to graph file for initialization
            taxonomy_file: Path to taxonomy CSV file
        """
        self.base_dir = base_dir
        self.nodes_dir = os.path.join(base_dir, "nodes")
        self.edges_dir = os.path.join(base_dir, "edges")
        self.snapshots_dir = os.path.join(base_dir, "snapshots")
        self.provenance_dir = os.path.join(base_dir, "provenance")

        # Setup logging
        self.logger = logging.getLogger("KnowledgeGraph")

        # Support for both initialization methods (backward compatibility)
        self.taxonomy_path = taxonomy_file if taxonomy_file else taxonomy_path
        self.graph_file = graph_file

        # Track discovered types for dynamic schema
        self.discovered_relation_types = set()
        self.discovered_node_types = set()

        # Initialize NetworkX graph
        self.graph = nx.DiGraph()
        
        # Graph provenance tracking
        self.graph_id = None
        self.graph_version = 0
        self.provenance_chain = []
        self.parent_graph_id = None

        # Create directories if they don't exist
        os.makedirs(self.nodes_dir, exist_ok=True)
        os.makedirs(self.edges_dir, exist_ok=True)
        os.makedirs(self.snapshots_dir, exist_ok=True)
        os.makedirs(self.provenance_dir, exist_ok=True)

        # Load taxonomy if available
        self.taxonomy = self._load_taxonomy()

        # Node and edge counters for ID generation
        self._node_counter = 0
        self._edge_counter = 0

        # Load existing graph if available
        if graph_file and os.path.exists(graph_file):
            self._load_graph_from_file(graph_file)
            self.logger.info(f"Loaded knowledge graph from {graph_file}")
        else:
            self._load_graph()
            self.logger.info(f"Loaded knowledge graph from directory structure at {base_dir}")
            
        # Initialize graph ID if not already set
        if not self.graph_id:
            self._initialize_graph_provenance()

    def _initialize_graph_provenance(self):
        """Initialize the graph provenance with a new ID if not already set"""
        if not self.graph_id:
            self.graph_id = f"graph_{uuid.uuid4().hex[:16]}"
            self.graph_version = 1
            self.provenance_chain = [{
                "timestamp": datetime.now().isoformat(),
                "operation": "initialize",
                "graph_id": self.graph_id,
                "graph_version": self.graph_version,
                "parent_graph_id": None,
                "description": "Initial graph creation"
            }]
            self.logger.info(f"Initialized new graph with ID: {self.graph_id}, version: {self.graph_version}")
            
            # Save initial provenance record
            self._save_provenance_record()

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
                        self.discovered_node_types.add(name)
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
                            self.discovered_relation_types.add(name)

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

                        # Track node type
                        node_type = node_data.get("type", "unknown")
                        if node_type not in self.discovered_node_types:
                            self.discovered_node_types.add(node_type)

                        # Update node counter
                        numeric_id = int(node_id.split('_')[-1]) if '_' in node_id else 0
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

                        # Track relation type
                        if relation and relation not in self.discovered_relation_types:
                            self.discovered_relation_types.add(relation)

                        if source_id and target_id:
                            # Add edge to graph
                            self.graph.add_edge(source_id, target_id, id=edge_id, **edge_data)

                            # Update edge counter
                            numeric_id = int(edge_id.split('_')[-1]) if '_' in edge_id else 0
                            if numeric_id > self._edge_counter:
                                self._edge_counter = numeric_id

                            edge_count += 1
                    except Exception as e:
                        self.logger.error(f"Error loading edge {edge_id}: {e}")

        self.logger.info(f"Loaded {node_count} nodes and {edge_count} edges from directory structure")
        
        # Load graph provenance
        self._load_graph_provenance()

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

            # Load graph metadata and provenance
            if "metadata" in graph_data:
                metadata = graph_data.get("metadata", {})
                self.graph_id = metadata.get("graph_id")
                self.graph_version = metadata.get("graph_version", 1)
                self.parent_graph_id = metadata.get("parent_graph_id")
                self.provenance_chain = metadata.get("provenance_chain", [])
                
                if not self.graph_id:
                    self._initialize_graph_provenance()
            else:
                self._initialize_graph_provenance()

            # Load nodes
            nodes = graph_data.get("nodes", {})
            for node_id, node_data in nodes.items():
                self.graph.add_node(node_id, **node_data)
                
                # Track node type
                node_type = node_data.get("type", "unknown")
                if node_type not in self.discovered_node_types:
                    self.discovered_node_types.add(node_type)

                # Update node counter
                numeric_id = int(node_id.split('_')[-1]) if '_' in node_id else 0
                if numeric_id > self._node_counter:
                    self._node_counter = numeric_id

            # Load edges
            edges = graph_data.get("edges", [])
            for edge_data in edges:
                source_id = edge_data.get("source")
                target_id = edge_data.get("target")
                relation = edge_data.get("relation")

                # Track relation type
                if relation and relation not in self.discovered_relation_types:
                    self.discovered_relation_types.add(relation)

                if source_id and target_id:
                    # Remove source and target from edge data to avoid duplication
                    edge_attrs = {k: v for k, v in edge_data.items() if k not in ["source", "target"]}
                    self.graph.add_edge(source_id, target_id, **edge_attrs)

                    # Update edge counter if id is present
                    if "id" in edge_data:
                        numeric_id = int(edge_data["id"].split('_')[-1]) if '_' in edge_data["id"] else 0
                        if numeric_id > self._edge_counter:
                            self._edge_counter = numeric_id

            self.logger.info(f"Loaded graph from file: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
        except Exception as e:
            self.logger.error(f"Error loading graph from file {graph_file}: {e}")
            # Initialize empty graph
            self.graph.clear()
            self._initialize_graph_provenance()

    def _load_graph_provenance(self):
        """Load graph provenance from provenance file if available"""
        provenance_file = os.path.join(self.provenance_dir, "graph_provenance.json")
        if os.path.exists(provenance_file):
            try:
                with open(provenance_file, 'r', encoding='utf-8') as f:
                    provenance_data = json.load(f)
                
                self.graph_id = provenance_data.get("graph_id")
                self.graph_version = provenance_data.get("graph_version", 1)
                self.parent_graph_id = provenance_data.get("parent_graph_id")
                self.provenance_chain = provenance_data.get("provenance_chain", [])
                
                self.logger.info(f"Loaded graph provenance: ID={self.graph_id}, version={self.graph_version}")
            except Exception as e:
                self.logger.error(f"Error loading graph provenance: {e}")
                self._initialize_graph_provenance()
        else:
            self._initialize_graph_provenance()

    def _save_provenance_record(self):
        """Save the current graph provenance information"""
        provenance_data = {
            "graph_id": self.graph_id,
            "graph_version": self.graph_version,
            "parent_graph_id": self.parent_graph_id,
            "provenance_chain": self.provenance_chain,
            "last_updated": datetime.now().isoformat()
        }
        
        provenance_file = os.path.join(self.provenance_dir, "graph_provenance.json")
        try:
            with open(provenance_file, 'w', encoding='utf-8') as f:
                json.dump(provenance_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved graph provenance record: {self.graph_id} (version {self.graph_version})")
        except Exception as e:
            self.logger.error(f"Error saving provenance record: {e}")

    def update_provenance(self, operation: str, description: str, source_ids: List[str] = None):
        """
        Update the graph provenance with a new operation
        
        Args:
            operation: Type of operation performed
            description: Description of the operation
            source_ids: List of source IDs (documents, analyses) that contributed to this update
        """
        # Increment graph version
        self.graph_version += 1
        
        # Create provenance record
        record = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "graph_id": self.graph_id,
            "graph_version": self.graph_version,
            "parent_graph_id": self.parent_graph_id,
            "description": description
        }
        
        # Add source IDs if provided
        if source_ids:
            record["source_ids"] = source_ids
            
        # Add content hash of the graph
        record["content_hash"] = self._compute_graph_hash()
        
        # Add to provenance chain
        self.provenance_chain.append(record)
        
        # Save updated provenance
        self._save_provenance_record()
        
        self.logger.info(f"Updated graph provenance: {operation}, version {self.graph_version}")
        
    def _compute_graph_hash(self) -> str:
        """Compute a hash of the graph content for integrity verification"""
        # Convert graph to a stable representation for hashing
        nodes_str = json.dumps(
            {node: {k: v for k, v in data.items() if k != "timestamp" and k != "created_at"} 
             for node, data in sorted(self.graph.nodes(data=True))}, 
            sort_keys=True
        )
        
        edges_str = json.dumps(
            [{
                "source": source,
                "target": target,
                **{k: v for k, v in data.items() if k != "timestamp" and k != "created_at"}
            } for source, target, data in sorted(self.graph.edges(data=True))], 
            sort_keys=True
        )
        
        # Compute hash of combined content
        hash_obj = hashlib.sha256((nodes_str + edges_str).encode('utf-8'))
        return hash_obj.hexdigest()

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
        # Add node type to discovered types if new
        if node_type not in self.discovered_node_types:
            self.discovered_node_types.add(node_type)
            self.logger.info(f"Discovered new node type: {node_type}")

        # Check if a similar node already exists
        existing_node = self._find_similar_node(node_type, name)
        if existing_node:
            # Update existing node with any new attributes
            existing_data = self.graph.nodes[existing_node]
            if attributes:
                if "attributes" not in existing_data:
                    existing_data["attributes"] = {}
                for key, value in attributes.items():
                    if key not in existing_data["attributes"]:
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
            },
            "graph_id": self.graph_id,
            "graph_version": self.graph_version
        }

        # Add node to graph
        self.graph.add_node(node_id, **node_data)

        # Save node to disk
        self._save_node(node_id, node_data)

        self.logger.debug(f"Added node {node_id}: {node_type} - {name}")
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
        # Add relation type to discovered types if new
        if relation not in self.discovered_relation_types:
            self.discovered_relation_types.add(relation)
            self.logger.info(f"Discovered new relation type: {relation}")

        # Validate source and target nodes
        if not self.graph.has_node(source_id):
            self.logger.warning(f"Source node {source_id} not in graph")
            return ""

        if not self.graph.has_node(target_id):
            self.logger.warning(f"Target node {target_id} not in graph")
            return ""

        # Check if a similar edge already exists
        existing_edge = self._find_similar_edge(source_id, relation, target_id)
        if existing_edge:
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
                if "attributes" not in existing_data:
                    existing_data["attributes"] = {}
                for key, value in attributes.items():
                    if key not in existing_data["attributes"]:
                        existing_data["attributes"][key] = value

            # Extract existing edge ID
            edge_id = existing_data.get("id", "")

            # Save updated edge
            if edge_id:
                self._save_edge(edge_id, existing_data)

            return edge_id

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
            "attributes": attributes or {},
            "graph_id": self.graph_id,
            "graph_version": self.graph_version
        }

        # Add edge to graph
        edge_attrs = {k: v for k, v in edge_data.items() if k != 'id'}  # Remove id to avoid duplicate
        self.graph.add_edge(source_id, target_id, **edge_attrs)

        # Save edge to disk
        self._save_edge(edge_id, edge_data)

        self.logger.debug(f"Added edge {edge_id}: {source_id} --[{relation}]--> {target_id}")
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

            # Update provenance if anything was added
            if nodes_added > 0 or edges_added > 0:
                analysis_id = kg_analysis.get("provenance_id", "unknown")
                self.update_provenance(
                    operation="add_article_analysis",
                    description=f"Added analysis from {article.get('title')}",
                    source_ids=[document_id, analysis_id]
                )

            return {
                "status": "success",
                "article_id": article_id,
                "document_id": document_id,
                "article_node_id": article_node_id,
                "nodes_added": nodes_added,
                "edges_added": edges_added,
                "graph_id": self.graph_id,
                "graph_version": self.graph_version
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
            "graph_id": self.graph_id,
            "graph_version": self.graph_version,
            "content_hash": self._compute_graph_hash(),
            "nodes": {node: data for node, data in self.graph.nodes(data=True)},
            "edges": {f"{source}_{target}": data for source, target, data in self.graph.edges(data=True)}
        }

        # Save snapshot to disk
        filepath = os.path.join(self.snapshots_dir, f"{snapshot_id}.json")
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(snapshot_data, f, indent=2, ensure_ascii=False)
            
            # Update provenance to record snapshot creation
            self.update_provenance(
                operation="create_snapshot",
                description=f"Created snapshot: {snapshot_name}",
                source_ids=[snapshot_id]
            )
            
            self.logger.info(f"Saved snapshot {snapshot_id}: {snapshot_name}")
            return snapshot_id
        except Exception as e:
            self.logger.error(f"Error saving snapshot {snapshot_id}: {e}")
            return ""

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

        # Prepare graph data with provenance metadata
        graph_data = {
            "metadata": {
                "graph_id": self.graph_id,
                "graph_version": self.graph_version,
                "parent_graph_id": self.parent_graph_id,
                "timestamp": datetime.now().isoformat(),
                "node_count": len(self.graph.nodes),
                "edge_count": len(self.graph.edges),
                "content_hash": self._compute_graph_hash(),
                "provenance_chain": self.provenance_chain
            },
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
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2, ensure_ascii=False)
            
            # Update provenance to record save
            self.update_provenance(
                operation="save_graph",
                description=f"Saved graph to {os.path.basename(filepath)}",
                source_ids=None
            )
            
            self.logger.info(f"Saved graph to {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"Error saving graph to {filepath}: {e}")
            return ""

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

        # Update provenance if any relationships were added
        if relationships_added > 0:
            self.update_provenance(
                operation="infer_temporal_relationships",
                description=f"Inferred {relationships_added} temporal relationships between events",
                source_ids=None
            )

        return relationships_added

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
            "discovered_node_types": list(self.discovered_node_types),
            "discovered_relation_types": list(self.discovered_relation_types),
            "graph_id": self.graph_id,
            "graph_version": self.graph_version,
            "content_hash": self._compute_graph_hash(),
            "provenance_chain_length": len(self.provenance_chain),
            "timestamp": datetime.now().isoformat()
        }
    
    def merge_graph(self, other_graph_file: str, description: str = None) -> bool:
        """
        Merge another graph into this one
        
        Args:
            other_graph_file: Path to the other graph file
            description: Description of the merge operation
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load the other graph
            with open(other_graph_file, 'r', encoding='utf-8') as f:
                other_graph_data = json.load(f)
                
            # Extract metadata
            other_metadata = other_graph_data.get("metadata", {})
            other_graph_id = other_metadata.get("graph_id")
            
            if not other_graph_id:
                self.logger.error(f"No graph ID found in {other_graph_file}")
                return False
                
            # Set as parent graph
            self.parent_graph_id = other_graph_id
            
            # Track added elements
            nodes_added = 0
            edges_added = 0
            
            # Add nodes from other graph
            for node_id, node_data in other_graph_data.get("nodes", {}).items():
                if not self.graph.has_node(node_id):
                    self.graph.add_node(node_id, **node_data)
                    nodes_added += 1
                    
                    # Update node counter if needed
                    numeric_id = int(node_id.split('_')[-1]) if '_' in node_id else 0
                    if numeric_id > self._node_counter:
                        self._node_counter = numeric_id
                        
                    # Save node file
                    self._save_node(node_id, node_data)
            
            # Add edges from other graph
            for edge_data in other_graph_data.get("edges", []):
                source = edge_data.get("source")
                target = edge_data.get("target")
                
                if source and target and not self.graph.has_edge(source, target):
                    # Remove source and target to avoid duplicate in attributes
                    edge_attrs = {k: v for k, v in edge_data.items() if k not in ["source", "target"]}
                    self.graph.add_edge(source, target, **edge_attrs)
                    edges_added += 1
                    
                    # Update edge counter if needed
                    if "id" in edge_data:
                        edge_id = edge_data["id"]
                        numeric_id = int(edge_id.split('_')[-1]) if '_' in edge_id else 0
                        if numeric_id > self._edge_counter:
                            self._edge_counter = numeric_id
                            
                        # Save edge file
                        self._save_edge(edge_id, edge_data)
            
            # Update provenance
            merge_description = description or f"Merged graph {other_graph_id} into {self.graph_id}"
            self.update_provenance(
                operation="merge_graph",
                description=merge_description,
                source_ids=[other_graph_id]
            )
            
            self.logger.info(f"Merged graph {other_graph_id}: added {nodes_added} nodes, {edges_added} edges")
            return True
            
        except Exception as e:
            self.logger.error(f"Error merging graph {other_graph_file}: {e}")
            return False
            
    def get_provenance_info(self) -> Dict[str, Any]:
        """
        Get detailed provenance information for the graph
        
        Returns:
            Dictionary with provenance info
        """
        return {
            "graph_id": self.graph_id,
            "graph_version": self.graph_version,
            "parent_graph_id": self.parent_graph_id,
            "content_hash": self._compute_graph_hash(),
            "first_created": self.provenance_chain[0]["timestamp"] if self.provenance_chain else None,
            "last_updated": self.provenance_chain[-1]["timestamp"] if self.provenance_chain else None,
            "update_count": len(self.provenance_chain),
            "provenance_chain": self.provenance_chain
        }
"""
Night_watcher Knowledge Graph
Manages the knowledge graph for storing entities, events, and relationships with dynamic relation types.
"""

import os
import json
import logging
import csv
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Union, Tuple
import networkx as nx

# Configure logging
logger = logging.getLogger(__name__)

class KnowledgeGraph:
    """
    Knowledge graph for storing entities, events, and relationships.
    Implements a graph-based data structure using NetworkX with JSON serialization.
    Dynamically accepts new node and relation types for flexibility.
    """

    def __init__(self, base_dir: str = "data/knowledge_graph", taxonomy_path: str = "KG_Taxonomy.csv",
                 graph_file: str = None, taxonomy_file: str = None):
        """
        Initialize the knowledge graph.

        Args:
            base_dir: Base directory for knowledge graph storage
            taxonomy_path: Path to taxonomy CSV file (deprecated, use taxonomy_file)
            graph_file: Path to graph file for initialization
            taxonomy_file: Path to taxonomy CSV file
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

        # Track discovered types for dynamic schema
        self.discovered_relation_types = set()
        self.discovered_node_types = set()

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
                        self.discovered_node_types.add(name)
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
                            self.discovered_relation_types.add(name)

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

                        # Track node type
                        node_type = node_data.get("type", "unknown")
                        if node_type not in self.discovered_node_types:
                            self.discovered_node_types.add(node_type)

                        # Update node counter
                        numeric_id = int(node_id.split('_')[-1]) if '_' in node_id else 0
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

                        # Track relation type
                        if relation and relation not in self.discovered_relation_types:
                            self.discovered_relation_types.add(relation)

                        if source_id and target_id:
                            # Add edge to graph
                            self.graph.add_edge(source_id, target_id, id=edge_id, **edge_data)

                            # Update edge counter
                            numeric_id = int(edge_id.split('_')[-1]) if '_' in edge_id else 0
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
                
                # Track node type
                node_type = node_data.get("type", "unknown")
                if node_type not in self.discovered_node_types:
                    self.discovered_node_types.add(node_type)

                # Update node counter
                numeric_id = int(node_id.split('_')[-1]) if '_' in node_id else 0
                if numeric_id > self._node_counter:
                    self._node_counter = numeric_id

            # Load edges
            edges = graph_data.get("edges", [])
            for edge_data in edges:
                source_id = edge_data.get("source")
                target_id = edge_data.get("target")
                relation = edge_data.get("relation")

                # Track relation type
                if relation and relation not in self.discovered_relation_types:
                    self.discovered_relation_types.add(relation)

                if source_id and target_id:
                    # Remove source and target from edge data to avoid duplication
                    edge_attrs = {k: v for k, v in edge_data.items() if k not in ["source", "target"]}
                    self.graph.add_edge(source_id, target_id, **edge_attrs)

                    # Update edge counter if id is present
                    if "id" in edge_data:
                        numeric_id = int(edge_data["id"].split('_')[-1]) if '_' in edge_data["id"] else 0
                        if numeric_id > self._edge_counter:
                            self._edge_counter = numeric_id

            self.logger.info(f"Loaded graph from file: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
        except Exception as e:
            self.logger.error(f"Error loading graph from file {graph_file}: {e}")
            # Initialize empty graph
            self.graph.clear()

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
        # Add node type to discovered types if new
        if node_type not in self.discovered_node_types:
            self.discovered_node_types.add(node_type)
            self.logger.info(f"Discovered new node type: {node_type}")

        # Check if a similar node already exists
        existing_node = self._find_similar_node(node_type, name)
        if existing_node:
            # Update existing node with any new attributes
            existing_data = self.graph.nodes[existing_node]
            if attributes:
                if "attributes" not in existing_data:
                    existing_data["attributes"] = {}
                for key, value in attributes.items():
                    if key not in existing_data["attributes"]:
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

        # Add node to graph
        self.graph.add_node(node_id, **node_data)

        # Save node to disk
        self._save_node(node_id, node_data)

        self.logger.debug(f"Added node {node_id}: {node_type} - {name}")
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
        # Add relation type to discovered types if new
        if relation not in self.discovered_relation_types:
            self.discovered_relation_types.add(relation)
            self.logger.info(f"Discovered new relation type: {relation}")

        # Validate source and target nodes
        if not self.graph.has_node(source_id):
            self.logger.warning(f"Source node {source_id} not in graph")
            return ""

        if not self.graph.has_node(target_id):
            self.logger.warning(f"Target node {target_id} not in graph")
            return ""

        # Check if a similar edge already exists
        existing_edge = self._find_similar_edge(source_id, relation, target_id)
        if existing_edge:
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
                if "attributes" not in existing_data:
                    existing_data["attributes"] = {}
                for key, value in attributes.items():
                    if key not in existing_data["attributes"]:
                        existing_data["attributes"][key] = value

            # Extract existing edge ID
            edge_id = existing_data.get("id", "")

            # Save updated edge
            if edge_id:
                self._save_edge(edge_id, existing_data)

            return edge_id

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

        # Add edge to graph
        edge_attrs = {k: v for k, v in edge_data.items() if k != 'id'}  # Remove id to avoid duplicate
        self.graph.add_edge(source_id, target_id, **edge_attrs)

        # Save edge to disk
        self._save_edge(edge_id, edge_data)

        self.logger.debug(f"Added edge {edge_id}: {source_id} --[{relation}]--> {target_id}")
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
                "edges_added": edges_added
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
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved graph to {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"Error saving graph to {filepath}: {e}")
            return ""

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
            "discovered_node_types": list(self.discovered_node_types),
            "discovered_relation_types": list(self.discovered_relation_types),
            "timestamp": datetime.now().isoformat()
        }
    