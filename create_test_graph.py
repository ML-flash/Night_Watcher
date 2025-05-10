#!/usr/bin/env python3
"""
Script to create a test knowledge graph with sample data.
This will verify that the knowledge_graph.py module is working correctly.
"""

import os
import sys
import logging
from datetime import datetime

# Ensure knowledge_graph module can be found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from knowledge_graph import KnowledgeGraph

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("create_test_graph")

def main():
    """Create a test knowledge graph with sample data"""
    
    print("\n=== Creating Test Knowledge Graph ===")
    
    # Initialize knowledge graph
    output_dir = "data/test_kg"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the knowledge graph
    kg = KnowledgeGraph(base_dir=output_dir, taxonomy_file="KG_Taxonomy.csv")
    
    # Create some test nodes
    print("\nAdding test nodes...")
    
    # Political actors
    president_id = kg.add_node(
        node_type="actor",
        name="President Smith",
        attributes={
            "role": "President",
            "party": "United Party",
            "appointment_source": "Election"
        },
        timestamp=datetime.now().isoformat()
    )
    
    doj_id = kg.add_node(
        node_type="institution",
        name="Department of Justice",
        attributes={
            "branch": "executive",
            "independence_score": 7
        },
        timestamp=datetime.now().isoformat()
    )
    
    judge_id = kg.add_node(
        node_type="actor",
        name="Judge Johnson",
        attributes={
            "role": "Federal Judge",
            "party": "Independent",
            "appointment_source": "Senate Confirmation"
        },
        timestamp=datetime.now().isoformat()
    )
    
    court_id = kg.add_node(
        node_type="institution",
        name="Federal Court System",
        attributes={
            "branch": "judicial",
            "independence_score": 8
        },
        timestamp=datetime.now().isoformat()
    )
    
    policy_id = kg.add_node(
        node_type="policy",
        name="Executive Order 2025-01",
        attributes={
            "issue_area": "Judicial Oversight",
            "enactment_date": "2025-01-30"
        },
        timestamp=datetime.now().isoformat()
    )
    
    event_id = kg.add_node(
        node_type="event",
        name="DOJ Review of Judicial Decisions",
        attributes={
            "date": datetime.now().strftime("%Y-%m-%d"),
            "location": "Washington, DC"
        },
        timestamp=datetime.now().isoformat(),
        source_sentence="The Department of Justice announced a review of recent judicial decisions."
    )
    
    media_id = kg.add_node(
        node_type="media_outlet",
        name="National News Network",
        attributes={
            "bias_label": "center-left",
            "format": "television"
        },
        timestamp=datetime.now().isoformat()
    )
    
    aba_id = kg.add_node(
        node_type="civil_society",
        name="American Bar Association",
        attributes={
            "sector": "bar association",
            "founding_year": "1878" 
        },
        timestamp=datetime.now().isoformat()
    )
    
    # Create relationships
    print("\nAdding test relationships...")
    
    # President authorizes the policy
    kg.add_edge(
        source_id=president_id,
        relation="authorizes",
        target_id=policy_id,
        timestamp=datetime.now().isoformat(),
        evidence_quote="The President signed Executive Order 2025-01 today."
    )
    
    # DOJ is part of executive branch
    kg.add_edge(
        source_id=doj_id,
        relation="part_of",
        target_id=president_id,
        timestamp=datetime.now().isoformat()
    )
    
    # Policy restricts courts
    kg.add_edge(
        source_id=policy_id,
        relation="restricts",
        target_id=court_id,
        timestamp=datetime.now().isoformat(),
        evidence_quote="The executive order places new restrictions on federal courts."
    )
    
    # Judge opposes policy
    kg.add_edge(
        source_id=judge_id,
        relation="opposes",
        target_id=policy_id,
        timestamp=datetime.now().isoformat(),
        evidence_quote="Judge Johnson expressed concerns about the executive order's impact on judicial independence."
    )
    
    # Media reports on event
    kg.add_edge(
        source_id=media_id,
        relation="mentions",  # Changed to 'mentions' which is in the taxonomy
        target_id=event_id,
        timestamp=datetime.now().isoformat()
    )
    
    # ABA supports judge
    kg.add_edge(
        source_id=aba_id,
        relation="supports",
        target_id=judge_id,
        timestamp=datetime.now().isoformat(),
        evidence_quote="The American Bar Association issued a statement supporting Judge Johnson's position."
    )
    
    # DOJ event undermines court
    kg.add_edge(
        source_id=event_id,
        relation="undermines",
        target_id=court_id,
        timestamp=datetime.now().isoformat(),
        evidence_quote="The review has been criticized as undermining the independence of the judiciary."
    )
    
    # Save the graph
    graph_file = os.path.join(output_dir, "graph.json")
    kg.save_graph(filepath=graph_file)
    
    # Display basic statistics
    stats = kg.get_basic_statistics()
    
    print("\n=== Knowledge Graph Created Successfully ===")
    print(f"Nodes: {stats['node_count']}")
    print(f"Edges: {stats['edge_count']}")
    print(f"Node types: {list(stats['node_types'].keys())}")
    print(f"Relation types: {list(stats['relation_types'].keys())}")
    print(f"\nGraph saved to: {graph_file}")
    
    # Create a snapshot
    snapshot_id = kg.save_snapshot(name="Test Graph Initial State")
    print(f"Snapshot created: {snapshot_id}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())