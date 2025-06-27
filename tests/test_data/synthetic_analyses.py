from typing import List, Dict

def create_test_scenario_1():
    """Executive Order scenario: 3 documents, 2 templates each."""
    return {
        "documents": [
            {"id": "doc_001", "title": "Biden Signs Climate Executive Order", "source": "CNN", "url": "https://cnn.com/climate-order"},
            {"id": "doc_002", "title": "President Issues Environmental Directive", "source": "Reuters", "url": "https://reuters.com/env-directive"},
            {"id": "doc_003", "title": "New Climate Rules Announced", "source": "Fox", "url": "https://fox.com/climate-rules"}
        ],
        "expected_events": [
            {
                "name": "Climate Executive Order Signing",
                "participants": ["President Biden", "EPA", "Environmental Groups"],
                "expected_weight": 3.0
            }
        ],
        "expected_unified_nodes": {
            "President Biden": {"min_weight": 2.5, "event_appearances": 1},
            "EPA": {"min_weight": 1.8, "event_appearances": 1},
            "Climate Executive Order": {"min_weight": 2.2, "event_appearances": 1}
        }
    }

def create_synthetic_analysis_files() -> List[Dict]:
    """Generate synthetic analysis JSON objects for testing."""
    analyses = []

    doc1_event_analysis = {
        "analysis_id": "doc_001_event_analysis",
        "template_name": "event_analysis",
        "article": {"document_id": "doc_001", "source": "CNN"},
        "facts_data": {
            "events": [{
                "name": "Climate Executive Order Signing",
                "date": "2025-01-24",
                "actors": ["President Biden"],
                "description": "President signs executive order on climate"
            }]
        },
        "kg_payload": {
            "nodes": [
                {"id": 1, "node_type": "event", "name": "Climate Executive Order Signing", "confidence": 0.9, "timestamp": "2025-01-24"},
                {"id": 2, "node_type": "actor", "name": "President Biden", "confidence": 0.95}
            ],
            "edges": [
                {"source_id": 2, "relation": "performs", "target_id": 1, "confidence": 0.9}
            ]
        }
    }

    doc1_actor_analysis = {
        "analysis_id": "doc_001_actor_analysis",
        "template_name": "actor_analysis",
        "article": {"document_id": "doc_001", "source": "CNN"},
        "kg_payload": {
            "nodes": [
                {"id": 1, "node_type": "actor", "name": "President Biden", "confidence": 0.9},
                {"id": 2, "node_type": "institution", "name": "EPA", "confidence": 0.8}
            ],
            "edges": [
                {"source_id": 1, "relation": "directs", "target_id": 2, "confidence": 0.85}
            ]
        }
    }

    doc2_event_analysis = {
        "analysis_id": "doc_002_event_analysis",
        "template_name": "event_analysis",
        "article": {"document_id": "doc_002", "source": "Reuters"},
        "facts_data": {
            "events": [{
                "name": "Climate Executive Order Signing",
                "date": "2025-01-24",
                "actors": ["President Biden"],
                "description": "President signs order"
            }]
        },
        "kg_payload": {
            "nodes": [
                {"id": 1, "node_type": "event", "name": "Climate Executive Order Signing", "confidence": 0.8, "timestamp": "2025-01-24"},
                {"id": 2, "node_type": "actor", "name": "President Biden", "confidence": 0.9}
            ],
            "edges": [
                {"source_id": 2, "relation": "performs", "target_id": 1, "confidence": 0.8}
            ]
        }
    }

    doc2_actor_analysis = {
        "analysis_id": "doc_002_actor_analysis",
        "template_name": "actor_analysis",
        "article": {"document_id": "doc_002", "source": "Reuters"},
        "kg_payload": {
            "nodes": [
                {"id": 1, "node_type": "actor", "name": "President Biden", "confidence": 0.88},
                {"id": 2, "node_type": "institution", "name": "EPA", "confidence": 0.85}
            ],
            "edges": [
                {"source_id": 1, "relation": "directs", "target_id": 2, "confidence": 0.8}
            ]
        }
    }

    doc3_event_analysis = {
        "analysis_id": "doc_003_event_analysis",
        "template_name": "event_analysis",
        "article": {"document_id": "doc_003", "source": "Fox"},
        "facts_data": {
            "events": [{
                "name": "Climate Executive Order Signing",
                "date": "2025-01-24",
                "actors": ["President Biden"],
                "description": "Order on climate signed"
            }]
        },
        "kg_payload": {
            "nodes": [
                {"id": 1, "node_type": "event", "name": "Climate Executive Order Signing", "confidence": 0.85, "timestamp": "2025-01-24"},
                {"id": 2, "node_type": "actor", "name": "President Biden", "confidence": 0.9}
            ],
            "edges": [
                {"source_id": 2, "relation": "performs", "target_id": 1, "confidence": 0.85}
            ]
        }
    }

    doc3_actor_analysis = {
        "analysis_id": "doc_003_actor_analysis",
        "template_name": "actor_analysis",
        "article": {"document_id": "doc_003", "source": "Fox"},
        "kg_payload": {
            "nodes": [
                {"id": 1, "node_type": "actor", "name": "President Biden", "confidence": 0.87},
                {"id": 2, "node_type": "institution", "name": "EPA", "confidence": 0.8}
            ],
            "edges": [
                {"source_id": 1, "relation": "directs", "target_id": 2, "confidence": 0.82}
            ]
        }
    }

    analyses.extend([
        doc1_event_analysis,
        doc1_actor_analysis,
        doc2_event_analysis,
        doc2_actor_analysis,
        doc3_event_analysis,
        doc3_actor_analysis,
    ])

    return analyses
