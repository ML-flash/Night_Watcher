import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from document_aggregator import aggregate_document_analyses


def test_multi_analysis_aggregation():
    """Test aggregating multiple analyses of same document."""

    analysis1 = {
        "analysis_id": "test_001_standard",
        "facts_data": {
            "events": [
                {
                    "name": "President signs executive order",
                    "date": "2025-01-24",
                    "actors": ["President"],
                    "description": "President signed EO limiting protests"
                }
            ]
        },
        "kg_payload": {
            "nodes": [
                {"id": 1, "node_type": "actor", "name": "President"},
                {"id": 2, "node_type": "target", "name": "Protesters"}
            ],
            "edges": [
                {"source_id": 1, "relation": "targets", "target_id": 2}
            ]
        }
    }

    analysis2 = {
        "analysis_id": "test_001_narrative",
        "kg_payload": {
            "nodes": [
                {"id": 1, "node_type": "narrative", "name": "Maintaining public order"},
                {"id": 2, "node_type": "event", "name": "Executive order signed", "timestamp": "2025-01-24", "attributes": {"actor": "President"}}
            ],
            "edges": [
                {"source_id": 1, "relation": "justifies", "target_id": 2}
            ]
        }
    }

    result = aggregate_document_analyses("test_doc_001", [analysis1, analysis2])

    assert len(result["events"]) == 1
    assert result["events"][0]["source_count"] == 2

    kg = result["kg_payload"]
    node_types = [n["node_type"] for n in kg["nodes"]]
    assert "event" in node_types
    assert "actor" in node_types
    assert "narrative" in node_types

    print("\u2713 Aggregation test passed!")
    print(f"Events: {len(result['events'])}")
    print(f"Nodes: {len(kg['nodes'])}")
    print(f"Edges: {len(kg['edges'])}")


if __name__ == "__main__":
    test_multi_analysis_aggregation()
