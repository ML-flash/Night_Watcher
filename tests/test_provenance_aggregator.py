import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from provenance_preserving_aggregator import ProvenancePreservingAggregator


def test_equivalence_building():
    agg = ProvenancePreservingAggregator()
    analysis1 = {
        "analysis_id": "a1",
        "article": {"url": "http://a.com", "domain": "a.com"},
        "kg_payload": {
            "nodes": [
                {"id": "n1", "node_type": "event", "name": "Launch", "confidence": 0.9}
            ],
            "edges": []
        }
    }
    analysis2 = {
        "analysis_id": "a2",
        "article": {"url": "http://b.com", "domain": "b.com"},
        "kg_payload": {
            "nodes": [
                {"id": "n2", "node_type": "event", "name": "Launch", "confidence": 0.8}
            ],
            "edges": []
        }
    }
    agg.process_analysis_batch([analysis1, analysis2])
    assert len(agg.equivalence_sets) == 1
    eq = next(iter(agg.equivalence_sets.values()))
    import pytest
    assert eq["aggregate_weight"] == pytest.approx(1.7)
    assert eq["canonical_name"] == "launch"
