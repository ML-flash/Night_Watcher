import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from event_aggregator import EventAggregator


def test_event_similarity_basic():
    agg = EventAggregator(similarity_threshold=0.7)
    e1 = {
        "name": "Protest in Washington",
        "date": "2025-06-01",
        "attributes": {"location": "Washington DC, USA", "description": "Mass protest"},
    }
    e2 = {
        "name": "Protest rally Washington",
        "date": "2025-06-01",
        "attributes": {"location": "Washington, DC, USA", "description": "Large protest"},
    }
    assert agg._events_similar(e1, e2)


def test_event_similarity_temporal():
    agg = EventAggregator(similarity_threshold=0.7)
    e1 = {"name": "Court ruling", "date": "2025-06-01", "attributes": {"location": "NY, USA"}}
    e2 = {"name": "Court ruling", "date": "2025-06-02", "attributes": {"location": "NY, USA"}}
    assert agg._events_similar(e1, e2)


def test_event_similarity_distant_dates():
    agg = EventAggregator(similarity_threshold=0.8)
    e1 = {"name": "Election results", "date": "2025-06-01", "attributes": {"location": "CA, USA"}}
    e2 = {"name": "Election results", "date": "2025-06-10", "attributes": {"location": "CA, USA"}}
    assert not agg._events_similar(e1, e2)
