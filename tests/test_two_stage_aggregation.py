import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from event_aggregator import EventAggregator
from tests.test_data.synthetic_analyses import create_synthetic_analysis_files


def test_event_matching():
    analyses = create_synthetic_analysis_files()
    agg = EventAggregator()
    groups = agg.match_events_across_analyses(analyses)
    assert len(groups) == 1
    key = next(iter(groups))
    assert len(groups[key]) == 6


def test_stage1_consolidation():
    analyses = create_synthetic_analysis_files()
    agg = EventAggregator()
    groups = agg.match_events_across_analyses(analyses)
    key = next(iter(groups))
    graph = agg.consolidate_event_group(groups[key])
    assert len(graph["nodes"]) >= 2
    actor_key = ("actor", "president biden")
    assert actor_key in graph["nodes"]
    assert graph["nodes"][actor_key]["weight"] > 2.5


def test_stage2_unified_graph():
    analyses = create_synthetic_analysis_files()
    agg = EventAggregator()
    groups = agg.match_events_across_analyses(analyses)
    event_graphs = {k: agg.consolidate_event_group(v) for k, v in groups.items()}
    unified = agg.build_unified_graph(event_graphs)
    assert unified["stats"]["total_nodes"] >= 2
    assert unified["stats"]["events_processed"] == 1


def test_provenance_preservation():
    analyses = create_synthetic_analysis_files()
    agg = EventAggregator()
    groups = agg.match_events_across_analyses(analyses)
    key = next(iter(groups))
    graph = agg.consolidate_event_group(groups[key])
    assert len(graph["contributing_analyses"]) == 6
    assert len(graph["document_sources"]) == 3


def test_weight_accumulation():
    analyses = create_synthetic_analysis_files()
    agg = EventAggregator()
    groups = agg.match_events_across_analyses(analyses)
    event_graphs = {k: agg.consolidate_event_group(v) for k, v in groups.items()}
    unified = agg.build_unified_graph(event_graphs)
    actor_key = ("actor", "president biden")
    node = unified["nodes"][actor_key]
    assert node["total_weight"] > 5.0
