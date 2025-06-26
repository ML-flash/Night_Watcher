from collections import defaultdict
from typing import Any, Dict, List, Tuple
from datetime import datetime
from difflib import SequenceMatcher
import os
import json
import hashlib


class EventAggregator:
    """Two-stage event-centric aggregator."""

    def __init__(self, similarity_threshold: float = 0.8) -> None:
        self.similarity_threshold = similarity_threshold

    def match_events_across_analyses(self, analyses: List[Dict]) -> Dict[str, List[Dict]]:
        """Group analyses by equivalent events."""
        event_groups: Dict[str, List[Dict]] = defaultdict(list)

        for analysis in analyses:
            events = self._extract_events_from_analysis(analysis)
            for event in events:
                event_key = self._normalize_event_key(event)
                event_groups[event_key].append(analysis)

        return dict(event_groups)

    def _normalize_event_key(self, event: Dict) -> str:
        """Create consistent key for event matching."""
        name = event.get("name", "").lower().strip()
        date = event.get("date", "")
        return f"{name}_{date}"

    def _extract_events_from_analysis(self, analysis: Dict) -> List[Dict]:
        """Get events from analysis, whether from facts or kg_payload."""
        events: List[Dict] = []

        if "facts_data" in analysis:
            events.extend(analysis["facts_data"].get("events", []))

        kg_payload = analysis.get("kg_payload", {})
        for node in kg_payload.get("nodes", []):
            if node.get("node_type") == "event":
                events.append({
                    "name": node.get("name"),
                    "date": node.get("timestamp", "N/A"),
                    "description": node.get("attributes", {}).get("description", ""),
                })

        return events

    def consolidate_event_group(self, analyses: List[Dict]) -> Dict:
        """Merge all analyses for a single event into consolidated graph."""
        consolidated = {
            "nodes": {},
            "edges": {},
            "contributing_analyses": [a.get("analysis_id") for a in analyses],
            "document_sources": list({a.get("article", {}).get("document_id") for a in analyses}),
        }

        for analysis in analyses:
            kg_payload = analysis.get("kg_payload", {})
            id_map: Dict[Any, Tuple[str, str]] = {}

            for node in kg_payload.get("nodes", []):
                node_key = (node.get("node_type"), node.get("name", "").lower())
                if node_key in consolidated["nodes"]:
                    existing = consolidated["nodes"][node_key]
                    existing["weight"] += node.get("confidence", 1.0)
                    existing["sources"].append(analysis.get("analysis_id"))
                    existing["attributes"].update(node.get("attributes", {}))
                else:
                    consolidated["nodes"][node_key] = {
                        "node_type": node.get("node_type"),
                        "name": node.get("name"),
                        "weight": node.get("confidence", 1.0),
                        "sources": [analysis.get("analysis_id")],
                        "attributes": node.get("attributes", {}),
                        "timestamp": node.get("timestamp", "N/A"),
                    }
                original_id = node.get("id")
                if original_id is not None:
                    id_map[original_id] = node_key

            for edge in kg_payload.get("edges", []):
                src_key = id_map.get(edge.get("source_id"))
                tgt_key = id_map.get(edge.get("target_id"))
                if not src_key or not tgt_key:
                    continue
                edge_key = (src_key, edge.get("relation"), tgt_key)
                if edge_key in consolidated["edges"]:
                    existing = consolidated["edges"][edge_key]
                    existing["weight"] += edge.get("confidence", 1.0)
                    existing["sources"].append(analysis.get("analysis_id"))
                else:
                    consolidated["edges"][edge_key] = {
                        "source_id": src_key,
                        "relation": edge.get("relation"),
                        "target_id": tgt_key,
                        "weight": edge.get("confidence", 1.0),
                        "sources": [analysis.get("analysis_id")],
                    }

        return consolidated

    def build_unified_graph(self, event_graphs: Dict[str, Dict]) -> Dict:
        """Merge entities/relationships across all events."""
        unified = {
            "nodes": {},
            "edges": {},
            "event_linkages": {},
            "stats": {},
        }

        for event_key, event_graph in event_graphs.items():
            for node_key, node in event_graph.get("nodes", {}).items():
                if node_key in unified["nodes"]:
                    unified_node = unified["nodes"][node_key]
                    unified_node["total_weight"] += node["weight"]
                    unified_node["event_appearances"] += 1
                    unified_node.setdefault("event_linkages", []).append({
                        "event": event_key,
                        "weight_contribution": node["weight"],
                        "sources": node["sources"],
                    })
                else:
                    unified["nodes"][node_key] = {
                        "node_type": node["node_type"],
                        "name": node["name"],
                        "total_weight": node["weight"],
                        "event_appearances": 1,
                        "event_linkages": [{
                            "event": event_key,
                            "weight_contribution": node["weight"],
                            "sources": node["sources"],
                        }],
                        "attributes": node.get("attributes", {}),
                    }

        for event_key, event_graph in event_graphs.items():
            for edge_key, edge in event_graph.get("edges", {}).items():
                if edge_key in unified["edges"]:
                    unified_edge = unified["edges"][edge_key]
                    unified_edge["total_weight"] += edge["weight"]
                    unified_edge["event_appearances"] += 1
                    unified_edge.setdefault("event_linkages", []).append({
                        "event": event_key,
                        "weight_contribution": edge["weight"],
                        "sources": edge["sources"],
                    })
                else:
                    unified["edges"][edge_key] = {
                        "source_id": edge["source_id"],
                        "relation": edge["relation"],
                        "target_id": edge["target_id"],
                        "total_weight": edge["weight"],
                        "event_appearances": 1,
                        "event_linkages": [{
                            "event": event_key,
                            "weight_contribution": edge["weight"],
                            "sources": edge["sources"],
                        }],
                    }

        unified["stats"] = {
            "total_nodes": len(unified["nodes"]),
            "total_edges": len(unified["edges"]),
            "events_processed": len(event_graphs),
            "total_documents": sum(len(g.get("document_sources", [])) for g in event_graphs.values()),
        }

        return unified

    # ------------------------------------------------------------------
    # Legacy similarity helpers for compatibility with existing tests
    # ------------------------------------------------------------------
    def _events_similar(self, event1: Dict[str, Any], event2: Dict[str, Any]) -> bool:
        name1 = event1.get("name", "") or ""
        name2 = event2.get("name", "") or ""

        desc1 = (event1.get("attributes", {}).get("description") or "")
        desc2 = (event2.get("attributes", {}).get("description") or "")

        name_score = self._fuzzy_ratio(name1.lower(), name2.lower())
        desc_score = self._fuzzy_ratio(desc1.lower(), desc2.lower())

        date_score = self._temporal_score(event1.get("date"), event2.get("date"))
        loc_score = self._location_score(
            event1.get("attributes", {}).get("location"),
            event2.get("attributes", {}).get("location"),
        )

        total = 0.6 * name_score + 0.2 * desc_score + 0.1 * date_score + 0.1 * loc_score
        return total >= self.similarity_threshold

    @staticmethod
    def _fuzzy_ratio(text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0
        return SequenceMatcher(None, text1, text2).ratio()

    def _temporal_score(self, d1: str, d2: str) -> float:
        date1 = self._parse_date(d1)
        date2 = self._parse_date(d2)
        if not date1 or not date2:
            return 0.0
        diff = abs((date1 - date2).days)
        if diff == 0:
            return 1.0
        if diff == 1:
            return 0.8
        if diff <= 2:
            return 0.6
        return 0.0

    def _location_score(self, loc1: str, loc2: str) -> float:
        if not loc1 or not loc2:
            return 0.0

        parts1 = [p.strip().lower() for p in loc1.split(',')]
        parts2 = [p.strip().lower() for p in loc2.split(',')]
        if parts1 == parts2:
            return 1.0
        if len(parts1) >= 2 and len(parts2) >= 2 and parts1[-2:] == parts2[-2:]:
            return 0.6
        if parts1 and parts2 and parts1[-1] == parts2[-1]:
            return 0.3
        return 0.0

    def _parse_date(self, date_str: str) -> datetime | None:
        if not date_str or date_str == "N/A":
            return None
        try:
            if "T" in date_str:
                return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            return datetime.strptime(date_str, "%Y-%m-%d")
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Crypto lineage enhancements
    # ------------------------------------------------------------------
    def aggregate_with_crypto_lineage(self, analyses: List[Dict]) -> Dict:
        """Aggregate events while preserving crypto lineage."""
        standard_result = self._aggregate_standard(analyses)
        try:
            return self._add_aggregation_lineage(standard_result, analyses)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Lineage preservation failed: {e}")
            return standard_result

    def _aggregate_standard(self, analyses: List[Dict]) -> Dict:
        """Original aggregation logic broken out for reuse."""
        event_groups = self.match_events_across_analyses(analyses)
        event_graphs = {}
        for event_key, event_analyses in event_groups.items():
            event_graphs[event_key] = self.consolidate_event_group(event_analyses)

        unified_graph = self.build_unified_graph(event_graphs)

        return {
            "event_graphs": event_graphs,
            "unified_graph": unified_graph,
            "analyses_processed": len(analyses),
            "aggregation_timestamp": datetime.now().isoformat(),
        }

    def _add_aggregation_lineage(self, standard_result: Dict, analyses: List[Dict]) -> Dict:
        """Attach aggregation lineage information."""
        input_lineages = []
        analysis_ids = []
        for analysis in analyses:
            if "crypto_lineage" in analysis:
                input_lineages.append(analysis["crypto_lineage"])
                analysis_ids.append(analysis["crypto_lineage"].get("analysis_id"))
            else:
                analysis_ids.append(analysis.get("analysis_id", "legacy"))

        aggregation_input = ":".join(sorted(analysis_ids))
        aggregation_id = hashlib.sha256(aggregation_input.encode("utf-8")).hexdigest()

        aggregation_output_hash = hashlib.sha256(json.dumps(standard_result, sort_keys=True).encode("utf-8")).hexdigest()

        lineage_enhanced_result = {
            **standard_result,
            "crypto_lineage": {
                "aggregation_id": aggregation_id,
                "output_hash": aggregation_output_hash,
                "derived_from_analyses": analysis_ids,
                "input_lineages": input_lineages,
                "aggregation_method": "two_stage_event_centric",
                "aggregation_timestamp": standard_result["aggregation_timestamp"],
            },
        }

        self._store_aggregation_lineage(aggregation_id, lineage_enhanced_result)
        return lineage_enhanced_result

    def _store_aggregation_lineage(self, aggregation_id: str, lineage_result: Dict) -> None:
        """Persist aggregation lineage for export."""
        try:
            lineage_dir = "data/aggregation_lineage"
            os.makedirs(lineage_dir, exist_ok=True)
            lineage_file = os.path.join(lineage_dir, f"{aggregation_id}_lineage.json")

            lineage_data = {
                "aggregation_id": aggregation_id,
                "crypto_lineage": lineage_result.get("crypto_lineage"),
                "lineage_type": "aggregation_derivation",
                "stored_at": datetime.now().isoformat(),
            }

            with open(lineage_file, "w", encoding="utf-8") as f:
                json.dump(lineage_data, f, indent=2)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Could not store aggregation lineage: {e}")

    def collect_all_aggregation_lineages(self) -> List[Dict]:
        """Load all aggregation lineage records."""
        lineages = []
        lineage_dir = "data/aggregation_lineage"
        if not os.path.exists(lineage_dir):
            return lineages

        for filename in os.listdir(lineage_dir):
            if filename.endswith("_lineage.json"):
                try:
                    with open(os.path.join(lineage_dir, filename), "r", encoding="utf-8") as f:
                        lineage = json.load(f)
                    lineages.append(lineage)
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(f"Could not load aggregation lineage {filename}: {e}")

        return lineages
