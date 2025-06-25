"""Document-level event aggregator.

Combines multiple analyses of the same document into a single knowledge graph
with events as the central nodes.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class EventExtractor:
    """Extract events from analysis results."""

    def extract_events(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []

        facts = analysis.get("facts_data", {})
        if isinstance(facts, dict) and "events" in facts:
            for event in facts["events"]:
                events.append(self._normalize_event(event))

        kg = analysis.get("kg_payload", {})
        for node in kg.get("nodes", []):
            if node.get("node_type") == "event":
                events.append(self._kg_node_to_event(node))

        for patt_event in self._extract_from_patterns(analysis):
            events.append(patt_event)

        return events

    def _normalize_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "name": event.get("name", ""),
            "date": self._normalize_date(event.get("date", "")),
            "actors": self._extract_actors(event),
            "action": self._extract_action(event),
            "targets": event.get("targets", []),
            "location": event.get("location", ""),
            "description": event.get("description", ""),
            "source_data": event,
        }

    def _kg_node_to_event(self, node: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "name": node.get("name", ""),
            "date": self._normalize_date(node.get("timestamp", "")),
            "actors": [node.get("attributes", {}).get("actor")],
            "action": node.get("name", ""),
            "targets": node.get("attributes", {}).get("targets", []),
            "location": node.get("attributes", {}).get("location", ""),
            "description": node.get("source_sentence", ""),
            "source_data": node,
        }

    def _extract_from_patterns(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        patt_events: List[Dict[str, Any]] = []
        title = analysis.get("article", {}).get("title", "")
        if "signs" in title:
            patt_events.append({
                "name": title,
                "date": analysis.get("article", {}).get("published", ""),
                "actors": [title.split(" signs ")[0]],
                "action": "signs",
                "targets": [],
                "location": "",
                "description": title,
                "source_data": analysis,
            })
        return patt_events

    def _extract_actors(self, event: Dict[str, Any]) -> List[str]:
        actors = event.get("actors", [])
        if isinstance(actors, list):
            return actors
        if actors:
            return [actors]
        return []

    def _extract_action(self, event: Dict[str, Any]) -> str:
        return event.get("action") or event.get("name", "")

    def _normalize_date(self, date_str: str) -> str:
        if not date_str:
            return ""
        for fmt in ["%Y-%m-%d", "%B %d, %Y", "%b %d, %Y"]:
            try:
                return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
            except Exception:
                continue
        return date_str


class EventMatcher:
    """Match and deduplicate events."""

    def deduplicate_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        unique: List[Dict[str, Any]] = []
        groups: List[List[Dict[str, Any]]] = []

        for event in events:
            matched = False
            for idx, ue in enumerate(unique):
                if self._events_match(event, ue):
                    groups[idx].append(event)
                    self._merge_event_data(ue, event)
                    matched = True
                    break
            if not matched:
                unique.append(event)
                groups.append([event])

        for idx, ue in enumerate(unique):
            ue["source_count"] = len(groups[idx])
            ue["merged_from"] = groups[idx]
        return unique

    def _events_match(self, a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        if a.get("date") and b.get("date") and a.get("date") != b.get("date"):
            return False
        score = 0.0
        if self._actors_match(a.get("actors"), b.get("actors")):
            score += 1.0
        if self._actions_match(a.get("action"), b.get("action")):
            score += 1.0
        if a.get("location") == b.get("location"):
            score += 0.5
        return score >= 1.5

    def _actors_match(self, a1: List[str] | None, a2: List[str] | None) -> bool:
        if not a1 or not a2:
            return False
        return bool(set(map(str.lower, a1)) & set(map(str.lower, a2)))

    def _actions_match(self, act1: str, act2: str) -> bool:
        if not act1 or not act2:
            return False
        return act1.lower() == act2.lower()

    def _merge_event_data(self, base: Dict[str, Any], incoming: Dict[str, Any]) -> None:
        if not base.get("description"):
            base["description"] = incoming.get("description", "")
        if not base.get("location"):
            base["location"] = incoming.get("location", "")
        actors = set(base.get("actors", [])) | set(incoming.get("actors", []))
        base["actors"] = list(actors)


class EventCentricGraphBuilder:
    """Build event-centric knowledge graphs."""

    def build_graph(self, events: List[Dict[str, Any]], analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        kg: Dict[str, Any] = {"nodes": [], "edges": []}
        event_node_map: Dict[str, int] = {}

        for event in events:
            node_id = len(kg["nodes"]) + 1
            kg["nodes"].append({
                "id": node_id,
                "node_type": "event",
                "name": event.get("name"),
                "attributes": {
                    "date": event.get("date"),
                    "description": event.get("description", ""),
                    "location": event.get("location", ""),
                    "source_count": event.get("source_count", 1),
                },
                "timestamp": event.get("date"),
            })
            event_node_map[event.get("name")] = node_id

        for analysis in analyses:
            self._add_actors(kg, analysis, event_node_map)
            self._add_targets(kg, analysis, event_node_map)
            self._add_narratives(kg, analysis, event_node_map)
            self._add_relationships(kg, analysis, event_node_map)
        return kg

    def _add_or_get_node(self, kg: Dict[str, Any], node: Dict[str, Any]) -> Dict[str, Any]:
        for existing in kg["nodes"]:
            if existing.get("node_type") == node.get("node_type") and existing.get("name") == node.get("name"):
                return existing
        node_id = len(kg["nodes"]) + 1
        new_node = {"id": node_id, **node}
        kg["nodes"].append(new_node)
        return new_node

    def _add_actors(self, kg: Dict[str, Any], analysis: Dict[str, Any], event_map: Dict[str, int]) -> None:
        actors = self._extract_all_actors(analysis)
        for actor in actors:
            actor_node = self._add_or_get_node(kg, {
                "node_type": "actor",
                "name": actor.get("name"),
                "attributes": actor.get("attributes", {}),
            })
            for event_name, event_id in event_map.items():
                if self._actor_relates_to_event(actor, event_name, analysis):
                    kg["edges"].append({
                        "source_id": actor_node["id"],
                        "relation": "performs",
                        "target_id": event_id,
                        "evidence": actor.get("evidence", ""),
                        "source_analysis": analysis.get("analysis_id"),
                    })

    def _add_targets(self, kg: Dict[str, Any], analysis: Dict[str, Any], event_map: Dict[str, int]) -> None:
        pass

    def _add_narratives(self, kg: Dict[str, Any], analysis: Dict[str, Any], event_map: Dict[str, int]) -> None:
        pass

    def _add_relationships(self, kg: Dict[str, Any], analysis: Dict[str, Any], event_map: Dict[str, int]) -> None:
        for edge in analysis.get("kg_payload", {}).get("edges", []):
            kg["edges"].append({
                "source_id": edge.get("source_id"),
                "relation": edge.get("relation"),
                "target_id": edge.get("target_id"),
                "source_analysis": analysis.get("analysis_id"),
            })

    def _extract_all_actors(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        actors: List[Dict[str, Any]] = []
        for node in analysis.get("kg_payload", {}).get("nodes", []):
            if node.get("node_type") == "actor":
                actors.append({"name": node.get("name"), "attributes": node.get("attributes", {})})
        return actors

    def _actor_relates_to_event(self, actor: Dict[str, Any], event_name: str, analysis: Dict[str, Any]) -> bool:
        return actor.get("name") in event_name


class RelationshipMapper:
    """Map relationships between entities through events."""

    def map_relationships(self, kg: Dict[str, Any], analyses: List[Dict[str, Any]]) -> None:
        for analysis in analyses:
            kg_payload = analysis.get("kg_payload", {})
            for edge in kg_payload.get("edges", []):
                self._add_relationship(kg, edge, analysis)
        self._infer_event_relationships(kg)

    def _add_relationship(self, kg: Dict[str, Any], edge: Dict[str, Any], analysis: Dict[str, Any]) -> None:
        kg["edges"].append({
            "source_id": edge.get("source_id"),
            "relation": edge.get("relation"),
            "target_id": edge.get("target_id"),
            "source_analysis": analysis.get("analysis_id"),
        })

    def _infer_event_relationships(self, kg: Dict[str, Any]) -> None:
        pass


class DocumentAggregator:
    """High level aggregator for a single document."""

    def __init__(self) -> None:
        self.extractor = EventExtractor()
        self.matcher = EventMatcher()
        self.builder = EventCentricGraphBuilder()
        self.mapper = RelationshipMapper()


def aggregate_document_analyses(document_id: str, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    aggregator = DocumentAggregator()
    all_events: List[Dict[str, Any]] = []
    for analysis in analyses:
        all_events.extend(aggregator.extractor.extract_events(analysis))
    unique_events = aggregator.matcher.deduplicate_events(all_events)
    unified_kg = aggregator.builder.build_graph(unique_events, analyses)
    aggregator.mapper.map_relationships(unified_kg, analyses)
    return {
        "document_id": document_id,
        "events": unique_events,
        "kg_payload": unified_kg,
        "source_analyses": [a.get("analysis_id") for a in analyses],
        "aggregation_timestamp": datetime.now().isoformat(),
    }
