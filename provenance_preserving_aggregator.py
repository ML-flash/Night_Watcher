"""Provenance Preserving Aggregator.

Stores individual knowledge graphs from multiple analyses and
builds equivalence sets between entities. Provides simple cross-
source weighting while keeping full provenance for each node.
"""
from collections import defaultdict
from typing import Any, Dict, List, Tuple


class ProvenancePreservingAggregator:
    """Aggregate analyses without losing provenance."""

    def __init__(self) -> None:
        self.individual_kgs: Dict[str, Dict[str, Any]] = {}
        self.equivalence_sets: Dict[str, Dict[str, Any]] = {}
        self.weighted_relationships: Dict[str, Dict[str, Any]] = {}
        self._set_counter = 0

    def process_analysis_batch(self, analyses: List[Dict[str, Any]]) -> None:
        """Store each analysis KG and build equivalence mappings."""
        for analysis in analyses:
            kg_id = f"kg_{analysis.get('analysis_id', len(self.individual_kgs))}"
            self.individual_kgs[kg_id] = self._extract_complete_kg(analysis)

        self._build_equivalence_sets()
        self._build_weighted_relationships()

    def _extract_complete_kg(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        payload = analysis.get("kg_payload") or {"nodes": [], "edges": []}
        payload.setdefault("source_info", analysis.get("article", {}))
        payload["analysis_id"] = analysis.get("analysis_id")
        return payload

    def _build_equivalence_sets(self) -> None:
        groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for kg_id, kg in self.individual_kgs.items():
            for node in kg.get("nodes", []):
                key = (node.get("node_type"), node.get("name", "").lower())
                groups[key].append({
                    "kg_id": kg_id,
                    "node_id": node.get("id"),
                    "node": node,
                    "source": kg.get("source_info", {}),
                    "analysis_id": kg.get("analysis_id"),
                    "confidence": node.get("confidence", 1.0),
                })

        for key, instances in groups.items():
            set_id = f"equiv_{self._set_counter}"
            self._set_counter += 1
            self.equivalence_sets[set_id] = {
                "set_id": set_id,
                "entity_type": key[0],
                "canonical_name": key[1],
                "instances": instances,
                "aggregate_weight": sum(i["confidence"] for i in instances),
                "cross_source_variance": 0.0,
            }

    def _build_weighted_relationships(self) -> None:
        for set_id_1, eq1 in self.equivalence_sets.items():
            for set_id_2, eq2 in self.equivalence_sets.items():
                if set_id_1 >= set_id_2:
                    continue
                supports = []
                target_ids = {i["node_id"] for i in eq2["instances"]}
                for inst1 in eq1["instances"]:
                    kg = self.individual_kgs.get(inst1["kg_id"], {})
                    for edge in kg.get("edges", []):
                        if edge.get("source_id") == inst1["node_id"] and edge.get("target_id") in target_ids:
                            supports.append({
                                "kg_id": inst1["kg_id"],
                                "relationship": edge,
                                "confidence": edge.get("confidence", 1.0),
                                "source": inst1.get("source", {}),
                            })
                if supports:
                    rel_id = f"rel_{len(self.weighted_relationships)}"
                    self.weighted_relationships[rel_id] = {
                        "relation_id": rel_id,
                        "source_equiv_set": set_id_1,
                        "target_equiv_set": set_id_2,
                        "supporting_instances": len(supports),
                        "total_weight": sum(s["confidence"] for s in supports),
                        "confidence": sum(s["confidence"] for s in supports) / len(supports),
                        "evidence": supports,
                    }


class ProvenanceTracker:
    """Track provenance lineage for equivalence sets."""

    def __init__(self) -> None:
        self.entity_lineage: Dict[str, Any] = {}
        self.analysis_metadata: Dict[str, Any] = {}

    def track_entity_provenance(self, set_id: str, instances: List[Dict[str, Any]]) -> None:
        self.entity_lineage[set_id] = {
            "canonical_id": set_id,
            "source_instances": instances,
            "corroboration_count": len(instances),
            "source_diversity": len({i.get("source", {}).get("domain") for i in instances}),
        }

    def get_provenance_chain(self, set_id: str) -> List[Dict[str, Any]]:
        lineage = self.entity_lineage.get(set_id, {})
        chain = []
        for inst in lineage.get("source_instances", []):
            meta = self.analysis_metadata.get(inst.get("analysis_id"), {})
            chain.append({
                "original_article": meta.get("article", {}).get("url"),
                "article_title": meta.get("article", {}).get("title"),
                "published_date": meta.get("article", {}).get("published"),
                "analysis_timestamp": meta.get("analysis_timestamp"),
                "kg_node_id": inst.get("node_id"),
                "confidence_score": inst.get("confidence", 1.0),
            })
        return chain


class AggregatedKnowledgeGraph:
    """Expose querying functions for aggregated data."""

    def __init__(self, aggregator: ProvenancePreservingAggregator, tracker: ProvenanceTracker) -> None:
        self.aggregator = aggregator
        self.provenance = tracker

    def get_entity(self, name: str, include_provenance: bool = True) -> Dict[str, Any]:
        for eq in self.aggregator.equivalence_sets.values():
            if eq["canonical_name"] == name.lower():
                result = {
                    "canonical_name": eq["canonical_name"],
                    "entity_type": eq["entity_type"],
                    "aggregate_weight": eq["aggregate_weight"],
                    "source_count": len(eq["instances"]),
                    "cross_source_variance": eq["cross_source_variance"],
                }
                if include_provenance:
                    self.provenance.track_entity_provenance(eq["set_id"], eq["instances"])
                    result["provenance"] = self.provenance.get_provenance_chain(eq["set_id"])
                return result
        return {}

    def get_relationship(self, name1: str, name2: str) -> Dict[str, Any]:
        set1 = None
        set2 = None
        for sid, eq in self.aggregator.equivalence_sets.items():
            if eq["canonical_name"] == name1.lower():
                set1 = sid
            if eq["canonical_name"] == name2.lower():
                set2 = sid
        if not set1 or not set2:
            return {}
        for rel in self.aggregator.weighted_relationships.values():
            if rel["source_equiv_set"] == set1 and rel["target_equiv_set"] == set2:
                return rel
            if rel["source_equiv_set"] == set2 and rel["target_equiv_set"] == set1:
                return rel
        return {}
