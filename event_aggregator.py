"""
Night_watcher Event Aggregator
Deduplicates and merges events across multiple article analyses.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)


class EventAggregator:
    """Aggregates events from multiple sources into unified event records."""
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        self.events = {}  # event_id -> merged event data
        self.article_to_events = defaultdict(set)  # article_id -> set of event_ids

        # Storage for consolidated knowledge graph data
        self.node_index: Dict[Tuple[str, str], str] = {}
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self._node_counter = 0

    def _add_nodes(self, nodes: List[Dict[str, Any]], source_doc: str) -> Dict[str, str]:
        """Add nodes from a single analysis to the consolidated store."""
        id_map = {}
        for node in nodes:
            node_type = node.get("node_type")
            name = node.get("name")
            if not node_type or not name:
                continue

            key = (node_type, name.lower())
            if key in self.node_index:
                node_id = self.node_index[key]
                existing = self.nodes[node_id]
                existing.setdefault("evidence_sources", []).append(source_doc)
                # Merge attributes conservatively
                for k, v in (node.get("attributes") or {}).items():
                    if k not in existing.get("attributes", {}):
                        existing.setdefault("attributes", {})[k] = v
            else:
                self._node_counter += 1
                node_id = f"agg_node_{self._node_counter}"
                self.node_index[key] = node_id
                self.nodes[node_id] = {
                    "id": node_id,
                    "node_type": node_type,
                    "name": name,
                    "attributes": node.get("attributes", {}),
                    "evidence_sources": [source_doc],
                }
            original_id = node.get("id")
            if original_id:
                id_map[original_id] = node_id
        return id_map

    def _add_edges(self, edges: List[Dict[str, Any]], id_map: Dict[str, str], source_doc: str) -> None:
        """Add edges from a single analysis using consolidated node IDs."""
        for edge in edges:
            src_orig = edge.get("source_id")
            tgt_orig = edge.get("target_id")
            relation = edge.get("relation")
            if not src_orig or not tgt_orig or not relation:
                continue

            src = id_map.get(src_orig)
            tgt = id_map.get(tgt_orig)
            if not src or not tgt:
                continue

            key = (src, tgt, relation)
            if key not in self.edges:
                self.edges[key] = {
                    "source": src,
                    "target": tgt,
                    "relationship": relation,
                    "weight": 1,
                    "evidence_sources": [source_doc],
                }
            else:
                self.edges[key]["weight"] += 1
                self.edges[key]["evidence_sources"].append(source_doc)

        
    def process_analysis_batch(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a batch of article analyses to extract events AND couple analyses together.
        
        Args:
            analyses: List of analysis results from analyzer.py
            
        Returns:
            Dict with aggregated events, patterns, and cross-references
        """
        raw_events = []
        all_actors = defaultdict(list)
        all_narratives = defaultdict(list)
        manipulation_patterns = defaultdict(list)
        authoritarian_indicators = defaultdict(list)
        
        # Extract comprehensive data from all analyses
        for analysis in analyses:
            article_id = analysis.get("article", {}).get("document_id", "")
            article_source = analysis.get("article", {}).get("source")
            article_bias = analysis.get("article", {}).get("bias_label")
            
            # Extract manipulation scores and concerns
            if "manipulation_score" in analysis:
                manipulation_patterns[article_source].append({
                    "score": analysis["manipulation_score"],
                    "article_id": article_id,
                    "techniques": self._extract_manipulation_techniques(analysis)
                })
            
            # Extract authoritarian indicators
            if "authoritarian_indicators" in analysis:
                for indicator in analysis["authoritarian_indicators"]:
                    authoritarian_indicators[indicator].append({
                        "article_id": article_id,
                        "source": article_source,
                        "concern_level": analysis.get("concern_level", "Unknown")
                    })
            
            # Process KG nodes for comprehensive extraction
            kg_payload = analysis.get("kg_payload", {})
            nodes = kg_payload.get("nodes", [])
            edges = kg_payload.get("edges", [])

            # Consolidate nodes and edges for master KG payload
            id_map = self._add_nodes(nodes, article_id)
            self._add_edges(edges, id_map, article_id)
            
            for node in nodes:
                node_type = node.get("node_type")
                
                if node_type == "event":
                    event = {
                        "name": node.get("name"),
                        "date": node.get("timestamp", "N/A"),
                        "attributes": node.get("attributes", {}),
                        "article_id": article_id,
                        "article_source": article_source,
                        "article_bias": article_bias,
                        "source_sentence": node.get("source_sentence", ""),
                        "manipulation_score": analysis.get("manipulation_score", 0),
                        "concern_level": analysis.get("concern_level", "Unknown")
                    }
                    raw_events.append(event)
                    
                elif node_type == "actor":
                    actor_data = {
                        "name": node.get("name"),
                        "attributes": node.get("attributes", {}),
                        "article_id": article_id,
                        "source": article_source,
                        "actions": self._extract_actor_actions(node, edges)
                    }
                    all_actors[node["name"]].append(actor_data)
                    
                elif node_type == "narrative":
                    narrative_data = {
                        "theme": node.get("name"),
                        "attributes": node.get("attributes", {}),
                        "article_id": article_id,
                        "source": article_source,
                        "weaponization_level": node.get("attributes", {}).get("weaponization_level", 0)
                    }
                    all_narratives[node["name"]].append(narrative_data)
        
        # Merge similar events
        merged_events = self._merge_similar_events(raw_events)
        
        # Analyze cross-source patterns
        pattern_analysis = self._analyze_cross_source_patterns(
            merged_events, all_actors, all_narratives, manipulation_patterns
        )
        
        # Detect coordinated campaigns
        campaigns = self._detect_coordinated_campaigns(
            merged_events, all_narratives, authoritarian_indicators
        )
        
        # Calculate urgency scores
        urgency_analysis = self._calculate_urgency_scores(
            merged_events, authoritarian_indicators, pattern_analysis
        )
        
        # Generate event IDs and build comprehensive index
        for event in merged_events:
            event_id = self._generate_event_id(event)
            event["event_id"] = event_id
            event["urgency_score"] = urgency_analysis.get(event["name"], 0)
            event["campaign_associations"] = [c["id"] for c in campaigns if event["name"] in c["events"]]
            self.events[event_id] = event
            
            # Track which articles mentioned this event
            for article_id in event["article_ids"]:
                self.article_to_events[article_id].add(event_id)
        
        return {
            "events": list(self.events.values()),
            "event_count": len(self.events),
            "article_event_map": dict(self.article_to_events),
            "cross_source_events": self._identify_cross_source_events(),
            "pattern_analysis": pattern_analysis,
            "coordinated_campaigns": campaigns,
            "urgency_scores": urgency_analysis,
            "actor_network": self._build_actor_network(all_actors),
            "narrative_tracking": self._track_narrative_evolution(all_narratives),
            "manipulation_analysis": self._analyze_manipulation_patterns(manipulation_patterns),
            "authoritarian_escalation": self._track_authoritarian_escalation(authoritarian_indicators),
            "kg_payload": {
                "nodes": list(self.nodes.values()),
                "edges": list(self.edges.values()),
            }
        }
    
    def _merge_similar_events(self, raw_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge similar events based on temporal and semantic similarity."""
        if not raw_events:
            return []
        
        # Group by rough time window (3 days)
        time_groups = defaultdict(list)
        
        for event in raw_events:
            event_date = self._parse_date(event.get("date", "N/A"))
            if event_date:
                # Round to 3-day window
                window_key = (event_date.toordinal() // 3) * 3
                time_groups[window_key].append(event)
            else:
                time_groups["undated"].append(event)
        
        merged = []
        
        # Within each time window, merge similar events
        for window, events in time_groups.items():
            clusters = self._cluster_events(events)
            
            for cluster in clusters:
                merged_event = self._merge_event_cluster(cluster)
                merged.append(merged_event)
        
        return merged
    
    def _cluster_events(self, events: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Cluster similar events within a time window."""
        if not events:
            return []
        
        clusters = []
        used = set()
        
        for i, event1 in enumerate(events):
            if i in used:
                continue
                
            cluster = [event1]
            used.add(i)
            
            for j, event2 in enumerate(events[i+1:], i+1):
                if j in used:
                    continue
                    
                if self._events_similar(event1, event2):
                    cluster.append(event2)
                    used.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def _events_similar(self, event1: Dict[str, Any], event2: Dict[str, Any]) -> bool:
        """Determine if two events are similar enough to merge."""
        # Extract key terms from event names
        terms1 = set(event1["name"].lower().split())
        terms2 = set(event2["name"].lower().split())
        
        # Remove common words
        stopwords = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or"}
        terms1 -= stopwords
        terms2 -= stopwords
        
        # Calculate Jaccard similarity
        if not terms1 or not terms2:
            return False
            
        intersection = len(terms1 & terms2)
        union = len(terms1 | terms2)
        similarity = intersection / union
        
        # Check if events share key entities or actions
        entities1 = self._extract_entities(event1)
        entities2 = self._extract_entities(event2)
        
        if entities1 & entities2:  # Share at least one entity
            similarity += 0.3
        
        return similarity >= self.similarity_threshold
    
    def _extract_entities(self, event: Dict[str, Any]) -> Set[str]:
        """Extract entity names from event data."""
        entities = set()
        
        # From name
        name_parts = event["name"].split()
        for part in name_parts:
            if part[0].isupper() and len(part) > 2:
                entities.add(part.lower())
        
        # From attributes
        attrs = event.get("attributes", {})
        if "actors" in attrs:
            entities.update(str(attrs["actors"]).lower().split())
        if "institution" in attrs:
            entities.add(attrs["institution"].lower())
            
        return entities
    
    def _merge_event_cluster(self, cluster: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge a cluster of similar events into one."""
        # Start with the most detailed event
        base_event = max(cluster, key=lambda e: len(e.get("attributes", {})))
        
        merged = {
            "name": base_event["name"],
            "date": base_event["date"],
            "attributes": base_event["attributes"].copy(),
            "article_ids": [],
            "sources": [],
            "source_perspectives": {},
            "reporting_variance": {}
        }
        
        # Merge data from all events
        all_dates = []
        all_attributes = defaultdict(list)
        
        for event in cluster:
            # Collect article references
            merged["article_ids"].append(event["article_id"])
            
            # Track source perspectives
            source = event["article_source"]
            bias = event["article_bias"]
            merged["sources"].append(source)
            merged["source_perspectives"][source] = {
                "bias": bias,
                "description": event.get("source_sentence", "")
            }
            
            # Collect dates
            if event["date"] != "N/A":
                all_dates.append(event["date"])
            
            # Collect all attribute values
            for key, value in event.get("attributes", {}).items():
                if value not in all_attributes[key]:
                    all_attributes[key].append(value)
        
        # Resolve date conflicts (use earliest)
        if all_dates:
            merged["date"] = min(all_dates)
            if len(set(all_dates)) > 1:
                merged["reporting_variance"]["dates"] = list(set(all_dates))
        
        # Merge attributes (keep all unique values)
        for key, values in all_attributes.items():
            if len(values) == 1:
                merged["attributes"][key] = values[0]
            else:
                merged["attributes"][key] = values
                merged["reporting_variance"][key] = values
        
        # Analyze cross-source patterns
        if len(merged["sources"]) > 1:
            merged["cross_source_analysis"] = self._analyze_cross_source(merged)
        
        return merged
    
    def _analyze_cross_source(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how different sources report the same event."""
        sources = event["source_perspectives"]
        bias_spectrum = {"left": -2, "center-left": -1, "center": 0, "center-right": 1, "right": 2}
        
        # Calculate bias spread
        biases = [bias_spectrum.get(s["bias"], 0) for s in sources.values()]
        bias_variance = max(biases) - min(biases) if biases else 0
        
        return {
            "source_count": len(sources),
            "bias_variance": bias_variance,
            "unanimous": bias_variance == 0,
            "controversial": bias_variance >= 3
        }
    
    def _identify_cross_source_events(self) -> List[str]:
        """Identify events reported by multiple sources."""
        cross_source = []
        
        for event_id, event in self.events.items():
            if len(set(event["sources"])) > 1:
                cross_source.append(event_id)
        
        return cross_source
    
    def _generate_event_id(self, event: Dict[str, Any]) -> str:
        """Generate a unique ID for an event."""
        # Create a fingerprint from key event properties
        fingerprint = f"{event['name']}:{event['date']}"
        return f"evt_{hashlib.md5(fingerprint.encode()).hexdigest()[:12]}"
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string to datetime."""
        if not date_str or date_str == "N/A":
            return None
            
        try:
            # Handle ISO format
            if "T" in date_str:
                return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            # Handle date only
            return datetime.strptime(date_str, "%Y-%m-%d")
        except:
            return None
    
    def create_event_nodes(self) -> List[Dict[str, Any]]:
        """
        Create knowledge graph nodes for aggregated events.
        Returns nodes ready for KG ingestion.
        """
        nodes = []
        
        for event_id, event in self.events.items():
            node = {
                "id": event_id,
                "node_type": "aggregated_event",
                "name": event["name"],
                "attributes": {
                    **event["attributes"],
                    "source_count": len(event["sources"]),
                    "cross_source": len(set(event["sources"])) > 1,
                    "reporting_variance": event.get("reporting_variance", {}),
                    "article_ids": event["article_ids"]
                },
                "timestamp": event["date"],
                "created_at": datetime.now().isoformat()
            }
            
            # Add cross-source analysis if available
            if "cross_source_analysis" in event:
                node["attributes"]["cross_source_analysis"] = event["cross_source_analysis"]
            
            nodes.append(node)
        
        return nodes
    
    def _extract_manipulation_techniques(self, analysis: Dict[str, Any]) -> List[str]:
        """Extract specific manipulation techniques from analysis."""
        techniques = []
        
        # Look in the article analysis text
        analysis_text = analysis.get("article_analysis", "").lower()
        
        technique_patterns = {
            "fear_mongering": ["fear", "threat", "danger", "crisis"],
            "false_dichotomy": ["only choice", "either or", "no alternative"],
            "scapegoating": ["blame", "fault of", "caused by"],
            "appeal_to_authority": ["must trust", "experts say", "officials confirm"],
            "emotional_manipulation": ["outrage", "anger", "betrayal"]
        }
        
        for technique, patterns in technique_patterns.items():
            if any(pattern in analysis_text for pattern in patterns):
                techniques.append(technique)
                
        return techniques
    
    def _extract_actor_actions(self, actor_node: Dict[str, Any], edges: List[Dict[str, Any]]) -> List[str]:
        """Extract what actions an actor took from edges."""
        actions = []
        actor_id = actor_node.get("id")
        
        for edge in edges:
            if edge.get("source_id") == actor_id:
                actions.append(edge.get("relation", "unknown"))
                
        return actions
    
    def _analyze_cross_source_patterns(self, events: List[Dict], actors: Dict, 
                                     narratives: Dict, manipulation: Dict) -> Dict[str, Any]:
        """Analyze patterns across different sources."""
        return {
            "synchronized_messaging": self._detect_synchronized_messaging(events),
            "narrative_coordination": self._analyze_narrative_coordination(narratives),
            "manipulation_consistency": self._analyze_manipulation_consistency(manipulation),
            "actor_prominence": self._calculate_actor_prominence(actors),
            "bias_correlation": self._analyze_bias_correlation(events)
        }
    
    def _detect_synchronized_messaging(self, events: List[Dict]) -> Dict[str, Any]:
        """Detect when multiple sources report same events simultaneously."""
        time_clusters = defaultdict(list)
        
        for event in events:
            if event["date"] != "N/A":
                # Group by date
                time_clusters[event["date"]].append(event)
        
        synchronized = []
        for date, cluster in time_clusters.items():
            if len(cluster) > 2:  # Multiple sources on same day
                sources = set()
                for evt in cluster:
                    sources.update(evt.get("sources", []))
                
                if len(sources) > 2:
                    synchronized.append({
                        "date": date,
                        "event_count": len(cluster),
                        "source_count": len(sources),
                        "events": [e["name"] for e in cluster]
                    })
        
        return {
            "synchronized_days": len(synchronized),
            "instances": synchronized
        }
    
    def _detect_coordinated_campaigns(self, events: List[Dict], narratives: Dict, 
                                    indicators: Dict) -> List[Dict[str, Any]]:
        """Detect potential coordinated information campaigns."""
        campaigns = []
        
        # Look for narrative + event + indicator clusters
        for narrative, instances in narratives.items():
            if len(instances) < 3:  # Need multiple sources
                continue
                
            # Find events that coincide with this narrative
            related_events = []
            for event in events:
                if any(narrative.lower() in str(e).lower() for e in event.values()):
                    related_events.append(event["name"])
            
            # Check for authoritarian indicators
            related_indicators = []
            for indicator, occurrences in indicators.items():
                if len(occurrences) > 1:
                    related_indicators.append(indicator)
            
            if related_events and related_indicators:
                campaigns.append({
                    "id": f"campaign_{hashlib.md5(narrative.encode()).hexdigest()[:8]}",
                    "narrative": narrative,
                    "events": related_events,
                    "indicators": related_indicators,
                    "source_count": len(set(inst["source"] for inst in instances)),
                    "intensity": len(instances)
                })
        
        return campaigns
    
    def _calculate_urgency_scores(self, events: List[Dict], indicators: Dict, 
                                patterns: Dict) -> Dict[str, float]:
        """Calculate urgency score for each event based on authoritarian risk."""
        scores = {}
        
        for event in events:
            score = 0.0
            
            # Base score from manipulation
            score += event.get("manipulation_score", 0) * 0.1
            
            # Concern level multiplier
            concern_multipliers = {"High": 2.0, "Moderate": 1.5, "Low": 1.0, "None": 0.5}
            score *= concern_multipliers.get(event.get("concern_level", "Unknown"), 1.0)
            
            # Cross-source amplification
            if len(event.get("sources", [])) > 2:
                score *= 1.5
            
            # Campaign association
            if event.get("campaign_associations"):
                score *= 1.3
            
            # Temporal clustering (rapid events)
            # This would need temporal analysis from the events list
            
            scores[event["name"]] = min(score, 10.0)  # Cap at 10
            
        return scores
    
    def _build_actor_network(self, actors: Dict[str, List]) -> Dict[str, Any]:
        """Build network showing actor relationships and influence."""
        network = {
            "nodes": [],
            "edges": [],
            "influence_scores": {}
        }
        
        for actor_name, instances in actors.items():
            # Calculate influence based on mentions and actions
            influence = len(instances) * sum(len(inst.get("actions", [])) for inst in instances)
            
            network["nodes"].append({
                "id": actor_name,
                "type": "actor",
                "mention_count": len(instances),
                "influence": influence
            })
            
            network["influence_scores"][actor_name] = influence
        
        return network
    
    def _track_narrative_evolution(self, narratives: Dict[str, List]) -> Dict[str, Any]:
        """Track how narratives evolve across sources and time."""
        evolution = {}
        
        for narrative, instances in narratives.items():
            # Sort by article date (would need date from article metadata)
            instances_by_source = defaultdict(list)
            
            for inst in instances:
                instances_by_source[inst["source"]].append(inst)
            
            evolution[narrative] = {
                "source_spread": len(instances_by_source),
                "total_instances": len(instances),
                "weaponization_trend": self._calculate_weaponization_trend(instances),
                "source_breakdown": dict(instances_by_source)
            }
        
        return evolution
    
    def _calculate_weaponization_trend(self, instances: List[Dict]) -> str:
        """Determine if narrative weaponization is increasing."""
        if not instances:
            return "stable"
            
        weaponization_levels = [inst.get("weaponization_level", 0) for inst in instances]
        
        if len(weaponization_levels) < 2:
            return "insufficient_data"
        
        # Simple trend detection
        first_half = sum(weaponization_levels[:len(weaponization_levels)//2])
        second_half = sum(weaponization_levels[len(weaponization_levels)//2:])
        
        if second_half > first_half * 1.2:
            return "increasing"
        elif second_half < first_half * 0.8:
            return "decreasing"
        else:
            return "stable"
    
    def _analyze_manipulation_patterns(self, manipulation: Dict[str, List]) -> Dict[str, Any]:
        """Analyze manipulation patterns across sources."""
        return {
            "average_scores_by_source": {
                source: sum(m["score"] for m in instances) / len(instances)
                for source, instances in manipulation.items()
                if instances
            },
            "technique_frequency": self._count_technique_frequency(manipulation),
            "high_manipulation_sources": [
                source for source, instances in manipulation.items()
                if any(m["score"] >= 7 for m in instances)
            ]
        }
    
    def _count_technique_frequency(self, manipulation: Dict[str, List]) -> Dict[str, int]:
        """Count frequency of manipulation techniques."""
        technique_counts = defaultdict(int)
        
        for instances in manipulation.values():
            for inst in instances:
                for technique in inst.get("techniques", []):
                    technique_counts[technique] += 1
                    
        return dict(technique_counts)
    
    def _analyze_manipulation_consistency(self, manipulation: Dict[str, List]) -> Dict[str, Any]:
        """Check if sources consistently use similar manipulation levels."""
        consistency = {}
        
        for source, instances in manipulation.items():
            if len(instances) < 2:
                continue
                
            scores = [m["score"] for m in instances]
            avg = sum(scores) / len(scores)
            variance = sum((s - avg) ** 2 for s in scores) / len(scores)
            
            consistency[source] = {
                "average": avg,
                "variance": variance,
                "consistent": variance < 2.0  # Low variance = consistent
            }
            
        return consistency
    
    def _analyze_narrative_coordination(self, narratives: Dict[str, List]) -> Dict[str, Any]:
        """Detect coordinated narrative deployment."""
        coordination_indicators = []
        
        for narrative, instances in narratives.items():
            if len(instances) < 3:
                continue
                
            sources = [inst["source"] for inst in instances]
            unique_sources = set(sources)
            
            # Check for rapid deployment (would need timestamps)
            if len(unique_sources) >= 3:
                coordination_indicators.append({
                    "narrative": narrative,
                    "sources": list(unique_sources),
                    "instance_count": len(instances),
                    "coordination_score": len(unique_sources) / len(instances)
                })
        
        return {
            "coordinated_narratives": len(coordination_indicators),
            "instances": coordination_indicators
        }
    
    def _calculate_actor_prominence(self, actors: Dict[str, List]) -> Dict[str, Any]:
        """Calculate which actors are most prominent across sources."""
        prominence = {}
        
        for actor, instances in actors.items():
            sources = set(inst["source"] for inst in instances)
            actions = []
            for inst in instances:
                actions.extend(inst.get("actions", []))
            
            prominence[actor] = {
                "mention_count": len(instances),
                "source_diversity": len(sources),
                "action_count": len(actions),
                "prominence_score": len(instances) * len(sources)
            }
        
        # Sort by prominence
        return dict(sorted(prominence.items(), key=lambda x: x[1]["prominence_score"], reverse=True))
    
    def _analyze_bias_correlation(self, events: List[Dict]) -> Dict[str, Any]:
        """Analyze how bias correlates with event reporting."""
        # Track how many events each bias reports and which sources mention
        # those events. The default entry contains a count and a set for
        # storing unique sources.
        bias_patterns = defaultdict(lambda: {"event_count": 0, "sources": set()})
        
        for event in events:
            if "source_perspectives" in event:
                for source, perspective in event["source_perspectives"].items():
                    bias = perspective.get("bias", "unknown")
                    bias_patterns[bias]["event_count"] += 1
                    bias_patterns[bias]["sources"].add(source)
        
        return {
            bias: {
                "events_reported": data["event_count"],
                "unique_sources": len(data.get("sources", set()))
            }
            for bias, data in bias_patterns.items()
        }
    
    def _track_authoritarian_escalation(self, indicators: Dict[str, List]) -> Dict[str, Any]:
        """Track escalation patterns in authoritarian indicators."""
        escalation_analysis = {
            "indicator_frequency": {},
            "high_concern_sources": [],
            "escalation_timeline": []
        }
        
        for indicator, occurrences in indicators.items():
            escalation_analysis["indicator_frequency"][indicator] = len(occurrences)
            
            # Track high concern instances
            high_concern = [occ for occ in occurrences if occ.get("concern_level") == "High"]
            if high_concern:
                sources = set(occ["source"] for occ in high_concern)
                escalation_analysis["high_concern_sources"].extend(sources)
        
        escalation_analysis["high_concern_sources"] = list(set(escalation_analysis["high_concern_sources"]))
        
        # Identify escalation patterns (would need temporal data)
        total_indicators = sum(len(occs) for occs in indicators.values())
        escalation_analysis["overall_threat_level"] = min(total_indicators / 10, 10.0)  # Scale to 10
        
        return escalation_analysis
    
    def create_event_relationships(self) -> List[Dict[str, Any]]:
        """Create edges showing event relationships."""
        edges = []
        
        # Sort events by date
        sorted_events = sorted(
            [(eid, e) for eid, e in self.events.items() if e["date"] != "N/A"],
            key=lambda x: x[1]["date"]
        )
        
        # Connect events in temporal sequence
        for i in range(len(sorted_events) - 1):
            event1_id, event1 = sorted_events[i]
            event2_id, event2 = sorted_events[i + 1]
            
            # Check if events are within 7 days
            date1 = self._parse_date(event1["date"])
            date2 = self._parse_date(event2["date"])
            
            if date1 and date2 and (date2 - date1).days <= 7:
                edges.append({
                    "source_id": event1_id,
                    "relation": "precedes", 
                    "target_id": event2_id,
                    "attributes": {
                        "days_between": (date2 - date1).days,
                        "temporal_proximity": "close"
                    }
                })
        
        return edges
