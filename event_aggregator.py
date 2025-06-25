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

    def _add_edges(
        self,
        edges: List[Dict[str, Any]],
        id_map: Dict[str, str],
        source_doc: str,
        source_name: str,
    ) -> None:
        """Add edges from a single analysis using consolidated node IDs.

        Args:
            edges: Edge list from the analysis payload.
            id_map: Mapping from original node IDs to consolidated IDs.
            source_doc: Document identifier for provenance tracking.
            source_name: Human friendly source name for diversity metrics.
        """
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
                    "sources": [source_name],  # Changed from set to list
                    "source_diversity": 1,
                }
            else:
                edge_entry = self.edges[key]
                edge_entry["weight"] += 1
                edge_entry["evidence_sources"].append(source_doc)
                # Convert to list, add item, remove duplicates
                sources_list = edge_entry.get("sources", [])
                if isinstance(sources_list, set):
                    sources_list = list(sources_list)
                if source_name not in sources_list:
                    sources_list.append(source_name)
                edge_entry["sources"] = sources_list
                edge_entry["source_diversity"] = len(sources_list)

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
            self._add_edges(edges, id_map, article_id, article_source)
            
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
                    all_actors[node.get("name")].append(actor_data)
                    
                elif node_type == "narrative":
                    narrative_data = {
                        "theme": node.get("name"),
                        "attributes": node.get("attributes", {}),
                        "article_id": article_id,
                        "source": article_source,
                        "timestamp": node.get("timestamp", "N/A")
                    }
                    all_narratives[node.get("name")].append(narrative_data)
        
        # Merge similar events
        merged_events = self._merge_similar_events(raw_events)
        
        # Assign IDs FIRST - before any analysis that needs them
        for event in merged_events:
            event_id = self._generate_event_id(event)
            event["event_id"] = event_id
            event["reporting_variance"] = self._calculate_reporting_variance(event)
        
        # Pattern analysis
        pattern_analysis = self._analyze_event_patterns(merged_events)
        
        # Detect coordinated campaigns
        campaigns = self._detect_coordinated_campaigns(merged_events, all_actors, all_narratives)
        
        # Calculate urgency scores - now events have IDs
        urgency_analysis = self._calculate_urgency_scores(merged_events, campaigns)
        
        # Build final event records
        for event in merged_events:
            event["campaigns"] = [c for c in campaigns if event["name"] in c["events"]]
            self.events[event["event_id"]] = event
            
            # Track which articles mentioned this event
            for article_id in event["article_ids"]:
                self.article_to_events[article_id].add(event["event_id"])
        
        return {
            "events": list(self.events.values()),
            "event_count": len(self.events),
            "article_event_map": {k: list(v) for k, v in self.article_to_events.items()},  # Convert sets to lists
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
                "edges": list(self.edges.values())  # Now safe for JSON serialization
            },
            "analyses_processed": len(analyses),
            "unique_events": len(self.events),
            "status": "success"
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
        """Check if two events are similar enough to merge."""
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

        # Weighted sum of scores
        total = (
            0.6 * name_score
            + 0.2 * desc_score
            + 0.1 * date_score
            + 0.1 * loc_score
        )

        return total >= self.similarity_threshold

    @staticmethod
    def _fuzzy_ratio(text1: str, text2: str) -> float:
        """Return a simple fuzzy match ratio between 0 and 1."""
        if not text1 or not text2:
            return 0.0
        from difflib import SequenceMatcher

        return SequenceMatcher(None, text1, text2).ratio()

    def _temporal_score(self, d1: str, d2: str) -> float:
        """Score temporal proximity between two dates."""
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
        """Score similarity between two location strings."""
        if not loc1 or not loc2:
            return 0.0

        parts1 = [p.strip().lower() for p in loc1.split(",")]
        parts2 = [p.strip().lower() for p in loc2.split(",")]
        if parts1 == parts2:
            return 1.0

        if len(parts1) >= 2 and len(parts2) >= 2 and parts1[-2:] == parts2[-2:]:
            return 0.6

        if parts1 and parts2 and parts1[-1] == parts2[-1]:
            return 0.3

        return 0.0
    
    def _merge_event_cluster(self, cluster: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge a cluster of similar events into one."""
        if not cluster:
            return {}
        
        if len(cluster) == 1:
            # Single event, just format it
            event = cluster[0]
            return {
                **event,
                "sources": [event["article_source"]],
                "article_ids": [event["article_id"]],
                "bias_distribution": {event.get("article_bias", "Unknown"): 1}
            }
        
        # Multiple events to merge
        base_event = cluster[0].copy()
        
        # Collect all sources and IDs
        sources = []
        article_ids = []
        bias_counts = defaultdict(int)
        
        for event in cluster:
            sources.append(event["article_source"])
            article_ids.append(event["article_id"])
            bias_counts[event.get("article_bias", "Unknown")] += 1
        
        # Update base event
        base_event["sources"] = list(set(sources))
        base_event["article_ids"] = article_ids
        base_event["bias_distribution"] = dict(bias_counts)
        base_event["source_count"] = len(set(sources))
        
        # Merge attributes
        all_attributes = defaultdict(list)
        for event in cluster:
            for key, value in event.get("attributes", {}).items():
                all_attributes[key].append(value)
        
        # Consolidate attributes - FIX: Handle unhashable types
        consolidated_attributes = {}
        for key, values in all_attributes.items():
            if not values:
                continue
            
            # If all values are the same, just use the first one
            if len(values) == 1:
                consolidated_attributes[key] = values[0]
            elif all(json.dumps(v, sort_keys=True) == json.dumps(values[0], sort_keys=True) for v in values):
                consolidated_attributes[key] = values[0]
            else:
                # For different values, check if they're simple types
                if all(isinstance(v, (str, int, float, bool, type(None))) for v in values):
                    # Find most common value
                    value_counts = defaultdict(int)
                    for v in values:
                        value_counts[v] += 1
                    consolidated_attributes[key] = max(value_counts.keys(), key=lambda k: value_counts[k])
                else:
                    # For complex types (lists, dicts), keep all values
                    consolidated_attributes[key] = values
        
        base_event["attributes"] = consolidated_attributes
        
        return base_event
    
    def _analyze_event_patterns(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in the aggregated events."""
        if not events:
            return {"status": "no_events"}
        
        # Temporal clustering
        temporal_clusters = self._find_temporal_clusters(events)
        
        # Topic clustering (simple keyword-based)
        topic_clusters = self._find_topic_clusters(events)
        
        # Source diversity analysis
        source_patterns = self._analyze_source_patterns(events)
        
        return {
            "temporal_clusters": temporal_clusters,
            "topic_clusters": topic_clusters,
            "source_patterns": source_patterns,
            "event_density": self._calculate_event_density(events)
        }
    
    def _detect_coordinated_campaigns(self, events: List[Dict[str, Any]], 
                                    actors: Dict[str, List], 
                                    narratives: Dict[str, List]) -> List[Dict[str, Any]]:
        """Detect potential coordinated campaigns."""
        campaigns = []
        
        # Look for events that:
        # 1. Happen close in time
        # 2. Share actors or narratives
        # 3. Appear across multiple sources
        
        for i, event1 in enumerate(events):
            if event1.get("source_count", 0) < 2:
                continue
                
            campaign_events = [event1["name"]]
            campaign_actors = set()
            campaign_narratives = set()
            
            event1_date = self._parse_date(event1.get("date", ""))
            if not event1_date:
                continue
            
            for event2 in events[i+1:]:
                event2_date = self._parse_date(event2.get("date", ""))
                if not event2_date:
                    continue
                
                # Check temporal proximity (within 7 days)
                if abs((event2_date - event1_date).days) <= 7:
                    # Check for shared elements
                    if self._events_connected(event1, event2, actors, narratives):
                        campaign_events.append(event2["name"])
            
            if len(campaign_events) >= 2:
                campaigns.append({
                    "events": campaign_events,
                    "time_span": "7 days",
                    "sources_involved": self._get_campaign_sources(campaign_events, events),
                    "confidence": "medium"
                })
        
        return campaigns
    
    def _calculate_urgency_scores(self, events: List[Dict[str, Any]], 
                                campaigns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate urgency scores for events and campaigns."""
        scores = {}
        
        # Factors:
        # 1. Recency
        # 2. Source count
        # 3. Manipulation score
        # 4. Part of campaign
        
        current_date = datetime.now()
        
        for event in events:
            event_date = self._parse_date(event.get("date", ""))
            if not event_date:
                continue
            
            # Base score from recency (0-10)
            days_old = (current_date - event_date).days
            recency_score = max(0, 10 - (days_old / 3))
            
            # Source multiplier
            source_mult = min(event.get("source_count", 1), 3)
            
            # Manipulation modifier
            manip_score = event.get("manipulation_score", 0)
            
            # Campaign modifier
            campaign_boost = 2 if any(event["name"] in c["events"] for c in campaigns) else 1
            
            urgency = (recency_score * source_mult + manip_score) * campaign_boost
            
            scores[event["event_id"]] = {
                "score": round(urgency, 2),
                "factors": {
                    "recency": round(recency_score, 2),
                    "sources": source_mult,
                    "manipulation": manip_score,
                    "campaign": campaign_boost > 1
                }
            }
        
        return scores
    
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
        
        # Look in rounds for manipulation analysis
        for round_name, round_data in analysis.get("rounds", {}).items():
            if "manipulation" in round_name.lower():
                response = round_data.get("response", "")
                # Simple extraction - could be enhanced
                if "disinformation" in response.lower():
                    techniques.append("disinformation")
                if "fear" in response.lower():
                    techniques.append("fear-mongering")
                if "division" in response.lower():
                    techniques.append("divisive rhetoric")
        
        return techniques
    
    def _extract_actor_actions(self, actor_node: Dict[str, Any], edges: List[Dict[str, Any]]) -> List[str]:
        """Extract actions performed by an actor from edges."""
        actions = []
        actor_id = actor_node.get("id")
        
        for edge in edges:
            if edge.get("source_id") == actor_id:
                actions.append(edge.get("relation", "unknown"))
        
        return actions
    
    def _calculate_reporting_variance(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate how reporting varies across sources."""
        if len(event.get("sources", [])) < 2:
            return {"variance": "single_source"}
        
        return {
            "source_count": len(event["sources"]),
            "bias_spread": event.get("bias_distribution", {}),
            "variance": "multi_source"
        }
    
    def _find_temporal_clusters(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find clusters of events happening close in time."""
        clusters = []
        
        # Sort events by date
        dated_events = [e for e in events if self._parse_date(e.get("date", ""))]
        dated_events.sort(key=lambda e: self._parse_date(e["date"]))
        
        if not dated_events:
            return clusters
        
        # Find clusters within 3-day windows
        current_cluster = [dated_events[0]]
        cluster_start = self._parse_date(dated_events[0]["date"])
        
        for event in dated_events[1:]:
            event_date = self._parse_date(event["date"])
            if (event_date - cluster_start).days <= 3:
                current_cluster.append(event)
            else:
                if len(current_cluster) >= 3:
                    clusters.append({
                        "start_date": cluster_start.isoformat(),
                        "end_date": self._parse_date(current_cluster[-1]["date"]).isoformat(),
                        "event_count": len(current_cluster),
                        "events": [e["name"] for e in current_cluster]
                    })
                current_cluster = [event]
                cluster_start = event_date
        
        # Don't forget last cluster
        if len(current_cluster) >= 3:
            clusters.append({
                "start_date": cluster_start.isoformat(),
                "end_date": self._parse_date(current_cluster[-1]["date"]).isoformat(),
                "event_count": len(current_cluster),
                "events": [e["name"] for e in current_cluster]
            })
        
        return clusters
    
    def _find_topic_clusters(self, events: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Group events by topic keywords."""
        topic_groups = defaultdict(list)
        
        # Simple keyword-based clustering
        keywords_map = {
            "legislative": ["bill", "vote", "senate", "house", "congress", "law"],
            "judicial": ["court", "judge", "ruling", "decision", "legal"],
            "executive": ["executive", "order", "president", "administration"],
            "media": ["media", "press", "journalist", "coverage", "report"],
            "protest": ["protest", "rally", "demonstration", "march"],
            "election": ["election", "voting", "ballot", "campaign"]
        }
        
        for event in events:
            event_name = event.get("name", "").lower()
            
            for topic, keywords in keywords_map.items():
                if any(kw in event_name for kw in keywords):
                    topic_groups[topic].append(event["name"])
        
        return dict(topic_groups)
    
    def _analyze_source_patterns(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in source reporting."""
        source_stats = defaultdict(lambda: {"event_count": 0, "bias": None})
        
        for event in events:
            for source in event.get("sources", []):
                source_stats[source]["event_count"] += 1
                # Track bias if available
                if "bias_distribution" in event:
                    # Simplified - just note the bias
                    for bias in event["bias_distribution"]:
                        source_stats[source]["bias"] = bias
                        break
        
        return dict(source_stats)
    
    def _calculate_event_density(self, events: List[Dict[str, Any]]) -> float:
        """Calculate events per day metric."""
        dated_events = [e for e in events if self._parse_date(e.get("date", ""))]
        
        if len(dated_events) < 2:
            return 0.0
        
        dates = [self._parse_date(e["date"]) for e in dated_events]
        date_range = (max(dates) - min(dates)).days + 1
        
        return round(len(dated_events) / date_range, 2)
    
    def _events_connected(self, event1: Dict[str, Any], event2: Dict[str, Any],
                         actors: Dict[str, List], narratives: Dict[str, List]) -> bool:
        """Check if two events share actors or narratives."""
        # Get article IDs for both events
        ids1 = set(event1.get("article_ids", []))
        ids2 = set(event2.get("article_ids", []))
        
        # Check actors
        for actor, occurrences in actors.items():
            actor_articles = {occ["article_id"] for occ in occurrences}
            if ids1 & actor_articles and ids2 & actor_articles:
                return True
        
        # Check narratives
        for narrative, occurrences in narratives.items():
            narrative_articles = {occ["article_id"] for occ in occurrences}
            if ids1 & narrative_articles and ids2 & narrative_articles:
                return True
        
        return False
    
    def _get_campaign_sources(self, event_names: List[str], 
                            all_events: List[Dict[str, Any]]) -> List[str]:
        """Get all sources involved in a campaign."""
        sources = set()
        
        for event in all_events:
            if event["name"] in event_names:
                sources.update(event.get("sources", []))
        
        return list(sources)
    
    def _build_actor_network(self, actors: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Build network analysis of actors."""
        network = {
            "nodes": [],
            "edges": [],
            "centrality": {}
        }
        
        # Create nodes for each actor
        for actor_name, occurrences in actors.items():
            node = {
                "id": actor_name,
                "occurrences": len(occurrences),
                "sources": list(set(occ["source"] for occ in occurrences)),
                "actions": list(set(action for occ in occurrences for action in occ.get("actions", [])))
            }
            network["nodes"].append(node)
        
        # Simple centrality based on occurrence count
        total_occurrences = sum(len(occs) for occs in actors.values())
        for actor_name, occurrences in actors.items():
            network["centrality"][actor_name] = round(len(occurrences) / total_occurrences, 3)
        
        return network
    
    def _track_narrative_evolution(self, narratives: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Track how narratives evolve over time."""
        evolution = {}
        
        for narrative, occurrences in narratives.items():
            # Sort by date
            dated_occs = [occ for occ in occurrences if self._parse_date(occ.get("timestamp", ""))]
            dated_occs.sort(key=lambda x: self._parse_date(x["timestamp"]))
            
            if dated_occs:
                evolution[narrative] = {
                    "first_seen": dated_occs[0]["timestamp"],
                    "last_seen": dated_occs[-1]["timestamp"],
                    "source_progression": [occ["source"] for occ in dated_occs],
                    "occurrence_count": len(occurrences)
                }
        
        return evolution
    
    def _analyze_manipulation_patterns(self, patterns: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze manipulation patterns by source."""
        analysis = {}
        
        for source, instances in patterns.items():
            if instances:
                scores = [inst["score"] for inst in instances]
                techniques_lists = [inst["techniques"] for inst in instances]
                all_techniques = [t for sublist in techniques_lists for t in sublist]
                
                analysis[source] = {
                    "average_score": round(sum(scores) / len(scores), 2),
                    "max_score": max(scores),
                    "instance_count": len(instances),
                    "common_techniques": list(set(all_techniques))
                }
        
        return analysis
    
    def _track_authoritarian_escalation(self, indicators: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Track escalation of authoritarian indicators."""
        escalation = {
            "indicator_frequency": {},
            "source_breakdown": defaultdict(list),
            "concern_levels": defaultdict(int),
            "timeline": []
        }
        
        for indicator, occurrences in indicators.items():
            escalation["indicator_frequency"][indicator] = len(occurrences)
            
            for occ in occurrences:
                escalation["source_breakdown"][occ["source"]].append(indicator)
                escalation["concern_levels"][occ["concern_level"]] += 1
        
        # Convert defaultdicts to regular dicts
        escalation["source_breakdown"] = dict(escalation["source_breakdown"])
        escalation["concern_levels"] = dict(escalation["concern_levels"])
        
        return escalation
    
    def create_event_relationships(self) -> List[Dict[str, Any]]:
        """
        Create edges between aggregated events based on temporal and thematic relationships.
        """
        edges = []
        events_list = list(self.events.values())
        
        for i, event1 in enumerate(events_list):
            event1_date = self._parse_date(event1.get("date", ""))
            if not event1_date:
                continue
                
            for event2 in events_list[i+1:]:
                event2_date = self._parse_date(event2.get("date", ""))
                if not event2_date:
                    continue
                
                # Check temporal relationship
                days_diff = (event2_date - event1_date).days
                
                if 0 < days_diff <= 7:
                    # Events within a week might be related
                    edge = {
                        "source_id": event1["event_id"],
                        "target_id": event2["event_id"], 
                        "relation": "precedes",
                        "attributes": {
                            "days_apart": days_diff,
                            "shared_sources": list(set(event1["sources"]) & set(event2["sources"]))
                        }
                    }
                    
                    # Check for shared campaign
                    shared_campaigns = set(c["events"] for c in event1.get("campaigns", [])) & \
                                     set(c["events"] for c in event2.get("campaigns", []))
                    if shared_campaigns:
                        edge["relation"] = "part_of_campaign"
                        edge["attributes"]["campaign"] = True
                    
                    edges.append(edge)
        
        return edges