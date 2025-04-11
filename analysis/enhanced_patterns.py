"""
Night_watcher Enhanced Pattern Recognition Module
Advanced pattern detection and analysis with predictive capabilities.
"""

import re
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import Counter, defaultdict

import numpy as np

from memory.system import MemorySystem
from memory.knowledge_graph import KnowledgeGraph, Entity, Relationship

logger = logging.getLogger(__name__)


class EnhancedPatternRecognition:
    """
    Advanced pattern detection and analysis for authoritarian trends, actors, and narratives.
    """

    def __init__(self, memory_system: MemorySystem, knowledge_graph: KnowledgeGraph = None):
        """Initialize with memory system and optional knowledge graph"""
        self.memory = memory_system
        self.knowledge_graph = knowledge_graph or KnowledgeGraph()
        self.logger = logging.getLogger("EnhancedPatternRecognition")

        # Define authoritarian indicators and map them to consistent names
        self.authoritarian_indicators = {
            "institutional_undermining": [
                "institutional undermining",
                "weakening institutions",
                "attack on institutions"
            ],
            "democratic_norm_violations": [
                "democratic norm violations",
                "norm violations",
                "violation of norms"
            ],
            "media_delegitimization": [
                "media delegitimization",
                "attack on media",
                "fake news claims"
            ],
            "opposition_targeting": [
                "opposition targeting",
                "attack on opposition",
                "enemy of the people"
            ],
            "power_concentration": [
                "power concentration",
                "executive overreach",
                "centralizing power"
            ],
            "accountability_evasion": [
                "accountability evasion",
                "avoiding oversight",
                "evading checks"
            ],
            "threat_exaggeration": [
                "threat exaggeration",
                "crisis exaggeration",
                "fearmongering"
            ],
            "authoritarian_rhetoric": [
                "authoritarian rhetoric",
                "strongman language",
                "cult of personality"
            ],
            "rule_of_law_undermining": [
                "rule of law undermining",
                "undermining courts",
                "law enforcement politicization"
            ]
        }

        # Named entities to track (can be expanded)
        self.tracked_entities = set()

        # Load tracked entities from memory if available
        self._load_tracked_entities()

    def _load_tracked_entities(self):
        """Load tracked entities from memory or initialize with defaults"""
        # For now, add some default entities to track
        default_entities = [
            "Donald Trump", "Joe Biden", "Supreme Court", "Congress", "FBI",
            "Justice Department", "White House", "Republican Party", "Democratic Party"
        ]

        for entity in default_entities:
            self.tracked_entities.add(entity)

    def extract_authoritarian_elements(self, analysis_text: str) -> Dict[str, Any]:
        """
        Extract authoritarian elements from analysis text with enhanced detection.

        Args:
            analysis_text: Analysis text to extract from

        Returns:
            Dictionary of authoritarian elements with presence, frequency, and examples
        """
        elements = {}

        # Extract authoritarian score
        score_match = re.search(r'AUTHORITARIAN SCORE:?\s*(\d+)[^\d]', analysis_text)
        if score_match:
            score = int(score_match.group(1))
            elements["authoritarian_score"] = score
        else:
            elements["authoritarian_score"] = 0

        # Extract indicators with enhanced pattern matching
        for indicator_key, indicator_patterns in self.authoritarian_indicators.items():
            # Create comprehensive search pattern
            indicator_variants = "|".join(indicator_patterns)
            pattern = rf'(?i)({indicator_variants})(.*?)(?=\n\n\d+\.|\n\n[A-Z]|\Z)'

            # Search for the indicator pattern
            matches = re.finditer(pattern, analysis_text, re.DOTALL)

            has_matches = False
            examples = []
            evidence_text = ""

            for match in matches:
                has_matches = True
                match_text = match.group(2).strip()
                evidence_text += match_text

                # Extract examples
                if match_text:
                    # Split by bullet points, periods, or other separators
                    for line in re.split(r'(?:•|\*|\-|\d+\.|\n)', match_text):
                        line = line.strip()
                        if line and len(line) > 10:
                            examples.append(line)

            # Check for explicit absence indicators
            negative_patterns = ["not present", "no evidence", "none found", "not identified"]
            is_present = has_matches and not any(neg in evidence_text.lower() for neg in negative_patterns)

            elements[indicator_key] = {
                "present": is_present,
                "examples": examples,
                "frequency": len(examples) if is_present else 0,
                "evidence_text": evidence_text if is_present else ""
            }

        # Extract named entities
        elements["entities"] = self._extract_named_entities(analysis_text)

        return elements

    def _extract_named_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text with basic pattern matching.

        Args:
            text: Text to extract entities from

        Returns:
            List of extracted entities with type and positions
        """
        entities = []

        # Check for tracked entities
        for entity_name in self.tracked_entities:
            if entity_name in text:
                # Find all occurrences
                for match in re.finditer(r'\b' + re.escape(entity_name) + r'\b', text):
                    start, end = match.span()
                    entities.append({
                        "text": entity_name,
                        "start": start,
                        "end": end,
                        "type": "PERSON" if any(name in entity_name for name in ["Trump", "Biden"]) else "ORG"
                    })

        # Add basic pattern matching for other possible entities
        # Organizations
        for match in re.finditer(r'\b(?:The\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,5})\b', text):
            entity_name = match.group(0)
            if entity_name not in self.tracked_entities and not any(word in entity_name for word in ["AUTHORITARIAN", "SCORE", "INSTITUTIONAL"]):
                start, end = match.span()
                entities.append({
                    "text": entity_name,
                    "start": start,
                    "end": end,
                    "type": "ORG"
                })

        return entities

    def analyze_authoritarian_trend_patterns(self, lookback_days: int = 90) -> Dict[str, Any]:
        """
        Analyze trends in authoritarian governance indicators over time with enhanced features.

        Args:
            lookback_days: Days to look back for analysis

        Returns:
            Dictionary containing trend analysis of authoritarian indicators
        """
        # Get analyses within the lookback period
        recent_analyses = self.memory.get_recent_analyses(lookback_days)

        if not recent_analyses:
            return {"error": "No recent analyses found"}

        # Track indicators and entities over time
        indicators = {indicator: [] for indicator in self.authoritarian_indicators.keys()}
        indicators["authoritarian_score"] = []
        entities_timeline = defaultdict(list)

        # Process analyses to extract indicators and entities
        for analysis in recent_analyses:
            metadata = analysis.get("metadata", {})
            timestamp = metadata.get("analysis_timestamp", "")
            if not timestamp:
                timestamp = datetime.now().isoformat()

            # Extract authoritarian indicators and score from analysis
            analysis_text = analysis.get("text", "")
            auth_elements = self.extract_authoritarian_elements(analysis_text)

            # Track authoritarian score
            if "authoritarian_score" in auth_elements:
                indicators["authoritarian_score"].append({
                    "date": timestamp,
                    "source": metadata.get("source", ""),
                    "title": metadata.get("title", ""),
                    "url": metadata.get("url", ""),
                    "score": auth_elements["authoritarian_score"]
                })

            # Track named entities
            for entity in auth_elements.get("entities", []):
                entity_name = entity["text"]
                entity_type = entity["type"]

                # Add to entities timeline
                entities_timeline[entity_name].append({
                    "date": timestamp,
                    "source": metadata.get("source", ""),
                    "title": metadata.get("title", ""),
                    "url": metadata.get("url", ""),
                    "entity_type": entity_type
                })

                # Also add entity to knowledge graph if not already there
                if self.knowledge_graph:
                    if entity_type == "PERSON":
                        self.knowledge_graph.find_or_create_actor(entity_name)
                    else:
                        self.knowledge_graph.find_or_create_entity(entity_name, self.knowledge_graph.ORGANIZATION)

            # Track indicators
            for indicator in self.authoritarian_indicators.keys():
                if indicator in auth_elements and auth_elements[indicator]["present"]:
                    indicator_data = {
                        "date": timestamp,
                        "source": metadata.get("source", ""),
                        "title": metadata.get("title", ""),
                        "url": metadata.get("url", ""),
                        "examples": auth_elements[indicator]["examples"],
                        "frequency": auth_elements[indicator]["frequency"]
                    }
                    indicators[indicator].append(indicator_data)

                    # Add to knowledge graph
                    if self.knowledge_graph:
                        # Find or create indicator entity
                        indicator_id = self.knowledge_graph.find_or_create_entity(
                            indicator.replace("_", " ").title(),
                            self.knowledge_graph.INDICATOR
                        )

                        # Add relationships for entities mentioned in the examples
                        for entity in auth_elements.get("entities", []):
                            entity_name = entity["text"]
                            if entity_type == "PERSON":
                                actor_id = self.knowledge_graph.find_or_create_actor(entity_name)
                                self.knowledge_graph.add_relationship(
                                    actor_id, indicator_id, self.knowledge_graph.DEMONSTRATES,
                                    weight=auth_elements[indicator]["frequency"],
                                    attributes={"date": timestamp, "source": metadata.get("source", "")}
                                )

        # Calculate trend strength for each indicator
        trend_analysis = {}
        for indicator, occurrences in indicators.items():
            trend_analysis[indicator] = {
                "count": len(occurrences),
                "examples": occurrences[:3],  # Top 3 examples
                "trend_strength": self._calculate_trend_strength(occurrences, lookback_days),
                "timeline": self._create_timeline(occurrences, lookback_days),
                "entities_association": self._analyze_entity_indicator_association(indicator, occurrences, entities_timeline)
            }

        # Calculate aggregate authoritarian risk
        aggregate_risk = self._calculate_aggregate_risk(trend_analysis)

        # Add additional insights if we have knowledge graph
        graph_insights = {}
        if self.knowledge_graph:
            graph_insights = self.knowledge_graph.analyze_authoritarian_trends()

        return {
            "lookback_days": lookback_days,
            "trend_analysis": trend_analysis,
            "entity_timeline": {k: v for k, v in entities_timeline.items() if len(v) > 1},
            "aggregate_authoritarian_risk": aggregate_risk,
            "risk_level": self._determine_risk_level(aggregate_risk),
            "graph_insights": graph_insights
        }

    def _analyze_entity_indicator_association(self, indicator: str, indicator_occurrences: List[Dict[str, Any]],
                                             entities_timeline: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Analyze which entities are most associated with a specific indicator.

        Args:
            indicator: The indicator name
            indicator_occurrences: List of indicator occurrences
            entities_timeline: Timeline of entity mentions

        Returns:
            List of entities associated with the indicator
        """
        # Get all dates when this indicator was observed
        indicator_dates = set(occ["date"] for occ in indicator_occurrences)

        # Calculate entity associations
        entity_associations = []

        for entity_name, occurrences in entities_timeline.items():
            # Count occurrences that match indicator dates
            matching_dates = sum(1 for occ in occurrences if occ["date"] in indicator_dates)

            if matching_dates > 0:
                # Calculate association strength (percentage of indicator occurrences with this entity)
                association_strength = matching_dates / len(indicator_occurrences)

                entity_associations.append({
                    "entity": entity_name,
                    "type": occurrences[0]["entity_type"],
                    "occurrences": matching_dates,
                    "association_strength": association_strength
                })

        # Sort by association strength
        entity_associations.sort(key=lambda x: x["association_strength"], reverse=True)

        return entity_associations[:5]  # Return top 5 associations

    def _calculate_trend_strength(self, occurrences: List[Dict[str, Any]], lookback_days: int) -> float:
        """
        Calculate the strength of a trend based on occurrences with enhanced weighting.

        Args:
            occurrences: List of occurrences
            lookback_days: Total days to look back

        Returns:
            Trend strength as a float from 0-1
        """
        if not occurrences:
            return 0.0

        # Calculate recent vs older ratio with recency bias
        midpoint = datetime.now() - timedelta(days=lookback_days/2)
        midpoint_str = midpoint.isoformat()

        # Count occurrences with frequency weighting
        recent_count = sum(occ.get("frequency", 1) for occ in occurrences if occ.get("date", "") >= midpoint_str)
        older_count = sum(occ.get("frequency", 1) for occ in occurrences if occ.get("date", "") < midpoint_str)

        # Calculate trend direction (positive means increasing)
        trend_direction = 0
        if older_count > 0:
            trend_direction = (recent_count / max(1, older_count)) - 1
        elif recent_count > 0:
            trend_direction = 1  # All occurrences are recent

        # Calculate frequency relative to total period with recency bias
        total_weighted_occurrences = sum(
            occ.get("frequency", 1) * (2.0 if occ.get("date", "") >= midpoint_str else 1.0)
            for occ in occurrences
        )
        frequency = total_weighted_occurrences / lookback_days

        # Combine into trend strength (0-1 scale)
        strength = min(1.0, (frequency * 10) * (0.5 + max(0, trend_direction)))

        return strength

    def _create_timeline(self, occurrences: List[Dict[str, Any]], lookback_days: int) -> List[Dict[str, Any]]:
        """
        Create a timeline of occurrences for visualization with enhanced features.

        Args:
            occurrences: List of occurrences
            lookback_days: Total days to look back

        Returns:
            Timeline data points
        """
        if not occurrences:
            return []

        # Sort by date
        sorted_occurrences = sorted(occurrences, key=lambda x: x.get("date", ""))

        # Group by week
        timeline = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        # Create weekly buckets
        current_date = start_date
        while current_date < end_date:
            week_end = current_date + timedelta(days=7)
            week_start_str = current_date.isoformat()
            week_end_str = min(week_end, end_date).isoformat()

            # Count occurrences in this week with frequency weighting
            week_occurrences = [
                o for o in sorted_occurrences
                if o.get("date", "") >= week_start_str and o.get("date", "") < week_end_str
            ]

            # Calculate total frequency in this week
            week_frequency = sum(o.get("frequency", 1) for o in week_occurrences)

            if week_occurrences:
                timeline.append({
                    "start_date": week_start_str,
                    "end_date": week_end_str,
                    "count": len(week_occurrences),
                    "frequency": week_frequency,
                    "items": week_occurrences
                })
            else:
                timeline.append({
                    "start_date": week_start_str,
                    "end_date": week_end_str,
                    "count": 0,
                    "frequency": 0,
                    "items": []
                })

            current_date = week_end

        return timeline

    def _calculate_aggregate_risk(self, trend_analysis: Dict[str, Any]) -> float:
        """
        Calculate aggregate authoritarian risk score with enhanced weighting.

        Args:
            trend_analysis: Trend analysis dictionary

        Returns:
            Aggregate risk score from 0-10
        """
        # Weights for different indicators with enhanced prioritization
        weights = {
            "institutional_undermining": 1.8,
            "democratic_norm_violations": 1.7,
            "media_delegitimization": 1.5,
            "opposition_targeting": 1.4,
            "power_concentration": 1.6,
            "accountability_evasion": 1.2,
            "threat_exaggeration": 1.0,
            "authoritarian_rhetoric": 1.1,
            "rule_of_law_undermining": 1.7,
            "authoritarian_score": 2.0
        }

        # Calculate weighted score
        total_weight = sum(weights.values())
        weighted_score = 0

        for indicator, weight in weights.items():
            if indicator in trend_analysis:
                indicator_data = trend_analysis[indicator]

                # For authoritarian_score, use average of scores
                if indicator == "authoritarian_score" and indicator_data["examples"]:
                    scores = [item.get("score", 0) for item in indicator_data["examples"] if "score" in item]
                    avg_score = sum(scores) / len(scores) if scores else 0
                    weighted_score += (avg_score / 10) * weight  # Normalize to 0-1
                else:
                    # For other indicators, use trend strength and frequency
                    trend_strength = indicator_data.get("trend_strength", 0)
                    count = indicator_data.get("count", 0)

                    # Combine trend strength with occurrence count for a more nuanced score
                    indicator_score = trend_strength * min(1.0, count / 10)
                    weighted_score += indicator_score * weight

        # Normalize to 0-10 scale with exponential emphasis on higher scores
        # This creates more sensitivity at higher risk levels
        aggregate_risk = (weighted_score / total_weight) * 10

        # Apply exponential emphasis to make moderate risks more apparent
        if aggregate_risk > 3:
            aggregate_risk = 3 + (aggregate_risk - 3) * 1.3

        # Cap at 10
        aggregate_risk = min(10, aggregate_risk)

        return aggregate_risk

    def _determine_risk_level(self, risk_score: float) -> str:
        """
        Determine risk level category from score with enhanced granularity.

        Args:
            risk_score: Aggregate risk score

        Returns:
            Risk level category
        """
        if risk_score < 2:
            return "Low"
        elif risk_score < 3.5:
            return "Moderate-Low"
        elif risk_score < 5:
            return "Moderate"
        elif risk_score < 6.5:
            return "Substantial"
        elif risk_score < 8:
            return "High"
        else:
            return "Severe"

    def analyze_authoritarian_actors(self, lookback_days: int = 90) -> Dict[str, Any]:
        """
        Analyze actors associated with authoritarian patterns.

        Args:
            lookback_days: Days to look back for analysis

        Returns:
            Dictionary containing actor analysis
        """
        # Get analyses within the lookback period
        recent_analyses = self.memory.get_recent_analyses(lookback_days)

        if not recent_analyses:
            return {"error": "No recent analyses found"}

        # Track actor mentions and associations with indicators
        actor_mentions = defaultdict(int)
        actor_indicators = defaultdict(lambda: defaultdict(int))
        actor_examples = defaultdict(list)
        actor_sources = defaultdict(set)

        # Process analyses
        for analysis in recent_analyses:
            metadata = analysis.get("metadata", {})
            timestamp = metadata.get("analysis_timestamp", "")
            source = metadata.get("source", "")
            title = metadata.get("title", "")
            url = metadata.get("url", "")

            # Extract authoritarian indicators and entities
            analysis_text = analysis.get("text", "")
            auth_elements = self.extract_authoritarian_elements(analysis_text)

            # Process entities
            for entity in auth_elements.get("entities", []):
                if entity["type"] == "PERSON":
                    actor_name = entity["text"]

                    # Count mention
                    actor_mentions[actor_name] += 1

                    # Track source
                    actor_sources[actor_name].add(source)

                    # Check for indicator associations in the same analysis
                    for indicator, data in auth_elements.items():
                        if indicator in self.authoritarian_indicators and data.get("present", False):
                            # Increment indicator count for this actor
                            actor_indicators[actor_name][indicator] += data.get("frequency", 1)

                            # Add example if we have room
                            if len(actor_examples[actor_name]) < 5:
                                actor_examples[actor_name].append({
                                    "indicator": indicator,
                                    "example": data["examples"][0] if data["examples"] else "",
                                    "source": source,
                                    "title": title,
                                    "date": timestamp,
                                    "url": url
                                })

        # Calculate authoritarian pattern scores
        actor_scores = {}

        for actor, indicators in actor_indicators.items():
            # Count total indicator instances
            total_instances = sum(indicators.values())

            # Count unique indicators
            unique_indicators = len(indicators)

            # Calculate authoritarian pattern score (0-10)
            # Based on breadth (unique indicators) and depth (total instances)
            breadth_score = unique_indicators / len(self.authoritarian_indicators)
            depth_score = min(1.0, total_instances / 20)  # Cap at 20 instances

            pattern_score = (breadth_score * 0.6 + depth_score * 0.4) * 10

            # Get most frequent indicator
            most_frequent = max(indicators.items(), key=lambda x: x[1]) if indicators else (None, 0)

            actor_scores[actor] = {
                "authoritarian_pattern_score": pattern_score,
                "risk_level": self._determine_risk_level(pattern_score),
                "total_mentions": actor_mentions[actor],
                "indicator_counts": dict(indicators),
                "unique_indicators": unique_indicators,
                "most_frequent_indicator": {
                    "indicator": most_frequent[0],
                    "count": most_frequent[1]
                },
                "sources": list(actor_sources[actor]),
                "examples": actor_examples[actor]
            }

        # Sort actors by score
        sorted_actors = sorted(
            [(actor, data) for actor, data in actor_scores.items()],
            key=lambda x: x[1]["authoritarian_pattern_score"],
            reverse=True
        )

        return {
            "lookback_days": lookback_days,
            "actor_patterns": {actor: data for actor, data in sorted_actors},
            "top_actors": [actor for actor, _ in sorted_actors[:5]],
            "timestamp": datetime.now().isoformat()
        }

    def identify_recurring_topics(self, min_count: int = 3) -> Dict[str, Any]:
        """
        Identify recurring topics and track their authoritarian scores over time.

        Args:
            min_count: Minimum number of occurrences to consider a topic recurring

        Returns:
            Dictionary containing recurring topics analysis
        """
        # Get topics summary with enhanced extraction
        topics_summary = self._extract_enhanced_topic_summary(limit=100)

        recurring_topics = {}

        for topic, data in topics_summary.get("top_topics", {}).items():
            if data["count"] >= min_count:
                # For each recurring topic, get all matching analyses
                query = topic
                matching_analyses = self.memory.find_similar_analyses(query, limit=20)

                if matching_analyses:
                    # Extract timestamps and scores
                    timeline = []
                    total_score = 0
                    total_auth_score = 0
                    auth_score_count = 0
                    indicator_counts = defaultdict(int)
                    entity_mentions = defaultdict(int)

                    for analysis in matching_analyses:
                        metadata = analysis.get("metadata", {})
                        timestamp = metadata.get("analysis_timestamp", "")
                        score = metadata.get("manipulation_score", 0)

                        # Extract authoritarian elements
                        auth_elements = self.extract_authoritarian_elements(analysis.get("text", ""))
                        auth_score = auth_elements.get("authoritarian_score", 0)

                        # Track indicators
                        for indicator, data in auth_elements.items():
                            if indicator in self.authoritarian_indicators and data.get("present", False):
                                indicator_counts[indicator] += 1

                        # Track entities
                        for entity in auth_elements.get("entities", []):
                            entity_mentions[entity["text"]] += 1

                        if auth_score > 0:
                            total_auth_score += auth_score
                            auth_score_count += 1

                        if timestamp:
                            timeline.append({
                                "id": analysis.get("id", ""),
                                "timestamp": timestamp,
                                "score": score,
                                "auth_score": auth_score,
                                "title": metadata.get("title", ""),
                                "source": metadata.get("source", ""),
                                "bias_label": metadata.get("bias_label", "unknown")
                            })
                            total_score += score

                    # Sort by timestamp
                    timeline.sort(key=lambda x: x["timestamp"])

                    # Identify trends
                    auth_trend = None
                    if len(timeline) >= 3:
                        first_half = timeline[:len(timeline)//2]
                        second_half = timeline[len(timeline)//2:]

                        first_avg = sum(item.get("auth_score", 0) for item in first_half) / len(first_half) if first_half else 0
                        second_avg = sum(item.get("auth_score", 0) for item in second_half) / len(second_half) if second_half else 0

                        if second_avg > first_avg * 1.25:
                            auth_trend = "increasing"
                        elif second_avg < first_avg * 0.75:
                            auth_trend = "decreasing"
                        else:
                            auth_trend = "stable"

                    # Sort entities by mention count
                    top_entities = sorted(
                        [(entity, count) for entity, count in entity_mentions.items()],
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]  # Top 5 entities

                    recurring_topics[topic] = {
                        "count": data["count"],
                        "average_score": total_score / len(timeline) if timeline else 0,
                        "average_auth_score": total_auth_score / auth_score_count if auth_score_count else 0,
                        "examples": data.get("examples", []),
                        "timeline": timeline,
                        "related_indicators": dict(indicator_counts),
                        "top_entities": [{"entity": e, "count": c} for e, c in top_entities],
                        "authoritarian_trend": auth_trend
                    }

        # Sort by count and auth score combined
        sorted_topics = sorted(
            [(topic, data) for topic, data in recurring_topics.items()],
            key=lambda x: (x[1]["average_auth_score"] * 0.7 + x[1]["count"] * 0.3),
            reverse=True
        )

        return {
            "recurring_topics": {topic: data for topic, data in sorted_topics},
            "total_topics_analyzed": len(topics_summary.get("top_topics", {})),
            "min_count_threshold": min_count,
            "timestamp": datetime.now().isoformat()
        }

    def _extract_enhanced_topic_summary(self, limit: int = 10) -> Dict[str, Any]:
        """
        Generate an enhanced summary of top topics in the memory system.

        Args:
            limit: Maximum number of topics to include

        Returns:
            Summary of topics with counts and examples
        """
        topics = {}
        bias_distribution = {}
        entity_correlations = defaultdict(lambda: defaultdict(int))

        # Scan all items
        for item_id, item in self.memory.store.items.items():
            metadata = item.get("metadata", {})

            if metadata.get("type") == "article_analysis":
                # Extract topics from analysis text
                analysis = item.get("text", "")
                extracted_topics = self._extract_topics(analysis)

                # Extract entities
                auth_elements = self.extract_authoritarian_elements(analysis)
                entities = auth_elements.get("entities", [])

                # Count topics
                for topic in extracted_topics:
                    if topic in topics:
                        topics[topic]["count"] += 1
                        if len(topics[topic]["examples"]) < 3:  # Keep up to 3 examples
                            topics[topic]["examples"].append({
                                "id": item_id,
                                "title": metadata.get("title", ""),
                                "source": metadata.get("source", "")
                            })
                    else:
                        topics[topic] = {
                            "count": 1,
                            "examples": [{
                                "id": item_id,
                                "title": metadata.get("title", ""),
                                "source": metadata.get("source", "")
                            }],
                            "entities": []
                        }

                    # Track entity correlations with topics
                    for entity in entities:
                        entity_name = entity["text"]
                        entity_correlations[topic][entity_name] += 1

                        # Add entity to topic if we have room and enough correlation
                        if entity_name not in [e["entity"] for e in topics[topic]["entities"]] and \
                           len(topics[topic]["entities"]) < 5:
                            topics[topic]["entities"].append({
                                "entity": entity_name,
                                "type": entity["type"]
                            })

                # Count bias labels
                bias = metadata.get("bias_label", "unknown")
                bias_distribution[bias] = bias_distribution.get(bias, 0) + 1

        # Sort topics by count
        sorted_topics = sorted(
            [(topic, data) for topic, data in topics.items()],
            key=lambda x: x[1]["count"],
            reverse=True
        )

        # Take top N
        top_topics = {topic: data for topic, data in sorted_topics[:limit]}

        return {
            "top_topics": top_topics,
            "bias_distribution": bias_distribution,
            "entity_correlations": {t: dict(e) for t, e in entity_correlations.items() if t in top_topics},
            "total_analyses": sum(1 for meta in self.memory.store.items.values()
                               if meta.get("metadata", {}).get("type") == "article_analysis")
        }

    def _extract_topics(self, analysis: str) -> List[str]:
        """Extract topics from analysis text with enhanced patterns"""
        topics = []

        try:
            if "MAIN TOPICS" in analysis:
                topics_section = analysis.split("MAIN TOPICS:")[1].split("\n\n")[0]

                # Extract with improved pattern matching
                # Look for standalone topics, bullet points, or numbered lists
                for item in re.split(r'[,\n•\*\-]|\d+\.', topics_section):
                    topic = item.strip()
                    if topic and len(topic) > 3:
                        # Clean up and normalize
                        topic = re.sub(r'^[\d\.\-\*]+\s*', '', topic)
                        topic = re.sub(r'["\']', '', topic)  # Remove quotes

                        if topic and not topic.upper() == topic:  # Skip all-caps headers
                            topics.append(topic)

            # Also look for key phrases in the analysis
            key_phrases = [
                r"discussion of ([^\.]+)",
                r"focus on ([^\.]+)",
                r"centered around ([^\.]+)",
                r"emphasizes ([^\.]+)",
                r"highlights ([^\.]+)",
            ]

            for pattern in key_phrases:
                for match in re.finditer(pattern, analysis, re.IGNORECASE):
                    phrase = match.group(1).strip()
                    if phrase and len(phrase) > 3 and len(phrase) < 50 and phrase not in topics:
                        topics.append(phrase)

        except Exception as e:
            self.logger.error(f"Error extracting topics: {str(e)}")

        return topics

    def predict_authoritarian_escalation(self, forecast_days: int = 30) -> Dict[str, Any]:
        """
        Predict potential authoritarian trend escalation in the near future.

        Args:
            forecast_days: Number of days to forecast

        Returns:
            Dictionary containing escalation predictions
        """
        # Get recent trend analysis
        trends = self.analyze_authoritarian_trend_patterns(lookback_days=90)

        if "error" in trends:
            return {"error": trends["error"]}

        # Analyze actors
        actors = self.analyze_authoritarian_actors(lookback_days=90)

        # Get trending topics
        topics = self.identify_recurring_topics(min_count=2)

        # Calculate escalation risk for indicators
        indicator_escalation = {}

        for indicator, data in trends.get("trend_analysis", {}).items():
            if indicator == "authoritarian_score":
                continue

            # Check if trend is already strong and consistent
            trend_strength = data.get("trend_strength", 0)
            count = data.get("count", 0)

            # Basic escalation probability based on recent trend strength
            escalation_probability = 0

            if trend_strength > 0.7:
                # Already strong trend, check if accelerating
                timeline = data.get("timeline", [])

                if timeline:
                    # Check last 3 points for acceleration
                    last_points = [point for point in timeline[-3:] if "frequency" in point]

                    if len(last_points) >= 2:
                        # Check for acceleration
                        frequencies = [point["frequency"] for point in last_points]
                        is_accelerating = True

                        for i in range(1, len(frequencies)):
                            if frequencies[i] <= frequencies[i-1]:
                                is_accelerating = False
                                break

                        if is_accelerating:
                            escalation_probability = 0.8
                        else:
                            escalation_probability = 0.5  # Strong but not accelerating
                    else:
                        escalation_probability = 0.5  # Strong but limited data points
                else:
                    escalation_probability = 0.4  # Strong but no timeline
            elif trend_strength > 0.4:
                # Moderate trend
                escalation_probability = 0.3
            elif count > 0:
                # Weak but present
                escalation_probability = 0.1

            # Adjust based on actor involvement
            actor_adjustment = 0

            # Check if high-risk actors are associated with this indicator
            for actor_name, actor_data in actors.get("actor_patterns", {}).items():
                actor_score = actor_data.get("authoritarian_pattern_score", 0)

                if actor_score > 7:  # High-risk actor
                    indicator_count = actor_data.get("indicator_counts", {}).get(indicator, 0)

                    if indicator_count > 0:
                        # This high-risk actor is associated with this indicator
                        actor_adjustment += 0.1

            # Cap adjustment
            actor_adjustment = min(0.3, actor_adjustment)

            # Final probability
            final_probability = min(0.95, escalation_probability + actor_adjustment)

            indicator_escalation[indicator] = {
                "current_trend_strength": trend_strength,
                "escalation_probability": final_probability,
                "risk_level": "High" if final_probability > 0.7 else
                             "Moderate" if final_probability > 0.3 else
                             "Low",
                "contributing_actors": [
                    actor for actor, data in actors.get("actor_patterns", {}).items()
                    if data.get("indicator_counts", {}).get(indicator, 0) > 0
                ][:3]  # Top 3 contributing actors
            }

        # Predict overall risk trajectory
        current_risk = trends.get("aggregate_authoritarian_risk", 0)
        risk_trajectory = "stable"

        # Count high-probability escalations
        high_probability_count = sum(1 for data in indicator_escalation.values()
                                   if data["escalation_probability"] > 0.7)

        if high_probability_count >= 3:
            risk_trajectory = "rapidly increasing"
            predicted_risk = min(10, current_risk + 1.5)
        elif high_probability_count >= 1:
            risk_trajectory = "increasing"
            predicted_risk = min(10, current_risk + 0.8)
        elif current_risk > 7:
            # Already high risk tends to sustain
            risk_trajectory = "sustained high"
            predicted_risk = current_risk
        else:
            predicted_risk = current_risk

        # Identify potential trigger topics
        trigger_topics = []

        for topic, data in topics.get("recurring_topics", {}).items():
            auth_score = data.get("average_auth_score", 0)
            auth_trend = data.get("authoritarian_trend")

            if auth_score > 6 and auth_trend == "increasing":
                trigger_topics.append({
                    "topic": topic,
                    "auth_score": auth_score,
                    "trend": auth_trend,
                    "related_indicators": data.get("related_indicators", {})
                })

        # Sort trigger topics by score
        trigger_topics.sort(key=lambda x: x["auth_score"], reverse=True)

        return {
            "forecast_days": forecast_days,
            "current_risk": current_risk,
            "current_risk_level": self._determine_risk_level(current_risk),
            "predicted_risk": predicted_risk,
            "predicted_risk_level": self._determine_risk_level(predicted_risk),
            "risk_trajectory": risk_trajectory,
            "indicator_escalation": indicator_escalation,
            "potential_trigger_topics": trigger_topics[:5],  # Top 5 trigger topics
            "key_actors_to_monitor": actors.get("top_actors", [])[:3],  # Top 3 actors to monitor
            "timestamp": datetime.now().isoformat()
        }