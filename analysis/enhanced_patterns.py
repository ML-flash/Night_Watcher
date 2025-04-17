"""
Night_watcher Enhanced Pattern Recognition Module - Fix
Adds authoritarian actors analysis functionality to the enhanced pattern recognition system.
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
    Enhanced pattern detection and analysis for authoritarian trends, actors, and narratives.
    With improved analysis capabilities.
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

    def analyze_authoritarian_actors(self, lookback_days: int = 90) -> Dict[str, Any]:
        """
        Analyze actors associated with authoritarian patterns with enhanced capabilities.

        Args:
            lookback_days: Days to look back for analysis

        Returns:
            Dictionary containing actor analysis with risk assessment
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
        actor_timeline = defaultdict(list)

        # Process analyses
        for analysis in recent_analyses:
            metadata = analysis.get("metadata", {})
            timestamp = metadata.get("analysis_timestamp", "")
            if not timestamp:
                timestamp = datetime.now().isoformat()

            source = metadata.get("source", "")
            title = metadata.get("title", "")
            url = metadata.get("url", "")

            # Extract authoritarian indicators and entities from analysis
            analysis_text = analysis.get("text", "")
            auth_elements = self.extract_authoritarian_elements(analysis_text)

            # Track named entities and their associations with indicators
            for entity in auth_elements.get("entities", []):
                entity_name = entity["text"]
                entity_type = entity["type"]

                # Only focus on PERSON type entities for actor analysis
                if entity_type == "PERSON":
                    # Count mention
                    actor_mentions[entity_name] += 1

                    # Track source
                    actor_sources[entity_name].add(source)

                    # Add to timeline
                    actor_timeline[entity_name].append({
                        "date": timestamp,
                        "source": source,
                        "title": title,
                        "url": url
                    })

                    # Track indicator associations
                    for indicator, data in auth_elements.items():
                        if indicator in self.authoritarian_indicators and data.get("present", False):
                            # Increment indicator count for this actor
                            actor_indicators[entity_name][indicator] += data.get("frequency", 1)

                            # Add example if we have room
                            if len(actor_examples[entity_name]) < 5 and data.get("examples"):
                                actor_examples[entity_name].append({
                                    "indicator": indicator,
                                    "example": data["examples"][0],
                                    "source": source,
                                    "title": title,
                                    "date": timestamp,
                                    "url": url
                                })

            # Also check for direct mentions of tracked entities
            for entity_name in self.tracked_entities:
                if entity_name in analysis_text:
                    # Count mention if not already counted
                    if entity_name not in actor_mentions:
                        actor_mentions[entity_name] += 1
                        actor_sources[entity_name].add(source)
                        actor_timeline[entity_name].append({
                            "date": timestamp,
                            "source": source,
                            "title": title,
                            "url": url
                        })

        # Calculate authoritarian pattern scores with enhanced metrics
        actor_scores = {}

        for actor, indicators in actor_indicators.items():
            # Count total indicator instances
            total_instances = sum(indicators.values())

            # Count unique indicators
            unique_indicators = len(indicators)

            # Calculate authoritarian pattern score (0-10) with enhanced weighting
            # Emphasize both breadth (unique indicators) and depth (total instances)
            breadth_score = unique_indicators / len(self.authoritarian_indicators)
            depth_score = min(1.0, total_instances / 20)  # Cap at 20 instances

            # Weight recency - check if mentions are increasing over time
            recency_factor = 1.0
            timeline = actor_timeline.get(actor, [])
            if len(timeline) >= 3:
                # Sort by date
                sorted_timeline = sorted(timeline, key=lambda x: x.get("date", ""))

                # Split into two halves
                midpoint = len(sorted_timeline) // 2
                first_half = sorted_timeline[:midpoint]
                second_half = sorted_timeline[midpoint:]

                # If mentions increasing over time, add recency weight
                if len(second_half) > len(first_half):
                    recency_factor = 1.2

            # Calculate final score with weighted factors
            pattern_score = (breadth_score * 0.5 + depth_score * 0.3 + (actor_mentions[actor] / 10) * 0.2) * 10 * recency_factor
            pattern_score = min(10.0, pattern_score)  # Cap at 10

            # Get most frequent indicator
            most_frequent = max(indicators.items(), key=lambda x: x[1]) if indicators else (None, 0)

            # Get most recent mention
            most_recent = max(timeline, key=lambda x: x.get("date", "")) if timeline else {}

            actor_scores[actor] = {
                "authoritarian_pattern_score": pattern_score,
                "risk_level": self._determine_risk_level(pattern_score),
                "total_mentions": actor_mentions[actor],
                "indicator_counts": dict(indicators),
                "unique_indicators": unique_indicators,
                "most_frequent_indicator": {
                    "indicator": most_frequent[0],
                    "count": most_frequent[1]
                } if most_frequent[0] else {},
                "sources": list(actor_sources[actor]),
                "examples": actor_examples[actor],
                "most_recent_mention": most_recent,
                "trend": "increasing" if recency_factor > 1 else "stable"
            }

            # Add to knowledge graph if available
            if self.knowledge_graph:
                actor_id = self.knowledge_graph.find_or_create_actor(actor)

                # Add indicator relationships
                for indicator, count in indicators.items():
                    if count > 0:
                        indicator_id = self.knowledge_graph.find_or_create_entity(
                            indicator.replace("_", " ").title(),
                            self.knowledge_graph.INDICATOR
                        )

                        self.knowledge_graph.add_relationship(
                            actor_id, indicator_id, self.knowledge_graph.DEMONSTRATES,
                            weight=count,
                            attributes={
                                "date": datetime.now().isoformat(),
                                "risk_level": self._determine_risk_level(pattern_score)
                            }
                        )

        # Sort actors by score
        sorted_actors = sorted(
            [(actor, data) for actor, data in actor_scores.items()],
            key=lambda x: x[1]["authoritarian_pattern_score"],
            reverse=True
        )

        # Calculate aggregate risk level
        top_actor_scores = [data["authoritarian_pattern_score"] for _, data in sorted_actors[:3]] if sorted_actors else [0]
        aggregate_risk = sum(top_actor_scores) / len(top_actor_scores) if top_actor_scores else 0

        return {
            "lookback_days": lookback_days,
            "actor_patterns": {actor: data for actor, data in sorted_actors},
            "top_actors": [actor for actor, _ in sorted_actors[:5]],
            "aggregate_risk": aggregate_risk,
            "risk_level": self._determine_risk_level(aggregate_risk),
            "timestamp": datetime.now().isoformat()
        }

    def _determine_risk_level(self, risk_score: float) -> str:
        """
        Determine risk level category from score with enhanced granularity.

        Args:
            risk_score: Risk score

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