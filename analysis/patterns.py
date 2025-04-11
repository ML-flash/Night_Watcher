"""
Night_watcher Enhanced Pattern Recognition Module
Analyzes stored data to identify patterns in authoritarian governance trends and media coverage.
"""

import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter

from memory.system import MemorySystem


class PatternRecognition:
    """
    Identifies patterns in authoritarian governance indicators, media coverage, and narrative strategies.
    """

    def __init__(self, memory_system: MemorySystem):
        """Initialize with memory system"""
        self.memory = memory_system
        self.logger = logging.getLogger("PatternRecognition")
        
        # Authoritarian indicators for tracking
        self.authoritarian_indicators = [
            "institutional_undermining",
            "democratic_norm_violations",
            "media_delegitimization",
            "opposition_targeting",
            "power_concentration",
            "accountability_evasion",
            "threat_exaggeration",
            "authoritarian_rhetoric",
            "rule_of_law_undermining"
        ]

    def analyze_source_bias_patterns(self, days: int = 30) -> Dict[str, Any]:
        """
        Analyze patterns in source bias and manipulation scores.

        Args:
            days: Number of days to look back for analysis

        Returns:
            Dictionary containing bias analysis results
        """
        # Get recent analyses
        recent_analyses = self.memory.get_recent_analyses(days)

        if not recent_analyses:
            return {"error": "No recent analyses found"}

        # Group by source and bias
        sources = {}
        biases = {
            "left": {"count": 0, "total_score": 0, "sources": set()},
            "center-left": {"count": 0, "total_score": 0, "sources": set()},
            "center": {"count": 0, "total_score": 0, "sources": set()},
            "center-right": {"count": 0, "total_score": 0, "sources": set()},
            "right": {"count": 0, "total_score": 0, "sources": set()},
            "unknown": {"count": 0, "total_score": 0, "sources": set()}
        }

        # Track highest scores by source and bias
        highest_scores = {
            "by_source": {},
            "by_bias": {k: {"score": 0, "article": None} for k in biases.keys()}
        }

        for item in recent_analyses:
            metadata = item.get("metadata", {})
            source = metadata.get("source", "Unknown")
            bias = metadata.get("bias_label", "unknown")
            score = metadata.get("manipulation_score", 0)

            # Track by source
            if source not in sources:
                sources[source] = {
                    "count": 0,
                    "total_score": 0,
                    "bias": bias
                }

            sources[source]["count"] += 1
            sources[source]["total_score"] += score

            # Track by bias category
            if bias not in biases:
                bias = "unknown"

            biases[bias]["count"] += 1
            biases[bias]["total_score"] += score
            biases[bias]["sources"].add(source)

            # Track highest scores
            if source not in highest_scores["by_source"] or score > highest_scores["by_source"][source]["score"]:
                highest_scores["by_source"][source] = {
                    "score": score,
                    "article": {
                        "id": item.get("id"),
                        "title": metadata.get("title", "Unknown"),
                        "url": metadata.get("url", "")
                    }
                }

            if score > highest_scores["by_bias"][bias]["score"]:
                highest_scores["by_bias"][bias] = {
                    "score": score,
                    "article": {
                        "id": item.get("id"),
                        "title": metadata.get("title", "Unknown"),
                        "source": source,
                        "url": metadata.get("url", "")
                    }
                }

        # Calculate averages
        for source, data in sources.items():
            data["average_score"] = data["total_score"] / data["count"] if data["count"] > 0 else 0

        for bias, data in biases.items():
            data["average_score"] = data["total_score"] / data["count"] if data["count"] > 0 else 0
            data["sources"] = list(data["sources"])  # Convert set to list for JSON serialization

        # Sort sources by average manipulation score
        sorted_sources = sorted(
            [(source, data) for source, data in sources.items()],
            key=lambda x: x[1]["average_score"],
            reverse=True
        )

        return {
            "period_days": days,
            "total_articles": len(recent_analyses),
            "source_analysis": {source: data for source, data in sorted_sources},
            "bias_analysis": biases,
            "highest_scores": highest_scores
        }

    def analyze_authoritarian_trend_patterns(self, lookback_days: int = 90) -> Dict[str, Any]:
        """
        Analyze trends in authoritarian governance indicators over time.
        
        Args:
            lookback_days: Days to look back for analysis
            
        Returns:
            Dictionary containing trend analysis of authoritarian indicators
        """
        # Get analyses within the lookback period
        recent_analyses = self.memory.get_recent_analyses(lookback_days)
        
        if not recent_analyses:
            return {"error": "No recent analyses found"}
            
        # Track indicators over time
        indicators = {indicator: [] for indicator in self.authoritarian_indicators}
        indicators["authoritarian_score"] = []
        
        # Process analyses to extract indicators
        for analysis in recent_analyses:
            metadata = analysis.get("metadata", {})
            timestamp = metadata.get("analysis_timestamp", "")
            
            # Extract authoritarian indicators and score from analysis
            analysis_text = analysis.get("text", "")
            auth_elements = self._extract_authoritarian_elements(analysis_text)
            
            # Track authoritarian score
            if "authoritarian_score" in auth_elements:
                indicators["authoritarian_score"].append({
                    "date": timestamp,
                    "source": metadata.get("source", ""),
                    "title": metadata.get("title", ""),
                    "score": auth_elements["authoritarian_score"]
                })
            
            # Track indicators
            for indicator in self.authoritarian_indicators:
                if indicator in auth_elements and auth_elements[indicator]["present"]:
                    indicators[indicator].append({
                        "date": timestamp,
                        "source": metadata.get("source", ""),
                        "title": metadata.get("title", ""),
                        "examples": auth_elements[indicator]["examples"]
                    })
        
        # Calculate trend strength for each indicator
        trend_analysis = {}
        for indicator, occurrences in indicators.items():
            trend_analysis[indicator] = {
                "count": len(occurrences),
                "examples": occurrences[:3],  # Top 3 examples
                "trend_strength": self._calculate_trend_strength(occurrences, lookback_days),
                "timeline": self._create_timeline(occurrences, lookback_days)
            }
        
        # Calculate aggregate authoritarian risk
        aggregate_risk = self._calculate_aggregate_risk(trend_analysis)
        
        return {
            "lookback_days": lookback_days,
            "trend_analysis": trend_analysis,
            "aggregate_authoritarian_risk": aggregate_risk,
            "risk_level": self._determine_risk_level(aggregate_risk)
        }
    
    def _extract_authoritarian_elements(self, analysis_text: str) -> Dict[str, Any]:
        """
        Extract authoritarian elements from analysis text.
        
        Args:
            analysis_text: Analysis text to extract from
            
        Returns:
            Dictionary of authoritarian elements
        """
        elements = {}
        
        # Extract authoritarian score
        score_match = re.search(r'AUTHORITARIAN SCORE:?\s*(\d+)[^\d]', analysis_text)
        if score_match:
            score = int(score_match.group(1))
            elements["authoritarian_score"] = score
        
        # Extract indicators
        for indicator in self.authoritarian_indicators:
            indicator_label = indicator.upper().replace("_", " ")
            if indicator_label in analysis_text:
                # Get the section for this indicator
                pattern = f"{indicator_label}:(.+?)(?=\n\n\d+\.|\n\n[A-Z]|\Z)"
                section_match = re.search(pattern, analysis_text, re.DOTALL)
                
                if section_match:
                    section_text = section_match.group(1).strip()
                    
                    # Determine if indicator is present
                    negative_patterns = ["not present", "no evidence", "none found", "not identified"]
                    is_present = True
                    
                    for pattern in negative_patterns:
                        if pattern in section_text.lower():
                            is_present = False
                            break
                    
                    # Extract examples
                    examples = []
                    if is_present:
                        # Split by bullet points, periods, or other separators
                        for line in re.split(r'(?:•|\*|\-|\d+\.|\n)', section_text):
                            line = line.strip()
                            if line and len(line) > 10:
                                examples.append(line)
                    
                    elements[indicator] = {
                        "present": is_present,
                        "examples": examples
                    }
                else:
                    elements[indicator] = {
                        "present": False,
                        "examples": []
                    }
        
        return elements
    
    def _calculate_trend_strength(self, occurrences: List[Dict[str, Any]], lookback_days: int) -> float:
        """
        Calculate the strength of a trend based on occurrences.
        
        Args:
            occurrences: List of occurrences
            lookback_days: Total days to look back
            
        Returns:
            Trend strength as a float from 0-1
        """
        if not occurrences:
            return 0.0
            
        # Calculate recent vs older ratio
        midpoint = datetime.now() - timedelta(days=lookback_days/2)
        midpoint_str = midpoint.isoformat()
        
        recent_count = len([o for o in occurrences if o.get("date", "") >= midpoint_str])
        older_count = len(occurrences) - recent_count
        
        # Calculate trend direction (positive means increasing)
        trend_direction = 0
        if older_count > 0:
            trend_direction = (recent_count / max(1, older_count)) - 1
        elif recent_count > 0:
            trend_direction = 1  # All occurrences are recent
        
        # Calculate frequency relative to total period
        frequency = len(occurrences) / lookback_days
        
        # Combine into trend strength (0-1 scale)
        strength = min(1.0, (frequency * 10) * (0.5 + max(0, trend_direction)))
        
        return strength
    
    def _create_timeline(self, occurrences: List[Dict[str, Any]], lookback_days: int) -> List[Dict[str, Any]]:
        """
        Create a timeline of occurrences for visualization.
        
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
            
            # Count occurrences in this week
            week_occurrences = [
                o for o in sorted_occurrences 
                if o.get("date", "") >= week_start_str and o.get("date", "") < week_end_str
            ]
            
            if week_occurrences:
                timeline.append({
                    "start_date": week_start_str,
                    "end_date": week_end_str,
                    "count": len(week_occurrences),
                    "items": week_occurrences
                })
            else:
                timeline.append({
                    "start_date": week_start_str,
                    "end_date": week_end_str,
                    "count": 0,
                    "items": []
                })
            
            current_date = week_end
        
        return timeline
    
    def _calculate_aggregate_risk(self, trend_analysis: Dict[str, Any]) -> float:
        """
        Calculate aggregate authoritarian risk score.
        
        Args:
            trend_analysis: Trend analysis dictionary
            
        Returns:
            Aggregate risk score from 0-10
        """
        # Weights for different indicators
        weights = {
            "institutional_undermining": 1.5,
            "democratic_norm_violations": 1.5,
            "media_delegitimization": 1.2,
            "opposition_targeting": 1.2,
            "power_concentration": 1.5,
            "accountability_evasion": 1.0,
            "threat_exaggeration": 0.8,
            "authoritarian_rhetoric": 1.0,
            "rule_of_law_undermining": 1.3,
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
                    # For other indicators, use trend strength
                    weighted_score += indicator_data.get("trend_strength", 0) * weight
        
        # Normalize to 0-10 scale
        aggregate_risk = (weighted_score / total_weight) * 10
        
        return aggregate_risk
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """
        Determine risk level category from score.
        
        Args:
            risk_score: Aggregate risk score
            
        Returns:
            Risk level category
        """
        if risk_score < 2:
            return "Low"
        elif risk_score < 4:
            return "Moderate"
        elif risk_score < 6:
            return "Substantial"
        elif risk_score < 8:
            return "High"
        else:
            return "Severe"

    def identify_recurring_topics(self, min_count: int = 3) -> Dict[str, Any]:
        """
        Identify recurring topics and track their manipulation scores over time.

        Args:
            min_count: Minimum number of occurrences to consider a topic recurring

        Returns:
            Dictionary containing recurring topics analysis
        """
        # Get topics summary
        topics_summary = self._extract_topic_summary(limit=100)  # Get a larger set to filter

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

                    for analysis in matching_analyses:
                        metadata = analysis.get("metadata", {})
                        timestamp = metadata.get("analysis_timestamp", "")
                        score = metadata.get("manipulation_score", 0)
                        
                        # Try to extract authoritarian score
                        auth_score = self._extract_authoritarian_score(analysis.get("text", ""))
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

                    recurring_topics[topic] = {
                        "count": data["count"],
                        "average_score": total_score / len(timeline) if timeline else 0,
                        "average_auth_score": total_auth_score / auth_score_count if auth_score_count else 0,
                        "examples": data.get("examples", []),
                        "timeline": timeline,
                        "related_indicators": self._find_related_indicators(topic, matching_analyses)
                    }

        # Sort by count
        sorted_topics = sorted(
            [(topic, data) for topic, data in recurring_topics.items()],
            key=lambda x: x[1]["count"],
            reverse=True
        )

        return {
            "recurring_topics": {topic: data for topic, data in sorted_topics},
            "total_topics_analyzed": len(topics_summary.get("top_topics", {})),
            "min_count_threshold": min_count
        }
    
    def _extract_authoritarian_score(self, text: str) -> int:
        """Extract authoritarian score from text"""
        score_match = re.search(r'AUTHORITARIAN SCORE:?\s*(\d+)[^\d]', text)
        if score_match:
            return int(score_match.group(1))
        return 0
    
    def _find_related_indicators(self, topic: str, analyses: List[Dict[str, Any]]) -> Dict[str, int]:
        """Find authoritarian indicators related to a topic"""
        indicator_counts = {indicator: 0 for indicator in self.authoritarian_indicators}
        
        for analysis in analyses:
            text = analysis.get("text", "")
            auth_elements = self._extract_authoritarian_elements(text)
            
            for indicator in self.authoritarian_indicators:
                if indicator in auth_elements and auth_elements[indicator]["present"]:
                    indicator_counts[indicator] += 1
        
        # Return only indicators that appear at least once
        return {k: v for k, v in indicator_counts.items() if v > 0}

    def _extract_topic_summary(self, limit: int = 10) -> Dict[str, Any]:
        """
        Generate a summary of top topics in the memory system.

        Args:
            limit: Maximum number of topics to include

        Returns:
            Summary of topics with counts and examples
        """
        topics = {}
        bias_distribution = {}

        # Scan all items
        for item_id, item in self.memory.store.items.items():
            metadata = item.get("metadata", {})

            if metadata.get("type") == "article_analysis":
                # Extract topics from analysis text
                analysis = item.get("text", "")
                extracted_topics = self._extract_topics(analysis)

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
                            }]
                        }

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
            "total_analyses": sum(1 for meta in self.memory.store.items.values()
                               if meta.get("metadata", {}).get("type") == "article_analysis")
        }

    def _extract_topics(self, analysis: str) -> List[str]:
        """Extract topics from analysis text"""
        topics = []

        try:
            if "MAIN TOPICS" in analysis:
                topics_section = analysis.split("MAIN TOPICS:")[1].split("\n\n")[0]

                # Simple extraction - split by commas, newlines, and clean up
                for item in re.split(r'[,\n]', topics_section):
                    topic = item.strip()
                    if topic and len(topic) > 3 and not topic.startswith("FRAMING"):
                        # Clean up bullet points and numbering
                        topic = re.sub(r'^[\d\.\-\*]+\s*', '', topic)
                        if topic:
                            topics.append(topic)
        except Exception as e:
            self.logger.error(f"Error extracting topics: {str(e)}")

        return topics

    def analyze_narrative_effectiveness(self, demographic: str = None) -> Dict[str, Any]:
        """
        Analyze counter-narrative effectiveness based on patterns and similarity.

        Args:
            demographic: Optional demographic to filter by

        Returns:
            Dictionary containing narrative effectiveness analysis
        """
        # Get all narrative content
        narratives = []

        # This is inefficient for large stores - would be better with a proper database
        for item_id, item in self.memory.store.items.items():
            metadata = item.get("metadata", {})

            if metadata.get("type") == "counter_narrative":
                if demographic is None or metadata.get("demographic") == demographic:
                    # Copy and add ID
                    narrative = item.copy()
                    narrative["id"] = item_id
                    narratives.append(narrative)

        if not narratives:
            return {"error": "No counter-narratives found"}

        # Group by demographic
        by_demographic = {}

        for narrative in narratives:
            demo = narrative.get("metadata", {}).get("demographic", "unknown")

            if demo not in by_demographic:
                by_demographic[demo] = []

            by_demographic[demo].append(narrative)

        # Analyze common themes in each demographic
        demographic_themes = {}

        for demo, demo_narratives in by_demographic.items():
            # Find common phrases and themes
            common_themes = self._extract_common_phrases(
                [n.get("text", "") for n in demo_narratives],
                min_count=2
            )

            # Get related article analyses
            parent_ids = []
            for narrative in demo_narratives:
                parent_id = narrative.get("metadata", {}).get("parent_id", "")
                if parent_id:
                    parent_ids.append(parent_id)

            parent_analyses = []
            for parent_id in parent_ids:
                item = self.memory.store.get_item(parent_id)
                if item:
                    parent = item.copy()
                    parent["id"] = parent_id
                    parent_analyses.append(parent)

            # Calculate average manipulation score of parent articles
            avg_parent_score = 0
            avg_auth_score = 0
            auth_score_count = 0
            
            if parent_analyses:
                total_score = sum(p.get("metadata", {}).get("manipulation_score", 0) for p in parent_analyses)
                avg_parent_score = total_score / len(parent_analyses)
                
                # Calculate average authoritarian score if available
                for p in parent_analyses:
                    auth_score = self._extract_authoritarian_score(p.get("text", ""))
                    if auth_score > 0:
                        avg_auth_score += auth_score
                        auth_score_count += 1
                
                if auth_score_count > 0:
                    avg_auth_score = avg_auth_score / auth_score_count

            demographic_themes[demo] = {
                "count": len(demo_narratives),
                "common_themes": common_themes,
                "avg_parent_manipulation_score": avg_parent_score,
                "avg_parent_authoritarian_score": avg_auth_score if auth_score_count > 0 else 0,
                "sample_narratives": demo_narratives[:3]  # Include a few examples
            }

        return {
            "demographic_themes": demographic_themes,
            "total_narratives": len(narratives),
            "demographic_distribution": {demo: len(items) for demo, items in by_demographic.items()}
        }