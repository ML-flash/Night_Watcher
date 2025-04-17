import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict

from memory.system import MemorySystem

logger = logging.getLogger(__name__)


class PatternRecognition:
    """
    Identifies patterns in media coverage and narrative strategies from the memory system data.
    """

    def __init__(self, memory_system: MemorySystem):
        """Initialize with memory system"""
        self.memory = memory_system
        self.logger = logging.getLogger("PatternRecognition")

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

                    for analysis in matching_analyses:
                        metadata = analysis.get("metadata", {})
                        timestamp = metadata.get("analysis_timestamp", "")
                        score = metadata.get("manipulation_score", 0)

                        if timestamp:
                            timeline.append({
                                "id": analysis.get("id", ""),
                                "timestamp": timestamp,
                                "score": score,
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
                        "examples": data.get("examples", []),
                        "timeline": timeline
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
            if parent_analyses:
                total_score = sum(p.get("metadata", {}).get("manipulation_score", 0) for p in parent_analyses)
                avg_parent_score = total_score / len(parent_analyses)

            demographic_themes[demo] = {
                "count": len(demo_narratives),
                "common_themes": common_themes,
                "avg_parent_manipulation_score": avg_parent_score,
                "sample_narratives": demo_narratives[:3]  # Include a few examples
            }

        return {
            "demographic_themes": demographic_themes,
            "total_narratives": len(narratives),
            "demographic_distribution": {demo: len(items) for demo, items in by_demographic.items()}
        }

    def _extract_common_phrases(self, texts: List[str], min_length: int = 3, min_count: int = 2) -> List[Dict[str, Any]]:
        """
        Extract common phrases from a list of texts.

        Args:
            texts: List of text strings to analyze
            min_length: Minimum number of words in a phrase
            min_count: Minimum number of occurrences to include a phrase

        Returns:
            List of common phrases with counts
        """
        # Extract n-grams from texts
        all_ngrams = Counter()

        for text in texts:
            # Normalize text
            text = text.lower()
            # Remove special characters
            text = re.sub(r'[^\w\s]', ' ', text)
            # Replace multiple spaces with single space
            text = re.sub(r'\s+', ' ', text).strip()

            words = text.split()

            # Extract n-grams
            for n in range(min_length, min(8, len(words))):
                for i in range(len(words) - n + 1):
                    ngram = ' '.join(words[i:i+n])
                    all_ngrams[ngram] += 1

        # Filter by minimum count
        common_phrases = [
            {"phrase": phrase, "count": count}
            for phrase, count in all_ngrams.items()
            if count >= min_count
        ]

        # Sort by count
        common_phrases.sort(key=lambda x: x["count"], reverse=True)

        # Return top phrases (limit to 20)
        return common_phrases[:20]

    def analyze_temporal_trends(self, lookback_days: int = 90, interval_days: int = 7) -> Dict[str, Any]:
        """
        Analyze trends over time, including manipulation scores and topic frequency.

        Args:
            lookback_days: Total days to look back
            interval_days: Interval for grouping data points

        Returns:
            Dictionary containing temporal trend analysis
        """
        # Get analyses within the lookback period
        recent_analyses = self.memory.get_recent_analyses(lookback_days)

        if not recent_analyses:
            return {"error": "No analyses found in the specified time period"}

        # Calculate time intervals
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        intervals = []
        current_date = start_date
        while current_date < end_date:
            interval_end = min(current_date + timedelta(days=interval_days), end_date)
            intervals.append((current_date, interval_end))
            current_date = interval_end

        # Group analyses by interval
        interval_data = []

        for interval_start, interval_end in intervals:
            interval_start_str = interval_start.isoformat()
            interval_end_str = interval_end.isoformat()

            # Filter analyses in this interval
            interval_analyses = [
                a for a in recent_analyses
                if (a.get("metadata", {}).get("analysis_timestamp", "") >= interval_start_str and
                    a.get("metadata", {}).get("analysis_timestamp", "") < interval_end_str)
            ]

            if interval_analyses:
                # Calculate metrics for this interval
                total_score = sum(a.get("metadata", {}).get("manipulation_score", 0) for a in interval_analyses)
                avg_score = total_score / len(interval_analyses)

                # Count sources and topics
                sources = {}
                all_topics = []

                for analysis in interval_analyses:
                    metadata = analysis.get("metadata", {})
                    source = metadata.get("source", "Unknown")
                    sources[source] = sources.get(source, 0) + 1

                    # Extract topics
                    analysis_text = analysis.get("text", "")
                    topics = self._extract_topics(analysis_text)
                    all_topics.extend(topics)

                # Count topic frequencies
                topic_counts = Counter(all_topics)
                top_topics = [{"topic": topic, "count": count}
                             for topic, count in topic_counts.most_common(5)]

                interval_data.append({
                    "interval_start": interval_start_str,
                    "interval_end": interval_end_str,
                    "article_count": len(interval_analyses),
                    "avg_manipulation_score": avg_score,
                    "sources": sources,
                    "top_topics": top_topics
                })

        return {
            "lookback_days": lookback_days,
            "interval_days": interval_days,
            "intervals": interval_data,
            "total_articles_analyzed": len(recent_analyses)
        }

    def analyze_source_correlation(self) -> Dict[str, Any]:
        """
        Analyze correlation between sources in terms of topic coverage and narrative framing.

        Returns:
            Dictionary containing source correlation analysis
        """
        # Get all article analyses
        analyses = []
        for item_id, item in self.memory.store.items.items():
            metadata = item.get("metadata", {})
            if metadata.get("type") == "article_analysis":
                analysis = item.copy()
                analysis["id"] = item_id
                analyses.append(analysis)

        if not analyses:
            return {"error": "No analyses found"}

        # Group by source
        by_source = {}
        for analysis in analyses:
            metadata = analysis.get("metadata", {})
            source = metadata.get("source", "Unknown")

            if source not in by_source:
                by_source[source] = []

            by_source[source].append(analysis)

        # Extract topics and frames by source
        source_profiles = {}
        for source, source_analyses in by_source.items():
            topics = []
            frames = []

            for analysis in source_analyses:
                analysis_text = analysis.get("text", "")
                topics.extend(self._extract_topics(analysis_text))
                frames.extend(self._extract_frames(analysis_text))

            # Count frequencies
            topic_counts = Counter(topics)
            frame_counts = Counter(frames)

            source_profiles[source] = {
                "count": len(source_analyses),
                "top_topics": [{"topic": topic, "count": count}
                             for topic, count in topic_counts.most_common(10)],
                "top_frames": [{"frame": frame, "count": count}
                              for frame, count in frame_counts.most_common(10)],
                "avg_manipulation_score": sum(a.get("metadata", {}).get("manipulation_score", 0)
                                          for a in source_analyses) / len(source_analyses) if source_analyses else 0
            }

        # Calculate correlation between sources
        correlations = []
        source_names = list(source_profiles.keys())

        for i, source1 in enumerate(source_names):
            for j in range(i+1, len(source_names)):
                source2 = source_names[j]

                # Calculate topic overlap
                source1_topics = {item["topic"] for item in source_profiles[source1]["top_topics"]}
                source2_topics = {item["topic"] for item in source_profiles[source2]["top_topics"]}
                topic_overlap = len(source1_topics.intersection(source2_topics)) / max(1, min(len(source1_topics), len(source2_topics)))

                # Calculate frame overlap
                source1_frames = {item["frame"] for item in source_profiles[source1]["top_frames"]}
                source2_frames = {item["frame"] for item in source_profiles[source2]["top_frames"]}
                frame_overlap = len(source1_frames.intersection(source2_frames)) / max(1, min(len(source1_frames), len(source2_frames)))

                # Calculate manipulation score difference
                score_diff = abs(source_profiles[source1]["avg_manipulation_score"] -
                                 source_profiles[source2]["avg_manipulation_score"])

                correlations.append({
                    "source1": source1,
                    "source2": source2,
                    "topic_overlap": topic_overlap,
                    "frame_overlap": frame_overlap,
                    "manipulation_score_diff": score_diff
                })

        # Sort by overall correlation (topic + frame overlap)
        correlations.sort(key=lambda x: x["topic_overlap"] + x["frame_overlap"], reverse=True)

        return {
            "source_profiles": source_profiles,
            "correlations": correlations
        }

    def _extract_frames(self, analysis: str) -> List[str]:
        """Extract frames from analysis text"""
        frames = []

        try:
            if "FRAMING" in analysis:
                frames_section = analysis.split("FRAMING:")[1].split("\n\n")[0]

                # Simple extraction - split by sentences and clean up
                for item in re.split(r'\.', frames_section):
                    frame = item.strip()
                    if frame and len(frame) > 5 and not frame.startswith("EMOTIONAL"):
                        frames.append(frame)
        except Exception as e:
            self.logger.error(f"Error extracting frames: {str(e)}")

        return frames

    def analyze_manipulation_techniques(self) -> Dict[str, Any]:
        """
        Analyze the prevalence of different manipulation techniques across sources.

        Returns:
            Dictionary containing manipulation technique analysis
        """
        # Get all article analyses
        analyses = []
        for item_id, item in self.memory.store.items.items():
            metadata = item.get("metadata", {})
            if metadata.get("type") == "article_analysis":
                analysis = item.copy()
                analysis["id"] = item_id
                analyses.append(analysis)

        if not analyses:
            return {"error": "No analyses found"}

        # Extract manipulation techniques from analyses
        technique_count = Counter()
        technique_by_source = {}
        technique_by_bias = {
            "left": Counter(),
            "center-left": Counter(),
            "center": Counter(),
            "center-right": Counter(),
            "right": Counter(),
            "unknown": Counter()
        }

        high_manipulation_examples = {}  # Track examples of each technique

        for analysis in analyses:
            metadata = analysis.get("metadata", {})
            source = metadata.get("source", "Unknown")
            bias = metadata.get("bias_label", "unknown")
            score = metadata.get("manipulation_score", 0)
            analysis_text = analysis.get("text", "")

            # Extract techniques
            techniques = self._extract_manipulation_techniques(analysis_text)

            # Count techniques
            for technique in techniques:
                technique_count[technique] += 1

                # Count by source
                if source not in technique_by_source:
                    technique_by_source[source] = Counter()
                technique_by_source[source][technique] += 1

                # Count by bias
                if bias not in technique_by_bias:
                    bias = "unknown"
                technique_by_bias[bias][technique] += 1

                # Track high manipulation examples
                if score >= 7:  # High manipulation threshold
                    if technique not in high_manipulation_examples:
                        high_manipulation_examples[technique] = []

                    if len(high_manipulation_examples[technique]) < 3:  # Keep up to 3 examples
                        high_manipulation_examples[technique].append({
                            "id": analysis.get("id", ""),
                            "title": metadata.get("title", ""),
                            "source": source,
                            "score": score
                        })

        # Prepare results
        technique_results = []
        for technique, count in technique_count.most_common():
            technique_results.append({
                "technique": technique,
                "count": count,
                "percentage": count / len(analyses) * 100 if analyses else 0,
                "examples": high_manipulation_examples.get(technique, [])
            })

        # Format source-specific results
        source_results = {}
        for source, counts in technique_by_source.items():
            total = sum(counts.values())
            techniques = [
                {"technique": technique, "count": count, "percentage": count / total * 100 if total else 0}
                for technique, count in counts.most_common(5)
            ]
            source_results[source] = {
                "total_articles": len([a for a in analyses if a.get("metadata", {}).get("source") == source]),
                "top_techniques": techniques
            }

        # Format bias-specific results
        bias_results = {}
        for bias, counts in technique_by_bias.items():
            if sum(counts.values()) > 0:  # Only include biases with data
                total = sum(counts.values())
                techniques = [
                    {"technique": technique, "count": count, "percentage": count / total * 100 if total else 0}
                    for technique, count in counts.most_common(5)
                ]
                bias_results[bias] = {
                    "total_articles": len([a for a in analyses if a.get("metadata", {}).get("bias_label") == bias]),
                    "top_techniques": techniques
                }

        return {
            "total_articles": len(analyses),
            "techniques": technique_results,
            "by_source": source_results,
            "by_bias": bias_results
        }

    def _extract_manipulation_techniques(self, analysis: str) -> List[str]:
        """Extract manipulation techniques from analysis text"""
        techniques = []
        technique_list = [
            "Appeal to fear or outrage",
            "False equivalence",
            "Cherry-picking of facts",
            "Ad hominem attacks",
            "Straw man arguments",
            "Bandwagon appeal",
            "Black-and-white fallacy"
        ]

        try:
            if "MANIPULATION TECHNIQUES" in analysis:
                techniques_section = analysis.split("MANIPULATION TECHNIQUES:")[1].split("MANIPULATION SCORE:")[0]

                for technique in technique_list:
                    if technique.lower() in techniques_section.lower():
                        techniques.append(technique)
        except Exception as e:
            self.logger.error(f"Error extracting manipulation techniques: {str(e)}")

        return techniques
        
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

        # Track actor mentions and indicators
        from collections import defaultdict
        actor_mentions = defaultdict(int)
        actor_indicators = defaultdict(lambda: defaultdict(int))
        actor_examples = defaultdict(list)
        actor_sources = defaultdict(set)

        # Define authoritarian indicators
        authoritarian_indicators = [
            "institutional_undermining",
            "democratic_norm_violations", 
            "media_delegitimization",
            "opposition_targeting",
            "power_concentration",
            "accountability_evasion"
        ]

        # Process analyses
        for analysis in recent_analyses:
            metadata = analysis.get("metadata", {})
            timestamp = metadata.get("analysis_timestamp", "")
            source = metadata.get("source", "")
            title = metadata.get("title", "")
            
            # Extract text and look for actor mentions
            analysis_text = analysis.get("text", "")
            
            # Simple pattern matching for key political actors
            key_actors = ["Donald Trump", "Joe Biden", "Congress", "Supreme Court", "White House"]
            
            for actor in key_actors:
                if actor in analysis_text:
                    # Count mention
                    actor_mentions[actor] += 1
                    
                    # Track source
                    actor_sources[actor].add(source)
                    
                    # Simple check for indicator associations
                    for indicator in authoritarian_indicators:
                        indicator_term = indicator.replace("_", " ")
                        if indicator_term in analysis_text.lower():
                            actor_indicators[actor][indicator] += 1
                            
                            # Add example
                            if len(actor_examples[actor]) < 3:
                                actor_examples[actor].append({
                                    "indicator": indicator,
                                    "source": source,
                                    "title": title
                                })

        # Calculate basic scores
        actor_patterns = {}
        for actor, indicators in actor_indicators.items():
            actor_patterns[actor] = {
                "authoritarian_pattern_score": min(10, sum(indicators.values())),
                "total_mentions": actor_mentions[actor],
                "indicator_counts": dict(indicators),
                "sources": list(actor_sources[actor]),
                "examples": actor_examples[actor]
            }

        # Sort actors by score
        top_actors = sorted(
            actor_patterns.keys(),
            key=lambda x: actor_patterns[x]["authoritarian_pattern_score"],
            reverse=True
        )[:5]

        return {
            "lookback_days": lookback_days,
            "actor_patterns": actor_patterns,
            "top_actors": top_actors,
            "timestamp": datetime.now().isoformat()
        }