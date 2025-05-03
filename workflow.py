"""
Night_watcher Workflow Orchestrator
Manages the Night_watcher workflow with focus on intelligence gathering and analysis.
"""

import os
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from providers import LLMProvider
from collector import ContentCollector
from analyzer import ContentAnalyzer
from memory import MemorySystem
from utils import save_to_file, create_slug, extract_topics

# Configure logging
logger = logging.getLogger(__name__)

# ==========================================
# Night_watcher Workflow
# ==========================================

class NightWatcherWorkflow:
    """Manages the Night_watcher workflow focused on intelligence gathering and analysis"""

    def __init__(self, llm_provider: LLMProvider, memory_system: Optional[MemorySystem] = None,
                 output_dir: str = "data"):
        """Initialize workflow with agents and output directory"""
        self.llm_provider = llm_provider
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Initialize memory system if not provided
        self.memory = memory_system or MemorySystem()

        # Initialize agents
        self.collector = ContentCollector()
        self.analyzer = ContentAnalyzer(llm_provider)

        # Set up logging
        self.logger = logging.getLogger("NightWatcherWorkflow")

        # Create output directories
        self._ensure_data_dirs()

    def _ensure_data_dirs(self):
        """Ensure all data directories exist"""
        directories = [
            f"{self.output_dir}/collected",
            f"{self.output_dir}/analyzed",
            f"{self.output_dir}/analysis"
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

        self.analysis_dir = f"{self.output_dir}/analysis/{self.timestamp}"
        os.makedirs(self.analysis_dir, exist_ok=True)

    def run(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run the Night_watcher intelligence gathering workflow"""
        if config is None:
            config = {}

        article_limit = config.get("article_limit", 5)
        sources = config.get("sources", None)
        pattern_analysis_days = config.get("pattern_analysis_days", 30)
        start_date = config.get("start_date", None)
        end_date = config.get("end_date", None)
        llm_provider_available = config.get("llm_provider_available", True)

        self.logger.info(f"Starting Night_watcher intelligence gathering workflow with timestamp {self.timestamp}")

        # 1. Collect articles with date range if specified
        self.logger.info("Collecting articles with focus on government/political content...")
        collection_params = {"limit": article_limit}
        
        if sources:
            collection_params["sources"] = sources
            
        if start_date:
            collection_params["start_date"] = start_date
            
        if end_date:
            collection_params["end_date"] = end_date

        collection_result = self.collector.process(collection_params)
        articles = collection_result.get("articles", [])
        self.logger.info(f"Collected {len(articles)} articles")

        # Save collected articles
        save_to_file(articles, f"{self.output_dir}/collected/articles_{self.timestamp}.json")
        
        # If LLM provider is not available, return early with collection results
        if not llm_provider_available:
            self.logger.info("LLM provider not available. Skipping analysis steps.")
            return {
                "timestamp": self.timestamp,
                "output_dir": self.output_dir,
                "articles_collected": len(articles),
                "articles_analyzed": 0,
                "pattern_analyses_generated": 0,
                "articles": articles
            }

        # 2. Analyze content for both divisive elements and authoritarian patterns
        self.logger.info("Analyzing articles for content and authoritarian patterns...")
        analysis_result = self.analyzer.process({"articles": articles})
        analyses = analysis_result.get("analyses", [])
        auth_analyses = analysis_result.get("authoritarian_analyses", [])

        # Save individual analyses and store in memory system
        for i, analysis in enumerate(analyses):
            if "article" in analysis:
                article_slug = create_slug(analysis['article']['title'])
                save_to_file(analysis, f"{self.output_dir}/analyzed/analysis_{article_slug}_{self.timestamp}.json")

                # Store in memory system
                analysis_id = self.memory.store_article_analysis(analysis)

                # If we have corresponding authoritarian analysis, save it
                if i < len(auth_analyses):
                    auth_analysis = auth_analyses[i]
                    save_to_file(auth_analysis,
                                 f"{self.output_dir}/analyzed/auth_analysis_{article_slug}_{self.timestamp}.json")

        # 3. Run pattern analysis
        self.logger.info(f"Running pattern analysis over the last {pattern_analysis_days} days...")
        
        # Get recent analyses for pattern recognition
        recent_analyses = self.memory.get_recent_analyses(pattern_analysis_days)
        
        if recent_analyses:
            # Analyze authoritarian patterns
            auth_trends = self._analyze_authoritarian_trends(recent_analyses)
            save_to_file(
                auth_trends,
                f"{self.analysis_dir}/authoritarian_trends_{self.timestamp}.json"
            )
            
            # Analyze topics
            topic_analysis = self._analyze_recurring_topics(recent_analyses)
            save_to_file(
                topic_analysis,
                f"{self.analysis_dir}/topic_analysis_{self.timestamp}.json"
            )
            
            # Analyze actors
            actor_analysis = self._analyze_authoritarian_actors(recent_analyses)
            save_to_file(
                actor_analysis,
                f"{self.analysis_dir}/actor_analysis_{self.timestamp}.json"
            )
            
            patterns_generated = 3
        else:
            self.logger.warning("No recent analyses found for pattern recognition")
            patterns_generated = 0

        # Return results
        return {
            "timestamp": self.timestamp,
            "output_dir": self.output_dir,
            "articles_collected": len(articles),
            "articles_analyzed": len(analyses),
            "pattern_analyses_generated": patterns_generated,
            "analyses": analyses,
            "auth_analyses": auth_analyses
        }
        
    def _analyze_authoritarian_trends(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends in authoritarian patterns"""
        # Define authoritarian indicators to track
        authoritarian_indicators = {
            "institutional_undermining": {
                "count": 0,
                "examples": [],
                "trend_strength": 0
            },
            "democratic_norm_violations": {
                "count": 0,
                "examples": [],
                "trend_strength": 0
            },
            "media_delegitimization": {
                "count": 0,
                "examples": [],
                "trend_strength": 0
            },
            "opposition_targeting": {
                "count": 0,
                "examples": [],
                "trend_strength": 0
            },
            "power_concentration": {
                "count": 0,
                "examples": [],
                "trend_strength": 0
            },
            "accountability_evasion": {
                "count": 0,
                "examples": [],
                "trend_strength": 0
            },
            "threat_exaggeration": {
                "count": 0,
                "examples": [],
                "trend_strength": 0
            },
            "authoritarian_rhetoric": {
                "count": 0,
                "examples": [],
                "trend_strength": 0
            },
            "rule_of_law_undermining": {
                "count": 0,
                "examples": [],
                "trend_strength": 0
            }
        }
        
        # Extract authoritarian scores and indicators from analyses
        authoritarian_scores = []
        
        for analysis in analyses:
            # Extract structured elements if available
            if "structured_elements" in analysis:
                elements = analysis.get("structured_elements", {})
                
                # Get authoritarian score
                auth_score = elements.get("authoritarian_score", 0)
                if auth_score > 0:
                    authoritarian_scores.append(auth_score)
                
                # Check for authoritarian indicators
                for indicator in authoritarian_indicators.keys():
                    if indicator in elements and elements[indicator].get("present", False):
                        authoritarian_indicators[indicator]["count"] += 1
                        
                        # Add example if available
                        examples = elements[indicator].get("examples", [])
                        if examples and len(authoritarian_indicators[indicator]["examples"]) < 3:
                            metadata = analysis.get("metadata", {})
                            example = {
                                "text": examples[0],
                                "source": metadata.get("source", "Unknown"),
                                "title": metadata.get("title", "")
                            }
                            authoritarian_indicators[indicator]["examples"].append(example)
        
        # Calculate trend strengths (0.0-1.0)
        total_analyses = len(analyses)
        if total_analyses > 0:
            for indicator, data in authoritarian_indicators.items():
                data["trend_strength"] = data["count"] / total_analyses
        
        # Calculate aggregate risk score (0-10)
        indicators_count = sum(1 for k, v in authoritarian_indicators.items() if v["count"] > 0)
        total_indicators = len(authoritarian_indicators)
        
        # Factors for overall calculation
        breadth_factor = indicators_count / total_indicators if total_indicators > 0 else 0
        
        trend_strengths = [v["trend_strength"] for v in authoritarian_indicators.values()]
        depth_factor = sum(trend_strengths) / len(trend_strengths) if trend_strengths else 0
        
        severity_factor = 0
        if authoritarian_scores:
            avg_score = sum(authoritarian_scores) / len(authoritarian_scores)
            severity_factor = avg_score / 10.0
        
        # Calculate weighted aggregate risk
        aggregate_risk = (breadth_factor * 0.3 + depth_factor * 0.3 + severity_factor * 0.4) * 10
        
        # Determine risk level
        risk_level = "Low"
        if aggregate_risk >= 7:
            risk_level = "Severe"
        elif aggregate_risk >= 5:
            risk_level = "High"
        elif aggregate_risk >= 3:
            risk_level = "Moderate"
            
        return {
            "lookback_days": len(analyses),
            "total_analyses": len(analyses),
            "trend_analysis": authoritarian_indicators,
            "aggregate_authoritarian_risk": aggregate_risk,
            "risk_level": risk_level,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _analyze_recurring_topics(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify recurring topics and track their manipulation scores over time"""
        # Extract all topics from analyses
        all_topics = {}
        
        for analysis in analyses:
            analysis_text = analysis.get("text", "")
            metadata = analysis.get("metadata", {})
            
            # Extract topics from analysis text
            topics = extract_topics(analysis_text)
            
            # Get manipulation score
            manipulation_score = metadata.get("manipulation_score", 0)
            
            # Track topics
            for topic in topics:
                if topic not in all_topics:
                    all_topics[topic] = {
                        "count": 0,
                        "total_score": 0,
                        "examples": []
                    }
                    
                all_topics[topic]["count"] += 1
                all_topics[topic]["total_score"] += manipulation_score
                
                # Add example if we have space
                if len(all_topics[topic]["examples"]) < 3:
                    all_topics[topic]["examples"].append({
                        "title": metadata.get("title", ""),
                        "source": metadata.get("source", ""),
                        "url": metadata.get("url", "")
                    })
        
        # Calculate average scores and filter by minimum count
        recurring_topics = {}
        min_count = 2  # At least 2 occurrences
        
        for topic, data in all_topics.items():
            if data["count"] >= min_count:
                avg_score = data["total_score"] / data["count"] if data["count"] > 0 else 0
                
                recurring_topics[topic] = {
                    "count": data["count"],
                    "average_score": avg_score,
                    "examples": data["examples"]
                }
        
        # Sort by count
        sorted_topics = sorted(
            recurring_topics.items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )
        
        return {
            "recurring_topics": {topic: data for topic, data in sorted_topics},
            "total_topics": len(all_topics),
            "recurring_count": len(recurring_topics),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _analyze_authoritarian_actors(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze actors associated with authoritarian patterns"""
        # Key political actors to track
        key_actors = ["Donald Trump", "Joe Biden", "Mike Johnson", "Chuck Schumer", "White House", "Congress", "Supreme Court"]
        
        # Track actor mentions and indicators
        actor_mentions = {}
        actor_indicators = {}
        actor_examples = {}
        
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
        for analysis in analyses:
            analysis_text = analysis.get("text", "")
            metadata = analysis.get("metadata", {})
            
            for actor in key_actors:
                if actor in analysis_text:
                    # Initialize actor data if needed
                    if actor not in actor_mentions:
                        actor_mentions[actor] = 0
                        actor_indicators[actor] = {indicator: 0 for indicator in authoritarian_indicators}
                        actor_examples[actor] = []
                    
                    # Count mention
                    actor_mentions[actor] += 1
                    
                    # Check for authoritarian indicators
                    structured_elements = analysis.get("structured_elements", {})
                    for indicator in authoritarian_indicators:
                        if indicator in structured_elements and structured_elements[indicator].get("present", False):
                            actor_indicators[actor][indicator] += 1
                            
                            # Add example if we have room
                            if len(actor_examples[actor]) < 3:
                                example = {
                                    "indicator": indicator,
                                    "source": metadata.get("source", ""),
                                    "title": metadata.get("title", "")
                                }
                                actor_examples[actor].append(example)
        
        # Calculate authoritarian pattern scores
        actor_patterns = {}
        for actor, indicators in actor_indicators.items():
            total_indicators = sum(indicators.values())
            risk_score = min(10, total_indicators)
            
            # Determine risk level
            risk_level = "Low"
            if risk_score >= 7:
                risk_level = "High"
            elif risk_score >= 4:
                risk_level = "Moderate"
            
            actor_patterns[actor] = {
                "authoritarian_pattern_score": risk_score,
                "risk_level": risk_level,
                "total_mentions": actor_mentions[actor],
                "indicator_counts": indicators,
                "examples": actor_examples[actor]
            }
        
        # Sort actors by score
        sorted_actors = sorted(
            actor_patterns.keys(),
            key=lambda x: actor_patterns[x]["authoritarian_pattern_score"],
            reverse=True
        )
        
        return {
            "actor_patterns": actor_patterns,
            "top_actors": sorted_actors[:5],  # Top 5 actors
            "analysis_timestamp": datetime.now().isoformat()
        }
