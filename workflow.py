"""
Night_watcher Workflow Orchestrator
Manages the workflow for intelligence gathering and analysis.
"""

import os
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union

from providers import LLMProvider
from memory import MemorySystem
from collector import ContentCollector
from analyzer import ContentAnalyzer
from knowledge_graph import KnowledgeGraph
from utils import save_to_file

# Configure logging
logger = logging.getLogger(__name__)

# ==========================================
# Workflow Orchestrator
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

        # Get knowledge graph from memory system
        self.knowledge_graph = self.memory.knowledge_graph

        # Initialize agents
        self.collector = ContentCollector()
        self.analyzer = ContentAnalyzer(llm_provider)

        # Set up logging
        self.logger = logging.getLogger("NightWatcherWorkflow")

        # Create output directories
        self._ensure_data_dirs()

    def _ensure_data_dirs(self):
        """Create necessary output directories"""
        # Main output directories
        self.collected_dir = os.path.join(self.output_dir, "collected")
        self.analyzed_dir = os.path.join(self.output_dir, "analyzed")
        self.analysis_dir = os.path.join(self.output_dir, "analysis")
        
        # Create directories if they don't exist
        os.makedirs(self.collected_dir, exist_ok=True)
        os.makedirs(self.analyzed_dir, exist_ok=True)
        os.makedirs(self.analysis_dir, exist_ok=True)

    def run(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run the Night_watcher intelligence gathering workflow"""
        if config is None:
            config = {}
            
        # Get configuration values
        article_limit = config.get("article_limit", 5)
        sources = config.get("sources", [])
        pattern_analysis_days = config.get("pattern_analysis_days", 30)
        start_date = config.get("start_date")
        end_date = config.get("end_date")
        llm_available = config.get("llm_provider_available", True)
        
        # 1. Content Collection
        self.logger.info("Starting content collection...")
        
        collection_input = {
            "limit": article_limit,
            "sources": sources,
            "start_date": start_date.isoformat() if isinstance(start_date, datetime) else start_date,
            "end_date": end_date.isoformat() if isinstance(end_date, datetime) else end_date
        }
        
        collection_result = self.collector.process(collection_input)
        articles = collection_result.get("articles", [])
        
        # Save collected articles
        for i, article in enumerate(articles):
            filename = f"article_{i+1}_{self.timestamp}.json"
            save_to_file(article, os.path.join(self.collected_dir, filename))
            
        self.logger.info(f"Collected {len(articles)} articles")
        
        # Skip analysis if no LLM available
        if not llm_available:
            self.logger.warning("LLM provider not available, skipping analysis")
            return {
                "articles_collected": len(articles),
                "articles_analyzed": 0,
                "pattern_analyses_generated": 0
            }
            
        # 2. Content Analysis
        if articles:
            self.logger.info("Starting content analysis...")
            
            analysis_input = {"articles": articles}
            analysis_results = self.analyzer.process(analysis_input)
            
            # Get results
            analyses = analysis_results.get("analyses", [])
            auth_analyses = analysis_results.get("authoritarian_analyses", [])
            kg_analyses = analysis_results.get("kg_analyses", [])
            
            # Save analysis results
            for i, analysis in enumerate(analyses):
                filename = f"analysis_{i+1}_{self.timestamp}.json"
                save_to_file(analysis, os.path.join(self.analyzed_dir, filename))
                
            for i, auth_analysis in enumerate(auth_analyses):
                filename = f"auth_analysis_{i+1}_{self.timestamp}.json"
                save_to_file(auth_analysis, os.path.join(self.analyzed_dir, filename))
                
            for i, kg_analysis in enumerate(kg_analyses):
                filename = f"kg_analysis_{i+1}_{self.timestamp}.json"
                save_to_file(kg_analysis, os.path.join(self.analyzed_dir, filename))
            
            # Store in memory system
            stored_analyses = []
            for analysis in analyses:
                analysis_id = self.memory.store_article_analysis(analysis)
                if analysis_id:
                    stored_analyses.append(analysis_id)
                    
            # Process knowledge graph analyses
            for kg_analysis in kg_analyses:
                # Add article ID if not present
                article = kg_analysis["article"]
                if "id" not in article:
                    article["id"] = f"article_{hash(article.get('title', ''))}"
                
                # Process structured data for knowledge graph
                self.knowledge_graph.process_article_analysis(article, kg_analysis)
            
            self.logger.info(f"Analyzed {len(analyses)} articles, stored {len(stored_analyses)} in memory")
            
            # 3. Run pattern analysis
            self.logger.info(f"Running pattern analysis over the last {pattern_analysis_days} days...")
            
            # Get recent analyses for pattern recognition
            recent_analyses = self.memory.get_recent_analyses(pattern_analysis_days)
            
            if recent_analyses:
                # Get authoritarian trends from knowledge graph
                auth_trends = self.knowledge_graph.get_authoritarian_trends(pattern_analysis_days)
                save_to_file(
                    auth_trends,
                    f"{self.analysis_dir}/authoritarian_trends_{self.timestamp}.json"
                )
                
                # Get influential actors
                influential_actors = self.knowledge_graph.get_influential_actors(10)
                save_to_file(
                    influential_actors,
                    f"{self.analysis_dir}/influential_actors_{self.timestamp}.json"
                )
                
                # Get comprehensive democratic erosion analysis
                democratic_erosion = self.knowledge_graph.analyze_democratic_erosion(pattern_analysis_days)
                save_to_file(
                    democratic_erosion,
                    f"{self.analysis_dir}/democratic_erosion_{self.timestamp}.json"
                )
                
                # Get actor coordination patterns
                coordination_patterns = self.knowledge_graph.detect_coordination_patterns(pattern_analysis_days)
                save_to_file(
                    coordination_patterns,
                    f"{self.analysis_dir}/coordination_patterns_{self.timestamp}.json"
                )
                
                # Generate comprehensive intelligence report
                intel_report = self._generate_intelligence_report(
                    democratic_erosion,
                    influential_actors,
                    coordination_patterns,
                    pattern_analysis_days
                )
                save_to_file(
                    intel_report,
                    f"{self.analysis_dir}/intelligence_report_{self.timestamp}.json"
                )
                
                patterns_generated = 5
            else:
                self.logger.warning("No recent analyses found for pattern recognition")
                patterns_generated = 0
                
            return {
                "articles_collected": len(articles),
                "articles_analyzed": len(analyses),
                "knowledge_graph_analyses": len(kg_analyses),
                "pattern_analyses_generated": patterns_generated
            }
        else:
            self.logger.warning("No articles collected, skipping analysis")
            return {
                "articles_collected": 0,
                "articles_analyzed": 0,
                "pattern_analyses_generated": 0
            }

    def _analyze_recurring_topics(self, recent_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze recurring topics across analyses.
        
        Args:
            recent_analyses: List of recent analyses
            
        Returns:
            Dict with topic analysis results
        """
        # This is a legacy method - delegate to knowledge graph for more advanced analysis
        # Get list of topics from analyses
        topics = {}
        
        for analysis in recent_analyses:
            metadata = analysis.get("metadata", {})
            
            # Extract topics from metadata or embedded structured elements
            if "structured_elements" in metadata:
                analysis_topics = metadata["structured_elements"].get("main_topics", [])
                
                for topic in analysis_topics:
                    if topic in topics:
                        topics[topic] += 1
                    else:
                        topics[topic] = 1
        
        # Sort topics by frequency
        sorted_topics = sorted(
            [(topic, count) for topic, count in topics.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            "topics": sorted_topics,
            "analysis_count": len(recent_analyses),
            "timestamp": datetime.now().isoformat()
        }

    def _generate_intelligence_report(self, democratic_erosion: Dict[str, Any],
                                    influential_actors: List[Dict[str, Any]],
                                    coordination_patterns: List[Dict[str, Any]],
                                    lookback_days: int) -> Dict[str, Any]:
        """
        Generate a comprehensive intelligence report.
        
        Args:
            democratic_erosion: Democratic erosion analysis
            influential_actors: List of influential actors
            coordination_patterns: List of coordination patterns
            lookback_days: Number of days in the analysis period
            
        Returns:
            Dict with intelligence report
        """
        # Prepare intelligence report
        report = {
            "title": f"Intelligence Report on Authoritarian Indicators - {datetime.now().strftime('%Y-%m-%d')}",
            "analysis_period": f"Last {lookback_days} days",
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "erosion_score": democratic_erosion.get("erosion_score", 0),
                "risk_level": democratic_erosion.get("risk_level", "Low"),
                "key_institutions_affected": len(democratic_erosion.get("affected_institutions", [])),
                "top_actors": [actor.get("name", "") for actor in influential_actors[:3]],
                "coordination_patterns_detected": len(coordination_patterns)
            },
            "democratic_erosion": democratic_erosion,
            "influential_actors": influential_actors,
            "coordination_patterns": coordination_patterns,
            "recommendations": []
        }
        
        # Generate recommendations based on erosion score
        erosion_score = democratic_erosion.get("erosion_score", 0)
        
        if erosion_score >= 7:
            report["recommendations"] = [
                "URGENT: Multiple clear indicators of authoritarian governance detected",
                "Focus monitoring on institutional undermining patterns",
                "Track coordination between powerful actors",
                "Monitor legislative attempts to reduce checks and balances",
                "Prepare response strategies for further democratic backsliding"
            ]
        elif erosion_score >= 5:
            report["recommendations"] = [
                "HIGH CONCERN: Significant authoritarian patterns detected",
                "Increase monitoring of key institutions under pressure",
                "Track rhetorical attacks on democratic processes",
                "Monitor normalization of anti-democratic narratives",
                "Identify key resistance points within democratic institutions"
            ]
        elif erosion_score >= 3:
            report["recommendations"] = [
                "MODERATE CONCERN: Early authoritarian indicators present",
                "Monitor rhetoric around institutional legitimacy",
                "Track attempts to delegitimize opposition",
                "Document norm violations for pattern analysis",
                "Identify key actors promoting authoritarian narratives"
            ]
        else:
            report["recommendations"] = [
                "LOW CONCERN: Minimal authoritarian indicators detected",
                "Continue monitoring key democratic institutions",
                "Track emerging narratives around governmental authority",
                "Monitor for early signs of norm erosion",
                "Maintain baseline monitoring of key political actors"
            ]
        
        return report
