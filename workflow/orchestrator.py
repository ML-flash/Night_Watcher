"""
Night_watcher Analysis Workflow Orchestrator
Manages the Night_watcher workflow focused on intelligence gathering and analysis.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from agents.base import LLMProvider
from agents.collector import ContentCollector
from agents.analyzer import ContentAnalyzer
from memory.system import MemorySystem
from utils.io import save_to_file
from utils.text import create_slug
from analysis.patterns import PatternRecognition


class NightWatcherWorkflow:
    """Manages the Night_watcher workflow focused on intelligence gathering and analysis"""

    def __init__(self, llm_provider: LLMProvider, memory_system: Optional[MemorySystem] = None,
                 output_dir: str = "data"):
        """Initialize workflow with agents and output directory"""
        # Set up logging first to avoid attribute errors
        self.logger = logging.getLogger("NightWatcherWorkflow")

        self.llm_provider = llm_provider
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Initialize memory system if not provided
        self.memory = memory_system or MemorySystem()

        # Initialize analysis agents only
        self.collector = ContentCollector(llm_provider)
        self.analyzer = ContentAnalyzer(llm_provider)

        # Add pattern recognition
        self.pattern_recognition = PatternRecognition(self.memory)

        # Ensure output directories exist
        self._ensure_data_dirs()

    def _ensure_data_dirs(self):
        """Ensure all data directories exist"""
        directories = [
            f"{self.output_dir}/collected",
            f"{self.output_dir}/analyzed",
            f"{self.output_dir}/reports",
            f"{self.output_dir}/analysis"
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

        self.report_dir = f"{self.output_dir}/reports/{self.timestamp}"
        self.analysis_dir = f"{self.output_dir}/analysis/{self.timestamp}"

        os.makedirs(self.report_dir, exist_ok=True)
        os.makedirs(self.analysis_dir, exist_ok=True)

    def run(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run the Night_watcher intelligence analysis workflow"""
        if config is None:
            config = {}

        article_limit = config.get("article_limit", 5)
        sources = config.get("sources", None)
        pattern_analysis_days = config.get("pattern_analysis_days", 30)

        self.logger.info(f"Starting Night_watcher intelligence analysis workflow with timestamp {self.timestamp}")

        # 1. Collect articles
        self.logger.info("Collecting articles with focus on government/political content...")
        collection_params = {"limit": article_limit}
        if sources:
            collection_params["sources"] = sources

        collection_result = self.collector.process(collection_params)
        articles = collection_result.get("articles", [])
        self.logger.info(f"Collected {len(articles)} articles")

        # Save collected articles
        save_to_file(articles, f"{self.output_dir}/collected/articles_{self.timestamp}.json")

        # 2. Analyze content for both divisive elements and authoritarian patterns
        self.logger.info("Analyzing articles for divisive content and authoritarian patterns...")
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
                    save_to_file(auth_analysis, f"{self.output_dir}/analyzed/auth_analysis_{article_slug}_{self.timestamp}.json")

                    # Store authoritarian analysis in memory (could enhance memory system to handle this specifically)
                    auth_meta = {
                        "type": "authoritarian_analysis",
                        "title": analysis['article']['title'],
                        "source": analysis['article']['source'],
                        "url": analysis['article'].get('url', ''),
                        "parent_id": analysis_id
                    }

                    # Extract authoritarian score if available
                    if "structured_elements" in auth_analysis:
                        auth_meta["authoritarian_score"] = auth_analysis["structured_elements"].get(
                            "authoritarian_score", 0)

                    self.memory.store.add_item(
                        f"auth_{analysis_id}",
                        auth_analysis.get("authoritarian_analysis", ""),
                        auth_meta
                    )

        # 3. Run pattern analysis to identify authoritarian trends
        self.logger.info(f"Running pattern analysis over the last {pattern_analysis_days} days...")

        # Authoritarian trend analysis
        auth_trends = self.pattern_recognition.analyze_source_bias_patterns(pattern_analysis_days)
        save_to_file(
            auth_trends,
            f"{self.analysis_dir}/authoritarian_trends_{self.timestamp}.json"
        )

        # Topic analysis
        topic_analysis = self.pattern_recognition.identify_recurring_topics()
        save_to_file(
            topic_analysis,
            f"{self.analysis_dir}/topic_analysis_{self.timestamp}.json"
        )

        # Actor analysis
        actor_analysis = self.pattern_recognition.analyze_authoritarian_actors(pattern_analysis_days)
        save_to_file(
            actor_analysis,
            f"{self.analysis_dir}/actor_analysis_{self.timestamp}.json"
        )

        # Correlation analysis
        correlation_analysis = self.pattern_recognition.analyze_source_correlation()
        save_to_file(
            correlation_analysis,
            f"{self.analysis_dir}/source_correlation_{self.timestamp}.json"
        )

        # Temporal trend analysis
        temporal_analysis = self.pattern_recognition.analyze_temporal_trends(lookback_days=pattern_analysis_days)
        save_to_file(
            temporal_analysis,
            f"{self.analysis_dir}/temporal_trends_{self.timestamp}.json"
        )

        # Manipulation techniques analysis
        techniques_analysis = self.pattern_recognition.analyze_manipulation_techniques()
        save_to_file(
            techniques_analysis,
            f"{self.analysis_dir}/manipulation_techniques_{self.timestamp}.json"
        )

        self.logger.info(f"Processing complete. All outputs saved in {self.output_dir}")

        return {
            "timestamp": self.timestamp,
            "output_dir": self.output_dir,
            "articles_collected": len(articles),
            "articles_analyzed": len(analyses),
            "pattern_analyses_generated": 6  # All the pattern analyses we generated
        }
