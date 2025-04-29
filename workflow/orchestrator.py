"""
Night_watcher Workflow Orchestrator
Manages the Night_watcher workflow with focus on authoritarian pattern detection and democratic resilience.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from agents.base import LLMProvider
from agents.collector import ContentCollector
from agents.analyzer import ContentAnalyzer
from agents.counter_narrative import CounterNarrativeGenerator
from agents.distribution import DistributionPlanner
from agents.strategic import StrategicMessaging
from memory.system import MemorySystem
from utils.io import save_to_file
from utils.text import create_slug
from utils.date_tracking import get_last_run_date, get_analysis_date_range, save_run_date
from analysis.patterns import PatternRecognition


class NightWatcherWorkflow:
    """Manages the enhanced Night_watcher workflow focused on democratic resilience"""

    def __init__(self, llm_provider: LLMProvider, memory_system: Optional[MemorySystem] = None,
                 output_dir: str = "data"):
        """Initialize workflow with agents and output directory"""
        self.llm_provider = llm_provider
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Set up logging first
        self.logger = logging.getLogger("NightWatcherWorkflow")

        # Initialize memory system if not provided
        self.memory = memory_system or MemorySystem()

        # Initialize agents
        self.collector = ContentCollector(llm_provider)
        self.analyzer = ContentAnalyzer(llm_provider)
        self.counter_narrative_gen = CounterNarrativeGenerator(llm_provider)
        self.distribution_planner = DistributionPlanner(llm_provider)
        self.strategic_messaging = StrategicMessaging(llm_provider)

        # Add pattern recognition
        self.pattern_recognition = PatternRecognition(self.memory)

        # Add report generator if available
        self.has_report_generator = False
        try:
            from agents.report_generator import DemocraticResilienceReportGenerator
            self.report_generator = DemocraticResilienceReportGenerator(llm_provider, self.memory)
            self.has_report_generator = True
        except ImportError:
            self.logger.warning("DemocraticResilienceReportGenerator not available")

        # Ensure output directories exist
        self._ensure_data_dirs()

    def _ensure_data_dirs(self):
        """Ensure all data directories exist"""
        directories = [
            f"{self.output_dir}/collected",
            f"{self.output_dir}/analyzed",
            f"{self.output_dir}/counter_narratives",
            f"{self.output_dir}/reports",
            f"{self.output_dir}/analysis"
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

        self.narrative_dir = f"{self.output_dir}/counter_narratives/{self.timestamp}"
        self.report_dir = f"{self.output_dir}/reports/{self.timestamp}"
        self.analysis_dir = f"{self.output_dir}/analysis/{self.timestamp}"

        os.makedirs(self.narrative_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)
        os.makedirs(self.analysis_dir, exist_ok=True)

    def run(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run the enhanced Night_watcher workflow"""
        if config is None:
            config = {}

        article_limit = config.get("article_limit", 5)
        manipulation_threshold = config.get("manipulation_threshold", 6)
        authoritarian_threshold = config.get("authoritarian_threshold", 5)
        sources = config.get("sources", None)
        generate_reports = config.get("generate_reports", True)
        pattern_analysis_days = config.get("pattern_analysis_days", 30)
        start_date = config.get("start_date", None)
        end_date = config.get("end_date", None)
        llm_provider_available = config.get("llm_provider_available", True)

        self.logger.info(f"Starting Night_watcher workflow with timestamp {self.timestamp}")

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
                "counter_narratives_generated": 0,
                "pattern_analyses_generated": 0,
                "articles": articles
            }

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
                    save_to_file(auth_analysis,
                                 f"{self.output_dir}/analyzed/auth_analysis_{article_slug}_{self.timestamp}.json")

        # 3. Generate counter-narratives with enhanced focus on democratic resilience
        self.logger.info("Generating counter-narratives for divisive and authoritarian content...")
        counter_narrative_result = self.counter_narrative_gen.process({
            "analyses": analyses,
            "authoritarian_analyses": auth_analyses,
            "manipulation_threshold": manipulation_threshold,
            "authoritarian_threshold": authoritarian_threshold
        })

        narratives = counter_narrative_result.get("counter_narratives", [])

        # Process counter-narratives
        for narrative in narratives:
            article_title = narrative.get("article_title", "untitled")
            article_slug = create_slug(article_title)

            # Save the full narrative result
            save_to_file(narrative, f"{self.narrative_dir}/{article_slug}_counter_narratives.json")

        # 4. Run pattern analysis to identify authoritarian trends
        self.logger.info(f"Running pattern analysis over the last {pattern_analysis_days} days...")

        # Authoritarian trend analysis
        auth_trends = self.pattern_recognition.analyze_authoritarian_trend_patterns(pattern_analysis_days)
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

        # Return results
        return {
            "timestamp": self.timestamp,
            "output_dir": self.output_dir,
            "articles_collected": len(articles),
            "articles_analyzed": len(analyses),
            "counter_narratives_generated": len(narratives),
            "pattern_analyses_generated": 3,  # Auth trends, topics, actors
            "analyses": analyses,
            "auth_analyses": auth_analyses
        }
