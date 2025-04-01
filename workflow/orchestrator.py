"""
Night_watcher Workflow Orchestrator
Manages the Night_watcher workflow and orchestrates agent interactions.
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from ..agents.base import LLMProvider
from ..agents.collector import ContentCollector
from ..agents.analyzer import ContentAnalyzer
from ..agents.counter_narrative import CounterNarrativeGenerator
from ..agents.distribution import DistributionPlanner
from ..agents.strategic import StrategicMessaging
from ..memory.system import MemorySystem
from ..utils.io import save_to_file
from ..utils.text import create_slug


class NightWatcherWorkflow:
    """Manages the Night_watcher workflow and orchestrates agent interactions"""

    def __init__(self, llm_provider: LLMProvider, memory_system: Optional[MemorySystem] = None,
                output_dir: str = "data"):
        """Initialize workflow with agents and output directory"""
        self.llm_provider = llm_provider
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Initialize memory system if not provided
        self.memory = memory_system or MemorySystem()

        # Initialize agents
        self.collector = ContentCollector(llm_provider)
        self.analyzer = ContentAnalyzer(llm_provider)
        self.counter_narrative_gen = CounterNarrativeGenerator(llm_provider)
        self.distribution_planner = DistributionPlanner(llm_provider)
        self.strategic_messaging = StrategicMessaging(llm_provider)

        # Set up logging
        self.logger = logging.getLogger("NightWatcherWorkflow")

        # Ensure output directories exist
        self._ensure_data_dirs()

    def _ensure_data_dirs(self):
        """Ensure all data directories exist"""
        directories = [
            f"{self.output_dir}/collected",
            f"{self.output_dir}/analyzed",
            f"{self.output_dir}/counter_narratives"
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

        self.narrative_dir = f"{self.output_dir}/counter_narratives/{self.timestamp}"
        os.makedirs(self.narrative_dir, exist_ok=True)

    def run(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run the Night_watcher workflow"""
        if config is None:
            config = {}

        article_limit = config.get("article_limit", 5)
        manipulation_threshold = config.get("manipulation_threshold", 6)
        sources = config.get("sources", None)

        self.logger.info(f"Starting Night_watcher workflow with timestamp {self.timestamp}")

        # 1. Collect articles
        self.logger.info("Collecting articles...")
        collection_params = {"limit": article_limit}
        if sources:
            collection_params["sources"] = sources

        collection_result = self.collector.process(collection_params)
        articles = collection_result.get("articles", [])
        self.logger.info(f"Collected {len(articles)} articles")

        # Save collected articles
        save_to_file(articles, f"{self.output_dir}/collected/articles_{self.timestamp}.json")

        # 2. Analyze content
        self.logger.info("Analyzing articles for divisive content...")
        analysis_result = self.analyzer.process({"articles": articles})
        analyses = analysis_result.get("analyses", [])

        # Save individual analyses and store in memory system
        for analysis in analyses:
            if "article" in analysis:
                article_slug = create_slug(analysis['article']['title'])
                save_to_file(analysis, f"{self.output_dir}/analyzed/analysis_{article_slug}_{self.timestamp}.json")

                # Store in memory system
                self.memory.store_article_analysis(analysis)

        # 3. Generate counter-narratives
        self.logger.info("Generating counter-narratives for divisive content...")
        counter_narrative_result = self.counter_narrative_gen.process({
            "analyses": analyses,
            "manipulation_threshold": manipulation_threshold
        })

        narratives = counter_narrative_result.get("counter_narratives", [])

        # Process counter-narratives
        for narrative in narratives:
            article_title = narrative.get("article_title", "untitled")
            article_slug = create_slug(article_title)

            # Save the full narrative result
            save_to_file(narrative, f"{self.narrative_dir}/{article_slug}_counter_narratives.json")

            # Find the corresponding analysis for memory storage
            analysis_id = ""
            for analysis in analyses:
                if analysis.get("article", {}).get("title") == article_title:
                    analysis_id = self.memory.store_article_analysis(analysis)
                    break

            # Save individual demographic-targeted narratives
            for demo_narrative in narrative.get("counter_narratives", []):
                demo_id = demo_narrative.get("demographic", "unknown")
                save_to_file(
                    demo_narrative.get("content", ""),
                    f"{self.narrative_dir}/{article_slug}_{demo_id}_narrative.txt"
                )

                # Store in memory system
                self.memory.store_counter_narrative(demo_narrative, analysis_id)

            # Save bridging content
            bridging_content = narrative.get("bridging_content", {})
            save_to_file(
                bridging_content.get("content", ""),
                f"{self.narrative_dir}/{article_slug}_bridging.txt"
            )

            # Store in memory system
            self.memory.store_bridging_content(bridging_content, analysis_id)

            # Generate and save strategic messaging
            for analysis in analyses:
                if analysis.get("article", {}).get("title") == article_title:
                    strategic_result = self.strategic_messaging.process({
                        "analyses": [analysis],
                        "manipulation_threshold": 0  # Set to 0 to guarantee processing
                    })

                    strategic_messages = strategic_result.get("strategic_messages", [])
                    if strategic_messages:
                        save_to_file(
                            strategic_messages[0],
                            f"{self.narrative_dir}/{article_slug}_strategic_messages.json"
                        )

                        # Save individual strategic messages
                        messages = strategic_messages[0].get("strategic_messaging", {})
                        for audience, content in messages.items():
                            save_to_file(
                                content,
                                f"{self.narrative_dir}/{article_slug}_{audience}_strategic.txt"
                            )

            # Generate and save distribution plan
            distribution_result = self.distribution_planner.process({
                "counter_narratives": [narrative]
            })

            distribution_plans = distribution_result.get("distribution_plans", [])
            if distribution_plans:
                save_to_file(
                    distribution_plans[0],
                    f"{self.narrative_dir}/{article_slug}_distribution_plan.json"
                )

                # Save distribution strategy
                save_to_file(
                    distribution_plans[0].get("distribution_strategy", ""),
                    f"{self.narrative_dir}/{article_slug}_distribution_strategy.txt"
                )

                # Save talking points for key demographics
                talking_points = distribution_plans[0].get("talking_points", {})
                for demo, content in talking_points.items():
                    save_to_file(
                        content,
                        f"{self.narrative_dir}/{article_slug}_{demo}_talking_points.txt"
                    )

        self.logger.info(f"Processing complete. All outputs saved in {self.narrative_dir}")

        return {
            "timestamp": self.timestamp,
            "output_dir": self.narrative_dir,
            "articles_collected": len(articles),
            "articles_analyzed": len(analyses),
            "counter_narratives_generated": len(narratives)
        }