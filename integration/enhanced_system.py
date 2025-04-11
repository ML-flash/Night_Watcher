"""
Night_watcher Enhanced Integration Module
Integrates the advanced memory and analysis system with the existing workflow.
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from memory.advanced_system import AdvancedMemorySystem
from analysis.enhanced_patterns import EnhancedPatternRecognition
from workflow.orchestrator import NightWatcherWorkflow
from agents.base import LLMProvider

logger = logging.getLogger(__name__)


class EnhancedNightWatcherWorkflow(NightWatcherWorkflow):
    """Enhanced version of the Night_watcher workflow with advanced memory and analysis"""

    def __init__(self, llm_provider: LLMProvider, memory_config: Dict[str, Any] = None,
                 output_dir: str = "data"):
        """
        Initialize with LLM provider and optional memory configuration.

        Args:
            llm_provider: LLM provider
            memory_config: Configuration for the advanced memory system
            output_dir: Output directory
        """
        self.llm_provider = llm_provider
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Initialize advanced memory system
        self.advanced_memory = AdvancedMemorySystem(memory_config or {})

        # Initialize the base workflow with the advanced memory's legacy system for compatibility
        super().__init__(
            llm_provider=llm_provider,
            memory_system=self.advanced_memory.legacy_memory,
            output_dir=output_dir
        )

        # Replace pattern recognition with enhanced version
        self.pattern_recognition = self.advanced_memory.pattern_recognition

        # Add report generator if available
        try:
            from agents.report_generator import DemocraticResilienceReportGenerator
            self.report_generator = DemocraticResilienceReportGenerator(
                llm_provider, self.advanced_memory.legacy_memory
            )
            self.has_report_generator = True
        except ImportError:
            self.has_report_generator = False
            self.logger.warning("DemocraticResilienceReportGenerator not available")

        # Set up logging
        self.logger = logging.getLogger("EnhancedNightWatcherWorkflow")

        # Ensure output directories exist
        self._ensure_data_dirs()

    def run(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run the enhanced Night_watcher workflow with advanced memory and analysis.

        Args:
            config: Configuration parameters

        Returns:
            Workflow results
        """
        if config is None:
            config = {}

        article_limit = config.get("article_limit", 5)
        manipulation_threshold = config.get("manipulation_threshold", 6)
        authoritarian_threshold = config.get("authoritarian_threshold", 5)
        sources = config.get("sources", None)
        generate_reports = config.get("generate_reports", True)
        pattern_analysis_days = config.get("pattern_analysis_days", 30)

        self.logger.info(f"Starting enhanced Night_watcher workflow with timestamp {self.timestamp}")

        # 1. Collect articles (use base implementation)
        self.logger.info("Collecting articles with focus on government/political content...")
        collection_params = {"limit": article_limit}
        if sources:
            collection_params["sources"] = sources

        collection_result = self.collector.process(collection_params)
        articles = collection_result.get("articles", [])
        self.logger.info(f"Collected {len(articles)} articles")

        # Save collected articles
        from utils.io import save_to_file
        save_to_file(articles, f"{self.output_dir}/collected/articles_{self.timestamp}.json")

        # 2. Analyze content for both divisive elements and authoritarian patterns
        self.logger.info("Analyzing articles for divisive content and authoritarian patterns...")
        analysis_result = self.analyzer.process({"articles": articles})
        analyses = analysis_result.get("analyses", [])
        auth_analyses = analysis_result.get("authoritarian_analyses", [])

        # Save individual analyses and store in memory system
        for i, analysis in enumerate(analyses):
            if "article" in analysis:
                article_slug = self._create_slug(analysis['article']['title'])
                save_to_file(analysis, f"{self.output_dir}/analyzed/analysis_{article_slug}_{self.timestamp}.json")

                # Store in advanced memory system instead of base memory
                analysis_id = self.advanced_memory.store_article_analysis(analysis)

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

        # Process counter-narratives (with advanced memory storage)
        for narrative in narratives:
            article_title = narrative.get("article_title", "untitled")
            article_slug = self._create_slug(article_title)

            # Save the full narrative result
            save_to_file(narrative, f"{self.narrative_dir}/{article_slug}_counter_narratives.json")

            # Find the corresponding analysis for memory storage
            analysis_id = ""
            for analysis in analyses:
                if analysis.get("article", {}).get("title") == article_title:
                    analysis_id = self.advanced_memory.store_article_analysis(analysis)
                    break

            # Save individual demographic-targeted narratives
            for demo_narrative in narrative.get("counter_narratives", []):
                demo_id = demo_narrative.get("demographic", "unknown")
                save_to_file(
                    demo_narrative.get("content", ""),
                    f"{self.narrative_dir}/{article_slug}_{demo_id}_narrative.txt"
                )

                # Store in advanced memory system
                self.advanced_memory.store_counter_narrative(demo_narrative, analysis_id)

            # Save authoritarian responses if available
            for auth_response in narrative.get("authoritarian_responses", []):
                demo_id = auth_response.get("demographic", "unknown")
                save_to_file(
                    auth_response.get("content", ""),
                    f"{self.narrative_dir}/{article_slug}_{demo_id}_auth_response.txt"
                )

            # Save democratic principles narrative if available
            if narrative.get("democratic_principles_narrative"):
                save_to_file(
                    narrative["democratic_principles_narrative"].get("content", ""),
                    f"{self.narrative_dir}/{article_slug}_democratic_principles.txt"
                )

            # Save bridging content
            bridging_content = narrative.get("bridging_content", {})
            save_to_file(
                bridging_content.get("content", ""),
                f"{self.narrative_dir}/{article_slug}_bridging.txt"
            )

            # Generate and save strategic messaging (same as base implementation)
            # [strategic messaging code from base unchanged]
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

            # Generate and save distribution plan (same as base implementation)
            # [distribution plan code from base unchanged]
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

        # 4. Run enhanced pattern analysis
        self.logger.info(f"Running enhanced pattern analysis over the last {pattern_analysis_days} days...")

        # Authoritarian trend analysis (using enhanced pattern recognition)
        auth_trends = self.pattern_recognition.analyze_authoritarian_trend_patterns(pattern_analysis_days)
        save_to_file(
            auth_trends,
            f"{self.analysis_dir}/authoritarian_trends_{self.timestamp}.json"
        )

        # Store in advanced memory
        self.advanced_memory.store_pattern_analysis("authoritarian_trends", auth_trends)

        # Actor analysis (using enhanced pattern recognition)
        actor_analysis = self.pattern_recognition.analyze_authoritarian_actors(pattern_analysis_days)
        save_to_file(
            actor_analysis,
            f"{self.analysis_dir}/actor_analysis_{self.timestamp}.json"
        )

        # Store in advanced memory
        self.advanced_memory.store_pattern_analysis("actor_analysis", actor_analysis)

        # Topic analysis (using enhanced pattern recognition)
        topic_analysis = self.pattern_recognition.identify_recurring_topics()
        save_to_file(
            topic_analysis,
            f"{self.analysis_dir}/topic_analysis_{self.timestamp}.json"
        )

        # Store in advanced memory
        self.advanced_memory.store_pattern_analysis("topic_analysis", topic_analysis)

        # NEW: Predictive analysis
        predictive_analysis = self.pattern_recognition.predict_authoritarian_escalation()
        save_to_file(
            predictive_analysis,
            f"{self.analysis_dir}/predictive_analysis_{self.timestamp}.json"
        )

        # Store in advanced memory
        self.advanced_memory.store_pattern_analysis("predictive_analysis", predictive_analysis)

        # 5. Generate enhanced intelligence brief
        intelligence_brief = self.advanced_memory.generate_intelligence_brief(
            focus="general",
            lookback_days=pattern_analysis_days
        )

        save_to_file(
            intelligence_brief,
            f"{self.report_dir}/intelligence_brief_{self.timestamp}.json"
        )

        # 6. Generate comprehensive reports if enabled and available (same as base implementation)
        reports_generated = 0
        if generate_reports and self.has_report_generator:
            self.logger.info("Generating democratic resilience reports...")

            # Weekly report
            weekly_report = self.report_generator.process({
                "report_type": "weekly",
                "lookback_days": 7,
                "pattern_findings": auth_trends
            })

            report_content = weekly_report.get("report", "")
            save_to_file(
                report_content,
                f"{self.report_dir}/weekly_resilience_report_{self.timestamp}.txt"
            )

            # Store in advanced memory
            self.advanced_memory.store_intelligence_report(
                report_type="weekly",
                report_content=report_content,
                report_data=weekly_report
            )

            # Action kit
            action_kit = self.report_generator.process({
                "report_type": "action_kit",
                "pattern_findings": auth_trends
            })

            action_kit_content = action_kit.get("action_kit", "")
            save_to_file(
                action_kit_content,
                f"{self.report_dir}/democratic_action_kit_{self.timestamp}.txt"
            )

            # Store in advanced memory
            self.advanced_memory.store_intelligence_report(
                report_type="action_kit",
                report_content=action_kit_content,
                report_data=action_kit
            )

            # Actor reports for top 3 concerning actors
            actor_data = actor_analysis.get("actor_patterns", {})
            sorted_actors = sorted(
                actor_data.items(),
                key=lambda x: x[1].get("authoritarian_pattern_score", 0),
                reverse=True
            )

            for actor, data in sorted_actors[:3]:
                actor_report = self.report_generator.process({
                    "report_type": "actor",
                    "actor": actor,
                    "actor_data": data
                })

                actor_report_content = actor_report.get("report", "")
                actor_slug = actor.lower().replace(" ", "_")
                save_to_file(
                    actor_report_content,
                    f"{self.report_dir}/actor_report_{actor_slug}_{self.timestamp}.txt"
                )

                # Store in advanced memory
                self.advanced_memory.store_intelligence_report(
                    report_type=f"actor_{actor_slug}",
                    report_content=actor_report_content,
                    report_data=actor_report
                )

            # Topic reports for top 3 concerning topics
            topic_data = topic_analysis.get("recurring_topics", {})
            sorted_topics = sorted(
                topic_data.items(),
                key=lambda x: x[1].get("average_auth_score", 0),
                reverse=True
            )

            for topic, data in sorted_topics[:3]:
                topic_report = self.report_generator.process({
                    "report_type": "topic",
                    "topic": topic,
                    "topic_data": data
                })

                topic_report_content = topic_report.get("report", "")
                topic_slug = self._create_slug(topic)
                save_to_file(
                    topic_report_content,
                    f"{self.report_dir}/topic_report_{topic_slug}_{self.timestamp}.txt"
                )

                # Store in advanced memory
                self.advanced_memory.store_intelligence_report(
                    report_type=f"topic_{topic_slug}",
                    report_content=topic_report_content,
                    report_data=topic_report
                )

            reports_generated = 5 + len(sorted_actors[:3]) + len(sorted_topics[:3])

        # Save the advanced memory system
        memory_dir = os.path.join(self.output_dir, "memory")
        memory_path = os.path.join(memory_dir, "night_watcher_memory.pkl")
        os.makedirs(memory_dir, exist_ok=True)
        self.advanced_memory.save(memory_path)

        self.logger.info(f"Enhanced processing complete. All outputs saved in {self.output_dir}")

        return {
            "timestamp": self.timestamp,
            "output_dir": self.output_dir,
            "articles_collected": len(articles),
            "articles_analyzed": len(analyses),
            "counter_narratives_generated": len(narratives),
            "pattern_analyses_generated": 4,  # Auth trends, topics, actors, predictive
            "reports_generated": reports_generated,
            "intelligence_brief_generated": True
        }

    def _create_slug(self, text: str) -> str:
        """Create a URL-safe slug from text"""
        import re
        # Remove non-alphanumeric characters and replace with hyphens
        slug = re.sub(r'[^\w\s-]', '', text.lower())
        # Replace whitespace with hyphens
        slug = re.sub(r'[\s]+', '-', slug)
        # Limit length
        slug = slug[:40].rstrip('-')
        return slug


def run_enhanced_workflow(config_path: str, llm_provider: LLMProvider,
                          output_dir: str = "data", memory_path: str = None) -> Dict[str, Any]:
    """
    Helper function to run the enhanced workflow.

    Args:
        config_path: Path to configuration file
        llm_provider: LLM provider
        output_dir: Output directory
        memory_path: Path to memory file

    Returns:
        Workflow results
    """
    from config import load_config

    # Load configuration
    config = load_config(config_path)

    # Create memory configuration
    memory_config = config.get("memory", {})

    # Initialize workflow
    workflow = EnhancedNightWatcherWorkflow(
        llm_provider=llm_provider,
        memory_config=memory_config,
        output_dir=output_dir
    )

    # Load existing memory if path provided
    if memory_path and os.path.exists(memory_path):
        logger.info(f"Loading memory from {memory_path}")
        workflow.advanced_memory.load(memory_path)

    # Run workflow
    workflow_config = {
        "article_limit": config["content_collection"]["article_limit"],
        "manipulation_threshold": config["content_analysis"]["manipulation_threshold"],
        "sources": config["content_collection"]["sources"]
    }

    result = workflow.run(workflow_config)

    return result