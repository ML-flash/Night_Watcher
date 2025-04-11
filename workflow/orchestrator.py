"""
Enhanced Night_watcher Workflow Orchestrator
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
from analysis.patterns import PatternRecognition


class NightWatcherWorkflow:
    """Manages the enhanced Night_watcher workflow focused on democratic resilience"""

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

        # Add pattern recognition
        self.pattern_recognition = PatternRecognition(self.memory)
        
        # Add report generator if available
        try:
            from agents.report_generator import DemocraticResilienceReportGenerator
            self.report_generator = DemocraticResilienceReportGenerator(llm_provider, self.memory)
            self.has_report_generator = True
        except ImportError:
            self.has_report_generator = False
            self.logger.warning("DemocraticResilienceReportGenerator not available")

        # Set up logging
        self.logger = logging.getLogger("NightWatcherWorkflow")

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

        self.logger.info(f"Starting Night_watcher workflow with timestamp {self.timestamp}")

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
                        auth_meta["authoritarian_score"] = auth_analysis["structured_elements"].get("authoritarian_score", 0)
                    
                    self.memory.store.add_item(
                        f"auth_{analysis_id}",
                        auth_analysis.get("authoritarian_analysis", ""),
                        auth_meta
                    )

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
        
        # 5. Generate comprehensive reports if enabled and available
        reports_generated = 0
        if generate_reports and self.has_report_generator:
            self.logger.info("Generating democratic resilience reports...")
            
            # Weekly report
            weekly_report = self.report_generator.process({
                "report_type": "weekly",
                "lookback_days": 7,
                "pattern_findings": auth_trends
            })
            
            save_to_file(
                weekly_report.get("report", ""),
                f"{self.report_dir}/weekly_resilience_report_{self.timestamp}.txt"
            )
            
            # Action kit
            action_kit = self.report_generator.process({
                "report_type": "action_kit",
                "pattern_findings": auth_trends
            })
            
            save_to_file(
                action_kit.get("action_kit", ""),
                f"{self.report_dir}/democratic_action_kit_{self.timestamp}.txt"
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
                
                actor_slug = actor.lower().replace(" ", "_")
                save_to_file(
                    actor_report.get("report", ""),
                    f"{self.report_dir}/actor_report_{actor_slug}_{self.timestamp}.txt"
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
                
                topic_slug = create_slug(topic)
                save_to_file(
                    topic_report.get("report", ""),
                    f"{self.report_dir}/topic_report_{topic_slug}_{self.timestamp}.txt"
                )
            
            reports_generated = 5 + len(sorted_actors[:3]) + len(sorted_topics[:3])

        self.logger.info(f"Processing complete. All outputs saved in {self.output_dir}")

        return {
            "timestamp": self.timestamp,
            "output_dir": self.output_dir,
            "articles_collected": len(articles),
            "articles_analyzed": len(analyses),
            "counter_narratives_generated": len(narratives),
            "pattern_analyses_generated": 3,  # Auth trends, topics, actors
            "reports_generated": reports_generated
        }