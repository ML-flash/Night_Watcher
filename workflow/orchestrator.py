"""
Night_watcher Workflow Orchestrator
Manages the Night_watcher workflow with focus on intelligence gathering and analysis.
"""

import os
import json
import uuid
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

# Import the new components
from content_repository import ContentRepository
from content_processor import ContentProcessor
from analysis_registry import AnalysisRegistry


class NightWatcherWorkflow:
    """Manages the enhanced Night_watcher workflow focused on intelligence analysis"""

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

        # Initialize agents - only for collection and analysis
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
            f"{self.output_dir}/analysis"
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

        self.analysis_dir = f"{self.output_dir}/analysis/{self.timestamp}"
        os.makedirs(self.analysis_dir, exist_ok=True)

    def run(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run the Night_watcher workflow with focus on intelligence analysis"""
        if config is None:
            config = {}

        article_limit = config.get("article_limit", 5)
        sources = config.get("sources", None)
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
                    save_to_file(auth_analysis,
                                 f"{self.output_dir}/analyzed/auth_analysis_{article_slug}_{self.timestamp}.json")

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

    # NEW METHODS BELOW FOR REPOSITORY-BASED ARCHITECTURE

    def initialize_repository_components(self):
        """Initialize repository components for the workflow"""
        # Initialize content repository
        self.content_repository = ContentRepository(os.path.join(self.output_dir, "raw"))
        
        # Initialize content processor
        self.content_processor = ContentProcessor(
            repository=self.content_repository,
            processed_dir=os.path.join(self.output_dir, "processed")
        )
        
        # Initialize analysis registry
        self.analysis_registry = AnalysisRegistry(os.path.join(self.output_dir, "registry"))
        
        # Ensure directories exist
        os.makedirs(os.path.join(self.output_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "processed"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "registry"), exist_ok=True)
        
        self.logger.info("Initialized repository components")

    def run_with_repository(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run the Night_watcher workflow using the repository architecture.
        
        Args:
            config: Configuration parameters
            
        Returns:
            Workflow results
        """
        if config is None:
            config = {}
        
        # Initialize repository components if not already done
        if not hasattr(self, 'content_repository'):
            self.initialize_repository_components()
        
        article_limit = config.get("article_limit", 5)
        manipulation_threshold = config.get("manipulation_threshold", 6)
        authoritarian_threshold = config.get("authoritarian_threshold", 5)
        sources = config.get("sources", None)
        pattern_analysis_days = config.get("pattern_analysis_days", 30)
        
        self.logger.info(f"Starting repository-based workflow with timestamp {self.timestamp}")
        
        # 1. Collect articles and store in repository
        self.logger.info("Collecting articles with focus on government/political content...")
        collection_params = {"limit": article_limit}
        if sources:
            collection_params["sources"] = sources
        
        collection_result = self.collector.process(collection_params)
        articles = collection_result.get("articles", [])
        self.logger.info(f"Collected {len(articles)} articles")
        
        # Store in repository
        content_ids = []
        for article in articles:
            content_id = self.content_repository.store_content(article, "rss")
            content_ids.append(content_id)
        
        # 2. Process the collected content
        processed_ids = []
        for content_id in content_ids:
            try:
                processed_id = self.content_processor.process_content(content_id)
                processed_ids.append(processed_id)
            except Exception as e:
                self.logger.error(f"Error processing content {content_id}: {str(e)}")
        
        # 3. Analyze content for both divisive elements and authoritarian patterns
        # Load processed content
        processed_items = []
        for proc_id in processed_ids:
            processed_content = self.content_processor.get_processed_content(proc_id)
            if processed_content:
                processed_items.append(processed_content)
        
        # Extract articles for analysis
        articles_for_analysis = [item["content"] for item in processed_items]
        
        self.logger.info("Analyzing articles for divisive content and authoritarian patterns...")
        analysis_result = self.analyzer.process({"articles": articles_for_analysis})
        analyses = analysis_result.get("analyses", [])
        auth_analyses = analysis_result.get("authoritarian_analyses", [])
        
        # Store analysis results
        analysis_ids = {
            "content_analysis": [],
            "authoritarian_analysis": []
        }
        
        analysis_dir = os.path.join(self.output_dir, "analyzed")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Store content analyses
        for i, analysis in enumerate(analyses):
            if "article" in analysis:
                article_slug = create_slug(analysis['article']['title'])
                
                # Generate analysis ID
                analysis_id = f"analysis_{article_slug}_{self.timestamp}"
                
                # Save analysis
                save_to_file(analysis, f"{analysis_dir}/analysis_{article_slug}_{self.timestamp}.json")
                
                # Register in registry
                processed_id = processed_ids[i] if i < len(processed_ids) else None
                source_ids = [processed_id] if processed_id else []
                
                self.analysis_registry.register_analysis(
                    analysis_id=analysis_id,
                    analysis_type="content_analysis",
                    source_ids=source_ids,
                    metadata={
                        "title": analysis["article"].get("title", ""),
                        "source": analysis["article"].get("source", ""),
                        "bias_label": analysis["article"].get("bias_label", "unknown"),
                        "timestamp": analysis.get("timestamp", "")
                    }
                )
                
                analysis_ids["content_analysis"].append(analysis_id)
                
                # Store in memory system (for backward compatibility)
                self.memory.store_article_analysis(analysis)
                
                # If we have corresponding authoritarian analysis, save it
                if i < len(auth_analyses):
                    auth_analysis = auth_analyses[i]
                    
                    # Generate analysis ID
                    auth_analysis_id = f"auth_analysis_{article_slug}_{self.timestamp}"
                    
                    # Save analysis
                    save_to_file(auth_analysis, f"{analysis_dir}/auth_analysis_{article_slug}_{self.timestamp}.json")
                    
                    # Register in registry
                    self.analysis_registry.register_analysis(
                        analysis_id=auth_analysis_id,
                        analysis_type="authoritarian_analysis",
                        source_ids=[processed_id, analysis_id] if processed_id else [analysis_id],
                        metadata={
                            "title": analysis["article"].get("title", ""),
                            "source": analysis["article"].get("source", ""),
                            "bias_label": analysis["article"].get("bias_label", "unknown"),
                            "timestamp": auth_analysis.get("timestamp", "")
                        }
                    )
                    
                    analysis_ids["authoritarian_analysis"].append(auth_analysis_id)
                    
                    # Store authoritarian analysis in memory (for backward compatibility)
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
        
        # 4. Run pattern analysis to identify authoritarian trends
        self.logger.info(f"Running pattern analysis over the last {pattern_analysis_days} days...")
        
        # Authoritarian trend analysis
        auth_trends = self.pattern_recognition.analyze_source_bias_patterns(pattern_analysis_days)
        
        # Store analysis and register
        pattern_dir = os.path.join(self.output_dir, "analysis", self.timestamp)
        os.makedirs(pattern_dir, exist_ok=True)
        
        trend_id = f"trends_{self.timestamp}"
        save_to_file(auth_trends, f"{pattern_dir}/authoritarian_trends_{self.timestamp}.json")
        
        self.analysis_registry.register_analysis(
            analysis_id=trend_id,
            analysis_type="authoritarian_trends",
            source_ids=analysis_ids.get("authoritarian_analysis", []),
            metadata={
                "lookback_days": pattern_analysis_days,
                "timestamp": auth_trends.get("timestamp", "")
            }
        )
        
        # Topic analysis
        topic_analysis = self.pattern_recognition.identify_recurring_topics()
        topic_id = f"topics_{self.timestamp}"
        save_to_file(topic_analysis, f"{pattern_dir}/topic_analysis_{self.timestamp}.json")
        
        self.analysis_registry.register_analysis(
            analysis_id=topic_id,
            analysis_type="topic_analysis",
            source_ids=analysis_ids.get("content_analysis", []),
            metadata={
                "timestamp": topic_analysis.get("timestamp", "")
            }
        )
        
        # Actor analysis
        actor_analysis = self.pattern_recognition.analyze_authoritarian_actors(pattern_analysis_days)
        actor_id = f"actors_{self.timestamp}"
        save_to_file(actor_analysis, f"{pattern_dir}/actor_analysis_{self.timestamp}.json")
        
        self.analysis_registry.register_analysis(
            analysis_id=actor_id,
            analysis_type="actor_analysis",
            source_ids=analysis_ids.get("authoritarian_analysis", []),
            metadata={
                "lookback_days": pattern_analysis_days,
                "timestamp": actor_analysis.get("timestamp", "")
            }
        )
        
        self.logger.info(f"Processing complete. All outputs saved in {self.output_dir}")
        
        return {
            "timestamp": self.timestamp,
            "output_dir": self.output_dir,
            "articles_collected": len(articles),
            "articles_analyzed": len(analyses),
            "content_ids": content_ids,
            "processed_ids": processed_ids,
            "analysis_ids": analysis_ids,
            "pattern_ids": {
                "authoritarian_trends": [trend_id],
                "topic_analysis": [topic_id],
                "actor_analysis": [actor_id]
            }
        }

    def export_analysis_history(self, content_id: str, output_path: str = None) -> Dict[str, Any]:
        """
        Export the full analytical history for a piece of content
        
        Args:
            content_id: ID of the content
            output_path: Optional path to save the export
            
        Returns:
            Dictionary with analysis history
        """
        # Initialize repository components if not already done
        if not hasattr(self, 'content_repository'):
            self.initialize_repository_components()
        
        # Get raw content metadata
        raw_metadata = self.content_repository.get_metadata(content_id)
        if not raw_metadata:
            return {"error": f"Content {content_id} not found"}
            
        # Get the full analysis chain
        analysis_chain = self.analysis_registry.get_downstream_chain(content_id)
        
        # Collect all analyses
        analyses = {}
        for analysis_type, analysis_ids in analysis_chain.items():
            analyses[analysis_type] = []
            
            for analysis_id in analysis_ids:
                # Get analysis provenance
                provenance = self.analysis_registry.get_analysis_provenance(analysis_id)
                
                # Get analysis file path
                file_path = None
                if analysis_type == "content_analysis":
                    file_path = os.path.join(self.output_dir, "analyzed", f"{analysis_id}.json")
                elif analysis_type == "authoritarian_analysis":
                    file_path = os.path.join(self.output_dir, "analyzed", f"{analysis_id}.json")
                elif analysis_type.startswith("proc_"):
                    file_path = os.path.join(self.output_dir, "processed", f"{analysis_id}.json")
                else:
                    # Try to find in pattern analysis directory
                    for pattern_dir in os.listdir(os.path.join(self.output_dir, "analysis")):
                        candidate_path = os.path.join(self.output_dir, "analysis", pattern_dir, f"{analysis_id}.json")
                        if os.path.exists(candidate_path):
                            file_path = candidate_path
                            break
                    
                # Load analysis content if available
                analysis_content = None
                if file_path and os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        analysis_content = json.load(f)
                        
                # Add to analyses
                analyses[analysis_type].append({
                    "analysis_id": analysis_id,
                    "provenance": provenance,
                    "content": analysis_content
                })
                
        # Create export
        export = {
            "content_id": content_id,
            "content_metadata": raw_metadata,
            "analysis_chain": analysis_chain,
            "analyses": analyses,
            "export_timestamp": datetime.now().isoformat()
        }
        
        # Save export if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export, f, indent=2)
                
        return export

    def reanalyze_content(self, content_ids: List[str], new_version: str = None) -> Dict[str, List[str]]:
        """
        Reanalyze existing content with updated algorithms
        
        Args:
            content_ids: IDs of content to reanalyze
            new_version: Version identifier for new analyses
            
        Returns:
            Dictionary mapping analysis types to new analysis IDs
        """
        # Initialize repository components if not already done
        if not hasattr(self, 'content_repository'):
            self.initialize_repository_components()
            
        self.logger.info(f"Starting reanalysis of content: {self.timestamp}")
        
        # Generate version string if not provided
        if new_version is None:
            new_version = f"v{self.timestamp}"
            
        # Process content if needed
        processed_ids = []
        for content_id in content_ids:
            # Check if we already have processed this content
            existing_analyses = self.analysis_registry.get_derived_analyses(content_id)
            processed_id = None
            
            for analysis in existing_analyses:
                if analysis.get("analysis_type") == "processed_content":
                    processed_id = analysis.get("analysis_id")
                    break
                    
            if processed_id:
                # Use existing processed content
                processed_ids.append(processed_id)
            else:
                # Process the content
                try:
                    processed_id = self.content_processor.process_content(content_id)
                    processed_ids.append(processed_id)
                except Exception as e:
                    self.logger.error(f"Error processing content {content_id}: {str(e)}")
        
        # Analyze processed content
        analysis_ids = {
            "content_analysis": [],
            "authoritarian_analysis": []
        }
        
        # Load processed content
        processed_items = []
        for proc_id in processed_ids:
            processed_content = self.content_processor.get_processed_content(proc_id)
            if processed_content:
                processed_items.append(processed_content)
                
        # Extract articles for analysis
        articles_for_analysis = [item["content"] for item in processed_items]
        
        # Perform analysis
        if articles_for_analysis:
            self.logger.info(f"Analyzing {len(articles_for_analysis)} processed items")
            analysis_result = self.analyzer.process({"articles": articles_for_analysis})
            analyses = analysis_result.get("analyses", [])
            auth_analyses = analysis_result.get("authoritarian_analyses", [])
            
            # Store analyses (similar to run_with_repository but with version info)
            analysis_dir = os.path.join(self.output_dir, "analyzed")
            
            for i, analysis in enumerate(analyses):
                if "article" in analysis:
                    article_slug = create_slug(analysis['article']['title'])
                    
                    # Generate analysis ID with version
                    analysis_id = f"analysis_{article_slug}_{new_version}_{self.timestamp}"
                    
                    # Save analysis
                    save_to_file(analysis, f"{analysis_dir}/{analysis_id}.json")
                    
                    # Register in registry with version
                    proc_id = processed_ids[i] if i < len(processed_ids) else None
                    
                    self.analysis_registry.register_analysis(
                        analysis_id=analysis_id,
                        analysis_type="content_analysis",
                        source_ids=[proc_id] if proc_id else [],
                        metadata={
                            "title": analysis["article"].get("title", ""),
                            "source": analysis["article"].get("source", ""),
                            "version": new_version,
                            "reanalysis": True,
                            "timestamp": analysis.get("timestamp", "")
                        },
                        version=new_version
                    )
                    
                    analysis_ids["content_analysis"].append(analysis_id)
                    
                    # Handle authoritarian analysis if available
                    if i < len(auth_analyses):
                        auth_analysis = auth_analyses[i]
                        
                        # Generate ID with version
                        auth_id = f"auth_analysis_{article_slug}_{new_version}_{self.timestamp}"
                        
                        # Save analysis
                        save_to_file(auth_analysis, f"{analysis_dir}/{auth_id}.json")
                        
                        # Register in registry
                        self.analysis_registry.register_analysis(
                            analysis_id=auth_id,
                            analysis_type="authoritarian_analysis",
                            source_ids=[proc_id, analysis_id] if proc_id else [analysis_id],
                            metadata={
                                "title": analysis["article"].get("title", ""),
                                "version": new_version,
                                "reanalysis": True,
                                "timestamp": auth_analysis.get("timestamp", "")
                            },
                            version=new_version
                        )
                        
                        analysis_ids["authoritarian_analysis"].append(auth_id)
        
        self.logger.info(f"Completed reanalysis with version {new_version}")
        return analysis_ids
