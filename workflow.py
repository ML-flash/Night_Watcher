"""
Night_watcher Workflow Orchestrator
Manages the workflow for intelligence gathering and analysis.
"""

import os
import logging
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Set

from providers import LLMProvider
from memory import MemorySystem
from collector import ContentCollector
from analyzer import ContentAnalyzer
from knowledge_graph import KnowledgeGraph
from document_repository import DocumentRepository
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
        
        # Initialize document repository
        self.doc_repo = DocumentRepository(base_dir=os.path.join(output_dir, "documents"))

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
        
        # Store collected articles in document repository
        doc_ids = []
        for article in articles:
            # Extract content and metadata
            content = article.get("content", "")
            metadata = {
                "title": article.get("title", "Untitled"),
                "url": article.get("url", ""),
                "source": article.get("source", "Unknown"),
                "bias_label": article.get("bias_label", "unknown"),
                "published": article.get("published"),
                "tags": article.get("tags", []),
                "collected_at": article.get("collected_at", datetime.now().isoformat()),
                "collection_method": "rss"
            }
            
            # Store in document repository
            doc_id = self.doc_repo.store_document(content, metadata)
            doc_ids.append(doc_id)
            
            # Add document ID to article for reference
            article["document_id"] = doc_id
            
            # Also save collected articles in the traditional way for backwards compatibility
            filename = f"article_{doc_id[:8]}_{self.timestamp}.json"
            save_to_file(article, os.path.join(self.collected_dir, filename))
            
        self.logger.info(f"Collected and stored {len(articles)} articles with document IDs")
        
        # Skip analysis if no LLM available
        if not llm_available:
            self.logger.warning("LLM provider not available, skipping analysis")
            return {
                "articles_collected": len(articles),
                "articles_analyzed": 0,
                "pattern_analyses_generated": 0,
                "document_ids": doc_ids
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
            
            # Save analysis results and update with document provenance
            for i, analysis in enumerate(analyses):
                # Add document ID references if not already present
                if "article" in analysis and "document_id" not in analysis["article"]:
                    if i < len(doc_ids):
                        analysis["article"]["document_id"] = doc_ids[i]
                
                # Add document citation
                doc_id = analysis["article"].get("document_id")
                if doc_id:
                    analysis["document_citation"] = self.doc_repo.get_document_citation(doc_id)
                
                # Save to file
                filename = f"analysis_{doc_id[:8] if doc_id else i+1}_{self.timestamp}.json"
                save_to_file(analysis, os.path.join(self.analyzed_dir, filename))
                
            # Similarly update authoritarian analyses with document references
            for i, auth_analysis in enumerate(auth_analyses):
                doc_id = None
                if i < len(doc_ids):
                    doc_id = doc_ids[i]
                    auth_analysis["document_id"] = doc_id
                    auth_analysis["document_citation"] = self.doc_repo.get_document_citation(doc_id)
                
                filename = f"auth_analysis_{doc_id[:8] if doc_id else i+1}_{self.timestamp}.json"
                save_to_file(auth_analysis, os.path.join(self.analyzed_dir, filename))
                
            # Update KG analyses with document references
            for i, kg_analysis in enumerate(kg_analyses):
                doc_id = None
                if "article" in kg_analysis and i < len(doc_ids):
                    doc_id = doc_ids[i]
                    kg_analysis["article"]["document_id"] = doc_id
                
                filename = f"kg_analysis_{doc_id[:8] if doc_id else i+1}_{self.timestamp}.json"
                save_to_file(kg_analysis, os.path.join(self.analyzed_dir, filename))
            
            # Store in memory system
            stored_analyses = []
            for analysis in analyses:
                # Always include document ID reference
                if "article" in analysis and "document_id" in analysis["article"]:
                    if "metadata" not in analysis:
                        analysis["metadata"] = {}
                    analysis["metadata"]["document_id"] = analysis["article"]["document_id"]
                
                analysis_id = self.memory.store_article_analysis(analysis)
                if analysis_id:
                    stored_analyses.append(analysis_id)
                    
            # Process knowledge graph analyses
            for kg_analysis in kg_analyses:
                # Add article ID if not present
                article = kg_analysis["article"]
                if "id" not in article:
                    article["id"] = article.get("document_id") or f"article_{hash(article.get('title', ''))}"
                
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
                trends_filename = f"{self.analysis_dir}/authoritarian_trends_{self.timestamp}.json"
                save_to_file(auth_trends, trends_filename)

                # Display a summary of the authoritarian trends
                self._display_summary(auth_trends, "Authoritarian Trends")

                # Get influential actors
                influential_actors = self.knowledge_graph.get_influential_actors(10)
                actors_filename = f"{self.analysis_dir}/influential_actors_{self.timestamp}.json"
                save_to_file(influential_actors, actors_filename)

                # Display a summary of influential actors
                self._display_summary(influential_actors, "Influential Actors")

                # Get comprehensive democratic erosion analysis
                democratic_erosion = self.knowledge_graph.analyze_democratic_erosion(pattern_analysis_days)
                erosion_filename = f"{self.analysis_dir}/democratic_erosion_{self.timestamp}.json"
                save_to_file(democratic_erosion, erosion_filename)

                # Display a summary of democratic erosion
                self._display_summary(democratic_erosion, "Democratic Erosion Analysis")

                # Get actor coordination patterns
                coordination_patterns = self.knowledge_graph.detect_coordination_patterns(pattern_analysis_days)
                patterns_filename = f"{self.analysis_dir}/coordination_patterns_{self.timestamp}.json"
                save_to_file(coordination_patterns, patterns_filename)

                # Display a summary of coordination patterns
                self._display_summary(coordination_patterns, "Coordination Patterns")

                # Generate comprehensive intelligence report
                intel_report = self._generate_intelligence_report(
                    democratic_erosion,
                    influential_actors,
                    coordination_patterns,
                    pattern_analysis_days
                )
                
                # Add document provenance to intelligence report
                intel_report["document_sources"] = []
                for doc_id in doc_ids:
                    intel_report["document_sources"].append({
                        "document_id": doc_id,
                        "citation": self.doc_repo.get_document_citation(doc_id)
                    })
                
                report_filename = f"{self.analysis_dir}/intelligence_report_{self.timestamp}.json"
                save_to_file(intel_report, report_filename)

                # Display a summary of the intelligence report
                self._display_summary(intel_report, "Intelligence Report")

                patterns_generated = 5

                # Print a more detailed summary
                print("\n=== Pattern Analysis Results ===")
                print(f"Time period analyzed: Last {pattern_analysis_days} days")
                print(f"Authoritarian Trends Score: {auth_trends.get('trend_score', 'N/A')}/10")
                print(f"Democratic Erosion Score: {democratic_erosion.get('erosion_score', 'N/A')}/10")
                print(f"Risk Level: {democratic_erosion.get('risk_level', 'N/A')}")
                print(f"Affected Institutions: {len(auth_trends.get('affected_institutions', []))}")
                print(f"Coordination Patterns: {len(coordination_patterns)}")
                print(f"Top Actors: {', '.join([actor.get('name', 'Unknown') for actor in influential_actors[:3]])}")
                print(f"\nAll analysis files saved to: {self.analysis_dir}")
                print(f"Document repository: {self.doc_repo.base_dir}")
                print(f"Document IDs: {', '.join([doc_id[:8] + '...' for doc_id in doc_ids])}")
            else:
                self.logger.warning("No recent analyses found for pattern recognition")
                patterns_generated = 0

            return {
                "articles_collected": len(articles),
                "articles_analyzed": len(analyses),
                "knowledge_graph_analyses": len(kg_analyses),
                "pattern_analyses_generated": patterns_generated,
                "document_ids": doc_ids
            }
        else:
            self.logger.warning("No articles collected, skipping analysis")
            return {
                "articles_collected": 0,
                "articles_analyzed": 0,
                "pattern_analyses_generated": 0,
                "document_ids": []
            }

    def _display_summary(self, data: Dict[str, Any], title: str) -> None:
        """Display a summary of analysis results"""
        self.logger.info(f"Generated {title}")

        # Extract key information based on the analysis type
        if title == "Authoritarian Trends":
            self.logger.info(f"Trend Score: {data.get('trend_score', 'N/A')}/10")
            self.logger.info(f"Affected Institutions: {len(data.get('affected_institutions', []))}")

        elif title == "Influential Actors":
            for i, actor in enumerate(data[:3], 1):
                self.logger.info(f"{i}. {actor.get('name', 'Unknown')} - Influence: {actor.get('influence_score', 'N/A')}")

        elif title == "Democratic Erosion Analysis":
            self.logger.info(f"Erosion Score: {data.get('erosion_score', 'N/A')}/10")
            self.logger.info(f"Risk Level: {data.get('risk_level', 'N/A')}")

        elif title == "Coordination Patterns":
            self.logger.info(f"Patterns Detected: {len(data)}")

        elif title == "Intelligence Report":
            self.logger.info(f"Report Generated: {data.get('title', 'Untitled')}")
            if "recommendations" in data and data["recommendations"]:
                self.logger.info(f"Top Recommendation: {data['recommendations'][0]}")
            if "document_sources" in data:
                self.logger.info(f"Sources: {len(data['document_sources'])} documents")
                for i, source in enumerate(data['document_sources'][:3], 1):
                    self.logger.info(f"  {i}. {source.get('citation', 'Unknown source')}")
