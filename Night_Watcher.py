#!/usr/bin/env python3
"""
Night_watcher Unified Main Controller
Single entry point for all Night_watcher operations with web API integration.
Replaces individual component scripts with a unified, streamlined interface.
"""

import os
import sys
import json
import logging
import argparse
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
import traceback
import uuid
from pathlib import Path

# Flask for web API
try:
    from flask import Flask, jsonify, request, render_template_string
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not available. Install with: pip install flask flask-cors")

# Import all Night_watcher components
try:
    from enhanced_collector import HistoricalContentCollector, create_historical_collection_config
    from analyzer import ContentAnalyzer
    from knowledge_graph import KnowledgeGraph
    from vector_store import VectorStore
    from kg_vector_integration import KGVectorIntegration
    from document_repository import DocumentRepository
    from analysis_provenance import AnalysisProvenanceTracker
    import providers
except ImportError as e:
    print(f"Night_watcher components not available: {e}")
    print("Ensure all Night_watcher modules are in the Python path")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@dataclass
class SystemStatus:
    """System status tracking"""
    collector_status: str = "idle"
    analyzer_status: str = "idle" 
    kg_status: str = "idle"
    vector_status: str = "idle"
    last_collection: Optional[str] = None
    last_analysis: Optional[str] = None
    articles_today: int = 0
    total_articles: int = 0
    total_nodes: int = 0
    total_edges: int = 0
    system_uptime: Optional[str] = None

@dataclass
class TaskResult:
    """Task execution result"""
    task_id: str
    task_type: str
    status: str  # "running", "completed", "failed"
    start_time: str
    end_time: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: int = 0

class NightWatcherMain:
    """
    Unified Night_watcher controller that manages all components
    and provides web API interface.
    """
    
    def __init__(self, config_path: str = "config.json", base_dir: str = "data"):
        """Initialize the unified Night_watcher system."""
        self.config_path = config_path
        self.base_dir = base_dir
        self.logger = logging.getLogger("NightWatcherMain")
        
        # System state
        self.status = SystemStatus()
        self.active_tasks: Dict[str, TaskResult] = {}
        self.system_start_time = datetime.now()
        
        # Component instances
        self.collector = None
        self.analyzer = None
        self.knowledge_graph = None
        self.vector_store = None
        self.kg_vector_integration = None
        self.document_repository = None
        self.provenance_tracker = None
        
        # Web API
        self.app = None
        self.api_thread = None
        
        # Initialize system
        self._load_config()
        self._setup_directories()
        self._initialize_components()
        
        if FLASK_AVAILABLE:
            self._setup_web_api()
    
    def _load_config(self):
        """Load configuration file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                self.logger.info(f"Loaded configuration from {self.config_path}")
            else:
                # Create default configuration
                self.config = self._create_default_config()
                self._save_config()
                self.logger.info(f"Created default configuration at {self.config_path}")
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            self.config = self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration."""
        return {
            "content_collection": {
                "article_limit": 50,
                "sources": [
                    {
                        "url": "https://feeds.bbci.co.uk/news/world/us_and_canada/rss.xml",
                        "type": "rss",
                        "bias": "center",
                        "name": "BBC US & Canada",
                        "enabled": True
                    },
                    {
                        "url": "https://www.federalregister.gov/presidential-documents.rss",
                        "type": "rss", 
                        "bias": "official-government",
                        "name": "Federal Register",
                        "enabled": True
                    },
                    {
                        "url": "https://feeds.foxnews.com/foxnews/politics",
                        "type": "rss",
                        "bias": "center-right", 
                        "name": "Fox News Politics",
                        "enabled": True
                    }
                ],
                "govt_keywords": [
                    "executive order", "administration", "white house", "president",
                    "congress", "senate", "house of representatives", "supreme court",
                    "federal", "government", "politics", "election", "democracy"
                ],
                "max_workers": 3,
                "request_timeout": 45,
                "delay_between_requests": 2.0,
                "use_llm_navigation_fallback": True
            },
            "llm_provider": {
                "type": "lm_studio",
                "host": "http://localhost:1234",
                "model": "default"
            },
            "analysis": {
                "max_articles": 20,
                "include_kg": True
            },
            "knowledge_graph": {
                "graph_file": f"{self.base_dir}/knowledge_graph/graph.json",
                "taxonomy_file": "KG_Taxonomy.csv"
            },
            "vector_store": {
                "base_dir": f"{self.base_dir}/vector_store",
                "embedding_provider": "local",
                "embedding_dim": 384,
                "index_type": "flat"
            },
            "output": {
                "base_dir": self.base_dir,
                "save_collected": True
            },
            "provenance": {
                "enabled": True,
                "dev_mode": True,
                "verify": True
            },
            "logging": {
                "level": "INFO",
                "log_dir": "logs"
            },
            "web_api": {
                "enabled": True,
                "host": "0.0.0.0",
                "port": 5000,
                "debug": False
            }
        }
    
    def _save_config(self):
        """Save configuration to file."""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
    
    def _setup_directories(self):
        """Create necessary directories."""
        directories = [
            self.base_dir,
            f"{self.base_dir}/collected",
            f"{self.base_dir}/analyzed", 
            f"{self.base_dir}/documents",
            f"{self.base_dir}/knowledge_graph",
            f"{self.base_dir}/vector_store",
            f"{self.base_dir}/analysis",
            f"{self.base_dir}/logs"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _initialize_components(self):
        """Initialize all Night_watcher components."""
        try:
            # Document Repository
            self.document_repository = DocumentRepository(
                base_dir=f"{self.base_dir}/documents",
                dev_mode=self.config.get("provenance", {}).get("dev_mode", True)
            )
            
            # Content Collector
            self.collector = HistoricalContentCollector(
                config=self.config,
                document_repository=self.document_repository,
                base_dir=self.base_dir
            )
            
            # LLM Provider
            self.llm_provider = providers.initialize_llm_provider(self.config)
            
            # Content Analyzer
            if self.llm_provider:
                self.analyzer = ContentAnalyzer(self.llm_provider)
            
            # Provenance Tracker
            if self.config.get("provenance", {}).get("enabled", True):
                self.provenance_tracker = AnalysisProvenanceTracker(
                    base_dir=f"{self.base_dir}/analysis_provenance",
                    dev_mode=self.config.get("provenance", {}).get("dev_mode", True)
                )
            
            # Knowledge Graph
            kg_config = self.config.get("knowledge_graph", {})
            self.knowledge_graph = KnowledgeGraph(
                graph_file=kg_config.get("graph_file", f"{self.base_dir}/knowledge_graph/graph.json"),
                taxonomy_file=kg_config.get("taxonomy_file", "KG_Taxonomy.csv")
            )
            
            # Vector Store
            vs_config = self.config.get("vector_store", {})
            self.vector_store = VectorStore(
                base_dir=vs_config.get("base_dir", f"{self.base_dir}/vector_store"),
                embedding_provider=vs_config.get("embedding_provider", "local"),
                embedding_dim=vs_config.get("embedding_dim", 384),
                index_type=vs_config.get("index_type", "flat"),
                kg_instance=self.knowledge_graph
            )
            
            # KG-Vector Integration
            self.kg_vector_integration = KGVectorIntegration(
                kg=self.knowledge_graph,
                vector_store=self.vector_store,
                enable_auto_sync=False
            )
            
            # Update system status
            self._update_system_status()
            
            self.logger.info("All Night_watcher components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            raise
    
    def _update_system_status(self):
        """Update system status information."""
        try:
            # Basic stats
            if self.knowledge_graph:
                self.status.total_nodes = len(self.knowledge_graph.graph.nodes)
                self.status.total_edges = len(self.knowledge_graph.graph.edges)
            
            if self.document_repository:
                docs = self.document_repository.list_documents()
                self.status.total_articles = len(docs)
                
                # Count today's articles
                today = datetime.now().date()
                today_count = 0
                for doc_id in docs:
                    try:
                        _, metadata, _ = self.document_repository.get_document(doc_id, verify=False)
                        if metadata and metadata.get("collected_at"):
                            collected_date = datetime.fromisoformat(metadata["collected_at"]).date()
                            if collected_date == today:
                                today_count += 1
                    except:
                        continue
                self.status.articles_today = today_count
            
            # System uptime
            uptime = datetime.now() - self.system_start_time
            self.status.system_uptime = str(uptime).split('.')[0]  # Remove microseconds
            
        except Exception as e:
            self.logger.error(f"Error updating system status: {e}")
    
    def _setup_web_api(self):
        """Setup Flask web API."""
        if not FLASK_AVAILABLE:
            return
            
        self.app = Flask(__name__)
        CORS(self.app)
        
        # API Routes
        @self.app.route('/api/status')
        def api_status():
            self._update_system_status()
            return jsonify(asdict(self.status))
        
        @self.app.route('/api/config')
        def api_config():
            return jsonify(self.config)
        
        @self.app.route('/api/config', methods=['POST'])
        def api_update_config():
            try:
                new_config = request.json
                self.config.update(new_config)
                self._save_config()
                return jsonify({"status": "success"})
            except Exception as e:
                return jsonify({"status": "error", "message": str(e)}), 400
        
        @self.app.route('/api/sources')
        def api_sources():
            return jsonify(self.config.get("content_collection", {}).get("sources", []))
        
        @self.app.route('/api/sources', methods=['POST'])
        def api_add_source():
            try:
                source_data = request.json
                success = self.collector.add_source(source_data)
                return jsonify({"status": "success" if success else "error"})
            except Exception as e:
                return jsonify({"status": "error", "message": str(e)}), 400
        
        @self.app.route('/api/collect', methods=['POST'])
        def api_collect():
            try:
                params = request.json or {}
                task_id = self._start_task("collection", self._run_collection, params)
                return jsonify({"task_id": task_id, "status": "started"})
            except Exception as e:
                return jsonify({"status": "error", "message": str(e)}), 400
        
        @self.app.route('/api/analyze', methods=['POST'])
        def api_analyze():
            try:
                params = request.json or {}
                task_id = self._start_task("analysis", self._run_analysis, params)
                return jsonify({"task_id": task_id, "status": "started"})
            except Exception as e:
                return jsonify({"status": "error", "message": str(e)}), 400
        
        @self.app.route('/api/build-kg', methods=['POST'])
        def api_build_kg():
            try:
                task_id = self._start_task("knowledge_graph", self._run_kg_build)
                return jsonify({"task_id": task_id, "status": "started"})
            except Exception as e:
                return jsonify({"status": "error", "message": str(e)}), 400
        
        @self.app.route('/api/sync-vectors', methods=['POST'])
        def api_sync_vectors():
            try:
                task_id = self._start_task("vector_sync", self._run_vector_sync)
                return jsonify({"task_id": task_id, "status": "started"})
            except Exception as e:
                return jsonify({"status": "error", "message": str(e)}), 400
        
        @self.app.route('/api/tasks/<task_id>')
        def api_task_status(task_id):
            if task_id in self.active_tasks:
                return jsonify(asdict(self.active_tasks[task_id]))
            else:
                return jsonify({"status": "not_found"}), 404
        
        @self.app.route('/api/tasks')
        def api_all_tasks():
            return jsonify({tid: asdict(task) for tid, task in self.active_tasks.items()})
        
        # Dashboard route
        @self.app.route('/')
        def dashboard():
            # Return the dashboard HTML (you can embed it here or serve from file)
            return render_template_string("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Night_watcher Dashboard</title>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
            </head>
            <body>
                <h1>Night_watcher Dashboard</h1>
                <p>Access the full dashboard at <a href="/dashboard.html">/dashboard.html</a></p>
                <p>API available at <a href="/api/status">/api/status</a></p>
            </body>
            </html>
            """)
    
    def _start_task(self, task_type: str, task_func, params: Dict[str, Any] = None) -> str:
        """Start a background task."""
        task_id = str(uuid.uuid4())
        task = TaskResult(
            task_id=task_id,
            task_type=task_type,
            status="running",
            start_time=datetime.now().isoformat()
        )
        
        self.active_tasks[task_id] = task
        
        # Start task in background thread
        def run_task():
            try:
                result = task_func(params or {})
                task.result = result
                task.status = "completed"
                task.progress = 100
            except Exception as e:
                self.logger.error(f"Task {task_id} failed: {e}")
                task.error = str(e)
                task.status = "failed"
            finally:
                task.end_time = datetime.now().isoformat()
        
        thread = threading.Thread(target=run_task)
        thread.daemon = True
        thread.start()
        
        return task_id
    
    def _run_collection(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run content collection."""
        self.status.collector_status = "running"
        
        try:
            force_mode = params.get("mode")
            reset_date = params.get("reset_date", False)
            
            if reset_date and os.path.exists(self.collector.last_run_file):
                os.remove(self.collector.last_run_file)
                self.logger.info("Reset collection date tracking")
            
            results = self.collector.collect_content(force_mode=force_mode)
            
            self.status.last_collection = datetime.now().isoformat()
            self.status.collector_status = "idle"
            
            return results
            
        except Exception as e:
            self.status.collector_status = "error"
            raise
    
    def _run_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run content analysis."""
        if not self.analyzer:
            raise Exception("Analyzer not available - LLM provider not initialized")
        
        self.status.analyzer_status = "running"
        
        try:
            max_articles = params.get("max_articles", 20)
            
            # Get unanalyzed documents
            doc_ids = self.document_repository.list_documents()
            analyzed_dir = f"{self.base_dir}/analyzed"
            
            # Find analyzed documents
            analyzed_files = []
            if os.path.exists(analyzed_dir):
                analyzed_files = [f for f in os.listdir(analyzed_dir) if f.startswith("analysis_")]
            
            analyzed_doc_ids = set()
            for filename in analyzed_files:
                try:
                    with open(os.path.join(analyzed_dir, filename), 'r') as f:
                        analysis = json.load(f)
                        doc_id = analysis.get("article", {}).get("document_id")
                        if doc_id:
                            analyzed_doc_ids.add(doc_id)
                except:
                    continue
            
            # Get unanalyzed documents
            unanalyzed_ids = [doc_id for doc_id in doc_ids if doc_id not in analyzed_doc_ids][:max_articles]
            
            if not unanalyzed_ids:
                return {"status": "no_new_documents", "analyzed": 0}
            
            # Prepare articles for analysis
            articles = []
            document_ids = []
            
            for doc_id in unanalyzed_ids:
                try:
                    content, metadata, verified = self.document_repository.get_document(doc_id)
                    if content and metadata:
                        article = {
                            "title": metadata.get("title", "Untitled"),
                            "content": content,
                            "url": metadata.get("url", ""),
                            "source": metadata.get("source", "Unknown"),
                            "bias_label": metadata.get("bias_label", "unknown"),
                            "published": metadata.get("published"),
                            "document_id": doc_id
                        }
                        articles.append(article)
                        document_ids.append(doc_id)
                except Exception as e:
                    self.logger.error(f"Error loading document {doc_id}: {e}")
                    continue
            
            if not articles:
                return {"status": "no_valid_documents", "analyzed": 0}
            
            # Run analysis
            analysis_input = {
                "articles": articles,
                "document_ids": document_ids
            }
            
            result = self.analyzer.process(analysis_input)
            analyses = result.get("analyses", [])
            
            # Save analysis results
            os.makedirs(analyzed_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for i, analysis in enumerate(analyses):
                article_info = analysis.get("article", {})
                doc_id = article_info.get("document_id", f"unknown_{i}")
                
                # Create provenance record if enabled
                if self.provenance_tracker:
                    try:
                        provenance_id = f"analysis_{doc_id}_{timestamp}"
                        self.provenance_tracker.create_analysis_record(
                            analysis_id=provenance_id,
                            document_ids=[doc_id],
                            analysis_type="content_analysis",
                            analysis_parameters={"timestamp": timestamp},
                            results=analysis,
                            analyzer_version="1.0.0"
                        )
                        analysis["provenance_id"] = provenance_id
                    except Exception as e:
                        self.logger.error(f"Error creating provenance record: {e}")
                
                # Save analysis
                filename = f"analysis_{doc_id}_{timestamp}.json"
                filepath = os.path.join(analyzed_dir, filename)
                
                try:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(analysis, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    self.logger.error(f"Error saving analysis: {e}")
            
            self.status.last_analysis = datetime.now().isoformat()
            self.status.analyzer_status = "idle"
            
            return {
                "status": "completed",
                "analyzed": len(analyses),
                "total_documents": len(articles)
            }
            
        except Exception as e:
            self.status.analyzer_status = "error"
            raise
    
    def _run_kg_build(self, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run knowledge graph building."""
        self.status.kg_status = "running"
        
        try:
            analyzed_dir = f"{self.base_dir}/analyzed"
            
            if not os.path.exists(analyzed_dir):
                return {"status": "no_analyses_found", "processed": 0}
            
            # Find analysis files
            analysis_files = [f for f in os.listdir(analyzed_dir) if f.startswith("analysis_")]
            
            if not analysis_files:
                return {"status": "no_analyses_found", "processed": 0}
            
            processed_count = 0
            initial_nodes = len(self.knowledge_graph.graph.nodes)
            initial_edges = len(self.knowledge_graph.graph.edges)
            
            for filename in analysis_files:
                try:
                    filepath = os.path.join(analyzed_dir, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        analysis = json.load(f)
                    
                    # Process analysis
                    article = analysis.get("article", {})
                    kg_payload = analysis.get("kg_payload", {})
                    
                    if article and kg_payload:
                        result = self.knowledge_graph.process_article_analysis(article, analysis)
                        if result.get("status") == "success":
                            processed_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Error processing analysis {filename}: {e}")
                    continue
            
            # Infer temporal relationships
            temporal_relations = self.knowledge_graph.infer_temporal_relationships()
            
            # Save graph
            self.knowledge_graph.save_graph()
            
            final_nodes = len(self.knowledge_graph.graph.nodes)
            final_edges = len(self.knowledge_graph.graph.edges)
            
            self.status.kg_status = "idle"
            
            return {
                "status": "completed",
                "processed_analyses": processed_count,
                "nodes_added": final_nodes - initial_nodes,
                "edges_added": final_edges - initial_edges,
                "temporal_relations": temporal_relations
            }
            
        except Exception as e:
            self.status.kg_status = "error"
            raise
    
    def _run_vector_sync(self, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run vector store synchronization."""
        self.status.vector_status = "running"
        
        try:
            # Sync vector store with knowledge graph
            stats = self.kg_vector_integration.sync()
            
            # Save vector store
            self.vector_store.save()
            
            self.status.vector_status = "idle"
            
            return {
                "status": "completed",
                "nodes_added": stats.get("nodes_added", 0),
                "total_nodes": stats.get("nodes_total", 0)
            }
            
        except Exception as e:
            self.status.vector_status = "error"
            raise
    
    def start_web_server(self, host: str = None, port: int = None, debug: bool = False):
        """Start the web server."""
        if not FLASK_AVAILABLE or not self.app:
            self.logger.error("Flask not available - cannot start web server")
            return
        
        web_config = self.config.get("web_api", {})
        host = host or web_config.get("host", "0.0.0.0")
        port = port or web_config.get("port", 5000)
        debug = debug or web_config.get("debug", False)
        
        self.logger.info(f"Starting web server on {host}:{port}")
        
        def run_server():
            self.app.run(host=host, port=port, debug=debug, use_reloader=False)
        
        self.api_thread = threading.Thread(target=run_server)
        self.api_thread.daemon = True
        self.api_thread.start()
    
    def run_full_pipeline(self, collection_mode: str = "auto") -> Dict[str, Any]:
        """Run the complete Night_watcher pipeline."""
        self.logger.info("Starting full Night_watcher pipeline")
        
        results = {
            "collection": None,
            "analysis": None,
            "knowledge_graph": None,
            "vector_sync": None,
            "overall_status": "running",
            "start_time": datetime.now().isoformat()
        }
        
        try:
            # Step 1: Content Collection
            self.logger.info("Step 1: Running content collection")
            collection_results = self._run_collection({"mode": collection_mode})
            results["collection"] = collection_results
            
            articles_collected = len(collection_results.get("articles", []))
            self.logger.info(f"Collected {articles_collected} articles")
            
            if articles_collected == 0:
                results["overall_status"] = "completed_no_new_content"
                return results
            
            # Step 2: Content Analysis
            self.logger.info("Step 2: Running content analysis")
            analysis_results = self._run_analysis({"max_articles": articles_collected})
            results["analysis"] = analysis_results
            
            analyses_completed = analysis_results.get("analyzed", 0)
            self.logger.info(f"Analyzed {analyses_completed} articles")
            
            if analyses_completed == 0:
                results["overall_status"] = "completed_no_analysis"
                return results
            
            # Step 3: Knowledge Graph Building
            self.logger.info("Step 3: Building knowledge graph")
            kg_results = self._run_kg_build()
            results["knowledge_graph"] = kg_results
            
            nodes_added = kg_results.get("nodes_added", 0)
            edges_added = kg_results.get("edges_added", 0)
            self.logger.info(f"Added {nodes_added} nodes and {edges_added} edges to knowledge graph")
            
            # Step 4: Vector Store Sync
            self.logger.info("Step 4: Synchronizing vector store")
            vector_results = self._run_vector_sync()
            results["vector_sync"] = vector_results
            
            vectors_added = vector_results.get("nodes_added", 0)
            self.logger.info(f"Added {vectors_added} vectors to vector store")
            
            results["overall_status"] = "completed_successfully"
            results["end_time"] = datetime.now().isoformat()
            
            self.logger.info("Full Night_watcher pipeline completed successfully")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            results["overall_status"] = "failed"
            results["error"] = str(e)
            results["end_time"] = datetime.now().isoformat()
        
        return results
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        self._update_system_status()
        
        # Component status
        components = {
            "collector": "available" if self.collector else "unavailable",
            "analyzer": "available" if self.analyzer else "unavailable",
            "knowledge_graph": "available" if self.knowledge_graph else "unavailable",
            "vector_store": "available" if self.vector_store else "unavailable",
            "document_repository": "available" if self.document_repository else "unavailable",
            "provenance_tracker": "available" if self.provenance_tracker else "unavailable",
            "llm_provider": "available" if self.llm_provider else "unavailable"
        }
        
        # Statistics
        stats = {}
        if self.document_repository:
            stats.update(self.document_repository.get_statistics())
        
        if self.knowledge_graph:
            stats.update(self.knowledge_graph.get_basic_statistics())
        
        if self.vector_store:
            stats.update(self.vector_store.get_statistics())
        
        return {
            "status": asdict(self.status),
            "components": components,
            "statistics": stats,
            "active_tasks": len(self.active_tasks),
            "config_file": self.config_path,
            "base_directory": self.base_dir
        }


def main():
    """Main entry point for Night_watcher."""
    parser = argparse.ArgumentParser(description="Night_watcher Unified Main Controller")
    parser.add_argument("--config", default="config.json", help="Configuration file path")
    parser.add_argument("--base-dir", default="data", help="Base data directory")
    
    # Operation modes
    parser.add_argument("--web", action="store_true", help="Start web server")
    parser.add_argument("--collect", action="store_true", help="Run content collection")
    parser.add_argument("--analyze", action="store_true", help="Run content analysis")
    parser.add_argument("--build-kg", action="store_true", help="Build knowledge graph")
    parser.add_argument("--sync-vectors", action="store_true", help="Sync vector store")
    parser.add_argument("--full-pipeline", action="store_true", help="Run full pipeline")
    
    # Collection options
    parser.add_argument("--collection-mode", choices=["auto", "first_run", "incremental", "full"],
                       default="auto", help="Collection mode")
    parser.add_argument("--reset-date", action="store_true", help="Reset collection date")
    
    # Web server options
    parser.add_argument("--host", default="0.0.0.0", help="Web server host")
    parser.add_argument("--port", type=int, default=5000, help="Web server port")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    # Source management
    parser.add_argument("--add-source", help="Add source: 'url,type,bias,name'")
    parser.add_argument("--add-article", help="Add direct article URL")
    parser.add_argument("--list-sources", action="store_true", help="List all sources")
    
    # System info
    parser.add_argument("--status", action="store_true", help="Show system status")
    parser.add_argument("--info", action="store_true", help="Show system info")
    
    args = parser.parse_args()
    
    try:
        # Initialize Night_watcher
        nw = NightWatcherMain(config_path=args.config, base_dir=args.base_dir)
        
        # Handle source management
        if args.add_source:
            parts = args.add_source.split(',')
            if len(parts) >= 2:
                source_data = {
                    "url": parts[0].strip(),
                    "type": parts[1].strip() if len(parts) > 1 else "auto",
                    "bias": parts[2].strip() if len(parts) > 2 else "unknown",
                    "name": parts[3].strip() if len(parts) > 3 else None
                }
                if nw.collector.add_source(source_data):
                    print(f"✓ Added source: {source_data['url']}")
                else:
                    print(f"✗ Failed to add source")
                    sys.exit(1)
            else:
                print("Error: Format should be 'url,type,bias,name'")
                sys.exit(1)
        
        elif args.add_article:
            source_data = {
                "url": args.add_article,
                "type": "article",
                "bias": "unknown"
            }
            if nw.collector.add_source(source_data):
                print(f"✓ Added article: {args.add_article}")
            else:
                print(f"✗ Failed to add article")
                sys.exit(1)
        
        elif args.list_sources:
            sources = nw.config.get("content_collection", {}).get("sources", [])
            print(f"\n=== Configured Sources ({len(sources)}) ===")
            for i, source in enumerate(sources, 1):
                status = "✓" if source.get("enabled", True) else "✗"
                print(f"{i}. {status} {source.get('name', 'Unknown')} ({source.get('type', 'unknown')})")
                print(f"   URL: {source.get('url', 'N/A')}")
                print(f"   Bias: {source.get('bias', 'unknown')}")
                print()
        
        # Handle operations
        elif args.status:
            info = nw.get_system_info()
            status = info["status"]
            print(f"\n=== Night_watcher System Status ===")
            print(f"Collector: {status['collector_status']}")
            print(f"Analyzer: {status['analyzer_status']}")
            print(f"Knowledge Graph: {status['kg_status']}")
            print(f"Vector Store: {status['vector_status']}")
            print(f"Articles Today: {status['articles_today']}")
            print(f"Total Articles: {status['total_articles']}")
            print(f"Total Nodes: {status['total_nodes']}")
            print(f"Total Edges: {status['total_edges']}")
            print(f"Uptime: {status['system_uptime']}")
        
        elif args.info:
            info = nw.get_system_info()
            print(f"\n=== Night_watcher System Information ===")
            print(json.dumps(info, indent=2))
        
        elif args.collect:
            print("Starting content collection...")
            params = {"mode": args.collection_mode, "reset_date": args.reset_date}
            results = nw._run_collection(params)
            print(f"✓ Collection completed: {len(results['articles'])} articles collected")
        
        elif args.analyze:
            print("Starting content analysis...")
            results = nw._run_analysis({})
            print(f"✓ Analysis completed: {results.get('analyzed', 0)} articles analyzed")
        
        elif args.build_kg:
            print("Building knowledge graph...")
            results = nw._run_kg_build()
            print(f"✓ Knowledge graph built: {results.get('nodes_added', 0)} nodes, {results.get('edges_added', 0)} edges added")
        
        elif args.sync_vectors:
            print("Synchronizing vector store...")
            results = nw._run_vector_sync()
            print(f"✓ Vector store synchronized: {results.get('nodes_added', 0)} vectors added")
        
        elif args.full_pipeline:
            print("Running full Night_watcher pipeline...")
            results = nw.run_full_pipeline(collection_mode=args.collection_mode)
            print(f"✓ Pipeline completed with status: {results['overall_status']}")
        
        elif args.web:
            print("Starting Night_watcher web server...")
            nw.start_web_server(host=args.host, port=args.port, debug=args.debug)
            print(f"Web server started at http://{args.host}:{args.port}")
            print("Press Ctrl+C to stop...")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nShutting down...")
        
        else:
            # Default: show help
            parser.print_help()
    
    except Exception as e:
        print(f"Error: {e}")
        if args.debug:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
