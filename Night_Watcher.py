#!/usr/bin/env python3
"""
Night_watcher Main Controller - Simplified Version
Streamlined entry point with reduced complexity.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

# Core imports
from collector import ContentCollector
from analyzer import ContentAnalyzer
from knowledge_graph import KnowledgeGraph
from vector_store import VectorStore
from document_repository import DocumentRepository
import providers

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class NightWatcher:
    """Simplified Night_watcher controller."""
    
    def __init__(self, config_path: str = "config.json", base_dir: str = "data"):
        self.config_path = config_path
        self.base_dir = base_dir
        self.logger = logging.getLogger("NightWatcher")
        
        # Load configuration
        self.config = self._load_config()
        self._setup_directories()
        
        # Initialize components
        self.document_repository = DocumentRepository(
            base_dir=f"{self.base_dir}/documents",
            dev_mode=True
        )
        
        self.collector = ContentCollector(
            config=self.config,
            document_repository=self.document_repository,
            base_dir=self.base_dir
        )
        
        self.llm_provider = providers.initialize_llm_provider(self.config)
        self.analyzer = ContentAnalyzer(self.llm_provider) if self.llm_provider else None
        
        self.knowledge_graph = KnowledgeGraph(
            graph_file=f"{self.base_dir}/knowledge_graph/graph.json",
            taxonomy_file="KG_Taxonomy.csv"
        )
        
        self.vector_store = VectorStore(
            base_dir=f"{self.base_dir}/vector_store"
        )
    
    def _load_config(self) -> Dict[str, Any]:
        """Load or create configuration."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        
        # Create default config
        config = {
            "content_collection": {
                "article_limit": 50,
                "sources": [
                    {
                        "url": "https://feeds.bbci.co.uk/news/world/us_and_canada/rss.xml",
                        "type": "rss",
                        "bias": "center",
                        "name": "BBC US & Canada",
                        "enabled": True
                    }
                ],
                "govt_keywords": [
                    "executive order", "administration", "white house", "president",
                    "congress", "senate", "supreme court", "federal", "government"
                ],
                "request_timeout": 45,
                "delay_between_requests": 2.0
            },
            "llm_provider": {
                "type": "lm_studio",
                "host": "http://localhost:1234"
            },
            "analysis": {
                "max_articles": 20
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return config
    
    def _setup_directories(self):
        """Create necessary directories."""
        dirs = ["collected", "analyzed", "documents", "knowledge_graph", "vector_store", "logs"]
        for d in dirs:
            os.makedirs(f"{self.base_dir}/{d}", exist_ok=True)
    
    def collect(self, mode: str = "auto") -> Dict[str, Any]:
        """Run content collection."""
        self.logger.info(f"Starting collection (mode: {mode})")
        return self.collector.collect_content(force_mode=mode if mode != "auto" else None)
    
    def analyze(self, max_articles: int = 20) -> Dict[str, Any]:
        """Run content analysis."""
        if not self.analyzer:
            raise Exception("Analyzer not available - check LLM provider")
        
        # Get unanalyzed documents
        all_docs = self.document_repository.list_documents()
        analyzed = self._get_analyzed_docs()
        unanalyzed = [d for d in all_docs if d not in analyzed][:max_articles]
        
        if not unanalyzed:
            return {"status": "no_new_documents", "analyzed": 0}
        
        # Prepare for analysis
        articles = []
        for doc_id in unanalyzed:
            content, metadata, _ = self.document_repository.get_document(doc_id)
            if content and metadata:
                articles.append({
                    "title": metadata.get("title", ""),
                    "content": content,
                    "url": metadata.get("url", ""),
                    "source": metadata.get("source", ""),
                    "bias_label": metadata.get("bias_label", ""),
                    "published": metadata.get("published"),
                    "document_id": doc_id
                })
        
        # Run analysis
        result = self.analyzer.process({"articles": articles, "document_ids": [a["document_id"] for a in articles]})
        
        # Save analyses with provenance
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, analysis in enumerate(result.get("analyses", [])):
            doc_id = articles[i]["document_id"]
            analysis_id = f"analysis_{doc_id}_{timestamp}"
            
            # Store analysis file
            filename = f"{analysis_id}.json"
            with open(f"{self.base_dir}/analyzed/{filename}", 'w') as f:
                json.dump(analysis, f, indent=2)
            
            # Store analysis provenance
            self.document_repository.store_analysis_provenance(
                analysis_id=analysis_id,
                document_ids=[doc_id],
                analysis_type="content_analysis",
                analysis_parameters={
                    "template": analysis.get("template_info", {}).get("file", "unknown"),
                    "max_articles": max_articles
                },
                results=analysis,
                analyzer_version=analysis.get("template_info", {}).get("version", "1.0")
            )
        
        return {"status": "completed", "analyzed": len(result.get("analyses", []))}
    
    def build_kg(self) -> Dict[str, Any]:
        """Build knowledge graph from analyses."""
        analyses_dir = f"{self.base_dir}/analyzed"
        if not os.path.exists(analyses_dir):
            return {"status": "no_analyses", "processed": 0}
        
        processed = 0
        for filename in os.listdir(analyses_dir):
            if filename.startswith("analysis_"):
                try:
                    with open(f"{analyses_dir}/{filename}", 'r') as f:
                        analysis = json.load(f)
                    
                    article = analysis.get("article", {})
                    if article and analysis.get("kg_payload"):
                        self.knowledge_graph.process_article_analysis(article, analysis)
                        processed += 1
                except Exception as e:
                    self.logger.error(f"Error processing {filename}: {e}")
        
        # Add temporal relationships
        temporal = self.knowledge_graph.infer_temporal_relationships()
        
        # Save graph
        self.knowledge_graph.save_graph()
        
        return {"status": "completed", "processed": processed, "temporal_relations": temporal}
    
    def sync_vectors(self) -> Dict[str, Any]:
        """Sync vector store with knowledge graph."""
        stats = self.vector_store.sync_with_knowledge_graph(self.knowledge_graph)
        return stats
    
    def _get_analyzed_docs(self) -> set:
        """Get set of analyzed document IDs."""
        analyzed = set()
        analyzed_dir = f"{self.base_dir}/analyzed"
        
        if os.path.exists(analyzed_dir):
            for filename in os.listdir(analyzed_dir):
                if filename.startswith("analysis_"):
                    try:
                        with open(f"{analyzed_dir}/{filename}", 'r') as f:
                            analysis = json.load(f)
                        doc_id = analysis.get("article", {}).get("document_id")
                        if doc_id:
                            analyzed.add(doc_id)
                    except:
                        continue
        
        return analyzed
    
    def status(self) -> Dict[str, Any]:
        """Get unified system status from all components."""
        # Get repository stats
        repo_stats = self.document_repository.get_statistics()
        
        # Get analyzed count
        analyzed = len(self._get_analyzed_docs())
        
        # Get KG stats
        kg_stats = self.knowledge_graph.get_basic_statistics()
        
        # Get vector stats
        vector_stats = self.vector_store.get_statistics()
        
        return {
            "documents": {
                "total": repo_stats["total_documents"],
                "analyzed": analyzed,
                "pending": repo_stats["total_documents"] - analyzed,
                "total_size_mb": repo_stats["total_size_mb"]
            },
            "analyses": {
                "total": repo_stats["total_analyses"]
            },
            "knowledge_graph": {
                "nodes": kg_stats["node_count"],
                "edges": kg_stats["edge_count"],
                "node_types": kg_stats["node_types"],
                "relation_types": kg_stats["relation_types"]
            },
            "vector_store": {
                "total_vectors": vector_stats["total_vectors"],
                "index_size_mb": vector_stats["index_size_mb"]
            },
            "system": {
                "llm_connected": self.llm_provider is not None,
                "base_dir": self.base_dir,
                "timestamp": datetime.now().isoformat()
            }
        }


def main():
    parser = argparse.ArgumentParser(description="Night_watcher - Political Intelligence Framework")
    parser.add_argument("--config", default="config.json", help="Config file")
    parser.add_argument("--base-dir", default="data", help="Data directory")
    
    # Commands
    parser.add_argument("--collect", action="store_true", help="Run collection")
    parser.add_argument("--analyze", action="store_true", help="Run analysis")
    parser.add_argument("--build-kg", action="store_true", help="Build knowledge graph")
    parser.add_argument("--sync-vectors", action="store_true", help="Sync vectors")
    parser.add_argument("--full", action="store_true", help="Run full pipeline")
    parser.add_argument("--status", action="store_true", help="Show status")
    
    # Options
    parser.add_argument("--mode", choices=["auto", "first_run", "incremental", "full"],
                       default="auto", help="Collection mode")
    parser.add_argument("--max-articles", type=int, default=20, help="Max articles to analyze")
    
    args = parser.parse_args()
    
    # Initialize
    nw = NightWatcher(config_path=args.config, base_dir=args.base_dir)
    
    try:
        if args.status:
            status = nw.status()
            print("\n=== Night_watcher Status ===")
            print(json.dumps(status, indent=2))
        
        elif args.collect:
            result = nw.collect(mode=args.mode)
            print(f"✓ Collected {len(result['articles'])} articles")
        
        elif args.analyze:
            result = nw.analyze(max_articles=args.max_articles)
            print(f"✓ Analyzed {result.get('analyzed', 0)} documents")
        
        elif args.build_kg:
            result = nw.build_kg()
            print(f"✓ Processed {result['processed']} analyses")
            print(f"✓ Added {result['temporal_relations']} temporal relations")
        
        elif args.sync_vectors:
            result = nw.sync_vectors()
            print(f"✓ Synced {result['nodes_added']} vectors")
        
        elif args.full:
            print("Running full pipeline...")
            
            # Collect
            collect_result = nw.collect()
            print(f"✓ Collected {len(collect_result['articles'])} articles")
            
            # Analyze
            if collect_result['articles']:
                analyze_result = nw.analyze()
                print(f"✓ Analyzed {analyze_result.get('analyzed', 0)} documents")
                
                # Build KG
                if analyze_result.get('analyzed', 0) > 0:
                    kg_result = nw.build_kg()
                    print(f"✓ Built knowledge graph")
                    
                    # Sync vectors
                    vector_result = nw.sync_vectors()
                    print(f"✓ Synced vectors")
        
        else:
            parser.print_help()
    
    except Exception as e:
        print(f"Error: {e}")
        if "--debug" in sys.argv:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Check if web server requested
    if "--web" in sys.argv:
        # Import and run web server
        try:
            from night_watcher_web import main as web_main
            sys.argv.remove("--web")  # Remove to avoid argument conflicts
            web_main()
        except ImportError:
            print("Error: night_watcher_web.py not found")
            print("Make sure night_watcher_web.py is in the same directory")
            sys.exit(1)
    else:
        main()
