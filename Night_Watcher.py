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
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

# Core imports
from collector import ContentCollector
from analyzer import ContentAnalyzer
from knowledge_graph import KnowledgeGraph
from vector_store import VectorStore
from document_repository import DocumentRepository
from document_aggregator import aggregate_document_analyses
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

        # Initialize event tracking
        try:
            from event_weight import EventManager
            from event_database import EventDatabase
            event_db_path = os.path.join(self.base_dir, "events.db")
            self.event_manager = EventManager(EventDatabase(event_db_path), self.llm_provider)
            self.logger.info("Event tracking system initialized")
        except Exception as e:
            self.logger.warning(f"Event tracking not available: {e}")
            self.event_manager = None
    
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
                        "enabled": True,
                        "limit": 50
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
    
    def collect(self, mode: str = "auto", callback=None) -> Dict[str, Any]:
        """Run content collection."""
        self.logger.info(f"Starting collection (mode: {mode})")
        return self.collector.collect_content(force_mode=mode if mode != "auto" else None, callback=callback)
    
    def analyze(self, max_articles: int = 20, templates: List[str] = None, 
                target: str = "unanalyzed", since_date: str = None) -> Dict[str, Any]:
        """Run content analysis with multiple templates.
        
        Args:
            max_articles: Maximum articles to analyze
            templates: List of template files to use
            target: "unanalyzed" (default), "all", or "recent" 
            since_date: ISO date string for custom date range (when target="all")
        """
        if not self.analyzer:
            raise Exception("Analyzer not available - check LLM provider")
        
        templates = templates or ["standard_analysis.json"]
        
        # Get documents based on target
        if target == "unanalyzed":
            all_docs = self.document_repository.list_documents()
            analyzed = self._get_analyzed_docs()
            target_docs = [d for d in all_docs if d not in analyzed][:max_articles]
        elif target == "recent":
            # Get documents from last collection run
            last_run_file = os.path.join(self.base_dir, "last_run_date.txt")
            if os.path.exists(last_run_file):
                with open(last_run_file, 'r') as f:
                    last_run = datetime.fromisoformat(f.read().strip())
                target_docs = self._get_docs_since(last_run)[:max_articles]
            else:
                target_docs = []
        elif target == "all" and since_date:
            # Get all docs since specified date
            since = datetime.fromisoformat(since_date)
            target_docs = self._get_docs_since(since)[:max_articles]
        else:
            # Default to unanalyzed
            all_docs = self.document_repository.list_documents()
            analyzed = self._get_analyzed_docs()
            target_docs = [d for d in all_docs if d not in analyzed][:max_articles]
        
        if not target_docs:
            return {"status": "no_documents", "analyzed": 0}

        articles = []
        for doc_id in target_docs:
            content, metadata, _ = self.document_repository.get_document(doc_id)
            if content and metadata:
                articles.append({
                    "id": doc_id,
                    "title": metadata.get("title", ""),
                    "content": content,
                    "url": metadata.get("url", ""),
                    "source": metadata.get("source", ""),
                    "bias_label": metadata.get("bias_label", ""),
                    "published": metadata.get("published")
                })

        analyzed_count = 0
        results = []

        for doc in articles:
            try:
                doc_analyses = []
                for template in templates:
                    self.analyzer.template = self.analyzer._load_template(template)
                    analysis = self.analyzer.analyze_article({
                        "content": doc["content"],
                        "title": doc["title"],
                        "url": doc["url"],
                        "source": doc["source"],
                        "published": doc["published"],
                        "document_id": doc["id"]
                    })
                    analysis["template_used"] = template
                    doc_analyses.append(analysis)

                if len(doc_analyses) > 1:
                    aggregated = aggregate_document_analyses(doc["id"], doc_analyses)
                    analysis_id = f"aggregated_{doc['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    aggregated["analysis_id"] = analysis_id
                    aggregated["sub_analyses"] = doc_analyses
                else:
                    aggregated = doc_analyses[0]

                analysis_file = os.path.join(
                    self.base_dir, "analyzed",
                    f"analysis_{aggregated['analysis_id']}.json"
                )
                with open(analysis_file, 'w') as f:
                    json.dump(aggregated, f, indent=2)

                if self.event_manager and aggregated.get("events"):
                    for event in aggregated["events"]:
                        self.event_manager.add_event_observation(
                            event_data={
                                "primary_actor": event.get("actors", ["Unknown"])[0] if event.get("actors") else "Unknown",
                                "action": event.get("action", event.get("name", "")),
                                "date": event.get("date", "N/A"),
                                "location": event.get("location", ""),
                                "description": event.get("description", ""),
                                "context": event.get("description", "")
                            },
                            source_doc={
                                "doc_id": doc["id"],
                                "source": doc["source"],
                                "bias_label": self._get_source_bias(doc["source"])
                            },
                            analysis_id=aggregated["analysis_id"]
                        )

                results.append(aggregated)
                analyzed_count += 1

            except Exception as e:
                self.logger.error(f"Error analyzing document {doc['id']}: {e}")

        return {
            "status": "completed",
            "analyzed": analyzed_count,
            "results": results
        }
    
    def _get_docs_since(self, since_date: datetime) -> List[str]:
        """Get document IDs collected since a given date."""
        docs = []
        for doc_id in self.document_repository.list_documents():
            _, metadata, _ = self.document_repository.get_document(doc_id)
            if metadata:
                collected_at = metadata.get("collected_at")
                if collected_at:
                    try:
                        doc_date = datetime.fromisoformat(collected_at.replace('Z', '+00:00'))
                        if doc_date >= since_date:
                            docs.append(doc_id)
                    except:
                        pass
        return docs
    
    def test_template(self, template: str, article_content: str = None, article_url: str = None) -> Dict[str, Any]:
        """Test a template with a single article."""
        if not self.analyzer:
            raise Exception("Analyzer not available - check LLM provider")
        
        # Use provided article or fetch from URL
        if article_url and not article_content:
            article_data = self.collector._extract_article(article_url)
            if not article_data:
                raise Exception("Failed to extract article from URL")
        elif article_content:
            article_data = {
                "title": "Test Article",
                "content": article_content,
                "url": article_url or "test://article",
                "published": datetime.now().isoformat()
            }
        else:
            raise Exception("Either article_content or article_url must be provided")
        
        # Run analysis
        analyzer = ContentAnalyzer(self.llm_provider, template_file=template)
        result = analyzer.process({"articles": [article_data], "document_ids": ["test_doc"]})
        
        if result.get("analyses"):
            return result["analyses"][0]
        else:
            return {"error": "No analysis produced"}
    
    def aggregate_events(self, analysis_window: int = 7, templates: List[str] = None) -> Dict[str, Any]:
        """Two-stage event-centric aggregation."""

        if not templates:
            templates = ["standard_analysis.json"]

        from event_aggregator import EventAggregator

        analyses = self._load_recent_multi_template_analyses(analysis_window, templates)

        if not analyses:
            return {"status": "no_analyses", "events": []}

        aggregator = EventAggregator()

        try:
            if hasattr(aggregator, "aggregate_with_crypto_lineage"):
                result = aggregator.aggregate_with_crypto_lineage(analyses)
                self.logger.info("Used crypto-enhanced aggregation")
            else:
                result = aggregator._aggregate_standard(analyses)
                self.logger.info("Used standard aggregation (crypto not available)")
        except Exception as e:
            self.logger.error(f"Enhanced aggregation failed: {e}")
            result = aggregator._aggregate_standard(analyses)

        output_file = f"{self.base_dir}/unified_graph/unified_kg_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

        if "unified_graph" in result:
            self._update_knowledge_graph_with_unified(result["unified_graph"])

        return {
            "status": "completed",
            "unique_events": len(result.get("event_graphs", {})),
            "unified_nodes": result.get("unified_graph", {}).get("stats", {}).get("total_nodes", 0),
            "analyses_processed": result.get("analyses_processed", len(analyses)),
            "crypto_lineage_included": "crypto_lineage" in result,
        }
    
    def _build_kg_original(self) -> Dict[str, Any]:
        """Original KG build method (renamed)."""
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
    
    def build_kg(self) -> Dict[str, Any]:
        """Enhanced KG build that includes event aggregation."""
        # First, do regular KG build from analyses
        regular_result = self._build_kg_original()
        
        # Then aggregate events
        event_result = self.aggregate_events()
        
        return {
            **regular_result,
            "event_aggregation": event_result
        }
    
    def sync_vectors(self) -> Dict[str, Any]:
        """Sync vector store with knowledge graph."""
        stats = self.vector_store.sync_with_knowledge_graph(self.knowledge_graph)
        return stats

    def _load_recent_multi_template_analyses(self, window_days: int, templates: List[str]) -> List[Dict]:
        """Load multi-template analyses within time window."""
        analyses: List[Dict] = []
        analyzed_dir = f"{self.base_dir}/analyzed"
        cutoff_date = datetime.now() - timedelta(days=window_days)

        if not os.path.exists(analyzed_dir):
            return analyses

        for filename in os.listdir(analyzed_dir):
            if not filename.startswith("analysis_"):
                continue

            filepath = os.path.join(analyzed_dir, filename)
            mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
            if mtime < cutoff_date:
                continue

            try:
                with open(filepath, "r") as f:
                    analysis = json.load(f)
                analyses.append(analysis)
            except Exception as e:
                self.logger.error(f"Error loading {filename}: {e}")

        return analyses

    def _update_knowledge_graph_with_unified(self, unified_graph: Dict[str, Any]) -> None:
        """Add unified graph data into the knowledge graph."""
        id_map: Dict[Tuple[str, str], str] = {}

        for node_key, node in unified_graph.get("nodes", {}).items():
            kg_id = self.knowledge_graph.add_node(
                node_type=node.get("node_type"),
                name=node.get("name"),
                attributes={
                    **node.get("attributes", {}),
                    "total_weight": node.get("total_weight"),
                    "event_appearances": node.get("event_appearances"),
                    "event_linkages": node.get("event_linkages", []),
                },
            )
            id_map[node_key] = kg_id

        for edge_key, edge in unified_graph.get("edges", {}).items():
            src_id = id_map.get(edge_key[0])
            tgt_id = id_map.get(edge_key[2])
            if not src_id or not tgt_id:
                continue
            self.knowledge_graph.add_edge(
                source_id=src_id,
                relation=edge.get("relation"),
                target_id=tgt_id,
                attributes={
                    "total_weight": edge.get("total_weight"),
                    "event_appearances": edge.get("event_appearances"),
                    "event_linkages": edge.get("event_linkages", []),
                },
            )

        self.knowledge_graph.save_graph()
    
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

    # ------------------------------------------------------------------
    # Crypto chain generation and export helpers
    # ------------------------------------------------------------------
    def generate_complete_crypto_chain(self) -> Dict[str, Any]:
        """Assemble full cryptographic lineage for export."""
        try:
            document_lineages = self.document_repository.collect_all_document_lineages()
            analysis_lineages = self.analyzer.collect_all_analysis_lineages() if hasattr(self.analyzer, "collect_all_analysis_lineages") else []

            aggregation_lineages = []
            try:
                from event_aggregator import EventAggregator
                aggregator = EventAggregator()
                if hasattr(aggregator, "collect_all_aggregation_lineages"):
                    aggregation_lineages = aggregator.collect_all_aggregation_lineages()
            except Exception as e:
                self.logger.warning(f"Could not collect aggregation lineages: {e}")

            lineage_tree = self._build_lineage_tree(document_lineages, analysis_lineages, aggregation_lineages)

            master_chain = {
                "chain_id": hashlib.sha256(json.dumps(lineage_tree, sort_keys=True).encode()).hexdigest(),
                "generation_timestamp": datetime.now().isoformat(),
                "master_instance_id": self._get_master_instance_id(),
                "lineage_tree": lineage_tree,
                "chain_statistics": {
                    "total_documents": len(document_lineages),
                    "total_analyses": len(analysis_lineages),
                    "total_aggregations": len(aggregation_lineages),
                },
            }

            return master_chain
        except Exception as e:
            self.logger.error(f"Crypto chain generation failed: {e}")
            return {
                "chain_id": "error",
                "generation_timestamp": datetime.now().isoformat(),
                "error": str(e),
                "lineage_tree": {"error": "chain_generation_failed"},
            }

    def _build_lineage_tree(self, documents: List[Dict], analyses: List[Dict], aggregations: List[Dict]) -> Dict:
        tree = {"documents": {}, "analyses": {}, "aggregations": {}, "derivation_map": {}}
        for doc_lineage in documents:
            doc_id = doc_lineage.get("document_id")
            if doc_id:
                tree["documents"][doc_id] = doc_lineage

        for analysis_lineage in analyses:
            analysis_id = analysis_lineage.get("analysis_id")
            if analysis_id:
                tree["analyses"][analysis_id] = analysis_lineage
                crypto_lineage = analysis_lineage.get("crypto_lineage", {})
                derivation = crypto_lineage.get("derivation", {})
                source_doc = derivation.get("derived_from_document")
                if source_doc:
                    tree.setdefault("derivation_map", {}).setdefault(source_doc, {"analyses": [], "aggregations": []})
                    tree["derivation_map"][source_doc]["analyses"].append(analysis_id)

        for agg_lineage in aggregations:
            agg_id = agg_lineage.get("aggregation_id")
            if agg_id:
                tree["aggregations"][agg_id] = agg_lineage
                crypto_lineage = agg_lineage.get("crypto_lineage", {})
                source_analyses = crypto_lineage.get("derived_from_analyses", [])
                for a_id in source_analyses:
                    for doc_id, derivations in tree.get("derivation_map", {}).items():
                        if a_id in derivations["analyses"]:
                            derivations["aggregations"].append(agg_id)
                            break

        return tree

    def _get_master_instance_id(self) -> str:
        try:
            instance_string = f"{self.config_path}:{self.base_dir}"
            return hashlib.sha256(instance_string.encode("utf-8")).hexdigest()[:16]
        except Exception:
            return "master_instance"

    def create_export_with_crypto_chain(self, version: str, private_key_path: str = None) -> Dict:
        try:
            crypto_chain = self.generate_complete_crypto_chain()

            intelligence_package = {
                "unified_graph": self._export_unified_graph(),
                "knowledge_graph": self._export_knowledge_graph_data(),
                "version": version,
                "export_timestamp": datetime.now().isoformat(),
            }

            export_record = {
                "version": version,
                "intelligence_package": intelligence_package,
                "crypto_chain": crypto_chain,
                "export_timestamp": datetime.now().isoformat(),
                "master_instance_id": crypto_chain.get("master_instance_id"),
            }

            export_hash = hashlib.sha256(json.dumps(export_record, sort_keys=True).encode("utf-8")).hexdigest()

            if private_key_path and os.path.exists(private_key_path):
                try:
                    export_signature = self._sign_export_with_private_key(export_record, private_key_path)
                    public_key = self._extract_public_key_from_private(private_key_path)
                except Exception as e:
                    self.logger.warning(f"Export signing failed: {e}")
                    export_signature = None
                    public_key = None
            else:
                export_signature = None
                public_key = None

            final_export = {
                "export_record": export_record,
                "export_hash": export_hash,
                "export_signature": export_signature,
                "public_key": public_key,
                "crypto_chain_included": True,
            }

            return final_export
        except Exception as e:
            self.logger.error(f"Export with crypto chain failed: {e}")
            return {"error": str(e), "crypto_chain_included": False}

    def _export_unified_graph(self) -> Dict:
        try:
            unified_dir = f"{self.base_dir}/unified_graph"
            if not os.path.exists(unified_dir):
                return {}
            files = [f for f in os.listdir(unified_dir) if f.startswith("unified_kg_")]
            if not files:
                return {}
            latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(unified_dir, f)))
            with open(os.path.join(unified_dir, latest_file), "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"Could not export unified graph: {e}")
            return {}

    def _export_knowledge_graph_data(self) -> Dict:
        try:
            return self.knowledge_graph.get_basic_statistics()
        except Exception as e:
            self.logger.warning(f"Could not export knowledge graph data: {e}")
            return {}

    def _sign_export_with_private_key(self, export_record: Dict, private_key_path: str) -> str:
        from cryptography.hazmat.primitives import serialization, hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        export_json = json.dumps(export_record, sort_keys=True).encode()
        with open(private_key_path, "rb") as f:
            private_key = serialization.load_pem_private_key(f.read(), password=None)

        signature = private_key.sign(
            export_json,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )

        return base64.b64encode(signature).decode("utf-8")

    def _extract_public_key_from_private(self, private_key_path: str) -> str:
        from cryptography.hazmat.primitives import serialization

        with open(private_key_path, "rb") as f:
            private_key = serialization.load_pem_private_key(f.read(), password=None)
        pub_bytes = private_key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        return pub_bytes.decode("utf-8")
    
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

    def get_export_orchestrator(self):
        """Get export orchestrator instance."""
        if not hasattr(self, '_export_orchestrator'):
            from export_orchestrator import ExportOrchestrator
            self._export_orchestrator = ExportOrchestrator(self)
        return self._export_orchestrator

    def create_distribution_package(self, package_type: str = "v001"):
        """Create distribution package through web interface."""
        orchestrator = self.get_export_orchestrator()
        if package_type == "v001":
            return orchestrator.create_v001_package()
        else:
            return orchestrator.create_update_package(package_type)


def main():
    parser = argparse.ArgumentParser(description="Night_watcher - Political Intelligence Framework")
    parser.add_argument("--config", default="config.json", help="Config file")
    parser.add_argument("--base-dir", default="data", help="Data directory")
    
    # Commands
    parser.add_argument("--collect", action="store_true", help="Run collection")
    parser.add_argument("--analyze", action="store_true", help="Run analysis")
    parser.add_argument("--build-kg", action="store_true", help="Build knowledge graph")
    parser.add_argument("--aggregate-events", action="store_true", help="Aggregate events from analyses")
    parser.add_argument("--show-events", action="store_true", help="Show weighted events")
    parser.add_argument("--event-details", help="Show details for specific event ID")
    parser.add_argument("--sync-vectors", action="store_true", help="Sync vectors")
    parser.add_argument("--full", action="store_true", help="Run full pipeline")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--export-signed", help="Export signed release artifact")
    parser.add_argument("--export-release", action="store_true", help="Export versioned release")
    parser.add_argument("--version", help="Version (v001, v002, etc)")
    parser.add_argument("--private-key", help="Private key file for signing")
    parser.add_argument("--previous-artifact", help="Previous artifact for chain")
    parser.add_argument("--bundle-files", nargs="+", help="Extra files to include")
    
    # Options
    parser.add_argument("--mode", choices=["auto", "first_run", "incremental", "full"],
                       default="auto", help="Collection mode")
    parser.add_argument("--max-articles", type=int, default=20, help="Max articles to analyze")
    parser.add_argument("--templates", nargs="+", help="Analysis templates to use")
    parser.add_argument("--aggregate", action="store_true", 
                       help="Use document aggregation for multiple templates")
    parser.add_argument("--target", choices=["unanalyzed", "recent", "all"],
                       default="unanalyzed", help="Analysis target")
    parser.add_argument("--since-date", help="Since date for target=all (ISO format)")
    parser.add_argument("--analysis-window", type=int, default=7, help="Days of analyses to aggregate")
    
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
            templates = args.templates or ["standard_analysis.json"]
            if args.aggregate and len(templates) > 1:
                pass  # analyze() will handle aggregation

            result = nw.analyze(
                max_articles=args.max_articles,
                templates=templates,
                target=args.target,
                since_date=args.since_date
            )
            print(f"✓ Analyzed {result.get('analyzed', 0)} documents with {len(templates)} templates")
        
        elif args.aggregate_events:
            result = nw.aggregate_events(analysis_window=args.analysis_window)
            print(f"✓ Aggregated {result['unique_events']} unique events")
            print(f"✓ Found {result['cross_source_events']} cross-source events")
            if result.get('coordinated_campaigns'):
                print(f"✓ Detected {len(result['coordinated_campaigns'])} coordinated campaigns")

        elif args.show_events:
            if nw.event_manager:
                events = nw.event_manager.get_weighted_events()
                print(f"\n=== Weighted Events (Total: {len(events)}) ===")
                for event in events[:10]:
                    attrs = json.loads(event['core_attributes'])
                    print(f"\nEvent: {event['event_id']}")
                    print(f"  Weight: {event['weight']:.1f} | Confidence: {event['confidence_score']:.2%}")
                    print(f"  Actor: {attrs.get('primary_actor', 'Unknown')}")
                    print(f"  Action: {attrs.get('action', 'Unknown')}")
                    print(f"  Date: {attrs.get('date', 'Unknown')}")
                    print(f"  Sources: {event['source_count']} | Diversity: {event['source_diversity']:.2%}")
            else:
                print("Event tracking not available")

        elif args.event_details:
            if nw.event_manager:
                ev = nw.event_manager.db.get_event(args.event_details)
                if not ev:
                    print("Event not found")
                else:
                    print(json.dumps(ev, indent=2))
            else:
                print("Event tracking not available")
        
        elif args.build_kg:
            result = nw.build_kg()
            print(f"✓ Processed {result['processed']} analyses")
            print(f"✓ Added {result['temporal_relations']} temporal relations")
            if 'event_aggregation' in result:
                print(f"✓ Aggregated {result['event_aggregation']['unique_events']} unique events")
        
        elif args.sync_vectors:
            result = nw.sync_vectors()
            print(f"✓ Synced {result['nodes_added']} vectors")

        elif args.export_signed:
            from export_signed_artifact import export_signed_artifact
            export_signed_artifact(
                output_path=args.export_signed,
                version=args.version,
                private_key_path=args.private_key,
                previous_artifact_path=args.previous_artifact,
                bundled_files=args.bundle_files,
            )
        elif args.export_release:
            from export_versioned_artifact import export_versioned_artifact
            out = f"night_watcher_{args.version}.tar.gz"
            export_versioned_artifact(
                output_path=out,
                version=args.version,
                private_key_path=args.private_key,
                previous_artifact_path=args.previous_artifact,
                bundled_files=args.bundle_files,
            )

        elif args.full:
            print("Running full pipeline...")
            
            # Collect
            collect_result = nw.collect()
            print(f"✓ Collected {len(collect_result['articles'])} articles")
            
            # Analyze
            if collect_result['articles']:
                analyze_result = nw.analyze()
                print(f"✓ Analyzed {analyze_result.get('analyzed', 0)} documents")
                
                # Build KG (includes event aggregation)
                if analyze_result.get('analyzed', 0) > 0:
                    kg_result = nw.build_kg()
                    print(f"✓ Built knowledge graph with event aggregation")
                    
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
