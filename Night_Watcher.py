#!/usr/bin/env python3
"""
Night_watcher - Political Intelligence Framework
Main orchestrator for collection, analysis, and knowledge graph building.
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import hashlib

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from file_utils import safe_json_load
from document_repository import DocumentRepository
from collector import ContentCollector
from analyzer import ContentAnalyzer
from knowledge_graph import KnowledgeGraph
from vector_store import VectorStore
from providers import get_llm_provider


class NightWatcher:
    """Main Night_watcher orchestrator."""

    def __init__(self, config_path: str = "config.json", base_dir: str = "data"):
        self.base_dir = base_dir
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = self._setup_logging()

        # Initialize components
        self.document_repository = DocumentRepository(base_dir)
        self.llm_provider = self._init_llm_provider()
        self.collector = ContentCollector(self.config, self.document_repository)
        self.analyzer = ContentAnalyzer(self.config, self.llm_provider, self.document_repository)
        self.knowledge_graph = KnowledgeGraph(base_dir)
        self.vector_store = VectorStore(base_dir, self.llm_provider)

        # Initialize event manager if available
        self.event_manager = self._init_event_manager()

        self.logger.info("Night_watcher initialized")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        return safe_json_load(self.config_path, default={})

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def _init_llm_provider(self):
        """Initialize LLM provider from config."""
        try:
            return get_llm_provider(self.config.get("llm_provider", {}))
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM provider: {e}")
            return None

    def _init_event_manager(self):
        """Initialize event manager if event database is available."""
        try:
            from event_weight import EventManager
            from event_database import EventDatabase

            db_path = os.path.join(self.base_dir, "events.db")
            if os.path.exists(db_path) or self.llm_provider:
                event_db = EventDatabase(db_path)
                return EventManager(event_db, self.llm_provider)
        except ImportError:
            self.logger.debug("Event tracking components not available")
        except Exception as e:
            self.logger.error(f"Failed to initialize event manager: {e}")
        return None

    def validate_system_state(self) -> List[str]:
        """Check system state and return any issues."""
        issues = []

        if not self.llm_provider:
            issues.append("LLM provider not configured")

        if not os.path.exists(self.base_dir):
            issues.append(f"Base directory {self.base_dir} does not exist")

        return issues

    def collect(self, mode: str = "auto") -> Dict[str, Any]:
        """Run content collection."""
        issues = self.validate_system_state()
        if issues:
            self.logger.warning(f"System issues detected: {issues}")

        return self.collector.collect_content(mode=mode)

    def analyze(self, max_articles: int = 20) -> Dict[str, Any]:
        """Run content analysis."""
        issues = self.validate_system_state()
        if issues:
            self.logger.warning(f"System issues detected: {issues}")

        if not self.llm_provider:
            return {"status": "error", "message": "LLM provider not available"}

        # Get unanalyzed documents
        analyzed_docs = self._get_analyzed_docs()
        all_docs = self.document_repository.list_documents()
        unanalyzed = [doc_id for doc_id in all_docs if doc_id not in analyzed_docs]

        if not unanalyzed:
            return {"status": "no_new_documents", "analyzed": 0}

        # Limit to max_articles
        to_analyze = unanalyzed[:max_articles]

        results = []
        analyzed_count = 0

        for doc_id in to_analyze:
            try:
                content, metadata, _ = self.document_repository.get_document(doc_id)
                if not content:
                    continue

                doc = {
                    "id": doc_id,
                    "content": content,
                    "url": metadata.get("url", ""),
                    "title": metadata.get("title", ""),
                    "source": metadata.get("source", ""),
                    "collected_at": metadata.get("collected_at", "")
                }

                # Run multi-template analysis
                doc_analyses = self.analyzer.analyze_document_multi_template(doc)

                if doc_analyses:
                    # Use document aggregator to consolidate
                    from document_aggregator import DocumentAggregator
                    aggregator = DocumentAggregator()
                    aggregated = aggregator.aggregate_document_analyses(doc_analyses)

                    # Save aggregated analysis
                    analysis_file = os.path.join(
                        self.base_dir, "analyzed",
                        f"analysis_{aggregated['analysis_id']}.json"
                    )
                    os.makedirs(os.path.dirname(analysis_file), exist_ok=True)
                    with open(analysis_file, 'w') as f:
                        json.dump(aggregated, f, indent=2)

                    # Add to event manager if available
                    if self.event_manager and aggregated.get("events"):
                        for event in aggregated["events"]:
                            self.event_manager.add_event_observation(
                                event_data={
                                    "primary_actor": event.get("actors", ["Unknown"])[0] if event.get(
                                        "actors") else "Unknown",
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
                self.logger.error(f"Error analyzing document {doc_id}: {e}")

        return {
            "status": "completed",
            "analyzed": analyzed_count,
            "results": results
        }

    def _get_source_bias(self, source: str) -> str:
        """Get bias label for source."""
        # Simple bias mapping - can be enhanced
        bias_map = self.config.get("source_bias", {})
        return bias_map.get(source, "unknown")

    def aggregate_events(self, analysis_window: int = 7, templates: Optional[List[str]] = None) -> Dict[str, Any]:
        """Aggregate events across analyses using EventAggregator."""
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

        # Save unified graph
        output_file = f"{self.base_dir}/unified_graph/unified_kg_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

        # Update knowledge graph with unified data
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
                    analysis_path = f"{analyses_dir}/{filename}"
                    analysis = safe_json_load(analysis_path, default=None)
                    if analysis is None:
                        raise ValueError("invalid json")

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
        """Enhanced KG build that runs event aggregation BEFORE KG creation."""
        issues = self.validate_system_state()
        if issues:
            self.logger.warning(f"System issues detected: {issues}")

        # First, aggregate events across analyses
        event_result = self.aggregate_events()

        # Then build KG from individual analyses AND aggregated events
        regular_result = self._build_kg_original()

        return {
            **regular_result,
            "event_aggregation": event_result
        }

    def sync_vectors(self) -> Dict[str, Any]:
        """Sync vector store with knowledge graph."""
        issues = self.validate_system_state()
        if issues:
            self.logger.warning(f"System issues detected: {issues}")
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
                analysis = safe_json_load(filepath, default=None)
                if analysis is None:
                    raise ValueError("invalid json")
                analyses.append(analysis)
            except Exception as e:
                self.logger.error(f"Error loading {filename}: {e}")

        return analyses

    def _update_knowledge_graph_with_unified(self, unified_graph: Dict[str, Any]) -> None:
        """Add unified graph data into the knowledge graph."""
        id_map: Dict[tuple, str] = {}

        # Add nodes
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

        # Add edges
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

        if not os.path.exists(analyzed_dir):
            return analyzed

        for filename in os.listdir(analyzed_dir):
            if filename.startswith("analysis_"):
                try:
                    analysis_path = f"{analyzed_dir}/{filename}"
                    analysis = safe_json_load(analysis_path, default=None)
                    if analysis and analysis.get("article", {}).get("id"):
                        analyzed.add(analysis["article"]["id"])
                except Exception as e:
                    self.logger.error(f"Error reading {filename}: {e}")

        return analyzed

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
        if not self.llm_provider:
            return {"error": "LLM provider not available"}

        if not article_content:
            # Use a sample article
            article_content = "Sample article content for testing"
            article_url = "https://example.com/test"

        return self.analyzer.test_template(template, article_content, article_url)

    def run_full_pipeline(self, mode: str = "auto", max_articles: int = 20) -> Dict[str, Any]:
        """Run the complete pipeline: collect -> analyze -> aggregate -> build KG -> sync vectors."""
        results = {}

        # Collection
        self.logger.info("Starting collection phase")
        collect_result = self.collect(mode=mode)
        results["collection"] = collect_result

        # Analysis
        self.logger.info("Starting analysis phase")
        analyze_result = self.analyze(max_articles=max_articles)
        results["analysis"] = analyze_result

        # Knowledge Graph building (includes event aggregation)
        self.logger.info("Building knowledge graph with event aggregation")
        kg_result = self.build_kg()
        results["knowledge_graph"] = kg_result

        # Vector sync
        self.logger.info("Synchronizing vectors")
        vector_result = self.sync_vectors()
        results["vectors"] = vector_result

        return results

    def get_status(self) -> Dict[str, Any]:
        """Get system status and statistics."""
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
    parser.add_argument("--template", help="Template to test")
    parser.add_argument("--test-content", help="Content for template testing")
    parser.add_argument("--test-url", help="URL for template testing")

    args = parser.parse_args()

    try:
        nw = NightWatcher(args.config, args.base_dir)

        if args.collect:
            result = nw.collect(mode=args.mode)
            print(f"✓ Collected {len(result.get('articles', []))} articles")

        elif args.analyze:
            result = nw.analyze(max_articles=args.max_articles)
            print(f"✓ Analyzed {result['analyzed']} documents")

        elif args.aggregate_events:
            result = nw.aggregate_events()
            print(f"✓ Aggregated {result['unique_events']} unique events")
            print(f"✓ Processed {result['analyses_processed']} analyses")

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
                base_dir=args.base_dir
            )
            print(f"✓ Exported signed artifact to {args.export_signed}")

        elif args.export_release:
            orchestrator = nw.get_export_orchestrator()
            result = orchestrator.create_update_package(args.version or "v001")
            print(f"✓ Created release package: {result}")

        elif args.template:
            result = nw.test_template(args.template, args.test_content, args.test_url)
            print(json.dumps(result, indent=2))

        elif args.full:
            result = nw.run_full_pipeline(mode=args.mode, max_articles=args.max_articles)
            print("✓ Full pipeline completed")
            for phase, phase_result in result.items():
                if isinstance(phase_result, dict):
                    if phase == "collection":
                        print(f"  Collection: {len(phase_result.get('articles', []))} articles")
                    elif phase == "analysis":
                        print(f"  Analysis: {phase_result.get('analyzed', 0)} documents")
                    elif phase == "knowledge_graph":
                        print(
                            f"  KG: {phase_result.get('processed', 0)} analyses, {phase_result.get('event_aggregation', {}).get('unique_events', 0)} events")
                    elif phase == "vectors":
                        print(f"  Vectors: {phase_result.get('nodes_added', 0)} synced")

        elif args.status:
            status = nw.get_status()
            print(json.dumps(status, indent=2))

        else:
            parser.print_help()

    except KeyboardInterrupt:
        print("\n✗ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()