#!/usr/bin/env python3
"""
Night_watcher Web Server
Flask-based API and dashboard server for Night_watcher control.
"""

import os
import sys
import json
import logging
import threading
import time
import glob
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS

# Import Night_watcher components
from Night_Watcher import NightWatcher
from analyzer import ContentAnalyzer
import providers

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NightWatcherWeb")

app = Flask(__name__)
CORS(app)

# Global instance and state
night_watcher = None
current_task = None
task_thread = None
task_status = {
    "running": False,
    "task": None,
    "progress": 0,
    "messages": []
}

# Statistics cache
stats_cache = {
    "last_update": None,
    "data": {}
}

# Review queue storage (simple file-based)
REVIEW_QUEUE_FILE = "data/review_queue.json"


def init_night_watcher():
    """Initialize Night_watcher instance."""
    global night_watcher
    if not night_watcher:
        night_watcher = NightWatcher()
        logger.info("Night_watcher initialized")


def add_log_message(level, message):
    """Add message to task status log."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    task_status["messages"].append({
        "time": timestamp,
        "level": level,
        "message": message
    })
    # Keep only last 100 messages
    if len(task_status["messages"]) > 100:
        task_status["messages"] = task_status["messages"][-100:]


def load_review_queue():
    """Load review queue from file."""
    if os.path.exists(REVIEW_QUEUE_FILE):
        try:
            with open(REVIEW_QUEUE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []


def save_review_queue(queue):
    """Save review queue to file."""
    try:
        os.makedirs(os.path.dirname(REVIEW_QUEUE_FILE), exist_ok=True)
        with open(REVIEW_QUEUE_FILE, 'w', encoding='utf-8') as f:
            json.dump(queue, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save review queue: {e}")


def scan_for_review_items():
    """Scan analyzed directory for items needing review."""
    review_items = []
    analyzed_dir = "data/analyzed"
    
    if not os.path.exists(analyzed_dir):
        return review_items
    
    for filename in os.listdir(analyzed_dir):
        if not filename.startswith("analysis_"):
            continue
            
        try:
            filepath = os.path.join(analyzed_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                analysis = json.load(f)
            
            validation = analysis.get("validation", {})
            if validation.get("status") == "REVIEW":
                # Create review item
                review_item = {
                    "id": filename[:-5],  # Remove .json
                    "filename": filename,
                    "timestamp": analysis.get("timestamp"),
                    "template_name": analysis.get("template_info", {}).get("name", "Unknown"),
                    "template_status": analysis.get("template_info", {}).get("status", "Unknown"),
                    "validation_status": validation.get("status"),
                    "validation_reason": validation.get("reason", ""),
                    "article": analysis.get("article", {}),
                    "manipulation_score": analysis.get("manipulation_score", 0),
                    "concern_level": analysis.get("concern_level", "Unknown"),
                    "authoritarian_indicators": analysis.get("authoritarian_indicators", [])
                }
                review_items.append(review_item)
                
        except Exception as e:
            logger.error(f"Error scanning {filename}: {e}")
    
    return review_items


@app.route('/')
def dashboard():
    """Serve the dashboard HTML."""
    # Read the dashboard HTML file with UTF-8 encoding
    dashboard_path = os.path.join(os.path.dirname(__file__), 'night_watcher_dashboard.html')
    if os.path.exists(dashboard_path):
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return "<h1>Dashboard not found. Please ensure night_watcher_dashboard.html is in the same directory.</h1>"


@app.route('/api/status')
def api_status():
    """Get current system status."""
    try:
        init_night_watcher()
        
        # Cache status for 5 seconds
        now = datetime.now()
        if stats_cache["last_update"] and (now - stats_cache["last_update"]).seconds < 5:
            # But always update certain fields
            status = stats_cache["data"].copy()
        else:
            status = night_watcher.status()
            stats_cache["last_update"] = now
            stats_cache["data"] = status
        
        # Always update these fields
        review_items = scan_for_review_items()
        status["pending_review"] = len(review_items)
        
        # Add specific document counts
        status["total_documents"] = status.get("documents", {}).get("total", 0)
        status["pending_analysis"] = status.get("documents", {}).get("pending", 0)
        status["analyzed_documents"] = status.get("documents", {}).get("analyzed", 0)
        
        # Add graph stats
        kg_stats = status.get("knowledge_graph", {})
        status["graph_nodes"] = kg_stats.get("nodes", 0)
        status["graph_edges"] = kg_stats.get("edges", 0)
        
        # Add vector count
        status["vector_count"] = status.get("vector_store", {}).get("total_vectors", 0)
        
        # Add task status
        status["task_status"] = {
            "running": task_status["running"],
            "current_task": task_status["task"],
            "progress": task_status["progress"]
        }
        
        # Add system info
        status["system"] = {
            "llm_connected": night_watcher.llm_provider is not None,
            "llm_type": night_watcher.config.get("llm_provider", {}).get("type", "unknown"),
            "current_model": getattr(night_watcher.llm_provider, 'model', None) if night_watcher.llm_provider else None
        }
        
        return jsonify(status)
    except Exception as e:
        logger.error(f"Status error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/templates')
def api_templates():
    """Get available analysis templates."""
    try:
        templates = []
        
        # Scan for template files
        template_files = glob.glob("*.json")
        template_files = [f for f in template_files if "analysis" in f.lower()]
        
        for filename in template_files:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    template_data = json.load(f)
                
                templates.append({
                    "filename": filename,
                    "name": template_data.get("name", filename),
                    "description": template_data.get("description", ""),
                    "version": template_data.get("version", ""),
                    "status": template_data.get("status", "UNKNOWN")
                })
            except Exception as e:
                logger.error(f"Error reading template {filename}: {e}")
        
        return jsonify(templates)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/review-queue')
def api_review_queue():
    """Get review queue items."""
    try:
        review_items = scan_for_review_items()
        return jsonify(review_items)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/review-queue/approve', methods=['POST'])
def api_approve_analysis():
    """Approve an analysis."""
    try:
        data = request.json
        analysis_id = data.get("analysis_id")
        notes = data.get("notes", "")
        
        # Load analysis file
        filepath = f"data/analyzed/{analysis_id}.json"
        if not os.path.exists(filepath):
            return jsonify({"error": "Analysis not found"}), 404
        
        with open(filepath, 'r', encoding='utf-8') as f:
            analysis = json.load(f)
        
        # Update validation status
        analysis["validation"]["status"] = "VALID"
        analysis["validation"]["approved_by"] = "dashboard_user"
        analysis["validation"]["approved_at"] = datetime.now().isoformat()
        analysis["validation"]["notes"] = notes
        
        # Save updated analysis
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2)
        
        # Clear cache to force status refresh
        stats_cache["last_update"] = None
        
        return jsonify({"status": "approved"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/review-queue/reject', methods=['POST'])
def api_reject_analysis():
    """Reject an analysis."""
    try:
        data = request.json
        analysis_id = data.get("analysis_id")
        notes = data.get("notes", "")
        
        # Load analysis file
        filepath = f"data/analyzed/{analysis_id}.json"
        if not os.path.exists(filepath):
            return jsonify({"error": "Analysis not found"}), 404
        
        with open(filepath, 'r', encoding='utf-8') as f:
            analysis = json.load(f)
        
        # Update validation status
        analysis["validation"]["status"] = "REJECTED"
        analysis["validation"]["rejected_by"] = "dashboard_user"
        analysis["validation"]["rejected_at"] = datetime.now().isoformat()
        analysis["validation"]["notes"] = notes
        
        # Save updated analysis
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2)
        
        # Clear cache to force status refresh
        stats_cache["last_update"] = None
        
        return jsonify({"status": "rejected"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/review-queue/retry', methods=['POST'])
def api_retry_analysis():
    """Retry a failed analysis."""
    try:
        data = request.json
        analysis_id = data.get("analysis_id")
        
        # For now, just mark as failed - actual retry would need more work
        filepath = f"data/analyzed/{analysis_id}.json"
        if not os.path.exists(filepath):
            return jsonify({"error": "Analysis not found"}), 404
        
        # Simple approach: just delete the file so it can be reprocessed
        os.remove(filepath)
        
        return jsonify({"status": "queued_for_retry"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/collect', methods=['POST'])
def api_collect():
    """Start content collection."""
    global task_thread, task_status
    
    if task_status["running"]:
        return jsonify({"error": "Another task is already running"}), 400
    
    data = request.json or {}
    mode = data.get("mode", "auto")
    
    def run_collection():
        task_status["running"] = True
        task_status["task"] = "collection"
        task_status["progress"] = 0
        add_log_message("info", f"Starting collection (mode: {mode})")

        def progress(event):
            if event.get("type") == "article":
                add_log_message("info", f"Collected: {event.get('title')}")
            elif event.get("type") == "error":
                add_log_message("error", f"{event.get('source')}: {event.get('message')}")
            elif event.get("type") == "cancelled":
                add_log_message("warning", "Collection cancelled")

        try:
            init_night_watcher()
            result = night_watcher.collect(mode=mode, callback=progress)
            
            articles_count = len(result.get("articles", []))
            if night_watcher.collector.cancelled:
                add_log_message("warning", "Collection stopped by user")
            else:
                add_log_message("success", f"Collection completed: {articles_count} articles")
            task_status["progress"] = 100
            
        except Exception as e:
            add_log_message("error", f"Collection failed: {str(e)}")
        finally:
            task_status["running"] = False
            task_status["task"] = None
            # Clear cache to update status
            stats_cache["last_update"] = None
    
    task_thread = threading.Thread(target=run_collection)
    task_thread.start()

    return jsonify({"status": "started"})


@app.route('/api/collect/stop', methods=['POST'])
def api_collect_stop():
    """Request stopping an ongoing collection."""
    try:
        if night_watcher and night_watcher.collector:
            night_watcher.collector.cancel()
            return jsonify({"status": "stopping"})
        return jsonify({"error": "Collector not running"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """Start content analysis."""
    global task_thread, task_status
    
    if task_status["running"]:
        return jsonify({"error": "Another task is already running"}), 400
    
    data = request.json or {}
    max_articles = data.get("max_articles", 20)
    templates = data.get("templates") or [data.get("template", "standard_analysis.json")]
    
    def run_analysis():
        task_status["running"] = True
        task_status["task"] = "analysis"
        task_status["progress"] = 0
        add_log_message("info", f"Starting analysis (max: {max_articles}, templates: {', '.join(templates)})")
        
        try:
            init_night_watcher()
            
            # Check if analyzer is available
            if not night_watcher.analyzer:
                add_log_message("error", "Analyzer not available - check LLM provider")
                task_status["progress"] = 100
                return
            
            # Get unanalyzed documents
            all_docs = night_watcher.document_repository.list_documents()
            analyzed = night_watcher._get_analyzed_docs()
            unanalyzed = [d for d in all_docs if d not in analyzed][:max_articles]
            
            if not unanalyzed:
                add_log_message("warning", "No new documents to analyze")
                task_status["progress"] = 100
                return
            
            add_log_message("info", f"Found {len(unanalyzed)} documents to analyze")
            
            # Run analysis
            result = night_watcher.analyze(max_articles=max_articles, templates=templates)
            analyzed_count = result.get("analyzed", 0)
            add_log_message("success", f"Analysis completed: {analyzed_count} documents across {result.get('templates', 1)} templates")
            task_status["progress"] = 100
            
        except Exception as e:
            add_log_message("error", f"Analysis failed: {str(e)}")
            logger.error(f"Analysis error details: {e}", exc_info=True)
        finally:
            task_status["running"] = False
            task_status["task"] = None
            # Clear cache to update status
            stats_cache["last_update"] = None
    
    task_thread = threading.Thread(target=run_analysis)
    task_thread.start()
    
    return jsonify({"status": "started"})


@app.route('/api/build-kg', methods=['POST'])
def api_build_kg():
    """Build knowledge graph."""
    global task_thread, task_status
    
    if task_status["running"]:
        return jsonify({"error": "Another task is already running"}), 400
    
    def run_kg_build():
        task_status["running"] = True
        task_status["task"] = "knowledge_graph"
        task_status["progress"] = 0
        add_log_message("info", "Building knowledge graph from analyses")
        
        try:
            init_night_watcher()
            result = night_watcher.build_kg()
            
            processed = result.get("processed", 0)
            temporal = result.get("temporal_relations", 0)
            add_log_message("success", f"KG built: {processed} analyses, {temporal} temporal relations")
            task_status["progress"] = 100
            
        except Exception as e:
            add_log_message("error", f"KG build failed: {str(e)}")
        finally:
            task_status["running"] = False
            task_status["task"] = None
            # Clear cache to update status
            stats_cache["last_update"] = None
    
    task_thread = threading.Thread(target=run_kg_build)
    task_thread.start()
    
    return jsonify({"status": "started"})


@app.route('/api/sync-vectors', methods=['POST'])
def api_sync_vectors():
    """Sync vector store."""
    global task_thread, task_status
    
    if task_status["running"]:
        return jsonify({"error": "Another task is already running"}), 400
    
    def run_sync():
        task_status["running"] = True
        task_status["task"] = "vector_sync"
        task_status["progress"] = 0
        add_log_message("info", "Synchronizing vector store with knowledge graph")
        
        try:
            init_night_watcher()
            result = night_watcher.sync_vectors()
            
            nodes_added = result.get("nodes_added", 0)
            add_log_message("success", f"Vector sync completed: {nodes_added} nodes added")
            task_status["progress"] = 100
            
        except Exception as e:
            add_log_message("error", f"Vector sync failed: {str(e)}")
        finally:
            task_status["running"] = False
            task_status["task"] = None
            # Clear cache to update status
            stats_cache["last_update"] = None
    
    task_thread = threading.Thread(target=run_sync)
    task_thread.start()
    
    return jsonify({"status": "started"})


@app.route('/api/task-status')
def api_task_status():
    """Get current task status and logs."""
    return jsonify({
        "running": task_status["running"],
        "task": task_status["task"],
        "progress": task_status["progress"],
        "messages": task_status["messages"][-20:]  # Last 20 messages
    })


@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    """Get or update configuration."""
    init_night_watcher()
    
    if request.method == 'GET':
        return jsonify(night_watcher.config)
    
    elif request.method == 'POST':
        try:
            new_config = request.json or {}

            def deep_update(orig, updates):
                for k, v in updates.items():
                    if isinstance(v, dict) and isinstance(orig.get(k), dict):
                        deep_update(orig[k], v)
                    else:
                        orig[k] = v
                return orig

            # Special handling for model updates
            if "llm_provider" in new_config and "model" in new_config["llm_provider"]:
                model = new_config["llm_provider"]["model"]
                # Update the provider directly if it exists
                if night_watcher.llm_provider and hasattr(night_watcher.llm_provider, 'update_model'):
                    night_watcher.llm_provider.update_model(model)
                    add_log_message("success", f"Model updated to: {model}")

            # Merge new values into existing config
            night_watcher.config = deep_update(night_watcher.config, new_config)

            # Save to file
            with open(night_watcher.config_path, 'w', encoding='utf-8') as f:
                json.dump(night_watcher.config, f, indent=2)

            # Reinitialize components if needed
            if "llm_provider" in new_config and ("type" in new_config["llm_provider"] or "api_key" in new_config["llm_provider"]):
                night_watcher.llm_provider = providers.initialize_llm_provider(night_watcher.config)
                night_watcher.analyzer = ContentAnalyzer(night_watcher.llm_provider) if night_watcher.llm_provider else None
                add_log_message("success", "LLM provider reinitialized")

            add_log_message("success", "Configuration updated")
            return jsonify({"status": "updated"})

        except Exception as e:
            logger.error(f"Config update error: {e}")
            return jsonify({"error": str(e)}), 400


@app.route('/api/llm-models')
def api_llm_models():
    """Return available models from LM Studio."""
    init_night_watcher()
    try:
        if night_watcher.llm_provider and hasattr(night_watcher.llm_provider, 'list_models'):
            models = night_watcher.llm_provider.list_models()
            return jsonify(models)
        else:
            return jsonify([])
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return jsonify([])


@app.route('/api/sources')
def api_sources():
    """Get content sources."""
    init_night_watcher()
    sources = night_watcher.config.get("content_collection", {}).get("sources", [])
    return jsonify(sources)


@app.route('/api/sources/add', methods=['POST'])
def api_add_source():
    """Add a new content source."""
    try:
        init_night_watcher()
        source_data = request.json
        
        if night_watcher.collector.add_source(source_data):
            # Save config
            with open(night_watcher.config_path, 'w', encoding='utf-8') as f:
                json.dump(night_watcher.config, f, indent=2)
            
            add_log_message("success", f"Added source: {source_data.get('name', source_data.get('url'))}")
            return jsonify({"status": "added"})
        else:
            return jsonify({"error": "Failed to add source"}), 400
            
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/sources/update', methods=['POST'])
def api_update_source():
    """Update an existing content source."""
    try:
        init_night_watcher()
        data = request.json
        url = data.get("url")
        limit = data.get("limit")

        if not url or limit is None:
            return jsonify({"error": "url and limit required"}), 400

        sources = night_watcher.config.get("content_collection", {}).get("sources", [])
        updated = False
        for src in sources:
            if src.get("url") == url:
                src["limit"] = int(limit)
                updated = True
                break

        if not updated:
            return jsonify({"error": "source not found"}), 404

        night_watcher.collector.sources = sources

        with open(night_watcher.config_path, 'w', encoding='utf-8') as f:
            json.dump(night_watcher.config, f, indent=2)

        add_log_message("success", f"Updated source limit for {url}")
        return jsonify({"status": "updated"})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/sources/set-limit', methods=['POST'])
def api_set_source_limit():
    """Set article limit for all sources."""
    try:
        init_night_watcher()
        data = request.json
        limit = data.get("limit")

        if limit is None:
            return jsonify({"error": "limit required"}), 400

        sources = night_watcher.config.get("content_collection", {}).get("sources", [])
        for src in sources:
            src["limit"] = int(limit)

        night_watcher.collector.sources = sources
        night_watcher.config["content_collection"]["article_limit"] = int(limit)

        with open(night_watcher.config_path, 'w', encoding='utf-8') as f:
            json.dump(night_watcher.config, f, indent=2)

        add_log_message("success", f"Set article limit for all sources: {limit}")
        return jsonify({"status": "updated"})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/search', methods=['POST'])
def api_search():
    """Search vector store."""
    try:
        init_night_watcher()
        data = request.json
        query = data.get("query", "")
        node_type = data.get("node_type")
        limit = data.get("limit", 10)
        
        results = night_watcher.vector_store.search(
            query=query,
            item_type=node_type,
            limit=limit
        )
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/test-llm', methods=['POST'])
def api_test_llm():
    """Test LLM connection."""
    try:
        init_night_watcher()
        
        if not night_watcher.llm_provider:
            return jsonify({"error": "No LLM provider configured"}), 400
        
        # Simple test prompt
        result = night_watcher.llm_provider.complete(
            prompt="Say 'Connection successful!' and nothing else.",
            max_tokens=20
        )
        
        if "error" in result:
            return jsonify({"error": result["error"]}), 400
        
        # Extract text from response
        text = result.get("choices", [{}])[0].get("text", "")
        
        return jsonify({
            "status": "connected",
            "response": text,
            "model": getattr(night_watcher.llm_provider, 'model', 'unknown')
        })
        
    except Exception as e:
        logger.error(f"LLM test error: {e}")
        return jsonify({"error": str(e)}), 400


@app.route('/api/pipeline', methods=['POST'])
def api_pipeline():
    """Run full pipeline."""
    global task_thread, task_status
    
    if task_status["running"]:
        return jsonify({"error": "Another task is already running"}), 400
    
    def run_pipeline():
        task_status["running"] = True
        task_status["task"] = "full_pipeline"
        task_status["progress"] = 0
        
        try:
            init_night_watcher()
            
            # Collection
            add_log_message("info", "Starting collection phase")
            task_status["progress"] = 10
            collect_result = night_watcher.collect()
            articles_count = len(collect_result.get("articles", []))
            add_log_message("success", f"Collected {articles_count} articles")
            
            if articles_count > 0:
                # Analysis
                add_log_message("info", "Starting analysis phase")
                task_status["progress"] = 40
                analyze_result = night_watcher.analyze()
                analyzed = analyze_result.get("analyzed", 0)
                add_log_message("success", f"Analyzed {analyzed} documents")
                
                if analyzed > 0:
                    # Knowledge Graph
                    add_log_message("info", "Building knowledge graph")
                    task_status["progress"] = 70
                    kg_result = night_watcher.build_kg()
                    add_log_message("success", f"Processed {kg_result.get('processed', 0)} analyses")
                    
                    # Vector Sync
                    add_log_message("info", "Synchronizing vectors")
                    task_status["progress"] = 90
                    vector_result = night_watcher.sync_vectors()
                    add_log_message("success", f"Synced {vector_result.get('nodes_added', 0)} vectors")
            
            task_status["progress"] = 100
            add_log_message("success", "Pipeline completed successfully")
            
        except Exception as e:
            add_log_message("error", f"Pipeline failed: {str(e)}")
        finally:
            task_status["running"] = False
            task_status["task"] = None
            # Clear cache to update status
            stats_cache["last_update"] = None
    
    task_thread = threading.Thread(target=run_pipeline)
    task_thread.start()
    
    return jsonify({"status": "started"})


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Night_watcher Web Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Initialize Night_watcher on startup
    init_night_watcher()
    
    # Run Flask app
    print(f"\n🌙 Night_watcher Web Server")
    print(f"Dashboard: http://{args.host}:{args.port}")
    print(f"API Base: http://{args.host}:{args.port}/api/")
    print("\nPress Ctrl+C to stop\n")
    
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
