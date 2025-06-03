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
from datetime import datetime
from flask import Flask, render_template_string, jsonify, request, send_from_directory
from flask_cors import CORS

# Import Night_watcher components
from Night_Watcher import NightWatcher

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


@app.route('/')
def dashboard():
    """Serve the dashboard HTML."""
    # Read the dashboard HTML file
    dashboard_path = os.path.join(os.path.dirname(__file__), 'night_watcher_dashboard.html')
    if os.path.exists(dashboard_path):
        with open(dashboard_path, 'r') as f:
            return f.read()
    else:
        return "<h1>Dashboard not found. Please ensure night_watcher_dashboard.html is in the same directory.</h1>"


@app.route('/api/status')
def api_status():
    """Get current system status."""
    try:
        init_night_watcher()
        
        # Cache status for 30 seconds
        now = datetime.now()
        if stats_cache["last_update"] and (now - stats_cache["last_update"]).seconds < 30:
            return jsonify(stats_cache["data"])
        
        status = night_watcher.status()
        
        # Add task status
        status["task_status"] = {
            "running": task_status["running"],
            "current_task": task_status["task"],
            "progress": task_status["progress"]
        }
        
        # Add system info
        status["system"] = {
            "llm_connected": night_watcher.llm_provider is not None,
            "llm_type": night_watcher.config.get("llm_provider", {}).get("type", "unknown")
        }
        
        # Cache it
        stats_cache["last_update"] = now
        stats_cache["data"] = status
        
        return jsonify(status)
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
        
        try:
            init_night_watcher()
            result = night_watcher.collect(mode=mode)
            
            articles_count = len(result.get("articles", []))
            add_log_message("success", f"Collection completed: {articles_count} articles")
            task_status["progress"] = 100
            
        except Exception as e:
            add_log_message("error", f"Collection failed: {str(e)}")
        finally:
            task_status["running"] = False
            task_status["task"] = None
    
    task_thread = threading.Thread(target=run_collection)
    task_thread.start()
    
    return jsonify({"status": "started"})


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """Start content analysis."""
    global task_thread, task_status
    
    if task_status["running"]:
        return jsonify({"error": "Another task is already running"}), 400
    
    data = request.json or {}
    max_articles = data.get("max_articles", 20)
    
    def run_analysis():
        task_status["running"] = True
        task_status["task"] = "analysis"
        task_status["progress"] = 0
        add_log_message("info", f"Starting analysis (max: {max_articles} articles)")
        
        try:
            init_night_watcher()
            result = night_watcher.analyze(max_articles=max_articles)
            
            analyzed_count = result.get("analyzed", 0)
            add_log_message("success", f"Analysis completed: {analyzed_count} documents")
            task_status["progress"] = 100
            
        except Exception as e:
            add_log_message("error", f"Analysis failed: {str(e)}")
        finally:
            task_status["running"] = False
            task_status["task"] = None
    
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
            new_config = request.json
            
            # Update config
            night_watcher.config.update(new_config)
            
            # Save to file
            with open(night_watcher.config_path, 'w') as f:
                json.dump(night_watcher.config, f, indent=2)
            
            add_log_message("success", "Configuration updated")
            return jsonify({"status": "updated"})
            
        except Exception as e:
            return jsonify({"error": str(e)}), 400


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
            with open(night_watcher.config_path, 'w') as f:
                json.dump(night_watcher.config, f, indent=2)
            
            add_log_message("success", f"Added source: {source_data.get('name', source_data.get('url'))}")
            return jsonify({"status": "added"})
        else:
            return jsonify({"error": "Failed to add source"}), 400
            
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
        
        results = night_watcher.vector_store.similar_nodes(
            query=query,
            node_type=node_type,
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
        
        return jsonify({
            "status": "connected",
            "response": result.get("choices", [{}])[0].get("text", "")
        })
        
    except Exception as e:
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
