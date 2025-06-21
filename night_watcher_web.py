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
from flask import Flask, jsonify, request, Response
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
task_status = {"running": False, "task": None, "progress": 0, "messages": []}

# Statistics cache
stats_cache = {"last_update": None, "data": {}}

# Review queue storage (simple file-based)
REVIEW_QUEUE_FILE = "data/review_queue.json"

# Public key storage
PUBLIC_KEY_FILE = "data/export_keys/public_key.pem"

# Event aggregation cache
EVENT_CACHE_FILE = "data/event_cache/latest_events.json"


def cache_aggregation_results(result):
    """Persist latest event aggregation results to disk."""
    try:
        os.makedirs(os.path.dirname(EVENT_CACHE_FILE), exist_ok=True)
        with open(EVENT_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to cache aggregation results: {e}")


def load_latest_aggregation():
    """Load most recent aggregation results if available."""
    if os.path.exists(EVENT_CACHE_FILE):
        try:
            with open(EVENT_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load aggregation results: {e}")
    return {}


def init_night_watcher():
    """Initialize Night_watcher instance."""
    global night_watcher
    if not night_watcher:
        night_watcher = NightWatcher()
        logger.info("Night_watcher initialized")


def add_log_message(level, message):
    """Add message to task status log."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    task_status["messages"].append(
        {"time": timestamp, "level": level, "message": message}
    )
    # Keep only last 100 messages
    if len(task_status["messages"]) > 100:
        task_status["messages"] = task_status["messages"][-100:]


def validate_key_pair(private_key_pem: str, public_key_pem: str) -> dict:
    """Validate that the provided private and public keys match."""
    try:
        from cryptography.hazmat.primitives import serialization, hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        private_key = serialization.load_pem_private_key(
            private_key_pem.encode(), password=None
        )
        public_key = serialization.load_pem_public_key(public_key_pem.encode())

        test_data = b"Night_watcher key validation test"
        signature = private_key.sign(
            test_data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256(),
        )

        public_key.verify(
            signature,
            test_data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256(),
        )

        return {"valid": True, "message": "Key pair validation successful"}

    except Exception as e:
        return {"valid": False, "message": str(e)}


def load_review_queue():
    """Load review queue from file."""
    if os.path.exists(REVIEW_QUEUE_FILE):
        try:
            with open(REVIEW_QUEUE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []


def save_review_queue(queue):
    """Save review queue to file."""
    try:
        os.makedirs(os.path.dirname(REVIEW_QUEUE_FILE), exist_ok=True)
        with open(REVIEW_QUEUE_FILE, "w", encoding="utf-8") as f:
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
            with open(filepath, "r", encoding="utf-8") as f:
                analysis = json.load(f)

            validation = analysis.get("validation", {})
            if validation.get("status") == "REVIEW":
                # Create review item
                review_item = {
                    "id": filename[:-5],  # Remove .json
                    "filename": filename,
                    "timestamp": analysis.get("timestamp"),
                    "template_name": analysis.get("template_info", {}).get(
                        "name", "Unknown"
                    ),
                    "template_status": analysis.get("template_info", {}).get(
                        "status", "Unknown"
                    ),
                    "validation_status": validation.get("status"),
                    "validation_reason": validation.get("reason", ""),
                    "article": analysis.get("article", {}),
                    "manipulation_score": analysis.get("manipulation_score", 0),
                    "concern_level": analysis.get("concern_level", "Unknown"),
                    "authoritarian_indicators": analysis.get(
                        "authoritarian_indicators", []
                    ),
                }
                review_items.append(review_item)

        except Exception as e:
            logger.error(f"Error scanning {filename}: {e}")

    return review_items


@app.route("/")
def dashboard():
    """Serve the dashboard HTML."""
    # Read the dashboard HTML file with UTF-8 encoding
    dashboard_path = os.path.join(
        os.path.dirname(__file__), "night_watcher_dashboard.html"
    )
    if os.path.exists(dashboard_path):
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return "<h1>Dashboard not found. Please ensure night_watcher_dashboard.html is in the same directory.</h1>"


@app.route("/api/status")
def api_status():
    """Get current system status."""
    try:
        init_night_watcher()

        # Cache status for 5 seconds
        now = datetime.now()
        if (
            stats_cache["last_update"]
            and (now - stats_cache["last_update"]).seconds < 5
        ):
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

        # Add latest event aggregation stats
        latest_events = load_latest_aggregation()
        status["event_stats"] = {
            "unique_events": latest_events.get("event_count", 0),
            "cross_source_events": len(latest_events.get("cross_source_events", [])),
            "active_campaigns": len(latest_events.get("coordinated_campaigns", [])),
            "threat_level": latest_events.get("authoritarian_escalation", {}).get("overall_threat_level", 0),
        }

        # Add task status
        status["task_status"] = {
            "running": task_status["running"],
            "current_task": task_status["task"],
            "progress": task_status["progress"],
        }

        # Add system info
        status["system"] = {
            "llm_connected": night_watcher.llm_provider is not None,
            "llm_type": night_watcher.config.get("llm_provider", {}).get(
                "type", "unknown"
            ),
            "current_model": (
                getattr(night_watcher.llm_provider, "model", None)
                if night_watcher.llm_provider
                else None
            ),
        }

        return jsonify(status)
    except Exception as e:
        logger.error(f"Status error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/templates")
def api_templates():
    """Get available analysis templates grouped by status."""
    try:
        templates = {"approved": [], "unapproved": []}

        # Scan for template files
        template_files = glob.glob("*_analysis.json")

        for filename in template_files:
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    template_data = json.load(f)

                template_info = {
                    "filename": filename,
                    "name": template_data.get("name", filename),
                    "description": template_data.get("description", ""),
                    "version": template_data.get("version", ""),
                    "status": template_data.get("status", "UNKNOWN"),
                    "rounds": len(template_data.get("rounds", [])),
                }

                # Group by status
                if template_data.get("status") == "PRODUCTION":
                    templates["approved"].append(template_info)
                else:
                    templates["unapproved"].append(template_info)

            except Exception as e:
                logger.error(f"Error reading template {filename}: {e}")

        return jsonify(templates)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/template/approve", methods=["POST"])
def api_approve_template():
    """Approve a template and mark its analyses as valid."""
    try:
        data = request.json
        template_file = data.get("template")

        if not template_file or not os.path.exists(template_file):
            return jsonify({"error": "Template not found"}), 404

        # Load template
        with open(template_file, "r", encoding="utf-8") as f:
            template_data = json.load(f)

        template_name = template_data.get("name", os.path.basename(template_file))

        # Update status only if template is not already in production
        if template_data.get("status") != "PRODUCTION":
            template_data["status"] = "PRODUCTION"
            template_data["approved_at"] = datetime.now().isoformat()
            template_data["approved_by"] = "dashboard_user"

        # Save updated template
        with open(template_file, "w", encoding="utf-8") as f:
            json.dump(template_data, f, indent=2)

        # Update analyses created with this template
        analyzed_dir = "data/analyzed"
        if os.path.exists(analyzed_dir):
            for filename in os.listdir(analyzed_dir):
                if not filename.startswith("analysis_"):
                    continue
                filepath = os.path.join(analyzed_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as af:
                        analysis = json.load(af)
                    if analysis.get("template_info", {}).get("name") != template_name:
                        continue
                    if analysis.get("validation", {}).get("status") == "REVIEW":
                        analysis["validation"]["status"] = "VALID"
                        analysis["validation"]["approved_by"] = "dashboard_user"
                        analysis["validation"][
                            "approved_at"
                        ] = datetime.now().isoformat()
                        with open(filepath, "w", encoding="utf-8") as af:
                            json.dump(analysis, af, indent=2)
                except Exception as e:
                    logger.error(f"Failed to update analysis {filename}: {e}")

        add_log_message("success", f"Template approved: {template_file}")
        return jsonify({"status": "approved"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/template/test", methods=["POST"])
def api_test_template():
    """Test a template with a single article."""
    try:
        init_night_watcher()

        data = request.json or {}
        template = data.get("template", "standard_analysis.json")
        article_content = data.get("article_content")
        article_url = data.get("article_url")

        # Run test
        result = night_watcher.test_template(
            template=template, article_content=article_content, article_url=article_url
        )

        return jsonify(result)

    except Exception as e:
        logger.error(f"Template test error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/analysis/<analysis_id>")
def api_get_analysis(analysis_id):
    """Get a specific analysis result."""
    try:
        filepath = f"data/analyzed/{analysis_id}.json"
        if not os.path.exists(filepath):
            return jsonify({"error": "Analysis not found"}), 404

        with open(filepath, "r", encoding="utf-8") as f:
            analysis = json.load(f)

        return jsonify(analysis)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/analysis/recent")
def api_recent_analyses():
    """Get recent analysis results."""
    try:
        analyses = []
        analyzed_dir = "data/analyzed"

        if not os.path.exists(analyzed_dir):
            return jsonify([])

        # Get all analysis files
        files = []
        for filename in os.listdir(analyzed_dir):
            if filename.startswith("analysis_"):
                filepath = os.path.join(analyzed_dir, filename)
                mtime = os.path.getmtime(filepath)
                files.append((filename, filepath, mtime))

        # Sort by modification time (newest first)
        files.sort(key=lambda x: x[2], reverse=True)

        # Get top 20
        for filename, filepath, mtime in files[:20]:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    analysis = json.load(f)

                analyses.append(
                    {
                        "id": filename[:-5],  # Remove .json
                        "filename": filename,
                        "timestamp": analysis.get("timestamp"),
                        "template": analysis.get("template_info", {}).get(
                            "name", "Unknown"
                        ),
                        "article_title": analysis.get("article", {}).get(
                            "title", "Unknown"
                        ),
                        "article_source": analysis.get("article", {}).get(
                            "source", "Unknown"
                        ),
                        "validation_status": analysis.get("validation", {}).get(
                            "status", "Unknown"
                        ),
                        "manipulation_score": analysis.get("manipulation_score", 0),
                        "concern_level": analysis.get("concern_level", "Unknown"),
                    }
                )
            except Exception as e:
                logger.error(f"Error reading {filename}: {e}")

        return jsonify(analyses)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/collect", methods=["POST"])
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
                add_log_message(
                    "error", f"{event.get('source')}: {event.get('message')}"
                )
            elif event.get("type") == "cancelled":
                add_log_message("warning", "Collection cancelled")

        try:
            init_night_watcher()
            result = night_watcher.collect(mode=mode, callback=progress)

            articles_count = len(result.get("articles", []))
            if night_watcher.collector.cancelled:
                add_log_message("warning", "Collection stopped by user")
            else:
                add_log_message(
                    "success", f"Collection completed: {articles_count} articles"
                )
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


@app.route("/api/collect/stop", methods=["POST"])
def api_collect_stop():
    """Request stopping an ongoing collection."""
    try:
        if night_watcher and night_watcher.collector:
            night_watcher.collector.cancel()
            return jsonify({"status": "stopping"})
        return jsonify({"error": "Collector not running"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """Start content analysis."""
    global task_thread, task_status

    if task_status["running"]:
        return jsonify({"error": "Another task is already running"}), 400

    data = request.json or {}
    max_articles = data.get("max_articles", 20)
    templates = data.get("templates", [])
    target = data.get("target", "unanalyzed")
    since_date = data.get("since_date")

    if not templates:
        return jsonify({"error": "No templates selected"}), 400

    def run_analysis():
        task_status["running"] = True
        task_status["task"] = "analysis"
        task_status["progress"] = 0
        add_log_message(
            "info", f"Starting analysis (templates: {len(templates)}, target: {target})"
        )

        try:
            init_night_watcher()

            # Check if analyzer is available
            if not night_watcher.analyzer:
                add_log_message("error", "Analyzer not available - check LLM provider")
                task_status["progress"] = 100
                return

            # Run analysis
            result = night_watcher.analyze(
                max_articles=max_articles,
                templates=templates,
                target=target,
                since_date=since_date,
            )

            if result.get("status") == "no_documents":
                add_log_message(
                    "warning", "No documents to analyze for selected target"
                )
            else:
                analyzed_count = result.get("analyzed", 0)
                add_log_message(
                    "success",
                    f"Analysis completed: {analyzed_count} documents with {len(templates)} templates",
                )

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


@app.route("/api/review-queue")
def api_review_queue():
    """Get review queue items."""
    try:
        review_items = scan_for_review_items()
        return jsonify(review_items)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/review-queue/approve", methods=["POST"])
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

        with open(filepath, "r", encoding="utf-8") as f:
            analysis = json.load(f)

        # Update validation status
        analysis["validation"]["status"] = "VALID"
        analysis["validation"]["approved_by"] = "dashboard_user"
        analysis["validation"]["approved_at"] = datetime.now().isoformat()
        analysis["validation"]["notes"] = notes

        # Save updated analysis
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2)

        # Clear cache to force status refresh
        stats_cache["last_update"] = None

        return jsonify({"status": "approved"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/review-queue/reject", methods=["POST"])
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

        with open(filepath, "r", encoding="utf-8") as f:
            analysis = json.load(f)

        # Update validation status
        analysis["validation"]["status"] = "REJECTED"
        analysis["validation"]["rejected_by"] = "dashboard_user"
        analysis["validation"]["rejected_at"] = datetime.now().isoformat()
        analysis["validation"]["notes"] = notes

        # Save updated analysis
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2)

        # Clear cache to force status refresh
        stats_cache["last_update"] = None

        return jsonify({"status": "rejected"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/build-kg", methods=["POST"])
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
            add_log_message(
                "success",
                f"KG built: {processed} analyses, {temporal} temporal relations",
            )
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


@app.route("/api/sync-vectors", methods=["POST"])
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
            add_log_message(
                "success", f"Vector sync completed: {nodes_added} nodes added"
            )
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


@app.route("/api/task-status")
def api_task_status():
    """Get current task status and logs."""
    return jsonify(
        {
            "running": task_status["running"],
            "task": task_status["task"],
            "progress": task_status["progress"],
            "messages": task_status["messages"][-20:],  # Last 20 messages
        }
    )


@app.route("/api/public-key", methods=["GET", "POST"])
def api_public_key():
    """Retrieve or update stored public key."""
    init_night_watcher()

    if request.method == "GET":
        if os.path.exists(PUBLIC_KEY_FILE):
            with open(PUBLIC_KEY_FILE, "r", encoding="utf-8") as f:
                key = f.read()
        else:
            key = ""
        return jsonify({"public_key": key})

    data = request.json or {}
    key = data.get("public_key", "")
    os.makedirs(os.path.dirname(PUBLIC_KEY_FILE), exist_ok=True)
    with open(PUBLIC_KEY_FILE, "w", encoding="utf-8") as f:
        f.write(key)

    night_watcher.config.setdefault("export", {})["public_key"] = PUBLIC_KEY_FILE
    with open(night_watcher.config_path, "w", encoding="utf-8") as f:
        json.dump(night_watcher.config, f, indent=2)

    add_log_message("success", "Public key saved")
    return jsonify({"status": "saved"})


@app.route("/api/export", methods=["POST"])
def api_export():
    """Create a signed export archive."""
    try:
        init_night_watcher()
        data = request.json or {}
        filename = (
            data.get("filename")
            or f"night_watcher_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tar.gz"
        )
        export_dir = os.path.join(night_watcher.base_dir, "exports")
        os.makedirs(export_dir, exist_ok=True)
        output_path = os.path.join(export_dir, filename)

        # Import here to avoid circular imports
        from export_artifact import export_artifact

        private_key = night_watcher.config.get("export", {}).get("private_key")
        public_key = night_watcher.config.get("export", {}).get(
            "public_key", PUBLIC_KEY_FILE if os.path.exists(PUBLIC_KEY_FILE) else None
        )

        export_artifact(
            output_path,
            kg_dir=os.path.join(night_watcher.base_dir, "knowledge_graph"),
            vector_dir=os.path.join(night_watcher.base_dir, "vector_store"),
            documents_dir=os.path.join(night_watcher.base_dir, "documents"),
            private_key_path=private_key,
            public_key_path=public_key,
        )

        add_log_message("success", f"Exported package: {filename}")
        return jsonify({"status": "exported", "path": output_path})

    except Exception as e:
        logger.error(f"Export error: {e}")
        return jsonify({"error": str(e)}), 400


@app.route("/api/export/status")
def api_export_status():
    """Get export system status and version info."""
    init_night_watcher()
    orchestrator = night_watcher.get_export_orchestrator()
    version = orchestrator.version_mgr.get_current_version()
    return jsonify({"current_version": version})


@app.route("/api/export/staging")
def api_staging_list():
    """List files in staging area."""
    init_night_watcher()
    orchestrator = night_watcher.get_export_orchestrator()
    files = orchestrator.staging_mgr.list_staged_files()
    return jsonify({"files": files})


@app.route("/api/export/staging/add", methods=["POST"])
def api_staging_add():
    """Add file to staging area."""
    init_night_watcher()
    path = (request.json or {}).get("file")
    orchestrator = night_watcher.get_export_orchestrator()
    orchestrator.staging_mgr.add_file(path)
    return jsonify({"status": "added"})


@app.route("/api/export/staging/remove", methods=["POST"])
def api_staging_remove():
    """Remove file from staging area."""
    init_night_watcher()
    data = request.json or {}
    orchestrator = night_watcher.get_export_orchestrator()
    if data.get("clear"):
        orchestrator.staging_mgr.clear_staging()
    else:
        orchestrator.staging_mgr.remove_file(data.get("file"))
    return jsonify({"status": "removed"})


@app.route("/api/export/validate-keys", methods=["POST"])
def api_validate_keys():
    """Validate private/public key pair."""
    data = request.json or {}
    result = validate_key_pair(data.get("private_key", ""), data.get("public_key", ""))
    status = 200 if result.get("valid") else 400
    return jsonify(result), status


@app.route("/api/export/create-package", methods=["POST"])
def api_create_package_v2():
    """Create package with user provided keys."""
    try:
        init_night_watcher()
        data = request.json or {}
        package_type = data.get("package_type", "v001")
        private_key = data.get("private_key", "")
        public_key = data.get("public_key", "")
        staging_files = data.get("staging_files", [])

        validation = validate_key_pair(private_key, public_key)
        if not validation["valid"]:
            return (
                jsonify({"error": f"Key validation failed: {validation['message']}"}),
                400,
            )

        orchestrator = night_watcher.get_export_orchestrator()
        result = orchestrator.create_package(
            package_type, private_key, public_key, staging_files
        )

        if result.get("success", True):
            return jsonify(result)
        else:
            return jsonify({"error": result.get("error", "unknown")}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/export/create", methods=["POST"])
def api_create_package():
    """Create distribution package."""
    init_night_watcher()
    data = request.json or {}
    orchestrator = night_watcher.get_export_orchestrator()
    pkg_type = data.get("type", "v001")

    priv = data.get("private_key")
    pub = data.get("public_key")
    full_v2 = data.get("full_since_v2", False)

    if pkg_type == "v001":
        result = orchestrator.create_v001_package(priv, pub)
    else:
        result = orchestrator.create_update_package(
            pkg_type, priv, pub, full_since_v2=full_v2
        )

    return jsonify(result)


@app.route("/api/export/history")
def api_export_history():
    """Get export history and logs."""
    init_night_watcher()
    orchestrator = night_watcher.get_export_orchestrator()
    history = orchestrator.version_mgr.get_export_history()
    return jsonify(history)


# ---------------------------------------------------------------------------
# Event aggregation endpoints
# ---------------------------------------------------------------------------

@app.route("/api/aggregate-events", methods=["POST"])
def api_aggregate_events():
    """Run event aggregation across recent analyses."""
    init_night_watcher()
    data = request.json or {}
    window = int(data.get("analysis_window", 7))

    result = night_watcher.aggregate_events(analysis_window=window)
    cache_aggregation_results(result)
    return jsonify(result)


@app.route("/api/events/latest")
def api_latest_events():
    """Return most recent aggregation results."""
    latest = load_latest_aggregation()
    return jsonify(latest)


@app.route("/api/events/timeline")
def api_event_timeline():
    """Timeline-ready list of events sorted by date."""
    latest = load_latest_aggregation()
    events = latest.get("events", [])
    events.sort(key=lambda e: e.get("date") or "")
    return jsonify(events)


@app.route("/api/events/campaigns")
def api_campaigns():
    """Return details on detected coordinated campaigns."""
    latest = load_latest_aggregation()
    return jsonify(latest.get("coordinated_campaigns", []))


@app.route("/api/config", methods=["GET", "POST"])
def api_config():
    """Get or update configuration."""
    init_night_watcher()

    if request.method == "GET":
        return jsonify(night_watcher.config)

    elif request.method == "POST":
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
                if night_watcher.llm_provider and hasattr(
                    night_watcher.llm_provider, "update_model"
                ):
                    night_watcher.llm_provider.update_model(model)
                    add_log_message("success", f"Model updated to: {model}")

            # Merge new values into existing config
            night_watcher.config = deep_update(night_watcher.config, new_config)

            # Save to file
            with open(night_watcher.config_path, "w", encoding="utf-8") as f:
                json.dump(night_watcher.config, f, indent=2)

            # Reinitialize components if needed
            if "llm_provider" in new_config and (
                "type" in new_config["llm_provider"]
                or "api_key" in new_config["llm_provider"]
            ):
                night_watcher.llm_provider = providers.initialize_llm_provider(
                    night_watcher.config
                )
                night_watcher.analyzer = (
                    ContentAnalyzer(night_watcher.llm_provider)
                    if night_watcher.llm_provider
                    else None
                )
                add_log_message("success", "LLM provider reinitialized")

            add_log_message("success", "Configuration updated")
            return jsonify({"status": "updated"})

        except Exception as e:
            logger.error(f"Config update error: {e}")
            return jsonify({"error": str(e)}), 400


@app.route("/api/llm-models")
def api_llm_models():
    """Return available models from LM Studio."""
    init_night_watcher()
    try:
        if night_watcher.llm_provider and hasattr(
            night_watcher.llm_provider, "list_models"
        ):
            models = night_watcher.llm_provider.list_models()
            return jsonify(models)
        else:
            return jsonify([])
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return jsonify([])


@app.route("/api/sources")
def api_sources():
    """Get content sources."""
    init_night_watcher()
    sources = night_watcher.config.get("content_collection", {}).get("sources", [])
    return jsonify(sources)


@app.route("/api/sources/add", methods=["POST"])
def api_add_source():
    """Add a new content source."""
    try:
        init_night_watcher()
        source_data = request.json

        if night_watcher.collector.add_source(source_data):
            # Save config
            with open(night_watcher.config_path, "w", encoding="utf-8") as f:
                json.dump(night_watcher.config, f, indent=2)

            add_log_message(
                "success",
                f"Added source: {source_data.get('name', source_data.get('url'))}",
            )
            return jsonify({"status": "added"})
        else:
            return jsonify({"error": "Failed to add source"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/sources/update", methods=["POST"])
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

        with open(night_watcher.config_path, "w", encoding="utf-8") as f:
            json.dump(night_watcher.config, f, indent=2)

        add_log_message("success", f"Updated source limit for {url}")
        return jsonify({"status": "updated"})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/sources/set-limit", methods=["POST"])
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

        with open(night_watcher.config_path, "w", encoding="utf-8") as f:
            json.dump(night_watcher.config, f, indent=2)

        add_log_message("success", f"Set article limit for all sources: {limit}")
        return jsonify({"status": "updated"})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/search", methods=["POST"])
def api_search():
    """Search vector store."""
    try:
        init_night_watcher()
        data = request.json
        query = data.get("query", "")
        node_type = data.get("node_type")
        limit = data.get("limit", 10)

        results = night_watcher.vector_store.search(
            query=query, item_type=node_type, limit=limit
        )

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/test-llm", methods=["POST"])
def api_test_llm():
    """Test LLM connection."""
    try:
        init_night_watcher()

        if not night_watcher.llm_provider:
            return jsonify({"error": "No LLM provider configured"}), 400

        # Simple test prompt
        result = night_watcher.llm_provider.complete(
            prompt="Say 'Connection successful!' and nothing else.", max_tokens=20
        )

        if "error" in result:
            return jsonify({"error": result["error"]}), 400

        # Extract text from response
        text = result.get("choices", [{}])[0].get("text", "")

        return jsonify(
            {
                "status": "connected",
                "response": text,
                "model": getattr(night_watcher.llm_provider, "model", "unknown"),
            }
        )

    except Exception as e:
        logger.error(f"LLM test error: {e}")
        return jsonify({"error": str(e)}), 400


@app.route("/api/pipeline", methods=["POST"])
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
                    add_log_message(
                        "success", f"Processed {kg_result.get('processed', 0)} analyses"
                    )

                    # Vector Sync
                    add_log_message("info", "Synchronizing vectors")
                    task_status["progress"] = 90
                    vector_result = night_watcher.sync_vectors()
                    add_log_message(
                        "success",
                        f"Synced {vector_result.get('nodes_added', 0)} vectors",
                    )

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
