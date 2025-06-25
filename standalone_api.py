#!/usr/bin/env python3
"""
Standalone API server for distribution client.
Provides web interface for installed intelligence data.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import sqlite3
import os

app = Flask(__name__)
CORS(app)

def get_db_connection():
    """Connect to distribution client database."""
    db_path = "intelligence.db"
    if not os.path.exists(db_path):
        raise FileNotFoundError("Intelligence database not found")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/api/status')
def api_status():
    """Get intelligence database status."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM nodes")
        node_count = cursor.fetchone()["count"]
        cursor.execute("SELECT COUNT(*) as count FROM edges")
        edge_count = cursor.fetchone()["count"]
        return jsonify({
            "nodes": node_count,
            "edges": edge_count,
            "database_ready": True
        })
    except Exception as e:
        return jsonify({"error": str(e), "database_ready": False})

@app.route('/api/search')
def api_search():
    """Search intelligence data."""
    pass

@app.route('/api/analyze/<entity_id>')
def api_analyze(entity_id):
    """Analyze specific entity."""
    pass

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
