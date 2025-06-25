import sqlite3
import json
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    event_signature TEXT NOT NULL,
    event_type TEXT,
    first_seen TEXT NOT NULL,
    last_updated TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    source_count INTEGER DEFAULT 1,
    source_diversity REAL DEFAULT 0.0,
    confidence_score REAL DEFAULT 0.0,
    status TEXT DEFAULT 'active',
    core_attributes TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS event_observations (
    observation_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    source_doc_id TEXT NOT NULL,
    source_name TEXT,
    source_bias TEXT,
    observed_at TEXT NOT NULL,
    extracted_data TEXT NOT NULL,
    analysis_id TEXT NOT NULL,
    citations TEXT,
    FOREIGN KEY (event_id) REFERENCES events(event_id)
);

CREATE TABLE IF NOT EXISTS match_decisions (
    decision_id TEXT PRIMARY KEY,
    signature_1 TEXT NOT NULL,
    signature_2 TEXT NOT NULL,
    decision TEXT NOT NULL,
    confidence REAL,
    method TEXT,
    reasoning TEXT,
    decided_at TEXT NOT NULL,
    llm_prompt TEXT,
    llm_response TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_weight ON events(weight DESC);
CREATE INDEX IF NOT EXISTS idx_events_signature ON events(event_signature);
CREATE INDEX IF NOT EXISTS idx_observations_event ON event_observations(event_id);
CREATE INDEX IF NOT EXISTS idx_observations_source ON event_observations(source_doc_id);
"""

class EventDatabase:
    def __init__(self, db_path: str = "data/events.db"):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        with self._get_connection() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def execute(self, query: str, params: List = None) -> None:
        with self._get_connection() as conn:
            cur = conn.cursor()
            if params:
                cur.execute(query, params)
            else:
                cur.execute(query)
            conn.commit()

    def query(self, query: str, params: List = None) -> List[Dict]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            if params:
                cur.execute(query, params)
            else:
                cur.execute(query)
            rows = cur.fetchall()
            return [dict(r) for r in rows]

    def get_event(self, event_id: str) -> Optional[Dict]:
        results = self.query("SELECT * FROM events WHERE event_id = ?", [event_id])
        if results:
            event = results[0]
            event['core_attributes'] = json.loads(event['core_attributes'])
            return event
        return None
