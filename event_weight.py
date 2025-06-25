"""Event Weight Tracking System.

Provides classes for generating event signatures, matching events,
calculating weights, and managing event observations.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class EventSignature:
    """Generate normalized signatures for event matching."""

    def __init__(self):
        self.action_equivalents = {
            "arrested": ["detained", "taken into custody", "apprehended"],
            "fired": ["dismissed", "removed", "terminated"],
            "passed": ["approved", "enacted", "adopted"],
        }

    def generate_signature(self, event_data: Dict) -> str:
        """Generate a normalized signature for event matching."""
        actor = self.normalize_actor(event_data.get("primary_actor", ""))
        action = self.normalize_action(event_data.get("action", ""))
        date = self.normalize_date(event_data.get("date", ""))
        location = self.normalize_location(event_data.get("location", ""))
        return f"{actor}|{action}|{date}|{location}"

    def normalize_actor(self, actor: str) -> str:
        """Normalize actor names by removing titles and lowercasing."""
        if not actor:
            return ""
        actor = actor.lower()
        for title in ["judge", "mr.", "mrs.", "ms.", "dr."]:
            actor = actor.replace(title, "")
        actor = actor.replace("  ", " ").strip()
        return actor.replace(" ", "_")

    def normalize_action(self, action: str) -> str:
        """Normalize actions with synonym mapping."""
        if not action:
            return ""
        action = action.lower().strip()
        for key, vals in self.action_equivalents.items():
            if action == key or action in vals:
                return key
        return action

    def normalize_date(self, date_str: str) -> str:
        """Normalize date to ISO format if possible."""
        if not date_str:
            return ""
        for fmt in ["%Y-%m-%d", "%B %d, %Y", "%b %d, %Y"]:
            try:
                return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
            except Exception:
                continue
        return date_str

    def normalize_location(self, location: str) -> str:
        """Normalize location names."""
        if not location:
            return ""
        return location.lower().replace(" ", "_")

    def calculate_similarity(self, sig1: str, sig2: str) -> float:
        """Calculate simple similarity between two signatures."""
        if not sig1 or not sig2:
            return 0.0
        parts1 = sig1.split("|")
        parts2 = sig2.split("|")
        matches = sum(1 for a, b in zip(parts1, parts2) if a and a == b)
        max_len = max(len(parts1), len(parts2))
        return matches / max_len


class EventMatcher:
    """Find matching events using signatures and optional LLM assistance."""

    def __init__(self, llm_provider=None):
        self.signature_gen = EventSignature()
        self.llm = llm_provider
        self.match_cache: Dict[str, Optional[str]] = {}

    def find_matching_event(self, new_event: Dict, existing_events: List[Dict]) -> Optional[str]:
        new_sig = self.signature_gen.generate_signature(new_event)
        matches = []
        for existing in existing_events:
            similarity = self.signature_gen.calculate_similarity(new_sig, existing.get("event_signature", ""))
            if similarity > 0.9:
                return existing.get("event_id")
            if similarity > 0.6:
                matches.append({"event_id": existing.get("event_id"), "similarity": similarity})
        if matches and self.llm:
            return self.resolve_with_llm(new_event, matches)
        return None

    def resolve_with_llm(self, new_event: Dict, potential_matches: List[Dict]) -> Optional[str]:
        cache_key = str((new_event, tuple(m["event_id"] for m in potential_matches)))
        if cache_key in self.match_cache:
            return self.match_cache[cache_key]
        for match in potential_matches:
            prompt = (
                "Are these two events the same?\n\n"
                f"Event 1 actor: {new_event.get('primary_actor')} action: {new_event.get('action')} date: {new_event.get('date')}\n"
                f"Event 2 actor: {match['event_id']}\n"
            )
            try:
                response = self.llm.complete(prompt, max_tokens=20)
            except Exception as e:
                logger.error(f"LLM error: {e}")
                response = "DIFFERENT"
            decision = response.strip().split()[0].upper()
            self.match_cache[cache_key] = match["event_id"] if decision == "SAME" else None
            if decision == "SAME":
                return match["event_id"]
        self.match_cache[cache_key] = None
        return None


class EventWeightCalculator:
    """Calculate event weight and confidence metrics."""

    def calculate_weight(self, event_observations: List[Dict]) -> Dict:
        base_weight = len(event_observations)
        unique_biases = {obs.get("source_bias") for obs in event_observations if obs.get("source_bias")}
        source_diversity = len(unique_biases) / base_weight if base_weight else 0
        dates = [obs.get("extracted_data", {}).get("date") for obs in event_observations]
        temporal_consistency = 1.0 if len(set(dates)) <= 1 else 0.5
        variation_score = 1.0
        weight = base_weight * source_diversity * temporal_consistency
        confidence = (source_diversity + temporal_consistency + variation_score) / 3
        return {
            "weight": weight,
            "source_count": base_weight,
            "source_diversity": source_diversity,
            "confidence_score": confidence,
            "metrics": {
                "temporal_consistency": temporal_consistency,
                "variation_score": variation_score,
                "unique_sources": len({obs.get("source_name") for obs in event_observations}),
            },
        }


class EventManager:
    """Manage events and observations."""

    def __init__(self, db_connection, llm_provider=None):
        self.db = db_connection
        self.matcher = EventMatcher(llm_provider)
        self.weight_calc = EventWeightCalculator()

    def add_event_observation(self, event_data: Dict, source_doc: Dict, analysis_id: str) -> str:
        existing = self.get_recent_events()
        match_id = self.matcher.find_matching_event(event_data, existing)
        if match_id:
            event_id = match_id
        else:
            event_id = self.create_new_event(event_data)
        observation = {
            "event_id": event_id,
            "source_doc_id": source_doc["doc_id"],
            "source_name": source_doc["source"],
            "source_bias": source_doc.get("bias_label"),
            "extracted_data": event_data,
            "analysis_id": analysis_id,
            "citations": event_data.get("citations", []),
        }
        self.save_observation(observation)
        self.update_event_weight(event_id)
        return event_id

    # --- Database helper methods (simple placeholders) ---
    def get_recent_events(self) -> List[Dict]:
        try:
            return self.db.query("SELECT * FROM events ORDER BY last_updated DESC LIMIT 100")
        except Exception:
            return []

    def create_new_event(self, event_data: Dict) -> str:
        event_id = f"evt_{datetime.utcnow().strftime('%Y%m%d_%H%M%S%f')}"
        signature = self.matcher.signature_gen.generate_signature(event_data)
        now = datetime.utcnow().isoformat() + "Z"
        self.db.execute(
            "INSERT INTO events (event_id, event_signature, event_type, first_seen, last_updated, weight) VALUES (?,?,?,?,?,?)",
            [event_id, signature, event_data.get("action"), now, now, 1],
        )
        return event_id

    def save_observation(self, observation: Dict) -> None:
        self.db.execute(
            "INSERT INTO observations (event_id, source_doc_id, source_name, source_bias, observed_at, data, analysis_id) VALUES (?,?,?,?,?,?,?)",
            [
                observation["event_id"],
                observation["source_doc_id"],
                observation["source_name"],
                observation.get("source_bias"),
                datetime.utcnow().isoformat() + "Z",
                str(observation.get("extracted_data")),
                observation["analysis_id"],
            ],
        )

    def update_event_weight(self, event_id: str) -> None:
        observations = self.db.query(
            "SELECT source_name, source_bias, data FROM observations WHERE event_id=?",
            [event_id],
        )
        obs_list = [
            {
                "source_name": o[0],
                "source_bias": o[1],
                "extracted_data": eval(o[2]),
            }
            for o in observations
        ]
        metrics = self.weight_calc.calculate_weight(obs_list)
        self.db.execute(
            "UPDATE events SET weight=?, source_count=?, source_diversity=?, confidence_score=?, last_updated=? WHERE event_id=?",
            [
                metrics["weight"],
                metrics["source_count"],
                metrics["source_diversity"],
                metrics["confidence_score"],
                datetime.utcnow().isoformat() + "Z",
                event_id,
            ],
        )

    def get_weighted_events(self, min_weight: float = 2.0) -> List[Dict]:
        return self.db.query("SELECT * FROM events WHERE weight >= ? ORDER BY weight DESC", [min_weight])

