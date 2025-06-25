"""Event Weight Tracking System.

Provides classes for generating event signatures, matching events,
calculating weights, and managing event observations.
"""

import logging
import json
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

    def __init__(self, llm_provider=None, db=None):
        self.signature_gen = EventSignature()
        self.llm = llm_provider
        self.db = db
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
        cache_key = self._get_cache_key(new_event, potential_matches)
        if cache_key in self.match_cache:
            return self.match_cache[cache_key]

        for match in potential_matches:
            existing_event = self.db.get_event(match["event_id"])
            if not existing_event:
                continue
            existing_attrs = existing_event.get("core_attributes", {})

            prompt = f"""Determine if these news events describe the SAME specific incident.

Event 1:
- Actor: {new_event.get('primary_actor', 'Unknown')}
- Action: {new_event.get('action', 'Unknown')}
- Date: {new_event.get('date', 'Unknown')}
- Location: {new_event.get('location', 'Unknown')}
- Context: {new_event.get('context', 'None provided')}

Event 2:
- Actor: {existing_attrs.get('primary_actor', 'Unknown')}
- Action: {existing_attrs.get('action', 'Unknown')}
- Date: {existing_attrs.get('date', 'Unknown')}
- Location: {existing_attrs.get('location', 'Unknown')}
- Context: {existing_attrs.get('context', 'None provided')}

Consider:
- Could date differences be due to timezone or reporting delays?
- Are different names/titles referring to the same person?
- Do action variations describe the same act?
- Is the location the same despite different descriptions?

Respond with ONLY one of these words followed by a brief reason:
SAME - if these describe the exact same incident
DIFFERENT - if these are separate incidents
RELATED - if connected but distinct events

Response:"""

            try:
                response = self.llm.complete(prompt, max_tokens=150, temperature=0.1)
                decision, reasoning = self._parse_llm_response(response)
                self._cache_decision(cache_key, decision, reasoning, prompt, response)
                if decision == "SAME":
                    return match["event_id"]
            except Exception as e:
                logger.error(f"LLM resolution error: {e}")
                continue

        self.match_cache[cache_key] = None
        return None

    def _get_cache_key(self, new_event: Dict, matches: List[Dict]) -> str:
        return str((json.dumps(new_event, sort_keys=True), tuple(m["event_id"] for m in matches)))

    def _parse_llm_response(self, text: str) -> (str, str):
        first_line = text.strip().split("\n")[0]
        parts = first_line.split(None, 1)
        decision = parts[0].upper() if parts else "DIFFERENT"
        reasoning = parts[1].strip() if len(parts) > 1 else ""
        return decision, reasoning

    def _cache_decision(self, key: str, decision: str, reasoning: str, prompt: str, response: str) -> None:
        self.match_cache[key] = None if decision != "SAME" else decision
        try:
            decision_id = f"dec_{datetime.utcnow().strftime('%Y%m%d_%H%M%S%f')}"
            self.db.execute(
                """INSERT INTO match_decisions (decision_id, signature_1, signature_2, decision, confidence, method, reasoning, decided_at, llm_prompt, llm_response)
                    VALUES (?,?,?,?,?,?,?,?,?,?)""",
                [
                    decision_id,
                    key,
                    '',
                    decision,
                    None,
                    'llm',
                    reasoning,
                    datetime.utcnow().isoformat() + 'Z',
                    prompt,
                    response,
                ],
            )
        except Exception as e:
            logger.error(f"Failed to cache decision: {e}")


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
        self.matcher = EventMatcher(llm_provider, db_connection)
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
            "INSERT INTO events (event_id, event_signature, event_type, first_seen, last_updated, weight, core_attributes) VALUES (?,?,?,?,?,?,?)",
            [
                event_id,
                signature,
                event_data.get("action"),
                now,
                now,
                1,
                json.dumps(event_data),
            ],
        )
        return event_id

    def save_observation(self, observation: Dict) -> None:
        self.db.execute(
            "INSERT INTO event_observations (observation_id, event_id, source_doc_id, source_name, source_bias, observed_at, extracted_data, analysis_id, citations) VALUES (?,?,?,?,?,?,?,?,?)",
            [
                f"obs_{datetime.utcnow().strftime('%Y%m%d_%H%M%S%f')}",
                observation["event_id"],
                observation["source_doc_id"],
                observation["source_name"],
                observation.get("source_bias"),
                datetime.utcnow().isoformat() + "Z",
                json.dumps(observation.get("extracted_data", {})),
                observation["analysis_id"],
                json.dumps(observation.get("citations", [])),
            ],
        )

    def update_event_weight(self, event_id: str) -> None:
        observations = self.db.query(
            "SELECT source_name, source_bias, extracted_data FROM event_observations WHERE event_id=?",
            [event_id],
        )
        obs_list = [
            {
                "source_name": o["source_name"],
                "source_bias": o["source_bias"],
                "extracted_data": json.loads(o["extracted_data"]),
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

