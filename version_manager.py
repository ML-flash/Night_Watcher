#!/usr/bin/env python3
"""
Version management for Night_watcher distribution releases.
"""

import os
import json
from typing import Optional, List
from datetime import datetime

class VersionManager:
    """Manages version tracking for distribution releases."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.version_file = os.path.join(data_dir, "release_versions.json")
        self.versions = self._load_versions()

    def _load_versions(self) -> dict:
        """Load version history from disk."""
        if os.path.exists(self.version_file):
            with open(self.version_file, 'r') as f:
                return json.load(f)
        return {"releases": [], "current": None}

    def _save_versions(self):
        """Save version history to disk."""
        with open(self.version_file, 'w') as f:
            json.dump(self.versions, f, indent=2)

    def get_next_version(self) -> str:
        """Calculate next version number."""
        if not self.versions["releases"]:
            return "v001"
        last_version = self.versions["releases"][-1]["version"]
        version_num = int(last_version[1:]) + 1
        return f"v{version_num:03d}"

    def validate_version_sequence(self, new_version: str, previous_version: str = None) -> bool:
        """Validate version follows proper sequence."""
        if new_version == "v001":
            return len(self.versions["releases"]) == 0
        if not previous_version:
            return False
        expected_prev = self.get_current_version()
        return previous_version == expected_prev

    def record_release(self, version: str, file_path: str, file_hash: str):
        """Record a new release in version history."""
        release_record = {
            "version": version,
            "file_path": file_path,
            "file_hash": file_hash,
            "created_at": datetime.now().isoformat()
        }
        self.versions["releases"].append(release_record)
        self.versions["current"] = version
        self._save_versions()

    def get_current_version(self) -> Optional[str]:
        """Get current/latest version."""
        return self.versions.get("current")

    def get_release_history(self) -> List[dict]:
        """Get all release history."""
        return self.versions.get("releases", [])
