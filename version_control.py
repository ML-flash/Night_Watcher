import os
import json
from file_utils import safe_json_load, safe_json_save
import logging
import shutil
from datetime import datetime
from typing import List, Dict


class IntegratedVersionManager:
    """Version control integrated with main Night_watcher system."""

    def __init__(self, night_watcher_instance):
        self.nw = night_watcher_instance
        self.version_file = "data/version_control.json"
        self.staging_dir = "data/export_staging"
        os.makedirs(self.staging_dir, exist_ok=True)
        self.logger = logging.getLogger("VersionManager")

    def get_current_version(self) -> str:
        data = self._load()
        return data.get("current_version")

    def get_next_version(self) -> str:
        cur = self.get_current_version()
        if not cur:
            return "v001"
        try:
            num = int(cur.lstrip('v'))
            return f"v{num+1:03d}"
        except Exception:
            return "v001"

    def validate_export_readiness(self) -> Dict[str, bool]:
        return {"ready": True}

    def log_export_attempt(self, version: str, success: bool, details: Dict):
        data = self._load()
        history = data.get("history", [])
        history.append({
            "version": version,
            "success": success,
            "details": details,
            "time": datetime.utcnow().isoformat() + "Z",
        })
        if success:
            data["current_version"] = version
        data["history"] = history
        self._save(data)

    def get_export_history(self) -> List[Dict]:
        return self._load().get("history", [])

    def _load(self) -> Dict:
        if os.path.exists(self.version_file):
            data = safe_json_load(self.version_file, default=None)
            if data is not None:
                return data
        return {}

    def _save(self, data: Dict):
        os.makedirs(os.path.dirname(self.version_file), exist_ok=True)
        safe_json_save(self.version_file, data)


class StagingManager:
    """Manages staging area for package preparation."""

    def __init__(self, staging_dir: str = "data/export_staging"):
        self.staging_dir = staging_dir
        os.makedirs(self.staging_dir, exist_ok=True)

    def add_file(self, file_path: str, category: str = "bundled"):
        if not file_path:
            return
        dest = os.path.join(self.staging_dir, os.path.basename(file_path))
        if os.path.isdir(file_path):
            shutil.copytree(file_path, dest, dirs_exist_ok=True)
        elif os.path.isfile(file_path):
            shutil.copy2(file_path, dest)

    def remove_file(self, file_path: str):
        target = os.path.join(self.staging_dir, os.path.basename(file_path))
        if os.path.exists(target):
            if os.path.isdir(target):
                shutil.rmtree(target)
            else:
                os.remove(target)

    def list_staged_files(self) -> List[str]:
        if not os.path.exists(self.staging_dir):
            return []
        return sorted(os.listdir(self.staging_dir))

    def clear_staging(self):
        if os.path.exists(self.staging_dir):
            for name in os.listdir(self.staging_dir):
                path = os.path.join(self.staging_dir, name)
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)

    def validate_staging(self) -> Dict[str, bool]:
        return {"valid": True}
