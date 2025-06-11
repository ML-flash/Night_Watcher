import os
import json
import logging
import tempfile
from typing import Dict, Optional

from version_control import IntegratedVersionManager, StagingManager


class ExportOrchestrator:
    """Manages complete export process with extensive logging."""

    def __init__(self, night_watcher_instance):
        self.nw = night_watcher_instance
        self.version_mgr = IntegratedVersionManager(night_watcher_instance)
        self.staging_mgr = StagingManager()
        self.logger = self._setup_export_logger()

    def _setup_export_logger(self):
        logger = logging.getLogger("ExportOrchestrator")
        if not logger.handlers:
            os.makedirs("data/export_history", exist_ok=True)
            handler = logging.FileHandler("data/export_history/orchestrator.log")
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def _log_export_step(self, step: str, status: str, details: Dict):
        self.logger.info(f"{step} - {status} - {details}")

    def _create_export_report(self, export_result: Dict):
        report_path = "data/export_history/export_report.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(export_result, f, indent=2)

    def _validate_prerequisites(self) -> Dict[str, bool]:
        return {
            "intelligence_data": True,
            "private_key": True,
            "staging_area": True,
            "version_sequence": True,
        }

    def _write_temp_key(self, text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        tmp = tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8")
        tmp.write(text)
        tmp.close()
        return tmp.name

    def create_v001_package(self, private_key: Optional[str] = None, public_key: Optional[str] = None) -> Dict:
        self._log_export_step("start", "info", {"type": "v001"})
        version = "v001"
        out_path = os.path.join("data/export_packages", f"night_watcher_{version}.tar.gz")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        pk_file = self._write_temp_key(private_key)
        try:
            from export_versioned_artifact import export_versioned_artifact
            export_versioned_artifact(
                out_path,
                version=version,
                private_key_path=pk_file,
                kg_dir=os.path.join(self.nw.base_dir, "knowledge_graph"),
                vector_dir=os.path.join(self.nw.base_dir, "vector_store"),
                documents_dir=os.path.join(self.nw.base_dir, "documents"),
                previous_artifact_path=None,
                bundled_files=[os.path.join(self.staging_mgr.staging_dir, f) for f in self.staging_mgr.list_staged_files()],
            )
            success = True
        except Exception as e:
            self._log_export_step("error", "failed", {"error": str(e)})
            success = False
            out_path = ""
        finally:
            if pk_file:
                os.unlink(pk_file)
        self.version_mgr.log_export_attempt(version, success, {"path": out_path})
        result = {"status": "created" if success else "error", "path": out_path}
        self._create_export_report(result)
        return result

    def create_update_package(self, version: str, private_key: Optional[str] = None, public_key: Optional[str] = None, *, full_since_v2: bool = False) -> Dict:
        self._log_export_step("start", "info", {"type": version, "full_since_v2": full_since_v2})
        target_version = self.version_mgr.get_next_version()
        out_path = os.path.join("data/export_packages", f"night_watcher_{target_version}.tar.gz")
        prev_num = int(target_version.lstrip("v")) - 1
        prev_path = os.path.join("data/export_packages", f"night_watcher_v{prev_num:03d}.tar.gz") if prev_num >= 1 else None
        if full_since_v2:
            prev_path = os.path.join("data/export_packages", "night_watcher_v001.tar.gz")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        pk_file = self._write_temp_key(private_key)
        try:
            from export_versioned_artifact import export_versioned_artifact
            export_versioned_artifact(
                out_path,
                version=target_version,
                private_key_path=pk_file,
                kg_dir=os.path.join(self.nw.base_dir, "knowledge_graph"),
                vector_dir=os.path.join(self.nw.base_dir, "vector_store"),
                documents_dir=os.path.join(self.nw.base_dir, "documents"),
                previous_artifact_path=prev_path,
                bundled_files=[os.path.join(self.staging_mgr.staging_dir, f) for f in self.staging_mgr.list_staged_files()],
            )
            success = True
        except Exception as e:
            self._log_export_step("error", "failed", {"error": str(e)})
            success = False
            out_path = ""
        finally:
            if pk_file:
                os.unlink(pk_file)
        self.version_mgr.log_export_attempt(target_version, success, {"path": out_path})
        result = {"status": "created" if success else "error", "path": out_path}
        self._create_export_report(result)
        return result
