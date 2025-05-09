#!/usr/bin/env python3
"""
Run the 7‑round Night_watcher analysis as a stand‑alone step.
Keeps LM‑Studio → Anthropic fallback logic unchanged.
"""

import argparse, glob, json, logging, os, pathlib, sys
from datetime import datetime
from typing import Any, Dict, List

# ── our own helpers / provider -------------------------------------------------
from analyzer  import ContentAnalyzer               # 7‑round chain (unchanged)
from utils     import save_to_file                  # existing helper

# ── minimal, built‑in config loader (replaces import night_watcher) ------------
DEFAULT_CFG: Dict[str, Any] = {
    "llm_provider": {
        "type": "lm_studio",
        "host": "http://localhost:1234",
        "model": "default"
    }
}

def load_config(path: str) -> Dict[str, Any]:
    """Return JSON config or fallback to DEFAULT_CFG."""
    if os.path.exists(path):
        try:
            with open(path, encoding="utf‑8") as f:
                cfg = json.load(f)
            # shallow‑merge defaults so those keys are always present
            merged = DEFAULT_CFG | cfg
            merged["llm_provider"] = DEFAULT_CFG["llm_provider"] | cfg.get("llm_provider", {})
            return merged
        except Exception as e:
            logging.warning(f"Bad config JSON – using defaults ({e})")
    return DEFAULT_CFG
