"""
Night_watcher LM Studio Provider
Implementation of LLM provider using LM Studio API.
"""

import logging
from typing import Dict, List, Any, Optional

from .base import LLMProvider


class LMStudioProvider(LLMProvider):
    """LM Studio implementation of LLM provider"""

    def __init__(self, host: str = "http://localhost:1234"):
        """Initialize LM Studio client"""
        self.host = host
        self.client = self._initialize_client()
        self.logger = logging.getLogger("LMStudioProvider")

    def _initialize_client(self):
        """Initialize the HTTP client for LM Studio"""
        import requests
        return requests.Session()

    def complete(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7,
                 stop: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute a completion request to LM Studio"""
        import requests

        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if stop:
            payload["stop"] = stop

        try:
            response = self.client.post(
                f"{self.host}/v1/completions",
                json=payload
            )

            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"LM Studio API error: {response.status_code} - {response.text}")
                return {"error": f"Failed to complete: {response.status_code}"}

        except Exception as e:
            self.logger.error(f"LM Studio completion error: {str(e)}")
            return {"error": f"Completion error: {str(e)}"}