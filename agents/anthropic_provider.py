"""
Night_watcher Anthropic Provider
Implementation of LLM provider using Anthropic API.
"""

import logging
from typing import Dict, List, Any, Optional

from agents.base import LLMProvider


class AnthropicProvider(LLMProvider):
    """Anthropic implementation of LLM provider"""

    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307"):
        """Initialize Anthropic client"""
        self.api_key = api_key
        self.model = model
        self.client = self._initialize_client()
        self.logger = logging.getLogger("AnthropicProvider")

    def _initialize_client(self):
        """Initialize the Anthropic client"""
        try:
            import anthropic
            return anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            self.logger.error("Anthropic SDK not installed. Install with: pip install anthropic")
            raise ImportError("Anthropic SDK not installed. Install with: pip install anthropic")

    def complete(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7,
                 stop: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute a completion request to Anthropic"""
        try:
            import anthropic

            # Convert to Claude API format
            message = anthropic.Message(
                role="user",
                content=prompt
            )

            params = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            if stop:
                params["stop_sequences"] = stop

            response = self.client.messages.create(
                **params,
                messages=[message]
            )

            # Convert Anthropic response to format expected by Night_watcher
            converted_response = {
                "choices": [
                    {
                        "text": response.content[0].text
                    }
                ]
            }

            return converted_response
        except Exception as e:
            self.logger.error(f"Anthropic API error: {str(e)}")
            return {"error": f"Failed to complete: {str(e)}"}

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available Anthropic models"""
        return [
            {"id": "claude-3-opus-20240229", "name": "Claude 3 Opus", "description": "Most powerful model, best for complex tasks"},
            {"id": "claude-3-sonnet-20240229", "name": "Claude 3 Sonnet", "description": "Balance of intelligence and speed"},
            {"id": "claude-3-haiku-20240307", "name": "Claude 3 Haiku", "description": "Fastest and most compact model"}
        ]
