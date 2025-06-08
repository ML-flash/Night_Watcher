"""
Night_watcher LLM Providers - Simplified
Minimal implementation for LM Studio and Anthropic support.
"""

import os
import json
import requests
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class LMStudioProvider:
    """LM Studio local LLM provider."""

    def __init__(self, host: str = "http://localhost:1234", model: Optional[str] = None):
        self.host = host.rstrip('/')
        self.model = model
        self.api_url = f"{self.host}/v1/completions"

    def update_model(self, model: Optional[str]):
        """Set the model to use for completions."""
        self.model = model

    def list_models(self) -> list:
        """Return available model IDs from LM Studio."""
        try:
            resp = requests.get(f"{self.host}/v1/models", timeout=5)
            if resp.status_code == 200:
                return [m.get("id") or m.get("model") for m in resp.json().get("data", [])]
        except Exception:
            pass
        return []
    
    def complete(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.1, stream: bool = False):
        """Get completion from LM Studio. If ``stream`` is True a generator of
        text chunks is returned."""
        try:
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": bool(stream)
            }
            if self.model:
                payload["model"] = self.model

            response = requests.post(
                self.api_url,
                json=payload,
                timeout=120,
                stream=bool(stream)
            )

            if response.status_code != 200:
                return {"error": f"LM Studio error: {response.status_code}"}

            if not stream:
                return response.json()

            def generate():
                for line in response.iter_lines():
                    if not line:
                        continue
                    if line.startswith(b"data:"):
                        line = line[5:]
                    if line.strip() == b"[DONE]":
                        break
                    try:
                        data = json.loads(line.decode("utf-8"))
                        yield data.get("choices", [{}])[0].get("text", "")
                    except Exception:
                        continue

            return generate()

        except Exception as e:
            return {"error": str(e)}


class AnthropicProvider:
    """Anthropic API provider."""
    
    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307"):
        self.api_key = api_key
        self.model = model
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")
    
    def complete(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.1) -> Dict[str, Any]:
        """Get completion from Anthropic."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Convert to standard format
            return {
                "choices": [{
                    "text": response.content[0].text if response.content else ""
                }]
            }
            
        except Exception as e:
            return {"error": str(e)}


def initialize_llm_provider(config: Dict[str, Any]) -> Optional[Any]:
    """Initialize LLM provider based on config."""
    llm_config = config.get("llm_provider", {})
    provider_type = llm_config.get("type", "lm_studio")
    
    logger.info(f"Attempting to initialize LLM provider: {provider_type}")
    
    if provider_type == "lm_studio":
        host = llm_config.get("host", "http://localhost:1234")
        model = llm_config.get("model")

        try:
            response = requests.get(f"{host}/v1/models", timeout=5)
            if response.status_code == 200:
                models = response.json().get("data", [])
                logger.info(f"LM Studio connected at {host} - {len(models)} models available")
            else:
                logger.warning(f"LM Studio /v1/models returned status {response.status_code}; proceeding anyway")
        except requests.exceptions.ConnectionError:
            logger.error(f"Could not connect to LM Studio at {host}")
            return None
        except Exception as e:
            logger.warning(f"LM Studio connection test failed: {e}; proceeding")

        provider = LMStudioProvider(host)
        provider.update_model(model)
        return provider
    
    elif provider_type == "anthropic":
        # Check for API key in multiple places
        api_key = llm_config.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        
        # Check for key file
        if not api_key and os.path.exists("key"):
            try:
                with open("key", "r") as f:
                    api_key = f.read().strip()
            except:
                pass
        
        if api_key:
            try:
                provider = AnthropicProvider(api_key, llm_config.get("model", "claude-3-haiku-20240307"))
                logger.info("Anthropic provider initialized")
                return provider
            except ImportError:
                logger.error("Anthropic package not installed - run: pip install anthropic")
            except Exception as e:
                logger.error(f"Anthropic init failed: {e}")
        else:
            logger.error("No Anthropic API key found in config, environment, or key file")
    
    logger.warning("No LLM provider available")
    return None
