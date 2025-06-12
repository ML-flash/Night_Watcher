"""
Night_watcher LLM Providers - Simplified
Minimal implementation for LM Studio and Anthropic support.
"""

import os
import json
import requests
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class LMStudioProvider:
    """LM Studio local LLM provider."""

    def __init__(self, host: str = "http://localhost:1234", model: Optional[str] = None):
        self.host = host.rstrip('/')
        self.model = model
        self.api_url = f"{self.host}/v1/completions"
        self.models_url = f"{self.host}/v1/models"
        self.logger = logging.getLogger("LMStudioProvider")

    def update_model(self, model: Optional[str]):
        """Set the model to use for completions."""
        self.model = model
        self.logger.info(f"Model updated to: {model}")

    def list_models(self) -> List[str]:
        """Return available model IDs from LM Studio."""
        try:
            resp = requests.get(self.models_url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                models = []
                for m in data.get("data", []):
                    model_id = m.get("id") or m.get("model")
                    if model_id:
                        models.append(model_id)
                self.logger.info(f"Found {len(models)} models in LM Studio")
                return models
            else:
                self.logger.error(f"Failed to list models: status {resp.status_code}")
        except requests.exceptions.ConnectionError:
            self.logger.error("Cannot connect to LM Studio - is it running?")
        except Exception as e:
            self.logger.error(f"Error listing models: {e}")
        return []
    
    def complete(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.1, stream: bool = False):
        """Get completion from LM Studio. If stream is True, returns a generator."""
        try:
            # Check if we have a model selected
            if not self.model:
                # Try to auto-select first available model
                models = self.list_models()
                if models:
                    self.model = models[0]
                    self.logger.info(f"Auto-selected model: {self.model}")
                else:
                    return {"error": "No model selected and no models available"}

            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": bool(stream),
                "model": self.model  # Always include model
            }

            self.logger.debug(f"Sending request to LM Studio with model: {self.model}")

            response = requests.post(
                self.api_url,
                json=payload,
                timeout=300,  # Increased timeout for long completions
                stream=bool(stream)
            )

            if response.status_code != 200:
                error_msg = f"LM Studio error: {response.status_code}"
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_msg += f" - {error_data['error']}"
                except:
                    error_msg += f" - {response.text}"
                self.logger.error(error_msg)
                return {"error": error_msg}

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

        except requests.exceptions.ConnectionError:
            error_msg = "Cannot connect to LM Studio - make sure it's running on port 1234"
            self.logger.error(error_msg)
            return {"error": error_msg}
        except requests.exceptions.Timeout:
            error_msg = "LM Studio request timed out - try a smaller max_tokens value"
            self.logger.error(error_msg)
            return {"error": error_msg}
        except Exception as e:
            self.logger.error(f"LM Studio completion error: {e}")
            return {"error": str(e)}


class AnthropicProvider:
    """Anthropic API provider."""
    
    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307"):
        self.api_key = api_key
        self.model = model
        self.logger = logging.getLogger("AnthropicProvider")
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")
    
    def update_model(self, model: str):
        """Update the model to use."""
        self.model = model
        self.logger.info(f"Model updated to: {model}")
    
    def list_models(self) -> List[str]:
        """Return available Anthropic models."""
        return [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229", 
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0"
        ]
    
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
            self.logger.error(f"Anthropic completion error: {e}")
            return {"error": str(e)}


def initialize_llm_provider(config: Dict[str, Any]) -> Optional[Any]:
    """Initialize LLM provider based on config."""
    llm_config = config.get("llm_provider", {})
    provider_type = llm_config.get("type", "lm_studio")
    
    logger.info(f"Attempting to initialize LLM provider: {provider_type}")
    
    if provider_type == "lm_studio":
        host = llm_config.get("host", "http://localhost:1234")
        model = llm_config.get("model")

        # Create provider first
        provider = LMStudioProvider(host, model)
        
        # Test connection
        try:
            models = provider.list_models()
            if models:
                logger.info(f"LM Studio connected at {host} - {len(models)} models available")
                # If no model specified in config, use first available
                if not model and models:
                    provider.update_model(models[0])
                    logger.info(f"Auto-selected model: {models[0]}")
            else:
                logger.warning(f"LM Studio connected but no models loaded")
                # Still return provider - user might load a model later
        except Exception as e:
            logger.error(f"Could not connect to LM Studio at {host}: {e}")
            return None

        return provider
    
    elif provider_type == "anthropic":
        # Check for API key in multiple places
        api_key = llm_config.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        
        # Check for key file
        if not api_key and os.path.exists("key"):
            try:
                with open("key", "r") as f:
                    api_key = f.read().strip()
                logger.info("Loaded API key from key file")
            except:
                pass
        
        if api_key:
            try:
                model = llm_config.get("model", "claude-3-haiku-20240307")
                provider = AnthropicProvider(api_key, model)
                logger.info(f"Anthropic provider initialized with model: {model}")
                return provider
            except ImportError:
                logger.error("Anthropic package not installed - run: pip install anthropic")
            except Exception as e:
                logger.error(f"Anthropic init failed: {e}")
        else:
            logger.error("No Anthropic API key found in config, environment, or key file")
    
    logger.warning("No LLM provider available")
    return None
