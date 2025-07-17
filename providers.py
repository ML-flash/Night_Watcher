"""
Night_watcher LLM Providers - Enhanced Context Window Management
Intelligent token allocation and model-specific configuration management.
"""

import os
import json
from file_utils import safe_json_load, safe_json_save
import requests
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Model context window defaults
MODEL_CONTEXT_WINDOWS = {
    # Common models with known context windows
    "qwen/qwen3-30b": 32768,
    "microsoft/phi-3-medium-4k": 4096,
    "microsoft/phi-3-medium-128k": 131072,
    "meta-llama/llama-3.1-8b": 131072,
    "meta-llama/llama-3.1-70b": 131072,
    "anthropic/claude-3-haiku": 200000,
    "anthropic/claude-3-sonnet": 200000,
    "anthropic/claude-3-opus": 200000,
    "mistral/mistral-7b": 32768,
    "mistral/mixtral-8x7b": 32768,
    "openai/gpt-3.5-turbo": 16385,
    "openai/gpt-4": 8192,
    "openai/gpt-4-32k": 32768,
    "openai/gpt-4-turbo": 128000,
    "openai/gpt-4o": 128000,
    "google/gemini-pro": 30720,
    "google/gemini-1.5-pro": 1048576,
    "google/gemini-1.5-flash": 1048576,
}

# Conservative token estimation (characters to tokens ratio)
CHARS_PER_TOKEN = 4


class ModelConfig:
    """Model-specific configuration management."""

    def __init__(self, config_dir: str = "data/model_configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("ModelConfig")

    def get_config_path(self, model_name: str) -> Path:
        """Get config file path for a model."""
        safe_name = model_name.replace("/", "_").replace("\\", "_")
        return self.config_dir / f"{safe_name}.json"

    def load_config(self, model_name: str) -> Dict[str, Any]:
        """Load model-specific configuration."""
        config_path = self.get_config_path(model_name)

        if config_path.exists():
            try:
                config = safe_json_load(str(config_path), default=None)
                if config is not None:
                    self.logger.info(f"Loaded config for {model_name}")
                    return config
            except Exception as e:
                self.logger.error(f"Failed to load config for {model_name}: {e}")

        # Return default config
        return self.get_default_config(model_name)

    def save_config(self, model_name: str, config: Dict[str, Any]):
        """Save model-specific configuration."""
        config_path = self.get_config_path(model_name)

        try:
            if safe_json_save(str(config_path), config):
                self.logger.info(f"Saved config for {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to save config for {model_name}: {e}")

    def get_default_config(self, model_name: str) -> Dict[str, Any]:
        """Get default configuration for a model."""
        # Try to match model name patterns
        context_window = 32768  # Conservative default

        for pattern, window in MODEL_CONTEXT_WINDOWS.items():
            if pattern.lower() in model_name.lower():
                context_window = window
                break

        return {
            "context_window": context_window,
            "max_tokens": min(4096, context_window // 2),  # Conservative output allocation
            "temperature": 0.1,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "created_at": "auto-generated",
            "notes": f"Auto-generated config for {model_name}"
        }


class ContextWindowManager:
    """Intelligent context window management."""

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.logger = logging.getLogger("ContextWindowManager")

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Simple character-based estimation
        # More accurate would use actual tokenizer, but this is pragmatic
        return max(1, len(text) // CHARS_PER_TOKEN)

    def calculate_optimal_tokens(self, model_name: str, prompt: str,
                               requested_max_tokens: Optional[int] = None) -> Tuple[int, Dict[str, Any]]:
        """
        Calculate optimal max_tokens based on context window and prompt size.

        Returns:
            (optimal_max_tokens, calculation_info)
        """
        config = self.model_config.load_config(model_name)
        context_window = config.get("context_window", 32768)

        prompt_tokens = self.estimate_tokens(prompt)

        # Reserve tokens for safety margin (10% of context window)
        safety_margin = max(256, context_window // 10)

        # Calculate available tokens for output
        available_tokens = context_window - prompt_tokens - safety_margin

        # Use requested max_tokens if provided and reasonable
        if requested_max_tokens:
            optimal_tokens = min(requested_max_tokens, available_tokens)
        else:
            # Use model's default max_tokens but respect context window
            default_max = config.get("max_tokens", 4096)
            optimal_tokens = min(default_max, available_tokens)

        # Ensure minimum viable output
        optimal_tokens = max(512, optimal_tokens)

        calculation_info = {
            "context_window": context_window,
            "prompt_tokens": prompt_tokens,
            "safety_margin": safety_margin,
            "available_tokens": available_tokens,
            "optimal_tokens": optimal_tokens,
            "requested_tokens": requested_max_tokens,
            "utilization": (prompt_tokens + optimal_tokens) / context_window
        }

        if calculation_info["utilization"] > 0.95:
            self.logger.warning(f"High context utilization: {calculation_info['utilization']:.2%}")

        return optimal_tokens, calculation_info


class LMStudioProvider:
    """Enhanced LM Studio provider with intelligent context management."""

    def __init__(self, host: str = "http://localhost:1234", model: Optional[str] = None):
        self.host = host.rstrip('/')
        self.model = model
        self.api_url = f"{self.host}/v1/completions"
        self.models_url = f"{self.host}/v1/models"
        self.logger = logging.getLogger("LMStudioProvider")

        # Initialize context management
        self.model_config = ModelConfig()
        self.context_manager = ContextWindowManager(self.model_config)

        # Try to detect current model if not provided
        if not self.model:
            self._auto_detect_model()

    def _auto_detect_model(self):
        """Auto-detect the currently loaded model."""
        try:
            models = self.list_models()
            if models:
                self.model = models[0]
                self.logger.info(f"Auto-detected model: {self.model}")
        except Exception as e:
            self.logger.debug(f"Could not auto-detect model: {e}")

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

    def get_model_config(self) -> Dict[str, Any]:
        """Get configuration for current model."""
        if not self.model:
            return self.model_config.get_default_config("unknown")
        return self.model_config.load_config(self.model)

    def save_model_config(self, config: Dict[str, Any]):
        """Save configuration for current model."""
        if self.model:
            self.model_config.save_config(self.model, config)

    def complete(self, prompt: str, max_tokens: Optional[int] = None,
                temperature: Optional[float] = None, stream: bool = False,
                auto_adjust_tokens: bool = True) -> Dict[str, Any]:
        """
        Get completion from LM Studio with intelligent token management.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate (auto-calculated if None)
            temperature: Sampling temperature (uses model config if None)
            stream: Whether to stream response
            auto_adjust_tokens: Whether to automatically adjust max_tokens for context window
        """
        try:
            # Check if we have a model selected
            if not self.model:
                models = self.list_models()
                if models:
                    self.model = models[0]
                    self.logger.info(f"Auto-selected model: {self.model}")
                else:
                    return {"error": "No model selected and no models available"}

            # Load model configuration
            model_config = self.get_model_config()

            # Calculate optimal tokens if auto-adjustment is enabled
            if auto_adjust_tokens:
                optimal_tokens, calc_info = self.context_manager.calculate_optimal_tokens(
                    self.model, prompt, max_tokens
                )

                # Log token calculation details
                self.logger.info(f"Token calculation for {self.model}: "
                               f"prompt={calc_info['prompt_tokens']}, "
                               f"optimal_output={optimal_tokens}, "
                               f"utilization={calc_info['utilization']:.1%}")

                # Warn if prompt is very large
                if calc_info['prompt_tokens'] > calc_info['context_window'] * 0.8:
                    self.logger.warning(f"Large prompt detected: {calc_info['prompt_tokens']} tokens "
                                      f"({calc_info['prompt_tokens']/calc_info['context_window']:.1%} of context)")

                max_tokens = optimal_tokens
            else:
                max_tokens = max_tokens or model_config.get("max_tokens", 2000)

            # Use model config defaults if not specified
            if temperature is None:
                temperature = model_config.get("temperature", 0.1)

            payload = {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": model_config.get("top_p", 0.9),
                "top_k": model_config.get("top_k", 40),
                "repeat_penalty": model_config.get("repeat_penalty", 1.15),  # Slightly higher to prevent loops
                "stream": bool(stream),
                "stop": ["\n\n---", "CONTENT:", "END_OF_RESPONSE", "}\n\n{"]  # Stop sequences to prevent repetition
            }

            self.logger.debug(f"Sending request to LM Studio: model={self.model}, "
                            f"max_tokens={max_tokens}, temp={temperature}")

            response = requests.post(
                self.api_url,
                json=payload,
                timeout=600,  # Increased timeout for large outputs
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
                result = response.json()

                # Log usage statistics if available
                if "usage" in result:
                    usage = result["usage"]
                    self.logger.info(f"Token usage: prompt={usage.get('prompt_tokens', 'N/A')}, "
                                   f"completion={usage.get('completion_tokens', 'N/A')}, "
                                   f"total={usage.get('total_tokens', 'N/A')}")

                return result

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
            error_msg = "LM Studio request timed out - try reducing prompt size or max_tokens"
            self.logger.error(error_msg)
            return {"error": error_msg}
        except Exception as e:
            self.logger.error(f"LM Studio completion error: {e}")
            return {"error": str(e)}


class AnthropicProvider:
    """Enhanced Anthropic API provider."""

    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307"):
        self.api_key = api_key
        self.model = model
        self.logger = logging.getLogger("AnthropicProvider")

        # Initialize context management
        self.model_config = ModelConfig()
        self.context_manager = ContextWindowManager(self.model_config)

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

    def get_model_config(self) -> Dict[str, Any]:
        """Get configuration for current model."""
        return self.model_config.load_config(f"anthropic/{self.model}")

    def save_model_config(self, config: Dict[str, Any]):
        """Save configuration for current model."""
        self.model_config.save_config(f"anthropic/{self.model}", config)

    def complete(self, prompt: str, max_tokens: Optional[int] = None,
                temperature: Optional[float] = None, auto_adjust_tokens: bool = True) -> Dict[str, Any]:
        """Get completion from Anthropic with intelligent token management."""
        try:
            # Load model configuration
            model_config = self.get_model_config()

            # Calculate optimal tokens if auto-adjustment is enabled
            if auto_adjust_tokens:
                optimal_tokens, calc_info = self.context_manager.calculate_optimal_tokens(
                    f"anthropic/{self.model}", prompt, max_tokens
                )
                max_tokens = optimal_tokens

                self.logger.info(f"Token calculation for {self.model}: "
                               f"prompt={calc_info['prompt_tokens']}, "
                               f"optimal_output={optimal_tokens}, "
                               f"utilization={calc_info['utilization']:.1%}")
            else:
                max_tokens = max_tokens or model_config.get("max_tokens", 2000)

            if temperature is None:
                temperature = model_config.get("temperature", 0.1)

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
                }],
                "usage": {
                    "prompt_tokens": response.usage.input_tokens if hasattr(response, 'usage') else 0,
                    "completion_tokens": response.usage.output_tokens if hasattr(response, 'usage') else 0,
                    "total_tokens": (response.usage.input_tokens + response.usage.output_tokens) if hasattr(response, 'usage') else 0
                }
            }

        except Exception as e:
            self.logger.error(f"Anthropic completion error: {e}")
            return {"error": str(e)}


class GeminiProvider:
    """Google Gemini API provider."""

    def __init__(self, api_key: str, model: str = "gemini-pro"):
        self.api_key = api_key
        self.model = model
        self.logger = logging.getLogger("GeminiProvider")

        # Initialize context management
        self.model_config = ModelConfig()
        self.context_manager = ContextWindowManager(self.model_config)

        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.genai = genai
            self.client = genai.GenerativeModel(model)
        except ImportError:
            raise ImportError("Install google-generativeai: pip install google-generativeai")

    def update_model(self, model: str):
        """Update the model to use."""
        self.model = model
        self.client = self.genai.GenerativeModel(model)
        self.logger.info(f"Model updated to: {model}")

    def list_models(self) -> List[str]:
        """Return available Gemini models."""
        try:
            models = self.genai.list_models()
            return [m.name for m in models]
        except Exception as e:
            self.logger.error(f"Error listing models: {e}")
            return []

    def get_model_config(self) -> Dict[str, Any]:
        """Get configuration for current model."""
        return self.model_config.load_config(f"gemini/{self.model}")

    def save_model_config(self, config: Dict[str, Any]):
        """Save configuration for current model."""
        self.model_config.save_config(f"gemini/{self.model}", config)

    def count_tokens(self, text: str) -> int:
        """Return token count for text."""
        try:
            resp = self.client.count_tokens(text)
            return int(getattr(resp, "total_tokens", 0))
        except Exception as e:
            self.logger.error(f"Gemini count_tokens error: {e}")
            return self.context_manager.estimate_tokens(text)

    def complete(self, prompt: str, max_tokens: Optional[int] = None,
                temperature: Optional[float] = None, auto_adjust_tokens: bool = True) -> Dict[str, Any]:
        """Get completion from Gemini with intelligent token management."""
        try:
            model_config = self.get_model_config()

            if auto_adjust_tokens:
                optimal_tokens, calc_info = self.context_manager.calculate_optimal_tokens(
                    f"gemini/{self.model}", prompt, max_tokens
                )
                max_tokens = optimal_tokens
                self.logger.info(
                    f"Token calculation for {self.model}: prompt={calc_info['prompt_tokens']}, "
                    f"optimal_output={optimal_tokens}, utilization={calc_info['utilization']:.1%}"
                )
            else:
                max_tokens = max_tokens or model_config.get("max_tokens", 2048)

            if temperature is None:
                temperature = model_config.get("temperature", 0.1)

            gen_cfg = {
                "max_output_tokens": max_tokens,
                "temperature": temperature,
                "top_p": model_config.get("top_p", 0.9),
            }

            response = self.client.generate_content(prompt, generation_config=gen_cfg)

            usage = {}
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = {
                    "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                    "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
                    "total_tokens": getattr(response.usage_metadata, "total_token_count", 0),
                }

            return {
                "choices": [{"text": getattr(response, "text", "")}],
                "usage": usage,
            }

        except Exception as e:
            self.logger.error(f"Gemini completion error: {e}")
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

    elif provider_type == "gemini":
        api_key = llm_config.get("api_key") or os.environ.get("GOOGLE_API_KEY")

        if api_key:
            try:
                model = llm_config.get("model", "gemini-pro")
                provider = GeminiProvider(api_key, model)
                logger.info(f"Gemini provider initialized with model: {model}")
                return provider
            except ImportError:
                logger.error(
                    "google-generativeai package not installed - run: pip install google-generativeai"
                )
            except Exception as e:
                logger.error(f"Gemini init failed: {e}")
        else:
            logger.error("No Google API key found in config or environment")

    logger.warning("No LLM provider available")
    return None