"""
Night_watcher LLM Providers
Implementations of LLM providers for the Night_watcher system using HTTP API only.
"""

import logging
import requests
import json
import sys
from typing import Dict, List, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

# ==========================================
# Base LLM Provider
# ==========================================

class LLMProvider:
    """Abstract base class for LLM provider implementations"""

    def complete(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7,
                 stop: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute a completion request to the LLM"""
        raise NotImplementedError("Subclasses must implement complete")

# ==========================================
# LM Studio Provider
# ==========================================

class LMStudioProvider(LLMProvider):
    """LM Studio implementation of LLM provider using HTTP API only"""

    def __init__(self, host: str = "http://localhost:1234", model: str = None, use_sdk: bool = False):
        """Initialize LM Studio client with HTTP API"""
        self.host = host
        self.model = model
        self.http_client = self._initialize_http_client()
        self.logger = logging.getLogger("LMStudioProvider")
        self.logger.info(f"Using LM Studio HTTP API at: {self.host}")

    def _initialize_http_client(self):
        """Initialize the HTTP client for LM Studio"""
        return requests.Session()

    def complete(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7,
                 stop: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute a completion request to LM Studio using HTTP API"""
        try:
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            if stop:
                payload["stop"] = stop

            response = self.http_client.post(
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

# ==========================================
# Anthropic Provider
# ==========================================

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
            # Filter out empty or whitespace-only stop sequences
            filtered_stop = None
            if stop:
                filtered_stop = [s for s in stop if s and s.strip()]
                if not filtered_stop:
                    filtered_stop = None

            # Check the Anthropic SDK version and use appropriate API
            import anthropic
            import pkg_resources
            anthropic_version = pkg_resources.get_distribution("anthropic").version

            if anthropic_version >= "0.5.0":
                # For newer versions of the SDK (claude-3 API)

                # Ensure proper model name format
                if self.model == "3" or not self.model.startswith("claude-"):
                    self.logger.warning(f"Invalid model name: {self.model}, using claude-3-haiku-20240307 instead")
                    model_name = "claude-3-haiku-20240307"
                else:
                    model_name = self.model

                self.logger.info(f"Using Anthropic model: {model_name}")

                params = {
                    "model": model_name,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }

                if filtered_stop:
                    params["stop_sequences"] = filtered_stop

                response = self.client.messages.create(
                    messages=[{"role": "user", "content": prompt}],
                    **params
                )

                # Extract the text content from the response
                text_content = ""
                for content_block in response.content:
                    if content_block.type == "text":
                        text_content = content_block.text
                        break

                # Convert Anthropic response to format expected by Night_watcher
                return {
                    "choices": [
                        {
                            "text": text_content
                        }
                    ]
                }
            else:
                # For older versions (claude-2 API)
                response = self.client.completion(
                    prompt=f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}",
                    max_tokens_to_sample=max_tokens,
                    temperature=temperature,
                    stop_sequences=filtered_stop if filtered_stop else None
                )

                # Convert Anthropic response to format expected by Night_watcher
                return {
                    "choices": [
                        {
                            "text": response["completion"]
                        }
                    ]
                }
        except Exception as e:
            self.logger.error(f"Anthropic API error: {str(e)}")
            return {"error": f"Failed to complete: {str(e)}"}

# ==========================================
# Provider Initialization
# ==========================================

def get_anthropic_credentials() -> tuple[str, str]:
    """
    Prompt the user for Anthropic API credentials.
    Returns a tuple of (api_key, model_name)
    """
    print("\nLM Studio server is not available.")
    print("Falling back to Anthropic API...\n")
    
    api_key = input("Enter your Anthropic API key: ").strip()
    if not api_key:
        print("No API key provided. Continuing without LLM capabilities.")
        return "", ""
        
    model = input("Enter Anthropic model (default: claude-3-haiku-20240307): ").strip()
    if not model or model == "3" or not model.startswith("claude-"):
        model = "claude-3-haiku-20240307"
        print(f"Using default model: {model}")
        
    return api_key, model

def initialize_llm_provider(config) -> Optional[LLMProvider]:
    """Initialize LLM provider based on configuration"""
    provider_type = config["llm_provider"].get("type", "lm_studio")

    if provider_type == "anthropic":
        # Check for Anthropic SDK
        try:
            import anthropic

            api_key = config["llm_provider"].get("api_key", "")
            model = config["llm_provider"].get("model", "claude-3-haiku-20240307")

            if not api_key:
                # Prompt for API key if not in config
                api_key, user_model = get_anthropic_credentials()
                if user_model:
                    model = user_model
                
                if not api_key:
                    logger.error("Anthropic API key is required but not provided")
                    return None

            logger.info(f"Using Anthropic provider with model {model}")
            return AnthropicProvider(api_key=api_key, model=model)
        except ImportError:
            logger.error("Anthropic SDK not installed. Install with: pip install anthropic")
            return None
    else:
        # Default to LM Studio
        host = config["llm_provider"].get("host", "http://localhost:1234")
        model = config["llm_provider"].get("model", None)

        # Check if LM Studio is running
        if not check_lm_studio_connection(host):
            logger.warning(f"Could not connect to LM Studio at {host}")

            # Try to use Anthropic as fallback
            try:
                # The import check is just to see if the library is installed
                import anthropic
                
                # Get credentials directly from user
                api_key, model = get_anthropic_credentials()
                
                if not api_key:
                    logger.warning("No Anthropic API key provided. Continuing without LLM capabilities.")
                    return None

                logger.info(f"Using Anthropic provider with model {model}")
                return AnthropicProvider(api_key=api_key, model=model)
            except ImportError:
                logger.error("Anthropic SDK not installed. Install with: pip install anthropic")
                logger.warning("Continuing without LLM capabilities")
                return None
        
        logger.info(f"Using LM Studio provider at {host}")
        return LMStudioProvider(host=host, model=model)

def check_lm_studio_connection(host: str) -> bool:
    """Check if LM Studio is running and accessible"""
    try:
        # Use only HTTP check
        response = requests.get(f"{host}/v1/models", timeout=5)
        return response.status_code == 200
    except Exception as e:
        logger.warning(f"LM Studio connection check failed: {e}")
        return False
