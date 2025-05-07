"""
Night_watcher LLM Providers (Enhanced)
Implementations of LLM providers for the Night_watcher system with LM Studio SDK support.
"""

import logging
import requests
import json
import os
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
    """LM Studio implementation of LLM provider with support for both HTTP API and Python SDK"""

    def __init__(self, host: str = "http://localhost:1234", model: str = None, use_sdk: bool = True):
        """Initialize LM Studio client with both HTTP and SDK options"""
        self.host = host
        self.model = model
        self.use_sdk = use_sdk
        
        # Try to use SDK first if available
        self.sdk_client = None
        if use_sdk:
            self.sdk_client = self._initialize_sdk()
            
        # Fallback to HTTP client if SDK not available/usable
        self.http_client = self._initialize_http_client()
        self.logger = logging.getLogger("LMStudioProvider")
        
        if self.sdk_client:
            self.logger.info(f"Using LM Studio SDK with model: {self.model or 'default'}")
        else:
            self.logger.info(f"Using LM Studio HTTP API at: {self.host}")

    def _initialize_sdk(self):
        """Initialize the LM Studio Python SDK client"""
        try:
            import lmstudio as lms
            
            # Create the client - if model is specified, load it
            if self.model:
                try:
                    return lms.llm(self.model)
                except Exception as e:
                    self.logger.warning(f"Failed to load specified model '{self.model}' with SDK: {e}")
                    # Try with default model
                    try:
                        return lms.llm()
                    except:
                        self.logger.warning("Failed to initialize LM Studio SDK with default model")
                        return None
            else:
                try:
                    # Use default model
                    return lms.llm()
                except Exception as e:
                    self.logger.warning(f"Failed to initialize LM Studio SDK: {e}")
                    return None
                    
        except ImportError:
            self.logger.warning("LM Studio SDK not installed. Install with: pip install lmstudio")
            return None

    def _initialize_http_client(self):
        """Initialize the HTTP client for LM Studio"""
        return requests.Session()

    def complete(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7,
                 stop: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute a completion request to LM Studio using SDK if available, fallback to HTTP"""
        
        # Try to use SDK if available
        if self.sdk_client:
            try:
                # Set up parameters
                kwargs = {"max_tokens": max_tokens, "temperature": temperature}
                if stop:
                    kwargs["stop"] = stop
                
                # Call the model using the SDK
                response = self.sdk_client.respond(prompt, **kwargs)
                
                # Format response to match the expected structure
                return {
                    "choices": [
                        {
                            "text": response
                        }
                    ]
                }
            except Exception as e:
                self.logger.warning(f"LM Studio SDK call failed, falling back to HTTP API: {e}")
                # If SDK fails, fall back to HTTP API
        
        # HTTP API fallback
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
        use_sdk = config["llm_provider"].get("use_sdk", True)
        
        # Check if LM Studio is running
        if not check_lm_studio_connection(host):
            logger.warning(f"Could not connect to LM Studio at {host}")
            
            # Try to use Anthropic as fallback
            if use_anthropic_api():
                try:
                    import anthropic
                    
                    # Get credentials from user
                    from night_watcher import get_anthropic_credentials
                    api_key, model = get_anthropic_credentials()
                    
                    if not api_key:
                        logger.warning("No Anthropic API key provided. Continuing without LLM capabilities.")
                        return None
                    
                    # Ensure proper model name format
                    if model == "3" or (model and not model.startswith("claude-")):
                        logger.warning(f"Invalid model name: {model}, using claude-3-haiku-20240307 instead")
                        model = "claude-3-haiku-20240307"
                    
                    logger.info(f"Using Anthropic provider with model {model}")
                    return AnthropicProvider(api_key=api_key, model=model)
                except ImportError:
                    logger.error("Anthropic SDK not installed. Install with: pip install anthropic")
                    return None
            else:
                logger.warning("Continuing without LLM capabilities")
                return None
        
        logger.info(f"Using LM Studio provider at {host}")
        return LMStudioProvider(host=host, model=model, use_sdk=use_sdk)

def check_lm_studio_connection(host: str) -> bool:
    """Check if LM Studio is running and accessible"""
    try:
        # First try SDK check
        try:
            import lmstudio as lms
            try:
                # Just initialize the client to check if it's available
                client = lms.llm()
                if client:
                    return True
            except:
                pass
        except ImportError:
            pass
            
        # Fall back to HTTP check
        response = requests.get(f"{host}/v1/models", timeout=5)
        return response.status_code == 200
    except Exception as e:
        logger.warning(f"LM Studio connection check failed: {e}")
        return False

def use_anthropic_api() -> bool:
    """Ask user if they want to use the Anthropic API"""
    print("\nLM Studio server is not available.")
    print("Would you like to use the Anthropic API instead? (requires API key)")
    response = input("Use Anthropic API? (y/n): ").strip().lower()
    return response == 'y' or response == 'yes'
