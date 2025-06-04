"""
Night_watcher LLM Providers - Simplified
Minimal implementation for LM Studio and Anthropic support.
"""

import os
import requests
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class LMStudioProvider:
    """LM Studio local LLM provider."""
    
    def __init__(self, host: str = "http://localhost:1234"):
        self.host = host.rstrip('/')
        self.api_url = f"{self.host}/v1/completions"
    
    def complete(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.1) -> Dict[str, Any]:
        """Get completion from LM Studio."""
        try:
            response = requests.post(
                self.api_url,
                json={
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False
                },
                timeout=120
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"LM Studio error: {response.status_code}"}
                
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
            logger.info(f"Anthropic client initialized with model: {model}")
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
        
        # Test connection
        try:
            response = requests.get(f"{host}/v1/models", timeout=5)
            if response.status_code == 200:
                logger.info(f"LM Studio connected at {host}")
                models = response.json().get("data", [])
                if models:
                    logger.info(f"Available models: {[m.get('id') for m in models]}")
                else:
                    logger.warning("No models loaded in LM Studio")
                return LMStudioProvider(host)
            else:
                logger.warning(f"LM Studio returned status {response.status_code}")
        except requests.exceptions.ConnectionError:
            logger.warning(f"Cannot connect to LM Studio at {host}")
        except Exception as e:
            logger.warning(f"LM Studio connection error: {e}")
    
    elif provider_type == "anthropic":
        # Check for API key in multiple places
        api_key = None
        
        # 1. Check config file
        if "api_key" in llm_config and llm_config["api_key"]:
            api_key = llm_config["api_key"]
            logger.info("Using API key from config file")
        
        # 2. Check environment variable
        elif os.environ.get("ANTHROPIC_API_KEY"):
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            logger.info("Using API key from environment variable")
        
        # 3. Check for key file (from document 7)
        elif os.path.exists("key"):
            try:
                with open("key", "r") as f:
                    api_key = f.read().strip()
                logger.info("Using API key from key file")
            except Exception as e:
                logger.error(f"Error reading key file: {e}")
        
        if api_key:
            try:
                # Import anthropic to check if it's installed
                import anthropic
                
                # Get model from config or use default
                model = llm_config.get("model", "claude-3-haiku-20240307")
                
                provider = AnthropicProvider(api_key, model)
                logger.info(f"Anthropic provider initialized with model: {model}")
                
                # Test the provider
                test_response = provider.complete("Say 'test'", max_tokens=10)
                if "error" not in test_response:
                    logger.info("Anthropic provider test successful")
                else:
                    logger.error(f"Anthropic provider test failed: {test_response.get('error')}")
                
                return provider
                
            except ImportError:
                logger.error("anthropic package not installed. Run: pip install anthropic")
            except Exception as e:
                logger.error(f"Anthropic initialization failed: {e}")
        else:
            logger.error("No Anthropic API key found in config, environment, or key file")
    
    logger.warning("No LLM provider available")
    return None
