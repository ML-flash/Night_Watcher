# providers.py — LM Studio + Anthropic fallback
import logging
from typing import Optional

from lmstudio import LMStudio

logger = logging.getLogger(__name__)

# LM Studio wrapper for a single model
class LMStudioModelWrapper:
    def __init__(self, client: LMStudio, model_id: str):
        self.client = client
        self.model_id = model_id

    def complete(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.3, stop=None) -> dict:
        try:
            response = self.client.complete(
                prompt=prompt,
                model=self.model_id,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop
            )
            return {"choices": [{"text": response.text}]}
        except Exception as e:
            logger.error(f"LMStudio model '{self.model_id}' completion error: {str(e)}")
            return {"error": str(e)}

# Anthropic fallback
class AnthropicProvider:
    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def complete(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.3, stop=None) -> dict:
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.client.messages.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop_sequences=stop if stop else None
            )
            for block in response.content:
                if block.type == "text":
                    return {"choices": [{"text": block.text}]}
            return {"error": "No valid text in response"}
        except Exception as e:
            logger.error(f"Anthropic completion error: {str(e)}")
            return {"error": str(e)}

# Initialization

def initialize_dual_llms(primary_id: str, secondary_id: str, fallback_api_key: Optional[str] = None, fallback_model: str = "claude-3-haiku-20240307"):
    try:
        client = LMStudio()
        models = [m.id for m in client.list_local_models()]

        if primary_id not in models:
            raise ValueError(f"Primary model '{primary_id}' not loaded in LM Studio")
        if secondary_id not in models:
            raise ValueError(f"Secondary model '{secondary_id}' not loaded in LM Studio")

        primary_llm = LMStudioModelWrapper(client, primary_id)
        secondary_llm = LMStudioModelWrapper(client, secondary_id)
        return primary_llm, secondary_llm

    except Exception as lm_error:
        logger.warning(f"LM Studio error: {lm_error}. Falling back to Anthropic.")

        if fallback_api_key:
            logger.info(f"Using Anthropic fallback model: {fallback_model}")
            fallback = AnthropicProvider(api_key=fallback_api_key, model=fallback_model)
            return fallback, fallback
        else:
            raise RuntimeError("No LLMs available and no Anthropic key provided.")
