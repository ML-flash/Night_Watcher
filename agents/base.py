"""
Night_watcher Base Agent Module
Defines abstract base classes for agents and LLM providers.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class LLMProvider(ABC):
    """Abstract base class for LLM provider implementations"""

    @abstractmethod
    def complete(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7,
                 stop: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute a completion request to the LLM"""
        pass


class Agent(ABC):
    """Abstract base class for all Night_watcher agents"""

    def __init__(self, llm_provider: LLMProvider, name: str = "BaseAgent"):
        """Initialize agent with an LLM provider"""
        self.llm_provider = llm_provider
        self.name = name
        self.logger = logging.getLogger(f"{self.name}")

    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return results"""
        pass

    def _call_llm(self, prompt: str, max_tokens: int = 1000,
                  temperature: float = 0.7, stop: Optional[List[str]] = None) -> str:
        """Helper method to call the LLM and extract text response"""
        response = self.llm_provider.complete(prompt, max_tokens, temperature, stop)

        if "error" in response:
            self.logger.error(f"LLM error: {response['error']}")
            return f"Error: {response['error']}"

        try:
            return response["choices"][0]["text"].strip()
        except (KeyError, IndexError) as e:
            self.logger.error(f"Error extracting text from LLM response: {str(e)}")
            return f"Error extracting response: {str(e)}"