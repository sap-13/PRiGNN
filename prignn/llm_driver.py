
import abc
import os
from typing import Dict, Type
import vertexai
from vertexai.preview.generative_models import GenerativeModel

class LLMDriver(abc.ABC):
    """The base class for all LLM drivers."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abc.abstractmethod
    def generate(self, prompt: str) -> str:
        """Generates a response from the LLM."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculates the cost of a generation."""
        raise NotImplementedError

class VertexAIDriver(LLMDriver):
    """Driver for Google's Vertex AI models."""

    def __init__(self, model_name: str, project_id: str, location: str):
        super().__init__(model_name)
        vertexai.init(project=project_id, location=location)
        self.client = GenerativeModel(model_name)

    def generate(self, prompt: str) -> str:
        response = self.client.generate_content(prompt)
        return response.text

    def get_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        # Placeholder pricing for Vertex AI (adjust as per actual pricing)
        return (prompt_tokens * 0.0001 + completion_tokens * 0.0002) / 1000

_DRIVER_REGISTRY: Dict[str, Type[LLMDriver]] = {
    "gemini": VertexAIDriver,
}

def get_llm_driver(
    driver_name: str, model_name: str, project_id: str = None, location: str = None
) -> LLMDriver:
    """Factory function to get an LLM driver instance."""
    if driver_name not in _DRIVER_REGISTRY:
        raise ValueError(f"Unknown LLM driver: {driver_name}")

    driver_class = _DRIVER_REGISTRY[driver_name]

    if driver_class is VertexAIDriver:
        return driver_class(model_name=model_name, project_id=project_id, location=location)

    return driver_class(model_name=model_name)

