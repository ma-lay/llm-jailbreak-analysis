"""Mistral-specific wrapper built on top of OllamaWrapper."""

from src.model.ollama_wrapper import OllamaWrapper


class MistralWrapper(OllamaWrapper):
    """Wrapper that defaults to the Mistral model."""

    def __init__(self, model_name: str = "mistral:7b"):
        super().__init__(model_name=model_name)
