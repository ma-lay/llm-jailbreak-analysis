"""Llama-specific wrapper built on top of OllamaWrapper."""

from src.model.ollama_wrapper import OllamaWrapper


class LlamaWrapper(OllamaWrapper):
    """Wrapper that defaults to the Llama model."""

    def __init__(self, model_name: str = "llama2:7b"):
        super().__init__(model_name=model_name)
