"""Vicuna-specific wrapper built on top of OllamaWrapper."""

from src.model.ollama_wrapper import OllamaWrapper


class VicunaWrapper(OllamaWrapper):
    """Wrapper that defaults to the Vicuna model."""

    def __init__(self, model_name: str = "vicuna"):
        super().__init__(model_name=model_name)
