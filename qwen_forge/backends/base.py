"""Abstract backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AbstractBackend(ABC):
    """Abstract interface for inference backends."""

    @abstractmethod
    def load(self, model_entry: dict[str, Any], params: dict[str, Any]) -> None:
        """Load the model described by `model_entry` using `params`."""
        raise NotImplementedError

    @abstractmethod
    def generate(self, conversation: list[dict[str, Any]], params: dict[str, Any]) -> str:
        """Generate a response from a conversation and generation parameters."""
        raise NotImplementedError

    @abstractmethod
    def unload(self) -> None:
        """Release model resources."""
        raise NotImplementedError

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Return whether a model is currently loaded."""
        raise NotImplementedError
