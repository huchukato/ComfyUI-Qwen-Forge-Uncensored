"""vLLM backend stub for future support."""

from __future__ import annotations

from typing import Any

from .base import AbstractBackend


class VLLMBackend(AbstractBackend):
    """Placeholder vLLM backend."""

    def load(self, model_entry: dict[str, Any], params: dict[str, Any]) -> None:
        raise NotImplementedError("vLLM backend is not implemented yet.")

    def generate(self, conversation: list[dict[str, Any]], params: dict[str, Any]) -> str:
        raise NotImplementedError("vLLM backend is not implemented yet.")

    def unload(self) -> None:
        pass

    @property
    def is_loaded(self) -> bool:
        return False
