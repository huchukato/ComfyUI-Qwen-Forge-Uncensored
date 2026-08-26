"""Shared node logic for Qwen-Uncensored ComfyUI nodes."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Make sure the repo root is importable
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from qwen_forge.cache import PromptCache
from qwen_forge.config import get_backend_class, load_model_catalog, resolve_model_entry
from qwen_forge.media import extract_frames, get_media_hash, tensor_to_pil
from qwen_forge.memory import normalize_device
from qwen_forge.output_cleaner import OutputCleanConfig, clean_model_output
from qwen_forge.prompts import build_prompt, build_text_prompt, load_prompt_config


class QwenUncensoredBaseNode:
    """Shared state and processing helpers for Qwen-Uncensored nodes."""

    def __init__(self) -> None:
        self.backend: Any = None
        self.backend_signature: tuple[Any, ...] | None = None
        self.prompt_cache = PromptCache()

    def _run_inference(
        self,
        model_name: str,
        prompt_text: str,
        params: dict[str, Any],
        backend_type: str,
        media: dict[str, Any] | None = None,
    ) -> str:
        catalog = load_model_catalog()
        entry = resolve_model_entry(model_name, catalog)

        if entry.get("backend") != backend_type:
            raise ValueError(f"Model '{model_name}' is not a {backend_type} model")

        # Conversation format used by backends
        conversation: list[dict[str, Any]] = []
        content: list[dict[str, Any]] = []

        if media:
            image = media.get("image")
            image2 = media.get("image2")
            frame_count = media.get("frame_count", 1)

            if image is not None:
                content.append({"type": "image", "image": image})

            if image2 is not None:
                frames = extract_frames(image2, frame_count=frame_count)
                for frame in frames:
                    content.append({"type": "image", "image": frame})

        content.append({"type": "text", "text": prompt_text})
        conversation.append({"role": "user", "content": content})

        # Reuse backend if signature matches
        signature = (model_name, backend_type, tuple(sorted(params.items())))
        if self.backend is None or self.backend_signature != signature:
            self._unload()
            backend_cls = get_backend_class(backend_type)
            self.backend = backend_cls()
            self.backend.load(entry, params)
            self.backend_signature = signature
        else:
            # Ensure headroom on cached model
            entry_params = {k: v for k, v in params.items() if k in ("device",)}
            self.backend.load(entry, {**params, **entry_params})

        raw = self.backend.generate(conversation, params)
        cleaned = clean_model_output(raw, OutputCleanConfig(mode="prompt")) or raw
        return cleaned

    @staticmethod
    def _model_default(model_name: str, key: str, fallback: Any) -> Any:
        entry = resolve_model_entry(model_name, load_model_catalog())
        defaults = entry.get("defaults") if isinstance(entry.get("defaults"), dict) else {}
        return entry.get(key, defaults.get(key, fallback))

    def _maybe_unload(self, keep_model_loaded: bool) -> None:
        if not keep_model_loaded:
            self._unload()

    def _unload(self) -> None:
        if self.backend is not None:
            self.backend.unload()
            self.backend = None
            self.backend_signature = None

    @staticmethod
    def _split_by_backend(catalog: dict[str, Any], backend: str, model_type: str | None = None) -> list[str]:
        names = []
        for name, info in catalog.items():
            if info.get("backend") != backend:
                continue
            if model_type is None or info.get("type") == model_type:
                names.append(name)
        return sorted(names)


# Global state for "last prompt" bypass across nodes
_LAST_SAVED_PROMPT: str | None = None


def get_last_prompt() -> str | None:
    return _LAST_SAVED_PROMPT


def set_last_prompt(value: str | None) -> None:
    global _LAST_SAVED_PROMPT
    _LAST_SAVED_PROMPT = value
