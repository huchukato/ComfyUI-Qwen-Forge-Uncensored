"""Persistent prompt cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class PromptCache:
    """Simple file-backed prompt cache."""

    def __init__(self, cache_path: str | Path | None = None) -> None:
        if cache_path is None:
            cache_path = Path(__file__).parent.parent / "prompt_cache.json"
        self.cache_path = Path(cache_path)
        self._cache: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._cache = data if isinstance(data, dict) else {}
            except Exception as exc:
                print(f"[QwenUncensored] Failed to load prompt cache: {exc}")
                self._cache = {}

    def save(self) -> None:
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2)
        except Exception as exc:
            print(f"[QwenUncensored] Failed to save prompt cache: {exc}")

    @staticmethod
    def make_key(
        model_name: str,
        preset: str,
        custom: str | None,
        image_hash: str | None,
        video_hash: str | None,
        seed: int | None,
    ) -> str:
        key_data = {
            "model": model_name,
            "preset": preset,
            "custom": (custom or "").strip(),
            "image": image_hash,
            "video": video_hash,
            "seed": seed,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, key: str) -> str | None:
        entry = self._cache.get(key)
        if isinstance(entry, dict):
            return entry.get("text")
        return None

    def set(self, key: str, text: str, **metadata: Any) -> None:
        self._cache[key] = {"text": text, **metadata}
        self.save()
