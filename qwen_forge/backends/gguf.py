"""GGUF backend via llama.cpp for Qwen models."""

from __future__ import annotations

import gc
import struct
from pathlib import Path
from typing import Any

import torch
from llama_cpp import Llama  # type: ignore

from ..media import tensor_to_base64_png
from ..memory import ensure_cuda_vram_headroom, normalize_device, torch_gc
from .base import AbstractBackend


def read_gguf_architecture(filepath: Path) -> str | None:
    """Read general.architecture from a GGUF header without loading weights."""
    _VTYPE_SIZE = {
        0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8,
    }
    _VTYPE_STRING = 8
    _VTYPE_ARRAY = 9

    def _read_string(f):
        length = struct.unpack("<Q", f.read(8))[0]
        return f.read(length).decode("utf-8", errors="replace")

    def _skip_value(f, vtype):
        if vtype in _VTYPE_SIZE:
            f.seek(_VTYPE_SIZE[vtype], 1)
        elif vtype == _VTYPE_STRING:
            length = struct.unpack("<Q", f.read(8))[0]
            f.seek(length, 1)
        elif vtype == _VTYPE_ARRAY:
            arr_type = struct.unpack("<I", f.read(4))[0]
            arr_len = struct.unpack("<Q", f.read(8))[0]
            for _ in range(arr_len):
                _skip_value(f, arr_type)
        else:
            return False
        return True

    try:
        with open(filepath, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                return None
            version = struct.unpack("<I", f.read(4))[0]
            if version not in (2, 3):
                return None
            _tensor_count = struct.unpack("<Q", f.read(8))[0]
            kv_count = struct.unpack("<Q", f.read(8))[0]
            for _ in range(kv_count):
                key = _read_string(f)
                vtype = struct.unpack("<I", f.read(4))[0]
                if key == "general.architecture":
                    if vtype == _VTYPE_STRING:
                        return _read_string(f)
                    return None
                if not _skip_value(f, vtype):
                    return None
    except Exception:
        return None
    return None


def _is_qwen3_gguf(arch: str | None, model_name: str) -> bool:
    if arch and "qwen3" in arch:
        return True
    return "qwen3" in model_name.lower()


class GGUFBackend(AbstractBackend):
    def __init__(self) -> None:
        self.llm: Any = None
        self.chat_handler: Any = None
        self.signature: tuple[Any, ...] | None = None
        self.is_qwen3 = False
        self.mmproj_path: Path | None = None

    @property
    def is_loaded(self) -> bool:
        return self.llm is not None

    def unload(self) -> None:
        if self.llm is not None:
            try:
                self.llm.close()
            except Exception:
                pass
            self.llm = None
        self.chat_handler = None
        self.signature = None
        self.is_qwen3 = False
        self.mmproj_path = None
        gc.collect()
        torch_gc()

    def load(self, model_entry: dict[str, Any], params: dict[str, Any]) -> None:
        model_name = model_entry.get("filename") or model_entry.get("repo_name") or "unknown"
        device_choice = params.get("device", "auto")
        ctx = params.get("ctx", 32768)
        n_batch = params.get("n_batch", 512)
        gpu_layers = params.get("gpu_layers", -1)
        pool_size = params.get("pool_size", 4194304)
        top_k = params.get("top_k", 20)

        device = normalize_device(device_choice)
        model_path = self._resolve_model_path(model_entry)
        if not model_path.exists():
            raise FileNotFoundError(f"[QwenUncensored GGUF] Model not found: {model_path}")

        signature = (
            str(model_path),
            device,
            ctx,
            n_batch,
            gpu_layers,
            top_k,
        )
        if self.signature == signature and self.is_loaded:
            ensure_cuda_vram_headroom("QwenUncensored GGUF")
            return

        self.unload()
        ensure_cuda_vram_headroom("QwenUncensored GGUF")

        mmproj = self._resolve_mmproj(model_entry, model_path)
        self.mmproj_path = mmproj

        n_gpu_layers = gpu_layers if device != "cpu" else 0

        llama_kwargs: dict[str, Any] = {
            "model_path": str(model_path),
            "n_ctx": ctx,
            "n_batch": n_batch,
            "n_gpu_layers": n_gpu_layers,
            "verbose": False,
            "chat_format": "qwen",
        }

        arch = read_gguf_architecture(model_path)
        self.is_qwen3 = _is_qwen3_gguf(arch, model_name)
        if self.is_qwen3:
            llama_kwargs["chat_template_kwargs"] = {"enable_thinking": False}
            print(f"[QwenUncensored GGUF] Qwen3 family detected (arch={arch}); disabling thinking.")

        if mmproj is not None:
            from llama_cpp.llama_chat_format import Llava15ChatHandler  # type: ignore
            self.chat_handler = Llava15ChatHandler(mmproj_path=str(mmproj))
            llama_kwargs["chat_handler"] = self.chat_handler
            llama_kwargs["image_min_tokens"] = 1024

        self.llm = Llama(**llama_kwargs)
        self.signature = signature

    def generate(self, conversation: list[dict[str, Any]], params: dict[str, Any]) -> str:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        ensure_cuda_vram_headroom("QwenUncensored GGUF")
        if hasattr(self.llm, "reset"):
            try:
                self.llm.reset()
            except Exception as exc:
                print(f"[QwenUncensored GGUF] llama context reset skipped: {exc}")

        max_tokens = params.get("max_tokens", 1024)
        temperature = params.get("temperature", 0.6)
        top_p = params.get("top_p", 0.9)
        top_k = params.get("top_k", 20)
        repetition_penalty = params.get("repetition_penalty", 1.0)
        seed = params.get("seed", 1)

        # Convert ComfyUI media tensors to base64 images
        messages: list[dict[str, Any]] = []
        for msg in conversation:
            role = msg.get("role", "user")
            content = msg.get("content", [])
            if isinstance(content, list):
                new_content: list[dict[str, Any]] = []
                for item in content:
                    if item.get("type") == "image":
                        b64 = tensor_to_base64_png(item.get("image"))
                        if b64:
                            new_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
                    elif item.get("type") == "text":
                        new_content.append({"type": "text", "text": item.get("text", "")})
                messages.append({"role": role, "content": new_content})
            else:
                messages.append({"role": role, "content": str(content)})

        output = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            repeat_penalty=float(repetition_penalty),
            seed=int(seed),
            stop=["<|im_start|>", "<|im_end|>", "<|endoftext|>"],
        )
        return output["choices"][0]["message"]["content"].strip()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_model_path(model_entry: dict[str, Any]) -> Path:
        filename = model_entry.get("filename")
        if filename:
            p = Path(filename)
            if p.is_absolute():
                return p
        repo_name = model_entry.get("repo_name") or model_entry.get("repo_id", "").split("/")[-1]
        author = model_entry.get("author") or "unknown"
        base_dir = Path("models/LLM/GGUF")
        try:
            import folder_paths  # type: ignore
            base_dir = Path(folder_paths.models_dir) / "LLM" / "GGUF"
        except Exception:
            pass
        if filename:
            return base_dir / author / repo_name / Path(filename).name
        raise ValueError("GGUF model entry missing filename")

    def _resolve_mmproj(self, model_entry: dict[str, Any], model_path: Path) -> Path | None:
        mmproj = model_entry.get("mmproj_file")
        if not mmproj:
            return None

        if Path(mmproj).is_absolute():
            return Path(mmproj) if Path(mmproj).exists() else None

        candidate = model_path.parent / mmproj
        if candidate.exists():
            return candidate

        # Search alternate LLM/GGUF paths
        try:
            import folder_paths  # type: ignore
            llm_roots = folder_paths.get_folder_paths("LLM")
        except Exception:
            llm_roots = []
        bare = Path(mmproj).name
        for root in llm_roots:
            matches = list(Path(root).rglob(bare))
            if matches:
                return matches[0]

        # If missing, raise informative error
        if candidate.exists():
            return candidate
        return None
