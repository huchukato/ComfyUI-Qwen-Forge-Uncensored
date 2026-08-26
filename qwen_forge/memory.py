"""VRAM cleanup and device utilities."""

from __future__ import annotations

import gc
from typing import Any

import torch


def normalize_device(device: str) -> str:
    """Normalize a user device choice into a valid torch device string."""
    device = (device or "auto").strip().lower()
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if device == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if device.startswith("cuda:"):
        if not torch.cuda.is_available():
            return "cpu"
        try:
            idx = int(device.split(":", 1)[1])
            if idx >= torch.cuda.device_count():
                return "cuda:0"
        except ValueError:
            return "cuda:0"
        return device

    if device == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if device == "cpu":
        return "cpu"

    # Fallback for unknown strings
    return "cpu"


def ensure_cuda_vram_headroom(
    module_name: str = "QwenUncensored",
    min_free_gb: float = 1.0,
    min_free_ratio: float = 0.08,
) -> bool:
    """Free CUDA reserved memory if available VRAM is below a threshold."""
    if not torch.cuda.is_available():
        return True

    try:
        torch.cuda.synchronize()
    except Exception:
        pass

    try:
        free_before, total = torch.cuda.mem_get_info()
    except Exception:
        gc.collect()
        torch.cuda.empty_cache()
        return True

    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    reclaimable = max(reserved - allocated, 0)
    threshold = max(int(min_free_gb * 1024**3), int(total * min_free_ratio))

    if free_before >= threshold and reclaimable < 512 * 1024 * 1024:
        return True

    print(
        f"[{module_name}] VRAM headroom low: "
        f"free={free_before / 1024**3:.2f}GB, "
        f"reserved={reserved / 1024**3:.2f}GB, "
        f"allocated={allocated / 1024**3:.2f}GB. Cleaning CUDA cache..."
    )

    gc.collect()
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
    try:
        torch.cuda.synchronize()
    except Exception:
        pass

    try:
        free_after, _ = torch.cuda.mem_get_info()
        print(f"[{module_name}] VRAM after cleanup: free={free_after / 1024**3:.2f}GB")
        return free_after >= threshold
    except Exception:
        return True


def torch_gc() -> None:
    """Aggressive garbage collection for torch."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
