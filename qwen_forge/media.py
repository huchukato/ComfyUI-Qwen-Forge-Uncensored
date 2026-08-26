"""Media conversion helpers for ComfyUI tensors."""

from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def tensor_to_pil(tensor: torch.Tensor | np.ndarray | None) -> Image.Image | None:
    """Convert a ComfyUI IMAGE tensor to a PIL Image."""
    if tensor is None:
        return None

    if isinstance(tensor, torch.Tensor):
        if tensor.dim() == 4:
            tensor = tensor[0]
        array = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    elif isinstance(tensor, np.ndarray):
        array = tensor
    else:
        return None

    if array.ndim == 2:
        return Image.fromarray(array, mode="L")
    if array.shape[-1] == 1:
        return Image.fromarray(array[..., 0], mode="L")
    if array.shape[-1] == 4:
        return Image.fromarray(array, mode="RGBA")
    return Image.fromarray(array[..., :3], mode="RGB")


def tensor_to_base64_png(tensor: torch.Tensor | None) -> str | None:
    """Convert a tensor to a base64 PNG string."""
    pil_img = tensor_to_pil(tensor)
    if pil_img is None:
        return None
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def extract_frames(
    video_tensor: torch.Tensor,
    frame_count: int = 1,
) -> list[Image.Image]:
    """Extract a fixed number of evenly spaced frames from a video tensor.

    Expected shape: [batch/float, frames, height, width, channels] or
    [frames, height, width, channels].
    """
    if video_tensor is None or video_tensor.numel() == 0:
        return []

    # Normalize to [frames, h, w, c]
    if video_tensor.dim() == 5:
        video_tensor = video_tensor[0]
    if video_tensor.dim() != 4:
        return []

    total_frames = int(video_tensor.shape[0])
    if total_frames <= frame_count:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, frame_count, dtype=int).tolist()

    return [tensor_to_pil(video_tensor[i]) for i in indices if tensor_to_pil(video_tensor[i]) is not None]


def get_media_hash(media: torch.Tensor | None) -> str | None:
    """Generate a short deterministic hash for a media tensor."""
    if media is None or media.numel() == 0:
        return None
    try:
        shape = str(media.shape)
        dtype = str(media.dtype)
        sample = media.flatten()[:50].tolist() if media.numel() > 0 else []
        content = f"{shape}_{dtype}_{sample[:10]}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    except Exception:
        return None
