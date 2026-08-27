"""HuggingFace Transformers vision/multimodal nodes."""

from __future__ import annotations

import torch

from .base import QwenUncensoredBaseNode, get_last_prompt, set_last_prompt
from qwen_forge.config import load_model_catalog


class QwenUncensoredVision(QwenUncensoredBaseNode):
    """Multimodal Qwen node using HuggingFace Transformers."""

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("RESPONSE",)
    FUNCTION = "process"
    CATEGORY = "Qwen Forge"

    @classmethod
    def INPUT_TYPES(cls):
        catalog = load_model_catalog()
        models = [n for n, info in catalog.items() if info.get("backend") == "hf" and info.get("type") == "vl"]
        default_model = models[0] if models else "(no HF VL models)"
        from qwen_forge.config import SYSTEM_PROMPTS_PATH
        from qwen_forge.prompts import load_prompt_config
        cfg = load_prompt_config(SYSTEM_PROMPTS_PATH)
        presets = cfg.get("_presets", [])
        preset_groups = cfg.get("_preset_groups", {})
        families = list(preset_groups) or ["Generic"]
        default_preset = preset_groups.get(families[0], presets)[0] if presets else "Describe this image in detail."

        return {
            "required": {
                "model_name": (models, {"default": default_model}),
                "quantization": (["None (FP16)", "8-bit (Balanced)", "4-bit (VRAM-friendly)"], {"default": "None (FP16)"}),
                "attention_mode": (["auto", "sage", "flash_attention_2", "sdpa"], {"default": "auto"}),
                "preset_family": (families, {"default": families[0]}),
                "preset_prompt": (presets, {"default": default_preset}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1}),
                "keep_last_prompt": ("BOOLEAN", {"default": False}),
                "image2_to_video": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image2_video": ("IMAGE",),
            },
        }

    def process(
        self,
        model_name,
        quantization,
        attention_mode,
        preset_family,
        preset_prompt,
        custom_prompt,
        keep_model_loaded,
        seed,
        keep_last_prompt,
        image2_to_video,
        image=None,
        image2_video=None,
    ):
        return self._process_common(
            model_name=model_name,
            quantization=quantization,
            attention_mode=attention_mode,
            preset_prompt=preset_prompt,
            custom_prompt=custom_prompt,
            keep_model_loaded=keep_model_loaded,
            seed=seed,
            keep_last_prompt=keep_last_prompt,
            image2_to_video=image2_to_video,
            image=image,
            image2=image2_video,
        )

    def _process_common(self, **kwargs):
        keep_last_prompt = kwargs.pop("keep_last_prompt", False)
        if keep_last_prompt:
            last = get_last_prompt()
            if last:
                return (last,)
            return ("",)

        from qwen_forge.prompts import load_prompt_config, build_prompt
        cfg = load_prompt_config("system_prompts.json")
        system_prompt = build_prompt(
            kwargs["preset_prompt"],
            kwargs["custom_prompt"],
            cfg.get("presets", {}),
            guard=False,
            sfw=False,
        )

        params = {
            "quantization": kwargs["quantization"],
            "attention_mode": kwargs["attention_mode"],
            "device": "auto",
            "max_tokens": self._model_default(kwargs["model_name"], "max_tokens", 8192),
            "temperature": 0.6,
            "top_p": 0.9,
            "repetition_penalty": 1.2,
            "num_beams": 1,
            "seed": kwargs["seed"],
            "use_torch_compile": False,
        }

        media = {
            "image": kwargs.get("image"),
            "image2": kwargs.get("image2"),
            "frame_count": 16 if kwargs.get("image2_to_video") else 1,
        }
        result = self._run_inference(
            model_name=kwargs["model_name"],
            prompt_text=system_prompt,
            params=params,
            backend_type="hf",
            media=media,
        )
        set_last_prompt(result)
        self._maybe_unload(kwargs["keep_model_loaded"])
        return (result,)


class QwenUncensoredVisionAdvanced(QwenUncensoredBaseNode):
    """Advanced multimodal Qwen node with full parameter control."""

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("RESPONSE",)
    FUNCTION = "process"
    CATEGORY = "Qwen Forge"

    @classmethod
    def INPUT_TYPES(cls):
        catalog = load_model_catalog()
        models = [n for n, info in catalog.items() if info.get("backend") == "hf" and info.get("type") == "vl"]
        default_model = models[0] if models else "(no HF VL models)"
        from qwen_forge.config import SYSTEM_PROMPTS_PATH
        from qwen_forge.prompts import load_prompt_config
        cfg = load_prompt_config(SYSTEM_PROMPTS_PATH)
        presets = cfg.get("_presets", [])
        preset_groups = cfg.get("_preset_groups", {})
        families = list(preset_groups) or ["Generic"]
        default_preset = preset_groups.get(families[0], presets)[0] if presets else "Describe this image in detail."

        num_gpus = torch.cuda.device_count()
        gpu_list = [f"cuda:{i}" for i in range(num_gpus)]
        device_options = ["auto", "cpu", "mps"] + gpu_list

        return {
            "required": {
                "model_name": (models, {"default": default_model}),
                "quantization": (["None (FP16)", "8-bit (Balanced)", "4-bit (VRAM-friendly)"], {"default": "None (FP16)"}),
                "attention_mode": (["auto", "sage", "flash_attention_2", "sdpa"], {"default": "auto"}),
                "use_torch_compile": ("BOOLEAN", {"default": False}),
                "device": (device_options, {"default": "auto"}),
                "preset_family": (families, {"default": families[0]}),
                "preset_prompt": (presets, {"default": default_preset}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True}),
                "max_tokens": ("INT", {"default": 8192, "min": 64, "max": 16384}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
                "num_beams": ("INT", {"default": 1, "min": 1, "max": 8}),
                "repetition_penalty": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05}),
                "frame_count": ("INT", {"default": 16, "min": 1, "max": 64}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1}),
                "keep_last_prompt": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image2": ("IMAGE",),
            },
        }

    def process(
        self,
        model_name,
        quantization,
        attention_mode,
        use_torch_compile,
        device,
        preset_family,
        preset_prompt,
        custom_prompt,
        max_tokens,
        temperature,
        top_p,
        num_beams,
        repetition_penalty,
        frame_count,
        keep_model_loaded,
        seed,
        keep_last_prompt,
        image=None,
        image2=None,
    ):
        keep_last_prompt = keep_last_prompt
        if keep_last_prompt:
            last = get_last_prompt()
            if last:
                return (last,)
            return ("",)

        from qwen_forge.prompts import load_prompt_config, build_prompt
        cfg = load_prompt_config("system_prompts.json")
        system_prompt = build_prompt(
            preset_prompt,
            custom_prompt,
            cfg.get("presets", {}),
            guard=False,
            sfw=False,
        )

        params = {
            "quantization": quantization,
            "attention_mode": attention_mode,
            "use_torch_compile": use_torch_compile,
            "device": device,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "num_beams": num_beams,
            "repetition_penalty": repetition_penalty,
            "seed": seed,
        }

        media = {"image": image, "image2": image2, "frame_count": frame_count}
        result = self._run_inference(
            model_name=model_name,
            prompt_text=system_prompt,
            params=params,
            backend_type="hf",
            media=media,
        )
        set_last_prompt(result)
        self._maybe_unload(keep_model_loaded)
        return (result,)


NODE_CLASS_MAPPINGS = {
    "QwenUncensoredVision": QwenUncensoredVision,
    "QwenUncensoredVisionAdvanced": QwenUncensoredVisionAdvanced,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenUncensoredVision": "Qwen Forge | Vision",
    "QwenUncensoredVisionAdvanced": "Qwen Forge | Vision (Advanced)",
}
