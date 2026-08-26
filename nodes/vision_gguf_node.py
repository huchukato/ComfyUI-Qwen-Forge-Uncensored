"""GGUF vision/multimodal nodes."""

from __future__ import annotations

from .base import QwenUncensoredBaseNode, get_last_prompt, set_last_prompt
from qwen_forge.config import load_model_catalog


class QwenUncensoredVisionGGUF(QwenUncensoredBaseNode):
    """Multimodal Qwen node using GGUF via llama.cpp."""

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("RESPONSE",)
    FUNCTION = "process"
    CATEGORY = "Qwen-Uncensored"

    @classmethod
    def INPUT_TYPES(cls):
        catalog = load_model_catalog()
        models = [n for n, info in catalog.items() if info.get("backend") == "gguf" and info.get("type") == "vl"]
        default_model = models[0] if models else "(no GGUF VL models)"
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
                "preset_family": (families, {"default": families[0]}),
                "preset_prompt": (presets, {"default": default_preset}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
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
            guard=True,
            sfw=False,
        )

        params = {
            "device": "auto",
            "max_tokens": self._model_default(model_name, "max_tokens", 8192),
            "temperature": 0.6,
            "top_p": 0.9,
            "repetition_penalty": 1.0,
            "seed": seed,
            "ctx": 32768,
            "n_batch": 512,
            "gpu_layers": -1,
            "top_k": 20,
            "pool_size": 4194304,
        }

        media = {"image": image, "image2": image2_video, "frame_count": 16 if image2_to_video else 1}
        result = self._run_inference(
            model_name=model_name,
            prompt_text=system_prompt,
            params=params,
            backend_type="gguf",
            media=media,
        )
        set_last_prompt(result)
        self._maybe_unload(keep_model_loaded)
        return (result,)


class QwenUncensoredVisionGGUFAdvanced(QwenUncensoredBaseNode):
    """Advanced multimodal GGUF node with full parameter control."""

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("RESPONSE",)
    FUNCTION = "process"
    CATEGORY = "Qwen-Uncensored"

    @classmethod
    def INPUT_TYPES(cls):
        catalog = load_model_catalog()
        models = [n for n, info in catalog.items() if info.get("backend") == "gguf" and info.get("type") == "vl"]
        default_model = models[0] if models else "(no GGUF VL models)"
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
                "preset_family": (families, {"default": families[0]}),
                "preset_prompt": (presets, {"default": default_preset}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True}),
                "max_tokens": ("INT", {"default": 8192, "min": 64, "max": 16384}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
                "repetition_penalty": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05}),
                "frame_count": ("INT", {"default": 16, "min": 1, "max": 64}),
                "ctx": ("INT", {"default": 32768, "min": 1024, "max": 262144, "step": 512}),
                "n_batch": ("INT", {"default": 512, "min": 64, "max": 32768, "step": 64}),
                "gpu_layers": ("INT", {"default": -1, "min": -1, "max": 200}),
                "image_max_tokens": ("INT", {"default": 4096, "min": 256, "max": 1024000, "step": 256}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 32768}),
                "pool_size": ("INT", {"default": 4194304, "min": 1048576, "max": 10485760, "step": 524288}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
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
        preset_family,
        preset_prompt,
        custom_prompt,
        max_tokens,
        temperature,
        top_p,
        repetition_penalty,
        frame_count,
        ctx,
        n_batch,
        gpu_layers,
        image_max_tokens,
        top_k,
        pool_size,
        keep_model_loaded,
        seed,
        keep_last_prompt,
        image=None,
        image2=None,
    ):
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
            guard=True,
            sfw=False,
        )

        params = {
            "device": "auto",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "seed": seed,
            "ctx": ctx,
            "n_batch": n_batch,
            "gpu_layers": gpu_layers,
            "top_k": top_k,
            "pool_size": pool_size,
            "image_max_tokens": image_max_tokens,
        }

        media = {"image": image, "image2": image2, "frame_count": frame_count}
        result = self._run_inference(
            model_name=model_name,
            prompt_text=system_prompt,
            params=params,
            backend_type="gguf",
            media=media,
        )
        set_last_prompt(result)
        self._maybe_unload(keep_model_loaded)
        return (result,)


NODE_CLASS_MAPPINGS = {
    "QwenUncensoredVisionGGUF": QwenUncensoredVisionGGUF,
    "QwenUncensoredVisionGGUFAdvanced": QwenUncensoredVisionGGUFAdvanced,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenUncensoredVisionGGUF": "Qwen-Uncensored Vision (GGUF)",
    "QwenUncensoredVisionGGUFAdvanced": "Qwen-Uncensored Vision GGUF (Advanced)",
}
