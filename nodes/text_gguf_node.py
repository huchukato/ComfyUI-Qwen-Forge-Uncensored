"""Text-only / prompt enhancer nodes using GGUF."""

from __future__ import annotations

from .base import QwenUncensoredBaseNode, get_last_prompt, set_last_prompt
from qwen_forge.config import load_model_catalog


class QwenUncensoredTextGGUF(QwenUncensoredBaseNode):
    """Text prompt enhancer using GGUF via llama.cpp."""

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ENHANCED_OUTPUT",)
    FUNCTION = "process"
    CATEGORY = "Qwen-Uncensored"

    @classmethod
    def INPUT_TYPES(cls):
        catalog = load_model_catalog()
        models = [n for n, info in catalog.items() if info.get("backend") == "gguf" and info.get("type") == "text"]
        default_model = models[0] if models else "(no GGUF text models)"
        from qwen_forge.config import SYSTEM_PROMPTS_PATH
        from qwen_forge.prompts import load_prompt_config
        cfg = load_prompt_config(SYSTEM_PROMPTS_PATH)
        styles = ["✍️ Custom Only (no preset)"] + list(cfg.get("text", {}).get("styles", {}).keys())
        default_style = "📝 Enhance" if "📝 Enhance" in styles else (styles[0] if styles else "✍️ Custom Only (no preset)")

        return {
            "required": {
                "model_name": (models, {"default": default_model}),
                "enhancement_style": (styles, {"default": default_style}),
                "prompt_text": ("STRING", {"default": "", "multiline": True}),
                "custom_system_prompt": ("STRING", {"default": "", "multiline": True}),
                "max_tokens": ("INT", {"default": 1024, "min": 32, "max": 16384}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 0.5, "max": 2.0, "step": 0.05}),
                "ctx": ("INT", {"default": 32768, "min": 1024, "max": 262144, "step": 512}),
                "n_batch": ("INT", {"default": 512, "min": 64, "max": 32768, "step": 64}),
                "gpu_layers": ("INT", {"default": -1, "min": -1, "max": 200}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 32768}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2**32 - 1}),
                "keep_last_prompt": ("BOOLEAN", {"default": False}),
            },
        }

    def process(
        self,
        model_name,
        enhancement_style,
        prompt_text,
        custom_system_prompt,
        max_tokens,
        temperature,
        top_p,
        repetition_penalty,
        ctx,
        n_batch,
        gpu_layers,
        top_k,
        keep_model_loaded,
        seed,
        keep_last_prompt,
    ):
        if keep_last_prompt:
            last = get_last_prompt()
            if last:
                return (last,)
            return ("",)

        from qwen_forge.prompts import load_prompt_config, build_text_prompt
        cfg = load_prompt_config("system_prompts.json")
        styles = cfg.get("text", {}).get("styles", {})
        merged_prompt = build_text_prompt(
            enhancement_style,
            custom_system_prompt,
            prompt_text,
            styles,
            guard=True,
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
        }

        result = self._run_inference(
            model_name=model_name,
            prompt_text=merged_prompt,
            params=params,
            backend_type="gguf",
            media=None,
        )
        set_last_prompt(result)
        self._maybe_unload(keep_model_loaded)
        return (result,)


NODE_CLASS_MAPPINGS = {"QwenUncensoredTextGGUF": QwenUncensoredTextGGUF}
NODE_DISPLAY_NAME_MAPPINGS = {"QwenUncensoredTextGGUF": "Qwen-Uncensored Text (GGUF)"}
