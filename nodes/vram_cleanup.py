import gc

import torch
from comfy import model_management


class QwenUncensoredVRAMCleanup:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("*",),
                "cleanup_mode": (
                    ["Cache Only", "Text Encoder", "Full Cleanup"],
                    {"default": "Cache Only"},
                ),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output",)
    FUNCTION = "cleanup_vram_memory"
    CATEGORY = "Qwen Forge/Tools"
    OUTPUT_NODE = True

    def cleanup_vram_memory(self, input, cleanup_mode):
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        print(f"[QwenUncensored] Starting VRAM cleanup: {cleanup_mode}")

        if cleanup_mode == "Cache Only":
            self._cache_only()
        elif cleanup_mode == "Text Encoder":
            self._text_encoder()
        elif cleanup_mode == "Full Cleanup":
            self._full_cleanup()

        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            freed_memory = max(initial_memory - final_memory, 0)
            print(
                f"[QwenUncensored] VRAM cleanup complete: "
                f"allocated={final_memory / 1024**3:.2f}GB, freed={freed_memory / 1024**3:.2f}GB"
            )
        return (input,)

    @staticmethod
    def _cache_only():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _text_encoder():
        gc.collect()
        if not torch.cuda.is_available():
            return
        for index in range(3):
            torch.cuda.empty_cache()
            if index == 1:
                torch.cuda.synchronize()

    @staticmethod
    def _full_cleanup():
        model_management.unload_all_models()
        gc.collect()
        if not torch.cuda.is_available():
            return
        for _ in range(5):
            torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        torch.cuda.synchronize()

NODE_CLASS_MAPPINGS = {"QwenUncensoredVRAMCleanup": QwenUncensoredVRAMCleanup}
NODE_DISPLAY_NAME_MAPPINGS = {"QwenUncensoredVRAMCleanup": "Qwen Forge | VRAM Cleanup"}
