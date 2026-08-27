"""HuggingFace Transformers backend for Qwen models."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch

try:
    # transformers >= 5.0 renamed the vision-seq class to image-text-to-text
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
except ImportError:
    from transformers import AutoModelForVision2Seq

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    snapshot_download = None  # type: ignore[assignment]
    hf_hub_download = None  # type: ignore[assignment]

from ..config import HF_LOCAL_DIR, _default_models_dir, _get_llm_paths
from ..media import tensor_to_pil
from ..memory import ensure_cuda_vram_headroom, normalize_device, torch_gc
from .base import AbstractBackend


def _read_model_type(model_dir: str) -> str | None:
    try:
        path = Path(model_dir) / "config.json"
        if not path.exists():
            return None
        import json

        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("model_type")
    except Exception:
        return None


def _is_fp8_name(name: str) -> bool:
    return any(ind in name.lower() for ind in ("-fp8", "_fp8"))


def _is_qwen3_family(model_type: str | None, model_name: str) -> bool:
    if model_type and "qwen3" in model_type:
        return True
    return "qwen3" in model_name.lower()


class HFTransformersBackend(AbstractBackend):
    def __init__(self) -> None:
        self.model: Any = None
        self.processor: Any = None
        self.tokenizer: Any = None
        self.signature: tuple[Any, ...] | None = None
        self.is_qwen3 = False
        self.is_text_only = False
        self._device: str = "cpu"

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def unload(self) -> None:
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.signature = None
        self.is_qwen3 = False
        self.is_text_only = False
        torch_gc()

    def load(self, model_entry: dict[str, Any], params: dict[str, Any]) -> None:
        model_name = model_entry.get("repo_id") or model_entry.get("local_path") or "unknown"
        quantization = params.get("quantization", "None (FP16)")
        attention_mode = params.get("attention_mode", "auto")
        use_compile = params.get("use_torch_compile", False)
        device_choice = params.get("device", "auto")
        backend_type = model_entry.get("type", "vl")

        device = normalize_device(device_choice)
        quant_value = self._parse_quantization(quantization)
        force_sdpa = self._needs_sdpa(model_entry, quant_value)
        attn_impl = self._resolve_attention_mode(attention_mode, force_sdpa)

        signature = (model_name, backend_type, quant_value, attn_impl, device, use_compile)
        if self.signature == signature and self.is_loaded:
            ensure_cuda_vram_headroom("QwenUncensored HF")
            return

        self.unload()
        ensure_cuda_vram_headroom("QwenUncensored HF")

        model_path = self._ensure_model_path(model_entry)
        quant_config, dtype = self._quantization_config(model_entry, quant_value, device)

        # BitsAndBytes must load directly onto GPU via device_map
        if quant_config is not None and device == "cpu":
            print("[QwenUncensored HF] BitsAndBytes requires a GPU; falling back to FP32 on CPU")
            quant_config = None
            dtype = torch.float32

        actual_attn = attn_impl
        if attn_impl == "sage":
            actual_attn = "sdpa"

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "attn_implementation": actual_attn,
            "use_safetensors": True,
            "low_cpu_mem_usage": True,
        }

        if quant_config is not None:
            bnb_device_map = device if device.startswith("cuda") else "auto"
            load_kwargs.update({
                "device_map": bnb_device_map,
                "quantization_config": quant_config,
            })
        else:
            load_kwargs["torch_dtype"] = dtype

        if backend_type == "text":
            self.is_text_only = True
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        else:
            self.model = AutoModelForVision2Seq.from_pretrained(model_path, **load_kwargs).eval()
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Move non-quantized models to target device
        if quant_config is None and device != "cpu":
            try:
                self.model = self.model.to(device)
            except Exception as exc:
                print(f"[QwenUncensored HF] Failed to move model to {device}: {exc}")

        # Apply SageAttention patching if requested
        if attn_impl == "sage":
            self._apply_sage_attention()

        # Compile
        if use_compile and device.startswith("cuda") and torch.cuda.is_available():
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("[QwenUncensored HF] torch.compile enabled")
            except Exception as exc:
                print(f"[QwenUncensored HF] torch.compile skipped: {exc}")

        # Model family detection
        model_type = _read_model_type(model_path)
        self.is_qwen3 = _is_qwen3_family(model_type, model_name)
        self._device = device
        self.signature = signature

        if self.is_qwen3:
            print(f"[QwenUncensored HF] Qwen3 family detected (model_type={model_type}); disabling thinking.")

    def generate(self, conversation: list[dict[str, Any]], params: dict[str, Any]) -> str:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        ensure_cuda_vram_headroom("QwenUncensored HF")

        max_new_tokens = params.get("max_tokens", 1024)
        temperature = params.get("temperature", 0.6)
        top_p = params.get("top_p", 0.9)
        repetition_penalty = params.get("repetition_penalty", 1.0)
        num_beams = params.get("num_beams", 1)
        seed = params.get("seed")

        chat_kwargs = {}
        if self.is_qwen3:
            chat_kwargs["enable_thinking"] = False

        if self.is_text_only:
            prompt_text = conversation[-1].get("content", "") if conversation else ""
            messages = [{"role": "user", "content": prompt_text}]
            template_kwargs = {"tokenize": False, "add_generation_prompt": True}
            if chat_kwargs:
                template_kwargs["chat_template_kwargs"] = chat_kwargs
            try:
                formatted = self.tokenizer.apply_chat_template(messages, **template_kwargs)
            except Exception:
                formatted = prompt_text
            inputs = self.tokenizer(formatted, return_tensors="pt").to(self._device)
            model_inputs = {k: v.to(self._device) for k, v in inputs.items() if torch.is_tensor(v)}
        else:
            # Multimodal: build processor inputs from conversation
            # Images are already PIL Images (converted in base.py), extract them
            # in the same order they appear in the conversation content
            images = [
                item.get("image")
                for item in conversation[-1].get("content", [])
                if item.get("type") == "image" and item.get("image") is not None
            ]
            prompt_text = ""
            for item in conversation[-1].get("content", []):
                if item.get("type") == "text":
                    prompt_text = item.get("text", "")

            # Qwen3.5: prepend /no_think to disable thinking at the text level too
            if self.is_qwen3 and prompt_text and not prompt_text.startswith("/no_think"):
                for item in conversation[-1].get("content", []):
                    if item.get("type") == "text":
                        item["text"] = "/no_think\n" + item["text"]
                        break

            chat = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
                chat_template_kwargs=chat_kwargs or None,
            )
            inputs = self.processor(text=chat, images=images or None, return_tensors="pt")
            model_device = next(self.model.parameters()).device
            model_inputs = {k: v.to(model_device, non_blocking=True) if torch.is_tensor(v) else v for k, v in inputs.items()}

        stop_tokens = [self.tokenizer.eos_token_id]
        if hasattr(self.tokenizer, "eot_id") and self.tokenizer.eot_id is not None:
            stop_tokens.append(self.tokenizer.eot_id)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
            "num_beams": num_beams,
            "eos_token_id": stop_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        if num_beams == 1:
            gen_kwargs.update({
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            })
            if self.is_qwen3:
                gen_kwargs["top_k"] = 20
        else:
            gen_kwargs["do_sample"] = False

        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        with torch.no_grad():
            try:
                outputs = self.model.generate(**model_inputs, **gen_kwargs)
            except torch.cuda.OutOfMemoryError:
                print("[QwenUncensored HF] OOM during generation; retrying with reduced tokens")
                torch_gc()
                gen_kwargs["max_new_tokens"] = max(256, max_new_tokens // 2)
                outputs = self.model.generate(**model_inputs, **gen_kwargs)

        input_len = model_inputs["input_ids"].shape[-1]
        result = self.tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)
        return result.strip()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_quantization(value: str) -> str:
        mapping = {
            "4-bit (VRAM-friendly)": "4bit",
            "8-bit (Balanced)": "8bit",
            "None (FP16)": "fp16",
        }
        return mapping.get(value, "fp16")

    def _needs_sdpa(self, model_entry: dict[str, Any], quant_value: str) -> bool:
        if model_entry.get("quantized") or _is_fp8_name(model_entry.get("repo_id", "")):
            return True
        return quant_value in ("4bit", "8bit")

    def _resolve_attention_mode(self, mode: str, force_sdpa: bool) -> str:
        if force_sdpa or mode == "sdpa":
            return "sdpa"

        if mode == "sage":
            if self._sage_available():
                return "sage"
            print("[QwenUncensored HF] SageAttention unavailable; falling back to SDPA")
            return "sdpa"

        if mode == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
                return "flash_attention_2"
            except Exception:
                print("[QwenUncensored HF] Flash Attention 2 unavailable; falling back to SDPA")
            return "sdpa"

        # auto: sage -> flash -> sdpa
        if self._sage_available():
            print("[QwenUncensored HF] Auto attention: SageAttention")
            return "sage"
        try:
            import flash_attn  # noqa: F401
            print("[QwenUncensored HF] Auto attention: Flash Attention 2")
            return "flash_attention_2"
        except Exception:
            pass
        print("[QwenUncensored HF] Auto attention: SDPA")
        return "sdpa"

    @staticmethod
    def _sage_available() -> bool:
        try:
            from sageattention import (  # type: ignore
                sageattn_qk_int8_pv_fp16_cuda,
            )
            return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        except Exception:
            return False

    def _apply_sage_attention(self) -> None:
        try:
            from sageattention_patch import set_sage_attention  # type: ignore
            set_sage_attention(self.model)
            print("[QwenUncensored HF] SageAttention patching applied")
        except Exception as exc:
            print(f"[QwenUncensored HF] SageAttention patching failed: {exc}")

    @staticmethod
    def _ensure_model_path(model_entry: dict[str, Any]) -> str:
        local = model_entry.get("local_path")
        if local:
            target = Path(local)
            if target.exists() and target.is_dir():
                return str(target)
            raise FileNotFoundError(
                f"[QwenUncensored HF] Local model directory not found: {target}"
            )

        repo_id = model_entry.get("repo_id")
        if not repo_id:
            raise ValueError("Model entry has no repo_id or local_path")

        if snapshot_download is None:
            raise RuntimeError(
                "huggingface_hub is not installed; cannot download models."
            )

        llm_paths = _get_llm_paths()
        models_dir = (
            llm_paths[0] / HF_LOCAL_DIR
            if llm_paths
            else _default_models_dir() / HF_LOCAL_DIR
        )
        models_dir.mkdir(parents=True, exist_ok=True)
        target = models_dir / repo_id.split("/")[-1]

        if target.exists() and target.is_dir():
            if any(target.glob("*.safetensors")) or any(target.glob("*.bin")):
                return str(target)

        print(f"[QwenUncensored HF] Downloading {repo_id} to {target}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target),
            ignore_patterns=["*.md", ".git*", "*.msgpack", "*.h5"],
        )

        # Some community uploads omit preprocessor_config.json even when the model is
        # multimodal. AutoProcessor fails without it, so fetch it from the official base.
        preproc_path = target / "preprocessor_config.json"
        if not preproc_path.exists():
            config_path = target / "config.json"
            needs_vision = False
            if config_path.exists():
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                    needs_vision = "vision_config" in cfg
                except Exception:
                    pass
            if needs_vision and hf_hub_download is not None:
                print(
                    "[QwenUncensored HF] preprocessor_config.json missing — "
                    "fetching from Qwen/Qwen3.5-4B"
                )
                try:
                    hf_hub_download(
                        repo_id="Qwen/Qwen3.5-4B",
                        filename="preprocessor_config.json",
                        local_dir=str(target),
                    )
                except Exception as exc:
                    print(
                        f"[QwenUncensored HF] Could not fetch preprocessor_config.json: {exc}"
                    )

        return str(target)

    def _quantization_config(
        self,
        model_entry: dict[str, Any],
        quant_value: str,
        device: str,
    ) -> tuple[Any, Any]:
        if model_entry.get("quantized"):
            # Pre-quantized models should not use BnB
            return None, torch.float16 if device != "cpu" else torch.float32

        if quant_value == "4bit":
            cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            return cfg, None
        if quant_value == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True), None
        return None, torch.float16 if device != "cpu" else torch.float32
