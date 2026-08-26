# ComfyUI-Qwen-Forge-Uncensored

![ComfyUI Qwen Forge Uncensored — Automated Prompting for Video Generation](img/qwen-forge.jpeg)

A clean, modular ComfyUI node pack for generating uncensored multimodal and text prompts with Qwen, designed specifically for AI video-generation workflows.

## Features

- **Pluggable backends**: HuggingFace Transformers and GGUF (llama.cpp) out of the box; vLLM and MLX stubs ready for future support.
- **Multimodal and text-only nodes** for video/animation prompt generation.
- **Grouped system prompt presets** for MiniMax H3, Wan 2.2, LTX and generic image/video analysis.
- **Smart prompt cache**, VRAM cleanup, and `keep_model_loaded` support.
- **Qwen3.x family detection** with automatic thinking disabled.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/huchukato/ComfyUI-Qwen-Forge-Uncensored.git
```

Install dependencies (if not already present):

```bash
cd ComfyUI-Qwen-Forge-Uncensored
pip install -r requirements.txt
```

For GGUF vision support, install a vision-capable `llama-cpp-python` wheel for your platform.

### Installing `llama-cpp-python`

The GGUF backend uses `llama-cpp-python`. The generic package on PyPI may compile `llama.cpp` from source on Linux/CUDA systems. If you prefer a faster, pre-built wheel, install one that matches your **Python version**, **OS**, and **CUDA version** before running `pip install -r requirements.txt`.

For example, on Linux with Python 3.12 and CUDA 13.1:

```bash
pip install https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.48-cu131-linux-20260821/llama_cpp_python-0.3.48+cu131-cp312-cp312-linux_x86_64.whl
```

Pre-built wheels are also available for macOS and Windows from the same source. If a compatible `llama-cpp-python` is already installed, the requirements step will not recompile or replace it.

## Nodes

| Node | Backend | Purpose |
|------|---------|---------|
| `Qwen Forge | Vision` | HF Transformers | Multimodal inference (image/video) |
| `Qwen Forge | Vision (Advanced)` | HF Transformers | Multimodal with full parameter control |
| `Qwen Forge | Vision (GGUF)` | GGUF | Multimodal inference via llama.cpp |
| `Qwen Forge | Vision (GGUF Advanced)` | GGUF | Multimodal GGUF with full parameters |
| `Qwen Forge | Text` | HF Transformers | Prompt enhancement / text-only |
| `Qwen Forge | Text (GGUF)` | GGUF | Prompt enhancement / text-only via llama.cpp |
| `Qwen Forge | VRAM Cleanup` | Utility | Pass-through VRAM cache cleanup and model unloading |
| `Qwen Forge | Story Split` | Utility | Split a story into four separate prompts |

## Configuration

- `models.json` — catalog of supported HF and GGUF models.
- `system_prompts.json` — vision presets and text styles.
- `custom_models.json` — optional user overrides (not tracked by git).

HF models are downloaded into `models/LLM/Qwen-Forge`; local models in `models/LLM/Qwen-Forge` and `models/LLM/GGUF` are discovered automatically.

## License

GPL-3.0 — see [LICENSE](LICENSE).
