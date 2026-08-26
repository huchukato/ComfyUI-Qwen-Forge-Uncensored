"""Model catalog and backend factory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

NODE_DIR = Path(__file__).parent.parent
MODELS_PATH = NODE_DIR / "models.json"
CUSTOM_MODELS_PATH = NODE_DIR / "custom_models.json"
SYSTEM_PROMPTS_PATH = NODE_DIR / "system_prompts.json"
HF_LOCAL_DIR = "Qwen-Forge"


def _safe_dirname(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return "unknown"
    return "".join(ch for ch in value if ch.isalnum() or ch in "._- ").strip() or "unknown"


def _get_llm_paths() -> list[Path]:
    """Return extra ComfyUI LLM folder paths if available."""
    try:
        import folder_paths  # type: ignore

        if "LLM" in folder_paths.folder_names_and_paths:
            return [Path(p) for p in folder_paths.get_folder_paths("LLM")]
    except Exception:
        pass
    return []


def _default_models_dir() -> Path:
    try:
        import folder_paths  # type: ignore

        return Path(folder_paths.models_dir) / "LLM"
    except Exception:
        return NODE_DIR / "models"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        print(f"[QwenUncensored] Failed to load {path.name}: {exc}")
        return {}


def _is_hf_model_dir(path: Path) -> bool:
    return path.is_dir() and (any(path.glob("*.safetensors")) or any(path.glob("*.bin")))


def _scan_local_hf_models(catalog: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Add locally available HF model directories not already in the catalog."""
    known_names: set[str] = set()
    for info in catalog.values():
        repo_id = info.get("repo_id")
        if isinstance(repo_id, str) and "/" in repo_id:
            known_names.add(repo_id.split("/")[-1])
        local = info.get("local_path")
        if local:
            known_names.add(Path(local).name)

    scan_dirs: set[Path] = set()
    for root in _get_llm_paths():
        scan_dirs.add(root / HF_LOCAL_DIR)
    scan_dirs.add(_default_models_dir() / HF_LOCAL_DIR)

    new_models: dict[str, Any] = {}
    for models_dir in scan_dirs:
        if not models_dir.exists():
            continue
        for entry in models_dir.iterdir():
            if entry.name in known_names or not _is_hf_model_dir(entry):
                continue
            display = f"[local] {entry.name}"
            new_models[display] = {
                "local_path": str(entry),
                "repo_id": None,
                "is_local": True,
                "quantized": False,
                "backend": "hf",
                "type": "vl",
            }
            known_names.add(entry.name)

    return new_models


def _scan_local_gguf_models(
    base_dir: Path,
    existing_filenames: set[str],
) -> dict[str, Any]:
    """Scan GGUF directories for local models not in the catalog."""
    models: dict[str, Any] = {}
    if not base_dir.exists() or not base_dir.is_dir():
        return models

    dirs_with_gguf: dict[Path, list[Path]] = {}
    try:
        for gguf_file in base_dir.rglob("*.gguf"):
            if gguf_file.is_file():
                dirs_with_gguf.setdefault(gguf_file.parent, []).append(gguf_file)
    except PermissionError:
        pass

    for dir_path, files in dirs_with_gguf.items():
        mmproj_files = [f for f in files if "mmproj" in f.name.lower()]
        model_files = [f for f in files if "mmproj" not in f.name.lower()]

        for model_file in model_files:
            if model_file.name in existing_filenames:
                continue
            display = f"[local] {model_file.name}"
            entry: dict[str, Any] = {
                "filename": str(model_file),
                "is_local": True,
                "repo_id": None,
                "author": None,
                "repo_dirname": dir_path.name,
                "backend": "gguf",
                "context_length": 32768,
                "n_batch": 512,
                "gpu_layers": -1,
                "top_k": 20,
            }
            if mmproj_files:
                entry["type"] = "vl"
                entry["mmproj_file"] = str(mmproj_files[0])
                entry["image_max_tokens"] = 4096
                entry["pool_size"] = 4194304
            else:
                entry["type"] = "text"
            models[display] = entry
            existing_filenames.add(model_file.name)

    return models


def _flatten_gguf_section(
    section: dict[str, Any] | None,
    backend_type: str,
) -> dict[str, Any]:
    """Flatten a GGUF catalog section (e.g. gguf_vl_models)."""
    flattened: dict[str, Any] = {}
    if not section:
        return flattened

    seen: set[str] = set()
    for repo_key, repo in section.items():
        if not isinstance(repo, dict):
            continue
        author = repo.get("author") or repo.get("publisher")
        repo_name = repo.get("repo_name") or repo_key
        model_files = repo.get("model_files") or []
        mmproj = repo.get("mmproj_file")

        for filename in model_files:
            display = Path(filename).name
            if display in seen:
                continue
            seen.add(display)
            entry = {
                **repo,
                "filename": filename,
                "repo_dirname": repo_name,
                "backend": backend_type,
            }
            if mmproj:
                entry["type"] = "vl"
            else:
                entry["type"] = "text"
            flattened[display] = entry

    return flattened


def load_model_catalog() -> dict[str, Any]:
    """Load unified model catalog, including custom overrides and local scans."""
    data = load_json(MODELS_PATH)
    custom = load_json(CUSTOM_MODELS_PATH)

    catalog: dict[str, Any] = {}

    # HF models
    hf_vl = {k: {**v, "backend": "hf", "type": "vl"} for k, v in data.get("hf_vl_models", {}).items()}
    hf_text = {k: {**v, "backend": "hf", "type": "text"} for k, v in data.get("hf_text_models", {}).items()}
    catalog.update(hf_vl)
    catalog.update(hf_text)

    if isinstance(custom.get("hf_vl_models"), dict):
        for k, v in custom["hf_vl_models"].items():
            catalog[k] = {**v, "backend": "hf", "type": "vl"}
    if isinstance(custom.get("hf_text_models"), dict):
        for k, v in custom["hf_text_models"].items():
            catalog[k] = {**v, "backend": "hf", "type": "text"}

    # Local HF scan
    catalog.update(_scan_local_hf_models(catalog))

    # GGUF models
    gguf_vl = _flatten_gguf_section(data.get("gguf_vl_models"), "gguf")
    gguf_text = _flatten_gguf_section(data.get("gguf_text_models"), "gguf")
    catalog.update(gguf_vl)
    catalog.update(gguf_text)

    if isinstance(custom.get("gguf_vl_models"), dict):
        catalog.update(_flatten_gguf_section(custom["gguf_vl_models"], "gguf"))
    if isinstance(custom.get("gguf_text_models"), dict):
        catalog.update(_flatten_gguf_section(custom["gguf_text_models"], "gguf"))

    # Local GGUF scan
    base_dir_value = data.get("base_dir") or custom.get("base_dir") or "LLM/GGUF"
    base_dir = Path(base_dir_value)
    if not base_dir.is_absolute():
        base_dir = _default_models_dir() / base_dir

    scan_dirs = {base_dir}
    for llm_root in _get_llm_paths():
        scan_dirs.add(llm_root / "GGUF")
        scan_dirs.add(llm_root)

    existing = {Path(e.get("filename", "")).name for e in catalog.values() if e.get("filename")}
    for scan_dir in scan_dirs:
        local_models = _scan_local_gguf_models(scan_dir, existing)
        catalog.update(local_models)
        existing.update({Path(e.get("filename", "")).name for e in local_models.values() if e.get("filename")})

    return catalog


def get_backend_class(backend: str):
    """Return the backend class for a given backend name."""
    if backend == "hf":
        from .backends.hf_transformers import HFTransformersBackend

        return HFTransformersBackend
    if backend == "gguf":
        from .backends.gguf import GGUFBackend

        return GGUFBackend
    raise ValueError(f"Unknown backend: {backend}")


def resolve_model_entry(model_name: str, catalog: dict[str, Any] | None = None) -> dict[str, Any]:
    """Look up a model entry by display name."""
    catalog = catalog or load_model_catalog()
    entry = catalog.get(model_name)
    if not entry:
        raise ValueError(f"Model '{model_name}' not found in catalog. Check models.json or local scan.")
    return entry
