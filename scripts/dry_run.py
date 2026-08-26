"""Dry-run script to validate imports and catalog loading without ComfyUI."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))


def main() -> int:
    print("[dry-run] Importing lightweight qwen_forge modules...")
    from qwen_forge import config, output_cleaner, prompts
    from qwen_forge.backends.base import AbstractBackend

    print("[dry-run] Loading model catalog...")
    catalog = config.load_model_catalog()
    print(f"[dry-run] Found {len(catalog)} model entries")
    for name, info in catalog.items():
        print(f"  - {name}: backend={info.get('backend')}, type={info.get('type')}")

    print("[dry-run] Loading prompt config...")
    cfg = prompts.load_prompt_config(repo_root / "system_prompts.json")
    print(f"[dry-run] Presets: {len(cfg.get('_presets', []))}")
    print(f"[dry-run] Text styles: {len(cfg.get('text', {}).get('styles', {}))}")

    print("[dry-run] Testing prompt builder...")
    sample = prompts.build_prompt(
        cfg["_presets"][0] if cfg["_presets"] else "Describe",
        "a cinematic scene",
        cfg.get("presets", {}),
        guard=True,
        sfw=False,
    )
    print(f"[dry-run] Prompt preview: {sample[:120]}...")

    print("[dry-run] AbstractBackend present: yes")

    print("[dry-run] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
