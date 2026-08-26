"""System prompt / preset loading and merging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .output_cleaner import prompt_output_guard


def load_prompt_config(path: str | Path) -> dict[str, Any]:
    """Load system prompts JSON."""
    path = Path(path)
    if not path.exists():
        return {"_presets": [], "text": {"translation_prompt": "", "styles": {}}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
    except Exception as exc:
        print(f"[QwenUncensored] Failed to load system prompts: {exc}")
        return {"_presets": [], "text": {"translation_prompt": "", "styles": {}}}

    presets = data.get("_presets", [])
    return {
        "_presets": presets,
        "_preset_groups": group_presets(presets),
        "presets": data.get("presets", {}),
        "text": data.get("text", {"translation_prompt": "", "styles": {}}),
    }


def group_presets(presets: list[str]) -> dict[str, list[str]]:
    groups = {"MiniMax H3": [], "Wan 2.2": [], "LTX": [], "Generic": []}
    for preset in presets:
        if "MiniMax H3" in preset:
            groups["MiniMax H3"].append(preset)
        elif "Wan 2.2" in preset:
            groups["Wan 2.2"].append(preset)
        elif "LTX" in preset:
            groups["LTX"].append(preset)
        else:
            groups["Generic"].append(preset)
    return {name: values for name, values in groups.items() if values}


def build_prompt(
    preset_name: str,
    custom_prompt: str,
    presets: dict[str, str],
    guard: bool = True,
    sfw: bool = False,
) -> str:
    """Merge a preset with custom input and optional guards.

    Args:
        preset_name: key in `presets`.
        custom_prompt: user extra text.
        presets: mapping of preset name -> system prompt string.
        guard: whether to append the output guard.
        sfw: if True, prepend an SFW instruction. Not implemented by default.
    """
    system = (presets.get(preset_name) or "").strip()
    custom = (custom_prompt or "").strip()

    parts = []
    if sfw:
        parts.append(
            "[MODE: SFW] Produce only general-audience content. "
            "No explicit nudity, sexual content, graphic violence, or gore."
        )
    if custom:
        parts.append(custom)
    if system:
        parts.append(system)
    if guard:
        parts.append(prompt_output_guard())

    return "\n\n".join(parts).strip()


def build_text_prompt(
    style_name: str,
    custom_system_prompt: str,
    prompt_text: str,
    styles: dict[str, dict[str, str]],
    guard: bool = True,
) -> str:
    """Build a text-only prompt from a style and custom input."""
    style_prompt = (styles.get(style_name, {}).get("system_prompt") or "").strip()
    custom = (custom_system_prompt or "").strip()

    if not style_prompt and not custom:
        raise ValueError("No system prompt or custom instruction provided.")

    system_parts = [p for p in (custom, style_prompt) if p]
    system = "\n\n".join(system_parts)
    if guard:
        system = f"{system}\n\n{prompt_output_guard()}".strip()

    user = (prompt_text or "").strip() or "Describe a scene vividly."
    return f"{user}\n\n{system}".strip()
