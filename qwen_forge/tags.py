"""Camera tag, style tag, and Danbooru tag guidance for Qwen Forge nodes.

Ported from ComfyUI-QwenVL-Mod AILab_QwenVL.py.
"""

from __future__ import annotations

# ── Camera tag dropdown options ───────────────────────────────────
CAMERA_TAG_OPTIONS = [
    "None",
    "[STATIC_CAMERA]",
    "[LOCKED_OFF]",
    "[SLOW_ZOOM_IN]",
    "[SLOW_ZOOM_OUT]",
    "[FAST_ZOOM_IN]",
    "[FAST_ZOOM_OUT]",
    "[PAN_LEFT]",
    "[PAN_RIGHT]",
    "[TILT_UP]",
    "[TILT_DOWN]",
    "[DOLLY_IN]",
    "[DOLLY_OUT]",
    "[TRACKING_LEFT]",
    "[TRACKING_RIGHT]",
    "[CRANE_UP]",
    "[CRANE_DOWN]",
    "[ORBIT]",
    "[HANDHELD]",
    "[ROLL]",
]

CAMERA_TAG_DESCRIPTIONS = {
    "STATIC_CAMERA":  "the camera MUST remain completely static. ABSOLUTELY NO zoom, pan, orbit, push-in, pull-out, tilt, tracking, or any motion whatsoever. You MUST explicitly state \"the camera remains locked-off, completely static throughout the entire clip\" and you MUST NOT describe any camera movement anywhere in the output.",
    "LOCKED_OFF":     "the camera MUST remain completely static. ABSOLUTELY NO zoom, pan, orbit, push-in, pull-out, tilt, tracking, or any motion whatsoever. You MUST explicitly state \"the camera remains locked-off, completely static throughout the entire clip\" and you MUST NOT describe any camera movement anywhere in the output.",
    "SLOW_ZOOM_IN":   "slow continuous push-in (dolly toward subject). The camera smoothly and continuously moves closer to the subject throughout the clip.",
    "SLOW_ZOOM_OUT":  "slow continuous pull-back (dolly away from subject). The camera smoothly and continuously moves away from the subject throughout the clip.",
    "FAST_ZOOM_IN":   "fast aggressive push-in, dramatic. The camera rapidly moves closer to the subject with energy.",
    "FAST_ZOOM_OUT":  "fast pull-back, reveal context. The camera rapidly moves away from the subject to reveal the wider scene.",
    "PAN_LEFT":       "smooth horizontal pan from right to left. The camera rotates smoothly on its axis, moving the framing from right to left.",
    "PAN_RIGHT":      "smooth horizontal pan from left to right. The camera rotates smoothly on its axis, moving the framing from left to right.",
    "TILT_UP":        "smooth vertical tilt from bottom to top, revealing the subject. The camera rotates upward on its axis.",
    "TILT_DOWN":      "smooth vertical tilt from top to bottom. The camera rotates downward on its axis.",
    "DOLLY_IN":       "physical dolly movement toward the subject (not optical zoom — the camera moves through space, creating parallax).",
    "DOLLY_OUT":      "physical dolly movement away from the subject (not optical zoom — the camera moves through space, creating parallax).",
    "TRACKING_LEFT":  "lateral tracking shot moving left, subject stays in frame. The camera physically moves left while keeping the subject centered.",
    "TRACKING_RIGHT": "lateral tracking shot moving right, subject stays in frame. The camera physically moves right while keeping the subject centered.",
    "CRANE_UP":       "crane/jib movement rising upward, revealing the scene from above. The camera physically rises.",
    "CRANE_DOWN":     "crane/jib movement descending toward the subject. The camera physically descends.",
    "ORBIT":          "smooth 360-degree orbit around the subject. The camera circles completely around the subject.",
    "HANDHELD":       "subtle handheld sway with natural micro-movements. The camera feels held by a person, with gentle bob and sway.",
    "ROLL":           "slow camera roll (rotation around the lens axis). The horizon slowly rotates.",
}

CAMERA_TAG_TOOLTIP = (
    "Camera movement override for video presets (MiniMax H3, WAN, etc). "
    "'None' lets the preset decide. Any other value is injected as a "
    "[TAG] and reinforced at the end of the prompt so Qwen respects it."
)

# ── Style tag dropdown ────────────────────────────────────────────────
STYLE_TAG_OPTIONS = [
    "None",
    "[ANIME]",
    "[PHOTOREALISTIC]",
    "[3DCG]",
    "[CARTOON]",
    "[CLAYMATION]",
    "[WATERCOLOR]",
    "[VINTAGE]",
    "[NOIR]",
    "[CYBERPUNK]",
    "[FANTASY]",
    "[SOFTFOCUS]",
    "[HENTAI]",
]

STYLE_TAG_DESCRIPTIONS = {
    "ANIME":        "2D-animated, cel-shaded, vibrant anime color palette, clean lineart, anime-style lighting",
    "PHOTOREALISTIC": "Live-action, cinematic photorealism, natural skin textures, realistic lighting, shallow depth of field",
    "3DCG":         "3D CG rendered, subsurface scattering, physically based rendering, cinematic 3D animation",
    "CARTOON":      "2D cartoon, bold outlines, flat colors, exaggerated expressions, cartoon-style animation",
    "CLAYMATION":   "Claymation, stop-motion clay texture, handcrafted look, visible fingerprints, studio lighting",
    "WATERCOLOR":   "Watercolor painting style, soft bleeding pigments, paper texture, hand-painted aesthetic",
    "VINTAGE":      "Vintage film, 35mm grain, muted colors, halation, film scratches, analog warmth",
    "NOIR":         "Film noir, high-contrast black and white, harsh shadows, venetian blind light, moody atmosphere",
    "CYBERPUNK":    "Cyberpunk, neon-lit, holographic displays, rainy night, chrome reflections, magenta-cyan palette",
    "FANTASY":      "Fantasy, ethereal lighting, magical particles, painterly atmosphere, mystical glow",
    "SOFTFOCUS":    "Soft focus, dreamy diffusion, bloom, pastel palette, romantic atmosphere",
    "HENTAI":       "2D-animated hentai, cel-shaded, explicit anime style, clean lineart, anime-style lighting",
}

STYLE_TAG_TOOLTIP = (
    "Visual style override for video presets (MiniMax H3, LTX 2.3, WAN, etc). "
    "'None' lets the preset decide. Any other value is injected as a "
    "[TAG] and reinforced at the end of the prompt so Qwen respects it. "
    "For image-reference modes, identity and composition remain anchored to the references."
)

# ── Danbooru input guidance ───────────────────────────────────────────
DANBOORU_INPUT_GUIDANCE = """INPUT TAG SUPPORT:
- The user input may be natural language, Danbooru-style comma-separated tags, or mixed text with TagComplete wildcards already resolved.
- Treat Danbooru tags as a visual blueprint and expand them into fluent cinematic English.
- Preserve every non-conflicting tag: subject count, identity, anatomy, hair, eyes, clothing, pose, expression, action, environment, lighting, framing, and style.
- Resolve conflicting tags logically instead of silently dropping them.
- Anime-typical tags default to 2D animation unless a style directive or reference image establishes another style.
- Text input takes priority for requested action and motion; reference images remain authoritative for visible identity, appearance, and composition unless the user explicitly requests a change."""


def add_danbooru_guidance(prompt: str, preset_name: str) -> str:
    """Append Danbooru tag guidance to LTX and MiniMax R2VA/FL2VA presets."""
    name = preset_name or ""
    if "LTX 2.3" not in name and not ("MiniMax H3" in name and ("R2VA" in name or "FL2VA" in name)):
        return prompt
    if "INPUT TAG SUPPORT:" in prompt or "Danbooru-style tags" in prompt:
        return prompt
    return f"{prompt}\n\n{DANBOORU_INPUT_GUIDANCE}"


def camera_directive_location(preset_name: str, prompt: str = "") -> str:
    """Return where the camera directive should be stated in the output."""
    context = f"{preset_name or ''}\n{prompt or ''}"
    if "LTX 2.3" in context:
        return "State it explicitly in the first sentence of the video description; do not introduce a [Shot 1] label unless the selected preset already requires one."
    if "MiniMax H3" in context:
        return "State it explicitly in the first sentence of [Shot 1]."
    return "State it explicitly in the first sentence of the generated video prompt."


def style_reference_guard(preset_name: str, prompt: str = "") -> str:
    """Return a guard for image-reference modes to preserve identity."""
    context = f"{preset_name or ''}\n{prompt or ''}"
    if any(mode in context for mode in ("I2V", "FL2VA", "R2VA", "L2VA")):
        return " Apply the style consistently while preserving the reference image subjects' identity, visible appearance, composition, and continuity unless the user explicitly requests a change."
    return ""


def inject_camera_tag(prompt: str, preset_name: str, camera_tag: str, custom_prompt: str = "") -> str:
    """Inject camera tag at start and end of prompt for recency bias."""
    camera_tags = list(CAMERA_TAG_DESCRIPTIONS.keys())
    found_cam_tag = None

    # Source 1: dropdown
    if camera_tag and camera_tag.strip() and camera_tag.strip().upper() != "NONE":
        tag_clean = camera_tag.strip().upper().strip("[]")
        if tag_clean in camera_tags:
            found_cam_tag = tag_clean

    # Source 2: manual tag in custom_prompt (only if dropdown is None)
    if not found_cam_tag and custom_prompt and custom_prompt.strip():
        upper = custom_prompt.upper()
        for tag in camera_tags:
            if f"[{tag}]" in upper:
                found_cam_tag = tag
                break

    if not found_cam_tag:
        return prompt

    desc = CAMERA_TAG_DESCRIPTIONS.get(found_cam_tag, "")
    tag_str = f"[{found_cam_tag}]"
    prefix = f"{tag_str}\n\n"
    reminder = (
        f"\n\n═══ FINAL CAMERA DIRECTIVE (HIGHEST PRIORITY) ═══\n"
        f"Camera: {tag_str} — {desc}\n"
        f"You MUST use this camera movement and NO other. "
        f"{camera_directive_location(preset_name, prompt)}\n"
        f"IMPORTANT: the camera tag controls ONLY the camera. "
        f"The subject MUST still have natural, lively action and "
        f"movement throughout the clip — breathing, gestures, "
        f"expression changes, body motion, interaction with the "
        f"environment. Do NOT freeze the subject just because the "
        f"camera is static or performing a specific move."
    )
    return f"{prefix}{prompt}{reminder}"


def inject_style_tag(prompt: str, preset_name: str, style_tag: str, custom_prompt: str = "") -> str:
    """Inject style tag at start and end of prompt for recency bias."""
    style_tags = list(STYLE_TAG_DESCRIPTIONS.keys())
    found_style_tag = None

    # Source 1: dropdown
    if style_tag and style_tag.strip() and style_tag.strip().upper() != "NONE":
        tag_clean = style_tag.strip().upper().strip("[]")
        if tag_clean in style_tags:
            found_style_tag = tag_clean

    # Source 2: manual tag in custom_prompt (only if dropdown is None)
    if not found_style_tag and custom_prompt and custom_prompt.strip():
        upper = custom_prompt.upper()
        for tag in style_tags:
            if f"[{tag}]" in upper:
                found_style_tag = tag
                break

    if not found_style_tag:
        return prompt

    desc = STYLE_TAG_DESCRIPTIONS.get(found_style_tag, "")
    tag_str = f"[{found_style_tag}]"
    guard = style_reference_guard(preset_name, prompt)
    prefix = f"{tag_str}\n\n"
    reminder = (
        f"\n\n═══ FINAL STYLE DIRECTIVE (HIGHEST PRIORITY) ═══\n"
        f"Visual style: {tag_str} — {desc}.{guard}\n"
        f"You MUST use this visual style for the entire clip and NO other."
    )
    return f"{prefix}{prompt}{reminder}"
