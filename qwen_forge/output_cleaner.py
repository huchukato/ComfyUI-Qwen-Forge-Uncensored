"""Output cleaning utilities for model-generated prompts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class OutputCleanConfig:
    mode: str = "prompt"
    strip_think: bool = True
    strip_code_fences: bool = True
    strip_role_prefixes: bool = True
    strip_json_wrappers: bool = True
    strip_leading_preamble: bool = True
    strip_planning: bool = True
    keep_first_paragraph_only: bool = False


_ROLE_PREFIX_RE = re.compile(
    r"^\s*(assistant|final|output|response|result|prompt)\s*:\s*",
    re.IGNORECASE,
)
_CODE_FENCE_RE = re.compile(r"^\s*```[\w-]*\s*$", re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(r"<think[^>]*>.*?</think\s*>", flags=re.IGNORECASE | re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think[^>]*>", flags=re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think\s*>", flags=re.IGNORECASE)
_IM_TOKEN_RE = re.compile(
    r"(?i)<\|?im_(start|end)\|?>|<im_(start|end)>|<\|endoftext\|>",
)
_MARKER_RE = re.compile(
    r"(?im)^\s*(final|final answer|answer|output|result|prompt)\s*[:\-]\s*",
)
_PLANNING_RE = re.compile(
    r"(?is)\b("
    r"final\s+plan\b|"
    r"final\s+check\b|"
    r"i\s+(should|need|must|will|want|am\s+going\s+to|have\s+to)\b|"
    r"let's\b|"
    r"wait\b|"
    r"ready\s+to\s+write\b|"
    r"writing\s+the\s+prompt\b|"
    r"so\s+i\s+need\s+to\b|"
    r"i\s+should\s+focus\s+on\b"
    r")"
)


def clean_model_output(text: str, config: OutputCleanConfig | None = None) -> str:
    """Clean raw model output into a usable prompt string."""
    if not text:
        return ""

    cfg = config or OutputCleanConfig()
    cleaned = text.strip()

    cleaned = _IM_TOKEN_RE.sub("", cleaned).strip()

    if cfg.strip_think:
        cleaned = _THINK_BLOCK_RE.sub("", cleaned)
        cleaned = _THINK_CLOSE_RE.sub("", cleaned)
        if _THINK_OPEN_RE.search(cleaned):
            cleaned = _THINK_OPEN_RE.sub("", cleaned)
            parts = re.split(r"\n\s*\n", cleaned, maxsplit=1)
            if len(parts) == 2:
                cleaned = parts[1]
        cleaned = cleaned.strip()

    cleaned = _IM_TOKEN_RE.sub("", cleaned).strip()

    if cfg.strip_code_fences and "```" in cleaned:
        lines = [ln for ln in cleaned.splitlines() if not _CODE_FENCE_RE.match(ln)]
        cleaned = "\n".join(lines).strip()

    if cfg.strip_json_wrappers:
        maybe = _extract_from_json(cleaned, mode=cfg.mode)
        if maybe is not None:
            cleaned = maybe.strip()

    if cfg.strip_leading_preamble:
        cleaned = _drop_preamble(cleaned).strip()

    if cfg.strip_planning and cfg.mode == "prompt":
        without_planning = _strip_planning_paragraphs(cleaned)
        if without_planning:
            cleaned = without_planning

    if cfg.strip_role_prefixes:
        lines = cleaned.splitlines()
        if lines:
            lines[0] = _ROLE_PREFIX_RE.sub("", lines[0])
        cleaned = "\n".join(lines).strip()

    cleaned = _MARKER_RE.sub("", cleaned).strip()

    if cfg.keep_first_paragraph_only:
        parts = re.split(r"\n\s*\n", cleaned, maxsplit=1)
        cleaned = parts[0].strip()

    return cleaned


def _extract_from_json(text: str, mode: str) -> str | None:
    candidate = text.strip()
    if not candidate:
        return None
    if not (candidate.startswith("{") and candidate.endswith("}")):
        return None
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    keys = ["prompt", "response", "output", "result", "text", "answer"]
    if mode == "tags":
        keys = ["tags", "prompt", "text"]

    for key in keys:
        if key in data and isinstance(data[key], str):
            return data[key]

    # If it's a single-key dict, return its value
    if len(data) == 1:
        value = next(iter(data.values()))
        if isinstance(value, str):
            return value
    return None


def _drop_preamble(text: str) -> str:
    preamble_patterns = [
        r"(?is)^\s*(okay[,!]?|sure[,!]?|here\s+(is|are)\s+(the|a)|certainly[,!]?)",
        r"(?is)^\s*I['']?ll\s+(write|create|generate)",
        r"(?is)^\s*Here\s+is\s+(?:a\s+)?(?:polished|enhanced|rewritten|final)\s+(?:prompt|description)",
    ]
    for pattern in preamble_patterns:
        text = re.sub(pattern, "", text, count=1).strip()
    return text


def _strip_planning_paragraphs(text: str) -> str | None:
    paragraphs = re.split(r"\n\s*\n", text)
    kept = [p for p in paragraphs if not _PLANNING_RE.search(p.strip())]
    if not kept:
        return None
    return "\n\n".join(kept).strip()


def prompt_output_guard() -> str:
    """Return a short instruction to append to prompts asking for clean output."""
    return (
        "Output rules that override all style preferences: return only the final usable prompt text. "
        "Do not include analysis, planning, bullet points, headings, markdown, JSON, explanations, alternatives, self-corrections, or notes. "
        "Do not write phrases such as Final Plan, Final Check, Wait, Okay, First, Next, Then, I will, or I need. "
        "Start directly with the required prompt text."
    )
