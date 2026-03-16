"""Shared utility helpers."""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

from backend.config import settings


def slugify(name: str) -> str:
    """Normalise a concept name to a stable slug id (Unicode-aware)."""
    name = unicodedata.normalize("NFD", name.lower())
    name = re.sub(r"[^a-z0-9\s-]", "", name)
    return re.sub(r"\s+", "-", name.strip())


def safe_temperature(requested: float) -> float:
    """Return a model-safe temperature.

    Gemini 3 models (gemini-3-*) require temperature=1.0 — values below 1.0
    cause degraded reasoning and silent failures (empty responses).
    """
    if "gemini-3" in settings.llm_model.lower():
        return 1.0
    return requested


def parse_llm_json_list(raw: str, primary_key: str) -> list[Any]:
    """Parse an LLM JSON response that may be a bare list or a dict wrapping one.

    The LLM may return either ``[...]`` or ``{"primary_key": [...], ...}``.
    Falls back to the first value of the dict if *primary_key* is absent.
    Returns an empty list on any parse error.
    """
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        if primary_key in parsed:
            result = parsed[primary_key]
        else:
            result = next(iter(parsed.values()), [])
        return result if isinstance(result, list) else []
    return []
