"""
LinguisticResources — central, externalized linguistic configuration.

Background: Phase 7 originally embedded English-specific stoplists, regex
patterns, narrative templates, and keyword lists directly into Python source.
That was bad engineering — it locked the codebase to English, made the
linguistic surface invisible to non-developers, and required code changes
for every i18n or domain-tuning need.

This module loads ALL linguistic resources from a single JSON file. The
default ships at `src/core/locales/en.json`. Override at runtime by setting
the env var `LINGUISTIC_CONFIG_PATH` to a different JSON file path.

Schema (see locales/en.json for the canonical reference):

    {
      "locale": str,
      "version": str,
      "entity_detection": { "named_entity_regex": str, "min_entity_length": int },
      "stopwords": { "<consumer_name>": [str, ...] },
      "trajectory_keywords": { "past_state": [...], "transition": [...], ... },
      "definitional_pattern": { "regex_template": str, "min_description_chars": int, "head_chars": int },
      "hybrid_encoder": { "filler_tokens": [...], "filler_phrases": [...], "high_signal_keywords": [...] },
      "paraphrase_rules": { "speaker_prefix_regex": str, "rules": [...] },
      "wake_summary_templates": { "no_idle_known": str, ... }
    }

All consumers should import the module-level helpers (e.g.
`get_stopwords("schema_extractor")`) rather than reaching into the cached
dict directly. That gives us a single place to layer overrides, validation,
and reload semantics.
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional


_DEFAULT_LOCALE_PATH = Path(__file__).parent / "locales" / "en.json"
_ENV_OVERRIDE = "LINGUISTIC_CONFIG_PATH"


_lock = threading.Lock()
_cache: Optional[Dict[str, Any]] = None
_loaded_from: Optional[str] = None


# ─── Public API ────────────────────────────────────────────────────────────


def get_resources() -> Dict[str, Any]:
    """Return the full linguistic resources dict (cached)."""
    global _cache, _loaded_from
    if _cache is not None:
        return _cache
    with _lock:
        if _cache is not None:
            return _cache
        path = _resolve_path()
        try:
            with open(path) as f:
                _cache = json.load(f)
            _loaded_from = str(path)
        except Exception as exc:
            # Hard fail loud — running without resources will silently
            # corrupt downstream behavior. Better to surface immediately.
            raise RuntimeError(
                f"Failed to load linguistic resources from {path}: {exc}"
            ) from exc
    return _cache


def reload() -> Dict[str, Any]:
    """Force a re-read of the resource file (for tests / SIGHUP-style flows)."""
    global _cache, _loaded_from
    with _lock:
        _cache = None
        _loaded_from = None
    return get_resources()


def loaded_from() -> Optional[str]:
    """Path the resources were loaded from. Useful for diagnostics."""
    if _cache is None:
        get_resources()  # populates _loaded_from
    return _loaded_from


def _resolve_path() -> Path:
    override = os.getenv(_ENV_OVERRIDE)
    if override:
        return Path(override)
    return _DEFAULT_LOCALE_PATH


# ─── Typed accessors per consumer ──────────────────────────────────────────


def get_stopwords(consumer: str) -> set:
    """
    Stoplist of generic / filler tokens that consumer-specific code should
    never treat as meaningful entities. Returned as a lowercased set.
    """
    res = get_resources().get("stopwords", {})
    items = res.get(consumer, [])
    return {str(s).lower() for s in items}


def get_entity_detection() -> Dict[str, Any]:
    """Settings for capitalized-entity extraction."""
    res = get_resources().get("entity_detection", {})
    return {
        "named_entity_regex": str(res.get("named_entity_regex", r"\b([A-Z][a-zA-Z'-]{2,}|[A-Z]{2,})\b")),
        "min_entity_length": int(res.get("min_entity_length", 4)),
    }


def get_trajectory_keywords() -> List[Dict[str, Any]]:
    """
    Returns ordered list of dicts: [{"kind": "past_state", "keywords": [...]}, ...].
    Consumers compile the keyword list into a single regex.
    """
    res = get_resources().get("trajectory_keywords", {})
    out = []
    for kind in ("past_state", "transition", "current_state"):
        kws = res.get(kind, [])
        if kws:
            out.append({"kind": kind, "keywords": list(kws)})
    return out


def get_definitional_pattern() -> Dict[str, Any]:
    res = get_resources().get("definitional_pattern", {})
    return {
        "regex_template": str(res.get("regex_template", r"\b{entity}\s+(?:is|was)\s+(?:a|an)\b")),
        "min_description_chars": int(res.get("min_description_chars", 25)),
        "head_chars": int(res.get("head_chars", 80)),
    }


def get_hybrid_encoder_keywords() -> Dict[str, list]:
    res = get_resources().get("hybrid_encoder", {})
    return {
        "filler_tokens": list(res.get("filler_tokens", [])),
        "filler_phrases": list(res.get("filler_phrases", [])),
        "high_signal_keywords": list(res.get("high_signal_keywords", [])),
    }


def get_paraphrase_rules() -> Dict[str, Any]:
    res = get_resources().get("paraphrase_rules", {})
    return {
        "speaker_prefix_regex": str(res.get("speaker_prefix_regex", "")),
        "rules": list(res.get("rules", [])),
    }


def get_wake_summary_templates() -> Dict[str, str]:
    res = get_resources().get("wake_summary_templates", {})
    return {k: v for k, v in res.items() if not k.startswith("_") and isinstance(v, str)}


# ─── Helpers for compiling rules ───────────────────────────────────────────


def compile_paraphrase_rules() -> List[tuple]:
    """
    Returns a list of (compiled_pattern, template_str) tuples ready to use.
    The {SP} placeholder in patterns is substituted with the speaker_prefix_regex.
    """
    cfg = get_paraphrase_rules()
    sp = cfg["speaker_prefix_regex"]
    out: List[tuple] = []
    for rule in cfg["rules"]:
        pattern = rule.get("pattern", "")
        template = rule.get("template", "")
        flags_str = rule.get("flags", "") or ""
        flags = 0
        if "IGNORECASE" in flags_str:
            flags |= re.IGNORECASE
        if "DOTALL" in flags_str:
            flags |= re.DOTALL
        if "MULTILINE" in flags_str:
            flags |= re.MULTILINE
        # Substitute {SP} placeholder
        pattern_str = pattern.replace("{SP}", sp)
        try:
            compiled = re.compile(pattern_str, flags)
        except re.error:
            continue
        out.append((compiled, template))
    return out


def compile_trajectory_patterns() -> List[tuple]:
    """
    Returns list of (compiled_regex, kind) tuples for trajectory detection.
    Each compiled regex matches any of the keywords for that kind.
    """
    out: List[tuple] = []
    for entry in get_trajectory_keywords():
        kws = entry["keywords"]
        if not kws:
            continue
        # Sort by length desc so multi-word phrases match before substrings.
        kws_sorted = sorted({k for k in kws if k}, key=len, reverse=True)
        joined = "|".join(re.escape(k) for k in kws_sorted)
        try:
            pattern = re.compile(rf"\b(?:{joined})\b", re.IGNORECASE)
        except re.error:
            continue
        out.append((pattern, entry["kind"]))
    return out


def compile_named_entity_regex():
    """Returns the compiled regex used to extract capitalized entity tokens."""
    cfg = get_entity_detection()
    return re.compile(cfg["named_entity_regex"])


def compile_definitional_regex(entity: str):
    """Returns a compiled regex matching the definitional pattern for `entity`."""
    cfg = get_definitional_pattern()
    template = cfg["regex_template"]
    pattern_str = template.replace("{entity}", re.escape(entity))
    return re.compile(pattern_str, re.IGNORECASE)
