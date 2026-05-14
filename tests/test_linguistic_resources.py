"""
Tests for the LinguisticResources loader + the de-hardcoding refactor.

Contracts under test:
  - Default English bundle loads cleanly.
  - Override via LINGUISTIC_CONFIG_PATH env var works.
  - Each consumer (schema_extractor, curiosity, paraphrase, hybrid_encoder,
    wake_summary) actually reads from the loader at instance time.
  - Reload picks up changes to the resource file (test isolation).
  - Missing/broken resource file fails LOUDLY (no silent corruption).
  - LLMSource for curiosity: stub LLM returning a brief is integrated correctly.
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import linguistic_resources as lr


# ─── Loader tests ───────────────────────────────────────────────────────────


def test_default_english_bundle_loads():
    lr.reload()
    res = lr.get_resources()
    assert res["locale"] == "en"
    assert "stopwords" in res
    assert "trajectory_keywords" in res
    assert "wake_summary_templates" in res
    assert "paraphrase_rules" in res


def test_loaded_from_reports_path():
    lr.reload()
    path = lr.loaded_from()
    assert path is not None
    assert "en.json" in path


def test_get_stopwords_per_consumer():
    lr.reload()
    sw_schema = lr.get_stopwords("schema_extractor")
    sw_curi = lr.get_stopwords("curiosity_engine")
    assert isinstance(sw_schema, set)
    assert isinstance(sw_curi, set)
    assert "today" in sw_schema
    assert "talking" in sw_curi
    # Empty for unknown consumer (graceful, not crash)
    assert lr.get_stopwords("nobody_uses_this_name") == set()


def test_compile_named_entity_regex():
    lr.reload()
    rx = lr.compile_named_entity_regex()
    matches = [m.group(1) for m in rx.finditer("I use Datadog at TechCorp.")]
    assert "Datadog" in matches
    assert "TechCorp" in matches


def test_compile_trajectory_patterns():
    lr.reload()
    pats = lr.compile_trajectory_patterns()
    assert len(pats) >= 3
    kinds = {kind for _, kind in pats}
    assert "past_state" in kinds
    assert "transition" in kinds


def test_compile_paraphrase_rules_returns_callable_pairs():
    lr.reload()
    rules = lr.compile_paraphrase_rules()
    assert len(rules) > 0
    # Each entry is (compiled_regex, template_string)
    pat, tmpl = rules[0]
    assert hasattr(pat, "match")
    assert isinstance(tmpl, str)


def test_compile_definitional_regex_substitutes_entity():
    lr.reload()
    rx = lr.compile_definitional_regex("Datadog")
    assert rx.search("Datadog is a monitoring platform.")
    assert not rx.search("Caroline mentioned Datadog yesterday.")


def test_wake_summary_templates_exposed():
    lr.reload()
    T = lr.get_wake_summary_templates()
    assert "closer" in T
    assert "patterns_intro_singular" in T
    assert "curiosity_insight_template" in T
    # Comments (keys starting with _) are filtered out
    assert all(not k.startswith("_") for k in T.keys())


# ─── Override behavior ──────────────────────────────────────────────────────


def test_override_via_env_var(tmp_path, monkeypatch):
    """A custom JSON file pointed to by LINGUISTIC_CONFIG_PATH should win."""
    custom = tmp_path / "custom.json"
    custom.write_text(json.dumps({
        "locale": "custom-test",
        "version": "1.0",
        "entity_detection": {"named_entity_regex": r"\b([A-Z][a-z]{4,})\b", "min_entity_length": 5},
        "stopwords": {"schema_extractor": ["customstopword"], "curiosity_engine": []},
        "trajectory_keywords": {"past_state": [], "transition": ["pivoted"], "current_state": []},
        "definitional_pattern": {"regex_template": r"\b{entity}\b", "min_description_chars": 0, "head_chars": 200},
        "hybrid_encoder": {"filler_tokens": ["custom"], "filler_phrases": [], "high_signal_keywords": []},
        "paraphrase_rules": {"speaker_prefix_regex": "", "rules": []},
        "wake_summary_templates": {"closer": "Custom closer."},
    }))
    monkeypatch.setenv("LINGUISTIC_CONFIG_PATH", str(custom))
    lr.reload()
    assert lr.get_resources()["locale"] == "custom-test"
    assert "customstopword" in lr.get_stopwords("schema_extractor")
    # Cleanup
    monkeypatch.delenv("LINGUISTIC_CONFIG_PATH", raising=False)
    lr.reload()


def test_broken_resource_fails_loudly(tmp_path, monkeypatch):
    bad = tmp_path / "broken.json"
    bad.write_text("not valid json{")
    monkeypatch.setenv("LINGUISTIC_CONFIG_PATH", str(bad))
    # Clear the cache without triggering load so the next read fails loudly.
    with lr._lock:
        lr._cache = None
        lr._loaded_from = None
    with pytest.raises(RuntimeError):
        lr.get_resources()
    monkeypatch.delenv("LINGUISTIC_CONFIG_PATH", raising=False)
    lr.reload()


# ─── Downstream consumers actually read from the loader ────────────────────


def test_schema_extractor_uses_loader_stopwords():
    """If we patch the resources, schema extractor instances should see it."""
    from src.sleep.schema_extractor import SchemaExtractor, SchemaExtractorConfig
    lr.reload()
    ext = SchemaExtractor(SchemaExtractorConfig(min_repetitions=2))
    # The instance should have the locale stoplist loaded
    assert "today" in ext._effective_stoplist


def test_curiosity_uses_loader_stopwords():
    from src.lifecycle.curiosity import CuriosityEngine, CuriosityConfig
    lr.reload()
    eng = CuriosityEngine(sources=[], config=CuriosityConfig(enabled=True))
    assert "talking" in eng._effective_stoplist


def test_hybrid_encoder_uses_loader_keywords():
    from src.core.encoder import HybridEncoder
    lr.reload()
    h = HybridEncoder(llm=None)
    assert "lol" in h._filler_tokens
    # Salience proxy still works the same on a known filler input
    assert h.compute_salience_proxy("lol") == 0.0


def test_paraphrase_uses_loader_rules():
    from src.sleep.paraphrase import HeuristicParaphraser
    lr.reload()
    hp = HeuristicParaphraser()
    out = hp.paraphrase("Caroline says: I work at GreenLeaf Cafe.")
    assert out is not None
    assert "works" in out.lower()


def test_wake_summary_uses_loader_templates():
    from src.lifecycle.wake_summary import WakeSummaryBuilder, WakeSummary
    from src.core.time_utils import utc_now
    lr.reload()
    # Stub engine
    class _LTM:
        def get_all_concepts(self, include_suppressed=False): return []
    class _WM:
        def get_all(self): return []
    class _Engine:
        session_id = "x"
        _sleep_history = []
        long_term_memory = _LTM()
        working_memory = _WM()
        cross_session_pool = None
    summary = WakeSummaryBuilder(_Engine()).build()
    # Closer text comes from the loader (default = "I'm ready when you are.")
    assert summary.narrative.endswith(lr.get_wake_summary_templates()["closer"])


# ─── LLMSource ──────────────────────────────────────────────────────────────


class _StubLLM:
    def __init__(self, response: str = "Datadog is a cloud monitoring platform."):
        self.response = response
        self.calls: List[str] = []
    def _chat(self, prompt: str, num_predict: int = 64) -> str:
        self.calls.append(prompt)
        return self.response


def test_llm_source_returns_llm_brief():
    from src.lifecycle.curiosity import LLMSource
    src = LLMSource(llm=_StubLLM("Datadog is a cloud monitoring and observability platform."))
    out = src.lookup("Datadog")
    assert out is not None
    assert "monitoring" in out.lower()


def test_llm_source_unknown_response_returns_none():
    from src.lifecycle.curiosity import LLMSource
    src = LLMSource(llm=_StubLLM("UNKNOWN"))
    assert src.lookup("Datadog") is None


def test_llm_source_short_response_returns_none():
    from src.lifecycle.curiosity import LLMSource
    src = LLMSource(llm=_StubLLM("ok"))
    assert src.lookup("Datadog") is None


def test_llm_source_falls_through_on_exception():
    from src.lifecycle.curiosity import LLMSource

    class BoomLLM:
        def _chat(self, *a, **kw): raise RuntimeError("network down")

    src = LLMSource(llm=BoomLLM())
    # Returns None, doesn't raise
    assert src.lookup("Datadog") is None


def test_llm_source_is_unavailable_without_llm():
    from src.lifecycle.curiosity import LLMSource
    src = LLMSource(llm=None)
    assert not src.is_available()
    assert src.lookup("anything") is None


def test_llm_source_used_inside_curiosity_engine():
    """End-to-end: gap detection → LLMSource fills it."""
    from src.lifecycle.curiosity import (
        CuriosityConfig, CuriosityEngine, LLMSource,
    )
    import uuid
    from src.core.models import Episode, ImportanceVector

    def _ep(text: str):
        return Episode(
            id=str(uuid.uuid4()),
            timestamp=datetime(2026, 5, 1),
            concept_ids=[],
            raw_content=text,
            context={"session_id": "s1"},
            importance=ImportanceVector(),
            source="user",
        )

    eng = CuriosityEngine(
        sources=[LLMSource(llm=_StubLLM("Snowflake is a cloud data warehouse."))],
        config=CuriosityConfig(enabled=True, min_occurrences=2),
    )
    eps = [_ep("Snowflake is mandated."), _ep("Snowflake queries are expensive."), _ep("Using Snowflake.")]
    filled = eng.run(eps, [])
    assert len(filled) == 1
    assert filled[0].source_name == "llm"
    assert filled[0].entity == "Snowflake"
    assert "warehouse" in filled[0].brief.lower()
