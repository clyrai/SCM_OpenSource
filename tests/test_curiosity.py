"""
Tests for Phase 7 CuriosityEngine and its sources.

Contracts:
  - Disabled by default → returns [].
  - StaticDictionarySource: lookups are case-insensitive.
  - LocalDocsSource: only reads files in the configured folder; safe when
    the folder doesn't exist.
  - Gap detection: skips entities that already have an LTM concept covering them.
  - Gap detection: respects min_occurrences threshold.
  - Gap detection: stoplisted entities never become candidates.
  - max_gaps_per_cycle is honored.
  - max_brief_chars is honored.
  - Filled gaps emit Concepts with `_curiosity=True`, source name, audit metadata.
  - Engine never crashes on broken sources or missing data.
  - End-to-end: deep-sleep with curiosity emits FACT-typed concepts that land
    in updated_concepts.
  - Wake summary M4 surfaces curiosity-filled concepts as `learned` insights.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.models import (
    Concept,
    ConceptType,
    Episode,
    ImportanceVector,
    MemoryState,
)
from src.lifecycle.curiosity import (
    CuriosityConfig,
    CuriosityEngine,
    CuriositySource,
    KnowledgeGap,
    LocalDocsSource,
    StaticDictionarySource,
)


# ─── Helpers ────────────────────────────────────────────────────────────────


def _ep(text: str, ts: Optional[datetime] = None, session: str = "s1") -> Episode:
    return Episode(
        id=str(uuid.uuid4()),
        timestamp=ts or datetime(2026, 5, 1, 10),
        concept_ids=[],
        raw_content=text,
        context={"session_id": session, "_origin_session": session},
        importance=ImportanceVector(),
        source="user",
    )


def _fact(desc: str) -> Concept:
    return Concept(
        type=ConceptType.FACT,
        description=desc,
        importance=ImportanceVector(),
        state=MemoryState.ACTIVE,
    )


# ─── StaticDictionarySource tests ──────────────────────────────────────────


def test_static_dict_basic_lookup():
    src = StaticDictionarySource({
        "Datadog": "Datadog is an observability platform for cloud-scale applications.",
        "GreenLeaf": "GreenLeaf Cafe is a local coffee shop.",
    })
    assert "observability" in src.lookup("Datadog").lower()
    # Case-insensitive
    assert "observability" in src.lookup("datadog").lower()


def test_static_dict_misses_return_none():
    src = StaticDictionarySource({"Datadog": "..."})
    assert src.lookup("NeverHeardOf") is None
    assert src.lookup("") is None


def test_static_dict_from_json(tmp_path):
    p = tmp_path / "glossary.json"
    p.write_text(json.dumps({"Datadog": "Observability."}))
    src = StaticDictionarySource.from_json(p)
    assert src.lookup("Datadog") == "Observability."


def test_static_dict_from_json_invalid_returns_empty(tmp_path):
    p = tmp_path / "broken.json"
    p.write_text("not valid json{")
    src = StaticDictionarySource.from_json(p)
    assert src.lookup("anything") is None


# ─── LocalDocsSource tests ─────────────────────────────────────────────────


def test_local_docs_finds_paragraph(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "datadog.md").write_text(
        "# Tools\n\nDatadog is a monitoring and observability platform.\n\n"
        "I use it for cloud monitoring.\n"
    )
    src = LocalDocsSource(folder=notes)
    assert src.is_available()
    out = src.lookup("Datadog")
    assert out is not None
    assert "monitoring" in out.lower()


def test_local_docs_missing_folder_safe():
    src = LocalDocsSource(folder=Path("/nonexistent/path/we/hope"))
    assert not src.is_available()
    assert src.lookup("anything") is None


def test_local_docs_only_reads_supported_suffixes(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "datadog.exe").write_text("Datadog is a binary.")  # ignored
    (notes / "real.md").write_text("Datadog is a real platform.")
    src = LocalDocsSource(folder=notes)
    out = src.lookup("Datadog")
    assert out is not None
    assert "real" in out.lower()


# ─── Gap detection tests ───────────────────────────────────────────────────


def test_disabled_returns_empty():
    eng = CuriosityEngine(
        sources=[StaticDictionarySource({"X": "Y"})],
        config=CuriosityConfig(enabled=False),
    )
    eps = [_ep("X is mentioned a lot."), _ep("X again.")]
    assert eng.run(eps, []) == []


def test_no_sources_returns_empty():
    eng = CuriosityEngine(
        sources=[],
        config=CuriosityConfig(enabled=True),
    )
    eps = [_ep("Datadog mentioned.")] * 3
    assert eng.run(eps, []) == []


def test_gap_detection_respects_min_occurrences():
    eng = CuriosityEngine(
        sources=[StaticDictionarySource({"Datadog": "Observability platform."})],
        config=CuriosityConfig(enabled=True, min_occurrences=3),
    )
    # Only 2 mentions → not a gap
    eps = [
        _ep("I use Datadog for monitoring."),
        _ep("Datadog dashboards are slow."),
    ]
    assert eng.run(eps, []) == []
    # 3 mentions → it IS a gap
    eps.append(_ep("Datadog stack down again."))
    filled = eng.run(eps, [])
    assert len(filled) == 1
    assert filled[0].entity == "Datadog"


def test_gap_detection_skips_entities_already_in_ltm():
    """Entities described by an existing concept should NOT be candidates."""
    eng = CuriosityEngine(
        sources=[StaticDictionarySource({"GreenLeaf": "A cafe."})],
        config=CuriosityConfig(enabled=True, min_occurrences=2),
    )
    eps = [
        _ep("GreenLeaf was crowded today."),
        _ep("Met someone at GreenLeaf."),
        _ep("GreenLeaf playlist was good."),
    ]
    existing_concepts = [_fact("GreenLeaf is a coffee shop downtown.")]
    filled = eng.run(eps, existing_concepts)
    assert filled == []


def test_stoplist_filters_generic_tokens():
    eng = CuriosityEngine(
        sources=[StaticDictionarySource({
            "Today": "irrelevant placeholder content for stoplist test",
            "Datadog": "Real observability platform.",
        })],
        config=CuriosityConfig(enabled=True, min_occurrences=2),
    )
    eps = [_ep("Today was busy. I used Datadog.")] * 3
    filled = eng.run(eps, [])
    entities = {fg.entity for fg in filled}
    assert "Today" not in entities
    # Datadog gets filled (it's in dict)
    assert "Datadog" in entities


def test_max_gaps_per_cycle_honored():
    src = StaticDictionarySource({
        "Datadog": "Observability platform.",
        "Snowflake": "Cloud data warehouse.",
        "Kubernetes": "Container orchestrator.",
        "Mongo": "Document database.",
        "Redis": "In-memory key-value store.",
    })
    eng = CuriosityEngine(
        sources=[src],
        config=CuriosityConfig(enabled=True, min_occurrences=2, max_gaps_per_cycle=2),
    )
    eps = []
    for ent in ["Datadog", "Snowflake", "Kubernetes", "Mongo", "Redis"]:
        for i in range(3):
            eps.append(_ep(f"Talking about {ent} again."))
    filled = eng.run(eps, [])
    assert len(filled) == 2


def test_max_brief_chars_truncates():
    long_brief = "X" * 1000
    eng = CuriosityEngine(
        sources=[StaticDictionarySource({"Datadog": long_brief})],
        config=CuriosityConfig(enabled=True, min_occurrences=2, max_brief_chars=50),
    )
    eps = [_ep("Datadog is here.")] * 3
    filled = eng.run(eps, [])
    assert len(filled) == 1
    assert len(filled[0].brief) == 50


def test_filled_gap_concept_has_audit_metadata():
    eng = CuriosityEngine(
        sources=[StaticDictionarySource({"Datadog": "An observability platform."})],
        config=CuriosityConfig(enabled=True, min_occurrences=2),
    )
    eps = [
        _ep("Using Datadog now.", session="day1"),
        _ep("Datadog dashboards.", session="day1"),
        _ep("Datadog alerts.", session="day2"),
    ]
    filled = eng.run(eps, [])
    assert len(filled) == 1
    c = filled[0].concept
    assert c.context_tags.get("_curiosity") is True
    assert c.context_tags.get("curiosity_entity") == "Datadog"
    assert c.context_tags.get("curiosity_source") == "static_dictionary"
    assert c.context_tags.get("occurrence_count") == 3
    assert "day1" in c.context_tags.get("source_sessions", [])
    assert "day2" in c.context_tags.get("source_sessions", [])


def test_source_priority_first_hit_wins():
    """Two sources, both have Datadog. First in list should win."""
    s1 = StaticDictionarySource({"Datadog": "Source ONE answer."})
    s2 = StaticDictionarySource({"Datadog": "Source TWO answer."})
    eng = CuriosityEngine(
        sources=[s1, s2],
        config=CuriosityConfig(enabled=True, min_occurrences=2),
    )
    eps = [_ep("Datadog!")] * 3
    filled = eng.run(eps, [])
    assert "ONE" in filled[0].brief


def test_source_failure_falls_through():
    """If first source raises, the engine should try the next one."""
    class BrokenSource(CuriositySource):
        name = "broken"
        def lookup(self, entity):
            raise RuntimeError("boom")

    s2 = StaticDictionarySource({"Datadog": "Survived."})
    eng = CuriosityEngine(
        sources=[BrokenSource(), s2],
        config=CuriosityConfig(enabled=True, min_occurrences=2),
    )
    eps = [_ep("Datadog mentioned.")] * 3
    filled = eng.run(eps, [])
    assert len(filled) == 1
    assert filled[0].source_name == "static_dictionary"


def test_engine_handles_empty_episodes_gracefully():
    eng = CuriosityEngine(
        sources=[StaticDictionarySource({"X": "Y"})],
        config=CuriosityConfig(enabled=True),
    )
    assert eng.run([], []) == []


def test_stats_populated_after_run():
    eng = CuriosityEngine(
        sources=[StaticDictionarySource({"Datadog": "An observability platform."})],
        config=CuriosityConfig(enabled=True, min_occurrences=2),
    )
    eps = [_ep("Datadog!")] * 3
    eng.run(eps, [])
    assert eng.last_stats["enabled"] is True
    assert "Datadog" in eng.last_stats["filled_entities"]
    assert eng.last_stats["gaps_filled"] == 1


# ─── Integration: deep sleep with curiosity ────────────────────────────────


def test_deep_sleep_emits_curiosity_concepts():
    from src.sleep.deep_sleep import DeepSleep

    eng = CuriosityEngine(
        sources=[StaticDictionarySource({
            "Datadog": "Datadog is a cloud monitoring and observability platform.",
        })],
        config=CuriosityConfig(enabled=True, min_occurrences=2),
    )
    deep = DeepSleep(
        enable_synthesis=False,
        enable_schema_extraction=False,
        enable_curiosity=True,
        curiosity_engine=eng,
    )
    eps = [
        _ep("Caroline says: I use Datadog at work."),
        _ep("Caroline says: Datadog dashboards help me debug."),
        _ep("Caroline says: Datadog alerts woke me up last night."),
    ]
    concepts = [_fact("Caroline mentioned something.")]
    updated, _, stats = deep.run(concepts=concepts, relations=[], episodes=eps)
    # Curiosity concept appears in stats and in updated_concepts
    assert len(stats.get("curiosity_filled", [])) == 1
    assert stats["curiosity_filled"][0]["entity"] == "Datadog"
    cur_concepts = [
        c for c in updated
        if isinstance(c.context_tags, dict) and c.context_tags.get("_curiosity")
    ]
    assert len(cur_concepts) == 1


# ─── Wake summary surfaces curiosity insights ─────────────────────────────


def test_wake_summary_surfaces_curiosity_as_learned_insights():
    from src.lifecycle.wake_summary import WakeSummaryBuilder
    from src.core.time_utils import utc_now

    # A FakeEngine carrying a curiosity-filled concept
    class FakeWM:
        def get_all(self): return []
    class FakeLTM:
        def __init__(self, cs): self._cs = cs
        def get_all_concepts(self, include_suppressed=False): return list(self._cs)
    class FakeEngine:
        session_id = "u1"
        _sleep_history = []
        long_term_memory = None
        working_memory = FakeWM()
        cross_session_pool = None

    cur_concept = Concept(
        type=ConceptType.FACT,
        description="Datadog is a cloud monitoring platform.",
        importance=ImportanceVector(),
        state=MemoryState.ACTIVE,
    )
    cur_concept.created_at = utc_now()
    cur_concept.context_tags = {
        "_curiosity": True,
        "curiosity_entity": "Datadog",
        "curiosity_source": "static_dictionary",
    }

    eng = FakeEngine()
    eng.long_term_memory = FakeLTM([cur_concept])
    summary = WakeSummaryBuilder(eng).build()
    assert len(summary.insights) == 1
    assert summary.insights[0].insight_type == "learned"
    assert "Datadog" in summary.insights[0].text
    assert "static_dictionary" in summary.insights[0].text
    # Narrative should include the entity
    assert "Datadog" in summary.narrative
