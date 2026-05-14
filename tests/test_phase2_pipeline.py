"""Phase 2 integration tests for ChatEngine pipeline."""

import uuid

import src.chat.engine as chat_engine_module
from src.chat.engine import ChatEngine
from src.core.models import Concept, ConceptType


class _DummyLLM:
    def _chat(self, prompt: str, num_predict: int = 512) -> str:
        return "Acknowledged."


class _StubEncoder:
    def extract(self, text: str):
        return [
            Concept(type=ConceptType.PERSON, description="Alice"),
            Concept(type=ConceptType.LOCATION, description="Seattle"),
            Concept(type=ConceptType.FACT, description="Moved for a new job"),
        ]

    def _get_embedding(self, text: str):
        seed = sum(ord(ch) for ch in text) % 97
        value = (seed + 1) / 100.0
        return [value] * 384


class TestPhase2Pipeline:
    def test_event_and_association_binding_when_hme_enabled(self, monkeypatch):
        monkeypatch.setattr(chat_engine_module, "HME_ENABLED", True)

        engine = ChatEngine(
            llm=_DummyLLM(),
            encoder=_StubEncoder(),
            enable_auto_sleep=False,
            session_id=f"phase2_{uuid.uuid4().hex}",
        )

        initial_relations = engine.long_term_memory.get_stats()["total_relations"]

        engine._extract_and_store(
            "I moved to Seattle today because I started a new job.",
            source="user",
        )

        latest_episode = engine.working_memory.get_recent(1)[0]

        assert latest_episode.who == "user"
        assert latest_episode.where_ == "Seattle"
        assert latest_episode.when_ is not None
        assert latest_episode.why is not None
        assert "association_stats" in latest_episode.context
        assert latest_episode.context["association_stats"]["edges_created"] > 0

        final_relations = engine.long_term_memory.get_stats()["total_relations"]
        assert final_relations > initial_relations

    def test_duplicate_events_do_not_rebind(self, monkeypatch):
        monkeypatch.setattr(chat_engine_module, "HME_ENABLED", True)

        engine = ChatEngine(
            llm=_DummyLLM(),
            encoder=_StubEncoder(),
            enable_auto_sleep=False,
            session_id=f"phase2_{uuid.uuid4().hex}",
        )

        engine._extract_and_store(
            "I moved to Seattle today because I started a new job.",
            source="user",
        )
        relations_after_first = engine.long_term_memory.get_stats()["total_relations"]

        engine._extract_and_store(
            "I moved to Seattle today because I started a new job.",
            source="user",
        )
        relations_after_second = engine.long_term_memory.get_stats()["total_relations"]

        latest_episode = engine.working_memory.get_recent(1)[0]
        assert latest_episode.context.get("is_duplicate_event") is True
        assert relations_after_second == relations_after_first
