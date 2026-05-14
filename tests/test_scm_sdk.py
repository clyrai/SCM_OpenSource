"""Smoke tests for the SCM Python SDK wrapper."""

from scm import SCMEngine, list_profiles


def test_sdk_profiles_exposed():
    profiles = list_profiles()
    assert "chatbot" in profiles
    assert "agent" in profiles
    assert "research" in profiles


def test_sdk_sandbox_export_import_roundtrip():
    engine = SCMEngine(session_id="sdk_test", profile="chatbot", sandbox=True)
    baseline = engine.memory_report()["long_term_memory"]["total_concepts"]

    _, _ = engine.message("My name is Alice.")
    _, _ = engine.message("I live in Seattle.")

    exported = engine.export_memory()
    assert exported["counts"]["concepts"] > 0

    engine.reset(clear_persistence=False)
    post_reset = engine.memory_report()
    assert post_reset["long_term_memory"]["total_concepts"] == baseline

    stats = engine.import_memory(exported, replace_existing=True)
    assert stats["concepts_imported"] == exported["counts"]["concepts"]
