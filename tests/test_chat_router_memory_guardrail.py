from src.api.chat_router import (
    _is_explicit_correction,
    _profile_fact_signal_count,
    _should_force_profile_ingest,
)


def test_profile_signal_count_detects_multi_fact_intro():
    text = "Hi, I'm Alex. I'm a backend engineer in Lisbon, and I have a peanut allergy."
    assert _profile_fact_signal_count(text) >= 3


def test_force_ingest_when_agent_stores_partial_profile():
    user = "Hi, I'm Alex. I'm a backend engineer in Lisbon, and I have a peanut allergy."
    tool_payloads = ["I have a peanut allergy."]
    assert _should_force_profile_ingest(user, tool_payloads) is True


def test_skip_force_ingest_when_full_message_already_stored():
    user = "Hi, I'm Alex. I'm a backend engineer in Lisbon, and I have a peanut allergy."
    tool_payloads = [user]
    assert _should_force_profile_ingest(user, tool_payloads) is False


def test_skip_force_ingest_for_non_profile_chitchat():
    assert _should_force_profile_ingest("lol thanks", ["lol thanks"]) is False


def test_detects_explicit_correction_language():
    assert _is_explicit_correction("Sorry it was not peanut but milk lol my bad") is True
    assert _is_explicit_correction("I like milk") is False
