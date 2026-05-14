"""Brutal end-to-end test of SCM as a memory layer behind a real
LangChain agent driven by a real LLM (Ollama llama3.2).

Tests cover:
    Tier 1 — multi-day recall
    Tier 2 — contradiction handling
    Tier 3 — idle-fired wake summary surfacing
    Tier 4 — cross-session synthesis
    Tier 5 — adversarial (all-noise, contradiction storm)
    Tier 6 — multi-user isolation
    Tier 7 — failure mode (SCM endpoint dies mid-run)
"""
