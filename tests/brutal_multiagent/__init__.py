"""
Brutal multi-agent harness — three LangChain agents (Researcher, Coder,
Reviewer) each backed by their own SCM memory namespace, all driven by
DeepSeek v4 as the LLM.

Tests cover:
    Tier 1 — per-agent memory specialty (each agent learns its niche)
    Tier 2 — shared user-memory namespace for cross-agent handoff
    Tier 3 — contradiction across agents (each keeps its own view)
    Tier 4 — per-agent autonomous wake summary
    Tier 5 — collaborative task with per-agent retrieval
    Tier 6 — strict isolation (Agent A's facts don't leak to Agent B)
    Tier 7 — DeepSeek extraction depth on ambiguous statements
"""
