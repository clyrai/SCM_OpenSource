"""
Three specialist LangChain agents, each with its own SCM memory namespace.

The product pattern this exercises:
  - Each agent has a distinct "user_id" in SCM (its own memory pool).
  - A shared user_id ("shared_user") models facts about the actual user
    that all agents can read.
  - Each agent calls SCM /v1/memories/search before responding, then
    /v1/memories after.
  - The LLM is DeepSeek v4 Flash via OpenAI-compatible API.

This is the same pattern Mem0 / Letta / LangGraph multi-agent setups use:
agents share a "user" memory namespace + each maintain a private one.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.integrations.langchain_adapter import SCMClient


SYSTEM_PROMPTS = {
    "researcher": (
        "You are the Researcher, a specialist in product research and user "
        "preferences. You have a private SCM memory of past research findings "
        "AND access to a shared memory of facts about the user. Use both. "
        "Keep responses to 2-3 short sentences. Cite specific facts when you "
        "have them. Do not invent facts not in your memory."
    ),
    "coder": (
        "You are the Coder, a specialist in implementation. You have a "
        "private SCM memory of past coding choices AND access to a shared "
        "memory of facts about the user. Use both. Keep responses to 2-3 "
        "short sentences. Cite specific facts when you have them. Do not "
        "invent facts not in your memory."
    ),
    "reviewer": (
        "You are the Reviewer, a specialist in code quality and standards. "
        "You have a private SCM memory of past review decisions AND access "
        "to a shared memory of facts about the user. Use both. Keep "
        "responses to 2-3 short sentences. Cite specific facts when you "
        "have them. Do not invent facts not in your memory."
    ),
}


@dataclass
class AgentTurn:
    role: str
    user: str
    response: str
    private_retrieved: List[str] = field(default_factory=list)
    shared_retrieved: List[str] = field(default_factory=list)
    private_context: str = ""
    shared_context: str = ""
    wake_summary_pending: Optional[dict] = None
    error: Optional[str] = None


def _make_deepseek_llm(model: str, temperature: float = 0.0) -> ChatOpenAI:
    """Construct a LangChain ChatOpenAI configured for DeepSeek v4."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        timeout=60,
        max_retries=2,
    )


class SpecialistAgent:
    """One of {Researcher, Coder, Reviewer}.

    Holds:
      - a private SCMClient (its own user_id namespace)
      - a shared SCMClient pointing to the per-user memory ("shared_user")
      - a LangChain ChatOpenAI bound to DeepSeek v4

    The agent's chat() call:
      1. searches BOTH private and shared SCM
      2. injects both into the system prompt
      3. generates via DeepSeek v4
      4. writes the user's input into BOTH private AND shared memory
         (so the next agent sees user-stated facts, but each agent's
         responses stay in its own private memory)
    """

    def __init__(
        self,
        role: str,
        scm_url: str,
        shared_user_id: str,
        model: str = "deepseek-v4-flash",
        temperature: float = 0.0,
        max_history: int = 6,
        retrieval_limit: int = 4,
    ):
        if role not in SYSTEM_PROMPTS:
            raise ValueError(f"unknown agent role: {role}")
        self.role = role
        self.private_id = f"agent_{role}"
        self.shared_id = shared_user_id
        self.private_scm = SCMClient(user_id=self.private_id, base_url=scm_url, timeout=60.0)
        self.shared_scm = SCMClient(user_id=self.shared_id, base_url=scm_url, timeout=60.0)
        self.llm = _make_deepseek_llm(model=model, temperature=temperature)
        self.max_history = max_history
        self.retrieval_limit = retrieval_limit
        self.history: List = []
        self.turns: List[AgentTurn] = []

    def chat(self, user_input: str, store_user: bool = True) -> AgentTurn:
        turn = AgentTurn(role=self.role, user=user_input, response="")

        # 1. Retrieve from BOTH namespaces. wait_for_pending=True so prior
        #    async ingests are visible to the search (v0.7.2 read-your-writes).
        try:
            priv = self.private_scm.search_memory(
                user_input, limit=self.retrieval_limit, wait_for_pending=True,
            )
            for m in priv.get("memories", []) or []:
                if m.get("description"):
                    turn.private_retrieved.append(m["description"])
            turn.private_context = priv.get("memory_context", "") or ""
            turn.wake_summary_pending = priv.get("wake_summary_pending")
        except Exception as e:
            turn.error = (turn.error or "") + f" private_search: {e!r}"

        try:
            shared = self.shared_scm.search_memory(
                user_input, limit=self.retrieval_limit, wait_for_pending=True,
            )
            for m in shared.get("memories", []) or []:
                if m.get("description"):
                    turn.shared_retrieved.append(m["description"])
            turn.shared_context = shared.get("memory_context", "") or ""
        except Exception as e:
            turn.error = (turn.error or "") + f" shared_search: {e!r}"

        # 2. Build system prompt with both contexts
        sys_prompt = SYSTEM_PROMPTS[self.role]
        if turn.shared_retrieved or turn.shared_context:
            blob = turn.shared_context or " | ".join(turn.shared_retrieved)
            sys_prompt += f"\n\n[Shared memory about the user]\n{blob}"
        if turn.private_retrieved or turn.private_context:
            blob = turn.private_context or " | ".join(turn.private_retrieved)
            sys_prompt += f"\n\n[Your private memory]\n{blob}"
        if turn.wake_summary_pending and turn.wake_summary_pending.get("narrative"):
            sys_prompt += (
                f"\n\n[While the user was away, you autonomously noticed:]\n"
                f"{turn.wake_summary_pending['narrative']}"
            )

        # 3. Generate via DeepSeek v4
        messages = [SystemMessage(content=sys_prompt)]
        for m in self.history[-self.max_history * 2:]:
            messages.append(m)
        messages.append(HumanMessage(content=user_input))
        try:
            ai = self.llm.invoke(messages)
            turn.response = ai.content if hasattr(ai, "content") else str(ai)
        except Exception as e:
            turn.error = (turn.error or "") + f" llm.invoke: {e!r}"

        self.history.append(HumanMessage(content=user_input))
        if turn.response:
            self.history.append(AIMessage(content=turn.response))

        # 4. Store user statement in both namespaces (user facts are
        #    universal); responses stay private (agent-specific reasoning).
        if store_user:
            try:
                self.shared_scm.add_memory(text=user_input, metadata={"surfaced_by": self.role})
            except Exception as e:
                turn.error = (turn.error or "") + f" shared_add: {e!r}"
            try:
                self.private_scm.add_memory(
                    text=f"[{self.role} session] {user_input}",
                    metadata={"role": self.role},
                )
            except Exception as e:
                turn.error = (turn.error or "") + f" private_add: {e!r}"

        self.turns.append(turn)
        return turn

    def remember_decision(self, fact: str) -> None:
        """Persist an agent-internal decision to the agent's private memory only."""
        try:
            self.private_scm.add_memory(
                text=fact,
                metadata={"role": self.role, "kind": "decision"},
            )
        except Exception:
            pass

    def force_consolidate(self, mode: str = "deep") -> dict:
        """Trigger sleep on this agent's private memory (and shared too)."""
        out = {"private": {}, "shared": {}}
        try:
            out["private"] = self.private_scm.consolidate(mode=mode)
        except Exception as e:
            out["private"] = {"error": str(e)}
        try:
            out["shared"] = self.shared_scm.consolidate(mode=mode)
        except Exception as e:
            out["shared"] = {"error": str(e)}
        return out
