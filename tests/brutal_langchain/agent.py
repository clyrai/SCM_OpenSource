"""
A real LangChain agent backed by SCM memory.

Uses modern LangChain 1.x patterns: ChatOllama for the LLM, message lists
for conversation state, and direct SCM HTTP calls for memory ops. This
mirrors what a real third-party developer would build when integrating
SCM into their LangChain app.

The agent's loop is intentionally simple:
    1. On each user turn:
        a. search_memory(query=user_msg) → retrieve relevant context
        b. inject retrieved context + any pending wake-summary into the
           system prompt
        c. call ChatOllama with system + history + user_msg
        d. add_memory(text=user_msg) → store user's input
    2. (Sleep cycles fire automatically in the SCM background sweeper.)

This is exactly the pattern documented in docs/INTEGRATIONS.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

# Defer SCM client import so this module is safe to load when the SCM
# server isn't running yet.
from src.integrations.langchain_adapter import SCMClient


SYSTEM_PROMPT_BASE = (
    "You are a helpful assistant with persistent memory provided by SCM. "
    "Before responding, you will receive any relevant memories the system "
    "retrieved for you, plus any wake-summary the system produced "
    "autonomously while the user was away. Use them naturally in your "
    "response — if the wake-summary mentions something relevant, surface "
    "it to the user; otherwise just answer the question. Keep your "
    "responses concise (1-3 sentences). Do not invent facts not in your "
    "memory or the user's current message."
)


@dataclass
class AgentTurn:
    """One turn of the conversation, with memory diagnostics."""
    user: str
    response: str
    retrieved_memories: List[str] = field(default_factory=list)
    memory_context: str = ""              # spreading-activation context blob
    wake_summary_pending: Optional[dict] = None
    error: Optional[str] = None


class SCMLangChainAgent:
    """A LangChain agent that uses SCM as its memory layer."""

    def __init__(
        self,
        user_id: str,
        scm_url: str = "http://localhost:8000/v1",
        model: str = "llama3.2:latest",
        temperature: float = 0.0,
        max_history: int = 10,
        retrieval_limit: int = 5,
    ):
        self.user_id = user_id
        self.scm = SCMClient(user_id=user_id, base_url=scm_url, timeout=60.0)
        self.llm = ChatOllama(model=model, temperature=temperature)
        self.history: List = []
        self.max_history = max_history
        self.retrieval_limit = retrieval_limit
        self.turns: List[AgentTurn] = []

    def chat(self, user_input: str, store_user: bool = True) -> AgentTurn:
        """One conversational turn: retrieve, generate, store.

        With v0.7.2 async ingest, search_memory passes wait_for_pending
        so the test sees writes from prior turns even if the background
        ingest worker hasn't drained yet — preserves write-then-read
        consistency that the brutal scoring depends on.
        """
        turn = AgentTurn(user=user_input, response="")

        # 1. Retrieve memories from SCM
        retrieved_text = ""
        wake_pending = None
        try:
            search = self.scm.search_memory(
                user_input, limit=self.retrieval_limit, wait_for_pending=True,
            )
            mems = search.get("memories", []) or []
            ctx = search.get("memory_context", "") or ""
            retrieved_text = ctx
            turn.memory_context = ctx
            for m in mems:
                if m.get("description"):
                    turn.retrieved_memories.append(m["description"])
                    if not retrieved_text:
                        retrieved_text = m["description"]
                    else:
                        retrieved_text += "\n- " + m["description"]
            wake_pending = search.get("wake_summary_pending")
            turn.wake_summary_pending = wake_pending
        except Exception as e:
            turn.error = f"search_memory failed: {e!r}"

        # 2. Build the system prompt with retrieved context
        sys_prompt = SYSTEM_PROMPT_BASE
        if retrieved_text:
            sys_prompt += f"\n\nRelevant memories:\n{retrieved_text}"
        if wake_pending and wake_pending.get("narrative"):
            sys_prompt += (
                f"\n\n[While the user was away, you noticed:]\n"
                f"{wake_pending['narrative']}"
            )

        # 3. Generate via LangChain ChatOllama
        messages = [SystemMessage(content=sys_prompt)]
        # Trim history to max_history exchanges to keep context bounded
        for m in self.history[-self.max_history * 2 :]:
            messages.append(m)
        messages.append(HumanMessage(content=user_input))

        try:
            ai_msg = self.llm.invoke(messages)
            turn.response = ai_msg.content if hasattr(ai_msg, "content") else str(ai_msg)
        except Exception as e:
            turn.error = (turn.error or "") + f" llm.invoke failed: {e!r}"
            turn.response = ""

        # 4. Update local history (used for short-term coherence, not for memory)
        self.history.append(HumanMessage(content=user_input))
        if turn.response:
            self.history.append(AIMessage(content=turn.response))

        # 5. Store the user input in SCM (the agent response is optional;
        # we don't store it here because the user's facts are what matter
        # for memory, not the agent's reformulations).
        if store_user:
            try:
                self.scm.add_memory(text=user_input)
            except Exception as e:
                turn.error = (turn.error or "") + f" add_memory failed: {e!r}"

        self.turns.append(turn)
        return turn

    def force_consolidate(self, mode: str = "deep") -> dict:
        """Manual override — most callers don't need this; SCM auto-fires."""
        return self.scm.consolidate(mode=mode)

    def wake_summary(self, since_hours: float = 24.0) -> dict:
        return self.scm.wake_summary(since_hours=since_hours)
