"""
LangChain memory adapter for SCM.

Drop-in replacement for ConversationBufferMemory / VectorStoreMemory etc.
Calls the SCM HTTP API (/v1/memories, /v1/memories/search) — does NOT
require LangChain's vector-store abstraction.

Usage:

    from langchain.chains import ConversationChain
    from langchain.llms import OpenAI

    from src.integrations.langchain_adapter import SCMMemory

    memory = SCMMemory(
        user_id="alice@example.com",
        base_url="http://localhost:8000/v1",
    )

    chain = ConversationChain(
        llm=OpenAI(temperature=0),
        memory=memory,
        verbose=True,
    )

    chain.predict(input="Hi, my name is Alice.")
    chain.predict(input="What did I just tell you?")
    # SCM searches its memory, returns Alice's intro, the chain works.

Idle detection is automatic — once the user has not called the chain for
5 minutes (configurable), SCM fires a sleep cycle in the background.
The next call to chain.predict() automatically surfaces any cached
wake-summary as part of the loaded memory variables.

This adapter does not import langchain at module load — that keeps it
optional. SCMMemory subclasses langchain's BaseChatMemory only when
actually instantiated.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests


# ─── Standalone (langchain-free) variant ──────────────────────────────────


class SCMClient:
    """Plain HTTP client for the SCM /v1 API. Useful without LangChain."""

    def __init__(
        self,
        user_id: str = "default",
        base_url: str = "http://localhost:8000/v1",
        timeout: float = 30.0,
        api_key: Optional[str] = None,
    ):
        self.user_id = user_id
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    # Method names match the tool names in src/integrations/tools.py
    # so the Python SDK, JS SDK, and tool definitions all use the same
    # vocabulary.

    def add_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._post("/memories", {
            "text": text, "user_id": self.user_id, "metadata": metadata or {},
        })

    def search_memory(
        self,
        query: str,
        limit: int = 5,
        wait_for_pending: bool = False,
    ) -> Dict[str, Any]:
        body = {"query": query, "user_id": self.user_id, "limit": limit}
        if wait_for_pending:
            body["wait_for_pending"] = True
        return self._post("/memories/search", body)

    # Aliases for backward compat — older docs may have used these.
    add = add_memory
    search = search_memory

    def consolidate(self, mode: str = "deep") -> Dict[str, Any]:
        return self._post("/memories/consolidate", {
            "user_id": self.user_id, "mode": mode,
        })

    def wake_summary(self, since_hours: float = 24.0) -> Dict[str, Any]:
        return self._get("/wake-summary", {
            "user_id": self.user_id, "since_hours": since_hours,
        })

    def forget(self, memory_id: str) -> Dict[str, Any]:
        r = requests.delete(
            f"{self.base_url}/memories/{memory_id}",
            params={"user_id": self.user_id},
            headers=self._headers,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def memory_lineage(self, memory_id: str) -> Dict[str, Any]:
        memory_id = (memory_id or "").strip()
        if not memory_id:
            raise ValueError("memory_id is required")
        encoded = quote(memory_id, safe="")
        return self._get(
            f"/memories/{encoded}/lineage",
            {"user_id": self.user_id},
        )

    lineage = memory_lineage

    def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(
            f"{self.base_url}{path}",
            json=body,
            headers=self._headers,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.get(
            f"{self.base_url}{path}",
            params=params,
            headers=self._headers,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()


# ─── LangChain BaseChatMemory adapter ─────────────────────────────────────


def _import_langchain():
    """Defer the import so this module is importable without LangChain installed."""
    try:
        from langchain.memory.chat_memory import BaseChatMemory  # type: ignore
        from langchain.schema import HumanMessage, AIMessage  # type: ignore
        return BaseChatMemory, HumanMessage, AIMessage
    except ImportError as e:
        raise ImportError(
            "LangChain is not installed. Run: pip install langchain"
        ) from e


def SCMMemory(
    user_id: str = "default",
    base_url: str = "http://localhost:8000/v1",
    timeout: float = 30.0,
    api_key: Optional[str] = None,
    memory_key: str = "history",
    input_key: str = "input",
    output_key: str = "output",
    search_limit: int = 5,
    return_messages: bool = False,
):
    """Construct a LangChain-compatible memory backed by SCM.

    Returns a `BaseChatMemory` subclass instance. The class is constructed
    dynamically so importing this module does not require LangChain.
    """
    BaseChatMemory, HumanMessage, AIMessage = _import_langchain()

    client = SCMClient(user_id=user_id, base_url=base_url,
                       timeout=timeout, api_key=api_key)

    class _SCMMemoryImpl(BaseChatMemory):
        memory_key_: str = memory_key  # type: ignore[misc]
        input_key_: Optional[str] = input_key  # type: ignore[misc]
        output_key_: Optional[str] = output_key  # type: ignore[misc]

        @property
        def memory_variables(self) -> List[str]:
            return [memory_key]

        def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            query = inputs.get(input_key) or inputs.get("query") or ""
            if not query:
                return {memory_key: [] if return_messages else ""}

            try:
                resp = client.search_memory(query, limit=search_limit)
            except Exception:
                resp = {"memories": [], "memory_context": ""}

            memories = resp.get("memories", []) or []
            context_str = resp.get("memory_context", "") or ""

            # If a wake-summary is pending (auto-surfaced by SCM after idle
            # consolidation), prepend it to the context so the LLM sees it.
            wake = resp.get("wake_summary_pending")
            if wake:
                ws_text = wake.get("narrative") or ""
                if ws_text:
                    context_str = f"[While you were away: {ws_text}]\n\n" + context_str

            if return_messages:
                msgs = []
                for m in memories:
                    msgs.append(AIMessage(content=m.get("description", "")))
                if wake:
                    ns = wake.get("narrative")
                    if ns:
                        msgs.insert(0, AIMessage(content=f"[Wake summary] {ns}"))
                return {memory_key: msgs}

            # Build a plain-text history block for non-message-format chains.
            blob_lines = []
            if context_str:
                blob_lines.append(context_str)
            for m in memories:
                d = m.get("description")
                if d:
                    blob_lines.append(f"- {d}")
            return {memory_key: "\n".join(blob_lines)}

        def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
            user_text = inputs.get(input_key) or inputs.get("query") or ""
            ai_text = outputs.get(output_key) or ""
            try:
                if user_text:
                    client.add_memory(text=user_text, metadata={"role": "human"})
                if ai_text:
                    client.add_memory(text=ai_text, metadata={"role": "assistant"})
            except Exception:
                # Don't break the chain if SCM is unreachable.
                pass

        def clear(self) -> None:
            # SCM memories persist across sessions by design; clearing
            # deletes user data, which we do not do silently. Use the
            # /v1/memories DELETE endpoint explicitly.
            pass

    return _SCMMemoryImpl()
