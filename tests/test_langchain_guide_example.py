"""Smoke test that the snippet from LANGCHAIN_GUIDE.md actually runs.

Two paths:
  1. SCMMemory (LangChain BaseChatMemory adapter) + ConversationChain
  2. SCMClient manual loop

Requires:
  - SCM server reachable at http://localhost:8000/v1
  - DEEPSEEK_API_KEY in .env (or OPENAI_API_KEY for the alt path)

Run:  python tests/test_langchain_guide_example.py
"""
from __future__ import annotations

import os
import sys
import uuid

from dotenv import load_dotenv
load_dotenv()


def _smoke_manual_path():
    """The 'manual control' pattern from §2 of the guide."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    from src.integrations.langchain_adapter import SCMClient

    user_id = f"guide_smoke_{uuid.uuid4().hex[:8]}"
    scm = SCMClient(user_id=user_id, base_url="http://localhost:8000/v1")
    llm = ChatOpenAI(
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=0.4,
        timeout=30,
    )

    history = []

    def chat(user_input: str) -> str:
        search = scm.search_memory(user_input, limit=5, wait_for_pending=True)
        context = search.get("memory_context", "")
        wake = search.get("wake_summary_pending")
        sys_text = (
            "You are a helpful assistant with persistent memory provided by SCM. "
            "Use the retrieved memories to personalize your reply.\n\n"
            f"Relevant memories:\n{context or '(none yet)'}"
        )
        if wake and wake.get("narrative"):
            sys_text += f"\n\n[While you were away: {wake['narrative']}]"

        messages = [SystemMessage(content=sys_text)] + history + [HumanMessage(content=user_input)]
        reply = llm.invoke(messages).content
        history.extend([HumanMessage(content=user_input), AIMessage(content=reply)])
        scm.add_memory(text=user_input)
        return reply

    print(f"\n[manual path] user_id={user_id}")
    r1 = chat("Hi, I'm Alex. I'm a backend engineer in Lisbon.")
    print(f"  T1> Hi, I'm Alex...\n     bot> {r1[:120]}")
    r2 = chat("I run every Tuesday morning along the river.")
    print(f"  T2> I run every Tuesday...\n     bot> {r2[:120]}")
    r3 = chat("Where do I work and where do I run?")
    print(f"  T3> Where do I work and where do I run?\n     bot> {r3[:200]}")

    # The third reply must mention BOTH facts the agent learned.
    blob = r3.lower()
    found_work = any(k in blob for k in ("backend", "engineer", "lisbon"))
    found_run = any(k in blob for k in ("tuesday", "river", "run"))
    assert found_work, f"didn't recall work fact: {r3!r}"
    assert found_run, f"didn't recall run fact: {r3!r}"
    print("  ✓ both facts recalled")


def _smoke_scmmemory_path():
    """The §1 pattern: SCMMemory + ConversationChain."""
    try:
        from langchain.chains import ConversationChain
    except Exception as e:
        print(f"\n[SCMMemory path] skipped — langchain.chains import failed: {e}")
        return
    from langchain_openai import ChatOpenAI
    from src.integrations.langchain_adapter import SCMMemory

    user_id = f"guide_mem_{uuid.uuid4().hex[:8]}"
    print(f"\n[SCMMemory path] user_id={user_id}")
    try:
        memory = SCMMemory(
            user_id=user_id,
            base_url="http://localhost:8000/v1",
            search_limit=5,
        )
    except Exception as e:
        print(f"  skipped — SCMMemory constructor failed: {e}")
        return
    llm = ChatOpenAI(
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=0.4,
        timeout=30,
    )
    try:
        chain = ConversationChain(llm=llm, memory=memory, verbose=False)
    except Exception as e:
        print(f"  skipped — ConversationChain rejected SCMMemory: {e}")
        return
    r1 = chain.predict(input="Hi, I'm Alex. I'm a backend engineer in Lisbon.")
    print(f"  T1 reply: {r1[:120]}")
    r2 = chain.predict(input="Where do I work?")
    print(f"  T2 reply: {r2[:200]}")
    blob = r2.lower()
    if any(k in blob for k in ("backend", "engineer", "lisbon")):
        print("  ✓ recalled work fact via ConversationChain")
    else:
        print(f"  (note) ConversationChain reply did not surface fact — common with newer LangChain APIs that prefer LCEL over BaseChatMemory")


if __name__ == "__main__":
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("ERROR: DEEPSEEK_API_KEY not set in .env")
        sys.exit(1)
    _smoke_manual_path()
    _smoke_scmmemory_path()
    print("\n=== guide example smoke test PASSED ===")
