"""SCM integrations: MCP server, REST API endpoints, framework adapters,
and exported tool definitions for ChatGPT/Claude/Gemini.

This package is the integration surface for the SCM memory layer. The
core SCM modules (src.chat, src.core, src.sleep, src.lifecycle) are
LLM-agnostic and harness-agnostic; this package wires them into specific
ecosystems.

Available submodules:
    mcp_server      Model Context Protocol server (Claude Desktop, Cursor,
                    ChatGPT-with-MCP, any MCP-compatible client).
    tools           The five canonical SCM tool definitions, exported in
                    multiple formats (OpenAI, Anthropic, Gemini, OpenAPI).
    langchain_      LangChain BaseMemory adapter.
    llamaindex_     LlamaIndex BaseMemory adapter.
"""
