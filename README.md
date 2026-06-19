# SCM — Sleep-Consolidated Memory for Language Agents

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-322-green.svg)](tests/)

**Other memory layers store facts. SCM learns from them while you're idle.**

SCM is the first open-source memory architecture for language agents that implements a complete biological memory lifecycle: bounded working memory, selective encoding, sleep-stage consolidation (NREM + REM), adaptive forgetting, contradiction-safe versioning, and autonomous learning during idle time.

## Key Results

| Metric | Value |
|--------|-------|
| Disambiguation recall (with sleep) | **0.9052** |
| Disambiguation recall (awake-only) | **0.0** |
| Noise reduction | **90.9%** |
| One-shot recall accuracy | **1.0** |
| Retrieval latency | **<0.3ms** |
| Regression tests | **322 passing** |

## Quick Start

```bash
pip install scm-memory
```

```python
from scm import SCMEngine

engine = SCMEngine(profile="chatbot")
engine.chat("My name is Saish and I live in Bangalore.")
engine.chat("I love filter coffee.")

# Force a sleep cycle to consolidate
engine.force_sleep("deep")

# Ask about stored memories
response, meta = engine.chat("Where do I live?")
print(response)  # "You live in Bangalore."
```

## Architecture

SCM implements seven phases of a biological memory lifecycle:

| Phase | Component | Function |
|-------|-----------|----------|
| 1 | AttentionGate | Selective encoding with 4-tier intensity |
| 2 | EventCompiler | Structured event frames (who/what/when/where/why) |
| 3 | SpreadingActivation | Cue-driven graph propagation retrieval |
| 4 | SleepKernel | Micro-sleep + Deep-sleep (NREM + REM) |
| 5 | ForgettingDynamics | Adaptive value-based forgetting |
| 6 | Guardrails | Paraphrase, evaluation harnesses |
| 7 | IdleLearner | Autonomous learning during user idle time |

## Deployment Profiles

| Profile | LLM | Embedding | Cost | Privacy |
|---------|-----|-----------|------|---------|
| A: Offline | heuristic | sentence-transformers | $0 | 100% local |
| B: Ollama | llama3.2 | nomic-embed-text | $0 | 100% local |
| C: Hybrid | DeepSeek | nomic-embed-text | ~$0.04/30 turns | text to cloud |
| D: All-cloud | GPT-4o-mini | text-embedding-3-large | ~$0.06/30 turns | all to cloud |

## Examples

```bash
# Run the quickstart
python examples/01_quickstart.py

# Run with wake summary
python examples/02_wake_summary.py

# Run with Ollama (local LLM)
python examples/03_with_ollama.py
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/ -k "sleep" -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Integrations

- **LangChain**: `SCMMemory` adapter + tool definitions
- **MCP Server**: stdio + HTTP transports for Claude Desktop, Cursor
- **REST API**: `/v1/memories`, `/v1/wake-summary`, `/v1/health`
- **Python SDK**: `from scm import SCMEngine`
- **JavaScript SDK**: `npm install scm-memory`

## Documentation

- [API Reference](docs/DEPLOYMENT.md)
- [LangChain Guide](docs/LANGCHAIN_GUIDE.md)
- [Integration Recipes](docs/INTEGRATIONS.md)
- [Benchmark Results](docs/BENCHMARKS.md)

## How It Works

### Wake Phase
During conversation, SCM encodes user input into typed semantic concepts, tags each with a 4-dimensional importance vector (novelty, emotion, task relevance, repetition), and stores recent episodes in a bounded 7-item working memory buffer.

### Sleep Phase
When the user goes idle, SCM enters sleep mode:
- **NREM**: Replays episodes, strengthens co-occurring concepts via Hebbian plasticity, applies synaptic downscaling
- **REM**: Generates novel concept combinations, creates new associative links
- **Forgetting**: Removes low-value memories while preserving important ones

### The Result
When you return and ask "What did you notice while I was away?", SCM produces a narrative like:
> "While you were away I noticed three things: You've changed jobs — I've moved you from Northstar Robotics to Atlas Labs. Your Tuesday-morning runs and Friday-night dinners with Mara have become weekly patterns. You've mentioned 'OAuth flow' five times without explaining it; I read up on it."

## Citation

```bibtex
@article{scm2026,
  title={SCM: Autonomous Lifelong Learning for Language Agents via Sleep-Stage Memory Consolidation},
  author={SCM Research Team},
  year={2026}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contact

- Author: Saish Shinde
- Email: blobopera@proton.me
- GitHub: [github.com/clyrai/SCM_OpenSource](https://github.com/clyrai/SCM_OpenSource)
