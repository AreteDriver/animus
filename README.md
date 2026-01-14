# Animus

*An exocortex architecture for personal cognitive sovereignty*

[![CI](https://github.com/AreteDriver/animus/actions/workflows/ci.yml/badge.svg)](https://github.com/AreteDriver/animus/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What is this?

Animus is a framework for building a **personal AI** - one that persists, learns, and serves a single user by design.

Current AI assistants are rented. Your context exists at the discretion of platform providers. Memory is a feature that can be revoked. The relationship resets at their convenience.

Animus explores an alternative: an AI that is **yours**.

```
17,000+ lines of Python | 266 tests | Local-first by default
```

---

## Quick Start

```bash
# Install
pip install -e .

# Run the interactive CLI
animus

# Or with API server
pip install -e ".[api]"
animus --api
```

### Minimal Example

```python
from animus import AnimusConfig, MemoryLayer, CognitiveLayer, ModelConfig

# Initialize
config = AnimusConfig()
memory = MemoryLayer(config.data_dir)
cognitive = CognitiveLayer(ModelConfig(provider="ollama", model="llama3.2"))

# Remember something
memory.remember(
    content="User prefers concise responses",
    memory_type="semantic",
    tags=["preference", "communication"],
    confidence=0.9,
)

# Retrieve relevant context
context = memory.recall("How should I communicate?", limit=5)

# Generate response with context
response = cognitive.think(
    prompt="Summarize my communication preferences",
    context=context,
)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Interface Layer                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────┐  │
│  │   CLI    │ │   API    │ │  Voice   │ │     Sync      │  │
│  │ (Rich)   │ │(FastAPI) │ │(Whisper) │ │  (WebSocket)  │  │
│  └──────────┘ └──────────┘ └──────────┘ └───────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    Cognitive Layer                          │
│         CognitiveLayer, ReasoningMode, ToolRegistry         │
├─────────────────────────────────────────────────────────────┤
│                     Memory Layer                            │
│     MemoryLayer (ChromaDB), SemanticFact, Conversation      │
├─────────────────────────────────────────────────────────────┤
│                    Learning Layer                           │
│       LearningLayer, Guardrails, Patterns, Rollback         │
├─────────────────────────────────────────────────────────────┤
│                      Core Layer                             │
│       AnimusConfig, DecisionFramework, TaskTracker          │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

| Module | Purpose |
|--------|---------|
| `memory.py` | Vector storage with ChromaDB, episodic/semantic/procedural memory types |
| `cognitive.py` | Model-agnostic reasoning (Ollama, Anthropic, OpenAI) |
| `learning/` | Pattern detection, preference inference, guardrails |
| `tools.py` | Extensible tool registry for file system, web, integrations |
| `sync/` | Cross-device state synchronization via WebSocket |
| `api.py` | FastAPI server for external integrations |
| `voice.py` | Whisper STT + TTS for voice interaction |

---

## Core Principles

- **Persistence** - Context accumulates across sessions, devices, and years
- **Sovereignty** - Your data stays yours. Local-first by default.
- **Loyalty** - Aligned to you, not to a platform's incentives
- **Portability** - Moves with you: CLI, API, voice, sync across devices
- **Growth** - Learns your patterns, priorities, and goals over time
- **Safety** - Guardrails are user-defined but inviolable

---

## Learning System

Animus learns from interactions while respecting strict boundaries:

```python
from animus import LearningLayer, MemoryLayer, AnimusConfig

config = AnimusConfig()
memory = MemoryLayer(config.data_dir)
learning = LearningLayer(memory, config.data_dir)

# Scan memories for patterns
patterns = learning.scan_and_learn()

# Review what was learned (transparency)
active = learning.get_active_learnings()

# Approve or reject pending learnings
pending = learning.get_pending_learnings()
learning.approve(pending[0].id)

# Rollback if needed
checkpoints = learning.rollback.list_checkpoints()
learning.rollback.restore(checkpoints[0].id)
```

### Safety Guardrails

```python
from animus.learning import GuardrailManager, Guardrail

guardrails = GuardrailManager(config.data_dir)

# View default safety rules
for g in guardrails.get_all_guardrails():
    print(f"{g.rule} (immutable: {g.immutable})")

# Core guardrails cannot be overridden by learned behavior
```

---

## Installation Options

```bash
# Core only (Ollama + ChromaDB)
pip install -e .

# With Anthropic Claude support
pip install -e ".[anthropic]"

# With API server
pip install -e ".[api]"

# With voice (Whisper + TTS)
pip install -e ".[voice]"

# With integrations (Todoist, Google)
pip install -e ".[integrations]"

# With cross-device sync
pip install -e ".[sync]"

# Everything
pip install -e ".[all]"
```

---

## CLI Usage

```bash
# Interactive mode
animus

# Single query
animus "What's on my schedule today?"

# With specific model
animus --model claude-3-5-sonnet-20241022

# Start API server
animus --api --port 8000

# Voice mode
animus --voice
```

---

## API Server

```bash
animus --api
```

```python
import httpx

# Chat endpoint
response = httpx.post("http://localhost:8000/chat", json={
    "message": "What are my priorities today?",
    "context_limit": 10,
})
print(response.json()["response"])

# Memory endpoints
httpx.post("http://localhost:8000/memory", json={
    "content": "Meeting with team at 3pm",
    "type": "episodic",
    "tags": ["calendar", "work"],
})

memories = httpx.get("http://localhost:8000/memory/search", params={
    "query": "meetings",
    "limit": 5,
})
```

---

## Documentation

- [Architecture Overview](docs/ARCHITECTURE.md) - System design and data flow
- [Implementation Roadmap](docs/ROADMAP.md) - Development phases
- [Use Cases](docs/USE_CASES.md) - Practical applications
- [Connectivity & Interfaces](docs/CONNECTIVITY.md) - Device integration
- [Safety & Ethics](docs/SAFETY.md) - Guardrails and privacy
- [Contributing](CONTRIBUTING.md) - How to contribute

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=animus --cov-report=term-missing

# Lint
ruff check .
ruff format .

# Type check
mypy animus/
```

---

## Philosophy

> "You don't own it. You rent access."

This is the fundamental problem with current AI assistants. Your relationship with the AI - the context it has about you, the patterns it's learned, the history you've built - exists at the pleasure of corporations whose incentives may diverge from yours at any moment.

Animus is an attempt to build something different: an AI that serves you because it's *yours*, not because a company's business model temporarily aligns with your needs.

The goal isn't to replace cloud AI services entirely - they have capabilities that local systems can't match. The goal is **sovereignty**: you control the core, you own the memory, you decide what gets shared and what stays private.

---

## License

MIT License - See [LICENSE](LICENSE) for details.
