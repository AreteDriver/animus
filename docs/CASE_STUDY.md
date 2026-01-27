# Case Study: Animus

## Executive Summary

Animus is a framework for building a personal AI that persists, learns, and serves a single user by design. It implements a local-first exocortex — an external cognitive layer that accumulates memory across sessions and devices while keeping data under the user's control.

**By the numbers:**
- 18,000+ lines of Python
- 400+ tests across 10 test files
- 4 memory types (episodic, semantic, procedural, active context)
- 5 of 6 development phases complete
- 7 integrations (Ollama, Calendar, Gmail, Todoist, Whisper, ChromaDB, WebSocket sync)
- Version 0.6.0 — active development

---

## Problem

Current AI assistants are stateless, cloud-dependent, and generic. Every conversation starts from zero. There's no persistent memory of your preferences, decisions, or context. Your data trains someone else's models. Switching devices means losing context.

The gap:
- **No memory** — assistants forget everything between sessions
- **No personalization** — responses are generic, not adapted to you
- **No sovereignty** — your data lives on corporate servers, used for training
- **No continuity** — context doesn't follow you across devices
- **No learning** — the AI never gets better at helping *you* specifically

## Solution

Animus runs locally by default using Ollama for inference and ChromaDB for vector memory. It accumulates four types of memory across every interaction, syncs across devices via WebSocket, and learns your patterns within explicit guardrails.

Key capabilities:
- **Persistent memory** — Episodic (conversations), semantic (facts/knowledge), procedural (how-to), and active context (current session)
- **Local-first** — Ollama for inference, ChromaDB for storage, no cloud required
- **Cross-device sync** — WebSocket-based memory synchronization with Zeroconf discovery
- **Voice interface** — Whisper STT + TTS for hands-free interaction
- **Guardrailed learning** — Pattern detection bounded by safety rules, transparent and reversible
- **Integrations** — Google Calendar, Gmail, Todoist via OAuth2

---

## Architecture Decisions

### Local-First by Design

**Decision:** All core functionality runs locally. Cloud AI (Claude, OpenAI) is optional and opt-in.

**Why:** Personal AI handles sensitive data — decisions, health, finances, relationships. Cloud dependency creates a single point of failure and a privacy risk. Local-first means the system works offline, data stays on-device, and no third party can access or train on your information.

### Four-Layer Architecture

**Decision:** Separate the system into Interface, Cognitive, Memory, and Core layers.

**Why:** Each layer has different change rates and concerns. The interface layer (CLI, voice, API) changes frequently as new surfaces are added. The cognitive layer (reasoning, tools) evolves with AI capabilities. The memory layer (ChromaDB, sync) is stable infrastructure. The core layer (identity, security) rarely changes. Layering allows independent evolution.

### Four Memory Types

**Decision:** Implement episodic, semantic, procedural, and active context as distinct memory systems.

**Why:** Human cognition uses different memory types for different purposes. Episodic memory (past conversations) enables recall of specific interactions. Semantic memory (facts, preferences) enables personalization. Procedural memory (how-to) enables repeated task optimization. Active context (current session) enables coherent multi-turn interaction. Conflating these into a single vector store loses the structure that makes retrieval useful.

### Guardrailed Learning

**Decision:** Bound all learning with immutable safety rules — cannot harm user, cannot exfiltrate data, must be transparent, must be reversible.

**Why:** A personal AI that learns your patterns is powerful but dangerous without constraints. The guardrail system ensures the AI can't learn harmful behaviors, all inferences are visible to the user, and any learned pattern can be rolled back. This makes the learning system trustworthy enough to actually use.

### Phased Development

**Decision:** Build in six phases, each delivering a working system with incrementally more capability.

| Phase | Focus | Status |
|-------|-------|--------|
| 0 | Foundation — basic conversation + memory | Complete |
| 1 | Memory architecture — 3 memory types | Complete |
| 2 | Cognitive capabilities — tools, analysis | Complete |
| 3 | Multi-interface — voice + API + sync | Complete |
| 4 | Integration — calendar, email, tasks | Complete |
| 5 | Self-learning — pattern detection, guardrails | In progress |
| 6 | Ambient — wearable, vehicle integration | Future |

**Why:** Building the full vision at once would be a multi-year monolith. Phased delivery means each milestone is a working, testable system. Feedback from earlier phases informs later design.

---

## Technical Highlights

### Memory System

ChromaDB stores vector embeddings for semantic search across all memory types. Each memory entry is typed, timestamped, and tagged with source context. Retrieval uses cosine similarity with type-weighted scoring — episodic memories decay over time while semantic memories remain stable.

### Cross-Device Sync

WebSocket-based synchronization with Zeroconf for automatic device discovery on local networks. Memory changes propagate with eventual consistency. Conflict resolution uses last-write-wins with device priority ordering. All sync traffic is encrypted end-to-end.

### Voice Interface

OpenAI Whisper for speech-to-text, pyttsx3 or edge-tts for text-to-speech. Designed for hands-free operation — morning briefings, driving mode, meeting prep. Wake word detection triggers active listening.

### Security Model

- AES-256 encryption at rest
- Device pairing and trust verification
- Audit logging for all data access
- No telemetry without explicit consent
- Export and backup from day one

### Engineering Practices

- Type hints throughout (mypy checked)
- 400+ tests with pytest
- Ruff for linting and formatting
- Phased roadmap with clear milestones
- Comprehensive documentation (architecture, roadmap, use cases, safety)

---

## Results

| Metric | Value |
|--------|-------|
| Lines of code | 18,214 |
| Test count | 409 |
| Test files | 10 |
| Source files | 35 |
| Memory types | 4 |
| Phases complete | 5 of 6 |
| Integrations | 7 |
| Version | 0.6.0 |

---

## Tech Stack

Python 3.10+ · Ollama (Llama 3.2, Mistral) · ChromaDB · FastAPI · OpenAI Whisper · Rich CLI · WebSockets · Zeroconf · Google Calendar/Gmail API · Todoist API · pytest · Ruff · mypy
