# Implementation Roadmap

A phased approach to building Animus, from proof-of-concept to full personal AI system.

---

## Development Philosophy

**Start ugly, make it work**
- CLI before GUI
- Local before cloud
- Working before elegant

**Own your data at every step**
- No cloud dependencies for core function
- Export/backup from day one
- Encryption by default

**Build for one user first**
- You are the test case
- Solve your problems before generalizing
- Resist premature abstraction

**Iterate publicly**
- Document as you go
- Share learnings
- Let the repo tell the story

---

## Phase 0: Foundation

*Get the core loop working*

### Goal
A local AI that remembers conversations and can retrieve context.

### Duration
2-4 weeks

### Tasks

- [x] Set up local LLM (Ollama with Llama 3 or Mistral)
- [x] Implement basic conversation interface (CLI)
- [x] Add vector database for memory (ChromaDB)
- [x] Build simple retrieval: "What did we discuss about X?"
- [x] Establish project structure and config management
- [x] Create basic logging and error handling

### Technical Stack
- Python 3.11+
- Ollama for local LLM
- ChromaDB for vector storage
- SQLite for structured data
- Click or Typer for CLI

### Success Criteria
- Can have multi-turn conversation
- Can recall information from previous sessions
- All data stored locally
- Response latency < 5 seconds on modest hardware

### Output
You can talk to it, it remembers, you own the data.

---

## Phase 1: Memory Architecture

*Make it actually know you*

### Goal
Structured memory that accumulates meaningful context over time.

### Duration
4-6 weeks

### Tasks

- [x] Implement episodic memory (conversation logs with timestamps)
- [x] Add semantic memory (facts, preferences, learnings)
- [x] Build procedural memory (workflows, patterns)
- [x] Create memory management tools (review, edit, delete)
- [x] Design tagging/categorization system
- [x] Implement memory search and retrieval optimization
- [x] Add memory export/import functionality

### Memory Schema

```python
# Episodic
{
    "id": "uuid",
    "timestamp": "ISO8601",
    "type": "conversation|event|decision",
    "content": "...",
    "context": {...},
    "embeddings": [...]
}

# Semantic
{
    "id": "uuid",
    "category": "fact|preference|entity|relationship",
    "subject": "...",
    "predicate": "...",
    "object": "...",
    "confidence": 0.0-1.0,
    "source": "stated|inferred|learned",
    "timestamp": "ISO8601"
}

# Procedural
{
    "id": "uuid",
    "name": "...",
    "trigger": "...",
    "steps": [...],
    "frequency": 0,
    "last_used": "ISO8601"
}
```

### Success Criteria
- Can answer "What do you know about X?"
- Can show how knowledge was acquired
- User can edit/delete any memory
- Memory persists across sessions reliably

### Output
It doesn't just remember conversations - it builds a model of you.

---

## Phase 2: Cognitive Capabilities

*Make it useful*

### Goal
Beyond chat - actual analysis and assistance.

### Duration
6-8 weeks

### Tasks

- [x] Implement tool use framework
  - [x] File system access (read, search)
  - [x] Web search integration
  - [ ] API call framework
- [x] Add analysis modes (quick vs. deep reasoning)
- [ ] Build register translation (formal ↔ casual ↔ technical)
- [x] Create situation briefing capability
- [x] Develop decision support framework
- [x] Implement basic task tracking

### Tool Framework

```python
class Tool:
    name: str
    description: str
    parameters: dict
    requires_approval: bool
    
    def execute(self, params: dict) -> Result:
        ...
    
    def validate(self, params: dict) -> bool:
        ...
```

### Analysis Modes

| Mode | Trigger | Behavior |
|------|---------|----------|
| Quick | Default | Single inference, immediate response |
| Deep | "Think about..." / complex query | Multi-step reasoning, longer response |
| Research | "Research..." / "Find out..." | Web search + synthesis |

### Success Criteria
- Can read and summarize local files
- Can search web when needed
- Adjusts communication style to context
- Can help with actual work tasks

### Output
A co-pilot that can actually help with real tasks.

---

## Phase 3: Multi-Interface

*Make it portable*

### Goal
Access from anywhere, context follows you.

### Duration
8-12 weeks

### Tasks

**Voice + API (v0.3.0)**
- [x] Add HTTP API server (FastAPI)
  - [x] /chat, /status endpoints
  - [x] Memory CRUD endpoints
  - [x] Tool execution endpoints
  - [x] Task management endpoints
  - [x] Decision analysis endpoint
  - [x] WebSocket for streaming
- [x] Add voice input/output (Whisper + TTS)
  - [x] VoiceInput with Whisper STT
  - [x] VoiceOutput with pyttsx3/edge-tts
  - [x] /voice and /speak CLI commands
  - [x] /server start/stop/status commands
- [x] Add API and voice configuration

**Remaining Multi-Interface Work**
- [ ] Build mobile interface (PWA or React Native)
- [ ] Implement sync layer for cross-device memory
- [ ] Create notification/ambient mode
- [ ] Design minimal interaction patterns
- [ ] Build handoff protocol
- [ ] Test across device types

### Sync Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Desktop   │────▶│  Sync Layer │◀────│   Mobile    │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                    ┌──────┴──────┐
                    │  Encrypted  │
                    │   Storage   │
                    └─────────────┘
```

### Sync Options

| Option | Pros | Cons |
|--------|------|------|
| Self-hosted (Syncthing) | Full control | Requires setup |
| Local network only | Simple, secure | Limited range |
| Encrypted cloud (E2E) | Convenient | Some trust required |

### Success Criteria
- Can start conversation on desktop, continue on phone
- Voice interaction works reliably
- Sync latency < 2 seconds on local network
- Works offline, syncs when connected

### Output
Start on desktop, continue on phone, no context lost.

---

## Phase 4: Integration

*Make it connected*

### Goal
Animus works with your other tools.

### Duration
6-8 weeks

### Tasks

**Core Integration Framework (v0.4.0)**
- [x] Integration base framework (BaseIntegration, IntegrationManager)
- [x] OAuth2 authentication flow helper
- [x] Integration configuration system
- [x] CLI commands (/integrate, /integrations, /disconnect)
- [x] API endpoints (/integrations/*)

**File System Integration**
- [x] File system indexing and search
- [x] Content search with regex
- [x] File reading tool

**External Service Integrations**
- [x] Calendar integration (Google Calendar via OAuth2)
- [x] Email integration (Gmail - read, draft, send via OAuth2)
- [x] Task management sync (Todoist via API key)
- [x] Webhook support for external triggers

**Remaining Work**
- [ ] Calendar integration (Apple, CalDAV)
- [ ] Messages integration (where permitted)

### Integration Framework

```python
class Integration:
    name: str
    auth_type: str  # oauth, api_key, local
    capabilities: list[str]  # read, write, search
    
    def connect(self) -> bool:
        ...
    
    def query(self, request: dict) -> Response:
        ...
    
    def push(self, data: dict) -> bool:
        ...
```

### Success Criteria
- Knows your schedule without being told
- Can find files by description, not just name
- Can draft communications in your style
- Integrations fail gracefully

### Output
It knows your schedule, your files, your commitments.

---

## Phase 5: Self-Learning (in progress)

*Make it grow*

Initially planned 2025-Q3 as a future phase, but work has been
landing organically through the Bootstrap intelligence layer.
The original aspirational task list shifted to a more concrete
implementation:

### Shipped

- [x] **Self-improvement loop** — `bootstrap/intelligence/proactive/checks/self_heal.py`. Auto-detects tool failures, slow tools, and error rates every 6h; auto-proposes improvements with AI analysis; sandboxes changes via `ImprovementSandbox` (backup + rollback); measures impact via baseline/post metrics scored -100 to +100.
- [x] **Identity proposal manager** — 20% change threshold for any modification to identity files (`bootstrap/identity/`). Operator approves or rejects.
- [x] **Active inference IntentResolver** — *spec written, not yet built*. Replaces evidence-counting stability with surprise-weighted Bayesian posterior. See `docs/specs/quorum_v2_week3-4_active_inference_resolver.md` for the next-level "actually learns from evidence" upgrade.

### Still aspirational

- [ ] Pattern detection engine across the broader event stream (Quorum v2 Week 1's TickEvent log makes this newly possible — see Active Work below)
- [ ] Preference inference (currently only style/voice via PersonaEngine)
- [ ] Workflow optimization suggestions (Forge has the eval framework but no auto-suggest layer)

### Constitutional principles (already enforced)

P1–P9 in `docs/CONSTITUTIONAL_PRINCIPLES.md` are the immutable guardrails the original Phase 5 spec called for. They constrain every Forge action and every IntentNode write. No separate `Guardrail` class needed.

---

## Active Work

Live roadmap for current work has moved to `docs/ROADMAP_quorum_v2.md` (5-week plan extending Quorum, not replacing).

### Quorum v2 — 5-week extension

| Week | Status | Description |
|------|--------|-------------|
| 1 | **Shipped 2026-05-10** (PR #36) | EventLog bitemporal-lite + signal_bus bridge + 4 mutation sites wired |
| 2 | Spec ready | LivenessWatchdog over event stream |
| 3-4 | Spec ready | Active-inference IntentResolver scorer (the one behavior change) |
| 5 | Spec ready | Coupling MI dashboard (read-only) |

Per-week specs at `docs/specs/quorum_v2_week*.md`. Decision provenance: ADL-20260510-001 in `notes/decisions/2026-05.md`.

### Hardening pass — 2026-05-10

| Task | Outcome |
|------|---------|
| Plaintext API key in config.toml | Migrated to `~/.local/share/animus/secrets.env` (chmod 400). New `secrets.env > env > config.toml` resolution order in `ApiSection`. |
| Webchat tool-execution gap | Diagnosed root cause (HybridBackend routed agentic queries to Ollama whose `generate_structured` is a stub). Routing classifier extended with 45 agentic verbs + URL/path detection. Tool-use nudge appended to system prompt when ToolExecutor wired. |
| Systemd user units | Reference units at `packages/bootstrap/contrib/systemd/`. Local install switched to systemd-managed services with `EnvironmentFile=-%h/.local/share/animus/secrets.env`. |
| Mypy regression gate | Per-package baselines captured (~1,026 errors total). CI fails only when count grows. `scripts/mypy-count.sh` + `.github/mypy-baseline.json`. |
| Test count visibility | Live shields.io badge from `.github/test-counts.json`. Auto-refreshed by CI on push to main. `scripts/test-count.sh`. |

---

## Phase 6: Wearable / Ambient (deferred)

*Make it present*

Original aspiration: ring/wearable form factor with full Animus capability. **Not on the active roadmap.**

The current focus is making the existing software substrate (Bootstrap dashboard, message gateway, intelligence layer) actually work end-to-end. Until webchat reliably executes tools and the self-improvement loop is proven across more than the seed checks, hardware form factor is premature.

Reopen this phase only when:
- Software stack is operationally stable for 30+ consecutive days
- Real measured user value justifies the hardware build cost
- A specific use case can't be served by phone + earbuds (current best-available form factor)

The pre-existing detail (form factors, vehicle integration, storage device mode) is preserved in git history — see `git log -- docs/ROADMAP.md` for the original Phase 6 spec.

---

## Timeline Overview

```
Phase 0: Foundation          ████░░░░░░░░░░░░░░░░  Weeks 1-4         SHIPPED
Phase 1: Memory              ░░░░████████░░░░░░░░  Weeks 5-10        SHIPPED
Phase 2: Cognitive           ░░░░░░░░░░████████░░  Weeks 11-18       SHIPPED
Phase 3: Multi-Interface     ░░░░░░░░░░░░░░░░████  Weeks 19-30       SHIPPED
Phase 4: Integration         ░░░░░░░░░░░░░░░░░░░░  Weeks 31-38
Phase 5: Self-Learning       ░░░░░░░░░░░░░░░░░░░░  Weeks 39-48
Phase 6: Wearable            ░░░░░░░░░░░░░░░░░░░░  Weeks 49+
```

**Total estimated time to MVP (Phase 3): ~30 weeks**
**Total estimated time to full vision: 12-18 months**

---

## What's Buildable Now vs. Aspirational

### Buildable Today
- Local LLM with persistent memory
- Desktop + mobile text interface
- Basic voice integration
- Personal knowledge base with retrieval
- Simple integrations

### Near-term (6-12 months)
- Improved local models approaching API quality
- Better voice-first interfaces
- Wearable integrations (existing hardware)
- Robust cross-device sync

### Aspirational
- True seamless device handoff
- Real-time ambient awareness
- Ring/minimal form factor with full capability
- Self-improving personalization
- Full vehicle integration

---

## Getting Started

See [CONTRIBUTING.md](../CONTRIBUTING.md) for how to get involved.

Start with Phase 0. Get something working. Iterate from there.
