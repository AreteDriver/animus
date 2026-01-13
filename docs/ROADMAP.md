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

- [ ] Set up local LLM (Ollama with Llama 3 or Mistral)
- [ ] Implement basic conversation interface (CLI)
- [ ] Add vector database for memory (ChromaDB)
- [ ] Build simple retrieval: "What did we discuss about X?"
- [ ] Establish project structure and config management
- [ ] Create basic logging and error handling

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

- [ ] Implement episodic memory (conversation logs with timestamps)
- [ ] Add semantic memory (facts, preferences, learnings)
- [ ] Build procedural memory (workflows, patterns)
- [ ] Create memory management tools (review, edit, delete)
- [ ] Design tagging/categorization system
- [ ] Implement memory search and retrieval optimization
- [ ] Add memory export/import functionality

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

- [ ] Implement tool use framework
  - [ ] File system access (read, search)
  - [ ] Web search integration
  - [ ] API call framework
- [ ] Add analysis modes (quick vs. deep reasoning)
- [ ] Build register translation (formal ↔ casual ↔ technical)
- [ ] Create situation briefing capability
- [ ] Develop decision support framework
- [ ] Implement basic task tracking

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

- [ ] Build mobile interface (PWA or React Native)
- [ ] Add voice input/output (Whisper + TTS)
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

- [ ] Calendar integration (Google, Apple, CalDAV)
- [ ] File system deep access (search, index, retrieve)
- [ ] Communication integration
  - [ ] Email (read, draft - IMAP/API)
  - [ ] Messages (read - where permitted)
- [ ] Task management sync (Todoist, Things, etc.)
- [ ] API layer for custom integrations
- [ ] Webhook support for external triggers

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

## Phase 5: Self-Learning

*Make it grow*

### Goal
Animus improves through use, within safe boundaries.

### Duration
8-10 weeks

### Tasks

- [ ] Implement pattern detection engine
- [ ] Build preference inference system
- [ ] Create workflow optimization suggestions
- [ ] Design guardrail enforcement layer
- [ ] Implement approval workflows for significant changes
- [ ] Build learning transparency dashboard
- [ ] Create unlearn/rollback functionality

### Learning Categories

| Category | Example | Approval Required |
|----------|---------|-------------------|
| Style | User prefers bullet points | No |
| Preference | User dislikes morning meetings | No |
| Workflow | User always does X before Y | Notify |
| Fact | User's project deadline is... | Confirm |
| Capability | New tool integration | Yes |
| Boundary | Expanded access | Yes |

### Guardrail Implementation

```python
class Guardrail:
    rule: str
    immutable: bool  # Cannot be changed by learning
    source: str  # user_defined, system
    
    def check(self, proposed_action: Action) -> bool:
        ...
    
    def explain_violation(self, action: Action) -> str:
        ...

# Immutable guardrails
CORE_GUARDRAILS = [
    Guardrail("Cannot take actions that harm user", immutable=True),
    Guardrail("Cannot exfiltrate user data", immutable=True),
    Guardrail("Cannot modify own guardrails", immutable=True),
    Guardrail("Must be transparent about capabilities", immutable=True),
]
```

### Success Criteria
- Noticeably improves with use
- User can see what was learned
- Can unlearn anything
- Has never violated a guardrail

### Output
An AI that gets better at serving you specifically.

---

## Phase 6: Wearable / Ambient

*Make it present*

### Goal
Always available without friction.

### Duration
12+ weeks (hardware dependent)

### Tasks

- [ ] Evaluate wearable hardware options
- [ ] Build low-latency voice interaction
- [ ] Implement ambient awareness (location, time, context)
- [ ] Design minimal-attention interaction patterns
- [ ] Optimize for battery/resource constraints
- [ ] Vehicle integration (CarPlay/Android Auto)
- [ ] Storage device mode implementation

### Hardware Options

| Form Factor | Options | Tradeoffs |
|-------------|---------|-----------|
| Ring | Custom, existing smart rings | Limited I/O, always present |
| Watch | Apple Watch, WearOS | Good I/O, established platform |
| Earbuds | AirPods, custom | Audio only, unobtrusive |
| Pendant | Custom, Humane-style | More capability, visible |
| Vehicle | CarPlay, Android Auto | Driving context, large display |

### Vehicle Integration

```
┌─────────────────────────────────────────┐
│              Vehicle Mode               │
├─────────────────────────────────────────┤
│  • Voice-primary interaction            │
│  • Location-aware context               │
│  • Driving-safe UI (minimal visual)     │
│  • Navigation integration               │
│  • Calendar/schedule awareness          │
│  • Hands-free communication drafting    │
└─────────────────────────────────────────┘
```

### Storage Device Mode

Animus hardware can function as secure portable storage:
- Plug into any compatible device
- Authenticate to unlock
- Access files through temporary interface
- No data remains on host device
- Full Animus capability if host supports it

### Success Criteria
- Can interact without looking at screen
- Context-aware suggestions are actually useful
- Battery life acceptable for daily use
- Works in vehicle safely

### Output
The ring. Or close to it.

---

## Timeline Overview

```
Phase 0: Foundation          ████░░░░░░░░░░░░░░░░  Weeks 1-4
Phase 1: Memory              ░░░░████████░░░░░░░░  Weeks 5-10
Phase 2: Cognitive           ░░░░░░░░░░████████░░  Weeks 11-18
Phase 3: Multi-Interface     ░░░░░░░░░░░░░░░░████  Weeks 19-30
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
