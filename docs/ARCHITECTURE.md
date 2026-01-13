# Architecture Overview

Animus is designed as a modular system with four primary layers, built for extensibility, portability, and user sovereignty.

---

## System Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Interface Layer                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────┐  │
│  │ Desktop  │ │  Mobile  │ │ Wearable │ │    Vehicle    │  │
│  │   App    │ │   App    │ │  (Ring)  │ │  (CarPlay)    │  │
│  └──────────┘ └──────────┘ └──────────┘ └───────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    Cognitive Layer                          │
│         (reasoning, analysis, generation, learning)         │
├─────────────────────────────────────────────────────────────┤
│                     Memory Layer                            │
│       (context, knowledge, patterns, file storage)          │
├─────────────────────────────────────────────────────────────┤
│                      Core Layer                             │
│       (identity, security, preferences, guardrails)         │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Layer

The foundation. Defines *who* this Animus belongs to and what it cannot do.

### Components

**Identity**
- Cryptographic ownership verification
- This instance serves one user, authenticated
- Device pairing and trust establishment

**Preferences**
- Communication style (formal, casual, technical)
- Priorities and values
- Interaction patterns

**Security**
- Encryption at rest (AES-256 minimum)
- Access control and authentication
- Secure key storage
- Audit logging

**Guardrails**
- User-defined constraints on behavior
- Inviolable safety rules (cannot harm user)
- Boundary enforcement for self-learning
- Action approval thresholds

### Design Principles

- Core layer is immutable without explicit user authentication
- Guardrails cannot be overridden by learned behavior
- All access logged and auditable
- Fail-secure: if uncertain, restrict rather than permit

---

## Memory Layer

What makes it *yours* over time. The accumulated context that transforms a generic AI into a personal one.

### Memory Types

**Episodic Memory**
- Conversations with timestamps
- Events and decisions
- Context of interactions
- "What happened"

**Semantic Memory**
- Facts about user and their world
- Learned preferences and knowledge
- Relationships and entities
- "What you know"

**Procedural Memory**
- Workflows and patterns
- How you do things
- Repeated task sequences
- "How you work"

**Active Context**
- Current situation
- Recent conversation threads
- Live priorities and tasks
- "What's happening now"

**File Storage**
- User documents and files
- Portable across devices
- Encrypted at rest
- Accessible to AI for context

### Implementation Options

| Component | Options | Tradeoffs |
|-----------|---------|-----------|
| Vector DB | ChromaDB, Qdrant, Milvus | ChromaDB simplest for local |
| Knowledge Graph | Neo4j, local JSON-LD | Neo4j powerful but heavy |
| Structured Data | SQLite, DuckDB | SQLite most portable |
| File Storage | Local encrypted FS, IPFS | Local simpler, IPFS for sync |

### Memory Management

- User can review, edit, delete any memory
- Retention policies configurable
- Export to standard formats
- Import from other systems

---

## Cognitive Layer

The reasoning engine. Model-agnostic by design.

### Components

**Model Interface**
- Abstraction layer for different LLMs
- Local models: Llama, Mistral, Phi
- API models: Claude, GPT (optional, user choice)
- Hot-swappable based on task requirements

**Tool Use Framework**
- File system access
- Web search and retrieval
- API calls to external services
- Device control and automation

**Analysis Modes**

| Mode | Use Case | Resources |
|------|----------|-----------|
| Quick | Simple queries, chat | Fast, low compute |
| Standard | Most interactions | Balanced |
| Deep | Complex analysis, planning | High compute, slower |
| Background | Learning, pattern detection | Async, low priority |

**Register Translation**
- Formal ↔ Casual ↔ Technical
- Context-aware communication adjustment
- User-defined style preferences

**Self-Learning Engine**
- Pattern recognition from interactions
- Preference inference
- Workflow optimization suggestions
- Bounded by guardrails - cannot learn outside permitted domains

### Learning Constraints

The self-learning system operates within strict boundaries:

1. **Cannot learn to bypass guardrails** - Safety rules are immutable
2. **Cannot learn harmful patterns** - Even if user attempts to teach them
3. **Transparency required** - User can see what was learned
4. **Reversible** - Any learned behavior can be unlearned
5. **Approval thresholds** - Significant changes require user confirmation

---

## Interface Layer

How you interact across contexts. The key requirement is **seamless handoff** - context follows you across devices.

### Supported Interfaces

**Desktop Application**
- Full-featured interface
- Long-form work and complex tasks
- File management and organization
- Development and technical work

**Mobile Application**
- Voice-first design
- Quick queries and responses
- On-the-go access
- Notification management

**Wearable Interface**
- Minimal, ambient
- Voice-activated
- Haptic feedback
- Glanceable information

**Vehicle Integration**
- CarPlay / Android Auto compatible
- Voice-primary interaction
- Driving-safe UI patterns
- Location-aware context

**Storage Device Mode**
- Animus as portable storage
- Plug into any compatible device
- Secure file access
- Temporary interface on host device

### Handoff Protocol

```
Device A (active)
    │
    ├── Saves current context to sync layer
    ├── Marks conversation as "in handoff"
    │
    ▼
Sync Layer (encrypted)
    │
    ├── Validates Device B authorization
    ├── Transfers active context
    │
    ▼
Device B (activating)
    │
    ├── Retrieves context
    ├── Resumes conversation seamlessly
    └── User experiences no interruption
```

### Communication Protocol

For real-time communication across devices:

- WebSocket for low-latency sync
- End-to-end encryption
- Conflict resolution (last-write-wins with history)
- Offline-first with eventual consistency

---

## Data Flow

### Standard Interaction

```
User input (any interface)
         │
         ▼
┌─────────────────────┐
│   Interface Layer   │  ← Captures, normalizes input
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│     Core Layer      │  ← Authenticates, applies preferences
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│    Memory Layer     │  ← Retrieves relevant context
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Cognitive Layer   │  ← Reasons, generates response
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│    Memory Layer     │  ← Stores new context, updates patterns
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Interface Layer   │  ← Delivers response to user
└─────────────────────┘
```

### Learning Flow

```
Interaction occurs
         │
         ▼
┌─────────────────────┐
│ Pattern Detection   │  ← Identifies potential learning
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Guardrail Check     │  ← Is this learning permitted?
└─────────────────────┘
         │
    ┌────┴────┐
    │         │
   YES        NO → Log and discard
    │
    ▼
┌─────────────────────┐
│ Threshold Check     │  ← Does this need approval?
└─────────────────────┘
         │
    ┌────┴────┐
    │         │
  Minor     Major → Queue for user approval
    │
    ▼
┌─────────────────────┐
│ Apply Learning      │  ← Update relevant memory
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Log Change          │  ← Transparent, reversible
└─────────────────────┘
```

---

## Security Model

### Threat Model

| Threat | Mitigation |
|--------|------------|
| Device theft | Encryption at rest, secure boot |
| Network interception | E2E encryption, certificate pinning |
| Malicious input | Guardrails, input sanitization |
| Unauthorized access | Multi-factor auth, device trust |
| Data exfiltration | Local-first, explicit sharing only |

### Privacy Guarantees

1. **No telemetry without consent** - Nothing phones home by default
2. **No training data extraction** - Your data doesn't improve someone else's model
3. **Explicit sharing only** - Data leaves device only when you choose
4. **Audit trail** - You can see everything that happened

---

## Extensibility

### Plugin Architecture

Animus supports plugins for:
- Additional interfaces
- New tool integrations
- Custom memory backends
- Specialized cognitive modules

### API Layer

RESTful API for:
- Third-party integrations
- Custom clients
- Automation scripts
- Inter-system communication

All API access subject to Core Layer authentication and guardrails.
