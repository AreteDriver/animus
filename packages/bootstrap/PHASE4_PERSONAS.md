# Phase 4 — Persona & Voice Layer Spec
## Scope: Persona system, voice/tone customization, multi-persona routing, and personality dashboard

---

## WHAT WE'RE BUILDING

The piece that makes Animus feel like *yours*: a configurable personality
that adapts its tone, knowledge, and behavior per channel, per context,
and per user preference.

```
┌─────────────────────────────────────────────────────────────────┐
│  Persona Layer                                                  │
│  Defines who Animus "is" — configurable identity, voice, tone,  │
│  knowledge domains, and behavioral constraints.                 │
├─────────────────────────────────────────────────────────────────┤
│  Persona Engine                                                 │
│  Registry of persona profiles. Routes messages to the right     │
│  persona based on channel, context, or explicit switch.         │
├─────────────────────────────────────────────────────────────────┤
│  Voice System                                                   │
│  Tone presets (formal, casual, technical, mentor), custom        │
│  instructions, response length preferences, emoji policy.       │
├─────────────────────────────────────────────────────────────────┤
│  Knowledge Domains                                              │
│  Per-persona knowledge scoping — what the persona knows about,  │
│  what it refuses, what it defers to other personas.             │
├─────────────────────────────────────────────────────────────────┤
│  Context Awareness                                              │
│  Time-of-day tone shifts, channel-specific behavior,            │
│  conversation history-based adaptation.                         │
├─────────────────────────────────────────────────────────────────┤
│  Persona Dashboard (extends Phase 3 dashboard)                  │
│  /personas — manage persona profiles                            │
│  /voice — tone presets editor                                   │
│  /routing — channel-to-persona mapping                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## ARCHITECTURE

### Component 1: Persona Engine (`animus_bootstrap.personas.engine`)

```python
@dataclass
class PersonaProfile:
    """A complete persona definition."""
    id: str
    name: str                    # "Animus", "Work Assistant", "Creative Partner"
    description: str
    system_prompt: str           # Base personality instruction
    voice: VoiceConfig
    knowledge_domains: list[str] # Topics this persona handles
    excluded_topics: list[str]   # Topics to refuse/defer
    channel_bindings: dict[str, bool]  # Which channels this persona responds on
    active: bool = True
    is_default: bool = False

class PersonaEngine:
    """Routes messages to the appropriate persona."""

    def get_persona_for_message(self, message: GatewayMessage) -> PersonaProfile:
        """Select persona based on channel, context, explicit /persona switch."""

    def register_persona(self, persona: PersonaProfile) -> None: ...
    def list_personas(self) -> list[PersonaProfile]: ...
    def set_default(self, persona_id: str) -> None: ...
```

### Component 2: Voice System (`animus_bootstrap.personas.voice`)

```python
@dataclass
class VoiceConfig:
    """Tone and style configuration."""
    tone: str = "balanced"         # "formal" | "casual" | "technical" | "mentor" | "creative"
    max_response_length: str = "medium"  # "brief" | "medium" | "detailed"
    emoji_policy: str = "minimal"  # "none" | "minimal" | "expressive"
    language: str = "en"
    custom_instructions: str = ""  # Free-form additional instructions
    time_shifts: dict[str, str] = field(default_factory=dict)  # "morning": "energetic"

VOICE_PRESETS: dict[str, VoiceConfig]  # Built-in tone presets
```

### Component 3: Knowledge Domains (`animus_bootstrap.personas.knowledge`)

```python
class KnowledgeDomainRouter:
    """Routes questions to the persona with relevant domain expertise."""

    def classify_topic(self, text: str) -> list[str]:
        """Classify message into topic domains."""

    def find_best_persona(self, topics: list[str]) -> PersonaProfile | None:
        """Find persona with best domain match."""
```

### Component 4: Context Awareness (`animus_bootstrap.personas.context`)

```python
class ContextAdapter:
    """Adapts persona behavior based on context signals."""

    def adapt_prompt(
        self, persona: PersonaProfile, message: GatewayMessage,
        session_history: list[dict],
    ) -> str:
        """Build context-adapted system prompt."""
        # Consider: time of day, channel norms, conversation mood, user prefs
```

---

## CONFIG EXTENSIONS

```toml
[personas.default]
name = "Animus"
tone = "balanced"
max_response_length = "medium"
emoji_policy = "minimal"
system_prompt = "You are Animus, a personal AI assistant..."

[personas.profiles.work]
name = "Work Assistant"
tone = "formal"
knowledge_domains = ["coding", "architecture", "devops"]
channel_bindings = {slack = true, email = true}

[personas.profiles.creative]
name = "Creative Partner"
tone = "creative"
knowledge_domains = ["writing", "brainstorming", "art"]
channel_bindings = {discord = true, webchat = true}
```

---

## FILE STRUCTURE

```
src/animus_bootstrap/personas/
├── __init__.py
├── engine.py           # PersonaEngine, PersonaProfile
├── voice.py            # VoiceConfig, VOICE_PRESETS, prompt builder
├── knowledge.py        # KnowledgeDomainRouter, topic classification
├── context.py          # ContextAdapter, time/channel/mood awareness
└── storage.py          # SQLite persistence for persona profiles

src/animus_bootstrap/dashboard/
├── routers/
│   ├── personas.py     # /personas routes
│   └── routing.py      # /routing routes (channel-to-persona mapping)
├── templates/
│   ├── personas.html
│   └── routing.html
```

---

## INTEGRATION WITH PHASE 3

The IntelligentRouter gains persona awareness:

```python
class IntelligentRouter:
    async def handle_message(self, message):
        # 0. Select persona for this message
        persona = self._persona_engine.get_persona_for_message(message)
        # 1-8. Existing pipeline with persona.system_prompt injected
```

---

## IMPLEMENTATION ORDER

1. **Persona models** — PersonaProfile, VoiceConfig dataclasses
2. **Voice system** — Presets, prompt building, response formatting
3. **Persona engine** — Registry, channel binding, default selection
4. **Knowledge domains** — Topic classification, persona routing
5. **Context adapter** — Time shifts, channel norms, mood detection
6. **SQLite persistence** — Persona CRUD
7. **Router integration** — IntelligentRouter gains persona selection
8. **Dashboard** — /personas, /routing pages
9. **CLI** — `animus-bootstrap personas list/add/set-default`
10. **Tests** — Full coverage

---

## SUCCESS CRITERIA

- [ ] Default persona responds on all channels with consistent personality
- [ ] Channel-bound personas activate on their assigned channels
- [ ] Voice presets produce noticeably different response styles
- [ ] Time-of-day shifts adjust tone (energetic mornings, calm evenings)
- [ ] Persona profiles persistable via config and dashboard
- [ ] 80%+ test coverage on personas module
- [ ] All Phase 1-3 tests still pass
