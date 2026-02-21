# Phase 2 — Message Gateway Spec
## Scope: Unified messaging hub + channel integrations + conversation dashboard

---

## WHAT WE'RE BUILDING

The piece that makes Animus feel like OpenClaw: talk to your AI on any
channel, get responses routed through one brain, with context that
persists across channels and conversations.

```
┌─────────────────────────────────────────────────────────────────┐
│  MessageGateway (router)                                        │
│  Normalizes inbound messages from any channel into a common     │
│  format, routes through Animus cognitive layer, sends           │
│  responses back through the originating channel.                │
├─────────────────────────────────────────────────────────────────┤
│  Channel Adapters                                               │
│  Telegram | Discord | Slack | WhatsApp | Signal | Matrix |      │
│  Google Chat | WebChat | Email | SMS                            │
├─────────────────────────────────────────────────────────────────┤
│  Session Manager                                                │
│  Cross-channel conversation tracking. Start on Telegram,        │
│  continue on Slack, see full history in dashboard.              │
├─────────────────────────────────────────────────────────────────┤
│  Conversation Dashboard (extends Phase 1 dashboard)             │
│  /conversations — live message feed across all channels         │
│  /channels — connected channel health + onboarding              │
└─────────────────────────────────────────────────────────────────┘
```

---

## ARCHITECTURE

### Component 1: Message Gateway (`animus_bootstrap.gateway`)

**Core abstraction:** Every message from every channel gets normalized
into a `GatewayMessage`, routed through the AI, and the response gets
sent back through the originating channel.

```python
@dataclass
class GatewayMessage:
    """Normalized message format — channel-agnostic."""
    id: str                          # Unique message ID (UUID)
    channel: str                     # "telegram" | "discord" | "slack" | ...
    channel_message_id: str          # Original platform message ID
    sender_id: str                   # Platform-specific user ID
    sender_name: str                 # Display name
    text: str                        # Message content
    attachments: list[Attachment]    # Images, files, etc.
    timestamp: datetime              # When sent
    reply_to: str | None             # If replying to a previous message
    metadata: dict[str, Any]         # Channel-specific extras

@dataclass
class GatewayResponse:
    """Response to send back through a channel."""
    text: str
    attachments: list[Attachment]
    channel: str
    channel_message_id: str | None   # For threading/replies
    metadata: dict[str, Any]

@dataclass
class Attachment:
    """File attachment."""
    filename: str
    content_type: str
    data: bytes | None               # In-memory
    url: str | None                  # Remote URL
```

**MessageRouter class:**
```python
class MessageRouter:
    """Routes messages from channels through the AI and back."""

    def __init__(
        self,
        cognitive_backend: CognitiveBackend,  # LLM interface
        session_manager: SessionManager,       # Cross-channel sessions
        channels: dict[str, ChannelAdapter],   # Registered channels
    ) -> None: ...

    async def handle_message(self, message: GatewayMessage) -> GatewayResponse:
        """Process an inbound message and generate a response.

        1. Look up or create session for sender
        2. Add message to session context
        3. Route through cognitive backend (LLM)
        4. Record response in session
        5. Return response for channel delivery
        """

    async def broadcast(self, text: str, channels: list[str] | None = None) -> None:
        """Send a message to all (or specified) channels. For proactive nudges."""

    def register_channel(self, name: str, adapter: ChannelAdapter) -> None:
        """Register a channel adapter."""

    def unregister_channel(self, name: str) -> None:
        """Remove a channel adapter."""
```

### Component 2: Channel Adapters (`animus_bootstrap.gateway.channels`)

**Base protocol:**
```python
class ChannelAdapter(Protocol):
    """Interface that all channel adapters must implement."""

    name: str                        # "telegram", "discord", etc.
    is_connected: bool               # Current connection status

    async def connect(self) -> None:
        """Establish connection to the platform."""

    async def disconnect(self) -> None:
        """Clean disconnect."""

    async def send_message(self, response: GatewayResponse) -> str:
        """Send a response. Returns platform message ID."""

    async def on_message(self, callback: MessageCallback) -> None:
        """Register callback for inbound messages."""

    async def health_check(self) -> ChannelHealth:
        """Check connection health."""
```

**Channel implementations:**

| Channel | Library | Auth Method | Complexity |
|---------|---------|-------------|------------|
| Telegram | python-telegram-bot | Bot token | Low — Port existing bot |
| Discord | discord.py | Bot token | Low — Port existing bot |
| Slack | slack-bolt | OAuth2 / Bot token | Medium — Upgrade to bidirectional |
| WhatsApp | whatsapp-web.js (via bridge) or Baileys | QR code scan | High — Needs JS bridge or Business API |
| Signal | signal-cli (subprocess) | Phone number + captcha | High — CLI wrapper |
| Matrix | matrix-nio | Access token | Medium — Async native |
| Google Chat | google-auth + httpx | Service account | Medium — Webhook + API |
| WebChat | WebSocket (built-in) | Session token | Low — FastAPI WebSocket |
| Email | aiosmtplib + aioimaplib | IMAP/SMTP credentials | Medium — Polling loop |

**Priority order (build sequence):**
1. **WebChat** — Zero external deps, test the gateway locally
2. **Telegram** — Port existing Core bot, most users have it
3. **Discord** — Port existing Core bot
4. **Slack** — Upgrade notification channel to bidirectional
5. **Matrix** — Open protocol, good community
6. **WhatsApp** — High demand, complex auth
7. **Signal** — Privacy-focused users
8. **Google Chat** — Enterprise users
9. **Email** — Universal fallback

### Component 3: Session Manager (`animus_bootstrap.gateway.session`)

**Cross-channel conversation tracking:**

```python
class Session:
    """A conversation session that can span multiple channels."""
    id: str                          # Session UUID
    user_id: str                     # Canonical user ID (maps platform IDs)
    user_name: str                   # Display name
    messages: list[GatewayMessage]   # Full conversation history
    channel_ids: dict[str, str]      # channel_name -> platform_user_id
    created_at: datetime
    last_active: datetime
    context_tokens: int              # Current context window usage
    metadata: dict[str, Any]

class SessionManager:
    """Manages user sessions across channels."""

    async def get_or_create_session(self, message: GatewayMessage) -> Session:
        """Find existing session by sender, or create new one."""

    async def link_channel(self, session_id: str, channel: str, platform_id: str) -> None:
        """Link a new channel identity to an existing session.
        Enables: same user on Telegram and Discord = one session."""

    async def get_context(self, session: Session, max_tokens: int) -> list[dict]:
        """Build LLM context from session history (recent messages + memory)."""

    async def prune_old_sessions(self, max_age_days: int = 30) -> int:
        """Clean up inactive sessions."""
```

**Storage:** SQLite table in the existing Animus data directory.
```sql
CREATE TABLE gateway_sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    user_name TEXT,
    created_at TEXT NOT NULL,
    last_active TEXT NOT NULL,
    context_tokens INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{}'
);

CREATE TABLE gateway_messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES gateway_sessions(id),
    channel TEXT NOT NULL,
    channel_message_id TEXT,
    sender_id TEXT NOT NULL,
    sender_name TEXT,
    text TEXT NOT NULL,
    role TEXT NOT NULL,  -- 'user' | 'assistant'
    timestamp TEXT NOT NULL,
    metadata TEXT DEFAULT '{}'
);

CREATE TABLE channel_identities (
    session_id TEXT NOT NULL REFERENCES gateway_sessions(id),
    channel TEXT NOT NULL,
    platform_user_id TEXT NOT NULL,
    linked_at TEXT NOT NULL,
    PRIMARY KEY (channel, platform_user_id)
);
```

### Component 4: Cognitive Backend (`animus_bootstrap.gateway.cognitive`)

**Bridge to Animus Core's LLM layer:**

```python
class CognitiveBackend(Protocol):
    """Interface for LLM processing."""

    async def generate_response(
        self,
        messages: list[dict],        # OpenAI-format messages
        system_prompt: str | None,
        max_tokens: int,
    ) -> str: ...

class AnthropicBackend(CognitiveBackend):
    """Direct Anthropic API calls (standalone mode)."""

class ForgeBackend(CognitiveBackend):
    """Route through Animus Forge API (orchestrated mode)."""

class OllamaBackend(CognitiveBackend):
    """Local Ollama for offline operation."""
```

### Component 5: Dashboard Extensions

**New pages added to the Phase 1 dashboard:**

**/conversations** — Live message feed
- All recent messages across all channels, newest first
- Channel indicator (icon + color) per message
- Reply from dashboard (sends through originating channel)
- Search messages
- Click message to see full session context

**/channels** — Channel management
- Connected channels with health indicators
- Connect new channel (opens setup flow for that channel)
- Disconnect/reconnect buttons
- Per-channel message count and last activity
- Channel-specific settings (e.g., which Discord guilds)

### Component 6: Wizard Extension

**New wizard step (Step 4.5 — between Forge and Memory):**

```
Step 4.5: Channels
  → "Which channels do you want to connect?"
  → Show available channels with status:
    ✅ WebChat (always available — built in)
    ⬜ Telegram (requires bot token)
    ⬜ Discord (requires bot token)
    ⬜ Slack (requires OAuth or bot token)
    ⬜ WhatsApp (requires QR code scan)
    ⬜ Signal (requires phone number)
    ⬜ Matrix (requires homeserver + token)
  → For each selected: run channel-specific setup
  → Test each connection before saving
  → "You can add more channels later in the dashboard"
```

---

## PROJECT STRUCTURE

```
src/animus_bootstrap/
├── gateway/
│   ├── __init__.py
│   ├── router.py              ← MessageRouter: normalize, route, respond
│   ├── models.py              ← GatewayMessage, GatewayResponse, Attachment
│   ├── session.py             ← SessionManager: cross-channel sessions
│   ├── cognitive.py           ← CognitiveBackend: Anthropic/Forge/Ollama
│   ├── channels/
│   │   ├── __init__.py
│   │   ├── base.py            ← ChannelAdapter protocol
│   │   ├── webchat.py         ← WebSocket chat widget (built-in)
│   │   ├── telegram.py        ← Telegram Bot API
│   │   ├── discord.py         ← Discord.py
│   │   ├── slack.py           ← Slack Bolt SDK
│   │   ├── whatsapp.py        ← WhatsApp Business API / Baileys bridge
│   │   ├── signal.py          ← signal-cli wrapper
│   │   ├── matrix.py          ← matrix-nio
│   │   ├── google_chat.py     ← Google Chat API
│   │   └── email.py           ← IMAP/SMTP
│   └── middleware/
│       ├── __init__.py
│       ├── auth.py            ← Per-channel user verification
│       ├── ratelimit.py       ← Per-channel rate limits
│       └── logging.py         ← Message audit logging
│
├── dashboard/
│   ├── routers/
│   │   ├── conversations.py   ← NEW: /conversations page + API
│   │   └── channels.py        ← NEW: /channels page + API
│   └── templates/
│       ├── conversations.html ← NEW: message feed
│       └── channels.html      ← NEW: channel management
│
└── setup/
    └── steps/
        └── channels.py        ← NEW: channel onboarding wizard step
```

---

## ADDITIONAL DEPENDENCIES

```toml
[project.optional-dependencies]
telegram = ["python-telegram-bot>=22.0"]
discord = ["discord.py>=2.3"]
slack = ["slack-bolt>=1.18"]
matrix = ["matrix-nio>=0.24"]
signal = []  # Uses signal-cli subprocess
whatsapp = []  # Uses external bridge
email = ["aiosmtplib>=3.0", "aioimaplib>=1.0"]
all-channels = [
    "python-telegram-bot>=22.0",
    "discord.py>=2.3",
    "slack-bolt>=1.18",
    "matrix-nio>=0.24",
    "aiosmtplib>=3.0",
    "aioimaplib>=1.0",
]
```

---

## CONFIG EXTENSIONS

```toml
[gateway]
enabled = true
default_backend = "anthropic"  # anthropic | forge | ollama
system_prompt = ""             # Custom system prompt for chat
max_response_tokens = 4096

[gateway.channels.webchat]
enabled = true                 # Always available

[gateway.channels.telegram]
enabled = false
bot_token = ""

[gateway.channels.discord]
enabled = false
bot_token = ""
allowed_guilds = []

[gateway.channels.slack]
enabled = false
bot_token = ""
app_token = ""                 # For Socket Mode

[gateway.channels.whatsapp]
enabled = false
auth_method = "qr"             # qr | business_api
phone_number = ""

[gateway.channels.signal]
enabled = false
phone_number = ""

[gateway.channels.matrix]
enabled = false
homeserver = ""
access_token = ""
room_ids = []

[gateway.channels.email]
enabled = false
imap_host = ""
smtp_host = ""
username = ""
password = ""
poll_interval = 60
```

---

## BUILD SEQUENCE

### Phase 2a — Gateway Core (Prompts 09-11)

**Prompt 09 — Gateway Models + Router + Session Manager**
- `gateway/models.py` — GatewayMessage, GatewayResponse, Attachment dataclasses
- `gateway/router.py` — MessageRouter with handle_message, broadcast, register/unregister
- `gateway/session.py` — SessionManager with SQLite storage, cross-channel linking
- `gateway/cognitive.py` — CognitiveBackend protocol + AnthropicBackend + OllamaBackend
- Config schema extensions for `[gateway]` section
- Tests for all gateway core modules

**Prompt 10 — WebChat Channel + Dashboard Integration**
- `gateway/channels/base.py` — ChannelAdapter protocol
- `gateway/channels/webchat.py` — WebSocket-based browser chat
- `dashboard/routers/conversations.py` — Conversation feed page
- `dashboard/routers/channels.py` — Channel management page
- `dashboard/templates/conversations.html` — Message feed with channel indicators
- `dashboard/templates/channels.html` — Channel health + connect UI
- Wire WebSocket chat into dashboard sidebar
- Tests

**Prompt 11 — Telegram + Discord Ports**
- `gateway/channels/telegram.py` — Port from Core's telegram_bot.py
- `gateway/channels/discord.py` — Port from Core's discord_bot.py
- `setup/steps/channels.py` — Channel onboarding wizard step
- Update CLI: `animus-bootstrap channels` subcommand
- Tests

### Phase 2b — Extended Channels (Prompts 12-14)

**Prompt 12 — Slack + Matrix**
- `gateway/channels/slack.py` — Slack Bolt SDK, Socket Mode
- `gateway/channels/matrix.py` — matrix-nio async
- Tests

**Prompt 13 — WhatsApp + Signal**
- `gateway/channels/whatsapp.py` — WhatsApp Business API integration
- `gateway/channels/signal.py` — signal-cli subprocess wrapper
- Tests

**Prompt 14 — Gateway Middleware + Polish**
- `gateway/middleware/auth.py` — Per-channel user allowlists
- `gateway/middleware/ratelimit.py` — Token-bucket per channel per user
- `gateway/middleware/logging.py` — Audit logging of all messages
- Full integration test: message → gateway → LLM → response → channel
- Coverage target: 80%+

---

## DESIGN CONSTRAINTS

### Non-Negotiable (inherited from Phase 1)
- Local-first: messages stored locally, never synced to cloud
- No telemetry: message content never sent anywhere except the user's chosen LLM
- Single-user: one owner per instance
- Works offline: WebChat + Ollama = fully offline operation
- chmod 600 on config: API keys and bot tokens protected

### Gateway-Specific
- **Channel failures are non-blocking** — if Telegram is down, Discord still works
- **No message replay** — if a channel disconnects, missed messages are not retried
- **Graceful degradation** — gateway works with zero channels (dashboard-only mode)
- **Optional deps per channel** — base install includes only WebChat
- **Message retention** — configurable, default 30 days, user can export before pruning

### Performance
- Gateway handles messages sequentially per session (no parallel LLM calls for same user)
- Channel adapters run as independent async tasks
- WebSocket connections capped at 10 concurrent (local dashboard, not public)
- SQLite WAL mode for concurrent reads during message queries

---

## WHAT THIS ACHIEVES

After Phase 2, Animus has feature parity with OpenClaw's core experience:

| Feature | OpenClaw | Animus (Phase 1+2) |
|---------|----------|---------------------|
| One-command install | `npm install -g openclaw` | `pip install animus-bootstrap` |
| Onboarding wizard | `openclaw onboard` | `animus-bootstrap install` |
| Channel gateway | Yes (10+ channels) | Yes (9 channels) |
| WhatsApp | Yes | Yes |
| Telegram | Yes | Yes (ported from Core) |
| Discord | Yes | Yes (ported from Core) |
| Slack | Yes | Yes (upgraded to bidirectional) |
| Signal | Yes | Yes |
| Matrix | Yes | Yes |
| WebChat | Yes | Yes (built-in) |
| Cross-channel context | Yes | Yes (SessionManager) |
| Local dashboard | Yes | Yes (localhost:7700) |
| Conversation view | Yes | Yes (/conversations) |
| System service | Yes | Yes (systemd/launchd) |
| Offline operation | Partial | Full (Ollama + WebChat) |
| Multi-agent orchestration | No | Yes (Forge integration) |
| Coordination protocol | No | Yes (Quorum integration) |
| Memory system | Basic | Advanced (episodic/semantic/procedural) |

**Animus advantages over OpenClaw:**
- Python stack (no Node.js dependency)
- Multi-agent orchestration via Forge
- Coordination protocol via Quorum
- Advanced memory (ChromaDB vector search)
- Proactive engine (nudges, morning briefs)
- Full offline operation (Ollama)
- HTMX dashboard (no JS build step)

---

*Last updated: 2026-02-20*
*Depends on: Phase 1 (Bootstrap v0.1.0)*
*Target: Bootstrap v0.2.0*
