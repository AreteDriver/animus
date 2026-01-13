# Connectivity & Interfaces

How Animus connects across devices, vehicles, and contexts. The core requirement: **seamless handoff with zero context loss**.

---

## Design Philosophy

**You shouldn't think about which device you're on.** You should think about what you're trying to do. Animus handles the translation between your intent and the interface available.

**The AI follows you, not the other way around.** You don't go to where Animus is - Animus is wherever you are.

**Minimal friction, maximum utility.** Each interface is optimized for its context. Desktop for complex work, voice for hands-busy, wearable for ambient awareness.

---

## Interface Modes

### Desktop / Laptop

**Primary use:** Complex work, long-form interaction, file management

**Characteristics:**
- Full keyboard and screen
- Multi-window capability
- Extended sessions
- Deep work support

**Interface options:**
- Native application (Electron, Tauri)
- Web application (PWA)
- Terminal/CLI for power users
- System tray for quick access

**Unique capabilities:**
- File system deep integration
- Multi-document context
- Code/technical work support
- Full memory browsing and management

---

### Mobile (Phone/Tablet)

**Primary use:** On-the-go queries, quick capture, voice interaction

**Characteristics:**
- Touch-first interface
- Variable attention (full screen to glance)
- Voice as primary input option
- Notification integration

**Interface options:**
- Native app (iOS/Android)
- PWA for cross-platform
- Widget for quick access
- Share sheet integration

**Unique capabilities:**
- Location awareness
- Camera integration (document capture, visual queries)
- Push notifications
- Quick capture without unlocking (voice)

---

### Wearable

**Primary use:** Ambient awareness, quick queries, notifications

**Characteristics:**
- Extremely limited screen (or none)
- Voice-primary interaction
- Haptic feedback
- Always-present

**Form factors:**

| Type | Pros | Cons |
|------|------|------|
| Smart Ring | Always on, unobtrusive | Very limited I/O |
| Smart Watch | Screen + voice, established | Requires charging, visible |
| Earbuds | Audio-only, natural | No visual, battery life |
| Pendant/Pin | More capability | Visible, social acceptance |

**Unique capabilities:**
- Biometric context (heart rate, activity)
- Truly hands-free operation
- Proactive notifications based on context
- Social-acceptable always-on presence

---

### Vehicle Integration

**Primary use:** Driving-time productivity, navigation context, hands-free everything

**Characteristics:**
- Voice-only interaction (safety)
- Large display available but minimal use
- Location/navigation context
- Time-bounded sessions (trips)

**Integration methods:**

**CarPlay / Android Auto:**
- Standard integration path
- Limited customization
- Works with existing vehicles
- Voice-primary with minimal visuals

**Direct OBD-II / Vehicle API:**
- Deeper integration possible
- Vehicle data access (fuel, diagnostics)
- More complex implementation
- Vehicle-specific development

**Bluetooth Audio + Phone:**
- Simplest approach
- Works with any vehicle
- Voice through car speakers
- Phone provides processing

**Vehicle mode features:**
- Pre-trip briefing (destination context, calendar, prep)
- Hands-free communication drafting
- Traffic-aware schedule management
- Podcast/content recommendations for trip length
- Arrival preparation (parking, context for destination)

**Safety constraints:**
- No visual content requiring attention
- Voice interactions designed to be brief
- Non-urgent notifications queued
- Driver can always say "not now"

---

### Storage Device Mode

**Primary use:** Access files on non-personal devices, portable identity

**Characteristics:**
- Physical hardware you carry
- Works on devices you don't own
- Secure, temporary access
- No trace left behind

**How it works:**

```
┌─────────────────────────────────────────────────────────────┐
│                      Host Device                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Temporary Interface                     │   │
│  │  (runs from Animus hardware, sandboxed)             │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ▲                                  │
│                          │                                  │
│  ┌───────────────────────┴───────────────────────────┐     │
│  │                  USB / Wireless                    │     │
│  └───────────────────────────────────────────────────┘     │
│                          ▲                                  │
└──────────────────────────┼──────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────┐
│                   Animus Hardware                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Encrypted  │  │  Identity   │  │  Portable Runtime   │  │
│  │  Storage    │  │  & Auth     │  │  (for temp UI)      │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Security model:**
- Hardware encryption, key never leaves device
- Biometric or PIN authentication
- Timeout auto-lock
- No data written to host device
- Audit log of access

**Use cases:**
- Access files at a friend's computer
- Work from a hotel business center
- Emergency access when your phone dies
- Share specific files without sharing device

---

## Handoff Protocol

The critical capability: **conversation and context continuity across devices**.

### Requirements

1. **Seamless** - User shouldn't have to "transfer" anything manually
2. **Fast** - Latency under 2 seconds on local network
3. **Secure** - All sync encrypted end-to-end
4. **Resilient** - Works offline, syncs when connected
5. **Conflict-free** - Handles simultaneous edits gracefully

### Architecture

```
┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│   Device A   │         │  Sync Layer  │         │   Device B   │
│              │         │              │         │              │
│ ┌──────────┐ │         │ ┌──────────┐ │         │ ┌──────────┐ │
│ │  Local   │ │◀───────▶│ │ Encrypted│ │◀───────▶│ │  Local   │ │
│ │   State  │ │         │ │   Store  │ │         │ │   State  │ │
│ └──────────┘ │         │ └──────────┘ │         │ └──────────┘ │
└──────────────┘         └──────────────┘         └──────────────┘
```

### Sync Methods

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| Local network (mDNS) | Fast, private, no cloud | Requires same network | Home/office use |
| Self-hosted relay | Full control | Requires server setup | Privacy-focused users |
| Syncthing | Proven, open source | Some setup required | Technical users |
| E2E encrypted cloud | Convenient, works anywhere | Third party involved | General use |

### Handoff Flow

```
1. User starts conversation on Device A
   └── State: active on A, synced to relay

2. User picks up Device B
   └── B detects user presence
   └── B requests current context from relay
   └── B displays: "Continue conversation about X?"

3. User confirms (or auto-continues based on settings)
   └── A receives "handoff initiated" signal
   └── A goes to background mode
   └── B becomes primary

4. Conversation continues on B
   └── All new context syncs to relay
   └── A remains available as secondary
```

### Conflict Resolution

When multiple devices have divergent state:

1. **Conversations:** Merge by timestamp, interleave if needed
2. **Facts/Memory:** Last-write-wins with history preserved
3. **Preferences:** Device-specific where relevant, sync global
4. **Files:** Version history, surface conflicts for user resolution

---

## Communication Protocol

### Real-Time Sync

**WebSocket-based protocol for low-latency sync:**

```
Client                           Server
   │                                │
   │──── AUTH (token) ─────────────▶│
   │◀─── AUTH_OK ──────────────────│
   │                                │
   │──── SUBSCRIBE (channels) ─────▶│
   │◀─── SUBSCRIBED ───────────────│
   │                                │
   │◀─── UPDATE (delta) ───────────│
   │──── ACK ──────────────────────▶│
   │                                │
   │──── PUSH (new_data) ──────────▶│
   │◀─── PUSH_OK ──────────────────│
```

### Message Types

| Type | Direction | Purpose |
|------|-----------|---------|
| AUTH | Client→Server | Authenticate connection |
| SUBSCRIBE | Client→Server | Register for updates |
| UPDATE | Server→Client | Push new data |
| PUSH | Client→Server | Send new data |
| PING/PONG | Both | Keep-alive |
| HANDOFF | Both | Device transition signal |

### Encryption

- TLS 1.3 for transport
- E2E encryption for payload (even on trusted networks)
- Per-device keys derived from master secret
- Key rotation on schedule or demand

---

## API Layer

For integrations, custom clients, and automation.

### REST API

```
Base: https://localhost:PORT/api/v1  (or remote if configured)

Endpoints:
  POST   /chat                    # Send message
  GET    /chat/history            # Retrieve history
  GET    /memory/search           # Query memory
  POST   /memory/add              # Add to memory
  GET    /context/current         # Get active context
  POST   /tools/{tool}/execute    # Run a tool
  GET    /status                  # System status
  POST   /sync/trigger            # Force sync
```

### Authentication

- API key for local access
- OAuth2 for remote access
- Device certificates for trusted devices
- Scoped permissions per client

### Webhooks

Animus can call external services on events:

```json
{
  "event": "reminder_triggered",
  "timestamp": "2024-01-15T09:00:00Z",
  "payload": {
    "reminder_id": "...",
    "title": "...",
    "context": {...}
  }
}
```

Configurable events:
- Conversation events
- Reminder triggers
- Calendar alerts
- Custom triggers based on context

---

## Offline Behavior

**Core principle:** Animus should be useful even without connectivity.

### What works offline

- Conversation with local model
- Memory retrieval (local cache)
- File access (synced files)
- All core cognitive functions

### What requires connectivity

- Web search
- Real-time sync between devices
- Cloud API access (if configured)
- External integrations

### Sync on reconnect

When connectivity returns:
1. Queue of outbound changes pushed
2. Inbound changes pulled and merged
3. Conflicts surfaced if any
4. Background sync resumes

---

## Future Connectivity

### Mesh Networking

Devices form local mesh for sync even without internet:
- Phone ↔ Laptop ↔ Wearable all sync directly
- Any device with internet shares connectivity
- Resilient to individual device disconnection

### Vehicle-to-Home Handoff

Arriving home, context transfers proactively:
- "You were working on X in the car - continue on desktop?"
- Calendar updates from trip reflected
- Notes captured while driving available immediately

### Multi-User Contexts

Shared Animus instances for family/team:
- Shared calendars and coordination
- Per-user privacy boundaries
- Collaborative memory (shared facts)
- Individual memory (personal only)

---

## Implementation Priority

### Phase 3 (Multi-Interface) - Must Have
- Desktop ↔ Mobile sync
- Basic voice integration
- Local network handoff

### Phase 4 (Integration) - Should Have
- Vehicle integration (CarPlay)
- Storage device mode (basic)
- API layer

### Phase 6 (Wearable) - Nice to Have
- Dedicated wearable support
- Full storage device mode
- Mesh networking
