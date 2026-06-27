# Animus — Bootstrap, Interface & UX Vision

**Status:** Canonical assessment · Created 2026-06-14 · Owner: ARETE  
**Scope:** Bootstrap mechanism, every user-facing surface, honest UX audit, and the roadmap to make Animus feel like a true exocortex.

---

## 1. Bootstrap Mechanism

### 1.1 What it does today

The bootstrap layer (`packages/bootstrap/`) is a self-contained install-and-run system. Its job is to take a raw machine and produce a running Animus daemon with zero manual configuration.

**Flow:**

```
animus-bootstrap install
  ├─ check deps (python 3.11+, pip, ollama/ffmpeg optional)
  ├─ install missing required deps via system package manager
  ├─ register systemd / launchd / Windows service
  ├─ run 9-step Rich wizard (welcome → identity → API keys → forge → memory
  │   → device → sovereignty → channels)
  ├─ start service
  └─ open localhost:7700 dashboard
```

**Key components:**

| Component | File | Role |
|---|---|---|
| `AnimusInstaller` | `daemon/installer.py` | Detects OS/package manager, installs deps, registers services |
| `LinuxService` / `MacOSService` / `WindowsService` | `daemon/platforms/*.py` | Generates systemd units, launchd plists, or Windows services |
| `AnimusWizard` | `setup/wizard.py` | 9-step TUI onboarding |
| `AnimusRuntime` | `runtime.py` | Central orchestrator: boots identity, session manager, memory, tools, proactive engine, router, personas, gateway channels |
| Dashboard app | `dashboard/app.py` | FastAPI + Jinja2 + HTMX + TailwindCSS, port 7700 |
| `AnimusUpdater` | `daemon/updater.py` | Checks GitHub for newer versions, applies updates |
| Integrity gate | `daemon/__main__.py` | Refuses to boot if critical files drift (ed25519-signed manifest) |

### 1.2 Strengths

- **One-command install.** `animus-bootstrap install` is genuinely zero-config on a fresh Ubuntu/macOS machine.
- **Cross-platform services.** systemd (Linux), launchd (macOS), and Windows service wrappers are all implemented.
- **Integrity on boot.** The daemon verifies a cryptographic baseline before starting — tampered files → hard refusal with exit code 2/3.
- **gocryptfs vault integration.** Encryption at rest is wired: `ExecStartPre` mounts the vault, `ExecStopPost` unmounts. TPM-sealed passphrase via `systemd-creds`.
- **Auto-updater.** Version check against GitHub releases, with apply path.

### 1.3 Weaknesses / Gaps

| Gap | Severity | Detail |
|---|---|---|
| Wizard is terminal-only | Medium | Non-technical users hit a TUI before ever seeing a graphical interface. The first impression is a monospace wall of text. |
| No install-time health verification | Low | After install, the service may fail silently (e.g., Ollama not running, API key invalid). The CLI says "started" but the runtime may be in degraded mode. |
| Service logs are hard to reach | Low | `journalctl --user -u animus` is the debug path. There's no `animus-bootstrap logs` or in-dashboard log viewer with search/filter. |
| Updater is not transactional | Medium | Failed updates can leave the install in a half-updated state. No automatic rollback on update failure. |
| No install sandbox / container path | Low | The only install path is system-native. No Docker/podman path for ephemeral or isolated installs. |
| First-run identity generation is opaque | Low | The wizard asks for identity, but the resulting `CORE_VALUES.md` and identity files are invisible until the user knows to look in `~/.config/animus/identity/`. |

### 1.4 How the user interacts with bootstrap

**Primary paths:**

1. **First install:** `animus-bootstrap install` → wizard → browser opens.
2. **Daily ops:** `animus-bootstrap start | stop | restart | status`
3. **Config tuning:** `animus-bootstrap config get/set <key>`
4. **Channel mgmt:** `animus-bootstrap channels enable/disable <name>`
5. **Persona mgmt:** `animus-bootstrap personas add/list/delete/set-default`
6. **Feedback loop:** `animus-bootstrap feedback add up/down` → `animus-bootstrap reflect`
7. **Direct dashboard:** `animus-bootstrap dashboard` (skip service, run foreground)

**Observation:** Bootstrap is ops-first. It's excellent for the "install and administer" persona, but the *daily user* persona has no bootstrap interaction — they use the dashboard, PWA, or CLI.

---

## 2. Interface Inventory

### 2.1 Surface map

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE SURFACES                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  TERMINAL / CLI                                                             │
│  ├─ Core CLI          python -m animus          prompt-toolkit REPL       │
│  │                     40+ slash commands, rich panels, memory recall       │
│  │                     think_with_tools() + approval callback              │
│  ├─ Bootstrap CLI     animus-bootstrap <cmd>    typer + rich tables        │
│  │                     install, setup, start, stop, status, dashboard     │
│  │                     config, channels, tools, proactive, automations     │
│  │                     personas, feedback, reflect                           │
│  └─ Forge CLI         animus-forge <cmd>        typer, workflow runner      │
│                       run, eval, compare, self-improve                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  DESKTOP WEB                                                                  │
│  └─ Dashboard         http://localhost:7700     FastAPI + Jinja2 + HTMX     │
│                       20 pages: status, conversations, channels, config     │
│                       memory, logs, update, tools, automations, activity    │
│                       tasks, personas, routing, identity, proposals       │
│                       self-mod, forge, timers, feedback                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  MOBILE / PWA                                                                 │
│  └─ PWA               https://host (or localhost)  React 19 + Vite          │
│                       4 views: Chat, Capture, Status, Personas              │
│                       WebSocket chat, push notifications, voice input       │
│                       share_target capture, bearer-token auth               │
├─────────────────────────────────────────────────────────────────────────────┤
│  IDE / EDITOR                                                                 │
│  └─ MCP Server        python -m animus.mcp_server  10 tools exposed         │
│                       memory (remember/recall/search/stats)                  │
│                       tasks (list/create/complete), brief, run_workflow      │
│                       Configured in ~/.claude/mcp.json                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  MESSAGING / GATEWAY                                                          │
│  ├─ WebChat           WebSocket /ws/chat          Browser-based chat         │
│  ├─ Telegram          Bot token adapter          Bi-directional             │
│  ├─ Discord           Bot token adapter          Bi-directional             │
│  ├─ Slack             Bot token adapter          Bi-directional             │
│  ├─ Matrix            Client SDK adapter         Bi-directional             │
│  ├─ WhatsApp          Adapter stub               (not fully wired)          │
│  ├─ Signal            Adapter stub               (not fully wired)          │
│  └─ Email             IMAP/SMTP adapter          Bi-directional             │
├─────────────────────────────────────────────────────────────────────────────┤
│  VOICE                                                                        │
│  └─ Core Voice        Whisper STT + pyttsx3/edge-tts TTS                     │
│                       /voice and /speak CLI commands                         │
│                       PWA speech recognition (Web Speech API)               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Core CLI (`python -m animus`)

The original interface. A prompt-toolkit REPL with:

- **Natural-language default:** Type anything → routed to `think_with_tools()`
- **Slash commands:** `/help`, `/status`, `/stats`, `/memory`, `/tools`, `/voice`, `/speak`, `/calendar`, `/tasks`, `/build`, `/model`, `/auto`, `/quit`
- **Rich output:** Tables, panels, syntax-highlighted code
- **Memory recall:** Every query is enriched with episodic/semantic/procedural recall
- **Approval gate:** Sensitive tools prompt `Execute? [Y/n]` in terminal
- **Voice mode:** `/voice` activates continuous listening

**Verdict:** Excellent for power users and developers. Poor for everyone else. It's a terminal app — intimidating, no discoverability, no visual hierarchy.

### 2.3 Bootstrap Dashboard (`localhost:7700`)

A FastAPI server serving server-rendered HTML with HTMX for partial updates. Dark theme (Tailwind custom colors: `#0f0f0f` bg, `#00ff88` accent, mono fonts).

**Pages:**

| Page | Purpose | Quality |
|---|---|---|
| `/` (home) | Status cards: runtime, forge, memory, uptime | Functional, utilitarian |
| `/conversations` | Chat via WebSocket or HTMX polling | Basic message feed |
| `/channels` | Enable/disable messaging adapters | Table of toggles |
| `/config` | View/edit config values | Raw key-value editor |
| `/memory` | Browse/search memory | Minimal |
| `/logs` | System logs | Minimal |
| `/tools` | Registered tools + approval queue | List view |
| `/automations` | Trigger/condition/action rules | Minimal |
| `/activity` | Recent actions/events | Feed |
| `/tasks` | Task list | Minimal |
| `/personas` | Persona profiles | Basic table + forms |
| `/routing` | Channel → persona routing rules | Basic |
| `/identity` | Identity file viewer | Raw text display |
| `/proposals` | Identity change proposals (20% threshold) | Basic |
| `/self-mod` | Self-improvement status | Minimal |
| `/forge` | Forge workflow status/link | Redirect hint |
| `/timers` | Scheduled timers | Minimal |
| `/feedback` | Feedback entries + stats | Table |

**Verdict:** An ops dashboard masquerading as a user interface. It has *everything* but the kitchen sink, which means daily tasks are buried under admin cruft. The chat page exists but is not the default landing. The visual design is consistent (dark cyber) but rigid — no customization, no density modes, no collapsible sections.

### 2.4 PWA (`packages/pwa/`)

React 19 + Vite + TypeScript. Built as a true PWA with service worker, manifest, offline support, push notifications, and `share_target`.

**Views:**

| View | Features |
|---|---|
| **Chat** | WebSocket real-time, message history, voice input (Web Speech API), offline queuing, "Thinking..." indicator |
| **Capture** | Quick text entry → stored to memory, share_target integration (receive from other apps) |
| **Status** | Health polling, component list, push toggle, connection state |
| **Personas** | List personas, activate/deactivate |
| **Login** | Bearer token entry, validated against `/api/health` |

**Architecture strengths:**
- WebSocket with offline queuing (messages buffered until reconnect)
- Voice input via browser SpeechRecognition
- Push subscription via VAPID
- Share target for rapid capture from any mobile app
- Responsive CSS with safe-area insets for notched phones

**Verdict:** The most "modern" interface, but it's *thin*. Four views is not enough for a daily exocortex. No memory browsing, no tool invocation UI, no calendar view, no task management, no workflow trigger, no decision support, no settings. It feels like a chat app with a capture button, not a cognitive layer.

### 2.5 MCP Server

Invisible to the eye, but critical. Provides 10 tools to Claude Code:

- `animus_remember`, `animus_recall`, `animus_search_tags`, `animus_memory_stats`
- `animus_list_tasks`, `animus_create_task`, `animus_complete_task`
- `animus_brief` (context briefing)
- `animus_run_workflow` (trigger Forge pipelines)

**Verdict:** Correctly designed. The integration is ambient — Claude Code sessions automatically have Animus memory without opening a separate window. This is the *closest* the system gets to true exocortex behavior: present without being opened.

### 2.6 Gateway Channels

Eight adapters. In practice, WebChat (browser), Telegram, Discord, and Email are the most likely to be used. WhatsApp and Signal are stubs.

**Interaction model:** Message-in → `IntelligentRouter` → cognitive backend → response-out.

**Verdict:** Powerful but under-designed from a UX standpoint. Each channel is a *pipe*, not a *surface*. There's no channel-native UI optimization (e.g., Discord embeds for memory cards, Telegram inline keyboards for approvals). The response is always plain text.

---

## 3. UX / UI Audit — Honest Scores

Scored 1–10. A 10 means "best in class, rivals commercial products." A 5 means "functional, not delightful." Below 5 means "friction blocks usage."

| Dimension | Score | Notes |
|---|---|---|
| **Visual Coherence** | 4 | Three independent stacks (Rich/CLI, Tailwind/HTMX, React/CSS). Colors, typography, spacing, and motion do not translate across surfaces. No shared design system or component library. |
| **Information Architecture** | 5 | Dashboard has 20 pages with no hierarchy — everything is equally prominent in the sidebar. No "simple mode." PWA has the opposite problem: too few entry points. CLI has 40+ commands with no command palette or fuzzy search. |
| **Mobile Experience** | 5 | PWA is the only mobile path. It's installable, offline-capable, and has push. But it's missing too many features to be a daily driver. Dashboard is unusable on phone (sidebar, tables, small text). |
| **Desktop Experience** | 5 | Dashboard is desktop-optimized but admin-heavy. No desktop native wrapper (Electron/Tauri), so no system tray, global shortcut, or native notifications. |
| **Chat / Conversation** | 6 | Functional across CLI, dashboard, PWA, and channels. But: no markdown rendering in PWA, no code highlighting, no file attachments, no branching threads, no inline tool results, no message editing. "Thinking..." is the only status indicator. |
| **Voice Interaction** | 5 | Core has Whisper + TTS. PWA has Web Speech API input. But no always-listening mode, no wake word, no voice-first UI (everything else requires screen). |
| **Memory Visualization** | 3 | Memory is text-only everywhere. No graph view, no timeline, no spatial organization, no "related memories" preview. The user cannot *see* what Animus knows about them. |
| **Onboarding** | 5 | Wizard covers all config but is TUI-only and feels like a questionnaire, not a conversation. No progressive disclosure — asks for API keys before explaining why. No interactive tutorial after install. |
| **Proactive / Ambient** | 4 | Proactive engine exists (6 checks). Quiet hours are configurable. But the UX is notification-only — there's no ambient HUD, no glanceable status widget, no lock-screen presence. The user must open an app to see what Animus is thinking. |
| **Accessibility** | 4 | PWA has basic ARIA labels and alt text. Dashboard uses semantic HTML but no focus management or skip links. CLI is inaccessible to screen readers (Rich panels are visual-only). No colorblind-friendly palettes. |
| **Performance & Reliability** | 7 | Fast boot, WebSocket reconnect with backoff, offline queuing in PWA. But no skeleton screens, no optimistic UI, no perceptible instant states. Dashboard HTMX swaps can feel sluggish without loading indicators. |
| **Personalization** | 6 | Persona engine is robust (profiles, voice tones, domains, channel routing). But no UI for the user to *tune* personality on the fly (sliders for formality, verbosity, initiative). No visual themes beyond dark mode. |

**Aggregate UX Score: ~5/10.**  
Functional, multi-surface, architecturally sound — but not cohesive, not ambient, and not emotionally resonant. It still feels like a *tool collection*, not a *cognitive layer*.

---

## 4. Interaction Model Analysis

### 4.1 The exocortex concept

An exocortex is not an app you open. It is a layer that:

1. **Surrounds** you — present on all devices, always available, ambient.
2. **Remembers** — accumulates context across years, surfaces it without being asked.
3. **Acts** — takes initiative within guardrails, proposes rather than just responds.
4. **Adapts** — learns your patterns, preferences, and style.
5. **Speaks your language** — communicates in the register, channel, and timing you prefer.

### 4.2 Current model: "App you launch"

Today's Animus is **pull-oriented**:

- User opens CLI / dashboard / PWA / messaging app
- User asks a question or gives a command
- Animus responds
- User closes the app

Even the proactive engine fits this model: it runs checks in the background, then *pushes a notification* — which the user pulls open to read.

This is fundamentally a **messaging app** paradigm. It's not wrong, but it's insufficient for an exocortex.

### 4.3 Target model: "Layer that surrounds"

The shift required:

| From | To |
|---|---|
| Open app → ask → get answer | Need arises → Animus surfaces context *before* you ask |
| Single-turn chat | Continuous, threaded, branched conversations with persistence |
| Notification = interruption | Ambient suggestion = glanceable, dismissible, ranked by urgency |
| Memory is a database you query | Memory is a landscape you explore and prune |
| Persona is a config file | Persona is a relationship that evolves |
| Tool use is hidden | Tool use is transparent and collaborative |
| One interface per device | Context follows you; handoff is seamless |

---

## 5. Vision: "True to Concept"

What does 10/10 look like for an Animus interface?

### 5.1 Presence — Always there, never in the way

- **System tray / menu bar widget:** A small orb or glyph that pulses when Animus is thinking. Hover shows recent thoughts. Click expands to quick capture or command palette.
- **Lock screen / notification shade:** Morning digest, upcoming tasks, anomalies detected, briefings ready.
- **Ambient voice:** Wake word activation ("Hey Animus") on desktop and mobile. Whisper runs locally so audio never leaves the device.
- **Wearable bridge:** Apple Watch complication / WearOS tile showing memory count, pending tasks, and a quick-voice-capture button.

### 5.2 Memory as a landscape

- **Memory graph:** A visual network of people, projects, concepts, and events. Zoom in to see details, zoom out for patterns. Filter by time, type, confidence.
- **Timeline / lifelog:** Horizontal scroll through your history. Click a day to see what you discussed, decided, and captured.
- **Knowledge domains:** Radars or treemaps showing which domains Animus knows you well in (code, health, relationships, finance) and where knowledge is sparse.
- **Memory gardening UI:** Review, strengthen, merge, or prune memories with swipe gestures. "This is important / This is outdated / This is wrong."

### 5.3 Conversation as collaboration

- **Branched threads:** Conversations fork. "Explore this idea further" branches without losing the main thread.
- **Inline artifacts:** Code blocks, charts, decision matrices, file diffs rendered natively — not pasted as text.
- **Tool transparency:** When Animus uses a tool, the user sees *what* it did, *why*, and can edit/retry/undo. Not a black box.
- **Co-editing:** User and Animus edit the same document simultaneously. Changes are attributed and reversible.

### 5.4 Initiative without intrusion

- **Urgency ranking:** The proactive engine assigns a score (1–10) to every suggestion. 8+ = notification. 4–7 = badge/dot on the tray icon. 1–3 = logged silently, surfaced in daily digest.
- **User teaches interruption preferences:** "Only interrupt me for calendar conflicts and security alerts." Learns from dismissals.
- **Briefings, not notifications:** Morning briefing is a structured page (agenda, risks, opportunities, memory highlights), not a pile of alerts.

### 5.5 Cross-device handoff

- **Session continuity:** Start a thought on phone, finish on laptop. The context (draft message, half-written code, open research tabs) transfers automatically.
- **Awareness of which device:** "You're on your phone — I'll keep this brief." vs. "You're at your desk — here's the full analysis with charts."
- **Conflict resolution UI:** When two devices edit the same memory, a visual diff-merge tool appears. Not silent last-write-wins.

---

## 6. Roadmap: From 5/10 to 10/10

**Dependency rule:** Phases are ordered. Do not start Phase N until Phase N‑1 exits. Within a phase, items are parallelizable unless marked sequential.

**Pace:** Personal project pace — weeks, not days. Each "session" is a focused block with a test or check as its done criterion.

---

### Phase I — Design Foundation (weeks 1–4)
*Unify the visual language and build the shared substrate.*

| ID | Item | Acceptance | Effort |
|---|---|---|---|
| I1 | **Design system spec** — colors, typography, spacing, motion, elevation, iconography. Covers all three stacks (CLI/Rich, Dashboard/HTMX, PWA/React). Documented in `docs/DESIGN_SYSTEM.md`. | Three surfaces use the same tokens. A single source-of-truth CSS/JSON file is imported by dashboard and PWA; Rich styles reference the same hex codes. | S |
| I2 | **Component library** — shared HTML/React components for cards, buttons, inputs, badges, tables, modals, toasts, skeleton screens. Start with 10 primitives. | Components are in `packages/ui/` (web components or React). Used by both dashboard and PWA. Visual regressions caught by a single screenshot diff test. | M |
| I3 | **Iconography set** — 50 custom SVG icons for Animus concepts (memory, forge, quorum, persona, automation, etc.). Replace emoji in PWA nav. | No emoji in production UI. Icons are accessible (aria-label, currentColor fill). | S |
| I4 | **Motion & feedback spec** — loading states, transitions, hover/active micro-interactions, optimistic UI patterns. No jarring full-page reloads. | HTMX swaps fade/slide. Buttons show press states. Forms show skeleton screens while loading. Documented in design system. | S |
| I5 | **Accessibility baseline** — WCAG 2.1 AA audit for PWA and dashboard. Focus traps, skip links, color contrast ≥ 4.5:1, reduced-motion support. | Lighthouse accessibility score ≥ 90 for PWA. axe-core automated scan passes for dashboard critical paths. | M |

**Exit:** All surfaces look like they belong to the same product. A user moving from PWA to dashboard to CLI feels continuity, not whiplash.

---

### Phase II — PWA: The Daily Driver (weeks 3–8)
*Overlap with Phase I starting week 3. Make the PWA the primary mobile and desktop daily interface.*

| ID | Item | Acceptance | Effort |
|---|---|---|---|
| II1 | **Chat overhaul** — markdown rendering with syntax highlighting, inline code execution preview, file attachment (photos/docs → memory), message editing, branching threads, search within conversation. | Rendered markdown matches GitHub quality. Images display inline. Thread branches visually distinct. | L |
| II2 | **Memory browser** — full-text search, tag filter, type filter, sort (recent/relevant/confidence). Card-based results with expand-for-detail. Swipe actions (pin/archive/delete). | User can find any memory in < 3 taps from home. Search returns in < 500ms. | M |
| II3 | **Timeline / lifelog** — daily digest view, horizontal scroll, event clustering ("busy morning," "deep work block"). Tap to expand into conversation fragments and captures. | 30 days visible without scroll lag. Events grouped by AI-inferred context switches. | M |
| II4 | **Tasks & calendar** — task list with priorities, due dates, check-off. Calendar view (day/week) showing Animus-scheduled blocks and external calendar integrations. | Tasks sync with Core's TaskTracker. Calendar fetches from Google Calendar integration. | M |
| II5 | **Tool invocation UI** — when Animus wants to use a tool, show a card with params, allow edit-and-approve inline. Post-execution, show result summary with "undo." | Tool approval feels like Siri Shortcuts — clear, cancellable, editable. | M |
| II6 | **Capture v2** — voice-first capture (hold mic button), photo capture with OCR, location tag, mood tag. Automatic categorization suggestion. | 3-second voice note → transcribed → categorized → stored. Photo text extracted on-device. | M |
| II7 | **Settings & onboarding** — graphical onboarding (4-slide welcome + permission requests), settings pages for identity, personas, channels, privacy. No terminal required for first setup. | New user installs PWA, completes setup without opening a terminal. Wizard is still available for advanced config. | M |
| II8 | **Desktop PWA wrapper** — install as standalone app (already supported), but add global shortcut (Cmd/Ctrl+Shift+A) to summon quick-capture overlay. | Global hotkey opens a floating input bar (like Spotlight/Albert) anywhere on desktop. | S |

**Exit:** The PWA is good enough to be the only interface for 80% of daily interactions. Users check it 10+ times a day.

---

### Phase III — Dashboard Redesign (weeks 6–10)
*Overlap with Phase II starting week 6. Transform the dashboard from ops panel to mission control.*

| ID | Item | Acceptance | Effort |
|---|---|---|---|
| III1 | **Landing = Command Center** — default page is a customizable grid of widgets: chat preview, memory highlights, upcoming tasks, proactive suggestions, system health mini-cards. | User can drag to reorder widgets. Layout persists per-user. | M |
| III2 | **Conversation-first navigation** — chat is a persistent sidebar or floating panel, not a buried page. All other pages are reachable without losing chat context. | Split-pane layout: chat on left, selected module on right (like Slack or Discord). | L |
| III3 | **Memory graph page** — D3.js or Cytoscape force-directed graph of memories. Click to focus, double-click to open. Filter by domain, time, confidence. | Renders 500 nodes at 60fps. Zoom/pan smooth. | L |
| III4 | **Workflow canvas** — visual YAML editor for Forge workflows. Drag-and-drop nodes (agents, gates, checkpoints). Live validation and dry-run trigger. | Non-technical users can build a 3-step workflow without reading docs. | L |
| III5 | **Decision support UI** — decision framework rendered as pros/cons matrix, confidence sliders, outcome probability bars. Export to PDF/markdown. | A decision page looks like a consulting deliverable, not a form. | M |
| III6 | **Activity feed v2** — filterable, searchable event log with timeline scrubber. Group related events. Link to source (memory, task, workflow run). | User can trace "why did Animus suggest X?" in 2 clicks. | M |
| III7 | **Density modes** — compact (ops), comfortable (default), spacious (presentations). Font scale independent. | Three density modes switch instantly, no reload. | S |

**Exit:** The dashboard is where power users *live*. It competes with Notion or Obsidian in terms of information density and utility.

---

### Phase IV — Ambient & Presence (weeks 9–14)
*Make Animus present without being opened.*

| ID | Item | Acceptance | Effort |
|---|---|---|---|
| IV1 | **System tray / menu bar agent** — lightweight native wrapper (Tauri or Electron shell) around the PWA. Shows status orb, recent thoughts, quick capture, command palette. | Runs on macOS, Linux (AppIndicator), Windows. < 50MB RAM. | M |
| IV2 | **Global quick-capture** — system-wide hotkey summons a floating input (text or voice). Captures go straight to memory, no full app open. | Latency < 200ms from hotkey to ready-to-type. | M |
| IV3 | **Native notifications** — macOS/Windows/Linux native notification APIs (not just web push). Rich actions: "Approve," "Dismiss," "Remind me in 10 min." | Notifications look native, with icons and action buttons. | S |
| IV4 | **Morning briefing** — daily automated briefing generated by proactive engine. Delivered as a native notification/page at user-defined time. Structured: agenda, risks, memory highlights, tasks due. | User glances at briefing for 30 seconds and knows their day. | M |
| IV5 | **Ambient voice (desktop)** — Always-listening wake word ("Hey Animus") using local Porcupine or Whisper.js. No cloud audio. | False positive rate < 1/day. Latency < 800ms from wake to ready. | M |
| IV6 | **Wearable bridge** — Apple Watch app / WearOS tile. Complications: memory count, pending tasks, quick voice capture. Syncs via the PWA backend. | Watch can capture voice and receive brief text replies without phone open. | L |
| IV7 | **Vehicle mode** — CarPlay/Android Auto simplified UI. Voice-primary. Location-aware context. Driving-safe (large text, minimal visual). | Driver can interact eyes-free. Maps integration for ETA-aware suggestions. | L |

**Exit:** Animus feels *around* the user, not *in an app*. The average interaction time drops to < 10 seconds because most context is surfaced proactively.

---

### Phase V — Cross-Device Continuity (weeks 13–18)
*Context follows the user.*

| ID | Item | Acceptance | Effort |
|---|---|---|---|
| V1 | **Session sync protocol** — every device syncs `SyncableState` (conversations, drafts, open research, active tasks) via the Core sync layer. Encrypted, conflict-aware. | Start on phone → open laptop → conversation is there, cursor position restored. | L |
| V2 | **Handoff UI** — visual indicator of which device has "focus." Push current context to another device with one tap. | "Continue on laptop" button in PWA moves session state. | S |
| V3 | **Conflict resolution UI** — when two devices diverge (e.g., edited same task), show diff-merge tool. User picks winner or merges. | Conflict is rare (< 1% of syncs) but handled gracefully when it occurs. | M |
| V4 | **Device-aware behavior** — Animus adapts output length and modality based on device capabilities and user context (phone = brief, desktop = detailed, car = audio-only). | Automatic; no manual mode switching required. | M |

**Exit:** Users forget which device they started on. The boundary between phone, laptop, and watch disappears for Animbus interactions.

---

### Phase VI — Emotional Resonance (weeks 17–22)
*The polish that makes it feel like a companion, not software.*

| ID | Item | Acceptance | Effort |
|---|---|---|---|
| VI1 | **Sound design** — custom audio identity: notification chimes (gentle, non-jarring), voice activation tone, error sounds. No stock system sounds. | 5-second soundscape test: a user hears a chime and knows "that's Animus, and it's low urgency." | S |
| VI2 | **Haptic language** — PWA and watch use distinct haptic patterns for urgency levels, confirmations, and errors. | 3 patterns, distinguishable eyes-free. | S |
| VI3 | **Animation system** — spring physics for UI transitions, typing indicators that feel alive, memory graph particles that settle organically. | 60fps on mid-range phone. Feels physical, not mechanical. | M |
| VI4 | **Persona warmth** — Persona engine gains emotional axes: warmth, humor, directness, enthusiasm. Visual avatar or glyph per persona. | Switching personas changes not just text but tone, timing, and emoji/style. | M |
| VI5 | **Relationship memory** — Animus remembers how the user likes to interact ("you prefer short answers in the morning"). Surface this in a "Our Relationship" page. | User sees a dashboard of "Animus knows you prefer X. Correct?" | S |
| VI6 | **Onboarding as friendship** — First-run experience is a conversation, not a form. Animus asks about goals, learns name/pronouns/preferences through dialogue. | User smiles during setup. Feels like meeting someone, not configuring software. | M |

**Exit:** Users describe Animus as "my assistant" not "this app." Net Promoter Score proxy: user spontaneously recommends it to friends.

---

### Phase VII — Bootstrap Hardening (continuous / opportunistic)

| ID | Item | Acceptance | Effort |
|---|---|---|---|
| VII1 | **Graphical onboarding alternative** — PWA-based first setup that writes the same config the wizard produces. Terminal wizard becomes "advanced mode." | Non-technical users never see a terminal during install. | M |
| VII2 | **Install health verification** — after `animus-bootstrap install`, run a smoke test (API key valid? Ollama reachable? Memory DB writable?) and report pass/fail with remediation links. | Install exits with clear red/yellow/green status, not just "done." | S |
| VII3 | **Transactional updater** — updates are atomic: download to temp, verify checksum, swap, rollback on failure. | Failed update auto-rolls back to prior version. No half-states. | M |
| VII4 | **Docker path** — `docker run -v animus-data:/data aretedriver/animus` as a fully supported install path. | Single-command Docker start with persistent volume. | S |
| VII5 | **Bootstrap dashboard as PWA** — The dashboard itself is served as a Progressive Web App (service worker, manifest), so desktop users can "install" it without a separate build. | Dashboard is installable from Chrome/Edge "Install as app." | S |

---

## 7. Success Metrics (10/10 Definition)

How do we know we've arrived?

| Metric | Target | Measurement |
|---|---|---|
| Daily active sessions (any surface) | ≥ 1 per waking hour | Log aggregated, privacy-preserving |
| Average interaction time | ≤ 15 seconds | Unless explicitly in deep-work mode |
| Proactive suggestion acceptance rate | ≥ 40% | User acts on or approves surfaced suggestions |
| Cross-device session continuity | ≥ 90% | Context successfully handoff without user manually transferring |
| User-reported "exocortex feeling" | ≥ 4.0 / 5.0 | Quarterly subjective survey (1 = "just an app", 5 = "part of my mind") |
| Accessibility (Lighthouse) | ≥ 95 | PWA and dashboard |
| Visual consistency score | ≥ 9/10 | Third-party blind comparison: "do these screens belong to the same app?" |
| Onboarding completion rate | ≥ 80% | Of users who start install, finish first meaningful interaction |
| Voice interaction accuracy | ≥ 95% | Whisper STT word-error rate on user voice |
| Boot-to-ready time | ≤ 3 seconds | From `animus-bootstrap start` to first responsive chat message |

---

## 8. Immediate Next Steps (What to do Monday)

1. **Write the Design System spec** (`docs/DESIGN_SYSTEM.md`). Colors, type, spacing, motion. Single source of truth.
2. **Audit PWA CSS** against the spec. Replace emoji nav with SVG icons. Unify border-radius, shadow, and transition tokens.
3. **Add global shortcut support** to the Bootstrap dashboard (a small Electron/Tauri shim or a desktop-native hotkey daemon). This is the fastest path to "ambient."
4. **Build the Memory Browser view** in the PWA. It's the highest-impact missing feature for daily usage.
5. **Create a `packages/ui/` shared component package** with Storybook (or Ladle) so dashboard and PWA share the same button, card, and input components.

---

## 9. Related Documents

- `docs/ROADMAP_TO_10.md` — Security, cost, eval, and correctness remediation (orthogonal to UX)
- `docs/ARCHITECTURE.md` — System architecture overview
- `docs/CONSTITUTIONAL_PRINCIPLES.md` — P1-P9 behavioral constraints
- `packages/pwa/` — Current PWA implementation
- `packages/bootstrap/src/animus_bootstrap/dashboard/` — Current dashboard implementation
- `packages/core/animus/__main__.py` — Core CLI entry point

---

*End of document. Updated 2026-06-14. Canonical — when this changes, update the cross-references in ROADMAP_TO_10.md and CANON.md.*
