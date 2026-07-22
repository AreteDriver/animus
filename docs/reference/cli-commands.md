# CLI Commands Reference

> Complete reference for all REPL commands in the Animus interactive shell.

---

## Overview

The Animus REPL accepts two kinds of input:

1. **Natural language** — Just type what you want. Animus will use its cognitive layer to understand and act.
2. **Slash commands** — Structured commands for precise control.

**Important note for local model users:** When using Ollama (local models), natural language mode does **not** execute tools automatically. Use `/tool <name>` for direct tool invocation. See [Working with Local Models](../operators/local-models.md) for details.

---

## Basic Commands

| Command | Description |
|---|---|
| `help` or `/help` | Show the help summary |
| `exit` | Quit Animus |
| `/status` | Show system status (provider, backend, memory count) |
| `/stats` | Show detailed memory statistics + task success rate |
| `/model` | Show current model info. **Core REPL:** dual-model routing (primary + fallback). **Head REPL:** swap installed local models. See [Head REPL Commands](head-repl.md) |
| `/auto` | Toggle auto-approve for tool execution |

### Examples

```
>>> /status
# Shows data directory, model provider, memory backend, total memories

>>> /model
# Primary: anthropic/claude-sonnet-4-20250514
# Local:   ollama/llama3:8b
# Dual-model routing active.

>>> /auto
# Toggles ANIMUS_AUTO_APPROVE environment variable
```

---

## Memory Commands

| Command | Description |
|---|---|
| `/remember <text>` | Store a semantic memory |
| `/recall <query>` | Search memories by semantic similarity |
| `/forget <id>` | Delete a memory by ID (or partial ID prefix) |
| `/review <id>` | Show full details of a single memory |
| `/list [type]` | List recent memories. Optional type: `semantic`, `episodic`, `procedural`, `active` |
| `/history` | Show recent conversation sessions |

### Examples

```
>>> /remember User prefers dark mode interfaces
# Remembered: a1b2c3d4

>>> /recall dark mode
# a1b2c3d4 [semantic] User prefers dark mode interfaces

>>> /review a1b2c3d4
# Full memory details: type, tags, confidence, created date, content, metadata

>>> /list semantic
# Lists semantic memories sorted by most recently updated
```

---

## Tagging Commands

| Command | Description |
|---|---|
| `/tag <id> <tag1> [tag2...]` | Add tags to a memory |
| `/untag <id> <tag>` | Remove a tag from a memory |
| `/tags` | List all tags with usage counts |
| `/search-tags <tag1> [tag2...]` | Find memories that have ALL specified tags |

### Examples

```
>>> /tag a1b2c3d4 preference ui
# Tagged a1b2c3d4 with: preference, ui

>>> /search-tags preference ui
# Shows memories tagged with both "preference" AND "ui"
```

---

## Structured Memory

| Command | Description |
|---|---|
| `/fact <subject> \| <predicate> \| <object>` | Store a structured semantic fact |
| `/procedure <name> \| <trigger> \| <step1>; <step2>...` | Store a learned workflow |

### Examples

```
>>> /fact User | prefers | dark mode
# Fact stored: e5f6g7h8

>>> /procedure morning-routine | waking up | check email; review calendar; plan day
# Procedure stored: i9j0k1l2
```

---

## Export / Import / Backup

| Command | Description |
|---|---|
| `/export [path]` | Export all memories to JSON (default: `~/animus-export.json`) |
| `/import <path>` | Import memories from a JSON file |
| `/backup [path]` | Create a full ZIP backup (default: `~/animus-backup.zip`) |

### Examples

```
>>> /export /tmp/animus-export.json
# Exported to: /tmp/animus-export.json

>>> /import ~/animus-export.json
# Imported 42 memories
```

---

## Tools

| Command | Description |
|---|---|
| `/tools` | List all available tools with parameters |
| `/tool <name> [param=value ...]` | Execute a tool directly |

### Direct Tool Execution

When the agent loop is disabled (Ollama provider), `/tool` is the only way to execute tools.

```
>>> /tool get_datetime
# Tool output: 2026-07-01 14:30:00

>>> /tool read_file path=/etc/hostname
# Tool output: myhostname

>>> /tool web_search query="Python context managers"
# Tool output: [search results]
```

**Tool names with spaces:** Use positional arguments for simple tools:

```
>>> /tool web_search Python context managers
# params["query"] = "Python context managers"
```

**Available built-in tools:** See the [Tools Reference](tools.md) for the full list.

---

## Reasoning Modes

| Command | Description |
|---|---|
| `/deep <query>` | Use deep reasoning mode (step-by-step, multi-perspective) |
| `/research <query>` | Research mode with automatic web search |
| `/brief [topic]` | Generate a situation briefing from memory context |

### Examples

```
>>> /deep Should we migrate from SQLite to PostgreSQL?
# Animus thinks step-by-step before responding

>>> /research vector databases for semantic search
# Runs web search, then synthesizes findings

>>> /brief
# Generates morning briefing from recent memories and nudges
```

---

## Decision Support

| Command | Description |
|---|---|
| `/decide <question>` | Run structured decision analysis with pros/cons/scoring |

### Example

```
>>> /decide Should we use FastAPI or Flask for the API layer?
# Outputs structured analysis with dimensions, options, and recommendation
```

---

## Task Tracking

| Command | Description |
|---|---|
| `/task add <description>` | Add a new task |
| `/task list [--all]` | List pending tasks (add `--all` for completed too) |
| `/task done <id>` | Mark a task complete |
| `/task start <id>` | Mark a task in progress |
| `/task delete <id>` | Delete a task |

### Examples

```
>>> /task add Review pull request #105
# Task added: t1a2b3c4 - Review pull request #105

>>> /task list
# [pending]   t1a2b3c4 Review pull request #105

>>> /task done t1a2b3c4
# Task completed: t1a2b3c4
```

---

## API Server (Phase 3)

| Command | Description |
|---|---|
| `/server start [port]` | Start the FastAPI server (default: 8420) |
| `/server stop` | Stop the API server |
| `/server status` | Show server status |

---

## Voice (Phase 3)

| Command | Description |
|---|---|
| `/voice on` | Enable voice input (microphone listening) |
| `/voice off` | Disable voice input |
| `/speak <text>` | Speak text aloud using TTS |
| `/speak-toggle` | Toggle automatic TTS for all responses |

**Dependencies:** Install with `pip install 'animus[voice]'`

---

## Integrations (Phase 4)

| Command | Description |
|---|---|
| `/integrations` | List all integrations with connection status |
| `/integrate <service>` | Connect an integration (interactive prompts for credentials) |
| `/disconnect <service>` | Disconnect an integration |

**Available services:** `filesystem`, `todoist`, `google_calendar`, `gmail`, `webhooks`

### Example

```
>>> /integrate todoist
# Prompts for API key, then connects
```

---

## Self-Learning (Phase 5)

| Command | Description |
|---|---|
| `/learning` | Show learning dashboard (total learned, pending approval, guardrails) |
| `/learning scan` | Trigger pattern detection across memory |
| `/learning approve <id>` | Approve a pending learning |
| `/learning reject <id>` | Reject a pending learning |
| `/learning history` | Show recent learning events |
| `/learning rollback` | List rollback checkpoints |
| `/learning rollback <id>` | Rollback to a checkpoint (unlearns items) |
| `/unlearn <id>` | Remove a specific learned item |
| `/guardrails` | List all active guardrails |
| `/guardrail add <rule>` | Add a user-defined guardrail |

### Examples

```
>>> /learning
# Total learned: 12
# Pending approval: 3
# Guardrail blocks: 0

>>> /learning scan
# Detected 2 patterns
#   [habit] User reviews docs before commits (confidence: 85%)

>>> /guardrail add "Never commit API keys to git"
# Added guardrail: g-abc123def456
```

---

## Citizen Zero

| Command | Description |
|---|---|
| `/reflect` | Generate reflection candidates from session (requires approval to persist) |
| `/eval` | Generate eval report with owner scoring |

**Requires:** `citizen_zero.enabled = true` in config.

---

## Proactive Intelligence

| Command | Description |
|---|---|
| `/briefing` | Generate morning briefing from nudges and memory |
| `/nudges` | Show active nudges with priorities |
| `/nudges dismiss [id]` | Dismiss one nudge (or all if no ID provided) |
| `/meeting-prep <topic>` | Prepare context for a meeting |

### Example

```
>>> /meeting-prep quarterly review
# Generates panel with relevant memories, tasks, and entities
```

---

## Entities & Relationships

| Command | Description |
|---|---|
| `/entities` | List tracked entities (people, projects, organizations, etc.) |
| `/entity add <name> <type> [alias1,alias2]` | Add an entity |
| `/entity <name>` | Show entity details and related context |
| `/entity delete <name>` | Delete an entity |

**Valid types:** `person`, `project`, `organization`, `place`, `topic`, `event`, `tool`

### Example

```
>>> /entity add Animus project
# Entity added: Animus (project)

>>> /entity Animus
# Shows mention count, last mentioned date, aliases, related memories
```

---

## Cross-Device Sync (Phase 6)

| Command | Description |
|---|---|
| `/sync start` | Start sync server and device discovery |
| `/sync stop` | Stop sync server and discovery |
| `/sync status` | Show sync status and connected peers |
| `/sync discover` | List discovered devices on the local network |
| `/sync connect <addr>` | Connect to a peer (ws://host:port or device_id) |
| `/sync disconnect` | Disconnect from current peer |
| `/sync now` | Trigger manual sync with connected peer |
| `/sync pair` | Show pairing code (shared secret) |

**Dependencies:** Install with `pip install websockets zeroconf`

---

## Forge (Multi-Agent Workflows)

| Command | Description |
|---|---|
| `/forge run <path>` | Run a workflow YAML file |
| `/forge resume <name>` | Resume a paused/failed workflow |
| `/forge status [name]` | Show workflow status (or list all) |
| `/forge list` | List all workflows with checkpoints |
| `/forge pause <name>` | Pause a running workflow |

### Example

```
>>> /forge run workflows/examples/build_task.yaml
# Running workflow: build_task
# Agents: 3 | Gates: 2 | Budget: $2.00
# [green]Workflow complete: completed[/green]
```

---

## Agent Mode

| Command | Description |
|---|---|
| `/build <description>` | Autonomous build pipeline (plan → code → lint → test → fix) |

### Example

```
>>> /build add a health check endpoint
# Build Pipeline: add a health check endpoint
# Steps: planner → coder → verifier → fixer
# Budget: $2.00
```

---

## Command Quick Reference Table

| Category | Commands |
|---|---|
| **Basic** | `exit`, `help`, `/status`, `/stats`, `/model`, `/auto` |
| **Memory** | `/remember`, `/recall`, `/forget`, `/review`, `/list`, `/history` |
| **Tags** | `/tag`, `/untag`, `/tags`, `/search-tags` |
| **Structured** | `/fact`, `/procedure` |
| **Export** | `/export`, `/import`, `/backup` |
| **Tools** | `/tools`, `/tool` |
| **Reasoning** | `/deep`, `/research`, `/brief` |
| **Decision** | `/decide` |
| **Tasks** | `/task add/list/done/start/delete` |
| **Server** | `/server start/stop/status` |
| **Voice** | `/voice on/off`, `/speak`, `/speak-toggle` |
| **Integrations** | `/integrations`, `/integrate`, `/disconnect` |
| **Learning** | `/learning`, `/learning scan/approve/reject/history/rollback`, `/unlearn`, `/guardrails`, `/guardrail add` |
| **Citizen Zero** | `/reflect`, `/eval` |
| **Proactive** | `/briefing`, `/nudges`, `/nudges dismiss`, `/meeting-prep` |
| **Entities** | `/entities`, `/entity add/delete/<name>` |
| **Sync** | `/sync start/stop/status/discover/connect/disconnect/now/pair` |
| **Forge** | `/forge run/resume/status/list/pause` |
| **Build** | `/build` |

---

## See Also

- [Tools Reference](tools.md) — All available tools and parameters
- [Memory System Reference](memory.md) — How memory works under the hood
- [Working with Local Models](../operators/local-models.md) — Ollama-specific behavior
- [Configuration](../operators/configuration.md) — Tuning Animus behavior
