# Phase 3 — Intelligence Layer Spec
## Scope: Proactive engine, scheduled tasks, tool use, and memory-aware conversations

---

## WHAT WE'RE BUILDING

The piece that makes Animus feel alive: it doesn't just respond — it
thinks ahead, remembers context, uses tools, and reaches out when it
has something useful to say.

```
┌─────────────────────────────────────────────────────────────────┐
│  Intelligence Layer                                             │
│  Sits between the gateway and cognitive backends. Enriches      │
│  every conversation with memory, tools, and proactive context.  │
├─────────────────────────────────────────────────────────────────┤
│  Memory Integration                                             │
│  Episodic (conversations), Semantic (knowledge), Procedural     │
│  (how-to). Auto-injected into LLM context window.              │
├─────────────────────────────────────────────────────────────────┤
│  Tool Executor                                                  │
│  Function calling → tool dispatch → result injection.           │
│  Built-in tools + user-defined MCP tools.                       │
├─────────────────────────────────────────────────────────────────┤
│  Proactive Engine                                               │
│  Scheduled checks, nudges, morning briefs, reminders.           │
│  Sends messages unprompted when it has something useful.         │
├─────────────────────────────────────────────────────────────────┤
│  Automation Pipeline                                            │
│  User-defined triggers → conditions → actions.                  │
│  "When X happens, do Y" — configurable via dashboard.           │
├─────────────────────────────────────────────────────────────────┤
│  Intelligence Dashboard (extends Phase 2 dashboard)             │
│  /memory — browse and search memory stores                      │
│  /tools — registered tools, execution logs, MCP status          │
│  /automations — trigger/action rules, execution history         │
│  /activity — proactive engine log, nudge history                │
└─────────────────────────────────────────────────────────────────┘
```

---

## ARCHITECTURE

### Component 1: Memory Integration (`animus_bootstrap.intelligence.memory`)

**Core abstraction:** Every inbound message gets enriched with relevant
memories before hitting the LLM. Every conversation gets stored as new
memory after the response.

```python
@dataclass
class MemoryContext:
    """Injected into LLM prompt alongside the conversation."""
    episodic: list[str]      # Recent relevant conversations
    semantic: list[str]      # Knowledge graph facts
    procedural: list[str]    # How-to snippets
    user_prefs: dict         # Learned preferences

class MemoryManager:
    """Bridges bootstrap gateway with Animus Core memory layer."""

    async def recall(self, query: str, limit: int = 5) -> MemoryContext:
        """Retrieve relevant memories for a query."""

    async def store_conversation(self, session_id: str, messages: list[dict]) -> None:
        """Store a completed conversation turn as episodic memory."""

    async def store_fact(self, subject: str, predicate: str, obj: str) -> None:
        """Store a knowledge triple in semantic memory."""

    async def search(self, query: str, memory_type: str = "all") -> list[dict]:
        """Full-text search across memory stores."""
```

**Storage backends (configurable via config.toml):**

| Backend | Use Case | Config Key |
|---------|----------|------------|
| SQLite FTS5 | Default, zero-infra | `memory.backend = "sqlite"` |
| ChromaDB | Vector similarity search | `memory.backend = "chromadb"` |
| Animus Core | Full exocortex memory | `memory.backend = "animus"` |

**Memory injection pipeline:**
```
Inbound message
    → Extract query (last N messages)
    → MemoryManager.recall(query)
    → Build system prompt with memory context
    → Send to LLM
    → Store conversation turn
    → Return response
```

---

### Component 2: Tool Executor (`animus_bootstrap.intelligence.tools`)

**Core abstraction:** LLM responses can include tool calls. The executor
dispatches them, collects results, and feeds them back for a final response.

```python
@dataclass
class ToolDefinition:
    """A callable tool the LLM can invoke."""
    name: str
    description: str
    parameters: dict         # JSON Schema
    handler: Callable        # async def handler(**kwargs) -> str

@dataclass
class ToolResult:
    """Result of executing a tool."""
    tool_name: str
    success: bool
    output: str
    duration_ms: float

class ToolExecutor:
    """Manages tool registration and execution."""

    def register(self, tool: ToolDefinition) -> None: ...
    def unregister(self, name: str) -> None: ...
    def list_tools(self) -> list[ToolDefinition]: ...
    async def execute(self, name: str, arguments: dict) -> ToolResult: ...
```

**Built-in tools (ship with bootstrap):**

| Tool | Description | Category |
|------|-------------|----------|
| `web_search` | Search the web via DuckDuckGo/SearXNG | Information |
| `web_fetch` | Fetch and parse a URL | Information |
| `file_read` | Read a local file (sandboxed) | Filesystem |
| `file_write` | Write a local file (sandboxed) | Filesystem |
| `shell_exec` | Run a shell command (approval-gated) | System |
| `calendar_check` | Query Google Calendar (if configured) | Integration |
| `todoist_query` | Query Todoist tasks (if configured) | Integration |
| `send_message` | Send message on any connected channel | Gateway |
| `set_reminder` | Schedule a future notification | Proactive |
| `store_memory` | Explicitly store a fact/preference | Memory |

**MCP tool bridge:**
```python
class MCPToolBridge:
    """Discovers and wraps MCP server tools as ToolDefinitions."""

    async def discover_servers(self) -> list[str]:
        """Scan MCP config for available servers."""

    async def import_tools(self, server_name: str) -> list[ToolDefinition]:
        """Import all tools from an MCP server."""

    async def call_tool(self, server: str, tool: str, args: dict) -> str:
        """Execute a tool on an MCP server via stdio/SSE."""
```

**Tool execution pipeline:**
```
LLM response contains tool_use blocks
    → Parse tool calls
    → For each tool:
        → Check permissions (auto-approve / ask user / deny)
        → Execute via ToolExecutor
        → Collect ToolResult
    → Feed results back to LLM
    → Return final response
```

**Permission levels (per tool, configurable):**
- `auto` — Execute without asking (default for read-only tools)
- `approve` — Ask user via active channel before executing
- `deny` — Never execute (admin override)

---

### Component 3: Proactive Engine (`animus_bootstrap.intelligence.proactive`)

**Core abstraction:** Background scheduler that checks conditions and
sends messages unprompted when it has something useful to say.

```python
@dataclass
class ProactiveCheck:
    """A scheduled check that may produce a nudge."""
    name: str
    schedule: str            # cron expression or interval ("every 30m")
    checker: Callable        # async def() -> str | None (None = nothing to say)
    channels: list[str]      # Which channels to nudge on
    priority: str            # "low" | "normal" | "high"

class ProactiveEngine:
    """Runs scheduled checks and sends nudges."""

    def register_check(self, check: ProactiveCheck) -> None: ...
    def unregister_check(self, name: str) -> None: ...
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
```

**Built-in checks:**

| Check | Schedule | Description |
|-------|----------|-------------|
| `morning_brief` | Daily 7:00 AM | Weather, calendar, tasks summary |
| `calendar_reminder` | Every 15 min | Upcoming events in next 30 min |
| `task_nudge` | Every 2 hours | Overdue or stale tasks |
| `news_digest` | Daily 12:00 PM | Personalized news based on interests |
| `learning_prompt` | Daily 6:00 PM | Learning suggestion based on memory |

**Nudge delivery:**
```
Check fires
    → Checker returns message text (or None to skip)
    → Route through gateway to configured channels
    → Log in proactive_log table
    → Respect quiet hours (configurable)
```

**Config section:**
```toml
[proactive]
enabled = true
quiet_hours_start = "22:00"
quiet_hours_end = "07:00"
timezone = "America/New_York"

[proactive.checks.morning_brief]
enabled = true
schedule = "0 7 * * *"
channels = ["telegram", "webchat"]

[proactive.checks.calendar_reminder]
enabled = true
schedule = "every 15m"
channels = ["telegram"]
```

---

### Component 4: Automation Pipeline (`animus_bootstrap.intelligence.automations`)

**Core abstraction:** User-defined "if this then that" rules. Triggers
fire conditions, conditions gate actions.

```python
@dataclass
class AutomationRule:
    """A trigger → condition → action pipeline."""
    id: str
    name: str
    enabled: bool
    trigger: TriggerConfig       # What starts the rule
    conditions: list[Condition]  # All must be true
    actions: list[ActionConfig]  # Execute in order
    cooldown_seconds: int        # Min time between firings

@dataclass
class TriggerConfig:
    type: str                    # "message" | "schedule" | "webhook" | "event"
    params: dict

@dataclass
class Condition:
    type: str                    # "contains" | "from_channel" | "time_range" | "regex"
    params: dict

@dataclass
class ActionConfig:
    type: str                    # "reply" | "forward" | "run_tool" | "store_memory" | "webhook"
    params: dict
```

**Trigger types:**

| Trigger | Description | Example |
|---------|-------------|---------|
| `message` | Incoming message matches | Keywords, regex, from specific sender |
| `schedule` | Cron or interval | "Every Monday at 9 AM" |
| `webhook` | External HTTP POST | GitHub push event, CI notification |
| `event` | Internal system event | Channel connected, memory stored, tool executed |

**Action types:**

| Action | Description | Example |
|--------|-------------|---------|
| `reply` | Send message on a channel | Auto-reply to specific keywords |
| `forward` | Forward message to another channel | Telegram → Slack bridge |
| `run_tool` | Execute a registered tool | Run a script when triggered |
| `store_memory` | Save information to memory | Auto-capture meeting notes |
| `webhook` | POST to external URL | Notify external service |
| `run_workflow` | Trigger a Forge workflow | Complex multi-step automation |

**Automation storage:** SQLite table `automations` with JSON rule definitions.
Dashboard CRUD for creating/editing/disabling rules.

---

### Component 5: Intelligence Dashboard Extensions

**New routes (extend existing FastAPI app):**

#### `/memory` (enhanced)
- Browse episodic/semantic/procedural memory
- Full-text search with highlighted results
- Memory timeline visualization
- Manual add/edit/delete entries
- Memory stats (total entries, storage size, last indexed)

#### `/tools`
- Registered tools list with descriptions
- Execution log (tool name, args, result, duration, timestamp)
- MCP server status (connected/disconnected/error)
- Permission matrix editor
- Tool usage analytics (calls/day, avg duration, error rate)

#### `/automations`
- Rule list with enable/disable toggles
- Create/edit rule form (trigger → conditions → actions)
- Execution history per rule
- Rule testing (dry run)

#### `/activity`
- Proactive engine status (running/stopped)
- Nudge history with delivery status
- Check schedule overview
- Quiet hours indicator
- Manual trigger button for any check

---

## CONFIG EXTENSIONS

```toml
# Add to existing config.toml

[intelligence]
enabled = true
memory_backend = "sqlite"          # "sqlite" | "chromadb" | "animus"
memory_db_path = "~/.local/share/animus/memory.db"
tool_approval_default = "auto"     # "auto" | "approve" | "deny"
max_tool_calls_per_turn = 5
tool_timeout_seconds = 30

[intelligence.mcp]
config_path = "~/.config/animus/mcp.json"
auto_discover = true

[proactive]
enabled = true
quiet_hours_start = "22:00"
quiet_hours_end = "07:00"
timezone = "America/New_York"

[proactive.checks.morning_brief]
enabled = true
schedule = "0 7 * * *"
channels = ["telegram", "webchat"]

[proactive.checks.calendar_reminder]
enabled = false
schedule = "every 15m"
channels = ["telegram"]

[proactive.checks.task_nudge]
enabled = false
schedule = "0 */2 * * *"
channels = ["webchat"]
```

---

## FILE STRUCTURE

```
src/animus_bootstrap/intelligence/
├── __init__.py
├── memory.py               # MemoryManager, MemoryContext
├── memory_backends/
│   ├── __init__.py
│   ├── sqlite_backend.py   # SQLite FTS5 backend
│   ├── chromadb_backend.py # ChromaDB vector backend
│   └── animus_backend.py   # Animus Core bridge
├── tools/
│   ├── __init__.py
│   ├── executor.py         # ToolExecutor, ToolDefinition, ToolResult
│   ├── permissions.py      # Permission checking
│   ├── builtin/
│   │   ├── __init__.py
│   │   ├── web.py          # web_search, web_fetch
│   │   ├── filesystem.py   # file_read, file_write (sandboxed)
│   │   ├── system.py       # shell_exec (approval-gated)
│   │   ├── integrations.py # calendar, todoist
│   │   ├── gateway.py      # send_message
│   │   └── memory.py       # store_memory, set_reminder
│   └── mcp_bridge.py       # MCPToolBridge
├── proactive/
│   ├── __init__.py
│   ├── engine.py           # ProactiveEngine, scheduler
│   ├── checks/
│   │   ├── __init__.py
│   │   ├── morning_brief.py
│   │   ├── calendar.py
│   │   ├── tasks.py
│   │   ├── news.py
│   │   └── learning.py
│   └── schedule.py         # Cron + interval parser
└── automations/
    ├── __init__.py
    ├── engine.py           # AutomationEngine, rule evaluation
    ├── triggers.py         # Trigger implementations
    ├── conditions.py       # Condition implementations
    └── actions.py          # Action implementations
```

**Dashboard extensions:**
```
src/animus_bootstrap/dashboard/
├── routers/
│   ├── tools.py            # /tools routes
│   ├── automations.py      # /automations routes
│   └── activity.py         # /activity routes
│   # memory.py already exists — enhance it
├── templates/
│   ├── tools.html
│   ├── automations.html
│   └── activity.html
│   # memory.html already exists — enhance it
```

**Tests:**
```
tests/
├── test_memory.py          # MemoryManager, backends, injection
├── test_tools.py           # ToolExecutor, permissions, built-ins
├── test_mcp_bridge.py      # MCP tool discovery and execution
├── test_proactive.py       # ProactiveEngine, checks, scheduling
├── test_automations.py     # AutomationEngine, triggers, conditions, actions
└── test_intelligence_dashboard.py  # New dashboard routes
```

---

## INTEGRATION POINTS

### Gateway ↔ Intelligence

The intelligence layer wraps the existing cognitive backends:

```python
class IntelligentRouter(MessageRouter):
    """Extends MessageRouter with memory + tools + automations."""

    def __init__(self, memory: MemoryManager, tools: ToolExecutor, ...):
        super().__init__()
        self.memory = memory
        self.tools = tools

    async def handle_message(self, message: GatewayMessage) -> GatewayResponse:
        # 1. Check automations (may short-circuit)
        auto_response = await self._check_automations(message)
        if auto_response:
            return auto_response

        # 2. Recall relevant memories
        context = await self.memory.recall(message.text)

        # 3. Build enriched prompt
        prompt = self._build_prompt(message, context)

        # 4. Send to LLM (may return tool calls)
        response = await self._cognitive_loop(prompt, message)

        # 5. Store conversation
        await self.memory.store_conversation(...)

        return response

    async def _cognitive_loop(self, prompt, message):
        """Loop: LLM → tool calls → LLM → ... → final response."""
        for _ in range(self.max_iterations):
            response = await self.backend.generate(prompt)
            if not response.tool_calls:
                return response
            results = await self.tools.execute_batch(response.tool_calls)
            prompt = self._append_tool_results(prompt, results)
        return response  # Max iterations reached
```

### Bootstrap ↔ Animus Core

When `memory.backend = "animus"`, the memory manager bridges to Animus Core's
full memory system (ChromaDB episodic/semantic/procedural + entity memory):

```python
class AnimusMemoryBackend:
    """Bridges to Animus Core memory layer via import."""

    def __init__(self):
        from animus.memory import MemoryManager as CoreMemory
        self._core = CoreMemory()
```

### Bootstrap ↔ Forge

When Forge is available, automations can trigger Forge workflows:

```python
class ForgeWorkflowAction:
    """Triggers a Forge workflow via HTTP API."""

    async def execute(self, workflow_id: str, params: dict) -> str:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.forge_url}/api/workflows/{workflow_id}/run",
                json=params,
            )
            return resp.json()
```

---

## DEPENDENCY CHANGES

```toml
# New optional dependencies
[project.optional-dependencies]
intelligence = [
    "apscheduler>=3.10.0",      # Scheduler for proactive engine
    "duckduckgo-search>=4.0",   # Web search tool
    "trafilatura>=1.6.0",       # Web content extraction
    "croniter>=1.4.0",          # Cron expression parsing
]
mcp = [
    "mcp>=1.0.0",               # MCP SDK for tool bridge
]
chromadb = [
    "chromadb>=0.4.0",          # Vector memory backend
]
```

---

## IMPLEMENTATION ORDER

1. **Memory integration** — MemoryManager + SQLite FTS5 backend + injection pipeline
2. **Tool executor** — ToolDefinition + executor + permissions + 3 built-in tools (web_search, file_read, store_memory)
3. **Intelligent router** — Wraps gateway router with memory + tool loop
4. **Remaining built-in tools** — Complete the tool set
5. **MCP bridge** — Discover and import MCP server tools
6. **Proactive engine** — Scheduler + morning_brief check
7. **Remaining proactive checks** — Calendar, tasks, news, learning
8. **Automation engine** — Triggers, conditions, actions
9. **Dashboard extensions** — /tools, /automations, /activity, enhanced /memory
10. **Integration tests** — End-to-end: message → memory → tools → response

---

## SUCCESS CRITERIA

- [ ] Conversations remember previous interactions (memory recall)
- [ ] LLM can call tools and return results in conversation
- [ ] MCP tools discoverable and executable from any channel
- [ ] Morning brief arrives on Telegram at configured time
- [ ] User can create "if message contains X, forward to Y" automation via dashboard
- [ ] Tool permissions enforce approval gates for destructive operations
- [ ] 80%+ test coverage on intelligence module
- [ ] All existing Phase 1 + Phase 2 tests still pass
- [ ] Dashboard shows memory search, tool logs, automation rules, proactive activity
