# Animus Operations Center

The Animus Bootstrap dashboard has evolved from a static status page into a **Cognitive Operations Center** — a live, instrumented control surface for monitoring and managing the Animus runtime.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Dashboard (FastAPI)                    │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────┐  │
│  │  Home   │  │ Events  │  │  Tasks  │  │   Memory    │  │
│  │ (HTMX)  │  │ (HTMX+  │  │ (HTMX)  │  │   (HTMX)    │  │
│  │         │  │  SSE)   │  │         │  │             │  │
│  └────┬────┘  └────┬────┘  └────┬────┘  └──────┬──────┘  │
│       │            │            │               │         │
│  ┌────┴────────────┴────────────┴───────────────┴─────┐  │
│  │              Operational Controls                    │  │
│  │  /runtime/pause  /tasks/{id}/kill  /memory/clear   │  │
│  │  /tools/{name}/rerun  /events/export  /alerts/ack  │  │
│  └─────────────────────────────────────────────────────┘  │
│                           │                                │
│  ┌────────────────────────┴──────────────────────────┐  │
│  │              Event Ledger (SQLite + Ring Buffer)      │  │
│  │  Thread-safe append-only log of operational events  │  │
│  └─────────────────────────────────────────────────────┘  │
│                           │                                │
│  ┌────────────────────────┴──────────────────────────┐  │
│  │              Alert Manager (Threshold Monitoring)   │  │
│  │  error_rate_max: 5.0/min  tool_failure_rate_max: 3  │  │
│  │  60s cooldown per alert type  composite health 0-100 │  │
│  └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Event Types

| Event Type | Source | Description |
|---|---|---|
| `session_started` | runtime | Runtime boot with version |
| `session_ended` | runtime | Runtime shutdown |
| `runtime_paused` | dashboard | Operator paused the runtime |
| `runtime_resumed` | dashboard | Operator resumed the runtime |
| `tool_execution` | tool_executor | Tool ran (success/fail, duration) |
| `tool_rerun` | dashboard | Operator re-ran a tool from history |
| `task_created` | dashboard | New task created via form |
| `task_completed` | dashboard | Task marked done |
| `task_deleted` | dashboard | Task removed |
| `task_killed` | dashboard | Task force-removed |
| `feedback_recorded` | dashboard | User rated a response |
| `config_changed` | dashboard | Config key saved or deleted |
| `proposal_approved` | dashboard | Proposal accepted |
| `proposal_rejected` | dashboard | Proposal declined |
| `memory_cleared` | dashboard | Operator cleared all memories |
| `events_exported` | dashboard | JSON/CSV export downloaded |
| `alert_acknowledged` | dashboard | Operator acknowledged alert |
| `alert` | alert_manager | Threshold breach (error_rate, tool_failure_rate) |

## API Surface

### Operational Events

- `GET /events` — Full events page with stats
- `GET /events/feed` — HTMX fragment (polls every 5s)
- `GET /events/stream` — Server-Sent Events (persistent)
- `GET /events/export?format=json|csv` — Download event history

### Controls

- `POST /runtime/pause` — Pause proactive processing
- `POST /runtime/resume` — Resume paused runtime
- `POST /tasks/{id}/kill` — Force-delete a task
- `POST /memory/clear` — Clear all memories (with backend fallback)
- `POST /tools/{name}/rerun` — Re-run tool with JSON arguments
- `POST /alerts/acknowledge` — Acknowledge alert by type

### Health

- `GET /health` — JSON health payload (unauthenticated)
- Home page displays composite health score (0–100) with color coding:
  - **Green** (≥80): Healthy
  - **Yellow** (≥50): Degraded
  - **Red** (<50): Critical

## Security

All state-changing endpoints require CSRF token validation via `X-CSRF-Token` header. The middleware enforces this automatically for every `POST`, `PUT`, `DELETE`, and `PATCH` request except `/health` and `/api/health`.

HTMX requests inherit the token from a cookie-set configuration:

```javascript
htmx.config.headers['X-CSRF-Token'] = csrfToken;
```

## Instrumentation

The `ToolExecutor` exposes a non-invasive observer pattern:

```python
tool_executor.add_execution_observer(callback)
```

The runtime wires this to the `EventLedger` on startup, so every tool execution is recorded without modifying individual tools.

## Alert Thresholds

| Metric | Default Threshold | Severity Escalation |
|---|---|---|
| Error rate | 5.0 / min | Critical at 2× threshold |
| Tool failure rate | 3.0 / min | Critical at 2× threshold |

Cooldown: 60 seconds per alert type to prevent spam.

## Keyboard Shortcuts

Press `?` anywhere in the dashboard to open the help modal.

| Shortcut | Action |
|---|---|
| `?` | Toggle help modal |
| `/` | Quick page search |
| `g` `h` | Go to Home |
| `g` `e` | Go to Events |
| `g` `t` | Go to Tasks |
| `Esc` | Close modal/overlay |
