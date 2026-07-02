# API Reference

The Animus HTTP API provides REST and WebSocket access to core functionality. It is built on FastAPI and runs on `127.0.0.1:8420` by default.

## Installation

The API server requires the optional `api` dependency:

```bash
pip install 'animus[api]'
```

This installs `fastapi`, `uvicorn`, and `pydantic`.

## Starting the Server

From the REPL:

```
/server start
```

Or programmatically:

```python
from animus.api import APIServer

server = APIServer(
    memory=memory_layer,
    cognitive=cognitive_layer,
    tools=tool_registry,
    tasks=task_tracker,
    decisions=decision_framework,
    host="127.0.0.1",
    port=8420,
    api_key="your-secret-key",  # optional
)
server.start()
```

## Authentication

If `api_key` is configured, all endpoints (except `/status`) require the header:

```
X-API-Key: your-secret-key
```

Configure via:

```yaml
api:
  api_key: your-secret-key
```

Or environment variable: `ANIMUS_API_KEY`

## Base URL

```
http://127.0.0.1:8420
```

---

## Status

### `GET /status`

Get system status. No authentication required.

**Response:**

```json
{
  "status": "running",
  "version": "0.4.0",
  "memory_count": 1247,
  "task_count": 23,
  "model_provider": "anthropic",
  "model_name": "claude-sonnet-5"
}
```

---

## Chat

### `POST /chat`

Send a chat message and get a response.

**Request:**

```json
{
  "message": "What did we decide about the database migration?",
  "mode": "auto",
  "conversation_id": "abc-123"
}
```

- `mode`: `auto`, `quick`, `deep`, `research`
- `conversation_id`: Optional — continues an existing conversation

**Response:**

```json
{
  "response": "We decided to use PostgreSQL with...",
  "conversation_id": "abc-123",
  "mode_used": "deep"
}
```

---

## Memories

### `POST /memory`

Create a new memory.

**Request:**

```json
{
  "content": "User prefers dark mode in all applications",
  "memory_type": "semantic",
  "tags": ["preference", "ui"],
  "source": "stated",
  "confidence": 1.0
}
```

**Response:** `MemoryResponse`

### `GET /memory/search`

Search memories with semantic relevance.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `string` | required | Search query |
| `limit` | `integer` | `10` | Max results (≤ 100) |
| `tags` | `string` | — | Comma-separated tag filter |

**Response:** `MemorySearchResponse`

### `GET /memory/{memory_id}`

Get a specific memory by ID (or partial ID prefix).

### `DELETE /memory/{memory_id}`

Delete a memory.

### `GET /memory/export/csv`

Export all memories as CSV download.

### `POST /memory/consolidate`

Consolidate old memories into summaries.

**Query Parameters:**

| Parameter | Type | Default |
|-----------|------|---------|
| `max_age_days` | `integer` | `90` |
| `min_group_size` | `integer` | `3` |

---

## Tools

### `GET /tools`

List all available tools with parameters and approval requirements.

### `POST /tools/{tool_name}`

Execute a tool.

**Request:**

```json
{
  "params": {
    "path": "/home/user/project/README.md"
  }
}
```

**Note:** Tools marked `requires_approval=True` return HTTP 403 via API.

---

## Tasks

### `GET /tasks`

List tasks.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `status` | `string` | — | Filter by status: `pending`, `in_progress`, `completed`, `cancelled` |
| `include_completed` | `boolean` | `false` | Include completed tasks |

### `POST /tasks`

Create a task.

**Request:**

```json
{
  "description": "Review PR #42",
  "tags": ["review", "urgent"],
  "priority": 1
}
```

### `PATCH /tasks/{task_id}`

Update a task.

**Request:**

```json
{
  "status": "in_progress",
  "description": "Updated description"
}
```

### `DELETE /tasks/{task_id}`

Delete a task.

---

## Decisions

### `POST /decide`

Perform decision analysis.

**Request:**

```json
{
  "question": "Which database should we use?",
  "options": ["PostgreSQL", "SQLite", "MySQL"],
  "criteria": ["performance", "ease-of-use", "cost"]
}
```

**Response:** Full decision analysis with recommendation and reasoning.

---

## Briefings

### `GET /brief`

Generate a situation briefing.

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `topic` | `string` | Optional topic filter |

---

## WebSocket Chat

### `WS /ws/chat`

Real-time streaming chat.

**Message format (client → server):**

```json
{
  "message": "Hello Animus",
  "mode": "auto"
}
```

**Response format (server → client):**

```json
{
  "response": "Hello! How can I help?",
  "conversation_id": "uuid",
  "mode_used": "auto"
}
```

The WebSocket creates a new conversation per connection.

---

## Integrations

### `GET /integrations`

List all integrations with connection status.

### `POST /integrations/{service}/connect`

Connect to an integration.

**Request:**

```json
{
  "credentials": {
    "token": "xxx"
  }
}
```

### `DELETE /integrations/{service}`

Disconnect from an integration.

**Supported services** (when IntegrationManager is configured): Discord, GitHub, Notion, Calendar, Email.

---

## Learning

### `GET /learning/status`

Get learning dashboard data.

### `GET /learning/items`

List learned items.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `status` | `string` | `all` | `all`, `active`, `pending` |

### `POST /learning/scan`

Trigger pattern detection scan.

**Response:**

```json
{
  "patterns_detected": 5
}
```

### `POST /learning/{item_id}/approve`

Approve a pending learning.

### `POST /learning/{item_id}/reject`

Reject a learning.

**Query Parameters:**

| Parameter | Type | Default |
|-----------|------|---------|
| `reason` | `string` | `""` |

### `DELETE /learning/{item_id}`

Unlearn a specific item.

### `GET /learning/history`

Get learning event history.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | `integer` | `50` | Max events (≤ 500) |
| `event_type` | `string` | — | Filter by event type |

---

## Guardrails

### `GET /guardrails`

List all guardrails (system + user-defined).

### `POST /guardrails`

Add a user-defined guardrail.

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `rule` | `string` | The rule text (required) |
| `description` | `string` | Human-readable description |

---

## Rollback

### `GET /learning/rollback-points`

List available rollback checkpoints.

### `POST /learning/rollback/{point_id}`

Rollback to a specific checkpoint.

---

## Proactive Intelligence

### `GET /nudges`

Get active proactive nudges.

### `POST /nudges/briefing`

Generate morning briefing now.

### `POST /nudges/meeting-prep`

Prepare meeting context.

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `topic` | `string` | Person name or meeting topic |

### `POST /nudges/{nudge_id}/dismiss`

Dismiss a nudge.

### `GET /proactive/stats`

Get proactive engine statistics.

---

## Entities

### `GET /entities`

List tracked entities.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `entity_type` | `string` | — | Filter by type: `person`, `organization`, `project`, `concept` |
| `limit` | `integer` | `50` | Max results |

### `POST /entities`

Add a new entity.

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `string` | Entity name (required) |
| `entity_type` | `string` | Type: `person`, `organization`, `project`, `concept` |
| `aliases` | `string` | Comma-separated aliases |
| `notes` | `string` | Free-form notes |

### `GET /entities/search`

Search entities by name, alias, or content.

### `GET /entities/{entity_id}`

Get entity details with context and relationships.

### `DELETE /entities/{entity_id}`

Delete an entity.

### `GET /entities/{entity_id}/timeline`

Get interaction timeline.

### `GET /entities/stats`

Get entity memory statistics.

---

## Autonomous Actions

### `GET /autonomous/actions`

List recent autonomous actions.

### `GET /autonomous/pending`

List actions awaiting user approval.

### `POST /autonomous/actions/{action_id}/approve`

Approve a pending action.

### `POST /autonomous/actions/{action_id}/deny`

Deny a pending action.

### `GET /autonomous/stats`

Get autonomous executor statistics.

---

## Register Translation

### `GET /register`

Get current communication register context.

### `POST /register/{register_name}`

Override communication register.

**Valid values:** `formal`, `casual`, `technical`, `neutral`

Pass `neutral` to clear the override.

---

## Error Responses

| Status | Meaning |
|--------|---------|
| `400` | Bad request — invalid parameter or body |
| `401` | Unauthorized — invalid or missing API key |
| `403` | Forbidden — tool requires approval, not available via API |
| `404` | Not found — resource or tool does not exist |
| `500` | Internal server error |
| `503` | Service unavailable — subsystem (learning, proactive, etc.) not initialized |

## Data Models

### MemoryResponse

```json
{
  "id": "uuid",
  "content": "string",
  "memory_type": "semantic",
  "tags": ["tag1"],
  "source": "stated",
  "confidence": 1.0,
  "created_at": "2026-07-02T10:00:00",
  "updated_at": "2026-07-02T10:00:00"
}
```

### TaskResponse

```json
{
  "id": "uuid",
  "description": "string",
  "status": "pending",
  "tags": ["tag1"],
  "priority": 1,
  "created_at": "2026-07-02T10:00:00"
}
```

### ToolResponse

```json
{
  "name": "read_file",
  "success": true,
  "output": "file contents",
  "error": null
}
```

---

## Files

| File | Lines | Responsibility |
|------|-------|--------------|
| `animus/api.py` | 1448 | FastAPI app factory, `APIServer`, Pydantic models |
