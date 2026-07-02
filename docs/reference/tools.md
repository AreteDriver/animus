# Tool Reference

> **Last verified:** 2026-07-01 against `packages/core/animus/tools.py`

All tools are available via `/tool <name>` in the Animus REPL. Natural language tool use (the agentic loop) is unreliable with local models — use direct `/tool` invocation.

---

## Quick Reference

| Tool | What It Does | Approval |
|---|---|---|
| `read_file` | Read a file from disk | No |
| `list_files` | List files matching a glob | No |
| `write_file` | Create or overwrite a file | **Yes** |
| `edit_file` | Find/replace text in a file | **Yes** |
| `run_command` | Execute a shell command | **Yes** |
| `web_search` | DuckDuckGo search | No |
| `http_request` | HTTP GET/POST/etc | **Yes** |
| `get_datetime` | Current date/time | No |
| `search_memory` | Search stored memories | No |
| `remember` | Store a new memory | No |
| `recall` | Recall memories by query | No |
| `forget` | Delete a memory by ID | No |

---

## Filesystem Tools

### `read_file`

Read the contents of a local file.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `path` | string | **Yes** | — | Absolute or relative path |
| `max_size` | integer | No | 100,000 bytes | Read limit (prevents loading huge files) |

**Example:**

```bash
>>> /tool read_file path=README.md
>>> /tool read_file path=/etc/hostname max_size=1024
```

**Notes:**
- Respects `~` expansion (`~/projects/foo` → `/home/user/projects/foo`)
- Files outside `allowed_paths` (if security config is set) are blocked

---

### `list_files`

List files in a directory matching a glob pattern.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `directory` | string | No | current dir | Base directory |
| `pattern` | string | No | `*` | Glob pattern. Use `**` for recursive |
| `max_results` | integer | No | 100 | Result limit |

**Example:**

```bash
>>> /tool list_files pattern="*.py"
>>> /tool list_files directory=packages/core pattern="**/*.md" max_results=20
```

---

### `write_file`

Create or overwrite a file. **Requires approval.**

**Parameters:**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `path` | string | **Yes** | — | Target path |
| `content` | string | **Yes** | — | Full file content |

**Example:**

```bash
>>> /tool write_file path=/tmp/note.txt content="Hello world"
```

**Security:**
- Blocked if path matches a `blocked_paths` pattern
- Restricted to `write_roots` when sandbox mode is active
- Approval prompt shows the target path

---

### `edit_file`

Find-and-replace text in a file. **Requires approval.**

**Parameters:**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `path` | string | **Yes** | — | File to edit |
| `old_text` | string | **Yes** | — | Exact text to find (must be unique in file) |
| `new_text` | string | **Yes** | — | Replacement text |

**Example:**

```bash
>>> /tool edit_file path=config.yaml old_text="debug: true" new_text="debug: false"
```

**Important:** `old_text` must match **exactly** — including indentation and whitespace. Read the file first with `read_file` to get the exact text.

---

### `run_command`

Execute a shell command. **Requires approval.**

**Parameters:**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `command` | string | **Yes** | — | Shell command to run |
| `timeout` | integer | No | 30 | Timeout in seconds |

**Example:**

```bash
>>> /tool run_command command="git status"
>>> /tool run_command command="ruff check packages/ --fix" timeout=60
```

**Security:**
- Blocked commands are defined in `blocked_commands` (configurable)
- Approval prompt shows the full command before execution
- Output is truncated at 10,000 characters

---

## Web Tools

### `web_search`

Search the web via DuckDuckGo.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `query` | string | **Yes** | — | Search query |

**Example:**

```bash
>>> /tool web_search query="python dataclasses vs pydantic"
```

**Notes:**
- Returns top 10 results with title, URL, and snippet
- Requires internet connectivity
- No API key required

---

### `http_request`

Make an HTTP request to a REST API. **Requires approval.**

**Parameters:**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `url` | string | **Yes** | — | Full URL |
| `method` | string | No | `GET` | `GET`, `POST`, `PUT`, `PATCH`, `DELETE` |
| `headers` | object | No | `{}` | Key-value header pairs |
| `body` | string | No | — | Request body (JSON string for POST/PUT/PATCH) |
| `auth_type` | string | No | `none` | `none`, `bearer`, `basic`, `api_key` |
| `auth_value` | string | No | — | Token/key for auth |
| `timeout` | integer | No | 30 | Seconds (max 60) |

**Example:**

```bash
>>> /tool http_request url="https://api.github.com/repos/AreteDriver/animus"
>>> /tool http_request url="https://api.example.com/items" method=POST headers='{"Content-Type":"application/json"}' body='{"name":"foo"}'
```

---

## Utility Tools

### `get_datetime`

Get the current date and time.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `format` | string | No | `%Y-%m-%d %H:%M:%S` | strftime format string |

**Example:**

```bash
>>> /tool get_datetime
>>> /tool get_datetime format="%Y-%m-%d"
```

---

## Memory Tools

Available when a `MemoryLayer` is initialized (standard in REPL mode).

### `search_memory`

Semantic search over stored memories.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `query` | string | **Yes** | — | Search query |
| `limit` | integer | No | 5 | Max results |
| `tags` | string\|list | No | — | Filter by tags (comma-separated or list) |

**Example:**

```bash
>>> /tool search_memory query="CI badge fixes" limit=3
>>> /tool search_memory query="deployment" tags="production,urgent"
```

---

### `remember`

Store a new memory.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `content` | string | **Yes** | — | Memory text |
| `type` | string | No | `semantic` | `semantic`, `episodic`, `procedural` |

**Example:**

```bash
>>> /tool remember content="API key for staging: sk-test-123" type=semantic
>>> /tool remember content="Refactored auth module on 2026-07-01" type=episodic
```

---

### `recall`

Recall memories by query with optional type filter.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `query` | string | **Yes** | — | Recall query |
| `type` | string | No | — | Filter by memory type |
| `limit` | integer | No | 5 | Max results |

**Example:**

```bash
>>> /tool recall query="database migration" type=episodic limit=10
```

---

### `forget`

Delete a memory by ID.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `id` | string | **Yes** | — | Full or partial memory ID |

**Example:**

```bash
>>> /tool forget id=a3f7d2e1
```

---

## Lugh Tools (Repository Monitoring)

Requires Lugh subsystem. Available in `create_default_registry()`.

### `harvest_repo`

Clone and index a repository for monitoring.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `url` | string | **Yes** | Git remote URL |
| `branch` | string | No | Branch to checkout (default: default branch) |

### `animus_watchlist_add`

Add a repository to the watchlist.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `url` | string | **Yes** | Repository URL |
| `name` | string | **Yes** | Human-readable name |

### `animus_watchlist_remove`

Remove a repository from the watchlist.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `name` | string | **Yes** | Name of watched repo |

### `animus_watchlist_list`

List all watched repositories.

(No parameters.)

### `animus_watchlist_scan`

Scan watched repos for new commits/issues.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `name` | string | No | Scan a specific repo (default: all) |

---

## Working with Local Models

**Critical:** The agentic loop (natural language → automatic tool selection) is **unreliable** with local 7B–8B models. They struggle with the `TOOL: <number>` format and often output prose instead of structured tool calls.

### What Works Reliably

| Approach | Reliability |
|---|---|
| `/tool read_file path=...` | ✅ Perfect |
| `/tool list_files pattern=...` | ✅ Perfect |
| `/tool web_search query=...` | ✅ Perfect |
| `/tool get_datetime` | ✅ Perfect |
| "Show me the README" (agent loop) | ❌ Garbled output, max iterations |
| "Fix the bug in auth.py" (agent loop) | ❌ Fails after 2–3 attempts |

### Recommended Workflow

1. **Discover:** Use `/tool list_files` or `/tool read_file` to explore
2. **Plan:** Read the relevant files, understand the codebase
3. **Act:** Chain `/tool` commands manually
4. **Verify:** Use `/tool run_command` to run tests or lint

**Example session:**

```bash
>>> /tool list_files pattern="packages/core/animus/*.py"
>>> /tool read_file path=packages/core/animus/config.py
>>> /tool edit_file path=packages/core/animus/config.py old_text="debug = False" new_text="debug = True"
>>> /tool run_command command="python -m pytest packages/core/tests/test_config.py -v"
```

---

## Tool Approval

Tools marked **Requires approval** trigger a confirmation prompt:

```
Tool: write_file
  path: /tmp/test.txt
  content: Hello world
Execute? [Y/n]
```

**Auto-approve mode:** Set `ANIMUS_AUTO_APPROVE=1` before starting Animus, or toggle with `/auto` in the REPL.

**⚠️ Warning:** Auto-approve skips confirmation for all tools, including `run_command` and `write_file`. Use only in trusted environments.

---

## Security Configuration

Tools respect `ToolsSecurityConfig` if provided to `create_default_registry()`:

| Config Field | Effect |
|---|---|
| `allowed_paths` | Read access restricted to these directories |
| `blocked_paths` | Read/write explicitly blocked |
| `write_roots` | Write access restricted to these directories |
| `blocked_commands` | Shell commands that `run_command` rejects |

**Example:**

```python
from animus.config import ToolsSecurityConfig

config = ToolsSecurityConfig(
    allowed_paths=["/home/user/projects"],
    write_roots=["/home/user/projects/build"],
    blocked_commands=["rm -rf /", "sudo"],
)
tools = create_default_registry(security_config=config)
```

---

## Common Errors

| Error | Cause | Fix |
|---|---|---|
| `Access denied: path not in allowed directories` | File outside `allowed_paths` | Use absolute path inside allowed root, or relax config |
| `Write denied: path not in write_roots` | Sandbox mode active | Write to a directory under `write_roots` |
| `Format error. Use EXACTLY: TOOL: <number>` | Natural language mode failed | Switch to `/tool <name>` direct invocation |
| `Max constrained iterations (8) reached` | Agent loop exceeded limit | Already lowered to 3 in patched versions; use `/tool` |
| `Unknown tool: foo` | Typo or tool not registered | Check spelling; verify tool is in registry with `/tools` |

---

## See Also

- [Quickstart](../getting-started/quickstart.md) — Installing and running Animus
- [Ollama Setup](../getting-started/ollama-setup.md) — Local model configuration
- [Architecture → Overview](../architecture/overview.md) — How the tool system fits into the architecture
