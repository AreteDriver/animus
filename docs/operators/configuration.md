# Configuration

> Config files, environment variables, and secrets management.

---

## Config File

Animus reads `~/.config/animus/config.toml` (chmod 600).

### Core Settings

| Section | Key | Default | Description |
|---|---|---|---|
| `animus` | `version` | `"0.1.0"` | Package version |
| `animus` | `first_run` | `true` | Triggers wizard on install |
| `animus` | `data_dir` | `"~/.local/share/animus"` | Data storage |

### API Keys

| Section | Key | Default | Description |
|---|---|---|---|
| `api` | `anthropic_key` | `""` | Anthropic API key |
| `api` | `openai_key` | `""` | OpenAI API key (optional) |

**Resolution order** (first non-empty wins):
1. Environment variable (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`)
2. Secrets file (`$ANIMUS_SECRETS_FILE`, default `~/.local/share/animus/secrets.env`) — KEY=VAL lines, chmod 400 recommended
3. `config.toml` field (legacy / plaintext)
4. Empty string

### Forge

| Section | Key | Default | Description |
|---|---|---|---|
| `forge` | `enabled` | `false` | Connect to Forge orchestration |
| `forge` | `host` | `"localhost"` | Forge API host |
| `forge` | `port` | `8000` | Forge API port |
| `forge` | `api_key` | `""` | Forge API key |

### Memory

| Section | Key | Default | Description |
|---|---|---|---|
| `memory` | `backend` | `"sqlite"` | `sqlite` / `chromadb` / `weaviate` |
| `memory` | `path` | `"~/.local/share/animus/memory.db"` | DB path |
| `memory` | `max_context_tokens` | `100000` | Token budget for context window |

### Identity

| Section | Key | Default | Description |
|---|---|---|---|
| `identity` | `name` | `""` | What Animus calls you |
| `identity` | `timezone` | `""` | User timezone |
| `identity` | `locale` | `""` | User locale |
| `identity` | `identity_dir` | `"~/.config/animus/identity"` | Identity files path |

### Services (Dashboard)

| Section | Key | Default | Description |
|---|---|---|---|
| `services` | `autostart` | `true` | Start on boot |
| `services` | `host` | `"127.0.0.1"` | Bind address |
| `services` | `port` | `7700` | Dashboard port |
| `services` | `log_level` | `"info"` | Logging level |
| `services` | `update_check` | `true` | Check for updates |
| `services` | `auth_required` | `"auto"` | `auto` / `always` / `never` |
| `services` | `auth_token` | `""` | Bearer token (auto-generated) |

### Ollama

| Section | Key | Default | Description |
|---|---|---|---|
| `ollama` | `enabled` | `true` | Use local Ollama |
| `ollama` | `host` | `"localhost"` | Ollama host |
| `ollama` | `port` | `11434` | Ollama port |
| `ollama` | `model` | `"llama3.2"` | Default model |
| `ollama` | `code_model` | `""` | Dedicated code model (optional) |
| `ollama` | `autoinstall` | `true` | Auto-install missing models |

### Gateway

| Section | Key | Default | Description |
|---|---|---|---|
| `gateway` | `enabled` | `true` | Gateway active |
| `gateway` | `default_backend` | `"anthropic"` | Default LLM backend |
| `gateway` | `max_response_tokens` | `4096` | Max tokens per response |
| `gateway` | `message_log` | `false` | Audit log to SQLite |

### Intelligence

| Section | Key | Default | Description |
|---|---|---|---|
| `intelligence` | `enabled` | `true` | Intelligence layer active |
| `intelligence` | `memory_backend` | `"sqlite"` | `sqlite` / `chromadb` / `animus` |
| `intelligence` | `memory_db_path` | `"~/.local/share/animus/intelligence.db"` | Intelligence DB |
| `intelligence` | `tool_approval_default` | `"auto"` | `auto` / `approve` / `deny` |
| `intelligence` | `max_tool_calls_per_turn` | `5` | Tool call limit |
| `intelligence` | `tool_timeout_seconds` | `30` | Tool execution timeout |
| `intelligence.mcp` | `config_path` | `"~/.config/animus/mcp.json"` | MCP server config |
| `intelligence.mcp` | `auto_discover` | `true` | Auto-discover MCP servers |

### Channels

All channels default to `enabled: false` except `webchat`.

| Channel | Required Secret |
|---|---|
| `channels.webchat` | None |
| `channels.telegram` | `bot_token` |
| `channels.discord` | `bot_token`, `allowed_guilds` |
| `channels.slack` | `bot_token`, `app_token` |
| `channels.matrix` | `homeserver`, `access_token`, `room_ids` |
| `channels.signal` | `phone_number` |
| `channels.whatsapp` | `phone_number` |
| `channels.email` | `imap_host`, `smtp_host`, `username`, `password` |

### Self-Improvement

| Section | Key | Default | Description |
|---|---|---|---|
| `self_improvement` | `reflection_enabled` | `true` | Enable reflection loop |
| `self_improvement` | `reflection_interval_hours` | `24` | Hours between reflections |
| `self_improvement` | `reflection_min_interactions` | `10` | Min interactions to trigger |
| `self_improvement` | `approval_required` | `true` | Human approval for changes |
| `self_improvement` | `proposals_dir` | `"~/.config/animus/proposals"` | Change proposals path |

### Proactive Engine

| Section | Key | Default | Description |
|---|---|---|---|
| `proactive` | `enabled` | `true` | Proactive checks active |
| `proactive` | `quiet_hours_start` | `"22:00"` | Silent period start |
| `proactive` | `quiet_hours_end` | `"07:00"` | Silent period end |
| `proactive` | `timezone` | `"UTC"` | Quiet hours timezone |

### Personas

| Section | Key | Default | Description |
|---|---|---|---|
| `personas` | `enabled` | `true` | Persona system active |
| `personas` | `default_name` | `"Animus"` | Default persona name |
| `personas` | `default_tone` | `"balanced"` | Default tone |
| `personas` | `default_max_response_length` | `"medium"` | Response length |
| `personas` | `default_emoji_policy` | `"minimal"` | Emoji usage |
| `personas` | `default_system_prompt` | `"You are Animus..."` | Base system prompt |

---

## Environment Variables

These override config file values:

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `ANIMUS_SECRETS_FILE` | Path to secrets env file |
| `ANIMUS_SKIP_INTEGRATION_TESTS` | Skip integration tests in CI |

---

## Per-Package Config

Each package may read its own config:
- Core: `~/.config/animus/core.toml`
- Forge: `~/.config/animus/forge.toml`
- Bootstrap: `~/.config/animus/config.toml` (canonical)

---

## See Also

- [Deployment](deployment.md) — Service setup
- [Monitoring](monitoring.md) — Health and logs
- [Troubleshooting](troubleshooting.md) — Common config issues
