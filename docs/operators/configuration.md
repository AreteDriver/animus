# Configuration

> Config files, environment variables, and secrets management.

---

## Config File

Animus reads `~/.config/animus/config.toml` (chmod 600).

### Bootstrap Config Table

| Section | Key | Default | Description |
|---|---|---|---|
| `animus` | `version` | `0.1.0` | Package version |
| `animus` | `first_run` | `true` | Triggers wizard on install |
| `animus` | `data_dir` | `~/.local/share/animus` | Data storage |
| `api` | `anthropic_key` | `""` | Anthropic API key |
| `api` | `openai_key` | `""` | OpenAI API key (optional) |
| `forge` | `enabled` | `false` | Connect to Forge |
| `forge` | `host` | `localhost` | Forge API host |
| `forge` | `port` | `8000` | Forge API port |
| `memory` | `backend` | `sqlite` | `sqlite` / `chroma` / `weaviate` |
| `memory` | `path` | `~/.local/share/animus/memory.db` | DB path |
| `identity` | `name` | `""` | What Animus calls you |
| `services` | `port` | `7700` | Dashboard port |
| `services` | `log_level` | `info` | Logging level |
| `services` | `autostart` | `true` | Start on boot |

## Environment Variables

These override config file values:

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `ANIMUS_SKIP_INTEGRATION_TESTS` | Skip integration tests in CI |

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
