# Animus Bootstrap

Animus is your personal AI — sovereign, persistent, local-first.

```bash
pip install animus-bootstrap && animus-bootstrap install
```

That single command installs dependencies, runs the onboarding wizard, registers a system service, and opens the dashboard. First-run to working system in under 5 minutes.

<!-- screenshot -->

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  animus-bootstrap install (daemon)                  │
│  One-command bootstrap. Installs deps, registers    │
│  system service, manages config, checks updates.    │
├─────────────────────────────────────────────────────┤
│  animus-bootstrap setup (onboarding wizard)         │
│  Interactive first-run. API keys, identity,         │
│  Forge connection, device/memory config.            │
├─────────────────────────────────────────────────────┤
│  animus-bootstrap dashboard (local web UI)          │
│  Status, health, config management. Runs at         │
│  localhost:7700 after install.                      │
└─────────────────────────────────────────────────────┘
```

## Quickstart

1. **Install:** `pip install animus-bootstrap`
2. **Run:** `animus-bootstrap install`
3. **Wizard completes:** API keys, identity, memory backend
4. **Dashboard opens:** `http://localhost:7700`
5. **Done.** Animus is running as a system service.

## CLI Commands

| Command | Description |
|---------|-------------|
| `animus-bootstrap install` | Full install: deps, service, wizard, dashboard |
| `animus-bootstrap setup` | Re-run the onboarding wizard |
| `animus-bootstrap start` | Start the system service |
| `animus-bootstrap stop` | Stop the system service |
| `animus-bootstrap status` | Show system status table |
| `animus-bootstrap update` | Check for and apply updates |
| `animus-bootstrap dashboard` | Open dashboard at localhost:7700 |
| `animus-bootstrap config get <key>` | Print a config value |
| `animus-bootstrap config set <key> <value>` | Update a config value |

## Configuration

Config lives at `~/.config/animus/config.toml` (chmod 600).

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `animus` | `version` | `0.1.0` | Package version |
| `animus` | `first_run` | `true` | Triggers wizard on install |
| `animus` | `data_dir` | `~/.local/share/animus` | Data storage directory |
| `api` | `anthropic_key` | `""` | Anthropic API key (required) |
| `api` | `openai_key` | `""` | OpenAI API key (optional) |
| `forge` | `enabled` | `false` | Connect to Animus Forge |
| `forge` | `host` | `localhost` | Forge API host |
| `forge` | `port` | `8000` | Forge API port |
| `memory` | `backend` | `sqlite` | sqlite / chroma / weaviate |
| `memory` | `path` | `~/.local/share/animus/memory.db` | Memory database path |
| `memory` | `max_context_tokens` | `100000` | Max context window |
| `identity` | `name` | `""` | What Animus calls you |
| `identity` | `timezone` | `""` | Your timezone |
| `services` | `port` | `7700` | Dashboard port |
| `services` | `log_level` | `info` | Logging level |
| `services` | `autostart` | `true` | Start on boot |

## Platform Support

| Platform | Status | Service Manager |
|----------|--------|----------------|
| Linux | Supported | systemd (user service) |
| macOS | Supported | launchd |
| Windows | Stub | Planned |

## Relationship to Animus Ecosystem

- **Animus Core** — The exocortex engine (identity, memory, CLI)
- **Animus Forge** — Multi-agent orchestration engine (connects at wizard Step 4)
- **Animus Quorum** — Coordination protocol (coming in v0.3)
- **Animus Bootstrap** — This package. The install/setup/dashboard layer.

## Development

```bash
git clone git@github.com:AreteDriver/Animus.git
cd Animus/packages/bootstrap
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
ruff check src/ tests/ && ruff format --check src/ tests/
```

287 tests, 93% coverage, Python 3.11+.

## Status

**Alpha** — Not ready for production use. API keys are file-permission protected but the system is under active development.

## License

MIT - 2026, AreteDriver
