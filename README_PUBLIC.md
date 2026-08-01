# Animus — Personal AI Exocortex

**Local-first, open-source AI operating environment you own.**

![License](https://img.shields.io/github/license/your-org/animus)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Platform](https://img.shields.io/badge/platform-Linux-green)

Animus is a multi-agent orchestration framework that runs entirely on your hardware. No API keys required. No cloud dependency. Just you, your models, and a system that remembers everything.

---

## Prerequisites

- **Python 3.10+**
- **Ollama** — local model server
- **Linux** (macOS on the roadmap; Windows not supported)

---

## Quick Start (< 10 minutes)

### 1. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:32b
```

### 2. Install Animus

```bash
curl -fsSL https://raw.githubusercontent.com/your-org/animus/main/scripts/install.sh | bash
```

Or manually:

```bash
git clone https://github.com/your-org/animus.git
cd animus
pip install -e packages/core
animus init
animus brief
```

### 3. Verify

```bash
animus brief
# → Returns a situation briefing from your local model
```

---

## Architecture

```
┌─────────────────────────────────────────┐
│  Interface    │  Dashboard + PWA + CLI  │
├─────────────────────────────────────────┤
│  Cognitive    │  Forge + Quorum         │
├─────────────────────────────────────────┤
│  Memory       │  Kernel (SQLite/Chroma) │
├─────────────────────────────────────────┤
│  Core         │  Identity + Security    │
└─────────────────────────────────────────┘
```

Eight packages, one system:

| Package | Purpose |
|---------|---------|
| `core` | Personal AI assistant, CLI, MCP server |
| `kernel` | Builder engine + durable memory stores |
| `forge` | Workflow orchestration engine |
| `quorum` | Decentralized agent coordination |
| `bootstrap` | FastAPI dashboard + HTMX UI |
| `pwa` | React 19 Progressive Web App |
| `contracts` | Canonical JSON schemas |
| `types` | Shared type definitions |

---

## Optional — Cloud Provider (Opt-In)

Animus defaults to 100% local inference. If you want Claude or GPT-4 as a fallback:

```bash
export ANIMUS_CLOUD_PROVIDER=anthropic
export ANTHROPIC_API_KEY=<your-anthropic-api-key>
```

Cloud providers are **never enabled automatically**.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, test commands, and pull request guidelines.

---

## License

MIT — see [LICENSE](LICENSE).

---

> Built with Claude Code. Human-reviewed. Human-verified.
