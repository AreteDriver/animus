# Animus Content

Autonomous write-to-earn content module for the Animus exocortex. Generates and publishes crypto/web3 content to platforms that reward authors with tips, tokens, or ad revenue.

## Features

- **LLM content generation** — drafts articles using Ollama or Anthropic
- **Multi-platform publishing** — Mirror.xyz, Paragraph.xyz, Medium (more planned)
- **Earnings tracking** — per-article, per-platform revenue aggregation
- **Topic discovery** — LLM-suggested trending topics with engagement estimates
- **Duplicate avoidance** — cooldown-based topic dedup across platforms
- **JSONL persistence** — lightweight, append-friendly local storage
- **Animus MCP tools** — 4 tools for integration with the exocortex
- **CLI** — Typer-based command interface

## Installation

```bash
pip install -e "packages/content/[dev]"
```

## Quick Start

```bash
# Initialize config
animus-content init

# Generate an article
animus-content generate "DeFi yield strategies in 2026" --platform mirror

# List articles
animus-content list

# Suggest topics
animus-content topics --platform paragraph --count 5

# View earnings
animus-content earnings
```

## Configuration

Config lives at `~/.animus/content/config.yaml`:

```yaml
llm:
  provider: ollama
  model: qwen2.5:14b
  base_url: http://localhost:11434
platforms:
  mirror:
    api_key: your-mirror-key
  paragraph:
    api_key: your-paragraph-key
  medium:
    api_key: your-medium-token
default_tags:
  - crypto
  - web3
  - blockchain
max_articles_per_day: 5
min_word_count: 500
```

## Supported Platforms

| Platform | Monetization | Status |
|----------|-------------|--------|
| Mirror.xyz | On-chain tips (ETH) | Adapter ready |
| Paragraph.xyz | Subscriber tips (MATIC) | Adapter ready |
| Medium | Partner Program revenue | Adapter ready |
| Hashnode | Planned | -- |
| Dev.to | Planned | -- |
| Substack | Planned | -- |

## MCP Tools

| Tool | Description |
|------|-------------|
| `content_generate` | Generate an article on a topic |
| `content_publish` | Publish to a platform |
| `content_earnings` | Earnings report |
| `content_topics` | Suggest trending topics |

## Testing

```bash
cd packages/content
pytest tests/ -v --cov=src/animus_content --cov-report=term-missing
```

## License

MIT
