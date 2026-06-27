# Animus Prospector

Autonomous crypto opportunity hunter for the Animus exocortex.

## Features

- **Multi-source hunting**: Aggregator sites, web search, Reddit, Discord, Twitter
- **LLM-powered evaluation**: Scam detection, ROI scoring, auto-action decisions
- **Faucet module integration**: Auto-discovers and registers new faucets
- **Deduplication**: SHA-256 URL hashing prevents duplicate processing
- **Heuristic + LLM two-stage evaluation**: Fast scam filtering before expensive LLM calls
- **Animus tool integration**: 3 MCP-compatible tools (scan, feed, health)

## Quick Start

```bash
pip install -e "packages/prospector/[all,dev]"

# Run a scan
animus-prospector scan

# View results
animus-prospector feed summary
animus-prospector feed actionable
animus-prospector feed faucets

# Check hunter health
animus-prospector health
```

## Architecture

```
animus_prospector/
├── hunters/              # Per-source opportunity discovery
│   ├── aggregator_hunter.py  # airdrops.io, Layer3, Galxe, etc.
│   ├── web_hunter.py         # Search engine discovery
│   └── social_hunter.py      # Reddit, Twitter monitoring
├── evaluator.py          # LLM-scored scam detection + ROI ranking
├── orchestrator.py       # Hunt → dedup → evaluate → act pipeline
├── feed.py               # Unified opportunity feed for consumers
├── store.py              # JSONL persistence with dedup
├── tools/                # Animus MCP tools
└── cli.py                # Typer CLI
```

## Hunters

| Hunter | Sources | Signal Quality |
|--------|---------|---------------|
| Aggregator | airdrops.io, DeFi Llama, Layer3, Galxe, Zealy, faucetlink | High — curated |
| Web | Search engines (configurable endpoint) | Medium — needs LLM filtering |
| Social | Reddit (5 subreddits), Twitter (with API key) | Low-Medium — noisy |
