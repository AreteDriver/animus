# Animus Faucet

Autonomous crypto faucet farming module for the Animus exocortex.

## Features

- Multi-chain support (BTC, ETH, Base Sepolia, Sui Testnet, SOL, DOGE, LTC, MATIC)
- Three driver types: API, Browser (Playwright), CLI
- Vision-based captcha solving (Ollama LLaVA + Whisper audio fallback)
- Residential proxy rotation (Bright Data, custom lists)
- Human-like timing jitter and mouse movement
- Auto-disable on consecutive failures / bans
- Treasury consolidation (hot → cold wallet sweeps)
- Earnings tracking and reporting
- Discord webhook alerts
- Animus tool integration (MCP-compatible)

## Quick Start

```bash
# Install
pip install -e "packages/faucet/[all,dev]"

# Initialize config
animus-faucet init

# Add wallets
animus-faucet wallet add eth 0xYOUR_ADDRESS
animus-faucet wallet add sui_testnet 0xYOUR_SUI_ADDRESS

# Claim from all enabled faucets
animus-faucet claim

# Check status
animus-faucet status

# View earnings
animus-faucet earnings --days 30
```

## Configuration

Config lives at `~/.animus/faucet/faucets.yaml`. Run `animus-faucet init` to generate a starter config with known-good faucets.

## Architecture

```
animus_faucet/
├── drivers/          # Per-faucet interaction (API, Browser, CLI)
├── vision/           # Captcha solving (Ollama vision + Whisper audio)
├── wallet/           # Multi-chain wallet management
├── proxy/            # Residential IP rotation
├── tools/            # Animus tool definitions (MCP-compatible)
├── config.py         # YAML + env var configuration
├── scheduler.py      # Jittered claim orchestration
├── store.py          # Claim history + health persistence
├── treasury.py       # Balance tracking + consolidation
└── cli.py            # Typer CLI
```
