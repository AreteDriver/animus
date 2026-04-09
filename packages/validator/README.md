# Animus Validator

Autonomous validator and node rewards management module for the Animus exocortex.

## Features

- **Multi-chain staking position tracking** — Sui, Ethereum (Lido/Rocket Pool), Solana
- **Node health/uptime monitoring** — alerts on downtime and slashing
- **Auto-compound rewards** — restake when rewards exceed gas threshold
- **APY comparison** — compare validators across networks
- **Opportunity discovery** — find new incentivized testnets and node programs
- **Reward history and yield reporting** — track earnings over time
- **Animus MCP tools** — 4 tools for integration with the exocortex

## Installation

```bash
pip install -e ".[dev]"
```

## CLI

```bash
animus-validator positions          # List staking positions
animus-validator rewards --days 30  # Earnings report
animus-validator discover           # Find staking opportunities
animus-validator health             # Check network health
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `validator_positions` | List all staking positions |
| `validator_rewards` | Earnings report |
| `validator_compound` | Trigger compounding |
| `validator_discover` | Find new staking opportunities |

## Architecture

```
animus_validator/
├── config.py           # ValidatorConfig (YAML + env vars)
├── models.py           # Node, StakePosition, RewardRecord, CompoundAction, NetworkInfo
├── networks/
│   ├── base.py         # BaseNetwork abstract
│   ├── sui.py          # Sui staking via JSON-RPC
│   ├── ethereum.py     # ETH staking (Lido, Rocket Pool)
│   └── solana.py       # SOL native staking
├── monitor.py          # Node health monitoring
├── compounder.py       # Auto-compound rewards
├── discovery.py        # Find new staking opportunities
├── tracker.py          # Rewards tracking and reporting
├── store.py            # JSONL persistence
├── tools/
│   └── validator_tools.py  # Animus MCP tools
└── cli.py              # Typer CLI
```

## Testing

```bash
pytest tests/ -v --cov=animus_validator --cov-report=term-missing
```

## License

MIT
