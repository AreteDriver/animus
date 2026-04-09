# Animus Referral

Autonomous referral program stacking module for the Animus exocortex. Manages and maximizes referral programs across crypto platforms by cross-referring accounts and tracking bonuses.

## Features

- Registry of known referral programs (exchanges, DeFi, faucets)
- Account graph showing who referred whom
- Auto-generate referral links for new accounts
- Track which referrals converted and paid out
- Calculate optimal referral chains (A refers B, B refers C)
- Alert on high-value referral programs
- JSONL persistence for all data
- 3 Animus MCP tools for integration

## Installation

```bash
pip install -e "packages/referral/[dev]"
```

## CLI

```bash
# List referral programs
animus-referral programs

# Show earnings summary
animus-referral status --days 30

# Register an account
animus-referral add-account alice coinbase --code ALICE123

# Generate a referral link
animus-referral gen-link coinbase alice ALICE123

# Show referral graph
animus-referral graph

# Record a bonus
animus-referral record-bonus coinbase alice 10.0 --usd 10.0 --paid
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `referral_programs` | List or add referral programs |
| `referral_status` | Earnings summary with top programs |
| `referral_link` | Generate or retrieve referral links |

## Architecture

```
animus_referral/
├── __init__.py
├── config.py          # ReferralConfig (YAML-backed)
├── models.py          # ReferralProgram, Account, ReferralLink, ReferralBonus
├── programs/
│   ├── base.py        # BaseProgram abstract
│   ├── exchange.py    # CEX programs (Coinbase, Binance, Kraken, Bybit)
│   ├── defi.py        # DeFi protocols (Uniswap, Aave, 1inch)
│   └── faucet.py      # Faucet programs (FreeBitcoin, Cointiply, FireFaucet)
├── manager.py         # Cross-referral orchestration
├── tracker.py         # Earnings aggregation and reporting
├── store.py           # JSONL persistence
├── tools/
│   └── referral_tools.py  # Animus MCP tool definitions
└── cli.py             # Typer CLI
```

## Testing

```bash
cd packages/referral
pytest tests/ -v
pytest tests/ --cov=animus_referral --cov-report=term-missing
```

## License

MIT
