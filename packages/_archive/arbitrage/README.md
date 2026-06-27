# Animus Arbitrage

Autonomous DEX arbitrage detection and execution module for the Animus exocortex.

## Features

- Multi-DEX, multi-chain price monitoring (Uniswap V3, Jupiter, Cetus, and more)
- Gas-adjusted profit calculation
- Configurable minimum spread thresholds
- Risk management: max trade size, daily loss limits, cooldown periods, concurrent trade limits
- **DRY_RUN mode by default** — logs opportunities without executing trades
- P&L tracking and reporting
- JSONL persistence for opportunities and trade results
- Animus MCP tool integration (4 tools)
- CLI interface

## Installation

```bash
# Core (no chain adapters)
pip install -e packages/arbitrage/

# With EVM support (Uniswap, Sushiswap, Curve)
pip install -e "packages/arbitrage/[evm]"

# With Sui support (Cetus, Turbos)
pip install -e "packages/arbitrage/[sui]"

# Everything
pip install -e "packages/arbitrage/[all]"

# Development
pip install -e "packages/arbitrage/[dev]"
```

## Usage

### CLI

```bash
# Scan for opportunities
animus-arbitrage scan

# Scan with specific trade size
animus-arbitrage scan --size 500

# View P&L status
animus-arbitrage status

# View configuration
animus-arbitrage config

# Execute opportunity (requires live mode)
animus-arbitrage execute <opportunity_id>
```

### MCP Tools

Four tools are registered under the `arbitrage` category:

| Tool | Description |
|------|-------------|
| `arbitrage_scan` | Scan for current arbitrage opportunities |
| `arbitrage_status` | P&L summary and risk status |
| `arbitrage_config` | View/update risk parameters |
| `arbitrage_execute` | Execute a specific opportunity (requires live mode) |

### Safety

**Live execution requires explicit opt-in.** The default mode is `DRY_RUN`, which logs detected opportunities without touching any chain. To enable live trading:

```yaml
# ~/.animus/arbitrage/config.yaml
execution:
  mode: live
  max_trade_size_usd: 100.0
```

Or via environment variable:

```bash
export ARBITRAGE_EXECUTION_MODE=live
```

## Architecture

```
animus_arbitrage/
├── __init__.py          # Public API
├── config.py            # ArbitrageConfig with chains, thresholds, wallets
├── models.py            # TokenPair, PriceQuote, ArbitrageOpportunity, TradeResult
├── feeds/
│   ├── base.py          # BaseFeed abstract
│   ├── uniswap.py       # Uniswap V3 subgraph
│   ├── jupiter.py       # Jupiter (Solana) aggregator
│   └── cetus.py         # Cetus (Sui) DEX
├── detector.py          # Cross-DEX spread detection
├── executor.py          # Trade execution (dry-run by default)
├── risk.py              # Risk management
├── tracker.py           # P&L tracking
├── store.py             # JSONL persistence
├── tools/
│   └── arbitrage_tools.py  # Animus MCP tools
└── cli.py               # Typer CLI
```

## Testing

```bash
cd packages/arbitrage
pytest tests/ -v --cov=animus_arbitrage --cov-report=term-missing
```

## License

MIT
