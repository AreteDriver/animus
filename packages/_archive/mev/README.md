# Animus MEV

Autonomous MEV (Maximal Extractable Value) detection and extraction module for the Animus exocortex. Monitors mempool activity and DEX trades on L2 chains where competition is lower.

## Features

- **WebSocket mempool monitoring** — real-time pending transaction feeds
- **DEX trade event parsing** — Uniswap V3 Swap event detection
- **Backrun strategy** — capture price impact from large DEX swaps
- **Sandwich detection** — monitoring only, execution is blocked by design
- **Trade simulation** — estimate profitability before execution
- **Private transaction submission** — Flashbots Protect on L2s
- **Gas budget management** — daily limits, per-tx caps, concurrent tx limits
- **P&L tracking** — real-time profit/loss by chain and strategy type
- **JSONL persistence** — append-only storage with deduplication
- **Animus MCP tools** — 3 tools for integration with the exocortex

## Ethical Constraints

- **DRY_RUN by default** — live execution requires explicit `execution.live = true`
- **Sandwich execution BLOCKED** — detection/monitoring only, never executes sandwich attacks
- **L2 focus** — Base, Arbitrum, Optimism where competition is lower

## Installation

```bash
pip install -e "packages/mev/[dev]"

# With EVM support (web3)
pip install -e "packages/mev/[evm]"
```

## Usage

### CLI

```bash
# Initialize config
animus-mev init

# Check status
animus-mev status

# View opportunities
animus-mev opportunities
```

### Configuration

Config lives at `~/.animus/mev/config.yaml`:

```yaml
chains:
  base:
    enabled: true
    min_swap_usd: 1000.0
  arb:
    enabled: true
  op:
    enabled: true

execution:
  live: false        # DRY_RUN default
  private_tx: true

risk:
  max_gas_per_tx_usd: 5.0
  daily_gas_budget_usd: 50.0
  min_profit_usd: 0.50
  min_confidence: 0.7
  sandwich_execution_blocked: true  # ALWAYS
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `MEV_DATA_DIR` | Data directory (default: `~/.animus/mev`) |
| `MEV_EXECUTION_LIVE` | Enable live execution (`true`/`false`) |
| `MEV_BASE_RPC_URL` | Base chain RPC endpoint |
| `MEV_BASE_WS_URL` | Base chain WebSocket endpoint |
| `MEV_ARB_RPC_URL` | Arbitrum RPC endpoint |
| `MEV_OP_RPC_URL` | Optimism RPC endpoint |
| `MEV_DISCORD_WEBHOOK` | Discord webhook for alerts |

## Architecture

```
animus_mev/
├── config.py          # MEVConfig — chains, execution, risk, monitoring
├── models.py          # MempoolTx, MEVOpportunity, ExtractionResult, Chain, MEVType
├── watchers/
│   ├── base.py        # BaseWatcher abstract
│   ├── mempool.py     # WebSocket mempool subscription
│   └── dex_trades.py  # DEX Swap event log subscription
├── strategies/
│   ├── base.py        # BaseStrategy abstract
│   ├── backrun.py     # Backrun large DEX trades
│   └── sandwich.py    # Sandwich detection (MONITOR ONLY)
├── simulator.py       # Trade simulation and profit estimation
├── executor.py        # Bundle submission (DRY_RUN default)
├── risk.py            # Gas budget, confidence, concurrent tx limits
├── tracker.py         # P&L tracking by chain and strategy
├── store.py           # JSONL persistence with dedup
├── tools/
│   └── mev_tools.py   # 3 Animus MCP tools
└── cli.py             # Typer CLI
```

## Testing

```bash
cd packages/mev
pytest tests/ -v
pytest tests/ -v --cov=animus_mev --cov-report=term-missing
```

## License

MIT
