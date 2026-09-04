# Animus Quorum

**Decentralized multi-agent coordination without a supervisor bottleneck.**

Agents read a shared intent graph and self-adjust based on stability scores — no inter-agent messaging required. Includes triumvirate voting, flocking behaviors, and an optional Rust PyO3 backend for performance.

## Features

- **Intent graph** — Shared state that all agents can read and write
- **Stability scoring** — Consensus rate, alignment, and separation metrics
- **Triumvirate voting** — Decentralized decision making without a central coordinator
- **Flocking behaviors** — Emergent coordination patterns
- **Rust + Python** — Hot path in Rust via PyO3, Python bindings for ergonomics

## Install

```bash
pip install animus-quorum
```

Or from source:
```bash
pip install -e packages/quorum/
```

## Quick Start

```python
from animus_quorum import IntentNode, StabilityScorer

# Create an intent node
node = IntentNode(
    agent_id="agent-1",
    intent_type="propose",
    payload={"action": "deploy"},
)

# Score stability
scorer = StabilityScorer()
score = scorer.evaluate(node)
```

## Architecture

```
quorum/
├── python/animus_quorum/          # Python package: import animus_quorum
├── src/                        # Rust PyO3 core
├── tests/                      # 926 tests, 97% coverage
├── benches/                    # Benchmarks (Rust)
└── pyproject.toml
```

## Development

```bash
git clone git@github.com:AreteDriver/animus.git
cd animus/packages/quorum

# Python tests
pytest tests/ -v

# Rust tests
cargo test

# Rust benchmarks
cargo bench
```

Python 3.12+, Rust 1.75+.

## Part of the Animus Monorepo

- [Animus Core](https://github.com/AreteDriver/animus/tree/main/packages/core) — operating environment engine
- [Animus Forge](https://github.com/AreteDriver/animus/tree/main/packages/forge) — multi-agent orchestration
- [Animus Bootstrap](https://github.com/AreteDriver/animus/tree/main/packages/bootstrap) — system daemon

## License

MIT — 2026, AreteDriver
