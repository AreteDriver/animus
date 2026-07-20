# Project Status

| Attribute | Value |
|-----------|-------|
| Status | **OPERATIONAL** |
| Last verified | 2026-07-19 |
| Installable? | Yes — see [Quickstart](README.md#quickstart) |
| Tested? | 3566/3566 regression + contracts + integration + database + PWA tests green |
| Documented? | Yes — [MkDocs site](https://aretedriver.github.io/animus/) |
| CI | [![CI](https://github.com/AreteDriver/animus/workflows/CI/badge.svg)](https://github.com/AreteDriver/animus/actions/workflows/ci.yml) |
| First-Run | [![First-Run](https://github.com/AreteDriver/animus/workflows/First-Run%20Verification/badge.svg)](https://github.com/AreteDriver/animus/actions/workflows/first-run.yml) |

## What Works

- **Kernel:** 357/357 tests green (+167 new) — memory stores (durable, local, chroma, layer), tier management, types, redaction
- **Head:** 83/83 core tests green — intent parsing, planning, synthesis, quality gates
- **Citizens:** 71/71 tests green — Architect, Conversation Designer, Knowledge Curator, Test Oracle
- **Forge Integration:** Kill Criterion #3 satisfied — authenticated workflow registration and execution
- **Daemon Mode:** P0–P3 complete — ResourceGuard, SessionManager, TaskScheduler, signal-safe shutdown
- **Discovery:** P5 complete — MCP scanner, OpenAPI ingestion, annotated script discovery
- **Docs:** MkDocs site built with 0 warnings, deployed to GitHub Pages
- **Benchmarks:** 6 kernel benchmarks tracked in CI with regression alerts
- **Type Check:** Mypy ratchet now blocking CI (1,418 errors baselined across 4 packages)
- **Contracts:** All 25 JSON schemas validated in CI with 116 tests
- **Pre-commit:** Ruff + hooks configured for local quality gates
- **Database:** 22/22 tests green — migration execution, upgrade/downgrade idempotency, connection handling
- **PWA:** 25/25 Vitest tests green — auth, API client, retry logic, WebSocket scaffold
- **Integration:** 14/14 tests green — package imports, cross-package types, memory + contracts workflow

## What Doesn't Work Yet

- External adoption evidence thin — no verified installs by users outside build loop (see [#115](https://github.com/AreteDriver/animus/issues/115))
- macOS support on roadmap; Windows out of scope
- PyPI package not yet published for `animus-core` (install from source only)
- Mypy error count >0 (baselined and ratcheted, but not zero)

## Install

```bash
git clone https://github.com/AreteDriver/animus.git && cd animus
pip install -e packages/types/ -e "packages/core/[dev]" -e packages/kernel/ -e packages/contracts/
python -m animus.cli architect --focus codebase
```

> Full walkthrough: [`docs/user-scenarios/v2.3-first-run.md`](docs/user-scenarios/v2.3-first-run.md)
