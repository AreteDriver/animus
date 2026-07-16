# Project Status

| Attribute | Value |
|-----------|-------|
| Status | **OPERATIONAL** |
| Last verified | 2026-07-16 |
| Installable? | Yes — see [Quickstart](README.md#quickstart) |
| Tested? | 2928/2928 regression tests green |
| Documented? | Yes — [MkDocs site](https://aretedriver.github.io/animus/) |
| CI | [![CI](https://github.com/AreteDriver/animus/workflows/CI/badge.svg)](https://github.com/AreteDriver/animus/actions/workflows/ci.yml) |
| First-Run | [![First-Run](https://github.com/AreteDriver/animus/workflows/First-Run%20Verification/badge.svg)](https://github.com/AreteDriver/animus/actions/workflows/first-run.yml) |

## What Works

- **Kernel:** 190/190 tests green — session persistence, checkpoint/resume, continuity guarantee
- **Head:** 83/83 core tests green — intent parsing, planning, synthesis, quality gates
- **Citizens:** 71/71 tests green — Architect, Conversation Designer, Knowledge Curator, Test Oracle
- **Forge Integration:** Kill Criterion #3 satisfied — authenticated workflow registration and execution
- **Daemon Mode:** P0–P3 complete — ResourceGuard, SessionManager, TaskScheduler, signal-safe shutdown
- **Discovery:** P5 complete — MCP scanner, OpenAPI ingestion, annotated script discovery
- **Docs:** MkDocs site built with 0 warnings, deployed to GitHub Pages
- **Benchmarks:** 6 kernel benchmarks tracked in CI with regression alerts

## What Doesn't Work Yet

- External adoption evidence thin — no verified installs by users outside build loop (see [#115](https://github.com/AreteDriver/animus/issues/115))
- macOS support on roadmap; Windows out of scope
- PyPI package not yet published for `animus-core` (install from source only)

## Install

```bash
git clone https://github.com/AreteDriver/animus.git && cd animus
pip install -e packages/types/ -e "packages/core/[dev]" -e packages/kernel/ -e packages/contracts/
python -m animus.cli architect --focus codebase
```

> Full walkthrough: [`docs/user-scenarios/v2.3-first-run.md`](docs/user-scenarios/v2.3-first-run.md)
