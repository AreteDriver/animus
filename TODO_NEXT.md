# Animus — Next Roadmap

## Goal
Unify entry points, expose Animus as an MCP server, stabilize all packages.

## Phase 1: Core CLI Unification ✅
- [x] Merge `scripts/chat.py` agent loop into `packages/core/animus/__main__.py`
- [x] Keep old commands (40+ prompt-toolkit commands) as slash commands
- [x] Agent mode as default: natural language → `think_with_tools()` + approval callback
- [x] `/build`, `/model`, `/auto` available from main CLI
- [x] Dual-model routing: auto-detect ANTHROPIC_API_KEY, create fallback config
- [x] Task outcome tracking wired into main loop
- [x] Agent context (personality + memory recall + past outcomes) for all requests
- [x] Make `scripts/chat.py` a thin shim that imports `__main__`
- [x] Tests for unified CLI entry point (23 tests in test_cli_commands.py)

## Phase 2: MCP Server for Animus ✅
- [x] `packages/core/animus/mcp_server.py` — FastMCP server with 8 tools
- [x] Memory tools: `animus_remember`, `animus_recall`, `animus_search_tags`, `animus_memory_stats`
- [x] Task tools: `animus_list_tasks`, `animus_create_task`, `animus_complete_task`
- [x] Brief tool: `animus_brief` (context from memory)
- [x] Entry point: `python -m animus.mcp_server`
- [x] Optional dep: `pip install animus[mcp]`
- [x] 18 tests (skip gracefully without mcp package)
- [x] Add to Claude Code MCP config for live use (~/.claude/mcp.json)
- [x] Expose `animus_run_workflow` (trigger Forge pipelines via MCP)

## Phase 3: Bootstrap Stabilization ✅
- [x] All 1697 tests pass in venv
- [x] 96% coverage confirmed
- [x] Collection errors on system Python are dep-only (tomli-w, python-multipart)
- [x] Bootstrap test count updated in CLAUDE.md

## Phase 4: Release ✅
- [x] All packages: full test suites pass
- [x] Tag v2.2.0 pushed
- [x] GitHub Release created: https://github.com/AreteDriver/animus/releases/tag/v2.2.0
- [x] MEMORY.md updated with new test counts

## Phase 5: CLI Hardening & v2.3.0 ✅
- [x] Reconcile test counts across all packages (13,597 total)
- [x] MCP config validated (9 tools, ~/.claude/mcp.json)
- [x] Deprecation warning on scripts/chat.py
- [x] Graceful warnings for missing Ollama/API keys
- [x] Tag v2.3.0 pushed
- [x] GitHub Release created: https://github.com/AreteDriver/animus/releases/tag/v2.3.0

## Phase 6: Polish & PyPI ✅
- [x] Remove scripts/chat.py entirely
- [x] CI validation — all jobs green
- [x] MCP server auth (ANIMUS_MCP_API_KEY, write tools gated)
- [x] Build pipeline JSON retry (constrained tool format hint on malformed output)
- [x] Conversation persistence (last session context loaded at startup)
- [x] PyPI publish workflow for Core (publish-core.yml, triggered by core-v* tags)
- [x] Core README.md for PyPI listing
- [x] README.md updated with current numbers and MCP quickstart
- [x] Bootstrap coverage push (96→97%) — 1739 tests, 97.01% coverage
- [x] Configure PyPI OIDC Trusted Publisher for animus-core package
- [x] Tag core-v2.3.0 to trigger first PyPI publish (pip install animus-core)

## Context
- Core: 2108 tests, 97% coverage
- Forge: 8871 tests, 97% coverage
- Quorum: 926 tests, 97% coverage
- Bootstrap: 1739 tests (in venv), 97% coverage
- Chat Agent TODO complete (5/5 phases)
- Ollama live smoke test passing (deepseek-coder-v2 tool use works)

## Phase N: Whitepaper-audit follow-ups (2026-06-02)

> **Full remediation plan: [`docs/ROADMAP_TO_10.md`](docs/ROADMAP_TO_10.md)**
> — dependency-ordered, 8-dimension 10/10 scorecard, completeness matrix.
> The two items below are roadmap A1 + A2 (the deferred breaking changes);
> everything else is tracked in the roadmap. Working rule: when an item lands,
> flip its `CANON.md` status, tick the roadmap, and delete it here — same PR.

Context: branch `fix/p0-whitepaper-refinements` landed 5 P0 fixes from the
2026-06 whitepaper audit (egress unify, content-taxonomy wiring, eval exec
isolation, ET opt-in ceiling, tier-contract docs + `recall_for_egress`).
Two items were intentionally deferred because they are breaking design
decisions, not clean fixes:

- [ ] **Flip Effective-Tokens to the DEFAULT enforced budget unit.** The opt-in
      ceiling shipped (`BudgetConfig.effective_token_budget`, off by default).
      Making ET the default redefines what every workflow's `token_budget:`
      YAML field means (raw tokens → cost-weighted ET) and will touch a large
      slice of the 9.7k forge tests + 30+ workflow definitions. Decide the
      semantics first: (a) reinterpret `total_budget` as ET, or (b) keep raw
      `total_budget` and auto-derive `effective_token_budget` from it. Then
      migrate workflows + tests in one deliberate pass. **This is the "flip".**
- [ ] **`BudgetManager.allocate()` real reservation** (whitepaper refinement #5,
      P1). Currently `allocate()` checks `can_allocate` but records nothing, so
      parallel auto-steps can each pass and collectively overspend. Track
      pending allocations; decrement on `record_usage` or explicit release.
      Pairs naturally with the ET flip since both touch the accounting path.
- [ ] Optional: route the 3 MCP egress sites through `MemoryLayer.EGRESS_SCOPE`
      once test mocks are updated (left as literal `{Sensitivity.PUBLIC}` for
      now to keep MagicMock-based tests simple).
- [ ] `_restore_from_db()` does not rebuild `_total_effective` (DB stores raw
      tokens only); ET under-counts after a session restore. Fine while ET is
      opt-in; fix when the flip lands.
