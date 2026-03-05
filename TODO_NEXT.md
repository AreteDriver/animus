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

## Context
- Core: 2103 tests, 97% coverage
- Forge: 8871 tests, 97% coverage
- Quorum: 926 tests, 97% coverage
- Bootstrap: 1697 tests (in venv), 96% coverage
- Chat Agent TODO complete (5/5 phases)
- Ollama live smoke test passing (deepseek-coder-v2 tool use works)
