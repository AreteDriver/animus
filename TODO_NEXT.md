# Animus — Next Roadmap

## Goal
Unify entry points, expose Animus as an MCP server, stabilize all packages.

## Phase 1: Core CLI Unification
- [ ] Merge `scripts/chat.py` agent loop into `packages/core/animus/__main__.py`
- [ ] Keep old commands (40+ prompt-toolkit commands) as slash commands
- [ ] Agent mode as default: natural language → CognitiveLayer + ToolRegistry
- [ ] `/build`, `/model`, `/stats`, `/workflow` available from main CLI
- [ ] Remove `scripts/chat.py` (or make it a thin shim that imports `__main__`)
- [ ] Tests for unified CLI entry point

## Phase 2: MCP Server for Animus
- [ ] New module: `packages/core/animus/mcp_server.py`
- [ ] Expose as MCP tools: `animus_remember`, `animus_recall`, `animus_search_memory`
- [ ] Expose as MCP tools: `animus_list_tasks`, `animus_create_task`, `animus_complete_task`
- [ ] Expose as MCP tools: `animus_run_workflow` (trigger Forge pipelines)
- [ ] Expose as MCP tools: `animus_entity_search`, `animus_brief`
- [ ] MCP resources: memory stats, task list, entity graph
- [ ] Entry point: `python -m animus.mcp_server` or `animus --mcp`
- [ ] Tests with MCP test client
- [ ] Add to Claude Code MCP config for live use

## Phase 3: Bootstrap Stabilization
- [ ] Verify all 1697 tests pass in venv
- [ ] Run coverage check, ensure ≥96%
- [ ] Fix any real failures (collection errors are dep-only)
- [ ] Update Bootstrap test count in CLAUDE.md

## Phase 4: Release
- [ ] All packages: run full test suites
- [ ] Tag v2.2.0
- [ ] GitHub Release with changelog
- [ ] Update MEMORY.md with new test counts

## Context
- Core: 2046 tests, 97% coverage
- Forge: 8874 tests, 97% coverage
- Quorum: 920 tests, 97% coverage
- Bootstrap: 1697 tests (in venv), 96% coverage
- Chat Agent TODO complete (5/5 phases)
- Ollama live smoke test passing (deepseek-coder-v2 tool use works)
