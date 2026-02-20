# Animus Chat Agent — Next Steps

## Goal
Make Animus work like Claude Code: take ideas, autonomously build them.

## Phase 1: Wire chat.py to Animus Core
- [ ] Replace raw `urllib.request` → `CognitiveLayer` (Anthropic primary, Ollama fallback)
- [ ] Replace hand-rolled tool exec → `ToolRegistry` + `create_default_tools()`
- [ ] Add `MemoryLayer` so context persists across sessions
- [ ] Shrink chat.py from ~700 lines to ~50 (thin wrapper)

## Phase 2: Claude brain + Ollama hands
- [ ] Use Claude API for planning/reasoning/tool-selection (reliable tool calling)
- [ ] Use Ollama for cheap local subtasks (summarization, formatting, simple analysis)
- [ ] CognitiveLayer already supports this via `provider` + `fallback` args

## Phase 3: Forge workflows for "build X" tasks
- [ ] Create YAML workflow template: plan → code → lint → test → commit
- [ ] Wire ForgeEngine into chat agent for multi-step tasks
- [ ] Add quality gates (ruff + pytest must pass before continuing)
- [ ] Budget tracking so it doesn't burn unlimited tokens

## Phase 4: Constrained tool selection for Ollama-only mode
- [ ] Numbered menu approach instead of free-form tool calling
- [ ] Deterministic parsing of structured responses
- [ ] Fallback when no Claude API key is available

## Phase 5: Memory + Learning
- [ ] Store task outcomes in MemoryLayer (what worked, what broke)
- [ ] Recall similar past tasks for context enrichment
- [ ] Learning system detects patterns across sessions

## Context
- Animus Core already has: CognitiveLayer, ToolRegistry, MemoryLayer, ForgeEngine, SwarmEngine, 6 integrations, learning system, 40+ CLI commands
- chat.py currently bypasses ALL of it and talks raw HTTP to Ollama
- 7B-8B local models can't reliably do autonomous tool calling — need Claude for the brain
- See conversation from session 46 for full analysis
