# Animus Chat Agent — Next Steps

## Goal
Make Animus work like Claude Code: take ideas, autonomously build them.

## Phase 1: Wire chat.py to Animus Core ✅
- [x] Replace raw `urllib.request` → `CognitiveLayer` (Anthropic primary, Ollama fallback)
- [x] Replace hand-rolled tool exec → `ToolRegistry` + `create_default_tools()`
- [x] Add `MemoryLayer` so context persists across sessions
- [x] Shrink chat.py from ~700 lines to ~373 (thin wrapper over Core)

## Phase 2: Claude brain + Ollama hands ✅
- [x] `classify_task()` — route heavy tasks (planning, code gen) to Claude, light tasks (summarize, format) to Ollama
- [x] `CognitiveLayer.delegate_to_local()` — explicitly route subtask to local model
- [x] `CognitiveLayer.think_routed()` — auto-route based on task classification
- [x] `create_local_think_tool()` — tool that lets Claude offload cheap subtasks to Ollama during agentic loop
- [x] `CognitiveLayer.has_dual_models` property
- [x] `/model` slash command for routing visibility
- [x] 52 tests covering classify_task, delegate_to_local, think_routed, local_think tool

## Phase 3: Forge workflows for "build X" tasks ✅
- [x] `build_task.yaml` — 4-agent pipeline: planner → coder → verifier → fixer
- [x] New archetypes: `planner`, `coder`, `verifier` in ForgeAgent
- [x] Planner uses `read_file`/`list_files` tools to understand codebase before planning
- [x] Coder uses `read_file`/`write_file`/`edit_file`/`list_files` to implement changes
- [x] Verifier uses `run_command` to run ruff + pytest
- [x] Fixer uses `read_file`/`edit_file`/`run_command` to fix issues
- [x] Quality gate after fixer: halt if PASS not in output
- [x] Budget tracking: $2.00 max per build task
- [x] `/build <description>` slash command injects task into pipeline
- [x] Task description injected into planner system prompt
- [x] 27 tests covering archetypes, workflow loading, agent execution, gate validation

## Phase 4: Constrained tool selection for Ollama-only mode ✅
- [x] `get_numbered_menu()` — presents tools as numbered list instead of free-form JSON
- [x] `_think_with_tools_constrained()` — deterministic loop using `TOOL: N` + `key: value` format
- [x] `_parse_constrained_tool()` — parses numbered selection + key:value params
- [x] `_strip_tool_lines()` — cleans TOOL: lines from final response (TOOL:0 preserves content)
- [x] Dispatch updated: Ollama/Mock/OpenAI all route to constrained loop
- [x] Old markdown `\`\`\`tool` loop preserved as `_think_with_tools_markdown()` (not dispatched)
- [x] 26 tests covering menu generation, parsing, strip logic, constrained loop, dispatch
- [x] Existing tests updated to constrained format (3 tests migrated)

## Phase 5: Memory + Learning
- [ ] Store task outcomes in MemoryLayer (what worked, what broke)
- [ ] Recall similar past tasks for context enrichment
- [ ] Learning system detects patterns across sessions

## Context
- Animus Core already has: CognitiveLayer, ToolRegistry, MemoryLayer, ForgeEngine, SwarmEngine, 6 integrations, learning system, 40+ CLI commands
- chat.py currently bypasses ALL of it and talks raw HTTP to Ollama
- 7B-8B local models can't reliably do autonomous tool calling — need Claude for the brain
- See conversation from session 46 for full analysis
