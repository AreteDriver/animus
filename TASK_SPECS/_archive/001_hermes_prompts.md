# TASK-001: Hermes Prompt Templates

## Objective
Add Hermes-optimized system prompts and function-calling format to `kernel/agents/prompts/hermes/`.

## Constraints
- ≤ 500 lines total across all prompt files.
- Must use Hermes XML function-calling format: `<tool_call><name>...</name><params>...</params></tool_call>`.
- Must include 6 role prompts: Planner, Builder, Tester, Reviewer, Architect, Documenter.
- Each prompt must include a strict `SYSTEM` block that locks the agent to its role.
- Budget: 800 ET.

## Inputs
- Existing prompt modes: `packages/forge/prompts/modes/*.md`
- Existing agent prompts: `packages/kernel/src/animus_kernel/agents/supervisor.py` (cached prompts)
- Hermes documentation: https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-70B (function calling format)
- OpenAI function calling schema (for dual-format support)

## Outputs
- `packages/kernel/src/animus_kernel/agents/prompts/hermes/planner.xml`
- `packages/kernel/src/animus_kernel/agents/prompts/hermes/builder.xml`
- `packages/kernel/src/animus_kernel/agents/prompts/hermes/tester.xml`
- `packages/kernel/src/animus_kernel/agents/prompts/hermes/reviewer.xml`
- `packages/kernel/src/animus_kernel/agents/prompts/hermes/architect.xml`
- `packages/kernel/src/animus_kernel/agents/prompts/hermes/documenter.xml`
- `packages/kernel/src/animus_kernel/agents/prompts/hermes/__init__.py`
- `packages/kernel/src/animus_kernel/agents/prompts/hermes/README.md`

## Acceptance Criteria
1. All 6 role prompts are importable via `from animus_kernel.agents.prompts.hermes import get_role_prompt`.
2. Each prompt contains a `<system>` block that explicitly states the agent's role and available tools.
3. Each prompt's tool list matches the `ToolRegistry` schema (see `kernel/tools/registry.py`).
4. A sample conversation using the Builder prompt generates valid XML `tool_call` tags for `read_file` and `edit_file`.
5. `personal-quality` rubric score ≥ B (0.80) on a sample conversation evaluated by Reviewer prompt.

## Rubric
- correctness [3.0] — prompts compile, import, and format tags validate.
- format_compliance [2.0] — strict XML schema for Hermes; JSON schema available for dual-format.
- non-genericity [1.5] — role-specific instructions, not generic "AI assistant" language.

## Exclusions
- No GUI / dashboard integration.
- No Discord-specific formatting.
- No translation to languages other than English.
- No multi-turn conversation state management (state lives in `AgentContext`).

## Dependencies
- BLOCKS: TASK-002
- BLOCKED_BY: none

## Notes
Hermes expects the system prompt to declare tools in a specific XML block. The model will then emit `<tool_call>` tags instead of raw text. The `AgentTaskRunner` must parse these tags and dispatch to `FilesystemTools`.
