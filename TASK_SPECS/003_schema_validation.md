# TASK-003: Tool-Call Schema Validation

## Objective
Enforce Pydantic schema validation on every tool call emitted by an agent before it reaches the filesystem.

## Constraints
- Must work with both XML (Hermes) and JSON (OpenAI/Anthropic) formats.
- Must not add >50ms latency per validation call.
- Must raise a specific exception (`ContractViolation`) on failure with the exact field error.
- Budget: 700 ET.

## Inputs
- `packages/kernel/src/animus_kernel/tools/registry.py` (ToolRegistry, ToolDefinition)
- `packages/kernel/src/animus_kernel/contracts/validator.py`
- `packages/kernel/src/animus_kernel/agents/task_runner.py`

## Outputs
- `packages/kernel/src/animus_kernel/tools/schema_validator.py` (new)
- Updated `packages/kernel/src/animus_kernel/agents/task_runner.py`
- Updated `packages/kernel/src/animus_kernel/contracts/__init__.py`

## Acceptance Criteria
1. Invalid tool call (missing required arg) raises `ContractViolation` with `field_path="params.path"`.
2. Valid tool call passes through unchanged in < 50ms.
3. Supports nested Pydantic models in tool definitions.
4. 1000 validations run in < 1 second (benchmark script).
5. Works for `read_file`, `edit_file`, `run_command`, `search_code`.

## Rubric
- correctness [3.0] — catches invalid calls, passes valid ones.
- schema_valid [2.0] — aligns with ToolRegistry schema.
- concision [0.5] — no redundant validation layers.

## Exclusions
- No sandbox-level permission checks (those stay in `SafetyChecker`).
- No retry logic on validation failure.

## Dependencies
- BLOCKS: none
- BLOCKED_BY: TASK-002
