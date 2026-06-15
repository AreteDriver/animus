# TASK-004: CommandRunner

## Objective
Build a general-purpose subprocess runner with timeout, output capture, and safety gating.

## Constraints
- Must block dangerous commands (`rm -rf /`, `sudo`, shell injection).
- Must capture stdout/stderr up to configurable byte limit.
- Must respect `Sandbox` file/line limits.
- Must support async execution.
- Budget: 600 ET.

## Inputs
- `packages/kernel/src/animus_kernel/sandbox/sandbox.py`
- `packages/kernel/src/animus_kernel/tools/safety.py`
- `packages/kernel/src/animus_kernel/utils/validation.py`

## Outputs
- `packages/kernel/src/animus_kernel/builder/command_runner.py` (new)

## Acceptance Criteria
1. `run("pytest", cwd="/tmp/test")` returns `CommandResult(exit_code, stdout, stderr, duration_ms)`.
2. `run("rm -rf /")` raises `SecurityError` before execution.
3. `run("sleep 100", timeout=1)` kills process and returns `CommandResult(timeout=True)`.
4. Async version `async def arun(...)` works identically.
5. Output truncated at 10MB byte limit without error.

## Rubric
- correctness [3.0] — commands run and return accurate results.
- schema_valid [1.5] — clean `CommandResult` dataclass.
- hallucination_safety [2.5] — blocks dangerous commands robustly.

## Exclusions
- No persistent command history.
- No pseudo-TTY support.
- No shell emulation (bash/zsh syntax not parsed).

## Dependencies
- BLOCKS: TASK-005
- BLOCKED_BY: none
