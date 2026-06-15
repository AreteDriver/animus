# TASK-005: TerminalAgent Loop

## Objective
Wire `SupervisorAgent`, `FilesystemTools`, and `CommandRunner` into an iterative build loop: read → plan → edit → test → retry.

## Constraints
- Max 10 iterations per task.
- Budget gate after each iteration.
- Must checkpoint state after each step.
- Must roll back on failure.
- Budget: 1,200 ET.

## Inputs
- `packages/kernel/src/animus_kernel/agents/supervisor.py`
- `packages/kernel/src/animus_kernel/tools/filesystem.py`
- `packages/kernel/src/animus_kernel/builder/command_runner.py`
- `packages/kernel/src/animus_kernel/sandbox/rollback.py`
- `packages/kernel/src/animus_kernel/budget/manager.py`

## Outputs
- `packages/kernel/src/animus_kernel/builder/terminal_agent.py` (new)
- `packages/kernel/src/animus_kernel/builder/__init__.py`

## Acceptance Criteria
1. `terminal_agent.build("Add OAuth to gatekeeper", project_path=...)` returns `BuildResult(success, files_changed, tests_passed, et_consumed)`.
2. Loop stops when tests pass or max iterations reached.
3. Each iteration logs ET consumption to `BudgetManager`.
4. Failed builds roll back to pre-build git state using `RollbackManager`.
5. Checkpoint saves `iteration_count`, `files_touched`, `test_results` after each step.

## Rubric
- correctness [3.0] — loop terminates, produces correct code.
- actionability [2.0] — clear BuildResult, recoverable on failure.
- schema_valid [1.5] — fits existing kernel dataclass patterns.

## Exclusions
- No long-running background tasks (terminal agent is foreground).
- No distributed multi-machine builds.
- No auto-commit to git (waits for human approval).

## Dependencies
- BLOCKS: TASK-006
- BLOCKED_BY: TASK-003, TASK-004
