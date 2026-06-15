# TASK-010: Kernel Integration Tests

## Objective
Port or write tests for all new kernel modules introduced in TASK-001 through TASK-009.

## Constraints
- Must run with `pytest` in < 30 seconds.
- No network calls (mock providers).
- Must achieve ≥ 80% coverage for new modules.
- Budget: 1,500 ET.

## Inputs
- All files from TASK-001 through TASK-009.
- `packages/kernel/scripts/verify_imports.py`
- Existing test patterns: `packages/types/tests/*.py`

## Outputs
- `packages/kernel/tests/conftest.py`
- `packages/kernel/tests/test_hermes_prompts.py`
- `packages/kernel/tests/test_role_router.py`
- `packages/kernel/tests/test_schema_validation.py`
- `packages/kernel/tests/test_command_runner.py`
- `packages/kernel/tests/test_terminal_agent.py`
- `packages/kernel/tests/test_fastapi_endpoint.py`
- `packages/kernel/tests/test_mobile_ui.py` (optional)
- `packages/kernel/tests/test_discord_bot.py` (mock)
- `packages/kernel/tests/test_ollama_default.py`

## Acceptance Criteria
1. `pytest packages/kernel/tests/` passes with ≥ 80% coverage on new files.
2. All critical imports from `verify_imports.py` have test coverage.
3. Mock provider (`MockProvider`) used for LLM calls — no real API usage.
4. `pytest --tb=short` completes in < 30 seconds.
5. CI-ready: runs with `python -m pytest` with no extra setup.

## Rubric
- correctness [3.0] — tests catch real bugs.
- schema_valid [1.5] — uses pytest idioms.
- concision [0.5] — no bloated fixtures.

## Exclusions
- No integration with external CI (GitHub Actions stays in shell tier).
- No load/stress tests.
- No property-based testing (Hypothesis).

## Dependencies
- BLOCKS: none
- BLOCKED_BY: all above (TASK-001 through TASK-009)
