# TASK-002: ProviderRouter Role-Based Tuning

## Objective
Extend `ProviderRouter` to select provider and model by `AgentRole`, not just by task tier (REASONING, STANDARD, FAST).

## Constraints
- Backward compatible: existing `ProviderManager` and `TierRouter` must work unchanged.
- Must not add latency to the routing hot path (> 10ms per call).
- Must support offline mode: when no cloud keys, defaults to Ollama with role-specific models.
- Budget: 600 ET.

## Inputs
- `packages/kernel/src/animus_kernel/providers/router.py`
- `packages/kernel/src/animus_kernel/providers/manager.py`
- `packages/kernel/src/animus_kernel/agents/supervisor.py` (defines `AgentRole`)

## Outputs
- `packages/kernel/src/animus_kernel/providers/role_router.py` (new)
- Updated `packages/kernel/src/animus_kernel/providers/__init__.py`
- Updated `packages/kernel/src/animus_kernel/providers/router.py`

## Acceptance Criteria
1. `router.route(role=AgentRole.BUILDER, instruction="...")` returns a `RoutingDecision` pointing to the local Hermes model when offline.
2. `router.route(role=AgentRole.PLANNER, instruction="...")` returns a `RoutingDecision` pointing to Qwen (or best available reasoning model).
3. When `ANTHROPIC_API_KEY` is set, Builder → Claude, Planner → Claude, Tester → fast/cheap model.
4. If a role has no explicit mapping, falls back to `TierRouter` behavior (existing).
5. Routing latency measured via `time.perf_counter()` is < 10ms for 1000 calls.

## Rubric
- correctness [3.0] — routes match role expectations.
- schema_valid [1.5] — preserves existing `RoutingDecision` dataclass.
- actionability [1.0] — configuration is explicit and editable.

## Exclusions
- No dynamic model benchmarking (latencies learned at runtime).
- No cost-based routing beyond existing `BudgetManager`.
- No GPU detection logic (stays in `hardware.py`).

## Dependencies
- BLOCKS: TASK-003, TASK-005, TASK-008, TASK-009
- BLOCKED_BY: TASK-001
