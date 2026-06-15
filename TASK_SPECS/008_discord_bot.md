# TASK-008: Discord Slash Commands

## Objective
Lightweight Discord bot with `/build` commands for status, approve, queue.

## Constraints
- No conversation memory. Stateless.
- Reads from kernel events via `AgentMessageBus`.
- Must start with only `DISCORD_TOKEN` env var.
- Must send rich embeds.
- Budget: 900 ET.

## Inputs
- `packages/kernel/src/animus_kernel/agents/message_bus.py`
- `packages/kernel/src/animus_kernel/budget/manager.py`
- `packages/kernel/src/animus_kernel/sandbox/orchestrator.py`

## Outputs
- `packages/kernel/src/animus_kernel/channels/discord_bot.py` (new)
- `packages/kernel/src/animus_kernel/channels/__init__.py`

## Acceptance Criteria
1. `/build status` returns embed with active builds + budget remaining + ET today.
2. `/build approve job-id` calls `ApprovalGate.approve(job_id)` and returns confirmation.
3. `/build queue project` lists pending tasks with priority and ET estimate.
4. Bot auto-shards if added to >1 guild.
5. Error messages are user-friendly (no stack traces in Discord).

## Rubric
- correctness [3.0] — commands respond accurately.
- actionability [2.0] — user can approve from phone.
- schema_valid [1.0] — embed format matches Discord API.

## Exclusions
- No DM support (guild channels only).
- No role-based permissions (any user can /build).
- No conversation context (stateless).

## Dependencies
- BLOCKS: none
- BLOCKED_BY: TASK-002
