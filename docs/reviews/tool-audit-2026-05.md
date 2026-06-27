# Animus Tool Surface Audit — 2026-05-15

> Track 3 of `PERSONAL_ROADMAP.md`. First systematic audit of the registered tool surface.
> **Headline finding:** usage-data persistence is broken — only 1 tool call recorded across the system. Audit is therefore intuition-based, not data-driven. **Fixing that is the prerequisite for every future audit.**

---

## Tool inventory

Two registries exist. They have substantial functional overlap.

### Core registry (10 tools) — `packages/core/animus/tools.py`

General-purpose tools available to any caller of the Core CLI / cognitive layer.

| Tool | Purpose |
|---|---|
| `get_datetime` | Current time |
| `read_file` | Read file contents |
| `list_files` | List directory |
| `run_command` | Execute shell |
| `write_file` | Write file |
| `edit_file` | Edit file by pattern |
| `web_search` | Web search |
| `http_request` | HTTP request (generic) |
| `(plus 2 more — need to read full BUILTIN_TOOLS list)` | — |

### Bootstrap intelligence registry (37 tools)

Per-domain tools for the intelligence layer / proactive engine / message gateway. Grouped by file:

| Module | Tools |
|---|---|
| `code_edit.py` | `code_list`, `code_patch`, `code_read`, `code_write` |
| `filesystem.py` | `file_read`, `file_write` |
| `forge_ctl.py` | `forge_invoke`, `forge_start`, `forge_status`, `forge_stop` |
| `gateway_tools.py` | `send_message` |
| `identity_tools.py` | `identity_append_learned`, `identity_list`, `identity_read`, `identity_write` |
| `memory_tools.py` | `recall_memory`, `set_reminder`, `store_memory` |
| `self_improve.py` | `analyze_behavior`, `apply_improvement`, `list_improvements`, `measure_impact`, `propose_improvement`, `rollback_improvement`, `self_improve_loop` |
| `system.py` | `shell_exec` |
| `task_ctl.py` | `task_complete`, `task_create`, `task_delete`, `task_list` |
| `timer_ctl.py` | `timer_cancel`, `timer_create`, `timer_fire`, `timer_list`, `timer_update` |
| `web.py` | `web_fetch`, `web_search` |

**Total: 47 tools across both registries.**

---

## Headline finding — usage data not persisted

`~/.local/share/animus/tool_history.db` exists and has the correct schema (`tool_history` table with `tool_name`, `success`, `duration_ms`, `timestamp`, `arguments`). Schema is fine. But it contains **exactly one row**: a `forge_status` call from 2026-04-30T22:17:43Z. That's 15 days ago and a single call.

The conclusion isn't "animus has been idle." The conclusion is **tool invocations aren't being persisted to `tool_history.db` from the actual tool dispatch path.** Possible causes:
- The persistence wire-in is only in one code path (the forge_ctl handler); other handlers skip the audit insert
- The persistence path is wrapped in a `try/except` that swallows errors silently
- The dispatcher used by the message gateway / CLI / MCP server bypasses the path entirely
- A schema mismatch is causing inserts to fail and be discarded

**Until this is fixed, every "is this tool used" question is unanswerable.** That makes future audits intuition-only, which is exactly the failure mode an audit is supposed to prevent.

### What IS being recorded

Other DBs have real data:

- `proactive_outcomes.db` / `check_fires` has 117 rows across 5 proactive checks:
  - `task_nudge` — 67 fires (most active)
  - `self_heal` — 20
  - `verdict_sync` — 16
  - `reflection` — 8
  - `morning_brief` — 6
- Memory says there should be **6** proactive checks; only **5** have fired. The 6th is either disabled, renamed, or has never met its trigger conditions.

### Prerequisite for future audits

Wire tool invocation persistence into every tool-dispatch surface:
1. Bootstrap intelligence `ToolExecutor` — insert into `tool_history` on every call (success and failure)
2. Core CLI / cognitive-layer tool dispatch — same
3. Forge tool-use loop — same
4. MCP server tool calls — same

Single source of truth. After ~30 days of real usage data, the audit becomes data-driven.

---

## Functional overlap (clear duplication)

Core and Bootstrap registries duplicate functionality with different names:

| Core tool | Bootstrap tool | Recommendation |
|---|---|---|
| `read_file` | `file_read` | Pick one. Bootstrap's path-aware version probably wins for the intelligence layer; Core's for the cognitive layer. **Deprecate one name; alias for back-compat.** |
| `write_file` | `file_write` | Same — pick one canonical name |
| `run_command` | `shell_exec` | Functional overlap but different safety surface (`shell_exec` likely has tighter sandbox). **Keep both if safety surfaces actually differ; if not, deprecate one.** |
| `web_search` (Core) | `web_search` (Bootstrap) | **Same name, two registrations.** Resolve by aliasing to the same implementation |
| `http_request` | `web_fetch` | Probably overlapping; verify and dedupe |

**Action:** consolidate to one canonical name per function. The "27% smaller surface" win matters because every tool definition the LLM sees is context burned on routing.

---

## Intuition-based bucketing

Without usage data, these are educated guesses based on tool description + likely workflow fit. Re-bucket after persistence is fixed and 30+ days of real data accrue.

### Bucket A — Almost certainly active workhorses

Tools that are core to daily operation:
- `recall_memory`, `store_memory` — exocortex foundation
- `read_file`, `write_file` / `file_read`, `file_write` — table stakes
- `web_search`, `web_fetch` — table stakes
- `task_create`, `task_list`, `task_complete` — daily task management
- `forge_status`, `forge_start`, `forge_stop` — operational control
- `send_message` — message gateway core

### Bucket B — Likely occasional

- `edit_file` / `code_patch` — used when actively editing
- `code_read`, `code_write`, `code_list` — coding sessions
- `shell_exec` / `run_command` — periodic
- `set_reminder`, `timer_create`, `timer_list` — sporadic
- `task_delete` — occasional cleanup
- `analyze_behavior`, `list_improvements` — self-improvement loop reads
- `morning_brief` (proactive — 6 fires) — daily check

### Bucket C — Probably rotting (needs verification)

Tools whose surface area suggests they may rarely fire:
- `identity_write`, `identity_append_learned` — identity is mostly auto-managed by reflection; manual writes rare
- `timer_update`, `timer_cancel`, `timer_fire` — if timers are mostly set-and-forget, the update/cancel/fire paths may be cold
- `task_delete` — usually `task_complete` instead of delete
- `propose_improvement`, `apply_improvement`, `rollback_improvement`, `measure_impact`, `self_improve_loop` — the self-improvement loop is autonomous; manual surface may be infrequent
- `forge_invoke` — Forge is usually invoked via systemd / API, not via this tool

### Bucket D — Definitely rotting or never-implemented

- The 6th proactive check that has never fired. Identify it. Either fix its trigger or remove it.

---

## Recommended actions

**Phase 1 — wire usage persistence (the prerequisite).** Until this is done, audits are guesswork.

1. Find the `ToolExecutor` dispatch path in Bootstrap intelligence
2. Identify why the `tool_history` insert isn't running for most calls
3. Wire the same persistence into Core's cognitive-layer tool dispatch
4. Wire it into Forge's tool-use loop
5. Add a test: any tool call must produce a `tool_history` row (success or failure)

**Effort:** ~3 hours. Single biggest leverage item from this audit.

**Phase 2 — overlap consolidation.**

1. Pick canonical names for: file read/write, web search, shell execution, web fetch / HTTP
2. Add deprecation aliases for back-compat (`file_read` → calls `read_file` or vice versa)
3. Update tool registry to expose one name per function
4. Net result: registry surface shrinks from 47 to ~40 tools without losing capability

**Effort:** ~2 hours.

**Phase 3 — re-audit after 30 days of usage data.**

1. Run this audit again with real `tool_history` data
2. Move tools from intuition-buckets to data-buckets
3. Prune Bucket D (zero usage, not core to operations)
4. Consider archiving Bucket C if still cold

**Effort:** ~1 hour after the data exists.

**Phase 4 — investigate the 6th proactive check.**

`check_fires` shows 5 distinct check names. Memory says 6 exist. Find the missing one in the proactive engine source, decide if it's:
- Misconfigured (won't fire under current conditions)
- Disabled in `config.toml`
- Never wired into the scheduler
- Already removed but referenced in memory

**Effort:** ~30 min.

---

## Audit-of-audit reflection

Lessons for the next time this audit runs:

- **The data layer was the load-bearing finding, not the tool list.** Going in I expected "X tools haven't been called in 90 days." What I found was "no tool data is being captured." That's a more important finding because it invalidates every quantitative claim about tool usage in animus.
- **Memory drift.** The "37 tools" claim in animus README and memory was correct for Bootstrap, but didn't account for Core's 10. Total surface is 47. Future tool-count claims should specify which registry.
- **Functional duplication is invisible until enumerated.** `file_read` vs `read_file` is the kind of drift that's silent until someone reads both registries side by side.

---

## Next quarterly run

Due: 2026-08-15 (or sooner if Phase 1 persistence work lands).

By that date, expect:
- `tool_history` populated with real usage
- Actual bucketing replacing the intuition guesses above
- Functional overlap already consolidated
- 6th-proactive-check resolved
