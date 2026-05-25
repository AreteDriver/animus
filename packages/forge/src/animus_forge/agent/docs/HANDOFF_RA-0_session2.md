# RA-0 Handoff — pick-up notes for session 2

**Branch:** `feat/animus-agent-loop` (4 commits ahead of main)
**Session 1 end:** 2026-05-25 (early hours)
**RA-0 progress:** 6 of 12 sub-tasks done; 5 of 18 spec acceptance criteria green

---

## First three actions when you sit back down

```bash
# 1. STOP the service before anything else (gotcha rule — see
#    memory project_animus_dev_gotchas). It MAY already be stopped
#    from session 1; this is idempotent and free.
systemctl --user stop animus.service

# 2. Confirm you're on the right branch + nothing has whipsawed.
cd ~/projects/animus
git checkout feat/animus-agent-loop
git log --oneline -5

# 3. Sanity check the existing work still passes.
cd packages/forge
./.venv/bin/python -m pytest tests/agent/ -q
# Expect: 163 passed
```

If step 2 shows you're NOT on `feat/animus-agent-loop`, self_heal whipsawed
you again. Read `project_animus_dev_gotchas` §1 mid-session-recovery block.

---

## What landed in session 1

| Commit | Scope |
|---|---|
| `54a73a4` | Spec v0.1.2 — RD9 amendment: LlamaCpp/qwen3.6-research default, RD7 overridden |
| `3925e65` | Foundation — `types.py`, `identity_roots.py`, smolagents dep + 42 tests |
| `a82a6f0` | 5 starter tools (Read/Write/Edit/Bash/Grep) + 71 tests |
| `9258b13` | Skill registry + hook gates + identity guard + P1-P9 stub + 50 tests |

163 tests, **97.3% coverage** on the agent package, ruff lint + format clean.

Acceptance criteria green: **A3, A4, A5, A16, A18** (5 of 18).

---

## Remaining work — in dependency order

### #19 — Extend BudgetManager with compute-metric columns (RD1)

**Files to touch:**
- `packages/forge/src/animus_forge/budget/manager.py` — add `wall_seconds`, `watt_hours`, `usd_cost` to `UsageRecord` (lines 27-46 currently have `tokens`, `input_tokens`, `output_tokens`, `cache_read_tokens`, `model`)
- `packages/forge/src/animus_forge/budget/manager.py` lines 182-192 — extend `_persist_usage` to write the new fields
- `packages/forge/migrations/` — add a new SQL migration adding nullable columns to `budget_session_usage`. Check the highest existing migration number first.
- `tests/test_budget*` — add tests asserting new fields are persisted

**Scope:** ~50 lines code + ~30 lines test. Maybe 30-45 min.

**Why now:** Task #20 (agent loop) calls Provider.complete which records to BudgetManager — easier to land the schema change first so the agent loop just uses it.

**Spec refs:** R3, RD1, A6, C8

### #20 — Core agent loop (THE BIG ONE)

**Files to create:**
- `packages/forge/src/animus_forge/agent/loop.py` — orchestrator
- `packages/forge/src/animus_forge/agent/llm_client.py` — thin Provider adapter (LlamaCpp default, Ollama alternate per RD9)
- `packages/forge/src/animus_forge/agent/tool_dispatch.py` — maps tool name → tool function with hook chain integration (this is where `run_pre_tool_use` actually gates execution)
- `packages/forge/tests/agent/test_loop*.py` + `test_tool_dispatch.py`

**Approach (proposed — confirm before building):** Bypass smolagents.CodeAgent if its constraints fight us. We have all the building blocks already (skill registry, hooks, tools, types). A custom ReAct loop is ~150 lines and gives full control over termination conditions per R2 (agent declares complete / max-turns / budget / wall-time / tool error after one retry). smolagents was the original plan, but spec doesn't mandate using its CodeAgent class specifically — just that smolagents is a dep (C2). Re-evaluate at loop-design time.

**Termination matrix per R2:**
| Condition | Status | Exit code |
|---|---|---|
| Agent declares completion | success | 0 |
| max_turns exceeded | max_turns | 1 |
| budget_tokens exceeded | budget_exceeded | 2 |
| wall_seconds exceeded | wall_time_exceeded | 3 |
| Tool error after retry | tool_error | 4 |
| Agent quits voluntarily | agent_quit | 5 |
| Anything else | internal_error | 6 |

**Spec refs:** R2, R3, R4, R5, R7, R10, A1, A6, A7, A10, C8, C9, C10

**Scope:** ~250-350 lines code + ~150-200 lines test. Probably the full second session by itself.

### #21 — CLI `animus-agent run`

**Files to create:**
- `packages/forge/src/animus_forge/agent/cli.py` — argparse, env precedence, invoke loop, format receipt, return exit code
- Entry point in `packages/forge/pyproject.toml` `[project.scripts]` section: `animus-agent = "animus_forge.agent.cli:main"`
- `tests/agent/test_cli.py` — exit code semantics + receipt-to-stdout + `--receipt` file output

**Flag set per spec §4.1 (amended for RD9):**
```
--project, --provider, --model, --max-turns, --budget-tokens, --wall-seconds,
--receipt, --dry-run, --verbose, --hook-dir, --no-service-pause
```

**Env precedence (CLI > env > config > default):** `FORGE_AGENT_PROVIDER`, `FORGE_AGENT_MODEL`

**Error cases per §4.1:** Ollama unreachable / model not pulled / skill load failure / hook syntax error / `--project` nonexistent / `systemctl stop` failure — each has a specific exit code + message shape.

**Scope:** ~150 lines code + ~120 lines test. ~1 hour.

**Spec refs:** R1, R8, R9, §4.1, A1, A2, A10

### #22 — systemd service coordination (R15)

**Files to create:**
- `packages/forge/src/animus_forge/agent/service_coord.py` — `pause_animus_service()` + `restart_animus_service()` context manager
- `tests/agent/test_service_coord.py` — mock subprocess, assert failure of systemctl doesn't block agent

**Behavior:** `systemctl --user stop animus.service` on agent startup, restart in `finally` block on exit (success OR failure). Failures logged at WARN, NEVER block. `--no-service-pause` flag disables.

**Scope:** ~40 lines code + ~50 lines test. ~30 min.

**Spec refs:** R15, R20, C14, A17

### #23 — End-to-end acceptance run (all 18 ACs)

**Files to create:**
- `tests/agent/test_e2e_hello.py` — A1, A10 (write file, read back, exit 0 within 60 s on default rig)
- `tests/agent/test_receipt_schema.py` — A2 (jsonschema validation against §4.9)
- `tests/agent/test_pure_local_no_network.py` — A7 (mock socket.create_connection, agent completes file task)
- `tests/agent/test_path_escape.py` — A8 (ReadFile /etc/passwd → ToolError)
- `tests/agent/test_bash_policy.py` — A9 (rm -rf / denied)
- `tests/agent/test_exit_codes.py` — A10 (parametrized: each terminal status → correct exit)
- `tests/agent/test_budget_integration.py` — A6 (every LLM call recorded with non-null compute fields)
- `tests/agent/test_p1_p9_stub.py` — A18 (log capture verifies stub fired)
- `tests/agent/test_hook_perf.py` — A14 (C12 hook denial ≤ 50ms overhead)
- `packages/forge/src/animus_forge/agent/README.md` — A15 (install / first-run / tool reference / hook authoring / skill format / identity roots / service coordination)
- Eval suite stub at `~/projects/arete-evals/suites/animus-agent-v0/` — A12 (≥ 10 task YAMLs; A13 first run scored)

**Final gates:**
- 95% coverage on `agent/` per C5
- ruff check + format clean per C6/C7
- e2e hello task wall ≤ 60s per C10
- All 18 ACs explicitly mapped to a pytest test or benchmark

**Scope:** ~400 lines test + ~200 lines README + eval suite. 2-3 sessions on its own.

---

## Operating notes / gotchas

1. **animus.service stays stopped between sessions.** Don't restart it at session end. Just leave it. Saves you from re-triggering the whipsaw next time.
2. **`harden/stage-1-p0-quickwins` branch exists** from self_heal's 2026-05-25 whipsaw. Empty (same HEAD as main). Delete with `git branch -D harden/stage-1-p0-quickwins` whenever convenient — not urgent.
3. **`.claude/worktrees/agent-a6a25bb3` worktree exists** for unclear reasons (possibly auto-created). `git worktree list` shows it. Remove with `git worktree remove .claude/worktrees/agent-a6a25bb3` if not in use.
4. **Spec amendment RD9 is on this branch only** — when this branch eventually merges, the amendment goes to main with it.
5. **127 skill floor (C4)** is satisfied today. Re-check at v0 ship if you've added/removed skills since — bump or accept the regression.
6. **PyYAML is in forge venv** (6.0.3) — confirmed available for skill_registry.
7. **`smolagents` is installed** (1.25.0) but NOT YET IMPORTED anywhere in agent code. Decision deferred to loop-design session — if CodeAgent fights us, we have a clean exit (custom loop).
8. **LlamaCpp server runs as systemd unit `llama-qwen.service`** (port 11435) — independent of animus.service, won't be whipsawed.

---

## Memory entries to glance at before starting

- `project_animus_dev_gotchas` — gotcha rule (TIGHTENED 2026-05-25)
- `project_research_assistant_roadmap` — overall RA-0 → RA-4 plan
- `project_animus_agent_platform` — original direction commitment
- `feedback_dont_stop_unless_blocking` — terminal status only on real blockers
- `feedback_meta_rule_hardening` — when a rule breaks twice, tighten in place

---

## Suggested session-2 budget

- **30 min:** boot, sanity tests pass, task #19 (BudgetManager extension)
- **60-90 min:** task #20 (agent loop — the big one)
- **45 min:** task #21 (CLI)
- **30 min:** task #22 (service coordination)
- **plus 2-3 follow-on sessions:** task #23 (full e2e acceptance, eval suite, README)

If only one chunk fits: **do task #20** (agent loop). #19 is easy to fit alongside. #21/#22 are small enough to land in a third session.

---

End of handoff.
