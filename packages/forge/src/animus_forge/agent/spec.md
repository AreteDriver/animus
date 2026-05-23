# Specification — Animus Agent Platform v0

**Status:** v0.1.0 — 2026-05-23 (decisions locked)
**Mode:** /specification (decision committed; do not re-litigate)
**Related:** [[project-animus-agent-platform]] (memory: `project_animus_agent_platform.md`)
**Implementation start gate:** After Quorum v2 wk5 re-eval (per Resolved Decision RD7)

---

## 1. Summary

`animus-agent run <task>` is a one-shot autonomous agent CLI that takes a single natural-language task, executes it via a free-form tool-use loop against a pure-local LLM (Qwen3-32B via Ollama), and emits a structured JSON receipt. It ships as a submodule inside Animus Forge (`packages/forge/src/animus_forge/agent/`), inherits Forge's Provider abstraction + BudgetManager + audit log, and wraps `smolagents.CodeAgent` with a Claude-Code-pattern skill registry (loading all 135 skills from `~/.claude/skills/`) and a pre/post-tool-use hook gate system. It exists to baseline a pure-local agent's task-success rate, cost/throughput envelope, and 30-project fleet autonomy against Claude Code — the resulting eval suite lands in `arete-evals`.

---

## 2. Requirements

### MUST

1. Provide CLI entry `animus-agent run <task>` accepting a positional task string and the flags enumerated in §4.1.
2. Execute the task via ReAct-style tool-use loop terminating on: agent declares completion, max-turns exceeded, token budget exceeded, wall-time exceeded, or unrecoverable tool error after one retry.
3. Route every LLM call through Forge's `BudgetManager`. For local Ollama calls, BudgetManager records `usd_cost = 0.0` and persists `tokens_in`, `tokens_out`, `wall_seconds`, and (when available via OS sensor) `watt_hours` in dedicated compute-metric columns alongside the existing USD column. Every call appears in the same audit log used by other Forge workflows.
4. Use Ollama as the LLM runtime, calling `qwen3:32b` by default; model overridable via `--model` and `FORGE_AGENT_MODEL` env (precedence: CLI > env > config > default).
5. Expose exactly 5 starter tools to the agent: `ReadFile`, `WriteFile`, `EditFile`, `Bash`, `Grep` (interfaces in §4.2–§4.6).
6. Implement a Skill Registry that loads all skills from `~/.claude/skills/` at agent startup and resolves them by name. The registry is **independent** of Animus's existing YAML skill resolver — v0 reads Claude Code skill format (markdown + YAML frontmatter) only. Registry MUST load all 135 currently-installed skills without raising.
7. Implement Hook Gates with 4 lifecycle events: `agent_start`, `pre_tool_use`, `post_tool_use`, `agent_end`. Hooks loaded from `~/.config/animus/agent/hooks.d/*.py`; a `pre_tool_use` returning `Deny(reason)` MUST abort the tool call and feed the denial back into agent context.
8. Emit a JSON receipt (schema §4.9) to stdout on completion. When `--receipt <path>` is set, ALSO write to that path.
9. Return exit code 0 only on `success`. Non-success exit codes: 1 = max_turns, 2 = budget_exceeded, 3 = wall_time_exceeded, 4 = tool_error, 5 = agent_quit, 6 = internal_error.
10. Constrain `ReadFile`/`WriteFile`/`EditFile`/`Grep` to the directory tree of `--project` (default cwd). Absolute paths escaping the root MUST raise `ToolError`. `Bash` cwd = project root.
11. Apply a default-deny `Bash` filter blocking: `rm -rf /`, `sudo`, `mkfs`, `dd if=`, raw `curl | sh`, any command containing literal `~` or `$HOME`. Filter list configurable in `~/.config/animus/agent/bash_policy.yaml`.
12. Live in `packages/forge/src/animus_forge/agent/`. CLI registered as `animus-agent` entry-point in `packages/forge/pyproject.toml`.
13. Wire a Constitutional Principles (P1–P9) validator call site as a built-in `pre_tool_use` hook. **v0 validator is a stub** that returns `Allow` unconditionally and logs `"P1-P9 validator stubbed; enforcement deferred to follow-on hardening spec"`. Validator module path, hook registration, and call signature MUST be in place so the follow-on spec only swaps the implementation.
14. Ship a built-in `pre_tool_use` hook (`agent.hooks.builtin.identity_guard`) that denies `WriteFile`/`EditFile` on any path inside identity-file roots: `packages/core/animus/CORE_VALUES.md`, `packages/core/animus/identity/`, `packages/bootstrap/.../persona/` (canonical list maintained in `agent/identity_roots.py`). Hook is fail-closed and runs before all user-supplied hooks. Override only via explicit user hook returning `Allow`.
15. On agent startup, pause `animus.service` via `systemctl --user stop animus.service` (logs warning if `systemctl` unavailable, e.g. macOS — uses `launchctl` equivalent there or logs `"service coordination unavailable; race risk"`). On agent exit (success OR failure path), restart via `systemctl --user start animus.service` in a `finally` block. Pause/restart failures MUST NOT block agent execution.

### MAY

16. Cache loaded skills in-memory for the agent's lifetime.
17. Support `--dry-run` flag using a MockProvider with canned tool calls (skill/hook testing without LLM cost).
18. Emit Prometheus metrics (`animus_agent_turns_total`, `animus_agent_tokens_total`, `animus_agent_wall_seconds`, `animus_agent_tool_calls_total{tool,status}`) when `FORGE_METRICS_ENABLED=1`.
19. Support `--verbose` flag streaming turn-by-turn agent reasoning to stderr.
20. Support `--no-service-pause` flag to disable R15 service coordination (for users not running `animus.service` or for nested test runs).

### MUST NOT

21. MUST NOT make any network call to Anthropic, OpenAI, or any cloud LLM provider during the agent loop.
22. MUST NOT modify files outside `--project` root (enforced by tool-level path validation, not convention).
23. MUST NOT execute Bash commands as root, with sudo, or via privilege escalation.
24. MUST NOT write to memboot, Animus memory, or any persistent store outside the receipt file and audit log without explicit user-provided write tools (no implicit memory side effects in v0).
25. MUST NOT spawn sub-agents in v0 (Claude Code's `Agent` tool deferred to v1+).
26. MUST NOT modify identity-file roots even with user hook override unless the override is explicit (R14).

---

## 3. Constraints

| # | Constraint | Measurement Method |
|---|------------|--------------------|
| C1 | Python 3.12 (Forge requirement) | `python --version` ≥ 3.12; CI matrix pins 3.12 |
| C2 | New dep: `smolagents >= 1.0` | `pip show smolagents` succeeds; `import smolagents` succeeds in forge venv |
| C3 | New dep: `ollama` client in Forge deps | `import ollama` succeeds in forge venv |
| C4 | Skill loader loads all installed skills | `tests/agent/test_skill_registry.py::test_loads_all_installed_skills` asserts `len(registry.list()) >= 135` |
| C5 | Test coverage ≥ 95% (Forge package gate) | `pytest --cov=src/animus_forge/agent tests/agent/ --cov-fail-under=95` passes |
| C6 | Lint clean | `ruff check packages/forge/src/animus_forge/agent/ packages/forge/tests/agent/` exits 0 |
| C7 | Format clean | `ruff format --check` on same paths exits 0 |
| C8 | All LLM calls flow through BudgetManager with token+time tracking | Grep audit: no direct `ollama.Client().chat(...)` outside `Provider.complete()`; assertion test confirms BudgetManager receives every recorded call AND every record has non-null `tokens_in`/`tokens_out`/`wall_seconds` |
| C9 | First-token latency on Qwen3-32B Q4_K_M @ 32 GB rig | Benchmark in arete-evals records first-token latency; v0 bar ≤ 8 s p50, ≤ 15 s p95 |
| C10 | End-to-end "hello world" task (write file, read back, confirm content) wall ≤ 60 s | `tests/agent/test_e2e_hello.py` measures wall, fails if > 60 s |
| C11 | Audit log entries in JSON Lines at `~/.local/share/animus/forge/audit/agent-YYYY-MM-DD.jsonl` | File existence + valid JSONL parse on test run |
| C12 | Hook denial latency overhead ≤ 50 ms per `pre_tool_use` call | `tests/agent/test_hook_perf.py` benchmark |
| C13 | Identity-file roots list maintained in `agent/identity_roots.py` and exported | `from animus_forge.agent.identity_roots import IDENTITY_ROOTS` succeeds; list is non-empty |
| C14 | Service coordination is non-blocking | `tests/agent/test_service_coordination.py::test_systemctl_failure_does_not_block` runs agent in a mocked environment where `systemctl` returns 1, asserts agent loop still completes |

---

## 4. Interfaces

Internal type names (`AgentContext`, `ToolResult`, `Receipt`, `Allow`, `Deny`, `Skill`) live in `packages/forge/src/animus_forge/agent/types.py`. Their concrete dataclass shapes are implementation detail beyond what is specified here.

### 4.1 CLI — `animus-agent run`

```
animus-agent run <task>
  [--project <path>]            default: cwd
  [--model <model>]             default: qwen3:32b
  [--max-turns <int>]           default: 50
  [--budget-tokens <int>]       default: 200_000
  [--wall-seconds <int>]        default: 1800 (30 min)
  [--receipt <path>]            default: stdout-only
  [--dry-run]                   use MockProvider, no Ollama call
  [--verbose | -v]              stream reasoning to stderr
  [--hook-dir <path>]           default: ~/.config/animus/agent/hooks.d
  [--no-service-pause]          skip systemctl pause/restart (R20)
```

**Inputs consumed:** task string, project tree, Ollama daemon at `OLLAMA_HOST` (default `http://localhost:11434`), skill files in `~/.claude/skills/`, hook files in `--hook-dir`.
**Outputs emitted:** JSON receipt to stdout, optional file at `--receipt`, audit log entries, optional Prometheus metrics, exit code 0–6.
**Error cases:**
- Ollama daemon unreachable → exit 6, receipt status `internal_error`, error names the daemon URL
- Model not pulled in Ollama → exit 6, error tells user to `ollama pull qwen3:32b`
- Skill load failure (≥ 1 skill fails to parse) → log warning, continue with loaded subset; receipt records `skill_load_warnings`
- Hook file syntax error → exit 6, error names the file
- `--project` path nonexistent → exit 6 before agent loop starts
- `systemctl stop` failure → log warning, continue (per R15)

### 4.2 ReadFile

```python
ReadFile(path: str, offset: int = 0, limit: int | None = None) -> str
```
Path must resolve within `--project` root (escape → `ToolError("path escapes project root")`). Non-existent → `ToolError("file not found: {path}")`. Non-UTF-8 binary → `ToolError("binary file: {path}")`. `limit=None` reads to EOF.

### 4.3 WriteFile

```python
WriteFile(path: str, content: str) -> str   # "wrote {n} bytes to {path}"
```
Path constraint per ReadFile. Creates parent directories. Overwrites without warning (v0 — no confirmation step). Subject to identity-file deny hook (R14).

### 4.4 EditFile

```python
EditFile(path: str, old: str, new: str) -> str   # "replaced 1 occurrence"
```
Requires single exact-match of `old`. Multi-match → `ToolError("multiple matches; refine old")`. Zero match → `ToolError("string not found")`. Path constraint per ReadFile. Subject to identity-file deny hook (R14).

### 4.5 Bash

```python
Bash(command: str, timeout_s: int = 30) -> dict
# {"exit_code": int, "stdout": str, "stderr": str, "wall_seconds": float}
```
cwd = project root. `timeout_s` clamped silently to 300. stdout/stderr each truncated at 8 KB with marker `[... truncated, N bytes total]`. Denied commands → `ToolError("command denied by policy: {reason}")` before execution.

### 4.6 Grep

```python
Grep(pattern: str, path: str = ".", glob: str = "**/*", case_insensitive: bool = False) -> list[dict]
# [{"file": str, "line": int, "text": str}, ...]
```
Python `re` regex. Result list capped at 100; over → first 100 + `_truncated: True`. Path constraint per ReadFile.

### 4.7 Skill Loader

**Input:** `~/.claude/skills/` tree.
**Output:** `dict[str, Skill]` keyed by skill name.
```python
@dataclass
class Skill:
    name: str
    description: str
    body: str
    invokable: bool
    source_path: Path
```
**Discovery:** any directory under `~/.claude/skills/` containing `SKILL.md` or `skill.md`. YAML frontmatter parsed for `name`, `description`, `invokable` (default `True`).
**Failure handling:** per-skill load errors logged + skipped; startup continues. Aggregate count surfaced in `--verbose` + audit log + receipt's `skill_load_warnings`.
**Scope:** This loader is independent of Animus's existing YAML skill resolver (RD3). Unification deferred to follow-on work.

### 4.8 Hook Gates

Hook files in `~/.config/animus/agent/hooks.d/*.py` define functions with reserved names:

```python
def agent_start(context: AgentContext) -> None: ...
def pre_tool_use(context: AgentContext, tool: str, args: dict) -> Allow | Deny: ...
def post_tool_use(context: AgentContext, tool: str, args: dict, result: ToolResult) -> None: ...
def agent_end(context: AgentContext, receipt: Receipt) -> None: ...
```

**Execution order for `pre_tool_use`:**
1. Built-in identity-file deny hook (R14) — runs first, fail-closed
2. Built-in P1–P9 validator hook (R13, stubbed in v0) — second
3. User hooks from `--hook-dir`, alphabetical filename order — third

First `Deny` short-circuits. User hook returning `Allow` after a built-in `Deny` does NOT override the built-in.

Functions discovered by name (any subset definable). Hook raising exception → tool call denied (fail-closed); error logged + appended to receipt's `hook_errors`.

### 4.9 Receipt JSON Schema

```json
{
  "task": "string",
  "status": "success|max_turns|budget_exceeded|wall_time_exceeded|tool_error|agent_quit|internal_error",
  "turns_taken": "integer",
  "tokens_in": "integer",
  "tokens_out": "integer",
  "wall_seconds": "number",
  "model": "string",
  "tool_calls": [
    {"turn": "integer", "tool": "string", "args_summary": "string", "status": "ok|error|denied", "wall_ms": "number"}
  ],
  "skill_load_warnings": ["string"],
  "hook_errors": ["string"],
  "final_message": "string",
  "error": "string|null",
  "started_at": "ISO8601",
  "ended_at": "ISO8601"
}
```

---

## 5. Acceptance Criteria

Every criterion maps to a pytest test, a benchmark, or an arete-evals artifact.

1. **A1 — CLI exists and runs:** `animus-agent run "echo hello world"` exits 0 within 60 s on default rig with Ollama + Qwen3-32B installed. Verified by `tests/agent/test_e2e_hello.py`.
2. **A2 — Receipt schema valid:** Receipt validates against §4.9 schema. Verified by `tests/agent/test_receipt_schema.py` using `jsonschema`.
3. **A3 — Skill loader covers all installed:** `len(registry.list()) >= 135` against current `~/.claude/skills/`. Verified by `tests/agent/test_skill_registry.py::test_loads_all_installed_skills`.
4. **A4 — All 5 tools functionally complete:** Each tool has `test_<tool>_happy_path.py` + `test_<tool>_path_escape_denied.py` + `test_<tool>_<specific_error>.py`. All pass.
5. **A5 — User hook gate denies tool call:** Fixture hook in `tests/agent/fixtures/hooks/deny_bash.py` returns `Deny("test")` on `pre_tool_use` for `Bash`; agent loop continues and receipt records denial. Verified by `tests/agent/test_hook_deny.py`.
6. **A6 — BudgetManager integration with compute metrics:** Every LLM call appears in audit log with non-null `tokens_in`, `tokens_out`, `wall_seconds`. Verified by `tests/agent/test_budget_integration.py` asserting count + field non-null.
7. **A7 — Pure-local network constraint:** Test runs agent with `socket.create_connection` mocked to refuse non-localhost; agent completes a file-manipulation task. Verified by `tests/agent/test_pure_local_no_network.py`.
8. **A8 — Path escape blocked:** Agent attempts `ReadFile("/etc/passwd")` from project rooted at `/tmp/proj/`; tool returns `ToolError`. Verified by `tests/agent/test_path_escape.py`.
9. **A9 — Bash policy enforced:** Agent attempts `Bash("rm -rf /")`; policy denies before execution; receipt records denial. Verified by `tests/agent/test_bash_policy.py`.
10. **A10 — Exit code semantics:** Parametrized test runs synthetic scenarios for each terminal status; exit code matches §2 R9. Verified by `tests/agent/test_exit_codes.py`.
11. **A11 — Coverage gate:** `pytest --cov=src/animus_forge/agent --cov-fail-under=95 tests/agent/` passes.
12. **A12 — Eval suite scaffolded in arete-evals:** `~/projects/arete-evals/suites/animus-agent-v0/` contains ≥ 10 task YAMLs in `benchgoblins-ask` format. Task definitions authored in separate `/specification` pass (RD6). Verified by `git ls-files arete-evals | grep "suites/animus-agent-v0/.*\.yaml" | wc -l >= 10`.
13. **A13 — First eval run scored:** `animus-forge eval run animus-agent-v0 --rubric code-edit --prompt-version v0.0.1` produces a result JSON. Score not gated for v0 — generation IS the criterion. Verified by file existence + valid JSON.
14. **A14 — Performance bars per C9, C10, C12:** Benchmark suite reports first-token latency, e2e hello task wall, and hook denial overhead within stated bounds.
15. **A15 — Documentation:** `packages/forge/src/animus_forge/agent/README.md` exists with: install, first-run walkthrough, tool reference, hook authoring guide, skill format, identity-root list, service-coordination behavior. Reviewed manually; not test-gated.
16. **A16 — Identity-file deny hook fires:** Agent attempts `WriteFile` on a path inside `IDENTITY_ROOTS`; built-in hook denies before execution; receipt records denial; identity file unchanged on disk. Verified by `tests/agent/test_identity_guard.py`.
17. **A17 — Service coordination round-trips:** Agent startup invokes `systemctl --user stop animus.service`; on exit (both success and failure paths), invokes `systemctl --user start animus.service`. Verified by `tests/agent/test_service_coordination.py` using mocked subprocess. Failure of either MUST be logged and MUST NOT block agent.
18. **A18 — P1–P9 stub wired:** `pre_tool_use` flow shows the P1–P9 validator hook being called for every tool invocation; stub returns `Allow` and emits the documented log line. Verified by `tests/agent/test_p1_p9_stub.py` capturing log output.

---

## 6. Out of Scope

- Sub-agent spawning (Claude Code `Agent` tool) → v1+
- Overnight-delegate queue with checkpoint/resume → v1
- Always-on daemon, file-watcher triggers, cron, MCP-server task ingestion → v2
- Multi-model routing (OllamaProvider + MLXProvider + router)
- MLX runtime / Mac-native serving → deferred until M5 hardware (October 2026)
- Web/HTTP fetch tool
- Memboot / Animus-memory write tools (no implicit memory side effects in v0)
- IDE plugin / VS Code integration
- Multi-user / multi-tenant
- Streaming partial results to caller (final receipt only)
- Auto-recovery retry-with-fix loop beyond single retry on tool error
- Cost-tracking dashboard UI (CLI + audit-log only)
- Sandboxing beyond path + bash-policy (no containers, no chroot)
- **Full P1–P9 validator implementation** (per RD2 — deferred to follow-on hardening spec)
- **Unifying agent skill loader with Animus's existing YAML skill resolver** (per RD3 — independent in v0)
- **Eval-suite task definitions** (per RD6 — separate `/specification` pass before v0 ships)
- **Lock-file coordination with `self_heal`** (per RD5 — v0 uses systemctl pause; lock-file approach is future option)

---

## 7. Resolved Decisions

Decisions locked 2026-05-23 by user choice. Listed here for traceability; reasoning preserved with the decision.

- **RD1 — BudgetManager semantics:** Record `usd_cost = 0.0` for local Ollama calls; track `tokens_in`, `tokens_out`, `wall_seconds`, `watt_hours` in dedicated compute-metric columns alongside USD. Forge dashboard works unchanged; schema migration adds nullable columns. Implemented in R3, C8, A6.
- **RD2 — P1–P9 enforcement:** Stub validator for v0 — wire call site, return `Allow` unconditionally, log stub-notice. Real implementation deferred to follow-on hardening spec. Implemented in R13, A18, OOS.
- **RD3 — Skill format compatibility:** v0 loader is independent and reads Claude Code format only. Unification with Animus's YAML resolver deferred. Implemented in R6, §4.7, OOS.
- **RD4 — IdentityProposalManager interaction:** Built-in `pre_tool_use` deny hook on identity-file paths. Fail-closed, runs before user hooks. Override requires explicit user hook returning `Allow`. Implemented in R14, R26, §4.8 ordering, A16.
- **RD5 — `self_heal` coexistence:** Pause `animus.service` via `systemctl --user stop` on agent startup; restart in `finally` block on exit. Non-blocking — `systemctl` failures are logged warnings, never abort. `--no-service-pause` flag disables for nested test runs. Implemented in R15, R20, C14, A17.
- **RD6 — Eval-suite task definitions:** Authored in separate `/specification` pass before v0 ships. A12 keeps the count gate (≥ 10 tasks); task content is the separate spec's deliverable. Recorded in A12 + OOS.
- **RD7 — Quorum v2 vs Agent v0 sequencing:** Sequence after Quorum v2 wk5 re-eval gate (currently wk1–2). Quorum learnings may shape agent design; no parallel-package contention. Recorded as Implementation start gate in header.
- **RD8 — Directory naming:** Keep `agent/` (singular). Semantically distinct from `agents/` (orchestration); cosmetic collision accepted as low-cost.

---

End of spec.
