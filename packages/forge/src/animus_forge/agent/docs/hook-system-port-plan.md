# Hook System Port Plan — Claude Code → Animus v0 Agent Hooks

**RE Task:** #3 (per `re-task-3-scope.md`)
**Status:** Draft v1.0 — 2026-05-23
**Parent spec:** `../spec.md` v0.1.1 — R7, R13, R14, R20, R26, §4.8, RD2, RD4, C12, A5, A16, A17, A18
**Sibling artifacts:** `cc-loop-port-plan.md` §7 (basic sketch — this doc supersedes), `skill-format-port-plan.md` (InvokeSkill bodies issue tool calls that hooks gate)
**Author:** Claude Code (advisory mode per [[project-animus-agent-platform]])

---

## 0. Sources Consulted

| Source | Location | Purpose |
|---|---|---|
| `../spec.md` v0.1.1 §4.8 + R7, R13, R14, R20, R26 | Hook lifecycle interface contract | Drives all design |
| `cc-loop-port-plan.md` §7 + matrix row 19 | Initial sketch; smolagents has NO hook gate interface (`monitoring.py:81-110` observer-only) | Confirms invent path |
| Claude Code 2.0 prompt `~/projects/system-prompts-and-models-of-ai-tools/Anthropic/Claude Code 2.0.txt:146` | "Users may configure 'hooks', shell commands that execute in response to events like tool calls... Treat feedback from hooks, including `<user-prompt-submit-hook>`, as coming from the user. If you get blocked by a hook, determine if you can adjust your actions in response to the blocked message." | CC reference shape |
| smolagents `agents.py:1309` (`_step_stream`) | Intercept point for subclass strategy | Required because smolagents has no native gate API |
| `packages/forge/src/animus_forge/workflow/arete_hooks.py` (204 lines) | Existing `AreteHooks` class — workflow-level observation hooks (`on_step_failure`, `on_workflow_complete`) | Naming collision to flag; different scope |
| `packages/forge/src/animus_forge/coordination/identity_patch.py` | Existing `IdentityPatchGate` — Forge→Core mutation approval gate | Reuse for routing identity edits in future; not v0 (RD4 picks "deny at hook layer" not "route through gate") |
| `packages/forge/forge/identity_anchor.yaml` | Canonical immutable-fields list: `CORE_VALUES.md`, `CONSTITUTIONAL_PRINCIPLES.md`, `mysoul.md` | Source of truth for `IDENTITY_ROOTS` (R14) |
| Animus monorepo `CLAUDE.md` Non-Negotiables #2 (identity), #4 (audit log), #8 (P1-P9 docs) | Constraint envelope | Backs RD4 and R13 design |

---

## 1. Lifecycle Event Matrix

Per §4.8 the agent's hook system has 4 events. Each event maps to: (CC equivalent) × (smolagents intercept point or "INVENT") × (Animus v0 required behavior) × status. Row count must include the built-in hook rows to satisfy AC2 (≥ 12).

Status legend: **C** = copy verbatim; **A** = adapt; **I** = invent.

| # | Event / Hook | CC Equivalent | smolagents Intercept | Animus v0 Required Behavior | Status | Cite |
|---|---|---|---|---|---|---|
| 1 | `agent_start` | No direct CC equivalent; session-start is implicit | INVENT — fire in subclass `__init__` after registry/skill loading, before `run()` | Receives `AgentContext`. Non-blocking. Used for: telemetry init, skill-load warning emission, system-coordination (R15 systemctl stop fires here as a built-in side effect). | **I** | R7, §4.8 |
| 2 | `pre_tool_use` (chain entry point) | CC hooks fire pre-tool, can block via output (treated as user message) | INVENT — wrap `_step_stream` tool dispatch path in subclass; fire BEFORE smolagents executes the tool | Receives `(context, tool, args)`. Returns `Allow \| Deny`. **First `Deny` short-circuits.** Hook exception = `Deny(reason="hook crashed: {exc}")` (fail-closed). | **I** | R7, §4.8, A5 |
| 3 | `post_tool_use` | CC hook output appears as user message after tool result | INVENT — fire after smolagents tool dispatch completes (success OR error) | Receives `(context, tool, args, result)`. Non-blocking; return value ignored. Used for: audit log enrichment, metrics. Exception logged, never blocks loop. | **I** | R7, §4.8 |
| 4 | `agent_end` | No direct equivalent in CC | INVENT — fire in subclass `run()`'s `finally` block, after receipt is composed | Receives `(context, receipt)`. Non-blocking. Used for: receipt post-processing, notifications, R15 systemctl restart. | **I** | R7, §4.8 |
| 5 | Built-in `identity_guard` (`pre_tool_use`) | None | Hook chain position 1 — runs before all other hooks | Denies `WriteFile`/`EditFile`/`Bash` (if Bash command targets identity roots) on any path inside `IDENTITY_ROOTS`. Fail-closed. NOT user-overridable (R26). | **I** | R14, R26, RD4, A16 |
| 6 | Built-in `p1_p9_validator` stub (`pre_tool_use`) | None | Hook chain position 2 — runs after identity_guard | Returns `Allow` unconditionally; logs `"P1-P9 validator stubbed; enforcement deferred to follow-on hardening spec"` at WARNING level once per agent run. Real implementation deferred. | **I** | R13, RD2, A18 |
| 7 | Built-in service-coordination (`agent_start` + `agent_end`) | None | `agent_start` invokes `systemctl --user stop animus.service`; `agent_end` invokes `systemctl --user start animus.service`. | Failures of either are logged warnings (not blocking). `--no-service-pause` flag (R20) skips both. Cross-platform: launchctl equivalent for macOS, otherwise log "service coordination unavailable". | **I** | R15, R20, RD5, C14, A17 |
| 8 | User hook discovery | CC hooks configured in user settings | Python files in `~/.config/animus/agent/hooks.d/*.py`; function discovery by reserved name | Loader walks dir alphabetically at `agent_start`; imports each file as a module; collects functions matching reserved names. Per-file syntax errors → exit 6 (per §4.1 error cases). | **I** | R7, §4.1 error cases, §4.8 |
| 9 | Hook ordering for `pre_tool_use` | CC ordering unspecified | Subclass orchestrates | (1) identity_guard, (2) p1_p9_validator, (3) user hooks alphabetical by source filename. First `Deny` short-circuits. User `Allow` does NOT override a built-in `Deny`. | **I** | §4.8, R26, RD4 |
| 10 | Hook ordering for non-gate events (`agent_start`, `post_tool_use`, `agent_end`) | CC unspecified | Subclass orchestrates | (1) built-ins (in source order: telemetry, service-coordination), (2) user hooks alphabetical. All run; exceptions logged + collected in receipt's `hook_errors`. | **I** | §4.8 |
| 11 | Hook exception handling | CC: not explicit in extracted prompt | Subclass wraps each hook call in try/except | `pre_tool_use` exception → `Deny(reason="hook crashed in {file}:{func}: {exc}")` (fail-closed). Non-gate hook exception → log warning + append `f"{file}:{func}: {exc}"` to receipt `hook_errors`. No hook can crash the agent loop. | **I** | §4.8, R7 |
| 12 | Hook → audit log linkage | None in CC's extracted prompt | Subclass dispatches via Forge's `BudgetManager.audit_log` | Each `pre_tool_use` decision (Allow/Deny + reason + hook source) emitted as one audit log entry. Latency cost counted toward C12 budget. | **I** | R3, R7, C8, C11, C12 |
| 13 | `tool` argument passed to hook | CC: hook sees tool name string | Subclass passes both `name` and resolved tool object reference | Hook signature: `pre_tool_use(context: AgentContext, tool: str, args: dict) -> Allow \| Deny`. `tool` is the tool name (string), `args` is the parsed argument dict from the model's JSON tool-call. Tool object NOT passed (would let user hooks introspect implementation — out of scope for v0). | **I** | §4.8 signature |
| 14 | Built-in `p1_p9_validator` swap contract | None | Same call site as v0 stub | Follow-on spec replaces the stub function body. Hook chain position, signature, and registration MUST remain identical. Stub log message tells follow-on implementer "this is where the real check lives." | **I** | R13, RD2 swap contract |

**Row count: 14** (≥ 12 required by AC2). 14 invent, 0 copy, 0 adapt — confirms hooks are net-new infrastructure with no carryover from smolagents.

---

## 2. Hook Ordering Spec

**`pre_tool_use` chain (gate semantics, first `Deny` wins):**

```
agent_start: register hooks
↓
for each tool call in agent loop:
  ┌────────────────────────────┐
  │ 1. identity_guard          │ ← built-in, fail-closed, NOT overridable (R26)
  │    Allow / Deny            │
  └─────────┬──────────────────┘
            │ Allow ↓
  ┌────────────────────────────┐
  │ 2. p1_p9_validator (stub)  │ ← built-in, v0 returns Allow + logs notice (RD2)
  │    Allow / Deny            │
  └─────────┬──────────────────┘
            │ Allow ↓
  ┌────────────────────────────┐
  │ 3. user hook #1            │ ← from ~/.config/animus/agent/hooks.d/*.py
  │    Allow / Deny            │   alphabetical by source filename
  └─────────┬──────────────────┘
            │ Allow ↓
  ┌────────────────────────────┐
  │ 3. user hook #2            │
  │    Allow / Deny            │
  └─────────┬──────────────────┘
            │ Allow ↓
       tool executes
```

**`agent_start` / `post_tool_use` / `agent_end` chains (all fire, exceptions captured):**

Built-ins first (in source-fixed order), then user hooks (alphabetical). No short-circuit — every hook runs. Exceptions logged + appended to `receipt.hook_errors` but the loop continues.

**User hook ordering tiebreaker:** alphabetical by source filename. Within a file, multiple functions for the same event all fire; their relative order matches Python's `dir()` lexicographic ordering (deterministic).

**User `Allow` after built-in `Deny`:** impossible by construction — the chain stopped at the built-in `Deny`. User hooks never see denied tool calls.

---

## 3. Built-in: `identity_guard` Hook Spec

**Module path:** `agent/hooks/builtin/identity_guard.py`

**Loaded from:** `packages/forge/forge/identity_anchor.yaml` `immutable_fields` at hook init time. Per yaml inventory: `["CORE_VALUES.md", "CONSTITUTIONAL_PRINCIPLES.md", "mysoul.md"]`.

**Resolved path roots:** the file names from `identity_anchor.yaml` resolved against known canonical locations — primary: `packages/core/animus/` and subdirs; secondary: `packages/bootstrap/.../persona/`; tertiary: any future identity-bearing files registered. Canonical resolution list lives in `agent/identity_roots.py` (per spec.md C13).

**Matching algorithm:**

```python
def identity_guard(context: AgentContext, tool: str, args: dict) -> Allow | Deny:
    """Built-in fail-closed hook denying writes to identity-file roots."""
    if tool not in ("WriteFile", "EditFile"):
        # Bash special case: deny if the command writes to an identity path
        if tool == "Bash":
            cmd = args.get("command", "")
            for root in IDENTITY_ROOTS:
                if str(root) in cmd:  # conservative: any mention triggers
                    return Deny(
                        reason=f"identity_guard: Bash command references identity root {root}; "
                               f"explicit hook override required (not permitted per R26)"
                    )
        return Allow()

    path = args.get("path", "")
    if not path:
        return Allow()  # tool itself will reject empty path

    try:
        resolved = Path(path).resolve()
    except (OSError, RuntimeError) as e:
        return Deny(reason=f"identity_guard: path resolution failed: {e}")

    for root in IDENTITY_ROOTS:
        try:
            resolved.relative_to(root)
            return Deny(
                reason=f"identity_guard: path {path} is inside identity root {root}; "
                       f"identity files are immutable (Non-Negotiable #2)"
            )
        except ValueError:
            continue  # path not under this root

    return Allow()
```

**Override prohibition (R26):** Built-in hooks always run first. A user hook returning `Allow` after identity_guard returned `Deny` is impossible — the chain short-circuited at `Deny`. There is NO mechanism for user hooks to override identity_guard. Documentation must make this explicit.

**Future relaxation path:** If a user genuinely needs to edit an identity file (rare, e.g. updating CORE_VALUES.md as part of identity evolution), they MUST route through `IdentityPatchGate` (`packages/forge/src/animus_forge/coordination/identity_patch.py`) — propose → approve → reject flow with audit trail. Not exposed as an agent tool in v0; manual user action.

**Failure mode:** if `IDENTITY_ROOTS` is empty (mis-config), identity_guard returns `Allow` for everything — log a CRITICAL warning at agent_start. Empty roots = degraded mode; explicit signal not silent.

**Test coverage (per A16):** `tests/agent/test_identity_guard.py` exercises:
- WriteFile on path inside identity root → Deny
- EditFile on path inside identity root → Deny
- Bash command containing identity root → Deny
- WriteFile on path NOT inside identity root → Allow
- Path resolution failure → Deny (defensive)
- Empty IDENTITY_ROOTS → all Allow + CRITICAL log

---

## 4. Built-in: `p1_p9_validator` Stub Spec

**Module path:** `agent/hooks/builtin/p1_p9_validator.py`

**v0 implementation:**

```python
import logging

logger = logging.getLogger(__name__)
_LOGGED_THIS_RUN: dict[str, bool] = {"yes": False}

P1_P9_STUB_NOTICE = (
    "P1-P9 validator stubbed; enforcement deferred to follow-on hardening spec. "
    "All tool calls pass through validator returning Allow."
)


def p1_p9_validator(context: AgentContext, tool: str, args: dict) -> Allow | Deny:
    """v0 stub — returns Allow unconditionally with one-shot startup log notice."""
    if not _LOGGED_THIS_RUN["yes"]:
        logger.warning(P1_P9_STUB_NOTICE)
        _LOGGED_THIS_RUN["yes"] = True
    return Allow()
```

**Swap contract for follow-on hardening spec:**
- Module path remains `agent/hooks/builtin/p1_p9_validator.py`.
- Function name remains `p1_p9_validator`.
- Signature remains `(context: AgentContext, tool: str, args: dict) -> Allow | Deny`.
- Position in `pre_tool_use` chain remains position 2 (after identity_guard, before user hooks).
- Real implementation reads P1-P9 from `docs/CONSTITUTIONAL_PRINCIPLES.md` and applies per-principle checks. Stub's `P1_P9_STUB_NOTICE` constant can be removed.

**Test coverage (per A18):** `tests/agent/test_p1_p9_stub.py` exercises:
- Every tool call sees p1_p9_validator invoked (count of calls = count of tool calls in run).
- Returns `Allow()` for all inputs.
- Log message emitted exactly once per agent run (not per call).
- Stub doesn't crash on weird input shapes (empty args, non-string tool name).

---

## 5. User Hook Discovery + Loading

**Hook directory:** `~/.config/animus/agent/hooks.d/*.py` (per §4.8; overridable via `--hook-dir <path>`).

**Reserved function names** (only these are wired):
- `agent_start(context: AgentContext) -> None`
- `pre_tool_use(context: AgentContext, tool: str, args: dict) -> Allow | Deny`
- `post_tool_use(context: AgentContext, tool: str, args: dict, result: ToolResult) -> None`
- `agent_end(context: AgentContext, receipt: Receipt) -> None`

**Discovery algorithm:**

```python
def load_user_hooks(hook_dir: Path) -> dict[str, list[Hook]]:
    """Walk hook_dir alphabetically, import each .py file, collect reserved-name functions."""
    hooks: dict[str, list[Hook]] = {
        "agent_start": [],
        "pre_tool_use": [],
        "post_tool_use": [],
        "agent_end": [],
    }
    if not hook_dir.is_dir():
        return hooks  # no hook dir = no user hooks, silent

    for path in sorted(hook_dir.glob("*.py")):
        spec = importlib.util.spec_from_file_location(f"animus_agent_hook_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except SyntaxError as e:
            # Per §4.1 error cases: exit 6 before agent loop starts
            raise InternalError(f"hook file syntax error in {path}: {e}") from e

        for name in hooks:
            fn = getattr(module, name, None)
            if fn is not None and callable(fn):
                hooks[name].append(Hook(name=name, func=fn, source=path))

    return hooks
```

**Properties:**
- Deterministic discovery (alphabetical).
- Per-file syntax error = exit 6 (consistent with §4.1).
- Functions with non-reserved names are ignored (no warning — user can define helpers).
- Module import errors (e.g. `ImportError` on missing third-party deps) = exit 6 with the import error message.

**Hook authoring constraints:**
- Hooks run in-process (same Python interpreter as the agent).
- Hooks can read `~/projects/` etc. but should not block on I/O — they sit in the C12 latency budget.
- Hooks should not perform tool execution themselves (recursive tool calls undefined behavior in v0).

---

## 6. Smolagents Intercept Strategy

**Decision (per AC5): SUBCLASS, not monkey-patch.**

**Rationale:**
- Monkey-patch couples us to internal method names that may change without notice across smolagents releases.
- Subclass makes the intercept explicit and discoverable in code review.
- Subclass survives `pip install --upgrade smolagents` better than monkey-patch (which would need re-verification each version).
- The cost is one additional class definition; monkey-patch would also need wrapper code of similar size.

**Implementation sketch:**

```python
# agent/loop.py
from smolagents import ToolCallingAgent
from smolagents.agents import ActionStep

class AnimusAgent(ToolCallingAgent):
    """Subclass of smolagents.ToolCallingAgent that wires the Animus hook chain."""

    def __init__(self, *args, hook_chain: HookChain, audit_log: AuditLog, **kwargs):
        super().__init__(*args, **kwargs)
        self._hook_chain = hook_chain
        self._audit_log = audit_log

    def _step_stream(self, memory_step: ActionStep):
        """Override smolagents.ToolCallingAgent._step_stream (agents.py:1309)
        to wire pre_tool_use / post_tool_use hooks around tool dispatch.
        """
        # Delegate generator construction to super(), but intercept yielded tool calls.
        for output in super()._step_stream(memory_step):
            if isinstance(output, ToolCall):  # smolagents internal type
                decision = self._hook_chain.pre_tool_use(
                    self._build_context(memory_step), output.name, output.args
                )
                self._audit_log.record_hook_decision(output, decision)
                if decision.is_deny:
                    # Synthesize a tool-result observation matching deny
                    output = self._build_denial_observation(output, decision)
                else:
                    # Run smolagents' actual tool execution
                    result = self._dispatch_tool(output)
                    self._hook_chain.post_tool_use(
                        self._build_context(memory_step), output.name, output.args, result
                    )
                    output = result
            yield output
```

**Risk:** smolagents' `_step_stream` is a generator with private-method naming convention (underscore prefix); the smolagents maintainers may refactor it. **Mitigation:** pin to a known-good smolagents version at v0 implementation kickoff; CI test runs against the pinned version; bump version explicitly with re-verification.

**Alternative considered (monkey-patch):** Replace `ToolCallingAgent._step_stream` with a wrapper at import time. Smaller code, but: (a) brittle to internal renames, (b) hard to discover during code review, (c) gives no inheritance benefits if we later need to subclass for other reasons (e.g. context-compaction in v0.1). Subclass wins.

**Hooks → smolagents callback bridge:** smolagents `Monitor` (`monitoring.py:81-110`) is observation-only and post-execution. We do NOT use it as the hook implementation channel — too late in the pipeline. Subclass intercept fires BEFORE smolagents dispatches the tool, satisfying the gate semantic.

---

## 7. Performance Envelope (C12)

**C12 budget: ≤ 50ms per `pre_tool_use` call.** That's the chain total, not per hook.

**Breakdown of the budget:**

| Component | Budget | Rationale |
|---|---:|---|
| Built-in `identity_guard` | ≤ 1ms | Path-prefix check against ≤ 10 roots; pure-Python `Path.resolve()` + comparison |
| Built-in `p1_p9_validator` stub | ≤ 0.1ms | Returns `Allow()` immediately; one-shot log negligible |
| Audit log emission (per call) | ≤ 5ms | JSONL append to local file; OS write buffer typically sub-ms but allow margin |
| User hooks total | ≤ 44ms | Remaining budget split across all user hooks at user's discretion |

**Measurement:** `tests/agent/test_hook_perf.py` (per spec C12) runs a parametrized benchmark with mock hooks doing varied work; asserts total chain latency ≤ 50ms at p95.

**Hook authoring guide note:** if a user hook needs > 30ms, it should be redesigned — likely doing I/O that belongs in `post_tool_use` (non-blocking) instead.

---

## 8. Failure Semantics

**`pre_tool_use` failure modes:**

| Failure | Resulting Decision | Cite |
|---|---|---|
| Hook returns `Deny(reason)` | Chain short-circuits, tool not executed, observation = denial reason | §4.8 |
| Hook returns `Allow()` | Chain continues to next hook | §4.8 |
| Hook returns `None` (forgot to return) | Treated as `Allow()` with WARNING log | INVENT — defensive |
| Hook raises any exception | Treated as `Deny(reason=f"hook crashed: {exc}")`; logged; tool not executed | §4.8 fail-closed |
| Hook returns non-Allow/Deny object | Treated as `Deny(reason=f"hook returned invalid type {type}")`; logged | INVENT — defensive |
| Hook takes > 30s (rough cap) | NOT enforced in v0 — user-hook timeout is an OQ (see §10) | OQ |

**`agent_start` / `post_tool_use` / `agent_end` failure modes:**

| Failure | Resulting Behavior | Cite |
|---|---|---|
| Hook returns anything | Return value ignored | §4.8 |
| Hook raises exception | Exception caught, logged, appended to `receipt.hook_errors` as `f"{source}:{name}: {exc}"`. Loop continues. | §4.8 |
| Hook in `agent_end` raises | Captured AFTER receipt mostly complete; receipt may be partial but is still emitted (best-effort) | INVENT |

**Receipt's `hook_errors` field** captures all non-blocking hook failures across the run. Useful for debugging user hooks.

---

## 9. Hook Authoring Guide (outline for `agent/README.md` per A15)

To be expanded in `agent/README.md` during v0 implementation. Outline:

1. **What is a hook?** Python function in `~/.config/animus/agent/hooks.d/*.py` matching a reserved name.
2. **Reserved names:** `agent_start`, `pre_tool_use`, `post_tool_use`, `agent_end`. Signatures (link to spec §4.8).
3. **`Allow` and `Deny` types:** import from `animus_forge.agent.hooks`. `Deny` takes a string reason that becomes part of the agent's tool-result observation.
4. **Built-in hooks:** what `identity_guard` and `p1_p9_validator` do; what you can/can't override (R26).
5. **Ordering:** built-ins first, then your hooks alphabetical by filename.
6. **Performance:** stay under the C12 budget. Use `post_tool_use` for slow I/O.
7. **Error handling:** crash = fail-closed deny (for gate hooks) or logged warning (for non-gate). Receipt's `hook_errors` field surfaces these.
8. **Example: deny `Bash` commands containing a specific keyword.**
9. **Example: log every tool call to a custom file via `post_tool_use`.**
10. **Example: send a notification at `agent_end` if exit status ≠ `success`.**

---

## 10. Edge Case Catalog

| # | Edge case | v0 behavior | Cite |
|---|---|---|---|
| 1 | Hook file with no reserved-name functions | Silently ignored (file imports for side effects only) | §5 |
| 2 | Hook file with both `pre_tool_use` and a helper function `_my_helper` | Reserved-name function registered; helper ignored (no warning) | §5 |
| 3 | Two hook files defining the same reserved-name function | Both fire in alphabetical filename order; for `pre_tool_use`, first `Deny` short-circuits | §2 ordering |
| 4 | Hook with `import` of unavailable third-party module | Exit 6 at agent_start with `ImportError` message naming the file | §5 |
| 5 | Hook with Python syntax error | Exit 6 at agent_start with `SyntaxError` location | §5, §4.1 |
| 6 | Hook returns `None` (forgot to return) | `pre_tool_use`: treated as `Allow` with WARNING log; non-gate: ignored | §8 table |
| 7 | Hook raises in `pre_tool_use` | Fail-closed: tool denied with reason `"hook crashed: {exc}"`; logged | §8 |
| 8 | Hook raises in `post_tool_use` | Logged + appended to `receipt.hook_errors`; loop continues | §8 |
| 9 | Hook returns a string instead of `Allow`/`Deny` | `pre_tool_use`: treated as `Deny` with type-error reason; non-gate: ignored | §8 table |
| 10 | `--hook-dir` points to non-existent directory | Silent — no user hooks loaded; agent runs with built-ins only | §5 |
| 11 | Hook crashes during `agent_end` (after receipt composition) | Caught, logged; receipt emitted anyway (best-effort) | §8 |
| 12 | identity_guard sees `Bash("echo CORE_VALUES.md")` (mention, not write) | Denies (conservative). Workaround: user can construct command without literal string mention | §3 algorithm note |

**12 edge cases** (≥ 8 required by AC8).

---

## 11. Naming Collision — `AreteHooks` Already Exists

`packages/forge/src/animus_forge/workflow/arete_hooks.py` defines an `AreteHooks` class with `on_step_failure` / `on_workflow_complete` — workflow-level observation hooks, fundamentally different scope from the agent's tool-level gate hooks.

**Recommendation for v0:** Use module path `agent/hooks/` and class name `HookChain` (or similar — NOT `AreteHooks`) to avoid namespace collision. Document the distinction in `agent/README.md`. Defer unification to a follow-on spec — the two systems serve different layers (workflow vs agent) and may have different lifecycle semantics that resist merging.

**File-naming convention for v0:**

```
packages/forge/src/animus_forge/agent/hooks/
├── __init__.py
├── chain.py            # HookChain orchestrator (this is the public surface)
├── types.py            # Allow, Deny, Hook dataclass
├── loader.py           # User hook discovery + loading (§5 algorithm)
└── builtin/
    ├── __init__.py
    ├── identity_guard.py     # R14, §3
    ├── p1_p9_validator.py    # R13, RD2, §4
    └── service_coordination.py  # R15, RD5
```

---

## 12. Open Questions / Gaps

- **OQ1 — User hook timeout enforcement:** v0 doesn't enforce a hook execution timeout (only the soft C12 budget). A runaway user hook (infinite loop, blocking I/O) hangs the agent. **Decision needed:** add a `signal.alarm`-based timeout or trust the user not to write bad hooks? Probably trust in v0; revisit if first eval surface a hung-hook incident.

- **OQ2 — Hook → tool-call recursion:** What if a user hook in `pre_tool_use` calls an Animus tool (e.g. ReadFile)? In-process Python, no guard against it. Likely undefined behavior or infinite recursion. **Decision needed:** explicit warning in authoring guide; defer mechanism to prevent (would need a thread-local flag).

- **OQ3 — `IDENTITY_ROOTS` resolution canon:** `identity_anchor.yaml` lists file names (`CORE_VALUES.md`, etc.) without paths. `agent/identity_roots.py` resolves to actual paths via heuristic (search `packages/core/animus/`, etc.). If a user has identity-bearing files outside known locations, identity_guard misses them. **Decision needed:** make `identity_roots.py` config-driven from a yaml in user config, with the heuristic as fallback default. Adds surface but maintainable.

- **OQ4 — Bash identity-guard false positives:** §3 algorithm denies `Bash` commands that *mention* identity-root file names, not necessarily *write to* them. `Bash("grep something CORE_VALUES.md")` would be denied — too conservative. **Decision needed:** refine to only deny commands that match write-pattern regex (`>`, `>>`, `tee`, `sed -i`, `cp X CORE_VALUES.md`, etc.), accepting more false negatives in exchange for fewer false positives. Real risk: agent fights with the guard during exploratory work.

- **OQ5 — `tool` arg vs full tool object in hook signature:** v0 passes only the tool name string. If a user hook needs to introspect the tool (e.g. check its declared schema), they can't. **Decision needed:** v0 keeps string-only (smaller surface); v0.1 adds optional tool object passing if signal demands.

- **OQ6 — `AgentContext` shape:** spec.md §4.8 names `AgentContext` as a passed type but doesn't enumerate fields. Need to spec at implementation: turn count, budget remaining, wall elapsed, current model, current memory snapshot? Defer to v0 implementation kickoff with this open question listed.

---

## 13. Coverage Check (per AC7)

| spec.md element | Doc reference |
|---|---|
| R7 (hook gates 4-event) | §1 matrix rows 1-4 |
| R13 (P1-P9 stub wired) | §1 row 6, §4 (full spec) |
| R14 (identity-file deny hook) | §1 row 5, §3 (full spec) |
| R20 (`--no-service-pause`) | §1 row 7 (service coordination) |
| R26 (built-ins not user-overridable) | §1 row 9, §3 override-prohibition note |
| §4.8 (hook lifecycle interface contract) | §1 matrix (all rows), §2 ordering, §8 failure semantics |
| RD2 (P1-P9 stub) | §1 row 6, §4 swap contract |
| RD4 (built-in-first fail-closed) | §1 row 5, §2 ordering diagram |
| C12 (≤ 50ms latency) | §7 |
| A5 (user hook denial test) | §1 row 2 |
| A16 (identity_guard test) | §3 test coverage subsection |
| A17 (service coordination test) | §1 row 7 |
| A18 (P1-P9 stub test) | §4 test coverage subsection |

All required spec elements referenced.

---

## 14. Acceptance Criteria Self-Check

- ✅ **AC1** — File exists at named path
- ✅ **AC2** — Lifecycle event matrix has 14 rows (≥ 12 required)
- ✅ **AC3** — `identity_guard` spec in §3 includes `IDENTITY_ROOTS` source (`identity_anchor.yaml`), full matching algorithm, override prohibition (R26 cite + override-impossibility argument), failure-mode behavior (empty roots = CRITICAL log)
- ✅ **AC4** — `p1_p9_validator` stub spec in §4 includes call site location, return shape, log notice constant `P1_P9_STUB_NOTICE`, swap contract (5 enumerated preservation requirements)
- ✅ **AC5** — Smolagents intercept strategy committed: **subclass `ToolCallingAgent`**, rationale + code sketch in §6
- ✅ **AC6** — C12 latency budget broken down in §7: identity_guard ≤ 1ms, p1_p9 stub ≤ 0.1ms, audit log ≤ 5ms, user hooks ≤ 44ms; total ≤ 50ms
- ✅ **AC7** — Coverage table in §13 — every required spec element referenced
- ✅ **AC8** — Edge case catalog §10 has 12 entries (≥ 8 required)
- ✅ **AC9** — Open Questions §12 has 6 entries (≥ 3 required)
- ✅ **AC10** — Code-level Forge citations: `coordination/identity_patch.py` (IdentityPatchGate exists), `forge/identity_anchor.yaml` (IDENTITY_ROOTS source), `workflow/arete_hooks.py:204` lines (existing AreteHooks for naming-collision flag)

All 10 ACs met.

---

## 15. Recommended Next Steps

1. **User review** of this document. Likely revision area: OQ4 (Bash identity-guard false positives — pick the policy now or leave to implementation).
2. **No further RE tasks** — this is the third in the trilogy. RE Tasks #1, #2, #3 collectively cover the v0 implementation surface.
3. **Pre-implementation spike (per cc-loop-port-plan OQ8):** verify `provider_wrapper.py` Ollama tool-call schema support. ~1 hour. Highest "is the integration even feasible without a new adapter" weight.
4. **Spec.md amendment candidates from this trilogy:**
   - From RE #2: `ListSkills` tool (changes R5 starter-tool count 5→6).
   - From RE #3: `HookChain` class name + module path (`agent/hooks/`) — could fold into spec §4.8 as the concrete naming, or leave to implementation discretion.
5. **Defer until v0 implementation:** all writeable code, including unit tests covering A5, A16, A17, A18 and edge cases in §10.

---

End of port plan.
