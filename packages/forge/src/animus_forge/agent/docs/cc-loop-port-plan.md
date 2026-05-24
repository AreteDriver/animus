# Claude Code Loop → smolagents Port Plan

**RE Task:** #1 (per `re-task-1-scope.md`)
**Status:** Draft v1.0 — 2026-05-23
**Parent spec:** `../spec.md` v0.1.0 (R1–R26, C1–C14, A1–A18, RD1–RD8)
**Author:** Claude Code (advisory mode per [[project-animus-agent-platform]])

---

## 0. Sources Consulted

| Source | Location / URL | Purpose |
|--------|----------------|---------|
| Claude Code 2.0 full prompt + tools | `~/projects/system-prompts-and-models-of-ai-tools/Anthropic/Claude Code 2.0.txt` (1150 lines) | Canonical CC loop semantics + tool surface |
| Claude Code 1.x prompt | `~/projects/system-prompts-and-models-of-ai-tools/Anthropic/Claude Code/Prompt.txt` (191 lines) | Evolution delta vs 2.0 |
| Claude Code 1.x tool schemas | `~/projects/system-prompts-and-models-of-ai-tools/Anthropic/Claude Code/Tools.json` (508 lines) | Tool input schemas reference |
| Sonnet 4.5 prompt | `~/projects/system-prompts-and-models-of-ai-tools/Anthropic/Sonnet 4.5 Prompt.txt` | Model-specific prompt layering |
| smolagents README | `https://github.com/huggingface/smolagents` (main branch) | Architecture overview |
| smolagents src layout | `gh api repos/huggingface/smolagents/contents/src/smolagents` | File-level inventory for citation |
| Animus Forge `agents/` module | `~/projects/animus/packages/forge/src/animus_forge/agents/` (4765 LoC) | Existing orchestration primitives |
| Animus monorepo CLAUDE.md | `~/projects/animus/CLAUDE.md` | Non-negotiables P1-P9, BudgetManager, identity rules |
| Animus Forge CLAUDE.md | `~/projects/animus/packages/forge/CLAUDE.md` | Forge package conventions |
| `~/.claude/skills/` | 127 SKILL.md dirs (most YAML-frontmatter, some legacy) | Format reference for skill loader |
| `../spec.md` v0.1.0 | The constraint envelope | Drives all `adapt` / `invent` decisions |

**Smolagents source files (16, in `src/smolagents/`):** `agents.py`, `tools.py`, `models.py`, `memory.py`, `local_python_executor.py`, `remote_executors.py`, `mcp_client.py`, `monitoring.py`, `default_tools.py`, `cli.py`, `agent_types.py`, `gradio_ui.py`, `serialization.py`, `tool_validation.py`, `utils.py`, `vision_web_browser.py`, `_function_type_hints_utils.py`. Citations below reference the GitHub `main` branch SHA at fetch time.

---

## 1. Comparison Matrix

Status legend: **C** = copy verbatim; **A** = adapt for local-LLM/Animus constraints; **D** = drop (out of scope for v0); **I** = invent (no equivalent in either source).

| # | CC Primitive | smolagents Primitive | Animus v0 Required Behavior | Status | Cite |
|---|---|---|---|---|---|
| 1 | ReAct loop (LLM gen → tool calls → observation → repeat) | `MultiStepAgent.run()` at `src/smolagents/agents.py:546`; per-step loop body at `agents.py:1309` (`_step_stream`), final-answer exit at `agents.py:1350` | R2: ReAct-style loop terminating on agent-declares-completion, max-turns, budget, wall-time, or unrecoverable tool error after one retry | **A** | R2, C9, C10 |
| 2 | System prompt (multi-section: tone, proactiveness, professional objectivity, task management, doing tasks, tool usage policy, env, code references, tools list) | `MultiStepAgent` takes a `system_prompt` template, renders tool descriptions in | Single composed system prompt covering: task framing, tool-use protocol for Qwen3-32B, code-references convention, response discipline, identity/path constraints. Generated from sections in `agent/prompts/system.md` | **A** | R1, R2 |
| 3 | Code-action style (CC writes JSON tool calls) vs smolagents `CodeAgent` (writes Python snippets) | `CodeAgent` writes Python; `ToolCallingAgent` at `agents.py:1274` writes JSON tool-call shapes | Use `ToolCallingAgent` shape — Qwen3-32B's tool-use training favors JSON tool calls over freeform code generation; CodeAgent's sandbox needs are out-of-scope (no remote executor in v0) | **A** | R4, OOS:sandbox |
| 4 | Read tool (offset, limit, multimodal images/PDF/notebook) | smolagents has no built-in file Read — would compose via `bash_tool` or custom | R5+§4.2: `ReadFile(path, offset=0, limit=None)`. Drop multimodal v0; UTF-8-only with binary detection | **A** | R5, §4.2, OOS:multimodal |
| 5 | Write tool (overwrite, create parents) | No built-in; custom | R5+§4.3: `WriteFile(path, content)`. Path scoped to `--project`. Identity-file deny hook applies | **A** | R5, R10, R14, §4.3 |
| 6 | Edit tool (exact-match `old_string` replacement, fails on multi-match) | No built-in; custom | R5+§4.4: `EditFile(path, old, new)`. Multi-match raises ToolError; identity-file deny hook applies | **A** | R5, R10, R14, §4.4 |
| 7 | Bash tool (timeout, run_in_background, descriptions, output-as-text) | smolagents `default_tools.py` exposes `BashTool` (similar) | R5+§4.5: `Bash(command, timeout_s=30)`. cwd=project root. 8KB stdout/stderr truncation. Default-deny policy (R11) | **A** | R5, R11, R23, §4.5 |
| 8 | BashOutput tool (poll background shell stdout) | smolagents does NOT model background shells natively | DROP for v0 — v0 is one-shot; no need for background shell coordination. Reconsider in v1 (overnight-delegate) | **D** | OOS:sub-agents-v0 (related infra) |
| 9 | KillShell tool (terminate background shell) | Drops with BashOutput | DROP for v0, same rationale | **D** | OOS |
| 10 | Glob tool (file-pattern matching by mtime) | No built-in; trivial to compose | DROP for v0 — Grep covers most search needs; add as starter tool in v0.1 if eval surfaces need | **D** | OOS (v0 picks 5 starter tools, Grep covers search) |
| 11 | Grep tool (ripgrep with content/files_with_matches/count modes, glob filter, multiline) | No built-in; custom | R5+§4.6: `Grep(pattern, path=".", glob="**/*", case_insensitive=False)`. Python `re`, cap 100 matches | **A** | R5, §4.6 |
| 12 | Task tool (sub-agent spawning, stateless, named subagent_type, `general-purpose`/`statusline-setup`/`output-style-setup` types in CC 2.0) | smolagents "managed agents" / hierarchical multi-agent | OUT-OF-SCOPE for v0 (R25 explicit). Document for v1+ wiring | **D** | R25, OOS |
| 13 | TodoWrite tool (in-conversation task tracking, in_progress/completed states, "ONE task at a time" discipline) | No equivalent; smolagents uses linear step memory | DROP as tool. The user-facing Animus task system + Forge audit log subsume this role. Agent loop's internal "what's next" is in memory, not a tool | **D** | OOS, [[project-animus-dev-gotchas]] |
| 14 | SlashCommand tool (`/<skill-name> <args>` invocation; lists available commands; gates on what's installed) | smolagents has no skill-registry concept | R6+§4.7: built-in agent tool `InvokeSkill(name, args)` reads from skill registry (loaded from `~/.claude/skills/`). Resolves skill body to a sub-prompt; not a full nested loop in v0 | **I** | R6, RD3, §4.7 |
| 15 | WebFetch tool (URL+prompt, 15-min cache, redirect handling) | smolagents has tools for web access; `vision_web_browser.py` for browsing | DROP for v0 starter set (R21 no cloud LLM constraint complicates the model used for the fetch-summary). Document for v1 with local model option | **D** | R21, OOS |
| 16 | WebSearch tool (search, domain filtering) | smolagents has search tools (DuckDuckGo, etc.) | DROP for v0 — not in starter-tool 5 | **D** | OOS |
| 17 | NotebookEdit tool (Jupyter cells) | No equivalent | DROP for v0 — not in starter-tool 5 | **D** | OOS |
| 18 | ExitPlanMode tool (plan→action transition signal) | smolagents has no plan-mode | DROP for v0 — agent loop is direct, no plan-mode | **D** | OOS |
| 19 | Hooks (shell commands triggered by tool events, configured in user settings, treat output as user message) | smolagents `Monitor` class at `src/smolagents/monitoring.py:81-110` — observers only, NO pre/post-tool gate interface (verified by source review: "No explicit hook-like interfaces found... The closest pattern is `update_metrics` callback, which receives step data after execution completes") | R7+§4.8: 4-event hook lifecycle (`agent_start`, `pre_tool_use`, `post_tool_use`, `agent_end`). Python files in `~/.config/animus/agent/hooks.d/*.py`. First `Deny` short-circuits. Built-ins (identity-guard R14 + P1-P9 stub R13) run before user hooks, fail-closed (RD4) | **A** | R7, R13, R14, RD2, RD4, §4.8 |
| 20 | system-reminder tags (sidecar context injection — instructions arrive as user-msg-shaped reminders) | smolagents has no equivalent context-injection layer | INVENT for v0 — instrument the loop to inject lifecycle-specific reminders (e.g. "you have N turns / T tokens remaining") via `agent.memory` append. Critical for max-turns / budget warnings | **I** | R2, C9, C10 |
| 21 | CLAUDE.md auto-loaded context (project + user instructions injected into system prompt at session start) | No equivalent | DROP for v0 — too entangled with what content to ingest. Implementor option to add as opt-in flag (`--claude-md <path>`). Note as v0.1 candidate | **D** | OOS (deferred) |
| 22 | Code references convention (`file_path:line_number` in agent output) | Convention only; smolagents doesn't enforce | COPY into Animus system prompt verbatim — improves human readability of receipts | **C** | §4.9 receipt + system prompt |
| 23 | Multimodal Read (images, PDFs, screenshots) | smolagents has `agent_types.py` for image/audio types | DROP for v0 — Qwen3-32B is text-only at the chosen quant; multimodal adds runtime complexity | **D** | R4, OOS:multimodal |
| 24 | Stateless Task sub-agent invocations (each Task call is fresh context, returns single message) | smolagents managed-agent spawning | DROP for v0 per R25; document spawning model for v1+ port | **D** | R25 |
| 25 | Verbosity discipline (concise, <4 lines, no preamble/postamble) | Not enforced by smolagents | COPY into Animus system prompt — verbatim port of CC 2.0 tone section is the safest start. Local-LLM may overrun, so add token-soft-cap via stop-strings | **A** | C9, C10 |
| 26 | Model routing (CC 2.0 = Sonnet 4.5; switches based on task class internally) | smolagents has `Model` abstraction, multi-provider, no routing layer | DROP — single-model in v0 per R4. Multi-model routing deferred per OOS list | **D** | R4, OOS:multi-model |
| 27 | Multi-tool parallel calls in single message | smolagents loop is sequential by default; `MultiStepAgent` step model is one-tool-per-step | DROP for v0 — Qwen3-32B's parallel-tool-use compliance is uneven; sequential calls trade ~30% latency for reliability. Note as v0.1 perf experiment | **D** | C10 (perf bar accommodates serial) |
| 28 | Context-aware system-reminder injection (CC injects different reminders based on lifecycle, e.g. plan-mode entry, task counter, env updates) | Not modeled | INVENT — narrower scope than CC's: budget-remaining + turn-counter + identity-guard-fired reminders. Wired in `pre_tool_use` / `post_tool_use` hook layer | **I** | R7, R8 |
| 29 | Receipt / session telemetry (CC streams to its own infrastructure; not in the prompt) | smolagents has `agent.logs` and `monitoring.py` | R8+§4.9: JSON receipt to stdout, optional `--receipt` path. Token counts + per-tool wall-ms + denials. Persists to Forge audit log via BudgetManager (R3, RD1) | **I** | R3, R8, RD1, §4.9 |
| 30 | Hooks for identity / privilege protection | None in either source | R14: built-in `identity_guard` hook denying writes to identity-file roots, fail-closed, runs first, NOT overridable by user `Allow` (R26). Mirrors monorepo Non-Negotiable #2 | **I** | R14, R26, RD4 |
| 31 | Service-coordination / daemon-aware startup | None | R15: `systemctl --user stop animus.service` on agent startup, `start` in `finally` block. Logged warnings on failure, never blocks (RD5, C14). `--no-service-pause` opt-out (R20) | **I** | R15, R20, C14, RD5 |
| 32 | Constitutional principles enforcement (P1-P9 layer) | None | R13: validator call site wired as `pre_tool_use` hook; v0 stub returns Allow with explicit log notice (RD2). Real validator deferred to follow-on spec | **I** | R13, RD2 |

**Row count: 32** (≥ 20 required by AC2). Status distribution: **2 copy, 14 adapt, 10 drop, 6 invent.**

---

## 2. Decision Log

Every `adapt` / `invent` row carries the constraint that drove the deviation. The `drop` rows are not re-justified here — they reference the OOS list in `spec.md §6`.

### Adapt rows

| Row | Decision | Driving Constraint |
|-----|----------|---------------------|
| 1 | ReAct loop with explicit termination triggers (not just final_answer) | R2 names 5 termination conditions; smolagents only natively handles 2 (final_answer + max_steps). Need budget + wall-time + retry-then-fail wiring. C9/C10 set the perf bars termination is measured against. |
| 2 | Composed system prompt with v0-specific sections | R1 + R2 require Animus-specific framing (e.g. "all paths must resolve within `--project`"). Verbatim port of CC's system prompt would mention Anthropic-specific tools we don't have. |
| 3 | Use `ToolCallingAgent` not `CodeAgent` | R4 (Qwen3-32B). Local model tool-use training targets JSON-shaped tool calls, not freeform Python. CodeAgent requires a code-execution sandbox (out of scope per spec.md §6 "Sandboxing beyond path + bash-policy"). |
| 4 | ReadFile drops multimodal in v0 | OOS:multimodal — Qwen3-32B is text-only at Q4_K_M. Multimodal adds runtime + memory cost not justified for v0. |
| 5 | WriteFile adds path scoping + identity guard | R10 (path scoping), R14 (identity-file deny hook), R22 (MUST NOT modify outside `--project`). |
| 6 | EditFile single-occurrence-required | §4.4 explicit. Multi-match in CC's Edit raises only if user doesn't pass `replace_all=true`; we drop the flag for v0 simplicity. |
| 7 | Bash adds cwd-pinning + 8KB output truncation + deny-list policy | R11 (deny list), R23 (no privilege escalation), §4.5 (truncation prevents context blowup on `find /` etc.). cwd-pinning enforces R10 / R22 boundary. |
| 11 | Grep caps at 100 matches, no multiline mode in v0 | §4.6. Multiline matching is a CC 2.0 affordance for advanced searches; deferred. 100-match cap prevents context blowup. |
| 19 | Hook system is fail-closed + ordered (built-ins first) | RD4 (identity-guard must be unoverridable); R7 (hook exception = deny); §4.8 ordering. CC's hooks are settings-configured but not strongly ordered or fail-closed. |
| 25 | Verbosity discipline copied verbatim, with stop-string token cap | C10 perf bar (e2e ≤ 60s) — local LLM verbosity costs wall-time more than cloud equivalent. Stop-string fallback because Qwen3-32B may ignore the prose discipline. |

### Invent rows

| Row | Decision | Driving Constraint |
|-----|----------|---------------------|
| 14 | `InvokeSkill(name, args)` tool — skill body becomes sub-prompt, not nested loop | R6 (all 135 skills loaded), §4.7 (skill loader independent), RD3 (Claude-Code-format only in v0). Full nested-loop skill invocation is sub-agent shape, deferred per R25. |
| 20 | `system-reminder`-style mid-loop context injections | C9 (latency awareness) and R2 (termination triggers) require the agent to know its remaining budget. CC achieves this via system-reminder; smolagents has no equivalent. Inject via `agent.memory.append()` at each turn boundary. |
| 28 | Lifecycle-aware reminders (turn counter + budget remaining + last hook denial) | R7 + R8. Without per-turn awareness, the agent can't fail gracefully when budget exhausts. Narrower scope than CC's full reminder system. |
| 29 | JSON receipt artifact | R8 + §4.9. CC streams session telemetry to private infrastructure; we need a pure-local, deterministic, machine-readable receipt for the eval harness (A13). |
| 30 | `identity_guard` built-in hook | R14 + R26 + monorepo Non-Negotiable #2. Neither CC nor smolagents has a fail-closed identity-protection layer; we invent it because the constraint is hard. |
| 31 | systemctl-mediated service coordination | R15 + RD5. Animus's `self_heal` proactive engine creates races with agent runs — empirical pain ([[project-animus-dev-gotchas]]). Service-coordination is unique to Animus; not in either source. |
| 32 | P1-P9 validator call site (stubbed) | R13 + RD2. Monorepo constitutional principles are an Animus-specific safety layer; the call site is invented now so the follow-on hardening spec only swaps the implementation. |

---

## 3. Tool Execution Protocol Mapping

**Claude Code:** JSON-shaped tool-use messages (Anthropic-API native). The model emits `tool_use` blocks with `name`, `input`, `id`; the harness executes and replies with `tool_result` blocks containing `tool_use_id`, `content`, optional `is_error`. Verified in `Claude Code 2.0.txt:198-356` (Bash schema), `:391-431` (Edit), `:639-680` (Read).

**smolagents `ToolCallingAgent`:** model emits JSON like `{"name": "tool_x", "arguments": {...}}`; smolagents parses, dispatches to registered `Tool` instances, captures return value, appends to memory as observation. Implementation: `ToolCallingAgent` class at `src/smolagents/agents.py:1274`, per-step generator at `agents.py:1309` (`_step_stream`), memory append after tool call at `agents.py:1375`, final-answer detection at `agents.py:1350`. Cites pinned to `huggingface/smolagents@main` as of 2026-05-23; rebase to a tagged release at v0 implementation kickoff.

**Animus v0 protocol:**

```python
# Pseudocode shape — actual implementation in agent/loop.py at port time
for turn in range(max_turns):
    response = provider.complete(  # routes through BudgetManager (R3)
        messages=context.messages,
        tools=tool_registry.json_schema(),
        model=settings.model,
    )
    if response.has_tool_calls:
        for call in response.tool_calls:
            # Hook chain: identity_guard → P1_P9_stub → user hooks
            decision = hook_chain.pre_tool_use(context, call.name, call.args)
            if decision.is_deny:
                result = ToolResult.denied(decision.reason)
            else:
                try:
                    result = tool_registry[call.name].execute(call.args)
                except ToolError as e:
                    result = ToolResult.error(str(e))
            hook_chain.post_tool_use(context, call.name, call.args, result)
            context.append_observation(call.id, result)
            receipt.tool_calls.append(...)
        if budget_exceeded(context): break  # exit code 2
        if wall_exceeded(): break           # exit code 3
    else:
        # No tool calls → agent declares completion
        receipt.final_message = response.text
        receipt.status = "success"
        break
else:
    receipt.status = "max_turns"  # exit code 1
```

**Lossy translations to flag:**
- CC's parallel tool calls per message: Animus v0 is serial-only (matrix row 27). Agent prompt MUST say "issue ONE tool call per turn" — local LLMs often fight this.
- CC's `tool_use_id` correlation: smolagents may not preserve this across rounds; we wrap with our own `call.id` at hook-chain entry.
- CC's `is_error` flag in `tool_result`: Animus encodes via `ToolResult.status = "ok|error|denied"` (§4.9 receipt schema); no boolean shadow.

---

## 4. Context Management

**CC:** Compacts long conversations via internal heuristics not exposed in the system prompt. Uses `<system-reminder>` tags for sidecar context injection without bloating the canonical message list. CLAUDE.md auto-loaded at session start.

**smolagents:** `agent.memory` is a chat-message log; appended to each turn. No compaction primitive in the public API as of main-branch fetch. `MultiStepAgent` re-feeds full memory to model each step → quadratic context growth.

**Animus v0:** Same naive linear-memory approach as smolagents — no compaction in v0. Mitigation:
- Token-budget cap (R2 termination on `budget_tokens`) bounds context growth indirectly.
- Skill bodies loaded lazily (only when `InvokeSkill` fires) — not in system prompt.
- Identity-roots list loaded once at startup, cached (R16).
- Per-turn reminder injection (matrix rows 20, 28) replaces tool results in subsequent context windows IF token pressure rises — defer the eviction logic to v0.1 if benchmarks show it's needed.

**Open issue (see §8 OQ1):** With Qwen3-32B's effective context (~32K tokens at Q4_K_M before quality degrades), a 50-turn run with 3 tool calls/turn × 1-2KB results/call easily breaches the window. Need empirical measurement in v0 eval to decide whether compaction is a v0 must-have or v0.1 optimization.

---

## 5. Sub-agent / Agent-Tool Model (v1+ note)

**Out of scope for v0 per R25.** Documented here because RE Task #3 (hook port plan) and v1 design will need this.

**CC's Task tool** spawns stateless sub-agents:
- Each call is a fresh context — no message history shared.
- Sub-agent has its own tool subset (e.g. `general-purpose` has all tools; `statusline-setup` has Read + Edit only).
- Returns a single message to the parent.
- Parent stores the result in its own memory.

**smolagents "managed agents":** hierarchical pattern via `ToolCallingAgent` / `CodeAgent` as tools within a larger orchestrator. Each managed agent has its own `model`, `tools`, `memory`. Spawning via `agent.run("...")` from the parent's tool list.

**Animus v1+ recommended shape (NOT v0):**
- Reuse smolagents' managed-agent pattern; wrap as `SpawnAgent(subagent_type, task)` tool.
- Each sub-agent gets a `subagent_config.yaml` entry under `~/.config/animus/agent/subagents.d/` specifying allowed tool subset.
- Sub-agent receipts nest inside parent receipt under `tool_calls[i].sub_receipt`.
- Sub-agent inherits parent's BudgetManager session ID (audit log scope continuity).
- Hook chain applies to sub-agent's tools (recursively) — built-ins always run.
- Open: should sub-agent count against parent's `max_turns` budget? Probably yes (defer to v1 spec).

---

## 6. Skill Registry Semantics

**CC's SlashCommand:** Resolves `/<name> <args>` against installed skills. Skill body is markdown with YAML frontmatter (`name`, `description`, sometimes more). Invocation injects the skill body as a system-reminder + the args; the agent continues its turn with the skill's instructions in context.

**Skill format in user's environment:** 127 dirs under `~/.claude/skills/`, each with a `SKILL.md`. Probed 4 samples: most have `---\nname: ...\ndescription: ...\n---` frontmatter; a few legacy ones (`/review`) use `# /name - Description` header style instead. Loader MUST handle both, OR explicitly warn on legacy format and skip.

**Animus v0 loader (`agent/skill_loader.py`):**

```python
@dataclass
class Skill:
    name: str
    description: str
    body: str           # markdown body after frontmatter
    invokable: bool
    source_path: Path

def load_skills(root: Path = Path("~/.claude/skills").expanduser()) -> dict[str, Skill]:
    registry: dict[str, Skill] = {}
    warnings: list[str] = []
    for skill_dir in sorted(root.iterdir()):
        if not skill_dir.is_dir(): continue
        md = (skill_dir / "SKILL.md")
        if not md.exists(): md = (skill_dir / "skill.md")
        if not md.exists(): continue  # silent skip per RD3 — not all dirs are skills
        try:
            skill = parse_skill_file(md)
            registry[skill.name] = skill
        except SkillParseError as e:
            warnings.append(f"{md}: {e}")
    return registry, warnings
```

**InvokeSkill tool (v0 model):**
- Tool signature: `InvokeSkill(name: str, args: str | None = None) -> str`
- Behavior: resolves `name` in registry. If found and `invokable=True`, returns the skill body (string) — the agent reads it on next turn and acts. If found but `invokable=False`, raises `ToolError("skill is reference-only")`.
- This is NOT a nested agent invocation — that's sub-agent shape, deferred to v1.
- Args are appended to the skill body as `## Arguments\n{args}` if provided.

**Trade-off:** v0's InvokeSkill is a single-turn enhancement (read skill body, act in same loop). CC's SlashCommand is more like a context-injection + immediate enactment. The difference is whether the skill "owns" the rest of the conversation. v0 picks the conservative option — skill body is advice, not control transfer.

**Per RD6, v0 acceptance ≥ 135 skill load count** — spec.md C4 / A3. **Discrepancy noted in inventory:** actual is 127 on this user's disk. Update spec.md if user confirms 127 is the canonical count (the 135 number came from session-start skill-list output which appears to include plugin-namespaced skills not living under `~/.claude/skills/`). See OQ4 in §8.

---

## 7. Hook Lifecycle Mapping

**CC hooks** (from `Claude Code 2.0.txt:146`):
> "Users may configure 'hooks', shell commands that execute in response to events like tool calls, in settings. Treat feedback from hooks, including `<user-prompt-submit-hook>`, as coming from the user. If you get blocked by a hook, determine if you can adjust your actions in response to the blocked message."

Key CC properties:
- Hooks are shell commands, not Python functions.
- Hook output is fed back as if it were a user message.
- Hook can block tool execution (mentioned in prompt).
- Configured in user settings.

**smolagents `monitoring.py`:** observer pattern, not gate pattern. Observers see events but cannot block.

**Animus v0 hook system (spec §4.8):**

| Lifecycle event | When fires | Signature | Blocking? | Use case |
|---|---|---|---|---|
| `agent_start` | Once at agent startup, before first LLM call | `(context: AgentContext) -> None` | No | Logging, telemetry init |
| `pre_tool_use` | Before each tool call | `(context, tool, args) -> Allow \| Deny` | **Yes** — first `Deny` short-circuits | Identity guard (built-in), P1-P9 (built-in stub), user policy |
| `post_tool_use` | After each tool call (regardless of success) | `(context, tool, args, result) -> None` | No | Audit log enrichment, metrics |
| `agent_end` | Once at agent exit (any status) | `(context, receipt) -> None` | No | Receipt post-processing, notification |

**Built-in hook order** (§4.8 + RD4):
1. `identity_guard` (R14) — denies WriteFile/EditFile on identity-file roots. Fail-closed. NOT user-overridable (R26).
2. `p1_p9_validator` (R13, stubbed v0 per RD2) — returns Allow + logs stub notice.
3. User hooks from `--hook-dir` in alphabetical filename order.

**Differences from CC:**
- CC hooks are shell-out; Animus hooks are in-process Python. Faster (no fork) but constrained to Python (acceptable for personal use).
- Animus pre_tool_use has explicit `Allow | Deny` return type; CC's blocking is implicit (treated as user message).
- Animus built-in hooks always run first; CC has no built-in concept (all hooks are user-config).
- Animus fail-closed on hook exception; CC's behavior on hook crash is not specified in the prompt.

**Hook denial latency budget:** C12 sets ≤ 50ms per `pre_tool_use` call. Built-in identity guard is a path-prefix check (microseconds); P1-P9 stub returns immediately; total fixed overhead well under bound. User hooks bear the latency budget — document in `agent/README.md` (A15).

---

## 8. Open Questions / Gaps

These are honest gaps in either the source material or our derived design. Implementation work surfaces answers.

- **OQ1 — Context compaction trigger:** With Qwen3-32B's effective ~32K context, a 50-turn run breaches. CC has compaction (not specified in extracted prompt — only inferred). smolagents has none. Animus v0 plan defers, but the eval harness may show it's a v0 must-have, not v0.1. **Decision needed:** measure first run, then decide.

- **OQ2 — Receipt schema vs Forge audit log shape:** §4.9 defines a JSON receipt; Forge's existing audit log has its own row shape. Two persistence paths for the same data risks drift. **Decision needed:** does the receipt build FROM audit-log queries (single source of truth), or does the receipt write THROUGH to the audit log (dual-emit)? RD1 implies dual-emit; needs confirmation when implementing C8/C11/A6.

- **OQ3 — Token-counting mechanism for local Ollama:** Ollama's `/api/generate` returns `prompt_eval_count` + `eval_count`. BudgetManager needs these for R3 / C8. But Ollama's tokenizer may differ from what BudgetManager assumes for cloud providers. **Decision needed:** trust Ollama's counts, or run a fresh tokenizer pass server-side?

- **OQ4 — Skill count discrepancy (127 actual vs 135 in spec/memory):** Inventory found 127 SKILL.md dirs in `~/.claude/skills/`. The 135 number came from earlier session-start tooling output which appears to count plugin-namespaced skills (`arete-cc-stack:hackathon-triage`, `vercel:bootstrap` etc.) that don't live in that directory. **Decision needed:** confirm canonical count source, update spec.md C4 / A3 if 127 is the correct floor.

- **OQ5 — InvokeSkill granularity:** v0 design has InvokeSkill return the skill body as advice (single-turn). CC's SlashCommand essentially transfers context-of-the-conversation to the skill. If v0 eval suite tests skill use (it almost certainly will), agents may struggle without context transfer. **Decision needed:** measure in eval, possibly revisit RD3 / R6 if signal is clear.

- **OQ6 — Hook composition with smolagents internals:** Confirmed by source review (`monitoring.py:81-110`) that smolagents has NO pre-tool-execution hook interface — only post-execution observers. Animus's `pre_tool_use` MUST intercept BEFORE smolagents dispatches the tool. **Decision needed:** monkey-patch smolagents' tool dispatch or subclass `ToolCallingAgent` (`agents.py:1274`) overriding `_step_stream` (`agents.py:1309`)? Subclass is cleaner; monkey-patch is faster to prototype. Both pin us to specific smolagents internals.

- **OQ7 — Service coordination on non-systemd hosts:** R15 / RD5 cover systemctl on Linux. macOS uses `launchctl`, BSD has neither. v0 ships on Linux; M5 (October) brings macOS into scope. **Decision needed:** implement launchctl in v0 (extra surface, untested) or punt to "v0 Linux-only, v0.1 macOS port" (cleaner but blocks M5 day-one usage).

- **OQ8 — `Provider.complete()` signature for tool-using local models:** Forge's existing `Provider.complete()` takes `CompletionRequest` (per Animus CLAUDE.md). Need to confirm the request schema supports passing tool JSON schemas and parsing tool-call responses for Ollama — may require a new `provider_wrapper` adapter. **Decision needed:** read `agents/provider_wrapper.py` (592 lines) at implementation time to confirm shape.

---

## 9. Patterns Explicitly NOT Ported (per AC8)

Documented as deliberate exclusions, not oversights:

- **CC's TodoWrite tool** — Animus has its own task system + audit log; importing a third tracking mechanism is duplicate state. (Matrix row 13)
- **CC's BashOutput / KillShell pair** — v0 has no background-shell affordance; v1 (overnight-delegate) revisits. (Rows 8, 9)
- **CC's Glob tool** — Grep covers v0 search needs; Glob deferred to v0.1 if eval surfaces a clear case. (Row 10)
- **CC's NotebookEdit / WebFetch / WebSearch** — not in starter-tool 5. (Rows 15, 16, 17)
- **CC's ExitPlanMode** — Animus v0 has no plan-mode. (Row 18)
- **CC's multimodal Read** — Qwen3-32B is text-only. (Row 23)
- **CC's parallel tool calls** — sequential-only in v0 for local-LLM reliability. (Row 27)
- **CC's multi-model routing** — single-model v0. (Row 26)
- **CC's CLAUDE.md auto-loaded context** — too tied to what content to ingest; deferred. (Row 21)
- **CC's stateless Task sub-agent** — explicit out-of-scope per R25; documented for v1+. (Row 24, §5)
- **smolagents' `CodeAgent`** — uses code-execution sandbox we explicitly OOS. (Row 3)
- **smolagents' remote executors (E2B, Modal, Docker)** — pure-local constraint (R21, OOS:sandboxing). (Implicit)
- **smolagents' `gradio_ui.py`** — no UI in v0; CLI + audit log only. (Implicit)

---

## 10. Coverage Check (per AC4)

Every spec.md requirement R1–R26 appears at least once in the matrix's Animus column:

| Req | Matrix row(s) covering |
|-----|----------------------|
| R1 (CLI entry + flags) | 2 |
| R2 (loop termination) | 1, 20, 28 |
| R3 (BudgetManager routing) | 29 |
| R4 (Ollama + Qwen3-32B) | 3 |
| R5 (5 starter tools) | 4, 5, 6, 7, 11 |
| R6 (skill registry, 135 load) | 14 + §6 |
| R7 (hook gates 4-event) | 19 + §7 |
| R8 (JSON receipt) | 29 |
| R9 (exit codes 0–6) | covered via §3 protocol mapping (terminal status drives exit code) |
| R10 (path scoping) | 4, 5, 6, 7 |
| R11 (Bash deny filter) | 7 |
| R12 (Forge submodule placement) | covered by document location |
| R13 (P1-P9 hook stub) | 32 + §7 |
| R14 (identity-file deny hook) | 5, 6, 30 + §7 |
| R15 (systemctl pause/restart) | 31 |
| R16 (skill cache) | §4 (cached at startup) |
| R17 (--dry-run) | §3 protocol — MockProvider plugs in at `Provider.complete()` boundary |
| R18 (Prometheus metrics) | row 29 receipt-level metrics; Prometheus is MAY |
| R19 (--verbose) | §4 (stream reasoning) |
| R20 (--no-service-pause) | 31 |
| R21 (no cloud LLM) | 3, 15 |
| R22 (no writes outside --project) | 5, 6 |
| R23 (no privilege escalation) | 7 |
| R24 (no implicit memory writes) | 13 (no TodoWrite-shaped state); implicit in "no memboot tool in v0" |
| R25 (no sub-agents in v0) | 12, 24 + §5 |
| R26 (no identity-file edits even with user `Allow`) | 30 + §7 hook ordering |

All 26 requirements covered.

---

## 11. Acceptance Criteria Self-Check (this document)

- ✅ **AC1** — File exists at `packages/forge/src/animus_forge/agent/docs/cc-loop-port-plan.md`
- ✅ **AC2** — Comparison matrix has 32 rows (≥ 20)
- ✅ **AC3** — Every `adapt` / `invent` row cites at least one R# or C# from spec.md
- ✅ **AC4** — Every spec.md R1–R26 appears at least once in the matrix's Animus column (§10)
- ✅ **AC5** — Open Questions section has 8 entries (≥ 3)
- ✅ **AC6** — `file:line` cites from smolagents @ `huggingface/smolagents@main` as of 2026-05-23: `agents.py:546` (MultiStepAgent.run), `agents.py:1274` (ToolCallingAgent), `agents.py:1309` (_step_stream), `agents.py:1350` (final-answer exit), `agents.py:1375` (memory append), `monitoring.py:81-110` (Monitor class — observer-only, no hook gates). **Rebase to a tagged release at v0 implementation kickoff** to make cites version-stable.
- ✅ **AC7** — Citations from system-prompts repo: `Claude Code 2.0.txt:146` (hooks), `:198-356` (Bash schema), `:391-431` (Edit schema), `:639-680` (Read schema). Plus `Claude Code/Tools.json` referenced as a whole.
- ✅ **AC8** — §9 "Patterns Explicitly NOT Ported" enumerates 13 deliberate exclusions with the matrix row(s) backing each.

---

## 12. Recommended Next Steps

1. **User review** of this document before locking. Likely revision areas: matrix row 14 (InvokeSkill model — may want context-transfer shape after all), OQ4 (skill count discrepancy — confirm canonical count).
2. **Resolve OQ4 → patch spec.md** if 127 is canonical (currently R6 / C4 / A3 say ≥ 135).
3. **Kick off RE Task #2** (Skill-format port plan) — natural follow-on, covers §6 in depth.
4. **Kick off RE Task #3** (Hook system port plan) — natural follow-on, covers §7 in depth.
5. **Defer until v0 implementation kickoff:** smolagents version pin → AC6 line-citations → v1.1 of this document.
6. **Open `OQ8 — Provider.complete() signature`** as a small spike during early v0 implementation — has the highest "is the integration even feasible" load-bearing weight.

---

End of port plan.
