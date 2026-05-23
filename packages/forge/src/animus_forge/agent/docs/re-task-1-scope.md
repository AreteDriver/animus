# RE Task #1 — Scope: Claude Code Loop → Smolagents Port Plan

**Status:** Scoped 2026-05-23, ready to execute
**Parent spec:** `../spec.md` (Animus Agent Platform v0)
**Mode:** Research / port-plan artifact (no code)
**Timing:** Run NOW in parallel with Quorum v2 — doesn't violate RD7 implementation gate.

---

## Artifact

`packages/forge/src/animus_forge/agent/docs/cc-loop-port-plan.md`

## Objective

Produce a buildable port plan that maps Claude Code's tool-use loop architecture onto a smolagents-based implementation, identifying what to copy verbatim, what to adapt for local-LLM constraints, what to drop, and what to invent.

---

## Inputs (sources to study)

- `~/projects/system-prompts-and-models-of-ai-tools` — extracted prompts / tools / loop patterns from Claude Code, Cursor, Cline, Codex
- Claude Agent SDK public docs + source (`anthropic-ai/claude-agent-sdk-python`)
- `smolagents` source — particularly `CodeAgent`, tool registry, execution path
- Animus Forge `agents/` module (existing SupervisorAgent + `provider_wrapper` for patterns we already have)
- `~/.claude/skills/` — skill format reference
- `../spec.md` v0.1.0 — constraint envelope (R1–R26, C1–C14)
- Animus monorepo `CLAUDE.md` non-negotiables (BudgetManager, audit log, P1–P9)

---

## Output — `cc-loop-port-plan.md` sections

1. **Comparison matrix** — Claude Code loop primitive × smolagents primitive × Animus required behavior, with status per row (`copy` / `adapt` / `drop` / `invent`)
2. **Decision log** — for each `adapt` / `invent` row, the constraint that drove the deviation (cited from `spec.md`)
3. **Tool execution protocol mapping** — Claude Code tool-use JSON schema → smolagents code-execution model; identify lossy translations
4. **Context management** — Claude Code compaction / sliding-window behavior vs smolagents context handling; which patterns need re-implementation
5. **Sub-agent / Agent-tool model** — out-of-scope for v0 (R25) but documented for v1+
6. **Skill registry semantics** — how Claude Code resolves `/skill` invocations; what porting requires for R6
7. **Hook lifecycle mapping** — Claude Code hooks vs proposed Animus hook gates (R7, §4.8)
8. **Open questions / gaps** — what source material doesn't make clear

---

## Skills

- `/context-mapper` — inventory pass against all inputs before reading
- `/senior-software-analyst` — codebase reads of `smolagents` + Forge `agents/`
- `/explain` — deep-dive on specific Claude Code primitives when matrix row is unclear
- `/ogma` — synthesis pass to produce the matrix
- *Do NOT preemptively build a `/pattern-port` skill — note the friction points, decide after this artifact ships whether a new skill earns its keep*

---

## Method

1. **Inventory** (`/context-mapper`): list files / sections worth reading from each input source
2. **Read**: structured notes against the inventory
3. **Synthesis** (`/ogma`): assemble the comparison matrix
4. **Decision log**: per `adapt` / `invent` row, articulate the `spec.md` constraint
5. **Coverage check**: every R1–R26 appears at least once in the matrix's Animus column
6. **Self-review**: gaps section is honestly populated (target ≥ 3 entries)

---

## Acceptance Criteria

- **AC1** — File exists at the named path
- **AC2** — Comparison matrix ≥ 20 rows
- **AC3** — Every `adapt` / `invent` row cites a `spec.md` R# or C#
- **AC4** — Every `spec.md` requirement R1–R26 appears at least once in the matrix's Animus column
- **AC5** — Gaps / open-questions section has ≥ 3 entries (zero = analysis too shallow)
- **AC6** — ≥ 3 code-level citations (`file:line`) from `smolagents` source
- **AC7** — ≥ 3 code-level citations from Claude Code SDK / `system-prompts-and-models-of-ai-tools` repo
- **AC8** — Document explicitly names which Claude Code patterns are NOT being ported and why (deliberate exclusions, not oversights)

---

## Time Budget

4–8 hours focused work. **Hard cap: 12 hours.** If it sprawls past 12, scope is too big — subdivide (e.g., split tool-execution-protocol mapping into its own session).

---

## Out of Scope

- Writing any agent code
- Implementing the port (planning only)
- Picking specific smolagents version pins (deferred to v0 implementation start, after RD7)
- Evaluating alternative base loops (smolagents picked, not re-litigated)
- Skill-format porting work (RE Task #2 — separate scope)
- Hook-system porting work (RE Task #3 — separate scope)
- Eval-suite task authoring (separate `/specification` pass per RD6)

---

## Coupling

- **Blocks:** v0 implementation start (Quorum v2 wk5 gate releases v0; port plan SHOULD be in hand before v0 code begins)
- **Blocked by:** nothing — start whenever
- **Follow-on tasks queued:**
  - RE Task #2 — Skill-format port plan (R6 / §4.7 / RD3)
  - RE Task #3 — Hook system port plan (R7 / §4.8)
