# RE Task #3 — Scope: Hook System Port Plan

**Status:** Scoped 2026-05-23, executed same session
**Parent spec:** `../spec.md` v0.1.1 (R7, R13, R14, R26, §4.8, RD2, RD4, C12, A5, A16, A18)
**Sibling artifacts:** `cc-loop-port-plan.md` §7 (basic sketch), `skill-format-port-plan.md` (interaction surface — InvokeSkill body content can drive tool calls that hooks gate)
**Mode:** Research / port-plan artifact (no code)
**Timing:** Run NOW

---

## Artifact

`packages/forge/src/animus_forge/agent/docs/hook-system-port-plan.md`

## Objective

Produce a buildable port plan for the Animus v0 agent's hook system: 4-event lifecycle, built-in hooks (identity_guard, P1-P9 stub), user hook discovery + loading, fail-closed semantics, smolagents intercept strategy, and the performance envelope per C12 — sufficient for an engineer to implement `agent/hooks/` without re-deriving the design.

---

## Inputs

- `../spec.md` v0.1.1 — R7, R13, R14, R20 (`--no-service-pause`), R26, §4.8, RD2, RD4, C12 (≤ 50ms latency budget), A5/A16/A18
- `cc-loop-port-plan.md` §7 + matrix row 19 (confirmed smolagents has NO pre-tool hook interface — port requires subclass or monkey-patch)
- `skill-format-port-plan.md` (InvokeSkill bodies can issue tool calls that hooks gate — interaction surface)
- Animus monorepo `CLAUDE.md` — Non-Negotiable #2 (identity-file immutability), #4 (audit log sacred), #8 (P1-P9 documented)
- Claude Code 2.0 prompt — `~/projects/system-prompts-and-models-of-ai-tools/Anthropic/Claude Code 2.0.txt:146` (hook reference)
- smolagents source — `agents.py:1309` (`_step_stream` — where to intercept), `monitoring.py:81-110` (observer pattern, not gate pattern)
- IdentityProposalManager (per monorepo Non-Negotiable #2) — locate in animus repo for interaction reference

## Output — `hook-system-port-plan.md` sections

1. **Lifecycle event matrix** — 4 events × CC equivalent × smolagents intercept point × Animus required behavior. Target ≥ 12 rows including built-in hook rows.
2. **Hook ordering spec** — built-in identity_guard → built-in P1-P9 stub → user hooks alphabetical. Short-circuit on first Deny. Fail-closed on exception.
3. **Built-in: `identity_guard`** — full implementation spec including `IDENTITY_ROOTS` list, path-prefix matching, override prohibition (R26).
4. **Built-in: `p1_p9_validator` (stub)** — call site, stub return shape, log notice format, follow-on swap contract.
5. **User hook discovery + loading** — `~/.config/animus/agent/hooks.d/*.py` walk, function discovery by reserved name, multi-file ordering, missing-file behavior.
6. **smolagents intercept strategy** — subclass vs monkey-patch decision (with implementation sketch for chosen path).
7. **Performance envelope** — C12 (≤ 50ms per pre_tool_use); built-in latency budget; user-hook budget.
8. **Failure semantics** — hook exception = deny; receipt's `hook_errors` capture; user hooks can't catastrophically break the loop.
9. **Hook authoring guide outline** — what goes in `agent/README.md` per A15.
10. **Edge case catalog** — concrete handling for ≥ 8 cases.
11. **Open questions / gaps** — honest enumeration.
12. **Coverage check** — every spec element referenced.
13. **Acceptance criteria self-check.**

## Skills

- `/context-mapper` (light) — locate IdentityProposalManager + any existing hook patterns in Animus
- `/senior-software-analyst` — inspect Forge for hook-shaped patterns (monitoring, autonomy, sefirotic router)
- `/explain` — only if a specific Animus subsystem needs deep-reading
- `/ogma` — synthesis of the lifecycle event matrix
- No `/pattern-port` skill — third RE doc in the series; if the porting-delta friction hasn't surfaced by now, it doesn't need formalization

## Method

1. Locate `IdentityProposalManager` + any current hook patterns in Animus codebase (grep, ~5 min).
2. Confirm smolagents `_step_stream` is the intercept point and sketch the subclass approach (~10 min).
3. Build lifecycle event matrix with built-ins as first-class rows (~20 min).
4. Detail built-in `identity_guard` and `p1_p9_stub` implementations (~20 min).
5. Spec user hook discovery (~10 min).
6. Performance envelope analysis (~10 min).
7. Failure semantics + edge cases (~15 min).
8. Coverage check + AC self-check (~10 min).

## Acceptance Criteria

- **AC1** — File exists at `packages/forge/src/animus_forge/agent/docs/hook-system-port-plan.md`
- **AC2** — Lifecycle event matrix has ≥ 12 rows (4 events × built-in + user + edge variants)
- **AC3** — Built-in `identity_guard` spec includes: `IDENTITY_ROOTS` enumeration, matching algorithm, override prohibition cite (R26), failure-mode behavior
- **AC4** — Built-in `p1_p9_validator` stub spec includes: call site location, return shape, log notice format string, swap contract for follow-on hardening
- **AC5** — smolagents intercept strategy committed (subclass OR monkey-patch — single picked path with rationale)
- **AC6** — C12 latency budget broken down: built-in budget + user-hook budget; total ≤ 50ms
- **AC7** — Spec.md R7, R13, R14, R26, §4.8, RD2, RD4, C12, A5, A16, A18 each referenced at least once
- **AC8** — Edge case catalog ≥ 8 entries with explicit behavior
- **AC9** — Open Questions ≥ 3 entries
- **AC10** — At least one code-level citation from Forge codebase for identity-root path discovery (or honest "couldn't locate in this pass — see OQ" note)

## Time Budget

2–4 hours focused. **Hard cap: 5 hours** — narrower than #2.

## Out of Scope

- Writing agent code (planning only)
- Implementing built-in hooks
- Full P1-P9 validator implementation (deferred per RD2)
- Lock-file coordination with `self_heal` (deferred per RD5)
- Cross-platform launchctl path (deferred per OQ7 in cc-loop-port-plan)

## Coupling

- **Blocks:** v0 implementation of `agent/hooks/` module
- **Blocked by:** nothing — sibling docs complete
- **Informs:** v0 implementation of `Bash` tool (R11 deny filter is a sibling concept to hooks but operates inside the tool, not the hook chain)
- **Final document in the RE trilogy** — after this lands, all three port plans cover the v0 implementation surface

---

End of scope.
