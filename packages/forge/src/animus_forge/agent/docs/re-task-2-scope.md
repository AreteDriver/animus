# RE Task #2 — Scope: Skill-Format Port Plan

**Status:** Scoped 2026-05-23, ready to execute
**Parent spec:** `../spec.md` v0.1.1 (R6, §4.7, RD3)
**Sibling artifact:** `cc-loop-port-plan.md` §6 (initial sketch of v0 InvokeSkill model)
**Mode:** Research / port-plan artifact (no code)
**Timing:** Run NOW — independent of v0 implementation start gate

---

## Artifact

`packages/forge/src/animus_forge/agent/docs/skill-format-port-plan.md`

## Objective

Produce a buildable port plan for the Claude Code skill format → Animus v0 agent skill registry, identifying the format compatibility matrix, frontmatter parser semantics, discovery algorithm, InvokeSkill tool design, and edge-case handling — sufficient for an engineer to implement `agent/skill_loader.py` and `agent/tools/invoke_skill.py` without re-deriving the design.

---

## Inputs (sources to study)

- **`~/.claude/skills/`** (127 dirs, snapshot 2026-05-23) — full sampling, not just 3-4 spot checks
  - Probe distribution: how many use YAML frontmatter, how many use `# /name - desc` legacy header, how many have edge cases (no description, missing name, multi-line keys)
- **`~/projects/animus/packages/forge/skills/`** — Animus's existing skill format for reference comparison
  - Structurally different: `SKILL.md` + separate `schema.yaml` + `registry.yaml` index file
  - Confirms why RD3 (independent loader for v0) is correct
- **`../spec.md` §4.7** — Skill loader interface contract (signature, discovery, failure handling, scope)
- **`../spec.md` R6, R16, A3, C4** — Hard requirements / constraints
- **`cc-loop-port-plan.md` §6** — Starting point for InvokeSkill semantics; expand here
- **Claude Code skill docs** if public — WebFetch `https://docs.claude.com/en/docs/claude-code/` skill pages for canonical format reference
- **Animus monorepo CLAUDE.md** — skill-related non-negotiables (none directly, but identity-file invariants matter for hooks that touch skills)

---

## Output — `skill-format-port-plan.md` sections

1. **Format inventory** — distribution of skill formats found in `~/.claude/skills/` (frontmatter vs legacy header vs edge cases). Concrete numbers from a sweep, not estimates.
2. **Format comparison matrix** — Claude Code SKILL.md fields × Animus existing skill fields × Animus v0 loader behavior, status per row (`copy` / `adapt` / `drop` / `invent`). Target ≥ 15 rows.
3. **Frontmatter parser spec** — YAML keys honored, defaults, failure modes (malformed YAML, missing required fields, unknown keys).
4. **Discovery algorithm** — directory walking, SKILL.md vs skill.md fallback, ignore patterns (e.g. `_*` private, `.*` hidden), order semantics.
5. **InvokeSkill tool design** — interface, body-as-advice vs context-transfer semantics (v0 picks body-as-advice per cc-loop §6), arg substitution shape, error cases.
6. **Edge case catalog** — concrete handling for each:
   - Missing SKILL.md in a dir under `~/.claude/skills/`
   - Malformed YAML frontmatter
   - Duplicate skill names (different dirs, same `name:` key)
   - Missing required frontmatter fields (`name`, `description`)
   - Legacy header-only skills (no frontmatter)
   - Multi-line YAML values (description spanning newlines)
   - Non-UTF-8 encoded files
   - Symlinks (follow or skip)
   - Skill name collisions with plugin-namespaced skills not on disk
7. **Performance + caching** — load time vs cache hit semantics (per R16 cache for lifetime).
8. **Open questions / gaps**

---

## Skills

- `/context-mapper` — initial inventory pass over `~/.claude/skills/`
- `/senior-software-analyst` — read Animus existing skill loader code (`packages/forge/src/animus_forge/skills/` if exists) for comparison reference
- `/explain` — deep-read on any specific Animus skill resolver entry point
- `/ogma` — synthesis of the format comparison matrix
- **Skip `/pattern-port` decision:** evaluate at completion whether RE Task #1 + RE Task #2 surface enough porting-delta friction to justify formalizing it. Not preemptive.

---

## Method

1. **Distribution sweep** — script that walks `~/.claude/skills/`, categorizes each SKILL.md by format variant (frontmatter / legacy / edge case), produces a count table. This becomes §1.
2. **Sample reads** — at least one example from each format variant, copied as a code block into the document for reference.
3. **Animus loader read** — locate and read Animus's existing skill resolution code for comparison rows in §2.
4. **Synthesis** (`/ogma`) — assemble the format comparison matrix + frontmatter parser spec.
5. **InvokeSkill spec** — expand `cc-loop-port-plan.md §6` into a full tool design including arg handling, body-as-advice semantics, error cases.
6. **Edge case enumeration** — for each of the 9+ edge cases, name the loader's behavior + cite spec §4.7 if the case is governed there.
7. **Coverage check** — every spec.md R6, R16, A3, C4, §4.7 element appears at least once.
8. **Self-review** — gaps section honestly populated (target ≥ 3 entries).

---

## Acceptance Criteria

- **AC1** — File exists at `packages/forge/src/animus_forge/agent/docs/skill-format-port-plan.md`
- **AC2** — Format inventory in §1 has concrete count per variant (not "most" or "many") — counts MUST sum to the snapshot total (127)
- **AC3** — Format comparison matrix in §2 has ≥ 15 rows
- **AC4** — Every `adapt` / `invent` row cites a `spec.md` R# / C# / §
- **AC5** — Spec.md R6, R16, A3, C4, §4.7 each referenced at least once across the document
- **AC6** — Edge case catalog has ≥ 9 entries with explicit behavior named (not "TBD")
- **AC7** — At least one verbatim sample SKILL.md block per format variant detected in §1
- **AC8** — Gaps / open-questions section has ≥ 3 entries
- **AC9** — InvokeSkill tool design is implementable: signature, return shape, all named error cases have documented behavior
- **AC10** — Document explicitly names format compatibility limits (what we DON'T parse — e.g. nested YAML, custom directives)

---

## Time Budget

2–4 hours focused work. **Hard cap: 6 hours.** Smaller scope than RE Task #1 (port plan was a loop architecture; this is a single subsystem). If it sprawls past 6h, scope is wrong — either the format variants are wilder than expected (real signal, document it) or scope creep (split into 2a + 2b).

---

## Out of Scope

- Writing any agent code (planning only)
- Implementing the loader
- Unifying with Animus's existing YAML skill resolver (deferred per RD3)
- Sub-agent skill invocation pattern (deferred to v1+ per R25)
- Skill versioning / dependency resolution (not in v0)
- Skill marketplace / discovery beyond `~/.claude/skills/`
- Hook-system semantics — covered separately by RE Task #3

---

## Coupling

- **Blocks:** v0 implementation of `agent/skill_loader.py` and `agent/tools/invoke_skill.py`
- **Blocked by:** nothing — RE Task #1 is complete; OQ4 patch applied
- **Informs:**
  - RE Task #3 (hook system port plan) — hook ordering matters when InvokeSkill fires
  - v0 eval-suite spec — at least one eval task must exercise InvokeSkill
- **Output feeds back into:** potential revision of `spec.md §4.7` if open questions surface design issues
