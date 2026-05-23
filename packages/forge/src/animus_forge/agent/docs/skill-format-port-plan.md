# Skill-Format Port Plan — Claude Code → Animus v0 Agent Skill Registry

**RE Task:** #2 (per `re-task-2-scope.md`)
**Status:** Draft v1.0 — 2026-05-23
**Parent spec:** `../spec.md` v0.1.1 — R6, R16, §4.7, A3, C4, RD3
**Sibling artifact:** `cc-loop-port-plan.md` §6 (initial InvokeSkill sketch — superseded here)
**Author:** Claude Code (advisory mode per [[project-animus-agent-platform]])

---

## 0. Sources Consulted

| Source | Location | Purpose |
|---|---|---|
| `~/.claude/skills/` (full sweep) | 127 dirs, snapshot 2026-05-23 | Distribution + format-variant samples |
| `~/projects/animus/packages/forge/skills/` | `browser/`, `email/`, `integrations/`, `system/` + `registry.yaml` | Animus's existing skill format (for comparison) |
| `../spec.md` v0.1.1 §4.7 | Skill loader interface contract | Drives loader shape |
| `cc-loop-port-plan.md` §6 | Initial InvokeSkill design (this doc supersedes) | Baseline to expand |
| Sample SKILL.md files: `/test`, `/a11y`, Animus `browser/web_scrape/SKILL.md` + `schema.yaml` | Verbatim reference snippets in §1 | Format-variant evidence |

---

## 1. Format Inventory (per AC2)

**Distribution sweep — script: `for d in ~/.claude/skills/*/; do …; done`:**

| Variant | Count | % | Detection |
|---|---:|---:|---|
| **YAML frontmatter** (`---` on line 1) | 93 | 73% | First line is `---`; second `---` terminates frontmatter |
| **Legacy header-only** (`# /name - desc`) | 34 | 27% | First line matches `^# /[a-z-]+ - .*$` |
| **Missing SKILL.md / skill.md** | 0 | 0% | No SKILL.md or skill.md in directory |
| **Other** (neither pattern) | 0 | 0% | First line is something else |
| **TOTAL** | **127** | **100%** | Sums correctly to snapshot floor (R6, C4, A3) |

**Frontmatter key distribution (within the 93 frontmatter skills):**

| Key | Count | % | Tier |
|---|---:|---:|---|
| `name` | 93 | 100% | **Required (universal)** |
| `description` | 93 | 100% | **Required (universal)** |
| `version` | 76 | 82% | Standard |
| `type` | 76 | 82% | Standard |
| `risk_level` | 76 | 82% | Standard |
| `metadata` | 76 | 82% | Standard |
| `category` | 76 | 82% | Standard |
| `trust` | 17 | 18% | Extended |
| `tools` | 17 | 18% | Extended |
| `parallel_safe` | 17 | 18% | Extended |
| `consensus` | 17 | 18% | Extended |
| `agent` | 17 | 18% | Extended |

**Effective schema tiers detected:**

1. **Minimal** (only `name` + `description`): 17 skills (93 − 76 = 17)
2. **Standard** (Minimal + version, type, risk_level, metadata, category): 76 − 17 = 59 skills
3. **Extended** (Standard + trust, tools, parallel_safe, consensus, agent): 17 skills
4. **Legacy** (no frontmatter — header-only): 34 skills

**Edge case probes:**
- Multi-line YAML descriptions (`description: |` or `description: >`): **0 detected**
- Duplicate `name:` values across skills: **0 detected**
- Symlinks under `~/.claude/skills/`: not probed yet (loader treats per OS default — follows)

### Verbatim samples (per AC7)

**Frontmatter / Minimal — `/test`:**
```markdown
---
name: test
description: Run pytest with coverage summary. Provides quick feedback on test status and coverage. Invoke with /test or /test <path>.
---

# Test Skill

Run project tests with pytest and show coverage summary.

## Usage
...
```

**Legacy header-only — `/a11y`:**
```markdown
# /a11y - Accessibility Checklist

Web accessibility audit and guidelines (WCAG compliance).

## Usage
```
/a11y                            # General checklist
/a11y path/to/component.tsx      # Audit specific component
...
```

**Frontmatter / Extended (Animus existing) — `forge/skills/browser/web_scrape/SKILL.md`:**
```markdown
---
name: web_scrape
version: 1.0.0
agent: browser
risk_level: low
description: "Fetch and extract content from web pages. Parse HTML, extract text, tables, and structured data. Handle JavaScript-rendered content with headless browser."
---

# Web Scrape Skill
...
```

**Animus existing skills also have a separate `schema.yaml`** with typed inputs/outputs, rate limits, cache config — see `forge/skills/browser/web_scrape/schema.yaml` (215 lines). This is structurally **NOT** a CC-format skill; it's a typed plugin manifest. Confirms RD3 (independent loaders).

---

## 2. Format Comparison Matrix

Status legend: **C** = copy verbatim; **A** = adapt for Animus v0; **D** = drop (not in v0 scope); **I** = invent.

| # | CC Skill Field / Concept | Animus Existing (forge/skills) | v0 Agent Loader Behavior | Status | Cite |
|---|---|---|---|---|---|
| 1 | `name` in YAML frontmatter | `name` (or `skill_name` in schema.yaml) | R6 + §4.7: required; registry key; must be non-empty string; unique across registry (collision = last-wins + warning) | **A** | R6, §4.7 |
| 2 | `description` in YAML frontmatter | `description` (often multiline in schema.yaml) | §4.7: required; shown in `--verbose` listing and InvokeSkill tool's discovery payload | **A** | R6, §4.7 |
| 3 | Markdown body after frontmatter | Markdown body in SKILL.md + structured schema.yaml | §4.7 `Skill.body`: stored verbatim as string; consumed by `InvokeSkill` (matrix row 9) | **C** | §4.7 |
| 4 | `version`, `type`, `risk_level`, `metadata`, `category` (Standard tier) | Same keys exist in Animus format | Parse if present, store in `Skill.metadata: dict[str, Any]`; not gated for load success | **A** | §4.7 |
| 5 | `trust`, `tools`, `parallel_safe`, `consensus`, `agent` (Extended tier) | `agent`, `risk_level`, `consensus_required` in schema.yaml | Parse and store in `Skill.metadata`; ignored by v0 InvokeSkill but available to user hooks | **A** | §4.7 |
| 6 | Legacy header format: `# /name - description` (line 1) | N/A (Animus skills always have frontmatter or schema) | INVENT: legacy-parser fallback — extract `name` and `description` from line 1 via regex `^# /([a-z][a-z0-9-]*) - (.+)$`. If parse fails, skip with warning | **I** | §4.7 failure handling |
| 7 | SKILL.md filename | SKILL.md (uppercase canonical) | §4.7: try `SKILL.md` first, then `skill.md`; both probed, first found wins. Other filenames = skill not discovered | **C** | §4.7 |
| 8 | Multi-line YAML values (block style `|` or `>`) | Used in Animus schema.yaml descriptions | v0 loader: use `yaml.safe_load` — handles both styles natively. Tested only on flat values per inventory (0 multiline observed) | **A** | §4.7 |
| 9 | `/skill-name args...` invocation (CC SlashCommand) | N/A in agent loader (Animus skills are RPC-style with schema.yaml) | INVENT: `InvokeSkill(name, args=None) -> str` tool — returns skill body (+ optional `## Arguments\n{args}` appendix). NOT a nested loop in v0 (sub-agent shape deferred per R25) | **I** | R6, R25, §4.7, §5 below |
| 10 | Skill registry / catalog | `forge/skills/registry.yaml` (manually maintained index) | INVENT: auto-built from disk walk at startup; cached for agent lifetime (R16). No registry.yaml — registry is the directory layout | **I** | R6, R16 |
| 11 | Skill discovery order | Animus uses explicit registry.yaml entries | §4.7: alphabetical directory walk under `~/.claude/skills/`; deterministic ordering for reproducibility | **A** | §4.7 |
| 12 | Plugin-namespaced skills (`vercel:bootstrap`, `arete-cc-stack:hackathon-triage`) | Not present on disk; appear in session tooling output | DROP for v0 — these live in plugin manifests not in `~/.claude/skills/`; the 127→135 discrepancy patched in spec v0.1.1 reflects this. Loader sees only on-disk skills | **D** | OQ4 (resolved) |
| 13 | Skill cache | None in Animus | R16: load all skills at agent startup, hold in `dict[str, Skill]` for agent lifetime. No watch / reload mechanism in v0 (`--dry-run` reloads on next CLI invocation only) | **A** | R16 |
| 14 | Failure isolation per skill | Animus loads via registry.yaml — failure of one entry = error | §4.7: per-skill load errors logged + skipped; aggregate count in `--verbose` + audit log + receipt's `skill_load_warnings`. Loader returns successfully-loaded subset | **A** | §4.7 failure handling |
| 15 | `invokable` flag (skill is reference vs callable) | Implicit in Animus (everything is callable via schema) | §4.7 `Skill.invokable: bool`, default `True`. Future use: mark some skills as "reference only — read body for context but don't InvokeSkill" via explicit `invokable: false` frontmatter | **A** | §4.7 |
| 16 | Skill arguments | Animus uses typed `inputs` in schema.yaml | v0: free-form string passed verbatim into body as `## Arguments\n{args}` appendix. No type validation, no schema parsing — agent interprets per skill body's documentation | **I** | §4.7, R25 (deferred for sub-agent dispatch shape) |
| 17 | Skill body templating | None in CC; Animus skills don't template either | DROP for v0 — args appended literally, no `{{var}}` substitution. v0.1+ if eval surfaces need | **D** | OOS |
| 18 | UTF-8 encoding | Both formats assume UTF-8 | §4.7 fail mode: non-UTF-8 SKILL.md → log warning + skip (per failure-isolation row 14) | **A** | §4.7 |
| 19 | Hidden / private skill convention (`_foo` or `.foo`) | Animus has no convention | INVENT: directories starting with `.` or `_` are skipped silently (no warning) — convention for personal/work-in-progress skills | **I** | §4.7 |
| 20 | Symlink handling | Animus doesn't model | v0: follow symlinks (default Python behavior); document in README. v0.1 if loop-back symlinks cause issues | **A** | §4.7 |

**Row count: 20** (≥ 15 required by AC3). Status: **2 copy, 11 adapt, 5 invent, 2 drop.**

---

## 3. Frontmatter Parser Spec

**Parser:** `yaml.safe_load()` from PyYAML (already a Forge / Animus core dependency per monorepo CLAUDE.md).

**Algorithm** (per file):

```python
def parse_skill_file(path: Path) -> Skill:
    """Parse a SKILL.md file (frontmatter or legacy). Raises SkillParseError."""
    raw = path.read_text(encoding="utf-8")  # row 18: UTF-8 enforced
    lines = raw.splitlines(keepends=True)

    if lines and lines[0].rstrip() == "---":
        # Frontmatter variant (row 1, 2, 4, 5)
        end_idx = next(
            (i for i in range(1, len(lines)) if lines[i].rstrip() == "---"),
            None,
        )
        if end_idx is None:
            raise SkillParseError(f"{path}: frontmatter opened with --- but never closed")
        try:
            meta = yaml.safe_load("".join(lines[1:end_idx])) or {}
        except yaml.YAMLError as e:
            raise SkillParseError(f"{path}: malformed YAML frontmatter: {e}")
        body = "".join(lines[end_idx + 1:]).lstrip("\n")
    else:
        # Legacy header-only variant (row 6)
        match = LEGACY_HEADER_RE.match(lines[0]) if lines else None
        if not match:
            raise SkillParseError(f"{path}: neither frontmatter nor legacy header detected on line 1")
        meta = {"name": match.group(1), "description": match.group(2).strip()}
        body = "".join(lines[1:]).lstrip("\n")

    name = meta.get("name")
    if not isinstance(name, str) or not name:
        raise SkillParseError(f"{path}: missing or non-string 'name' field")
    desc = meta.get("description", "")
    if not isinstance(desc, str):
        raise SkillParseError(f"{path}: 'description' must be a string (got {type(desc).__name__})")

    return Skill(
        name=name,
        description=desc,
        body=body,
        invokable=bool(meta.get("invokable", True)),
        source_path=path,
        metadata={k: v for k, v in meta.items() if k not in ("name", "description", "invokable")},
    )


LEGACY_HEADER_RE = re.compile(r"^# /([a-z][a-z0-9-]*) - (.+)$")
```

**Honored keys (v0):** `name` (required, str), `description` (required, str — defaults `""` if absent, NOT a fatal error), `invokable` (optional bool, default `True`).
**Stored in `metadata`:** everything else found in frontmatter — kept as-is for user hooks / future use.
**Unknown / unexpected keys:** silently preserved in `metadata`. No "unknown key" warnings (would create noise across the diverse extant skills).

---

## 4. Discovery Algorithm

```python
def load_skills(root: Path = Path("~/.claude/skills").expanduser()) -> tuple[dict[str, Skill], list[str]]:
    """Walk root, load all skills, return (registry, warnings)."""
    registry: dict[str, Skill] = {}
    warnings: list[str] = []

    if not root.is_dir():
        warnings.append(f"skill root does not exist: {root}")
        return registry, warnings

    for entry in sorted(root.iterdir()):  # row 11: alphabetical
        if not entry.is_dir():
            continue
        if entry.name.startswith(".") or entry.name.startswith("_"):  # row 19: hidden/private
            continue

        skill_md = entry / "SKILL.md"
        if not skill_md.exists():
            skill_md = entry / "skill.md"  # row 7: lowercase fallback
        if not skill_md.exists():
            continue  # silent skip (most dirs without SKILL.md aren't skills)

        try:
            skill = parse_skill_file(skill_md)
        except SkillParseError as e:
            warnings.append(str(e))
            continue  # row 14: failure isolation
        except UnicodeDecodeError as e:
            warnings.append(f"{skill_md}: not UTF-8 ({e})")
            continue  # row 18

        if skill.name in registry:
            warnings.append(
                f"duplicate skill name '{skill.name}': "
                f"{skill.source_path} overrides {registry[skill.name].source_path}"
            )  # row 1: collision = last-wins + warning

        registry[skill.name] = skill

    return registry, warnings
```

**Properties:**
- Deterministic (alphabetical walk).
- Fail-soft (warnings, not exceptions).
- Cache-friendly (single pass, holds everything in memory per R16).
- O(N) where N = directories under root.

---

## 5. InvokeSkill Tool Design

**Purpose:** Let the agent retrieve a skill's body to use as context-of-the-moment guidance for its current turn.

**Signature:**
```python
def InvokeSkill(name: str, args: str | None = None) -> str:
    """Look up skill by name. Return skill body, optionally with args appended."""
```

**Behavior:**
1. Look up `name` in registry (case-sensitive).
2. If not found → raise `ToolError(f"skill not found: {name}. Use ListSkills tool to see available.")`. (`ListSkills` is a v0.1 candidate — for v0, agent has skill names in its system prompt's skill catalog.)
3. If found but `invokable=False` → raise `ToolError(f"skill '{name}' is reference-only (invokable=False)")`.
4. If `args` is None: return `skill.body` verbatim.
5. If `args` is provided: return `skill.body + "\n\n## Arguments\n" + args`.

**Return shape:** plain string. The agent reads it on next turn and uses it as context. The skill body is NOT a nested loop, NOT a sub-agent spawn — it is advice for the agent's current execution.

**Semantic implications:**
- Skill body becomes part of agent memory for subsequent turns (until context-compaction evicts it). May warrant explicit eviction after some turn count — see OQ2 below.
- An agent can chain InvokeSkill calls; bodies stack up in memory until compaction.
- No sandboxing of skill body content — skill author is trusted (it's the user's own `~/.claude/skills/` dir).

**Differences from CC's SlashCommand:**
- CC's `/skill args...` transfers conversation context to the skill (skill body essentially becomes the new system prompt). v0 InvokeSkill is much weaker — body is appended as observation, agent decides what to do with it.
- Trade-off: weaker semantics = simpler implementation; stronger semantics = sub-agent shape (deferred to v1 per R25).
- If v0 eval shows agents struggle to "actually follow" skill bodies via this weak semantic, v0.1 promotes InvokeSkill to spawn a fresh sub-loop with skill body as system prompt.

**Error cases (per AC9):**

| Case | Behavior |
|---|---|
| Name not in registry | `ToolError("skill not found: {name}")` |
| Name present, `invokable=False` | `ToolError("skill '{name}' is reference-only (invokable=False)")` |
| Name present, body is empty string | Return `""` + optional args appendix. Agent observes empty content. No error. |
| Args provided to a skill with no `## Arguments` documentation | Args still appended verbatim. Agent interprets per skill body. |
| Non-string args (e.g. dict from a confused model) | smolagents tool validation rejects before reaching the function. If it slips through: `ToolError("args must be a string")` |

---

## 6. Edge Case Catalog (per AC6 — ≥ 9 entries)

| # | Edge case | v0 loader / InvokeSkill behavior | Cite |
|---|---|---|---|
| 1 | Directory under `~/.claude/skills/` with no SKILL.md / skill.md | Silently skip (not all dirs are skills) | §4 algorithm |
| 2 | Malformed YAML frontmatter (parse error) | Log warning in `skill_load_warnings`; skip this skill; continue loader | §3 algorithm, §4 row 14 |
| 3 | Frontmatter opened with `---` but never closed | Treated as parse error → same as #2 | §3 algorithm |
| 4 | Duplicate `name:` across two skills | Last-wins (alphabetical order = later dir overrides); log warning | §4 row 1, §4 algorithm |
| 5 | Missing required field (`name` absent or non-string) | Parse error → warning + skip | §3 algorithm |
| 6 | `description` missing | Default to `""`; load succeeds (not fatal) | §3 algorithm |
| 7 | Legacy header-only skill without matching regex (e.g. uppercase name `# /Foo - desc`) | Parse error → warning + skip. Loader does not silently downcase; case mismatch surfaces as a clear failure | §3, row 6 regex strict |
| 8 | Multi-line YAML description (`description: \|` block style) | Handled natively by `yaml.safe_load` — no special-case code | §3, row 8 |
| 9 | Non-UTF-8 file encoding | `UnicodeDecodeError` caught; log warning; skip | §3, §4, row 18 |
| 10 | Symlink in `~/.claude/skills/` pointing into the tree | Followed by default Python behavior; document in README. Loop-back symlinks unaddressed in v0 (rare in practice) | row 20 |
| 11 | Hidden directory (`.foo`) or underscore-prefixed (`_foo`) | Silently skipped (private convention) | row 19, §4 algorithm |
| 12 | Skill body is empty | Loader succeeds; `Skill.body = ""`; InvokeSkill returns empty string + args appendix if any | §5 error table |
| 13 | InvokeSkill called with unknown skill name | `ToolError("skill not found: {name}")` — agent gets clear feedback | §5 |
| 14 | `invokable: "yes"` (string instead of bool) | `bool("yes") == True` → treated as invokable. Lenient. Document in README; YAML linting would catch upstream | §3 algorithm |

**14 edge cases** (≥ 9 required).

---

## 7. Performance + Caching

**Load cost:** 127 files × ~1KB avg + YAML parse → estimated <100ms cold on the user's hardware. Acceptable for agent startup; no lazy-load needed in v0.

**Cache strategy:** R16 — load once at `agent_start`, hold in `dict[str, Skill]` for the lifetime of the `animus-agent run` invocation. No watch/reload (no `inotify`, no polling). Files changed during agent execution will not be picked up — by design (deterministic execution).

**Memory cost:** 127 skills × ~3KB body avg = ~400KB resident. Negligible.

**Failure path performance:** parse errors caught per file; load loop continues. No per-failure retry. Total cost bounded by directory count even if all parses fail.

---

## 8. Open Questions / Gaps

- **OQ1 — InvokeSkill body-as-advice vs context-transfer:** v0 picks the weak semantic (body appended as observation). If eval suite shows agents fail to "actually follow" skill bodies (e.g. they read but don't enact `/review` instructions), this is the lever to pull. Promote to spawn a sub-loop with skill body as system prompt — but that's sub-agent shape (R25 deferred). **Decision needed:** measure first eval run, then revisit.

- **OQ2 — Skill body memory eviction:** Once InvokeSkill returns a 5KB body, it lives in context until compaction. An agent that calls InvokeSkill 10 times in a long run could blow context. **Decision needed:** add explicit `evict_skill(name)` tool to v0, or trust compaction to handle it? Probably no v0 tool — keep simple, observe behavior.

- **OQ3 — Plugin-namespaced skills (`vercel:*`, `arete-cc-stack:*`):** Not on disk; loader can't see them. v0 silently lives without them. **Decision needed:** is the agent supposed to know they exist (and refuse to InvokeSkill them with a clear "plugin skills not supported in v0" message), or stay silent? Either way is fine for v0; clear-message is friendlier.

- **OQ4 — `ListSkills` tool for discovery:** Without it, the agent's system prompt must enumerate the 127 skill names + descriptions to make InvokeSkill discoverable — that's a chunk of tokens. With `ListSkills(filter=None)` as a tool, agent fetches the catalog on demand. **Decision needed:** include in v0 starter tool list (changes "5 starter tools" → 6 — needs spec.md R5 amendment), or wait for v0.1?

- **OQ5 — Skill-name namespace collisions with built-in tool names:** What if a user has a skill named `Bash` or `WriteFile`? Loader would happily load it; InvokeSkill would shadow nothing (skill registry and tool registry are separate). But the agent might get confused. **Decision needed:** warn on loader pass if a skill name collides with a built-in tool name. Cheap to implement, valuable for surprise reduction.

- **OQ6 — Reload-on-edit during agent run:** v0 is deterministic (no reload). v1+ overnight-delegate use case may want reload (long-running agent picks up new skills). **Decision needed:** punt to v1 spec.

---

## 9. Patterns Explicitly NOT Ported (per AC10 / format compatibility limits)

- **Animus `schema.yaml` typed-inputs format** — v0 loader doesn't parse it. Animus's existing forge/skills/* skills are not loadable by the agent's CC-format loader. This is correct per RD3.
- **`registry.yaml` index file** — registry is the directory walk; no index file consumed or generated by v0 loader.
- **Nested skill packages** (e.g. `~/.claude/skills/python/lint/SKILL.md`) — v0 walks only top-level dirs. Nested skills not discovered. Document in README; defer to v0.1 if user creates nested skill layouts.
- **Skill dependencies / `requires:` clauses** — neither format has them; not invented for v0.
- **Skill versioning / pins** — `version:` field is parsed and stored in metadata but not enforced for compatibility.
- **Templating in skill bodies** (`{{var}}`, Jinja, mustache) — not in either format; not added in v0. Args appended literally.
- **Skill marketplace / remote loading** — `~/.claude/skills/` is the only source. No HTTP, no git pull, no plugin registry.
- **Multi-language skills** (skills written in non-Python languages) — body is markdown only; v0 doesn't execute skill code. The "skill" IS the markdown prompt.
- **CC's SlashCommand context-transfer semantics** — see OQ1 / §5; v0 uses weaker body-as-advice model.

---

## 10. Coverage Check (per AC5)

| spec.md element | Doc reference |
|---|---|
| R6 (Skill Registry, 127 floor) | §1 (count), §2 row 1, §4 algorithm |
| R16 (cache for agent lifetime) | §2 row 13, §7 |
| A3 (`len(registry.list()) >= 127`) | §1 inventory establishes 127 floor; loader §4 produces the registry |
| C4 (`tests/agent/test_skill_registry.py::test_loads_all_installed_skills`) | §4 algorithm directly implements; §1 confirms count |
| §4.7 Skill Loader interface contract | §2 row 1-2 (input/output), §3 (parser), §4 (discovery), §6 (edge cases) — every §4.7 sub-element addressed |
| §4.7 failure handling | §2 row 14, §3 algorithm, §4 algorithm, §6 edge cases |
| RD3 (independent loader) | §1 (Animus schema.yaml structurally different), §2 row 10, §9 (NOT ported list) |

All required spec elements referenced.

---

## 11. Acceptance Criteria Self-Check

- ✅ **AC1** — File exists at `packages/forge/src/animus_forge/agent/docs/skill-format-port-plan.md`
- ✅ **AC2** — Format inventory in §1 sums to 127 (93 + 34 + 0 + 0) — concrete counts per variant
- ✅ **AC3** — Format comparison matrix in §2 has 20 rows (≥ 15)
- ✅ **AC4** — Every `adapt` / `invent` row cites spec.md R#, C#, or §
- ✅ **AC5** — Coverage table in §10 — every required spec element referenced
- ✅ **AC6** — Edge case catalog in §6 has 14 entries (≥ 9), all with explicit behavior
- ✅ **AC7** — Three verbatim sample SKILL.md blocks in §1 (frontmatter/minimal, legacy, frontmatter/extended)
- ✅ **AC8** — Gaps section in §8 has 6 OQs (≥ 3)
- ✅ **AC9** — InvokeSkill tool design (§5) has signature, return shape, 5 named error cases
- ✅ **AC10** — §9 "Patterns Explicitly NOT Ported" enumerates 9 format compatibility limits

All 10 ACs met.

---

## 12. Recommended Next Steps

1. **User review** of this document before locking. Likely revision areas: OQ1 (InvokeSkill semantic strength), OQ4 (`ListSkills` tool — affects spec R5 starter-tool count).
2. **Resolve OQ4 → potential spec.md patch** if `ListSkills` belongs in v0 (would change "5 starter tools" → 6).
3. **Kick off RE Task #3** (Hook system port plan) — `cc-loop-port-plan.md` §7 already covered the basics; #3 will go deeper on hook ordering interaction with skill loader (built-in identity-guard runs before any InvokeSkill body could mutate files via tool calls within the body) and built-in P1-P9 stub call site.
4. **Defer until v0 implementation:**
   - `agent/skill_loader.py` writing — this doc spec is buildable
   - `agent/tools/invoke_skill.py` writing — spec'd in §5
   - Unit tests covering the 14 edge cases in §6 + the format-variant test fixtures
5. **One follow-on artifact during v0 implementation:** a `--list-skills` CLI flag separate from the agent's `InvokeSkill` tool — useful for debugging the loader and for the eval harness to enumerate available skills.

---

End of port plan.
