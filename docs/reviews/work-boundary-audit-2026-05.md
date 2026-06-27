# Work Boundary — Integration Audit + Tag System Scope

> Track 9 finishing work for `PERSONAL_ROADMAP.md`. Pairs with `WORK_BOUNDARY.md` (the policy).
> **Audit date:** 2026-05-15. **Purpose:** Steps 2 and 3 of Track 9 — inventory active animus integrations that could leak work context into personal memory, and scope the `personal` / `work-adjacent` tag system.

---

## Active integration inventory

Sources: `~/.config/animus/config.toml`, `packages/bootstrap/src/animus_bootstrap/gateway/channels/`, `packages/core/animus/integrations/`.

### Message gateway channels

| Channel | Status | Leak risk |
|---|---|---|
| `webchat` | **enabled** | LOW — local-only HTMX webchat at localhost:7700. Only the operator reaches it. |
| `discord` | **enabled** (no bot_token configured) | LOW today — adapter wired but no live token = no traffic. **Becomes MEDIUM if/when a Discord server is configured.** Cross-channel context can pull in non-personal conversations. |
| `telegram` | disabled | — |
| `slack` | disabled | — |
| `matrix` | disabled | — |
| `signal` | disabled | — |
| `whatsapp` | disabled | — |
| `email` | adapter exists, not in config | — (not active) |

**Action:** when activating any gateway channel beyond `webchat`, verify the channel target is personal-only (your private servers / groups). Never wire a work Discord, work Slack, or work email account into animus.

### Core integrations (Phase 1 personal-assistant layer)

Available in `packages/core/animus/integrations/`:

| Integration | Status | Leak risk |
|---|---|---|
| `google` (Calendar / Drive / Gmail) | Configurable via OAuth | **HIGH risk** if work account is OAuthed in. Work calendar events / emails / files would flow into personal memory by default. **Audit your current Google OAuth scope — make sure it's the personal account, not any work workspace.** |
| `todoist` | Configurable | LOW if used for personal todos only. Becomes MEDIUM if employer assigns work tasks via Todoist (rare). |
| `filesystem` | Active (configurable paths) | **HIGH risk** if any work directory is watched. Audit `[filesystem]` config — should only watch `~/projects/`, `~/Documents/WORK/` (your application materials, NOT employer files), and personal note dirs. **Never watch employer-issued device home dirs or mounted work network shares.** |
| `webhooks` | Configurable | LOW if outbound webhooks are to personal endpoints only. MEDIUM if inbound webhook exposes animus to external callers. |
| `arete_bridge.py` | Internal | N/A (animus-internal coordination) |
| `gorgon.py` | Legacy (Gorgon archived 2026-04-25) | Should be checked / removed if dead code |

### MCP servers (intelligence layer)

`~/.config/animus/mcp.json` registers MCP servers that animus consumes as tools.

- `auto_discover = true` — animus picks up new MCP server registrations automatically. **This is a leak surface.** If a future role provides a work MCP server (employer-issued context server), and it gets registered in this config, animus could pull work context into personal memory through that channel.

**Action:** audit `~/.config/animus/mcp.json` quarterly. Disable `auto_discover` if any work-adjacent MCP servers ever get installed on this machine.

### Forge integration

`[forge] enabled = true` — animus Forge runs workflows. Forge workflows can call tools, fetch URLs, access memory. If a workflow references work data sources (URLs, files), it carries that data through Forge's audit log.

**Action:** every workflow under `packages/forge/workflows/` should have its sources audited as personal-only. None should reference employer URLs / file paths.

---

## Leak risk summary

**Currently low risk** (this machine, today):
- Only `webchat` and (no-token) `discord` channels active
- Google integration scope unknown — needs verification
- Filesystem watcher scope unknown — needs verification
- MCP auto-discover is enabled (dormant risk)

**Highest-priority verifications before any FDE role lands:**

1. **Confirm Google OAuth is personal-account only.** Check the OAuth scope at https://myaccount.google.com/permissions — animus should appear under your personal Google account exclusively. If it ever appeared under a work Workspace account, revoke immediately.
2. **Confirm filesystem watcher paths.** `grep -A 10 '\[filesystem\]' ~/.config/animus/config.toml`. Paths should be personal-project dirs only. No `/work`, no `/employer`, no mounted shares.
3. **Confirm MCP server list.** `cat ~/.config/animus/mcp.json` — every entry should be a personally-installed MCP server. No employer-provided ones.
4. **Confirm Forge workflows don't reference work URLs.** `grep -rEn 'http[s]?://.+\..+' packages/forge/workflows/`. Spot-check.

**During-role posture (preemptive):**
- New rule: any time a new integration / MCP server / workflow is added, ask "could this pull employer data into personal memory?" If yes, configure scope strictly or don't add.
- New rule: `auto_discover = false` once a work environment exists on the same machine (defense in depth — but ideally the work environment is on separate hardware per `WORK_BOUNDARY.md`).

---

## Tag system scope

The `WORK_BOUNDARY.md` Step 3 calls for memory + task tags:

| Tag | Meaning | Default |
|---|---|---|
| `personal` | Self, family, portfolio, applications, financials | **default** |
| `portfolio` | Open-source projects, public Substack, public repos | explicit |
| `tiaid-engagement` | Anonymized methodology notes from TIAID consulting | explicit + post-scrub |
| `work-context` | Information about a current/past employer | **excluded — never stored** |
| `client-confidential` | Specific identifiable client data | **excluded — separate per-engagement encrypted store** |

### Current state

Inspected `packages/bootstrap/src/animus_bootstrap/intelligence/memory.py`. No `tag` field surfaces on the existing memory schema. Search by `tag`/`category`/`scope` came back empty. **The tag system doesn't exist yet — it has to be added.**

### Minimal implementation scope

**Schema change:**
- Add a `tags: list[str]` field to the Memory model (or a separate `memory_tags` table for many-to-many)
- Add a default tag of `personal` when no explicit tag is provided
- Add a `excluded_tags` set in MemoryManager: any memory with an excluded tag is rejected on write (raises an exception)

**Storage:**
- Backfill: existing memories get `tags=['personal']` by default
- New writes: explicit tag list, default `['personal']` if omitted

**Query surface:**
- `MemoryManager.search(query, tags=['personal'])` filters by tag
- `MemoryManager.search(query, exclude_tags=['work-context'])` excludes
- Default behavior: include all `personal` and `portfolio` and `tiaid-engagement` (post-scrub); never include `work-context` or `client-confidential`

**Excluded-tag enforcement:**
- `MemoryManager.store(content, tags=['work-context'])` raises `ExcludedTagError`
- Forces explicit operator override for any work-context storage (which the policy says shouldn't happen at all)

**Effort:** ~4-6 hours desktop work. Schema migration + tag-aware write/read paths + tests for the exclusion enforcement + backfill script.

**Test coverage required:**
- `test_default_tag_is_personal` — implicit `personal` on writes without explicit tags
- `test_excluded_tag_raises` — `work-context` write raises `ExcludedTagError`
- `test_search_filters_by_tag` — search with tag filter returns only matching
- `test_search_excludes_by_default` — search excludes `work-context` and `client-confidential` by default
- `test_backfill_migration` — existing memories get `personal` tag after migration runs

### When to implement

**Not urgent today** — current usage is overwhelmingly personal and the boundary is defended by the policy + filesystem-scope discipline. The tag system becomes urgent the moment work data could enter the system (i.e., new role starts).

**Trigger:** when an FDE role lands and you start any cross-pollination scenario (e.g., a colleague's question prompts a memory store), implement tags first. Until then, the system continues to work fine without explicit tags (everything is implicit-personal).

---

## Cross-track interactions

- **`TOOL_AUDIT_2026-05-15.md`** — `store_memory` tool will need a `tags` parameter once the schema lands. The tool registry surface is one of the audit's deduplication targets.
- **`LEARNED_AUDIT_2026-05-15.md`** — reflection-loop fix (`fix/reflection-no-feedback-bailout` branch) gathers memory signal via `MemoryManager.search`. Once tags exist, reflection should default to `personal` tag only and never reflect on `work-context` memories.
- **`PERSONAL_ROADMAP.md` Track 10** — "Resist productization" rule: don't extend the tag system to multi-tenant data-isolation tags. The 5-tag set above is intentional and small.

---

## Audit-of-audit reflection

- **Most integration risk is dormant today.** The currently-active surface (webchat, no-token Discord) is low-risk. The tag system isn't yet a forcing function for separation because there's effectively one tenant of data (personal).
- **The risk profile changes the day work data enters the system.** Tag system implementation is then time-critical, not a leisurely refactor.
- **MCP auto-discover is a quiet leak surface.** Future-you might install a work-adjacent MCP server without realizing it routes through animus. Worth disabling auto-discover even now as defense-in-depth.
- **Three quarterly verifications cost ~10 minutes each.** Google OAuth scope, filesystem watcher paths, MCP server list. Cheap insurance.

---

## Next quarterly run

Due: 2026-08-15 (or sooner if FDE role lands).

By that date, expect either:
- No change (no role yet) — re-verify Google OAuth + filesystem + MCP list, confirm tag system unbuilt-but-scoped is still acceptable
- Role landed — tag system implementation lands within first month; integration-leak audit re-runs with new role context
