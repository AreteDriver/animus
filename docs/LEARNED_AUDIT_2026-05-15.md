# Animus LEARNED.md Audit — 2026-05-15

> Track 5 of `PERSONAL_ROADMAP.md`. First systematic audit of the self-knowledge layer.
> **Headline finding:** the reflection loop has never produced an update. LEARNED.md is its 2026-02-27 seed content, unchanged across 78 days of animus use. The reflection check has fired 8 times in proactive outcomes, but the write path isn't reaching the file.

---

## Current state

`~/.config/animus/identity/LEARNED.md` — 25 lines, last meaningful line `*Last reflection: never (first run)*`.

Content:
- 5 high-level Observations (operating principle, EVE framing, legacy motivation, Kaizen, onboarding date)
- 5 Technical Patterns (Python conventions, Rust conventions, git, security tooling, coverage gates)
- 5 Communication Patterns (lead with answer, code blocks, bullets, no restating, no emojis)

All content is accurate but generic — it's the initial seed written at onboarding, not learned-from-real-use signal.

---

## Headline finding — the reflection loop is broken

`proactive_outcomes.db / check_fires` shows the `reflection` proactive check has fired **8 times** in recent activity. The check fires. The file doesn't update. That's the bug.

Possible failure modes:
1. The reflection check fires but doesn't actually invoke the prompt that updates LEARNED.md
2. The prompt fires but produces empty output (model declines, no signal observed worth recording)
3. The prompt produces output but the file-write path errors silently
4. The prompt produces output but writes to a different path than `~/.config/animus/identity/LEARNED.md`
5. The "Last reflection: never (first run)" line is being parsed as state and short-circuiting subsequent updates

Investigation entry-points (Bootstrap source):
- `packages/bootstrap/src/animus_bootstrap/intelligence/proactive/checks/reflection.py` — the check itself
- `packages/bootstrap/src/animus_bootstrap/identity/manager.py` — file writer
- `packages/bootstrap/src/animus_bootstrap/intelligence/tools/builtin/identity_tools.py` — write tool (`identity_append_learned`)

Cross-reference: `identity_append_learned` is one of the 37 Bootstrap tools (per `TOOL_AUDIT_2026-05-15.md`). It's bucketed as "Probably rotting" in that audit because identity writes were assumed rare. **The cross-audit observation is that the writes aren't rare — they're failing silently.** Both audits should be re-read together.

---

## What the seed content gets right

Even as a static file, the seed captures durable identity:
- Operating principle ("See it through. Do it better. Leave something real.") — accurate, well-phrased
- EVE Online as systemic-thinking proxy — accurate
- Kaizen framing — accurate to working style
- Communication patterns (lead with answer, no restating, no emojis) — accurate and visible across animus interactions today

Keep this. The seed is a fine baseline. The problem isn't the seed; it's that 78 days of evolution haven't been captured on top of it.

---

## What's missing (compared to what a working reflection loop would have captured)

Based on memory + decision log activity from the last 78 days, the reflection loop *should* have surfaced things like:
- **Workflow patterns:** "Phone-remote terminal during Toyota shifts is the default build pattern" (per memory, adopted 2026-05-08)
- **Operating-rule additions:** "Never plow through a push without `git fetch` first" (per the pre-push preflight memory rule)
- **Failure modes:** "Strategic statements are exploration not conviction — wait for explicit commit" (per the exploration-vs-conviction feedback memory)
- **Tool preferences:** "Avoid `rm` directly; `rm` aliased to `trash`" (per a gotcha)
- **Cost discipline:** ET metric integration into BudgetManager (PR #41, 2026-05-12)
- **Boundary discipline:** work-vs-personal separation (Track 9 of PERSONAL_ROADMAP)
- **Self-improvement arc:** "Refuse own previous recommendations cleanly" (per the recent feedback memory)

None of these have made it into LEARNED.md. They live in the memory system (auto-memory + decision log + topic files), but the *self-knowledge consolidation* layer that LEARNED.md is supposed to be hasn't done its job.

---

## Recommended fixes

### Phase 1 — diagnose the reflection write path (~1-2 hours)

1. Add explicit logging to `reflection.py` check: log when the check fires, what prompt was generated, what output came back, where it tried to write.
2. Re-run the check manually: `animus-bootstrap reflect` (or whatever the CLI command is) and observe.
3. Identify which of the 5 failure modes above applies.
4. Fix.

### Phase 2 — re-seed with the 78 days of accumulated context (~1 hour)

Once the write path works:
1. Run the reflection prompt against a snapshot of the last 78 days of audit log + decision log + memory entries
2. Let it propose 10-20 candidate LEARNED entries
3. Human-curate: keep the high-signal ones, reject the noise
4. Append to LEARNED.md as a single "2026-05-15 reseed" block

### Phase 3 — tune the reflection prompt for ongoing signal density (~1 hour)

The current implicit prompt produces seed-style entries (high-level, generic). For ongoing reflection to be useful, it needs to bias toward:
- **Pattern-level observations** ("ARETE tends to X under condition Y") — these compound across sessions
- **Anti-patterns observed in own behavior** ("X happened twice; the rule should be Y")
- **Tool-and-workflow preferences** (specific commands / sequences that work)
- **Decision-rule additions** (new heuristics for when to do X vs Y)

Bias AWAY from:
- **Transient session details** ("worked on memboot today") — already in session notes
- **Restating known principles** ("Kaizen is important") — already in CORE_VALUES.md
- **Verbose narration** — LEARNED.md should be terse claim/why pairs

Suggested prompt sketch (for someone to refine):

> "Read the last 24 hours of audit log, decision log, and memory entries. Surface up to 3 patterns that are NEW (not already in LEARNED.md), DURABLE (would still apply next week), and SPECIFIC (operational guidance, not generic principles). Output as terse one-line claims with a one-line 'why' beneath. Reject anything that's transient session content, restates known principles, or could equally describe any developer."

### Phase 4 — quarterly compaction (~30 min)

Once LEARNED.md starts growing:
1. Quarterly: archive entries older than 6 months whose pattern hasn't recurred
2. Surface the most-referenced patterns at the top
3. Cross-link to the memory entries that informed them

---

## Cross-audit links

- `TOOL_AUDIT_2026-05-15.md` — `identity_append_learned` was bucketed "Probably rotting" but is actually "silently failing." Re-read together for the full picture.
- Memory entries that should have made it into LEARNED.md but didn't: `feedback_strategic_exploration_vs_conviction.md`, `feedback_run_preflight_before_push.md`, `feedback_dont_stop_unless_blocking.md`, `feedback_conflict_surface_rule.md`, `project_phone_remote_terminal_pattern.md`.

---

## Audit-of-audit reflection

- **Two consecutive audits, two "the data persistence is broken" findings.** TOOL_AUDIT surfaced `tool_history.db` with 1 row; LEARNED_AUDIT surfaces `LEARNED.md` with seed-only content. Pattern: **animus's introspection layer (how it knows what it's doing and what it's learned) is the weakest part of the stack right now.** This is upstream of every other improvement; fixing it makes future audits possible.
- **Seed content was good enough to mask the problem for 78 days.** The seed captures durable identity well enough that nothing felt obviously wrong. Without an audit pass, this would have continued indefinitely.
- The PERSONAL_ROADMAP's Track 5 description anticipated some of this ("Is it tracking the right patterns?") but didn't anticipate the "never wrote anything at all" failure mode. Update Track 5 in the roadmap to account for the diagnosis-first phase.

---

## Next quarterly run

Due: 2026-08-15 (or sooner if Phase 1-3 lands).

By that date, expect:
- LEARNED.md populated with 78-day reseed + ongoing additions
- Quarterly compaction ritual established
- Reflection prompt tuned and stable
