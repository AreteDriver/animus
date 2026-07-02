# Documentation Gap Analysis

**Last updated:** 2026-07-01

**Completed in this session:**
- ✅ Created `docs/reference/tools.md` — 14 tools documented with parameters, examples, and local model reliability notes
- ✅ Added Tools link to README.md header navigation
- ✅ Added Tools to mkdocs.yml nav
- ✅ Fixed broken internal link in tools.md

---

## Critical Gaps (Users blocked without these)

### 1. ❌ CLI Commands Reference
**Where:** Nowhere in docs
**What exists:** `packages/core/animus/__main__.py` has 40+ REPL commands (`/status`, `/stats`, `/history`, `/recall`, `/remember`, `/tag`, `/untag`, `/search-tags`, `/fact`, `/procedure`, `/export`, `/import`, `/backup`, `/tools`, `/tool`, `/task`, `/decide`, `/research`, `/brief`, `/briefing`, `/nudges`, `/meeting-prep`, `/reflect`, `/eval`, `/entities`, `/entity`, `/sync`, `/forge`, `/server`, `/voice`, `/speak`, `/speak-toggle`, `/build`, `/model`, `/auto`, `/deep`, `/learning`, `/guardrails`, `/unlearn`)
**Impact:** Users must discover commands via `/help` or by reading 2600-line source file
**Fix:** Create `docs/reference/cli-commands.md`

### 2. ❌ Memory System Reference
**Where:** Nowhere in docs
**What exists:** `MemoryLayer` with semantic/episodic/procedural types, tags, confidence scores, `recall()`, `remember()`, `forget()`, search by tags
**Impact:** Users don't know how memory works, what gets stored, how to search effectively
**Fix:** Create `docs/reference/memory.md`

### 3. ❌ Configuration Reference
**Where:** `operators/configuration.md` exists but is it complete?
**What exists:** `AnimusConfig` with nested settings for model, memory, learning, proactive, entities, api, voice, autonomous, tools_security, citizen_zero, sync
**Impact:** Users can't tune behavior without reading source
**Fix:** Audit and expand `operators/configuration.md` with full config schema and examples

---

## Important Gaps (Significant friction)

### 4. ❌ API Reference
**Where:** Nowhere in docs
**What exists:** `APIServer` in `packages/core/animus/api.py` with FastAPI endpoints. No OpenAPI/Swagger docs generated.
**Impact:** Can't integrate with Animus API without reading source
**Fix:** Enable FastAPI auto-generated OpenAPI docs; create `docs/reference/api.md`

### 5. ❌ Local Model Troubleshooting
**Where:** Nowhere in docs
**What exists:** `docs/getting-started/ollama-setup.md` covers installation but not the constrained loop reliability issues
**Impact:** Users think Animus is broken when natural language mode garbles
**Fix:** Add "Working with Local Models" section to `ollama-setup.md` or create `docs/operators/local-models.md`

### 6. ❌ Architecture: Memory Layer
**Where:** `docs/architecture/overview.md` mentions Memory Layer but no dedicated doc
**What exists:** Kernel handles episodic/semantic/procedural memory with ChromaDB/SQLite/PostgreSQL backends
**Fix:** Create `docs/architecture/memory.md`

### 7. ❌ Architecture: Learning Layer
**Where:** Nowhere in architecture docs
**What exists:** Pattern detection, guardrails, approval workflow, rollback checkpoints
**Fix:** Create `docs/architecture/learning.md`

### 8. ❌ Architecture: Proactive Engine
**Where:** Nowhere in architecture docs
**What exists:** Morning briefings, nudges, meeting prep, background scanning
**Fix:** Create `docs/architecture/proactive.md`

### 9. ❌ Citizen Zero Documentation
**Where:** Only exists in `packages/core/animus/citizen_zero.py` and REPL help
**What exists:** Constitutional corpus, identity projection, reflection, eval reports
**Impact:** Users don't understand the identity/safety layer
**Fix:** Create `docs/architecture/citizen-zero.md` or `docs/reference/citizen-zero.md`

---

## Performance & Observability Gaps

### 10. ❌ No Performance Telemetry System
**Where:** Nowhere in codebase
**What exists:** Token counts are tracked per-workflow but not per-REPL-interaction. No latency histograms. No memory growth tracking.
**Impact:** Can't answer "Is Animus getting slower over time?" Can't optimize what isn't measured.
**Measurable audit factors available now:**
- Token burn rate per session (already tracked in Forge workflows, missing in REPL)
- Tool call latency distribution (not tracked)
- Memory recall speed by backend (SQLite vs PostgreSQL vs ChromaDB)
- Model inference time by provider (Ollama vs Anthropic vs OpenAI)
- Conversation fragmentation overhead (auto-save every 10 messages → new conversation ID)
- Database size growth over time
- Cache hit rate for file reads
- Approval callback latency (human-in-the-loop delay)
- Context window utilization ratio
- Dual-model routing frequency (how often fallback triggers)

**Fix:** Create `docs/operators/performance.md` + instrument `__main__.py` with lightweight metrics collection (no external dependencies — just logging with structured JSON)

### 11. ❌ No Self-Profiling Framework
**Where:** Nowhere in codebase
**What exists:** Benchmark tests exist (`test_benchmarks.py`) but no runtime profiling
**Overlooked factors:**
- The constrained agent loop itself is a black box — no measurement of parse failures, retries, or format correction rounds
- Memory layer query performance varies wildly by backend but isn't benchmarked at runtime
- File I/O is synchronous in an async event loop — potential blocking
- The `atexit` cleanup handler blocks on `loop.run_until_complete()` which can hang if sync client disconnect fails
- `conversation.add_message()` stores full content with no truncation — long tool outputs bloat context
- `last_session_context = last.content[:500]` is arbitrary truncation, not token-aware
- `AGENT_CONTEXT` + `last_session_context` + tool menu + user input + memory context can exceed 8K context window before generation even starts
- No measurement of "time to first token" for streaming output
- No tracking of which tools fail most often (useful for improving the constrained loop)
- No measurement of memory leak potential — `conversation.messages` grows unbounded between auto-saves

**Fix:** Add lightweight `animus.profiler` module with context managers for timing critical paths. Output to rotating log files, not external services.

### 12. ❌ No Cost-Per-Operation Analysis
**Where:** Nowhere in docs or code
**What exists:** Forge tracks `$cost_usd` per workflow agent, but REPL has no cost visibility
**Impact:** Users running local models think they're "free" but don't see electricity/GPU-wear costs. Users on API keys don't know their burn rate.
**Fix:** Add optional cost estimation to REPL status (`/stats` command) showing estimated tokens consumed, estimated cost, and local GPU utilization if available.

---

## Stale References (Cleanup needed)

### 13. ⚠️ `gorgon` references in docs
**Files affected:**
- `docs/reviews/2026-02-21-self-improve-deep-dive.md` — `.gorgon/snapshots/` paths (historical, may be intentional)
- `docs/whitepapers/ANIMUS_WHITEPAPER_2026-06.md` — multiple references to incomplete rename (legitimate gap analysis)
- `docs/planning/phase-0-results.md` — claims Forge README stale refs are "fixed" — **need to verify**
- `docs/planning/documentation-roadmap.md` — same claim

**Action:** Verify `packages/forge/README.md` has zero `gorgon` CLI references. If fixed, close tracking items. If not, fix README.

### 14. ⚠️ `packages/forge/README.md`
**Check:** `grep -n "gorgon" packages/forge/README.md` — if matches exist, those are stale references

---

## Orphaned / Unlinked Docs

### 15. ⚠️ `docs/getting-started/` docs not in mkdocs nav
**Check:** `use-cases.md`, `case-study.md`, `interface-vision.md`, `animus-context.md`, `ollama-setup.md` — are these linked from Quickstart/Installation? Some may not be discoverable via the site nav.

### 16. ⚠️ `docs/reference/safety.md` vs `docs/reference/security.md` vs `docs/reference/security-layer.md`
**Check:** Three security-related docs. Are they duplicates? Do they have distinct scopes?

### 17. ⚠️ `docs/operators/known-issues.md`, `docs/operators/migration-guide.md`, `docs/operators/recovery.md`
**Check:** Not in mkdocs nav. Are they reachable?

---

## Package README Gaps

### 18. ⚠️ `packages/core/README.md`
**Check:** Does it document the REPL, tools, and CLI commands? Or just the import API?

### 19. ⚠️ `packages/forge/README.md`
**Check:** Does `animus-forge` CLI syntax match current code? (Historical gorgon rename issue)

### 20. ⚠️ `packages/bootstrap/README.md`
**Check:** Does it cover all CLI subcommands? (`install`, `setup`, `start`, `stop`, `status`, `update`, `dashboard`, `config`, `channels`, `tools`, `proactive`, `automations`, `personas`, `feedback`)

---

## P4: Performance & Observability

### Where: Nowhere in codebase or docs
### What exists: Token counts tracked per-workflow (Forge), but REPL has zero telemetry

**Impact:** Can't answer "Is Animus getting slower over time?" Can't optimize what isn't measured. Users on local models don't know their token burn rate. Users on API keys don't know their spend.

**Measurable audit factors (collectable without performance degradation):**

| Factor | What It Measures | Collection Method |
|---|---|---|
| **Token burn rate** | Tokens consumed per REPL interaction | Already tracked in Forge workflows; add to REPL `think_with_tools` return metadata |
| **Tool call latency** | Time from tool selection to result | `@contextmanager` timing in `_think_with_tools_constrained` loop |
| **Model inference latency** | Time from prompt submission to first token | Hook into `primary.generate()` call in `cognitive.py` |
| **Memory recall speed** | Query time by backend (SQLite vs PostgreSQL vs ChromaDB) | Instrument `memory.recall()` with timing |
| **Context window utilization** | Input tokens / max context ratio | Calculate before each `generate()` call |
| **Conversation fragmentation** | How often auto-save triggers (every 10 messages) | Count `Conversation.new()` calls |
| **Database size growth** | SQLite/ChromaDB file size over time | Daily measurement via background task |
| **Cache hit rate** | File read cache effectiveness | Track `read_file` repeated access patterns |
| **Approval callback latency** | Human-in-the-loop delay | Time from prompt to user response |
| **Dual-model routing frequency** | How often fallback model triggers | Count fallback invocations |
| **Constrained loop failure rate** | Parse failures / total iterations | Already partially tracked via iteration count; log failure reason |
| **Memory leak indicators** | RSS growth over session lifetime | Simple `psutil.Process().memory_info().rss` check on exit |

**Overlooked factors:**
- **Synchronous file I/O in async loop:** `read_file`, `write_file`, `edit_file` use blocking `pathlib.Path.read_text()` / `write_text()` inside the asyncio event loop. This blocks other coroutines.
- **Unbounded message growth:** `conversation.messages` grows without token-aware truncation between auto-saves. Long tool outputs (like the 2000-line watchlist JSON we just saw) bloat context for subsequent interactions.
- **Arbitrary context truncation:** `last.content[:500]` is character-based, not token-aware. Could be 50 tokens or 500 tokens depending on content.
- **No warmup measurement:** First Ollama call after idle is slow (model load from disk). No distinction between cold-start latency and steady-state latency.
- **No GPU utilization tracking:** Local model users don't know if they're CPU-bound or GPU-bound.
- **No context compression ratio:** How much of the prompt is system instructions vs user input vs tool results? Useful for optimizing the constrained loop instructions.

**Fix:**
1. Add lightweight `animus.profiler` module with context managers (zero external dependencies)
2. Log structured JSON to rotating file: `~/.animus/logs/performance.log`
3. Fields: `timestamp`, `phase`, `duration_ms`, `tool_name`, `model_provider`, `success`, `context_tokens`, `response_tokens`
4. Document schema in `docs/operators/performance.md`
5. Add `/stats --perf` command to REPL for real-time summary

---

## P5: Fix Agent Loop for Local Models (Critical Reliability)

### Where: `packages/core/animus/__main__.py` line 2465+
### What exists: Constrained agent loop (`_think_with_tools_constrained`) assumes model can follow `TOOL: <number>` format

**Impact:** Local 7B-8B models cannot reliably follow the structured format. They output prose, get format error prompts, retry, and hit max iterations with garbled output. This is the #1 user-facing bug.

**Evidence:**
- `watchlist-list` → model outputs `TOOL: 12` repeatedly, never executes
- `animus_watchlist_scan` → same behavior
- `show me the README` → model outputs prose instead of `TOOL: 2` + `path: README.md`
- After 3 iterations (patched down from 8), bails with `WARNING: Max constrained iterations reached`

**Root cause:** The system prompt presents a numbered tool menu (`1. read_file`, `2. list_files`, etc.) and instructs the model to respond with `TOOL: <number>`. Local models lack the instruction-following precision for this format. They treat the menu as content to summarize rather than a UI to interact with.

**Options considered:**

| Option | Approach | Effort | Reliability |
|---|---|---|---|
| A | Skip agent loop when provider == Ollama; require `/tool` direct invocation | Small | ✅ Perfect |
| B | Simplify format to single-word tool names (no numbers) | Medium | ⚠️ Better but not fixed |
| C | Add retry with format examples | Small | ❌ Worse (more garbage accumulation) |
| D | Use structured output (JSON mode) if model supports it | Large | ❌ Ollama models don't consistently support JSON mode |

**Recommended: Option A** — Honest about local model limitations. Preserve agent loop for Anthropic/OpenAI users who have models with native tool_use support.

**Implementation:**
```python
# In __main__.py, before entering agent loop:
if cognitive.primary_config.provider.value == "ollama":
    console.print("[dim]Tip: Use /tool <name> for tool execution with local models.[/dim]")
    # Fall through to basic think() without tools
    response = cognitive.think(user_input, context, mode, citizen_context=citizen_context)
    console.print(response)
    continue
```

**Add to docs:** `docs/operators/local-models.md` — "Why natural language doesn't work with local models and how to use `/tool` instead"

---

## Recommended Priority Order

| Priority | Doc / Fix | Effort | Impact |
|---|---|---|---|
| **P0** | `docs/reference/cli-commands.md` | Medium | Users can't use Animus without this |
| **P0** | `docs/reference/memory.md` | Medium | Core feature completely undocumented |
| **P1** | Expand `operators/configuration.md` | Medium | Users can't tune behavior |
| **P1** | `docs/operators/local-models.md` | Small | Prevents "Animus is broken" support burden |
| **P1** | Fix stale `gorgon` refs in `packages/forge/README.md` | Small | False advertising if not actually fixed |
| **P2** | `docs/architecture/memory.md` | Medium | Developer onboarding |
| **P2** | `docs/architecture/learning.md` | Medium | Developer onboarding |
| **P2** | `docs/architecture/proactive.md` | Medium | Developer onboarding |
| **P2** | API docs (`docs/reference/api.md`) | Medium | Integration use case |
| **P3** | `docs/architecture/citizen-zero.md` | Large | Identity layer is complex |
| **P3** | Audit orphaned docs | Small | Discoverability |
| **P4** | Performance telemetry + profiling | Medium | Operational visibility; optimization foundation |
| **P5** | Disable agent loop for Ollama users | Small | Fixes #1 user-facing bug |

---

## Senior Engineer Prompt: OODA Loop for P0–P5 Completion

**Role:** Senior Staff Engineer (10+ years). You've been assigned to close all gaps P0–P5 before the next release. You operate with full autonomy. The user (AreteDriver) trusts your judgment on approach, scope, and sequencing.

**Goal:** Every item P0–P5 is documented, tested (where applicable), reviewed, and merged to `main`.

**Method:** Strict OODA loop. No action without observation. No decision without data. No verification without criteria.

---

### Phase 1: Observe (Gather State)

**For each P0–P5 item, before writing a single line:**

1. **Read existing code.** Find the source of truth:
   - CLI commands → `packages/core/animus/__main__.py`, `packages/core/animus/cli.py`
   - Memory API → `packages/core/animus/memory.py`, `packages/kernel/src/animus_kernel/memory/`
   - Config schema → `packages/core/animus/config.py`
   - Performance paths → `packages/core/animus/__main__.py` (REPL loop), `packages/core/animus/cognitive.py` (think_with_tools)
   - Agent loop → `packages/core/animus/__main__.py` line 2465+, `packages/core/animus/cognitive.py` `_think_with_tools_constrained()`

2. **Read existing docs.** Don't duplicate. Link, extend, or replace:
   - `docs/getting-started/quickstart.md`
   - `docs/architecture/overview.md`
   - `docs/operators/configuration.md`
   - Package READMEs

3. **Measure current state.** For performance items (P4):
   - Instrument `__main__.py` with `@contextmanager` timing around tool execution, model generation, and memory operations
   - Run 10 REPL interactions, log latency per phase
   - Report: mean, p95, max for each phase
   - Identify the slowest path

4. **Reproduce the bug.** For P5:
   - Start Animus with Ollama provider
   - Type `watchlist-list` (no `/tool` prefix)
   - Confirm garbled output and max iterations warning
   - Document exact failure mode

5. **Identify stale references.** Run:
   ```bash
   grep -rn "gorgon" docs/ packages/ --include="*.md" --include="*.py" | grep -v "Gorgon renamed\|archived\|Historical"
   ```

**Output:** A table per item:
| Item | Source of Truth | Existing Docs | Gap Size | Estimated Effort | Blockers |
|---|---|---|---|---|---|

---

### Phase 2: Decide (Prioritize and Plan)

**Decision criteria (ranked):**

1. **User impact** — Will a new user be blocked without this?
2. **Maintenance burden** — Will this doc reduce support questions?
3. **Accuracy risk** — Is the current state misleading or harmful?
4. **Effort / value ratio** — Can we ship a "good enough" version in <2 hours?

**Decision framework:**

- **If gap is small and source is clear** → Write doc directly, one-shot, no draft
- **If gap requires code changes** (P4 instrumentation, P5 agent loop disable) → Create a sub-task, implement first, document second
- **If gap spans multiple systems** (e.g., memory docs) → Write overview doc first, link to subsystem docs later
- **If item is stale reference** → Fix or delete. Never leave stale docs in place.

**Sequencing rule:** Complete all P0 before starting P1. Complete all P1 before P2. No skipping. The user learns the system in order — critical gaps first.

**Special handling for P5:** This is a code fix, not a doc. Implement before P4 because P4 measurement will be affected by whether the agent loop is active.

**Output:** A plan file at `.claude/plans/doc-gaps-P0-P5.md` with:
- Item order (strict P0 → P5)
- Per-item acceptance criteria (what "done" means)
- Dependencies between items (P5 before P4 instrumentation)
- Estimated time per item
- Risk: what could make this take 2× longer

---

### Phase 3: Act (Implement)

**Per-item workflow:**

1. **Create branch:** `docs/p0-cli-reference` (or similar, one branch per item or per priority level)
2. **Write doc:** Follow existing style from `docs/reference/tools.md`:
   - Parameter tables for structured data
   - Example blocks for commands
   - Honest notes about limitations
   - Links to related docs (no orphans)
3. **Test doc accuracy:**
   - For CLI commands: run each command in Animus REPL, verify output matches doc
   - For config: read source schema, verify default values
   - For architecture: cross-check with code comments
4. **Run docs validation:**
   ```bash
   python scripts/docs-validate.py
   ```
   - Fix broken internal links
   - Fix trailing whitespace
5. **Commit:** Conventional commit, `docs:` prefix
   ```
   docs: add CLI commands reference

   Documents all 40+ REPL commands with parameters and examples.
   Verified against packages/core/animus/__main__.py.

   Co-Authored-By: Claude <noreply@anthropic.com>
   ```

**P5 (Agent loop fix) specifics:**
- Modify `__main__.py` to detect Ollama provider before entering agent loop
- Preserve agent loop for Anthropic/OpenAI (they have native tool_use)
- Add user-facing hint: "Tip: Use /tool <name> for tool execution with local models"
- Test: Start Animus with Ollama, verify natural language input gets basic response without garbled tool calls
- Test: Start Animus with Anthropic key, verify agent loop still works

**P4 (Performance telemetry) specifics:**
- Add lightweight `animus.profiler` module with context managers (zero external dependencies)
- Log structured JSON to rotating file: `~/.animus/logs/performance.log`
- Fields: `timestamp`, `phase`, `duration_ms`, `tool_name`, `model_provider`, `success`, `context_tokens`, `response_tokens`
- Rotation: 7 days, max 10MB
- Document the schema in `docs/operators/performance.md`
- Add `/stats --perf` command to REPL for real-time summary
- Test: Run 10 interactions, verify log file contains all phases

**Output:** PR per priority level (or per item if large). Clean git history. No WIP commits in PR.

---

### Phase 4: Verify (Review and Close)

**Self-review checklist (before requesting user review):**

- [ ] Doc is accurate against current code (verified by running commands / reading source)
- [ ] No stale references (run grep for known stale terms)
- [ ] All internal links resolve (run `scripts/docs-validate.py`)
- [ ] No trailing whitespace in markdown files
- [ ] Commit message explains what was learned/decided, not just what changed
- [ ] MkDocs nav updated (if applicable)
- [ ] README updated with link (if applicable)
- [ ] Performance instrumentation produces valid JSON and doesn't crash on edge cases
- [ ] P5 fix tested with both Ollama and Anthropic providers
- [ ] P5 fix doesn't break Anthropic agent loop

**User review prompt:**
> "PR #XXX ready for review. This covers P0 items X, Y, Z. I've verified each command by running it in the REPL. Please spot-check 2-3 commands and confirm the examples work on your machine."

**Post-merge verification:**
- Deploy docs site (`mkdocs gh-deploy --force`)
- Verify new pages render correctly on GitHub Pages
- Check mobile rendering (narrow viewport)

**Output:** Closed PRs. Updated `docs/reference/tools.md` changelog if one exists. Session notes in `notes/sessions/YYYY-MM-DD.md`.

---

### Abort Conditions

Stop the loop and escalate to user if:
- Source code contradicts existing docs in a way that suggests a bug, not just stale docs
- Performance instrumentation reveals a critical performance regression (>2× slower than expected)
- Any doc requires secrets, API keys, or credentials to verify
- Estimated effort for any single item exceeds 4 hours (scope creep — split into follow-up)
- P5 fix breaks Anthropic agent loop (rollback immediately)

---

### Success Criteria

All P0–P5 items are "done" when:
1. ✅ Doc exists at the specified path
2. ✅ Doc is accurate against current code (verified)
3. ✅ Doc passes `scripts/docs-validate.py` (links, whitespace)
4. ✅ Doc is in mkdocs nav (if applicable)
5. ✅ Code changes (if any) have tests or are manually verified
6. ✅ PR is merged to `main`
7. ✅ User has spot-checked and confirmed usefulness

**Report format on completion:**

```
Documentation Gap Closure — Final Report
======================================
P0: X/X complete | P1: X/X complete | P2: X/X complete | P3: X/X complete | P4: X/X complete | P5: X/X complete
Total time: N hours
Unexpected findings: [list]
Follow-up items: [list]
```

---

*This prompt is designed to be self-contained. Paste it into a new Claude Code session, and the agent will execute the OODA loop autonomously until all items are closed or an abort condition is met.*

**Role:** Senior Staff Engineer (10+ years). You've been assigned to close all documentation gaps P0–P4 before the next release. You operate with full autonomy. The user (AreteDriver) trusts your judgment on approach, scope, and sequencing.

**Goal:** Every item P0–P4 is documented, tested (where applicable), reviewed, and merged to `main`.

**Method:** Strict OODA loop. No action without observation. No decision without data. No verification without criteria.

---

### Phase 1: Observe (Gather State)

**For each P0–P4 item, before writing a single line:**

1. **Read existing code.** Find the source of truth:
   - CLI commands → `packages/core/animus/__main__.py`, `packages/core/animus/cli.py`
   - Memory API → `packages/core/animus/memory.py`, `packages/kernel/src/animus_kernel/memory/`
   - Config schema → `packages/core/animus/config.py`
   - Performance paths → `packages/core/animus/__main__.py` (REPL loop), `packages/core/animus/cognitive.py` (think_with_tools)

2. **Read existing docs.** Don't duplicate. Link, extend, or replace:
   - `docs/getting-started/quickstart.md`
   - `docs/architecture/overview.md`
   - `docs/operators/configuration.md`
   - Package READMEs

3. **Measure current state.** For performance items:
   - Instrument `__main__.py` with `@contextmanager` timing around tool execution, model generation, and memory operations
   - Run 10 REPL interactions, log latency per phase
   - Report: mean, p95, max for each phase
   - Identify the slowest path

4. **Identify stale references.** Run:
   ```bash
   grep -rn "gorgon" docs/ packages/ --include="*.md" --include="*.py" | grep -v "Gorgon renamed\|archived\|Historical"
   ```

**Output:** A table per item:
| Item | Source of Truth | Existing Docs | Gap Size | Estimated Effort | Blockers |
|---|---|---|---|---|---|

---

### Phase 2: Decide (Prioritize and Plan)

**Decision criteria (ranked):**

1. **User impact** — Will a new user be blocked without this?
2. **Maintenance burden** — Will this doc reduce support questions?
3. **Accuracy risk** — Is the current state misleading or harmful?
4. **Effort / value ratio** — Can we ship a "good enough" version in <2 hours?

**Decision framework:**

- **If gap is small and source is clear** → Write doc directly, one-shot, no draft
- **If gap requires code changes** (e.g., performance instrumentation) → Create a sub-task, implement first, document second
- **If gap spans multiple systems** (e.g., memory docs) → Write overview doc first, link to subsystem docs later
- **If item is stale reference** → Fix or delete. Never leave stale docs in place.

**Sequencing rule:** Complete all P0 before starting P1. Complete all P1 before P2. No skipping. The user learns the system in order — critical gaps first.

**Output:** A plan file at `.claude/plans/doc-gaps-P0-P4.md` with:
- Item order (strict P0 → P4)
- Per-item acceptance criteria (what "done" means)
- Dependencies between items
- Estimated time per item
- Risk: what could make this take 2× longer

---

### Phase 3: Act (Implement)

**Per-item workflow:**

1. **Create branch:** `docs/p0-cli-reference` (or similar, one branch per item or per priority level)
2. **Write doc:** Follow existing style from `docs/reference/tools.md`:
   - Parameter tables for structured data
   - Example blocks for commands
   - Honest notes about limitations
   - Links to related docs (no orphans)
3. **Test doc accuracy:**
   - For CLI commands: run each command in Animus REPL, verify output matches doc
   - For config: read source schema, verify default values
   - For architecture: cross-check with code comments
4. **Run docs validation:**
   ```bash
   python scripts/docs-validate.py
   ```
   - Fix broken internal links
   - Fix trailing whitespace
5. **Commit:** Conventional commit, `docs:` prefix
   ```
   docs: add CLI commands reference

   Documents all 40+ REPL commands with parameters and examples.
   Verified against packages/core/animus/__main__.py.

   Co-Authored-By: Claude <noreply@anthropic.com>
   ```

**Performance instrumentation specifics:**
- Add lightweight timing to `__main__.py` using `time.perf_counter()` (no external deps)
- Log structured JSON to `~/.animus/logs/performance.log`
- Fields: `timestamp`, `phase` (tool_parse/model_generate/tool_execute/memory_recall), `duration_ms`, `tool_name`, `model_provider`, `success`
- Rotation: 7 days, max 10MB
- Document the schema in `docs/operators/performance.md`

**Output:** PR per priority level (or per item if large). Clean git history. No WIP commits in PR.

---

### Phase 4: Verify (Review and Close)

**Self-review checklist (before requesting user review):**

- [ ] Doc is accurate against current code (verified by running commands / reading source)
- [ ] No stale references (run grep for known stale terms)
- [ ] All internal links resolve (run `scripts/docs-validate.py`)
- [ ] No trailing whitespace in markdown files
- [ ] Commit message explains what was learned/decided, not just what changed
- [ ] MkDocs nav updated (if applicable)
- [ ] README updated with link (if applicable)
- [ ] Performance instrumentation produces valid JSON and doesn't crash on edge cases

**User review prompt:**
> "PR #XXX ready for review. This covers P0 items X, Y, Z. I've verified each command by running it in the REPL. Please spot-check 2-3 commands and confirm the examples work on your machine."

**Post-merge verification:**
- Deploy docs site (`mkdocs gh-deploy --force`)
- Verify new pages render correctly on GitHub Pages
- Check mobile rendering (narrow viewport)

**Output:** Closed PRs. Updated `docs/reference/tools.md` changelog if one exists. Session notes in `notes/sessions/YYYY-MM-DD.md`.

---

### Abort Conditions

Stop the loop and escalate to user if:
- Source code contradicts existing docs in a way that suggests a bug, not just stale docs
- Performance instrumentation reveals a critical performance regression (>2× slower than expected)
- Any doc requires secrets, API keys, or credentials to verify
- Estimated effort for any single item exceeds 4 hours (scope creep — split into follow-up)

---

### Success Criteria

All P0–P4 items are "done" when:
1. ✅ Doc exists at the specified path
2. ✅ Doc is accurate against current code (verified)
3. ✅ Doc passes `scripts/docs-validate.py` (links, whitespace)
4. ✅ Doc is in mkdocs nav (if applicable)
5. ✅ Code changes (if any) have tests or are manually verified
6. ✅ PR is merged to `main`
7. ✅ User has spot-checked and confirmed usefulness

**Report format on completion:**

```
Documentation Gap Closure — Final Report
======================================
P0: X/X complete | P1: X/X complete | P2: X/X complete | P3: X/X complete | P4: X/X complete
Total time: N hours
Unexpected findings: [list]
Follow-up items: [list]
```

---

*This prompt is designed to be self-contained. Paste it into a new Claude Code session, and the agent will execute the OODA loop autonomously until all items are closed or an abort condition is met.*