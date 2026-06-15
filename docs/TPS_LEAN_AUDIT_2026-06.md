# Animus — Lean Manufacturing Audit

**Method:** Toyota Production System (TPS) applied to software · Date: 2026-06-14  
**Auditor:** ARETE · Scope: 564 commits across 18 months · Outcome: Cut, Repurpose, Accelerate  
**Doctrine:** *"The floor is still the test."* — Principle: value is proven only by usage, not by architecture diagrams.

---

## Executive Summary

| Waste Type | Severity | Evidence | Count |
|---|---|---|---|
| **Overproduction** | 🔴 Critical | Features shipped that have never been used by the sole operator | 8 |
| **Inventory** | 🔴 Critical | Code written, tested, documented — and sitting idle | 6 |
| **Over-processing** | 🟡 High | 10/10 security audit on a personal tool; fortress built for a one-person house | 1 |
| **Defects** | 🟡 High | Fixes constitute 43/564 commits (7.6%); 56 security-related PRs chasing a moving standard | Ongoing |
| **Transport** | 🟡 Medium | Context fragmented across 7 interfaces; operator's memory, not Animus's, bridges the gaps | Daily |
| **Waiting** | 🟢 Low | Local inference waits; budget gates pause cloud calls; these are controlled waits, not waste | — |
| **Motion** | 🟢 Low | Most CLI commands are efficient; dashboard has too many clicks, but operator rarely uses it | — |

**Bottom line:** ~40% of the codebase is actively used. ~35% is speculative infrastructure (built for a future that may never come). ~25% is genuine waste (built, tested, then abandoned).

---

## The 7 Wastes — Evidence from the Gemba

### 1. Overproduction (Muda of Excess)

> *Building features before demand exists — or building for hypothetical users.*

| Feature | Commits | Proof of Non-Use | Assessment |
|---|---|---|---|
| **Discord bot** (`7b4f19c`, `3d1f788`) | 2 major commits, gateway adapter code | No Discord server for Animus exists. Bot token never generated. | 🔴 **Pure waste.** Built because "channels" were in the architecture. |
| **Competition watchlist + harvest** (`dd86085`, `635d40f`) | 3 commits, cron scheduling | Watchlist is empty. `animus_harvest` tool was never invoked. | 🔴 **Speculative.** Competitor monitoring is a product-company need, not a solo-operator need. |
| **WhatsApp/Signal channel adapters** | Stub implementations in `gateway/channels/` | No phone numbers, no API keys, no integration. | 🟡 **Placeholder inventory.** Stubs cost little but clutter the mental model. |
| **Windows service** (`e9f4d2f`) | Full `WindowsService` class with installer | Operator is Linux-only. No Windows machine in the portfolio. | 🔴 **Complete waste.** Built for "completeness," not need. |
| **LlamaCppProvider** (`79e1037`) | 1 commit, full provider | Never used. Ollama provider handles all local inference. | 🟡 **Duplication.** Ollama is the local path; this is a parallel track that serves the same need with more friction. |
| **Red-team suite** (`5ccf580`, `b66b5b3`) | 2 commits, test harness | Valuable in principle, but operator has never run it. The adversarial review was driven by the 10/10 audit, not an observed threat. | 🟡 **Over-preparation.** Security-by-threat-model is enterprise discipline; security-by-usage-context is solo discipline. |
| **Skill evolver + benchmark CI** (`62a6d2e`) | 1 commit, full module | `skills/` directory has evolver code. Benchmark results exist. Skill evolution never triggered on a real workflow. | 🟡 **Pipeline without intake.** The machine exists but no material feeds it. |
| **OGMA synthesis** (`bdb93f9`, `9da89d8`) | 2 commits | `ogma/` package. Never invoked in a real session. | 🟡 **Research artifact.** Valuable as a concept, not as shipped code. Belongs in a research note, not the repo. |

**Repurpose rule:** Before any feature ships, the acceptance criterion must include: *"I will use this within 7 days of merge, or it is cut."*

---

### 2. Inventory (Muda of Stock)

> *Code that is complete but sits unused — depreciation in real-time.*

| Inventory Item | Condition | Depreciation Rate | Action |
|---|---|---|---|
| **Reflection loop** (`2d7ff51`, `40fc1e7`) | Complete, wired, never executed. `LEARNED.md`: "Last reflection: never" | Increases as LEARNED.md gets stale | 🔴 **Cut the reflection framework.** Replace with a manual weekly review of `animus_recall` output. |
| **Self-improve sandbox** (`e9ad264`, `5d2970b`) | Complete, 52 references in code. No autonomous PR created by it. | High — code rots without exercise | 🟡 **Freeze.** Do not expand. Only reactivate if operator records 10 feedback entries/week sustained. |
| **Persona engine** (`23799a5`, `3d912de`) | 6 voice presets, 9 knowledge domains, channel routing | Increasing — personas drift uncalibrated | 🟡 **Reassess after 30 days of PWA daily use.** If not actively switched between, reduce to 1 default persona. |
| **Proactive checks** (`bd147a0`, `2e12944`) | 6 checks, SQLite persistence, outcome capture | Decaying — checks run but suggestions are dismissed/not seen | 🟡 **Cut to 3 checks.** Keep only: (1) cost anomaly, (2) memory stale, (3) daily task reminder. Delete the rest. |
| **Quorum v2 EventLog** (`6b259f6`) | Bitemporal event log, signal bridge | Moderate — not consumed by any downstream system | 🟡 **Repurpose.** Turn into a simple audit log for the operator's own quarterly review, not a distributed-coordination primitive. |
| **Dashboard 20 pages** | All implemented, most never opened | Stable — but maintenance burden on dependency upgrades | 🟡 **Cut to 5 pages.** Keep: (1) chat, (2) memory browser, (3) status, (4) config, (5) tasks. Merge or delete the rest. |
| **PWA 4 views** | Chat, Capture, Status, Personas | Increasing — missing too many features to be daily driver | 🟡 **Keep, but expand selectively.** Priority: memory browser, then tasks. Cut personas view if persona engine is cut. |

**Lean principle:** Inventory hides problems. If the reflection loop had been used once, the operator would have learned it produces low-signal output and iterated — or killed it. By letting it sit unused, we hid its uselessness behind a green test badge.

---

### 3. Over-Processing (Muda of Over-engineering)

> *Doing more than is required to solve the problem.*

**The 10/10 security audit** is the poster child.

| Phase | Effort (commits) | Real Threat Model | Return |
|---|---|---|---|
| A1–A8 (cost + security substrate) | ~15 commits, 3 sessions | Real — cost overruns and credential leaks happen | ✅ **High.** Protects wallet and keys. |
| B1–B7 (eval integrity) | ~20 commits, 3 sessions | Theoretical — eval framework exists but produces no actionable insights | 🟡 **Medium.** Harder to trust evals that are never used. |
| C0–C1 (security enforcement end-to-end) | ~25 commits, 2 PRs | Real-ish — prevents egress of secrets, but operator is sole user with no external untrusted input | 🟡 **Diminishing returns.** After A4 (content-aware egress), the marginal utility of C1-14 approaches zero. |
| D1i–D5i (tiered memory, active inference, replay) | 0 commits — still in roadmap | Aspirational | N/A |
| E9–E14 (systemd integrity, red-team loop, supply-chain) | ~10 commits | Extremely theoretical for a solo operator | 🔴 **Over-processing.** These are enterprise-grade controls. The threat model is "what if someone roots my laptop and tampers with the unit file?" The answer is: if they root my laptop, Animus is already compromised. |

**The jidoka moment:** The whitepaper audit found real gaps. The first 40% of fixes (A1–A4, A6, A7) were necessary. The next 40% (B1–B7, C0) were defensible but not urgent. The final 20% (E9–E14) were the software equivalent of machining parts to 0.001mm tolerance for a garden shed.

**Repurpose rule:** For any security/control feature, the stopping criterion is: *"Does the absence of this feature materially increase the probability of operator harm within the next 6 months?"* If the answer is "no," it doesn't ship.

---

### 4. Defects (Muda of Rework)

**Source of truth:**

```
564 total commits since 2025-01
  48 feat:  (features)
  43 fix:   (bug fixes)
  62 docs:  (documentation)
  22 test:  (test additions)
   2 refactor:
  ~387 misc/chore/merge/etc

7.6% of commits are fixes.
56 commits reference security/CodeQL corrections.
```

| Defect Category | Root Cause | Prevention (Poka-yoke) |
|---|---|---|
| **CodeQL alerts** (24 → 12 → 6 → 0 progression) | Dead code paths, empty `except`, unused imports | Pre-commit hook with `ruff check` + `mypy --strict`. Already partially in place. Needs zero-tolerance gate. |
| **Migration collisions** (`012` prefix collision) | Manual numbering | `E15` guard — test globs migrations, asserts uniqueness. Now in roadmap. |
| **API key leakage** (historical) | Credentials in config files | `systemd-creds` + `gocryptfs` already deployed. No plaintext keys in `~/.animus/`. Poka-yoke: CI rejects PRs with `sk-ant` patterns. |
| **Fix cascades** (egress unified → provider wiring missed → follow-up fix) | Feature crosses too many files | Single-owner principle: egress logic should live in `animus_types` only. Provider modules should import, not re-implement. (Fixed in C5.) |

**The lean read:** 7.6% defect rate is not terrible, but 56 security-related PRs in a solo project suggests the process is catching symptoms, not preventing disease. Poka-yoke beats QC.

---

### 5. Transport (Muda of Movement)

> *Moving information between tools, sessions, and surfaces.*

**The operator's daily context transport map:**

```
Claude Code session ←→ MCP tools ←→ Animus memory (great — 1 hop)
Claude Code session ←→ GitHub issues (manual — operator types URLs)
Claude Code session ←→ Todoist (integration exists but rarely used)
Claude Code session ←→ Google Calendar (OAuth works but operator doesn't check it)
Claude Code session ←→ EVE tools (separate projects, no bridge)
Animus dashboard ←→ PWA ←→ CLI (three islands; state doesn't transfer)
Portfolio updates ←→ Discord/Slack (operator copies/pastes manually)
```

**The bottleneck:** The only seamless transport is MCP server ↔ memory. Everything else requires the operator to *carry* context. This is the opposite of an exocortex.

**Lean prescription:** Don't build more transport mechanisms. Make the *existing* ones disappear.

| Target State | Current State | Gap |
|---|---|---|
| Animus suggests next task from GitHub issues + memory | Animus has GitHub integration but operator manually queries | Bridge: proactive engine reads open issues, suggests highest-value next task |
| Animus knows what EVE projects are active | EVE tools are separate repos, no bridge | Bridge: `animus_recall` should index `gatekeeper/`, `monolith/` etc. as part of semantic memory |
| Animus calendar suggestions are surfaced *in* Claude Code session | Calendar integration exists; operator forgets to check it | Bridge: MCP `animus_brief` should include upcoming calendar items when `working_hours=true` |

---

### 6. Waiting (Muda of Delay)

**Controlled waits (not waste):**

| Wait | Cause | Mitigation | Assessment |
|---|---|---|---|
| Local inference | RTX class GPU needed, currently CPU fallback | GPU box build (in progress) | ✅ **Investment wait.** Will resolve with hardware. |
| Budget gate | ET ceiling check before cloud call | Instant (lock acquisition) | ✅ **Necessary.** Sub-millisecond. |
| Test suite | 14,596 tests take 3–5 min | Parallel CI, selective test runs | 🟡 **Could be leaner.** Run only affected package tests on pre-push. Full suite nightly. |
| Reflection loop | Never triggered | N/A | N/A — not a wait, an abandonment |

**Uncontrolled waits (waste):**

| Wait | Cause | Fix |
|---|---|---|
| Switching contexts between 30 projects | No unified briefing | MCP `animus_brief` should auto-fire on `cd ~/projects/X` |
| Re-reading previous session | No session summary auto-generated | `animus_remember` should tag sessions with one-line summaries |
| Finding the right tool | 37 tools, no search | Tool registry needs a "most used" ranking surfaced in MCP |

---

### 7. Motion (Muda of Unnecessary Steps)

> *Extra actions to accomplish a task.*

| Task | Current Steps | Ideal Steps | Waste |
|---|---|---|---|
| Capture a thought | Open PWA → wait for load → type → save | Global hotkey → speak/type → auto-save | 3 steps + load time |
| Check task status | Open dashboard → click Tasks → wait for render | Glance at tray widget or MCP brief | 3 steps |
| Approve a tool | Terminal prompt: "Execute? [Y/n]" | Pre-approved safe tools + explicit block for risky ones | 1 keystroke + context switch |
| Recall a memory | `/memory` in CLI → type query → read results | Auto-recall enriched into every Claude Code prompt | 0 steps (already happens via MCP!) |
| Run a workflow | `cd packages/forge && animus-forge run workflow.yml` | Claude Code `/run workflow-name` via MCP | 1 command instead of directory + path |
| Review yesterday's work | Search git logs, open GitHub, check Todoist | "What did I do yesterday?" → Animus synthesizes from audit log | Multiple manual searches |

**The lean read:** The MCP server is the only interface that *eliminates* motion. Everything else adds it. The strategic priority is clear: **make the MCP layer omnivorous** — consume more operator context automatically — and **deprecate interfaces that require opening.**

---

## Kaizen: What to Do Now

### 🔴 CUT (Immediate — removes waste, frees capacity)

| # | Item | Why | How |
|---|---|---|---|
| C1 | **Discord bot + all gateway stubs** | Never used, never will be for a solo operator | Delete `gateway/channels/discord.py`, `telegram.py`, `slack.py` stubs. Keep WebSocket + Email only (if used). |
| C2 | **Windows service** | No Windows machine | Delete `daemon/platforms/windows.py`. Keep Linux + macOS only. |
| C3 | **Competition watchlist** | Solo operators don't compete; they build | Delete `watchlist.py`, `harvest.py`. Move conceptual notes to `notes/topics/competitive-intel.md` if useful. |
| C4 | **Reflection loop framework** | Never executed; manual weekly review is better | Delete `reflection.py`, `feedback.py` CLI commands. Keep `LEARNED.md` as a manual journal. |
| C5 | **Self-improve sandbox** | No production PRs; framework without fruit | Freeze. Comment out `auto_promote_on_improvement`. If 30 days pass with no manual trigger, delete. |
| C6 | **Dashboard pages 6–20** | Admin pages for a non-admin | Delete: `/automations`, `/activity`, `/routing`, `/self-mod`, `/forge`, `/timers`, `/feedback`, `/proposals`. Merge `/logs` into `/status`. |
| C7 | **Skill evolver + benchmark CI** | Evolver exists with no skills to evolve | Delete `evolver/`. Benchmarks: move to manual quarterly run, not CI. |
| C8 | **OGMA synthesis package** | Research artifact, not product code | Extract to `research/ogma-spec.md`. Delete `packages/ogma/`. |

**Estimated code reduction: ~15,000 lines (~30% of repo).**  
**Test reduction: ~2,000 tests (still leaving 12,000+).**  
**Maintenance burden: Significantly reduced.**

---

### 🟡 REPURPOSE (This quarter — redirect sunk cost into value)

| # | Item | From | To |
|---|---|---|---|
| R1 | **Proactive engine** | 6 checks, noisy, ignored | 3 checks only: cost anomaly, memory staleness, daily task digest. Output goes to MCP `animus_brief`, not notifications. |
| R2 | **Persona engine** | 6 presets, 9 domains, over-engineered | 1 persona: "ARETE assistant." Tone: direct, technical, terse. Voice: none (text-only). Delete persona dashboard page. |
| R3 | **Quorum EventLog** | Distributed-coordination primitive | Personal audit trail: auto-append every tool call + decision. Monthly review for pattern detection. |
| R4 | **LlamaCppProvider** | Parallel local inference path | Consolidate into OllamaProvider. Delete `llama.cpp` path entirely. Ollama handles all local inference. |
| R5 | **Dashboard (remaining 5 pages)** | Generic admin panel | Single-page: Command Center (status + chat + quick capture + task list). HTMX is fine. No SPA needed. |
| R6 | **PWA** | 4 thin views | **One purpose: mobile quick-capture + status glance.** Expand memory browser later. Cut personas/status views if redundant. |
| R7 | **Red-team suite** | Enterprise adversarial testing | **Quarterly ritual:** run once per quarter, file findings to `SECURITY.md`, fix only HIGH+. Don't maintain continuous suite. |

---

### 🟢 DOUBLE DOWN (Accelerate — these produce value)

| # | Item | Why | Investment |
|---|---|---|---|
| D1 | **MCP server** | Only interface that eliminates motion; used in every session | Expand to 15 tools: `animus_remember_context`, `animus_daily_brief`, `animus_task_next`, `animus_project_switch`, `animus_decision_log` |
| D2 | **BudgetManager + ET enforcement** | Saves real money; prevents runaway spend | Keep hardening. Add: monthly Pareto report (spend by project, by model). Surface in MCP brief. |
| D3 | **Memory tiering** | HOT/WARM/COLD works; access tracking is real | Finish incremental BM25 + LLM consolidation. This is the moat. |
| D4 | **Forge workflow executor** | Checkpoint/resume, budget gates, proven in CI-like use | Build 5 "portfolio hygiene" workflows and schedule them weekly via cron. Make them *the* automation layer. |
| D5 | **Ollama routing + GPU box** | Marginal cost → $0; sovereignty → absolute | Complete CUDA box build. Benchmark. Flip `ANIMUS_OFFLINE=1` to default. |
| D6 | **Quick-capture** | Highest leverage UI: lowest friction input | System-tray hotkey (Linux: `xdotool` or `eww` widget). Voice or text. One shot → memory. |
| D7 | **Integrity + encryption at rest** | Already deployed, working, tested | Maintain. The `gocryptfs` + `systemd-creds` stack is already correct. Don't add more. |

---

## Future-State Value Stream

### Current State (with waste)

```
[Need arises] → [Choose interface: CLI/dashboard/PWA/MCP] → [Navigate to right page/command]
              → [Query/remember manually] → [Wait for inference (local or cloud)]
              → [Review output] → [Decide to act] → [Switch to project/tool]
              → [Act manually] → [Forget to record outcome] → [Repeat next session]
```

**Lead time:** Variable, 5 min–30 min per interaction.  
**Touchpoints:** 4–7.  
**Automation:** Low. Operator carries context.

### Future State (lean)

```
[Need arises] → [MCP auto-brief already enriched context into Claude Code prompt]
              → [Claude Code acts with tools, auto-approved for safe ops]
              → [Outcome auto-remembered with project tag]
              → [Next session begins where last ended, no manual recall]
```

**Lead time:** Near-zero.  
**Touchpoints:** 1 (Claude Code).  
**Automation:** High. Animus carries context.

### The Constraint

The bottleneck is **not** the model or the memory or the UI.  
The bottleneck is **transport** — moving context between the operator's brain and the tools.

**Elevate the constraint:** Make MCP consume everything. Eliminate all other interfaces as primary surfaces. The dashboard becomes a read-only status page. The PWA becomes a capture-only satellite. The CLI becomes an emergency fallback.

---

## Measuring Kaizen

How do we know the cuts were correct?

| Metric | Now | Target (90 days) | How to Measure |
|---|---|---|---|
| Lines of code | ~45,000 | ~32,000 | `cloc` across `packages/*/src/` |
| Test count | 14,596 | ~12,500 | `pytest --collect-only` |
| CI time | ~8 min | ~5 min | GitHub Actions logs |
| Daily MCP tool calls | ~15 | ~40 | MCP audit log |
| Dashboard opens per week | ~2 | 0 | (self-reported) |
| PWA opens per week | ~5 | 3 (capture only) | (self-reported) |
| Cost per month (API) | $50–150 | $20–50 | `cost_audit.py` output |
| Proactive suggestions acted on | 0 | 3/week | Proactive check outcome log |
| Memory recalls per session | ~5 | ~8 | MCP audit log |
| Time to switch projects | 2 min | 10 sec | `/project switch X` via MCP |

---

## Standing Rule: The 7-Day Shikumi

*(Shikumi = mechanism/system)*

For any new feature, the acceptance test is not "it passes CI." It is:

> **"Within 7 days of shipping, the operator uses it in a real session, and the interaction is faster, cheaper, or produces a better outcome than the previous method. If not, it is reverted or cut."**

This is the software equivalent of *jidoka* — stop the line and fix the process when quality drops.

No exceptions. Not for security theater. Not for architectural elegance. Not for "it might be useful someday."

The floor is still the test.

---

## Related

- `docs/INTERFACE_BOOTSTRAP_VISION.md` — UX vision (now subject to this audit)
- `docs/CUDA_AI_BOX_SETUP.md` — GPU box build (D5, the hardware enabler)
- `docs/ROADMAP_TO_10.md` — Technical roadmap (now subject to E9–E14 cuts)
- `docs/PERSONAL_ROADMAP.md` — Solo-operator doctrine (aligns perfectly with this audit)
- `docs/CONSTITUTIONAL_PRINCIPLES.md` — P1-P9 (P4 and P7 reinforce the lean posture)

---

*Canonical. Review monthly. Update when cuts are executed. File follow-up ADL for any feature that violates the 7-day shikumi.*
