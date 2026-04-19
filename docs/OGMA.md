# Ogma — Reverse-Engineering Synthesis Persona

Ogma reads external work (papers, podcasts, HN, blogs) **and our own projects** and produces implementation-ready synthesis. Companion to [Lugh](../packages/core/animus/lugh/): where Lugh harvests, Ogma deciphers.

**Named for Ogma** — Celtic god of eloquence who invented Ogham script, the first written Gaelic. Patron of translation and learning. The fit: Ogma turns outside text (and our own code) into Animus-native understanding and rebuild proposals.

---

## Purpose

Four failure modes Ogma exists to prevent:

1. **Harvest fatigue** — Lugh pulls hundreds of items weekly. Without a reader, they pile up unread. Ogma is the reader.
2. **Generic summary-bot output** — public summary services (even LLM-powered ones) produce "this paper proposes a novel approach to…" slop. That's not useful. Ogma grounds every claim against the Animus codebase, so output is actionable, not encyclopedic.
3. **Copy-cat integration** — naive "implement what the paper said" produces dependency bloat and broken composition. Ogma's mandate is **build it better**: understand the mechanism, then design the Animus-native version — sharper contract, better composition, smaller footprint, reproducibility test baked in.
4. **Silent internal drift** — the portfolio accumulates 30+ projects, each with debt the author has stopped seeing. Tactical tools (`/review`, `/refactor`, `/simplify`) work at line level. Ogma works at subsystem level: "how does this module actually work, what's weak, what would we build today if we started fresh?" Same rigor applied inward.

---

## Ethos — Build It Better

From the design conversation (2026-04-18):

> "We want to be able to build the best possible tools, absorb what we can and keep moving forward. Not copy it exactly, but figure out how it works, then build it better."
>
> "It also needs to just be harvesting and rebuilding for our own projects. Improving anything possible to the highest standards and function."

Operational consequences:

- **Same standard outside and inside.** External sources and our own portfolio get the same treatment: understand the mechanism, critique honestly, propose the version that improves on it. No special pleading for our own code.
- **No refusal to propose** — if Animus has no matching abstraction (`animus_gap: NONE`), Ogma still fires. If an internal subsystem is "good enough," Ogma still names the rebuild that would be excellent. Greenfield proposals and principled rebuilds with explicit ROI are the whole point.
- **Critique honestly.** Every external synthesis has a `weaknesses` field naming what's hand-wavy or unreproducible in the source. Every internal audit has the same — no flattery of our own work.
- **Name the improvement.** External: how our version goes beyond the source (tighter contract, Forge/Quorum composition, reproducibility, smaller deps, fail-open I/O, constitutional alignment). Internal: how the rebuild raises standards (cleaner abstractions, consolidated duplication, retired debt, better test surface).

---

## Output Contracts

Two shapes — external and internal. Both saved to `~/projects/notes/ogma/YYYY-MM-DD-<slug>.md` so synthesis is durable and queryable. Missing fields fail the skill.

### External (for `read` and `brief`)

| Field | Purpose |
|-------|---------|
| `concept` | One-paragraph distillation of what the source actually does. No hedging. |
| `novelty` | What's new vs prior art. If reheated, say so. |
| `animus_gap` | `NONE \| PARTIAL \| FULL` + exact file(s):function(s) in Animus that overlap. Verified by Grep/Read — never guessed. |
| `weaknesses` | What's hand-wavy, unreproducible, missing ablations, or load-bearing on bad assumptions. |
| `proposal` | Concrete — module name, abstraction it composes with, sketch of the change. Explicitly names how our version is better. |
| `roi` | Value (what this unlocks), effort (trivial/moderate/substantial), priority (why now vs later). |
| `risks` | Reproducibility, maturity, licensing, perf, scope creep, coupling. |
| `confidence` | 0.0–1.0 + one-line justification. Below 0.6 means another read or a prototype before acting. |
| `sources_cited` | Source URL + prior art + Animus file:line where gap claim was verified. |

### Internal (for `audit` and `sweep`)

| Field | Purpose |
|-------|---------|
| `concept` | What this subsystem actually does. If muddled, name the muddle. |
| `what_works` | Load-bearing strengths the rebuild must preserve. |
| `weaknesses` | Architectural debt, duplicated logic, stale abstractions, test gaps, coupling violations, perf cliffs. file:line specific. |
| `cross_portfolio_overlap` | Another Arete project solve the same problem differently? Consolidation candidate, or divergence intentional? |
| `rebuild_proposal` | If starting today, what's the version-2? Contracts, abstractions, file layout, test surface. Strategic, not incremental. |
| `incremental_path` | If full rebuild isn't justified, the minimum intervention that captures most of the value. If it's fine as-is, say so. |
| `roi` | Capability / debt / perf / velocity gains. Effort estimate. Priority. |
| `risks` | Breaking consumers, data migration, contract drift, context loss, opportunity cost. |
| `confidence` | 0.0–1.0 + justification. |
| `sources_cited` | File:line references + related Arete projects + external work that informed the proposal. |

Shared fields (`concept`, `weaknesses`, `roi`, `risks`, `confidence`, `sources_cited`) are identical in spirit across modes.

---

## Surfaces

Three delivery modes, staggered by scope.

### v1 — Claude Code skill (`/ogma`)

Manual, in-session. Five verbs:

- `/ogma read <id>` — deep synthesis of one external item (paper / HN / podcast / URL)
- `/ogma brief [--since 7d] [--min-score 0.5]` — batch briefing across lugh cache
- `/ogma gap "<concept>"` — reverse query ("does Animus already do X?")
- `/ogma audit <target>` — internal subsystem audit ("how would we rebuild this to highest standards?")
- `/ogma sweep [--portfolio] [--min-age 60d]` — portfolio-wide audit sweep, ranked by rebuild ROI

**Spec:** `~/.claude/skills/ogma/SKILL.md`
**Persistence:** markdown at `~/projects/notes/ogma/YYYY-MM-DD-<slug>.md`
**Grounding (external):** Grep/Read on Animus repo + WebFetch on source.
**Grounding (internal):** Grep/Read on target project + git log for activity + pytest --collect-only for test surface.

**Status: v1 spec shipping now.**

### v2 — Forge workflow

Cron-triggered batch. YAML workflow in `packages/forge/workflows/ogma-weekly.yaml`. Fires weekly after Lugh harvests, reads top-N scored items, produces one composite briefing, posts to Discord + stores in ChromaDB.

**Depends on:**
- v1 output contract stabilized
- ChromaDB write path from lugh.sources (planned)
- Forge budget manager (existing)
- Forge provider routing (existing)

**Open questions:**
- Which provider tier? Opus for deep reads, Sonnet/Haiku for brief summaries, Ollama for keyword gating?
- How does the workflow handle source-unreachable items? Skip, retry, or escalate to human?

### v3 — Animus MCP tool

Exposes Ogma output as an MCP tool so any client (CC, another agent, the user's personal chatbot) can query accumulated synthesis:

- `ogma_query "<concept>"` — returns prior Ogma reads matching concept, ranked by confidence × recency
- `ogma_gap "<concept>"` — runs v1 gap flow
- `ogma_read <id>` — runs v1 read flow

Served from the existing Animus MCP server (`packages/core/animus/mcp_server.py`), following the same registration pattern as Lugh's `HARVEST_TOOL` + `WATCHLIST_*_TOOL`s.

---

## Architecture — How It Stays Grounded

```
┌─────────────┐     ┌────────────┐     ┌──────────────────────────┐
│ Lugh cache  │────▶│ /ogma read │────▶│ Per-item synthesis .md   │
│ (SQLite)    │     │            │     │ notes/ogma/2026-*.md     │
└─────────────┘     └────┬───────┘     └──────────────────────────┘
                         │
                         ▼
                    ┌──────────────┐
                    │ Animus repo  │ ◀── Grep / Read for
                    │ grounding    │     animus_gap verification
                    └──────────────┘
```

Key constraint: **no Animus file path appears in output unless it was actually read during synthesis.** This is the rule that separates Ogma from a summary bot.

The markdown archive at `~/projects/notes/ogma/` is Ogma's long-term memory in v1. In v2 it gets indexed into ChromaDB via `tools/animus_sync.py` so Ogma can self-query prior reads before re-synthesizing ("have I already read this concept?").

---

## Relationship to Animus Subsystems

| Subsystem | Interaction |
|-----------|-------------|
| **Lugh** (`packages/core/animus/lugh/`) | Upstream — provides `SourceItem`s + relevance scores Ogma reads from |
| **Forge** (`packages/forge/`) | v2 — runs `/ogma brief` as a workflow with budget-managed LLM calls |
| **Quorum** (`packages/quorum/`) | Out of scope for v1. v3 could route high-stakes proposals (>0.7 confidence, substantial effort) through triumvirate voting before acting |
| **Core memory** (`packages/core/animus/memory.py`) | v2 — Ogma writes synthesis as episodic memories; v3 queries them via MCP |
| **Constitutional Principles** (P1–P9) | Proposals evaluated against principles; violations → escalate or reject |

---

## Non-Goals

- **Not a code generator.** Proposals are specs, not patches. Forge is for code changes; Ogma is for deciding what's worth changing.
- **Not a peer reviewer.** Synthesis is opinionated by design — Ogma is not neutral. It serves the Animus project's goals.
- **Not a literature search tool.** Lugh harvests breadth; Ogma goes deep on what Lugh surfaced.
- **Not a hype-trend tracker.** Relevance scoring lives in Lugh; Ogma processes what scores well, not what trends.
- **Not a line-level refactor helper.** Use `/refactor` or `/simplify` for code-level changes. Ogma operates at subsystem / architectural scope.
- **Not a PR reviewer.** Use `/review` or `/code-reviewer`. Ogma is strategic ("is this the right subsystem?"), not tactical ("is this the right diff?").

---

## Roadmap

| Version | Surface | Status | Depends on |
|---------|---------|--------|------------|
| v1.0 | `/ogma` skill | **Shipping now** | Lugh sources (PR #24) |
| v1.1 | Markdown archive indexed by animus_sync | Planned | v1.0 merged |
| v2.0 | Forge weekly briefing workflow | Planned | ChromaDB write path for lugh |
| v2.1 | Discord embed of weekly Ogma brief | Planned | v2.0 |
| v3.0 | MCP tool surface | Planned | v2.0 |
| v3.1 | Quorum-gated proposals for high-effort changes | Speculative | v3.0 + Quorum IntentNode wiring |

---

## First Real Test

The design call-out — Google AI memory compression work (likely [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663), Behrouz et al. 2024). Directly relevant because Animus **is** a memory system. A good v1 output would:

- Distill the learned-memory-at-test-time mechanism
- Gap-check against `packages/core/animus/memory.py` (ChromaDB + BM25 hybrid, no learned compression) — `animus_gap: NONE`
- Weaknesses: reproducibility unclear, perf at scale unvalidated, interaction with existing retrieval not specified
- Proposal: learned-memory overlay in `packages/core/animus/learning/`, composed with existing retrieval, optional per-query compression
- ROI: moderate effort, high value — could differentiate Animus's memory tier
- Confidence: 0.6 (promising but early)

That's the shape. Summary bots can't produce it; Ogma's grounding rules make it possible.
