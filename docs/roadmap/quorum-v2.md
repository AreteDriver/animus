# Quorum v2 — Extension Roadmap

**Status:** Draft, 2026-05-10
**Scope:** 5-week extension plan for Quorum and IntentResolver
**Stance:** Extend current Quorum, do not replace it
**Source conversation:** Analysis of `~/Downloads/ANIMUS_ASYNC_CA_SPEC.md` (2026-05-09 draft)

---

## Why this exists

A spec landed proposing an asynchronous graph cellular automaton (GCA) substrate for Quorum, with coupling detection (MI/TE), adaptive scheduling, and liveness watchdog. Full analysis in conversation; the short version:

- The CA engine and Kuramoto-style oscillator alternatives solve coordination problems Quorum does not have at current scale (5–20 agents, step-function workloads, not continuous oscillator networks).
- Existing Quorum already covers the coordination surface: stigmergy field, TriumvirateVoter, IntentResolver stability scoring, signal_bus pub/sub.
- Three pieces of the spec are genuinely missing from current Animus and earn their keep as observability primitives.
- One alternative — active inference inside IntentResolver — is the single behavior change worth making.

This roadmap commits to those four pieces. Everything else from the spec is explicitly deferred or rejected.

---

## Out of scope (do not build)

| Item | Reason for rejection |
|---|---|
| AsyncGCAEngine (full CA substrate) | Solves continuous-coordination problem Animus does not have |
| ScheduleController (adaptive period control) | Will oscillate without strong damping; stigmergy does not need rate adjustment |
| Pure transition function contract | Wrong shape for LLM agents (non-deterministic outputs); replay claim breaks |
| Kuramoto oscillator substrate | Elegant but no current workload is oscillator-shaped |
| Gossip / CRDT state replication | Defer until multi-instance Animus is real |
| Reservoir computing for tick stream | No current stream needs it |
| Multi-scale CA | Already have multi-scale (Bootstrap proactive / Forge workflow / Quorum resolution); just label takt times |

If any of these become load-bearing later, revisit. Do not build speculatively.

---

## Dependencies

```
Week 1: TickEvent log ──┬──> Week 2: LivenessWatchdog
                        └──> Week 5: Coupling dashboard

Week 3-4: Active-inference IntentResolver  (independent, can run in parallel)
```

Week 1 is prerequisite for Weeks 2 and 5. Weeks 3–4 are independent and can start any time.

---

## Week 1 — TickEvent append-only log

**Goal:** Every Quorum mutation (intent add, stability change, vote, stigmergy mark, resolver decision) emits a content-addressed event onto the existing signal_bus.

**Why it earns its keep:**
- Replay debugging for the self-improvement loop. Forge checkpoints are coarse; rubric scores lag. Transition-level events let you bisect *which mutation broke a stable intent*.
- Bootstrap dashboard observability — drop the event stream into the existing localhost:7700 dashboard.
- Direct port of memboot's bitemporal-lite pattern (`valid_from` / `recorded_at`, locked 2026-05-07). Map memboot field names 1:1 to TickEvent's `wall_time` / `tx_time`. Pattern is already proven; do not re-derive.

**Deliverables:**
- `packages/core/ontology/events/tick_event.py` — frozen dataclass with fields: `id`, `subject_id` (intent / agent / stigmergy marker), `subject_kind`, `wall_time`, `tx_time`, `prior_state_hash`, `new_state_hash`, `mutation_kind`, `triggered_by`, `task_id` (nullable), `payload_blob_ref` (content-addressed, optional)
- `packages/core/repository/event_store.py` — append-only write, range-by-time query, range-by-subject query. SQLite backend matching current memory.db pattern.
- Hooks at five mutation sites: IntentResolver stability update, TriumvirateVoter vote commit, stigmergy.mark, intent_graph.add_intent, intent_graph.demote.
- Signal_bus emit on every TickEvent write (existing pub/sub plumbing, no new infra).

**Success criteria:**
- All five mutation sites emit events with deterministic hashes.
- 200 ticks/sec write throughput on local SQLite (well above expected load).
- `query_events(subject_id, since)` returns ordered stream that exactly reproduces a synthetic test case's mutation history.
- Existing 926 Quorum tests still pass.

**Out of scope for this week:**
- Replay engine (just the log; replay is later, if needed).
- Cross-machine event sync.
- Event schema versioning beyond a single `schema_version` int.

**Effort estimate:** 3 days build, 1 day wiring, 1 day tests.

---

## Week 2 — LivenessWatchdog over event stream

**Goal:** Detect when a Quorum agent or coordination loop has stopped emitting events and surface as alerts.

**Why it earns its keep:**
- fleet-monitor watches *services* (process up/down). Bootstrap proactive engine watches *the user*. Nothing watches *individual coordination steps*. This closes the gap.
- Reuses fleet-monitor's existing Discord webhook path — no new alert channel to maintain.

**Deliverables:**
- `packages/quorum/liveness/watchdog.py` — sweep loop over active subjects, severity ladder (`warn` / `stalled` / `dead`) keyed off expected cadence per subject_kind.
- `core/ontology/events/dead_zone_alert.py` — alert event type with resolution lifecycle.
- Discord webhook adapter reusing fleet-monitor pattern.
- One config knob per subject_kind: `expected_period_ms`. No adaptive period adjustment.

**Success criteria:**
- Synthetic test: kill an agent's coroutine, verify `stalled` fires within 5× expected period and `dead` within 30×.
- Failure-counter test: synthetic agent that raises 5× consecutive triggers `dead` regardless of timing.
- Zero false positives on a 1-hour healthy run.

**Out of scope:**
- Auto-respawn (alert-only; respawn decisions sit in Forge or the operator).
- Adaptive expected_period (manually tuned per subject_kind).

**Effort estimate:** 2 days.

---

## Week 3–4 — Active-inference IntentResolver

**Goal:** Replace IntentResolver's evidence-counting stability score with a Bayesian posterior that updates on surprise.

**Why it earns its keep:**
- Current stability score is `f(positive_evidence, negative_evidence)`. Saturates fast. Cannot distinguish "100 confirmations of an obvious thing" from "100 confirmations of a contested thing that just shifted." Vulnerable to evidence flooding.
- Surprise-weighted updates: confirmatory evidence on a settled intent stops moving the score; contradictory evidence moves it more. Naturally robust against the failure mode evidence-counting has.
- Curiosity gradient as free side-effect: intents with high prior uncertainty become "things worth investigating." Forge consumes this signal to direct self-improvement work toward intents that need evidence, not intents already settled. Real flywheel for the existing self-improvement loop.

**Deliverables:**
- `packages/quorum/resolver/active_inference.py` — `ActiveInferenceStabilityScorer` implementing `StabilityScorer` protocol.
- `StabilityScorer` protocol extracted from current IntentResolver (refactor; current scorer becomes `EvidenceCountingStabilityScorer`).
- Feature flag: `quorum.resolver.scorer = "evidence_counting" | "active_inference"`. Default stays evidence-counting until A/B passes.
- Curiosity readout: `intent.curiosity_score: float` derived from posterior variance, written on each stability update, exposed through the existing intent query surface.
- A/B harness using forge eval framework: replay a captured stream of historical IntentResolver events through both scorers, score against `personal-quality` rubric.

**Success criteria:**
- Drop-in: zero changes to IntentResolver public API.
- A/B win on the captured stream: surprise-weighted scorer recovers ground-truth intent stability with ≥10% lower MSE than evidence-counting on a hand-labeled subset.
- Curiosity score correlates with self-improvement loop's eventual focus targets (post-hoc check on 2 weeks of historical Forge runs).
- Feature flag flip is reversible without data migration.

**Out of scope:**
- Replacing TriumvirateVoter (vote semantics unchanged).
- Multi-modal priors (Gaussian posterior over a single stability scalar is fine for v1).
- Hyperparameter learning (priors hand-tuned per intent_kind for v1).

**Effort estimate:** 3 days math + scorer implementation, 2 days protocol refactor, 2 days A/B harness, 3 days tuning.

---

## Week 5 — Coupling dashboard (read-only)

**Goal:** Mutual-information heatmap over the stigmergy marker stream, surfaced in the existing localhost:7700 dashboard.

**Why it earns its keep:**
- Diagnostic for stuck tasks: "agents A and B are over-coupling on this task" or "agent C decoupled from the swarm" is actionable signal.
- Read-only by design. **Never wire this to a controller.** The spec's ScheduleController will oscillate; coupling is useful as observability, harmful as actuation.

**Deliverables:**
- `packages/quorum/coupling/monitor.py` — sliding-window MI estimator over stigmergy marker stream, partitioned by `task_id`, restricted to graph-adjacent agent pairs (skip O(N²) full sweep).
- `packages/core/ontology/edges/coupling_edge.py` — derived edge with `mutual_information`, `sample_count`, `confidence`, `window_start`, `window_end`, `task_id`.
- Bootstrap dashboard tile: agent × agent heatmap, task filter dropdown.

**Success criteria:**
- Synthetic test: 3-agent ring with known coupling pattern, monitor recovers MI > 0 for coupled pairs and ≈ 0 for independent pairs (Miller-Madow corrected, K = 8, window = 1000 ticks).
- Heatmap renders on dashboard within 200ms for up to 50 agents.
- Zero writes back into Quorum state. Pure read.

**Out of scope:**
- Transfer entropy (MI alone is enough for the dashboard signal; TE was for the controller, which we are not building).
- Coupling-driven scheduling.
- Learned embeddings for state discretization (hash-mod-K is fine for stigmergy markers, which are already low-cardinality).

**Effort estimate:** 4 days.

---

## Re-evaluation gate (after week 5)

Stop. Run the new infrastructure for at least 2 weeks against real workloads. Then ask:

1. Did the TickEvent log materially improve any debugging session? If not, was it because nothing broke or because the events weren't granular enough?
2. Did LivenessWatchdog catch a real stall? Any false positives cost dev time?
3. Did the active-inference scorer A/B win? Did Forge actually consume the curiosity signal, or did it sit unused?
4. Did the coupling heatmap surface anything useful, or was it pretty wallpaper?

Each "no" is a signal to remove or replace, not extend. Each "yes" is a signal to deepen — and only then revisit deferred pieces (replay engine, learned scheduler, etc.) with real demand-side justification.

Do not pre-commit to a v3 roadmap before this gate.

---

## Pointers into current code

For the next agent picking this up:

| What | Where |
|---|---|
| IntentResolver + stability scoring | `packages/quorum/src/.../resolver.py` |
| TriumvirateVoter | `packages/quorum/src/.../triumvirate.py` |
| Stigmergy field | `packages/quorum/src/.../stigmergy.py` |
| Signal bus (pub/sub) | `packages/quorum/src/.../signal_bus.py`, `signal_backend.py` |
| FlockingCoordinator (parse-only stub) | `packages/quorum/src/.../convergent/flocking.py` |
| Memory provider protocol | `packages/core/.../protocols/memory.py` |
| Memory layer facade | `packages/core/.../memory/layer.py` |
| Memboot bitemporal-lite reference | `~/projects/memboot/` (see `docs/source_attribution.md`, ontology.md in notes repo) |
| Forge eval framework | `packages/forge/src/animus_forge/evaluation/` |
| Forge rubrics | `packages/forge/rubrics/` |
| Bootstrap dashboard | localhost:7700 (entry: bootstrap package) |
| fleet-monitor Discord pattern | `~/projects/fleet-monitor/` |

---

## Decision provenance

- 2026-05-09: ARETE Async CA spec draft
- 2026-05-10: Spec analysis identified mismatches with current Animus state (no GraphRepo, no bitemporal in core, existing stigmergy and TriumvirateVoter were ignored by spec)
- 2026-05-10: Decision to extend Quorum, not replace; commit to four pieces; defer the rest
- This document supersedes any plan to build the spec wholesale or build Path B (Kuramoto)

Log to ADL after first week-1 commit lands.
