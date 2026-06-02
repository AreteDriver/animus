# Animus — Roadmap to 10/10

**Status: CANONICAL (remediation roadmap)** · Created 2026-06-02 · Owner: ARETE (solo)

## What this is

A personal work-queue to close every gap surfaced by the 2026-06 whitepaper
audit, the local adversarial security review, and the cost-enforcement critique
from the cold read. It is **not** a pitch and not a product roadmap. It is the
"what do I do next, in what order, and how do I know each piece is done" doc for
future-me.

Relationship to other docs (see [`CANON.md`](CANON.md)):
- `whitepapers/ANIMUS_WHITEPAPER_2026-06.md` — the audit that found these gaps.
- `PERSONAL_ROADMAP.md` — *direction* (what Animus is allowed to become).
- This file — *remediation* (closing the gaps to 10/10). When this and
  PERSONAL_ROADMAP disagree on scope, PERSONAL_ROADMAP wins (anti-productization).
- `TODO_NEXT.md` — short-horizon queue; points here for the audit work.

**Pace is yours.** Effort is sized S/M/L, never dated. Sequence is
dependency-driven, not calendar-driven. Account for Toyota OT + sleep recovery;
this is a months-long personal arc, not a sprint.

---

## Definition of 10/10

Eight dimensions. A dimension is 10/10 when its acceptance criteria all hold and
are test- or check-enforced (not just asserted in a doc). Current scores are
honest reads from the whitepaper maturity (68 production / 6 beta / 4 exp / 5 stub).

| # | Dimension | Now | 10/10 acceptance criteria |
|---|---|---|---|
| D1 | **Cost discipline** (keystone) | 6 | Effective-Tokens is the *enforced* unit; `allocate()` reserves; one pricing table; a parallel output-heavy opus run cannot overspend. The thesis is enforced, not reported. |
| D2 | **Memory** | 7 | Tiered HOT/WARM/COLD with real promotion; incremental BM25; no full second in-RAM copy; LLM-summarized consolidation; sync conflict-surfacing. |
| D3 | **Security & at-rest** | 7 | Encryption at rest; content-aware egress (not tier-trust); integrity baseline covers all critical-path files + self; per-install salt; signed memory or explicit decision not to. |
| D4 | **Eval integrity** | 7 | Judges calibrated against a human golden set + drift tracked; judge failure ≠ silent 0.5; content taxonomy wired (done); real code-exec sandbox; power/sample-size advisor on `compare`. |
| D5 | **Orchestration** | 7 | Budget reservation enforced; gates parsed where docs claim; impact score includes quality regression; no stale-pricing double view. |
| D6 | **Coordination (Quorum)** | 6 | Pluggable StabilityScorer (flood-resistant); EventLog wiring enforced at the bridge; content-addressed transition log enabling replay/bisect. |
| D7 | **Autonomy-readiness** | 5 | Durability/rebuild dry-run + `animus export --all`; encryption at rest; a measured one-week overnight-delegate run. Earn the right to run unattended. |
| D8 | **Doc coherence** | 8 | CANON kept current (status flips in the same PR as the feature); zero present-tense claims without code; no contradicting roadmaps. (Mostly closed in PR #80.) |

10/10 overall = all eight at criteria, **and** the cost keystone (D1) enforced,
because every other claim rests on it.

---

## Phase 0 — DONE (PR #80)

Closed already; recorded so future-me doesn't redo it.

- P0 refinements: content-taxonomy wired (#2), code-exec interim isolation via
  `sys.executable -I` (#3, real sandbox still pending → B4), egress unified into
  `animus_types` (#5), tier default-deny reframed + `recall_for_egress` contract
  (#1), ET **opt-in ceiling** mechanism shipped (#4 — the *flip* is A1).
- Test-warning cleanup (root-fixed dead `yaml_file` config + Security filters).
- Claim reconciliation + `CANON.md` + canonical whitepaper + clickable PDF.
- → Advances D8 to ~8; lays the mechanism for D1.

---

## Phase A — The keystone + trustworthy substrate

Do this first. A1 makes the central thesis true; the rest is what you must have
before it is sane to run Animus unattended.

| ID | Item | Source | Current → Target | Acceptance | Effort |
|---|---|---|---|---|---|
| **A1** | **Make Effective-Tokens the enforced budget unit ("the flip")** | §6 #6, §8.4, cold-read keystone | ET reporting-only → ET enforced | Decide semantics (reinterpret `total_budget` as ET, or auto-derive `effective_token_budget` from it); migrate 30+ workflow YAMLs + the forge suite; a parallel opus output-heavy run trips EXCEEDED in tests | L |
| **A2** | `BudgetManager.allocate()` real reservation | §6 #5, appendix | check-only → reserves | Pending allocations tracked; concurrent `can_allocate` cannot collectively overspend; released on record/cancel; test proves it | M |
| **A3** | Single pricing source | appendix (orchestration) | two tables (stale 2024 + multipliers) → one | `estimate_cost` and ET multipliers read one table; no dashboard/budget mismatch | S |
| **A4** | Content-aware egress (stop trusting caller tier) | §8.1, Qwen #1, residual gap | tier-trust only → tier + content scan | Egress gate runs DLP/secret scan on payload before allow; a CONFIDENTIAL body mis-tagged PUBLIC is blocked; test with mis-tag | M |
| **A5** | Encryption at rest | §7 significant, §8.1, ARCHITECTURE claim | plaintext ext4 → encrypted | gocryptfs vault (PR #67) finished + memory store inside it; documented recovery; CANON/ARCHITECTURE flip from PLANNED→done | M |
| **A6** | Expand integrity baseline | §6 #8, Qwen #3 | 4 files → all critical-path + self | tier-router, providers, pi_wrap, integrity checker itself hashed; tamper of any trips boot refusal; test each | M |
| **A7** | Per-install encryption salt | §6 #14 | hardcoded `gorgon-` salt → random per-install | Salt generated + persisted to `~/.config/animus`; legacy constant gone | S |
| **A8** | Durability: rebuild dry-run + `animus export --all` | §7 significant | none → portable archive + tested rebuild | `export --all` produces documented-schema archive; timed cold rebuild restores state; loss-of-machine is survivable | M |

Exit: D1 → 9, D3 → 9, D7 substrate in place. **This is the phase that matters most.**

---

## Phase B — Eval integrity + close the Kaizen loop

The self-improvement loop is only trustworthy if the evals it depends on are.

| ID | Item | Source | Acceptance | Effort |
|---|---|---|---|---|
| **B1** | Judge-failure visibility | §6 #7, §8.3 | LLM-judge exceptions → ERROR/`judge_error` (not silent 0.5); routed to `provider_error` bucket | S |
| **B2** | Judge calibration / meta-eval harness | §7 significant, §8.3 | Periodic scoring of judges vs a human golden set; per-model judge-drift tracked as a first-class metric | M |
| **B3** | Evolution loop: real experiment runner + CLI | appendix (forge) | Replace dry-run echo with an injected runner; expose via CLI/daemon so autoresearch is operator-accessible | M |
| **B4** | Real code-exec sandbox | §6 #3 (beyond interim) | `CodeExecutionMetric` runs in the self-improve Sandbox or a restricted namespace, not just `-I` | M |
| **B5** | Auto-promotion eval loop | §7 transformative | A statistically-significant `eval compare` win auto-opens a WorkflowEvolution pending patch pinning the prompt version | M |
| **B6** | `compare` power/sample-size advisor | §7 incremental | "not significant" on a tiny suite reports underpower, not "no difference" | S |
| **B7** | StabilityScorer protocol extraction | §6 #10 | `compute_stability()` delegates to an injectable scorer (prereq for D-phase active inference) | M |

Exit: D4 → 9-10, D5 → 9, Kaizen loop genuinely autonomous-within-gates.

---

## Phase C — Make it the daily tool

The point of the whole system. Validates the autonomy claims under real load.

| ID | Item | Source | Acceptance | Effort |
|---|---|---|---|---|
| **C1** | RA-1/RA-2 research capability layer | §7 transformative, RA roadmap | WebFetch(allowlist) + Retrieve + Cite + Synthesize; source-grounded output contract ("answer without sources" = fail) | L |
| **C2** | RA-3 overnight delegate | §7 significant, §8 | Persistent SQLite task queue + turn-level checkpoint/resume + morning digest (task→outcome→cost→citations); measured 1-week unattended intervention rate | L |

Exit: D7 → 10 (the delegate is the proof the substrate works). Claude Code reverts to advisory.

---

## Phase D — Differentiators (build last, on a proven substrate)

High-impact, but each depends on earlier work. Building these on an unproven base
makes demos you can't trust.

| ID | Item | Source | Dep | Effort |
|---|---|---|---|---|
| **D1i** | HOT/WARM/COLD tiered memory + lossless compaction | §7 transformative, §6, ANIMUS_MEMORY_GAPS | A8 durability | L |
| **D2i** | Active-inference IntentResolver | §7 significant, Quorum v2 | B7 scorer protocol, B2 calibration | L |
| **D3i** | Replay/bisect over content-addressed event log | §7 transformative | EventLog rework | M |
| **D4i** | EventLog wiring enforced at GorgonBridge | §6 #9 | — | S |
| **D5i** | LivenessWatchdog + coupling MI dashboard | §7 incremental, Quorum v2 | EventLog | M |

Exit: D2 → 10, D6 → 10.

---

## Phase E — Scale, hygiene, residual hardening (continuous)

Slot in opportunistically; none blocks the above. Includes the Qwen net-new items.

| ID | Item | Source | Effort |
|---|---|---|---|
| E1 | BM25 incremental index (no O(N) re-tokenize per write) | §6 #11 | M |
| E2 | ChromaMemoryStore: drop the full in-RAM mirror | §6 #12 | L |
| E3 | LLM-summarized consolidation (not first-tag concat) | appendix (memory) | M |
| E4 | Sync: conflict-surfacing (not silent last-write-wins) | appendix (memory) | M |
| E5 | Quality-gate YAML parsing reconcile (or remove the README example) | §6 #13 | M |
| E6 | Finish Gorgon→Forge rename; gitignore committed `*.db` | §6 #15 | S |
| E7 | Unified secret manager (age/pass backend, `secrets://`) | §7 significant, SECURITY_LAYER spec | L |
| E8 | Cost/quality Pareto optimizer over run history | §7 incremental | M |
| E9 | systemd unit files versioned in repo + integrity-checked | §8.1, Qwen #7 | S |
| E10 | PI-envelope cross-model testing (Qwen/Llama, not just Sonnet) | §8.1, Qwen #P1 | M |
| E11 | **Self-improve / red-team loop abuse hardening** | **Qwen #4 (net-new)** | M |
| E12 | **Local-model supply-chain: pin + checksum model blobs** | **Qwen #5 (net-new)** | M |
| E13 | **TOCTOU on tier labels: immutable tier at read / re-check at use** | **Qwen #6 (net-new)** | M |
| E14 | Ollama upgrade so the uncensored 35B Qwen3.6-A3B red-team model loads | this session | S |

---

## Completeness matrix (every audit issue → roadmap ID)

So nothing silently drops.

- §6 refinements: #1✅(P0) #2✅ #3✅interim/B4 #4✅mech/A1 #5→A2 #6→A1 #7→B1 #8→A6 #9→D4i #10→B7 #11→E1 #12→E2 #13→E5 #14→A7 #15→E6
- §7 future work: RA→C1/C2 · auto-promotion→B5 · tiered memory→D1i · replay/bisect→D3i · active-inference→D2i · overnight delegate→C2 · encryption+signing→A5 · durability/export→A8 · judge-calibration→B2 · secret manager→E7 · liveness/coupling→D5i · pareto→E8 · power-advisor→B6
- §8 risks: 8.1→A4/A5/A6/E9/E10 · 8.2→D8/CANON(✅) · 8.3→B1/B2 · 8.4→A1/A2/A3
- Qwen net-new: #4→E11 · #5→E12 · #6→E13 ; reinforced: egress-content→A4, encryption→A5, integrity→A6
- Appendix limitations not above: consolidation→E3 · sync→E4 · pricing→A3 · evolution-dry-run→B3 · impact-score-quality→(fold into B5/D5i)

## Working rule

When an item lands, in the **same PR**: flip its status in `CANON.md`, tick it
here, and remove it from `TODO_NEXT.md`. That is how this stays a live map and
not roadmap #5.
