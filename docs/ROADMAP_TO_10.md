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
| D1 | **Cost discipline** (keystone) | **10** ✅ | ET is the enforced unit (A1); `allocate()` atomically reserves so concurrent steps can't collectively overspend (A2); `estimate_cost` derives from one tier table (A3). Cost is now a hard, thread-safe constraint. |
| D2 | **Memory** | 7 | Tiered HOT/WARM/COLD with real promotion; incremental BM25; no full second in-RAM copy; LLM-summarized consolidation; sync conflict-surfacing. |
| D3 | **Security & at-rest** | 9 | Content-aware egress ✅ (A4); integrity baseline covers all critical-path files + self-hash + cross-package egress logic ✅ (A6); per-install salt ✅ (A7). Remaining for 10: encryption at rest (A5) + a signed-memory decision. |
| D4 | **Eval integrity** | 9 | Judge failure ≠ silent 0.5 ✅ (B1); content taxonomy wired ✅ (PR #80); real code-exec sandbox ✅ (B4); power/sample-size advisor on `compare` ✅ (B6). Remaining for 10: judges calibrated against a human golden set + drift tracked (B2, Session 6). |
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
| **A1 ✅** | **Make Effective-Tokens the enforced budget unit ("the flip")** — DONE Session 1, option (b) | §6 #6, §8.4, cold-read keystone | ET reporting-only → **ET enforced** | Chose (b): raw `total_budget` kept, ET ceiling auto-derived; status takes worse-of-raw/ET; executor halts on EXCEEDED; migration 020 persists ET across restart. Non-breaking (raw-only records neutral) so 0 workflow YAMLs changed. Done-criteria test green; full forge suite 10,210 passed | L |
| **A2 ✅** | `BudgetManager.allocate()` real reservation — DONE Session 2 | §6 #5, appendix | check-only → reserves | Lock + pending state; `allocate()` atomically check-and-reserves; parallel handler reserves once/sub-step + releases in finally; threaded test proves no collective overspend | M |
| **A3 ✅** | Single pricing source — DONE Session 2 | appendix (orchestration) | two tables → one | `estimate_cost` derives from `DEFAULT_MODEL_MULTIPLIERS` × base $9/1M; config overrides flow into cost; reproduces old opus/sonnet numbers | S |
| **A4 ✅** | Content-aware egress — DONE Session 3 | §8.1, Qwen #1, residual gap | tier-trust only → tier + content scan | Canonical credential scanner in `animus_types.secrets` (reused by core redaction — no drift); `is_egress_allowed(content=...)` denies credential-bearing payloads; 4 providers wired via `assert_egress_allowed`; secret mis-tagged PUBLIC is blocked | M |
| **A5** | Encryption at rest | §7 significant, §8.1, ARCHITECTURE claim | plaintext ext4 → encrypted | gocryptfs vault (PR #67) finished + memory store inside it; documented recovery; CANON/ARCHITECTURE flip from PLANNED→done | M |
| **A6 ✅** | Expand integrity baseline — DONE Session 3 | §6 #8, Qwen #3 | 4 files → all critical-path + self | Self-hashes the checker + guardrails; adds cross-package modules via importlib (`animus_types.egress`/`secrets` required, forge openrouter/pi_wrap optional) — closes the gap where core's egress was a re-export shim; tamper of any trips verify. **Operator must re-baseline** (`python -m animus.integrity.cli regenerate`) after this lands | M |
| **A7 ✅** | Per-install encryption salt — DONE Session 2 | §6 #14 | hardcoded `gorgon-` salt → random per-install | `get_install_salt` helper; random 16B persisted 0600 to `~/.config/animus`; wired at BOTH sites (field_encryption + settings/manager); env overrides kept. Verified no legacy-encrypted data exists, so zero migration cost | S |
| **A8** | Durability: rebuild dry-run + `animus export --all` | §7 significant | none → portable archive + tested rebuild | `export --all` produces documented-schema archive; timed cold rebuild restores state; loss-of-machine is survivable | M |

Exit: D1 → 9, D3 → 9, D7 substrate in place. **This is the phase that matters most.**

---

## Phase B — Eval integrity + close the Kaizen loop

The self-improvement loop is only trustworthy if the evals it depends on are.

| ID | Item | Source | Acceptance | Effort |
|---|---|---|---|---|
| **B1 ✅** | Judge-failure visibility — DONE Session 5 | §6 #7, §8.3 | judge raises `JudgeError` (no provider / call fails / unparseable) instead of 0.5; evaluator turns it into an ERROR result; classifier buckets `provider_error` | S |
| **B2** | Judge calibration / meta-eval harness | §7 significant, §8.3 | Periodic scoring of judges vs a human golden set; per-model judge-drift tracked as a first-class metric | M |
| **B3** | Evolution loop: real experiment runner + CLI | appendix (forge) | Replace dry-run echo with an injected runner; expose via CLI/daemon so autoresearch is operator-accessible | M |
| **B4 ✅** | Real code-exec sandbox — DONE Session 5 | §6 #3 (beyond interim) | `-I -S` + scrubbed env + isolated cwd + kernel RLIMITs (CPU/memory/file-size via preexec_fn); infinite loop → TIMEOUT, memory bomb bounded — not just `-I` | M |
| **B5** | Auto-promotion eval loop | §7 transformative | A statistically-significant `eval compare` win auto-opens a WorkflowEvolution pending patch pinning the prompt version | M |
| **B6 ✅** | `compare` power/sample-size advisor — DONE Session 5 | §7 incremental | `underpowered`/`power_note` on ComparisonReport: a non-significant result with CI wider than ±0.05 is flagged underpowered (vs tight-CI equivalence); surfaced in the `compare` CLI | S |
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
| E15 | **Migration-number collision guard** — a test/CI check that fails when two `migrations/*.sql` share a leading number | **Session 1 (the `012` collision that broke the first full run)** | S |

---

## Completeness matrix (every audit issue → roadmap ID)

So nothing silently drops.

- §6 refinements: #1✅(P0) #2✅ #3✅interim/B4 #4✅mech/A1 #5→A2 #6→A1 #7→B1 #8→A6 #9→D4i #10→B7 #11→E1 #12→E2 #13→E5 #14→A7 #15→E6
- §7 future work: RA→C1/C2 · auto-promotion→B5 · tiered memory→D1i · replay/bisect→D3i · active-inference→D2i · overnight delegate→C2 · encryption+signing→A5 · durability/export→A8 · judge-calibration→B2 · secret manager→E7 · liveness/coupling→D5i · pareto→E8 · power-advisor→B6
- §8 risks: 8.1→A4/A5/A6/E9/E10 · 8.2→D8/CANON(✅) · 8.3→B1/B2 · 8.4→A1/A2/A3
- Qwen net-new: #4→E11 · #5→E12 · #6→E13 ; reinforced: egress-content→A4, encryption→A5, integrity→A6
- Appendix limitations not above: consolidation→E3 · sync→E4 · pricing→A3 · evolution-dry-run→B3 · impact-score-quality→(fold into B5/D5i)

## Session plan — demo/structure → fully operational, ASAP, on-spec-or-better

**The goal of this arc:** take every feature that is currently *claimed but not
actually working* (reporting-only cost, opt-in security gates, dry-run loops,
plaintext-at-rest) to **fully operational, meeting or exceeding the whitepaper's
stated contract.** Not new capabilities — the existing system, made real.

**Definition of OPERATIONAL (the milestone, end of Session 6):** no feature in
the system is reporting-only, opt-in-by-accident, stubbed, or
trust-the-caller. Every claimed guarantee is enforced and has a test that
proves it. After this, Animus can be *relied on*, not just demoed.

**On-spec-or-better rule:** each "done when" criterion is at least the
whitepaper's contract; where cheap, it exceeds it (e.g. egress goes from
tier-trust to *content-aware*, which is stronger than the spec). Never ship a
criterion weaker than the documented contract.

**Critical path:** Session 1 (cost keystone) is the highest-leverage block — do
it first, alone. Sessions 1–6 reach OPERATIONAL. Session 7 is residual security
hardening (incl. the local security-review items) and can run in parallel with
later work — it does not gate "operational."

Each session: one coherent cluster, a definition of done that is **a passing
test or check**, and a clean PR. Pace is yours; a "session" is a focused block,
not a fixed number of hours. Don't start the next until the prior exits green.
Sessions can merge if a block runs short; split if it runs long.

### Session 1 — The keystone: enforce cost (roadmap A1) ✅ DONE
**Why first:** every other claim rests on cost being a hard constraint; today it
isn't. Breaking change, so it gets its own session.
- [x] **Decision gate:** chose **(b)** — keep raw `total_budget`, auto-derive the
      ET ceiling. Non-breaking.
- [x] Route the executor's `_check_budget_exceeded` + `record_usage` through the
      ET-aware `status` (worse-of-raw/ET governs; `effective_ceiling` derives
      from `total_budget`; `enforce_effective_tokens=False` is the escape hatch).
- [x] `_restore_from_db()` rebuilds `_total_effective` via migration 020 (persists
      per-record ET), with a raw-only fallback for un-migrated DBs.
- [x] Workflow YAMLs: **0 changed** — every production `record_usage` is raw-only,
      so ET == raw there (neutral). Migrated the 5 budget tests that encoded the
      old opt-in behavior + 3 migration-fixture lists.
- [x] **Done:** parallel opus run trips `EXCEEDED` at 30% raw (test green);
      executor halts on it; **full forge suite 10,210 passed, 0 failed**. D1 → 9.

### Session 2 — Finish the cost cluster + a quick security win (A2, A3, A7) ✅ DONE
- [x] **A2** `BudgetManager.allocate()` reserves under a lock; parallel handler
      reserves once/sub-step + releases in finally; threaded test proves 20
      concurrent 200-tok reservations grant exactly 10 against a 2000 budget.
- [x] **A3** `estimate_cost` derives from the one tier-multiplier table × base
      $9/1M (config overrides flow through); old opus/sonnet numbers preserved.
- [x] **A7** `get_install_salt` (random 16B, persisted 0600 to `~/.config/animus`)
      wired at both salt sites; env overrides kept. Confirmed **no** legacy-
      encrypted data exists (credentials/api_keys/mcp/settings all 0 rows), so
      zero migration cost.
- [x] **Done:** A2 reservation + threaded tests, A3 single-source tests, A7 salt
      tests all green; affected coverage tests migrated (removed-method gate).
      Prior full run 10,221 passed / 3 failed-now-fixed; CI confirms. **D1 → 10.**

### Session 3 — Egress + integrity (A4, A6) ✅ DONE
- [x] **A4** content-aware egress: canonical credential scanner in
      `animus_types.secrets` (core redaction now reuses it — drift removed);
      `is_egress_allowed(content=...)` denies credential-bearing payloads; the 4
      cloud providers gate via a shared `assert_egress_allowed`. Closes the
      §8.1 residual + Qwen #1/#2/#3.
- [x] **A6** integrity baseline self-hashes the checker + guardrails AND tracks
      cross-package modules via importlib (`animus_types.egress`/`secrets`
      required; forge `openrouter`/`pi_wrap` optional) — the core `network/egress.py`
      was a re-export shim, so the real logic was previously untracked.
- [x] **Done:** mis-tag egress test blocks; module-drift + tamper trip verify;
      verify_hardening 24/24; core security slice 92 passed; types 31; forge DLP
      78. **D3 → 9.** ⚠️ Operator re-baseline required: `python -m
      animus.integrity.cli regenerate`.

### Session 4 — At-rest + durability (A5, A8)
- [ ] **A5** finish the gocryptfs vault (PR #67) with the memory store inside it;
      documented recovery; flip ARCHITECTURE/CANON from PLANNED → done.
- [ ] **A8** `animus export --all` (documented schema) + a timed cold-rebuild
      that restores state.
- **Done when:** store is encrypted at rest; a from-scratch rebuild passes.
  D3 → 10, D7 substrate in place.

### Session 5 — Eval integrity fixes (B1, B4, B6) ✅ DONE
- [x] **B1** judge raises `JudgeError` (no provider / call fails / unparseable)
      → ERROR result → classified `provider_error`, never a silent 0.5.
- [x] **B4** code-exec sandbox: `-I -S` + scrubbed env + isolated cwd + kernel
      RLIMITs (CPU/memory/file-size). Infinite loop → TIMEOUT; memory bomb bounded.
- [x] **B6** `compare` advisor: `underpowered`/`power_note` distinguish a wide-CI
      underpowered "not significant" from tight-CI equivalence; shown in the CLI.
- [x] **Done:** broken-judge test surfaces an error + routes provider_error;
      sandbox bounds a runaway; small-suite compare flags underpower. eval 236,
      compare 22, sandbox/routing green; whole-tree ruff clean. **D4 → 9.**

> Note: Session 4 (A5 encryption-at-rest + A8 durability) was skipped to here;
> do it before the autonomy/daily-tool phases (it gates D7 + D3=10).

### Session 6 — Close the Kaizen loop (B2, B3, B5, B7)
- [ ] **B2** judge-calibration harness vs a human golden set; track judge drift.
- [ ] **B3** evolution loop: inject a real experiment runner + expose via CLI.
- [ ] **B5** auto-promotion: a significant `eval compare` win opens a
      WorkflowEvolution pending patch.
- [ ] **B7** extract the `StabilityScorer` protocol (prereq for D-phase active
      inference).
- **Done when:** calibration metric exists; evolution loop runs a real
  experiment from the CLI; auto-promotion opens a patch in a test. D4 → 10, D5 → 9.

### Session 7 — Security residual hardening (Qwen review + §8.1 leftovers)
Does NOT gate "operational" — run it parallel to Phase C/D or right after S6.
The 2026-06 local security review's 3 net-new items + the remaining §8.1 gaps.
- [ ] **E13 (Qwen #6) TOCTOU on tier labels** — verify A4 closed it; if A4
      inspects content at egress time the label-mutation window is moot, but add
      an explicit re-check-at-use test. *(Mostly covered by Session 3 / A4.)*
- [ ] **E11 (Qwen #4) self-improve / red-team loop abuse** — the loop that
      generates and applies changes is an injection surface. Harden: the
      red-team driver's generated probes can never reach an apply path; the
      self-improve sandbox rejects probe-shaped input; test an adversarial probe
      cannot escalate. *(Touches the Session 6 loop — can fold there if timing fits.)*
- [ ] **E12 (Qwen #5) local-model supply-chain** — pin + checksum the Ollama
      model blobs Animus trusts; refuse to run a red-team/eval model whose digest
      changed unexpectedly. Closes the "backdoored open-weight model" vector.
- [ ] **E9 systemd unit integrity** — version the unit files in the repo and add
      them to the A6 integrity baseline so an edited unit (which defeats the
      kernel-plane filter) is detected.
- [ ] **E10 PI-envelope cross-model testing** — the prompt-injection footer is
      only tested against Sonnet; add Qwen/Llama cases (the local-first models
      that are the *actual* primary consumers).
- **Done when:** supply-chain digest test trips on a changed blob; probe-can't-
  escalate test passes; unit-tamper trips boot refusal; PI envelope holds on
  Qwen/Llama. D3 → 10 (fully hardened).
- [ ] **E14** Ollama upgrade so the uncensored 35B Qwen3.6-A3B loads — then
      re-run the adversarial review with the *aggressive* model for a stronger pass.

**After Session 6 = OPERATIONAL.** The existing system is fully working,
on-spec-or-better, every claim enforced and tested. Session 7 hardens the
security tail. Only *then* do Phases C (daily-tool: RA layer, overnight delegate)
and D (differentiators: tiered memory, active inference, replay) begin — the
*build* arc, re-planned into sessions on the now-trustworthy substrate.

### Phase E sessions — scale, hygiene, guards (non-security; continuous)

Phase E is *opportunistic*: none of these gate OPERATIONAL, and any can be
slotted in when its area is already open. Security-tail E-items (E9–E14) live in
Session 7. The rest cluster into three optional sessions below. Same rule: a
"done when" that is a passing test, and a clean PR.

#### Session 8 — Memory scaling & correctness (E1–E4)
Lifts the memory layer off its scaling ceilings and removes silent data loss.
Best done when the memory store is otherwise quiet (no concurrent migrations).
- [ ] **E1** BM25 index updates incrementally on store/delete (no full O(N)
      re-tokenize per write). *Done when:* a write does not rebuild the whole
      corpus (assert via a spy/counter); retrieval results unchanged.
- [ ] **E2** `ChromaMemoryStore` stops holding a full second in-RAM dict; hydrate
      `Memory` objects on demand, keep only an id→metadata index for BM25.
      *Done when:* memory footprint scales sub-linearly in a large-corpus test;
      all store contract tests still pass. (L — the heaviest Phase E item.)
- [ ] **E3** `consolidate()` summarizes via the cognitive layer instead of
      first-tag 150-char concat. *Done when:* a consolidated memory is a real
      summary (LLM in test = mock returning a marker), originals untouched
      (append-only preserved).
- [ ] **E4** Cross-device sync surfaces conflicts instead of silent
      last-write-wins. *Done when:* two divergent edits to the same record
      produce a recorded conflict (not a dropped loser) in a sync test.
- **Exit:** D2 (memory) scaling ceilings removed; no silent loss. D2 → 10.

#### Session 9 — Hygiene & dev-tooling guards (E5, E6, E8, E15)
Cheap correctness + the poka-yoke that would have caught Session 1's own bug.
- [ ] **E15** Migration-number collision guard (NEW, from Session 1). A test
      globs `migrations/*.sql`, parses the leading integer, and asserts the set
      is unique. *Done when:* the test fails on a duplicate number and passes on
      the current tree. ~20 lines; do this **first** in the session — it is the
      poka-yoke for the exact failure that broke Session 1's first full run.
- [ ] **E5** Reconcile the quickstart `gates:` YAML: either parse it in the
      production `WorkflowConfig` loader (reuse Core's safe gate parser) or
      remove the example from the README. *Done when:* docs and the loader agree
      (a test asserts a `gates:` workflow either enforces or is rejected, not
      silently ignored).
- [ ] **E6** Finish the Gorgon→Forge rename (`branch_prefix` default, residual
      identifiers) and gitignore the committed runtime `*.db` files. *Done when:*
      `grep -ri gorgon src/` is clean of live identifiers; no `*.db` tracked.
- [ ] **E8** Cost/quality Pareto optimizer over the eval run store: recommend
      the cheapest (model, prompt_version) holding a target quality band.
      *Done when:* given seeded run history, it returns the Pareto-optimal config
      in a test. (Builds on the now-enforced Effective-Tokens from Session 1.)
- **Exit:** D5/D8 hygiene closed; the migration guard prevents a repeat of the
  Session 1 collision.

#### Session 10 (optional, large) — Unified secret manager (E7)
- [ ] **E7** Build the `SECURITY_LAYER.md` secret manager (age/pass backend,
      `animus secrets` CLI, `secrets://` URI resolution) to replace ad-hoc
      credential handling. Large and cross-cutting with security; sequence it
      near Session 7 or whenever credential handling next causes friction.
      *Done when:* a `secrets://` reference resolves at load; no plaintext
      secret in config; round-trip test green. Flip the SECURITY_LAYER spec
      `ASPIRATIONAL-SPEC → CANONICAL` in `CANON.md` when it lands.

## Working rule

When an item lands, in the **same PR**: flip its status in `CANON.md`, tick it
here, and remove it from `TODO_NEXT.md`. That is how this stays a live map and
not roadmap #5.
