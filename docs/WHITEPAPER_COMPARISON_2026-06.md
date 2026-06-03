# Current Code vs the 2026-06 Whitepaper — Gap Closure (as of 2026-06-03)

Re-running the canonical whitepaper's own gap list
([`whitepapers/ANIMUS_WHITEPAPER_2026-06.md`](whitepapers/ANIMUS_WHITEPAPER_2026-06.md))
against the code after Sessions 1-6, cross-checked against the 2026-06-03
adversarial [review](reviews/animus-10-10-review-2026-06-03.md). Status is
honest: ✅ closed, ◐ partial (closed in mechanism, gap remains), ✗ open.

## §6 Design Refinements (the P0-P3 backlog)

| # | Whitepaper refinement | Status | Notes |
|---|---|---|---|
| 1 | `allowed_tiers` default-deny | ✅ | Reframed (PR #80): egress surface already pins `{PUBLIC}`; the local-read default is the correct contract. Review confirmed. |
| 2 | Wire content taxonomy (F1-F8) | ✅ | PR #80. |
| 3 | Real code-exec sandbox | ◐ | RLIMIT bounding is real (B4). But it scores *crashing* code 1.0 when `expected` is None (review C6). |
| 4 | Unify the two egress copies | ✅ | `animus_types.egress` canonical; review confirmed no drift. |
| 5 | `allocate()` real reservation | ◐ | Raw-token reservation atomic + thread-safe (A2, confirmed). ET axis not enforced at admission (C8). |
| 6 | Effective-Tokens enforced | ◐ | Enforced post-hoc via `status==EXCEEDED` (A1). A single opus/output-heavy step bypasses it at admission (C8). |
| 7 | Judge-failure visibility | ✅ | B1; review confirmed end-to-end (raise → ERROR → `provider_error`). |
| 8 | Expand integrity baseline | ◐ | Mechanism tracks the cross-package primitive + self-hash (A6). But forge call-sites untracked (C5) and the deployed baseline is stale (boot-fail). |
| 9 | EventLog wiring enforced | ✗ | Not started (D6 tail). |
| 10 | `StabilityScorer` protocol | ✅ | B7; review confirmed byte-identical default + injectable seam. |
| 11 | BM25 incremental index | ✗ | Phase E (E1). |
| 12 | `ChromaMemoryStore` RAM | ✗ | Phase E (E2). |
| 13 | Quality-gate YAML parsing | ✗ | Phase E (E5). |
| 14 | Per-install encryption salt | ✅ | A7; review confirmed (no legacy data, zero-cost). |
| 15 | Gorgon rename + gitignore db | ◐ | `gorgon.yaml` removed; `CostTracker` still carries a stale "2024" table (C12). |

**§6 score: 7 closed, 5 partial, 3 open** (of 15). Up from 0 at whitepaper time.

## §7 Future Work

| Item | Status | Notes |
|---|---|---|
| Judge calibration / meta-eval | ✅ | B2; review confirmed statistics sound. |
| Auto-promotion eval loop | ◐ | Logic + staging/approval correct (B5). The experiment runner that feeds it drops the score (C7); zero production callers yet. |
| Active-inference IntentResolver | ✗ (seam ✅) | B7 laid the injectable scorer seam; the flood-resistant impl is unbuilt. |
| RA-1/2 research layer · overnight delegate (RA-3) | ✗ | Phase C — not started. |
| Encryption at rest + signed memory | ✗ | A5 — operator-gated (gocryptfs PR #67 unmerged). |
| Durability / `export --all` + rebuild | ✗ | A8 — Session 4 was skipped; **this is the biggest still-open "earn unattended operation" gap.** |
| Tiered memory · replay/bisect · secret manager · Pareto optimizer | ✗ | Phase D/E. |

## §8 Open Questions / Risks

| Risk | Movement | Notes |
|---|---|---|
| 8.1 Where do safety boundaries hold? | ◐ | Content-aware egress added (A4) but wired on only 4/6 providers + blind to tool blocks (C2-C4); encryption at rest still open (A5). The boundary is *stronger but still leaky*. |
| 8.2 Single-maintainer sustainability | ✅ (mitigated) | CANON.md + claim reconciliation + this review give the coherence machinery the whitepaper called for. |
| 8.3 Are the evals valid? | ✅ | Judge calibration (B2) + judge-failure visibility (B1) turn "assumed" into "measured." |
| 8.4 Cost at scale | ◐ | Raw reservation enforced (A2); ET enforcement is post-hoc only (C8); two pricing tables still diverge on the live path (C12). |

## Bottom line vs the whitepaper

The whitepaper's honest self-audit (68 production / 6 beta / 4 exp / 5 stub)
described a system with real cores and a long, named gap list. **Sessions 1-6
materially closed that list** — judge integrity, calibration, the scorer seam,
the per-install salt, the unified egress/credential source, raw-token cost
reservation, and the kaizen-loop scaffolding are genuinely in place and
adversarially verified.

But the same review that confirmed those also showed the work **over-claimed
"10/10" on D1/D3/D4/D5**: the enforcement edges (ET admission gate, Bedrock/Vertex
egress, Azure streaming, tool-block DLP, the integrity tracked-set + deployed
baseline, the sandbox crash-scoring, the auto-promotion runner) are bypassable
or broken, several masked by tests that mocked the failing path. The honest
delta from the whitepaper: **the catalogued gaps are mostly closed or in-flight,
the maturity is meaningfully higher, but the system is not yet 10/10 and not
fully unattended-ready** — durability/encryption (A5/A8) is untouched and the
[Phase C0 corrections](ROADMAP_TO_10.md) must land first.
