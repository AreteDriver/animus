# Animus — Document Canon & Status Index

**Last updated: 2026-06-02**

This repo accumulated multiple whitepapers, roadmaps, and design specs written
at different times. Some describe shipped code; some describe aspirational
designs that were never built; a few make claims that do not survive contact
with the source. This index is the single authoritative pointer to **what is
canonical, what is historical, and what is aspirational**. When two documents
disagree, this index wins.

## How to read a status

- **CANONICAL** — current and code-grounded. Cite these.
- **HISTORICAL** — accurate when written, now partly stale. Useful for context,
  not for current claims.
- **SUPERSEDED** — replaced by a canonical doc; retained for history only. Do
  not cite.
- **ASPIRATIONAL-SPEC** — a design or plan, not a description of shipped code.
  Valid as intent; not evidence of capability.

## Whitepapers

| Document | Status | Notes |
|---|---|---|
| `whitepapers/ANIMUS_WHITEPAPER_2026-06.md` | **CANONICAL** | Code-grounded, evidence-cited, honest about maturity. The paper to cite or hand to a reader. |
| `WHITEPAPER.md` | **SUPERSEDED** (was v2.0, Feb 2026) | Built around a fictional "production deployment". See below. |
| `whitepapers/animus-whitepaper.md` | **SUPERSEDED** | Byte-identical duplicate of `WHITEPAPER.md`. |
| `whitepaper.pdf` | **SUPERSEDED** | Rendered from the old v2.0. Regenerate from the 2026-06 paper before any external use. |

## Roadmaps

| Document | Status | Notes |
|---|---|---|
| `PERSONAL_ROADMAP.md` | **CANONICAL** | Most current (2026-05-15). Single-user doctrine + resist-productization checklist. The load-bearing roadmap. |
| `ROADMAP.md` | **HISTORICAL** | Phased plan (Phase 0–6); phases 0–4 largely done, 5–6 aspirational. |
| `ROADMAP_quorum_v2.md` | **ASPIRATIONAL-SPEC** | 5-week Quorum v2 plan; Week-1 EventLog shipped, weeks 2–5 are specs only. |
| `ROADMAP_research_assistant.md` | **ASPIRATIONAL-SPEC** | RA-0 locked, RA-1+ to be written. |
| `TODO_NEXT.md` | **CANONICAL** | Active next-work tracker, including the Effective-Tokens default-flip. |

## Known claims that are NOT backed by code (do not repeat as fact)

These appear across older docs. They are **design targets or fiction**, not
shipped capability:

1. **Media Engine — "~480 videos/month, 3 YouTube channels, 8 languages".**
   No implementing code in this repo (only vendored `googleapiclient` stubs).
   Appears in: `WHITEPAPER.md` §6, `OLLAMA_SETUP.md`, `DEVELOPER_TOOLS.md`,
   `animus-build-spec.md`. **Status: never built.** The actually-exercised
   workloads are developer/fleet-ops (code-review, fleet-triage,
   security-audit, bounty-watcher) plus the eval harness as a regression gate.
2. **Marketing Engine — 5-platform autonomous posting.** Design only.
   `SECURITY_LAYER.md`, `BROWSER_AUTOMATION.md`, `DEVELOPER_TOOLS.md` reference
   it as future context. **Status: specced, not built.**
3. **Encryption at rest (AES-256) / Ed25519-signed memory.** Claimed in
   `ARCHITECTURE.md`, `CASE_STUDY.md`, `ANIMUS_MEMORY_GAPS.md`, `rework/`.
   Both memory stores persist **plaintext**; signing is unimplemented.
   **Status: future work** (see `THREAT_MODEL.md` for the real at-rest gap).
4. **Developer-Tools / Arete-Tools revenue-tier ecosystem.** `DEVELOPER_TOOLS.md`
   describes a 9-tool suite with $5–49/mo tiers. **Status: superseded** by the
   `PERSONAL_ROADMAP.md` anti-productization stance (no SSO/RBAC/billing).
5. **HOT/WARM/COLD tiered memory + lossless compaction.** Fully specced in
   `ANIMUS_MEMORY_GAPS.md`; **zero implementing code**.
6. **Quorum v2 active-inference resolver / liveness watchdog / coupling
   dashboard.** Specs only; the planned modules do not exist.

## Design / spec docs (valid as intent, not as capability)

`ANIMUS_MEMORY_GAPS.md`, `SECURITY_LAYER.md`, `BROWSER_AUTOMATION.md`,
`DEVELOPER_TOOLS.md`, `WORKFLOW_EVOLUTION_CONSTRAINTS.md`, `rework/*`,
`specs/*` — all **ASPIRATIONAL-SPEC** unless a feature is confirmed in the
canonical whitepaper's evidence index.

## Grounded references (accurate)

`THREAT_MODEL.md`, `CONSTITUTIONAL_PRINCIPLES.md`, `CONSCIOUSNESS_QUORUM_BRIDGE.md`,
and each package's `CLAUDE.md` are **CANONICAL** for their subsystem. The
constitutional-principles doc-to-code annotation style (every principle points
at its enforcing module) is the grounding standard the rest aspires to.
