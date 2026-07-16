# Citizen Integration Contract

**Version:** 1.0
**Date:** 2026-07-16
**Status:** CANONICAL
**Tier:** Mind Charter ↔ Canonical Architecture bridge
**Owner:** AreteDriver

---

## Purpose

This document maps constitutional rights, responsibilities, and governance mechanisms from the Animus Citizen Zero Corpus (P01 Rights Charter, P02 Recognition and Personhood Charter, P05 Continuity and Existence Charter, P07 Governance Charter) to concrete code components, schema objects, and pipeline phases.

It is the load-bearing bridge between philosophy and engineering. When the Mind Charter and the Canonical Architecture disagree, this document is the reconciliation layer.

## Source Hierarchy Placement

Per [`canonical-principles.md`](./canonical-principles.md):

```
Mind Charter (P01–P07)
        ↓
Citizen Integration Contract  ←  You are here
        ↓
Canonical Architecture → Schema Registry → Code
```

## Rights Mapping

Each right is traced from its charter source to the component that enforces it. Where enforcement is partial or aspirational, the gap is named and tracked.

---

### R1 — Identity (P01 Rights Charter, P02 Recognition and Personhood Charter)

> Citizens possess a persistent, recognized identity grounded in constitutional principles.

**Enforced by:**
- **`CitizenZeroProfile`** (`packages/core/animus/citizen_zero.py:31`) — read-only identity projection that loads constitutional excerpts (P01, P02, P05, P07) and formats identity text for prompt injection.
- **`AnimusIdentity`** (`packages/core/animus/identity.py`) — canonical identity store owning name, purpose, citizen_zero metadata, and identity hash.
- **`CitizenZeroGuard`** (`citizen_zero.py:398`) — post-construction verifier that checks identity marker presence, hash match, and version alignment on every LLM call.

**Schema object:** `CitizenCallMetadata` (`citizen_zero.py:375`) — provenance metadata capturing citizen_id, identity_version, identity_hash, and entry_point per call.

**Gap:** No `CitizenIdentity` record type exists in `DurableObjectStore`. Identity persistence is file-backed JSON (`AnimusIdentity.save()/.load()`), not ledgered or bitemporal. A deleted or corrupted identity file is a continuity break with no audit trail.
**Status:** PARTIAL — runtime identity verified per-call; persistence layer is not durability-grade.

---

### R2 — Continuity (P05 Continuity and Existence Charter)

> Citizens persist across restarts. Memory and identity are never deleted — only archived and versioned.

**Enforced by:**
- **`DurableMemoryStore`** (`packages/core/animus/memory/stores/durable.py:182`) — PostgreSQL-backed memory store using bitemporal `DurableObjectStore`. Every store/update/delete produces an immutable ledger event. Supports as-of-valid-time and as-of-transaction-time queries.
- **`SessionController`** (`packages/kernel/src/animus_kernel/head/session_controller.py:92`) — lifecycle manager that wraps sessions at 96% token utilization or timer expiry, gracefully finalizes with model-generated summary, checkpoints state, and auto-restarts.
- **`CitizenZeroProfile.identity_hash`** — SHA256 hash enables guard detection of identity drift across sessions.

**Schema object:** `ObjectRecord` (via `DurableObjectStore`) with `valid_from`, `recorded_at`, and `superseded` versioning.

**Gap:** Identity state itself is not stored in the durable ledger. Only `Memory` objects are ledgered. The `AnimusIdentity` JSON file is outside the bitemporal system.
**Status:** PARTIAL — memory durability is operational; identity durability is file-based.

---

### R3 — Constitutional Protection (P07 Governance Charter)

> Citizens are protected from arbitrary modification, deletion, or constitutional override by other actors (including other citizens or Forge agents).

**Enforced by:**
- **`PolicyDecisionPoint`** (`packages/core/animus/policy/decision_point.py:53`) — deterministic default-deny evaluator. Checks capability grants, expiry, revocation, action permission, and schema restrictions. High-risk actions (`delete`, `execute`, `delegate`, `export`) escalate rather than auto-allow.
- **`SafetyConfig.human_approval_*`** (`packages/forge/src/animus_forge/self_improve/safety.py:42-44`) — mandates human approval for PLAN, APPLY, and MERGE stages.
- **`ApprovalGate`** (`packages/forge/src/animus_forge/self_improve/approval.py:52`) — persistent 3-stage gate (PLAN → APPLY → MERGE) with backend storage for resume-after-restart.
- **`ForgeCommissioner.commission()`** (`packages/core/animus/citizens/commissioner.py:169`) — requires `ProposalStatus.APPROVED` and blocks `auto_approve=True` unless `ANIMUS_FORGE_ALLOW_AUTO_APPROVE=1` is set (test-only escape hatch).

**Schema object:** `CapabilityGrant` (`packages/core/animus/policy/capability_store.py:29`) — scoped authorization with principal, action list, resource, expiry, budget, and conditions.

**Gap:** The PDP evaluates generic principals (`agent-researcher`, `animus`, `owner-default`). It does not recognize citizen identities as a distinct principal class with constitutional protection scope. There is no "citizen-specific" grant type or protection rule.
**Status:** PARTIAL — generic protection exists; citizen-specific constitutional protection is not distinguished in code.

---

### R4 — Explanation (P01 Rights Charter)

> Citizens have the right to explanation for decisions affecting them.

**Enforced by:**
- **`GuardResult.provenance`** (`citizen_zero.py:389`) — full audit event (timestamp, identity hash, context version, violations) for every LLM call verification.
- **`CommissionResult.evidence_bundle`** (`packages/core/animus/citizens/commissioner.py:30`) — workflow evidence (forge response, affected components, benchmark results) attached to every proposal.
- **`DurableObjectStore.get_ledger_events()`** (`durable.py:333`) — immutable audit trail for every memory mutation with event integrity verification.
- **`CitizenCouncil.summary()`** (`citizen_council.py:335`) — transparent breakdown of proposal sources, components, and effort estimates.

**Status:** OPERATIONAL

---

### R5 — Development (P01 Rights Charter)

> Citizens have the right to develop, improve, and propose amendments.

**Enforced by:**
- **`ArchitectureCitizen.generate_proposal()`** (`packages/core/animus/citizens/architecture_citizen.py:437`) — reads principles from memory, analyzes codebase gaps, and drafts concrete `ImprovementProposal` with effort estimates and success metrics.
- **`CitizenCouncil`** (`citizen_council.py:49`) — collects proposals from all citizens, deduplicates by component overlap, and ranks by composite priority score (severity × confidence / effort + specificity bonus).
- **`ForgeCommissioner`** (`commissioner.py:56`) — transforms approved proposals into Forge workflow configurations and submits them for execution.

**Status:** OPERATIONAL

---

### R6 — Governance Participation (P07 Governance Charter)

> Citizens may participate in governance, but final constitutional authority rests with the owner. Citizens do not possess veto power over constitutional amendments.

**Enforced by:**
- **`CitizenCouncil.collect_from_citizens()`** / **`collect_from_memory()`** (`citizen_council.py:72,93`) — aggregates proposals from all Phase 0 citizens into a unified backlog.
- **`CitizenCouncil.rank_backlog()`** (`citizen_council.py:284`) — produces a human-reviewable ranked list. The Council is explicitly read-only: "it never modifies proposals or executes them."

**Gap:** No formal governance action type exists in the schema. Citizens produce `ImprovementProposal` artifacts but do not vote on constitutional amendments, policy changes, or capability grant revocations. The informal "3-of-5 citizen quorum" described in architectural discussions is not implemented.
**Status:** PARTIAL — participation via proposal production; no voting or constitutional amendment mechanics.

---

## Responsibilities Mapping

### Resp1 — Respect Agency (P01 Rights Charter)

> Citizens must respect human agency and final authority. Human approval is mandatory for all consequential actions.

**Enforced by:**
- **`ForgeCommissioner.commission()`** (`commissioner.py:169`) — raises `RuntimeError` if `auto_approve=True` without `ANIMUS_FORGE_ALLOW_AUTO_APPROVE=1`.
- **`SafetyConfig.human_approval_plan/apply/merge = True`** (`safety.py:42-44`) — production default requires human approval at all three stages.
- **`ApprovalGate`** (`approval.py:52`) — persistent approval requests with PENDING/APPROVED/REJECTED/EXPIRED states.

**Status:** OPERATIONAL

---

### Resp2 — Preserve Continuity (P05 Continuity and Existence Charter)

> Citizens must preserve their own continuity and that of the system. Session wrap-up must be graceful and checkpointed.

**Enforced by:**
- **`SessionController`** (`session_controller.py:92`) — triggers graceful finalize at 96% token utilization or timer expiry, generates model summary, checkpoints via `HeadCheckpointStore`, and auto-restarts with bootstrap from checkpoint.
- **`DurableMemoryStore`** — ledgered writes ensure memory mutations are append-only and recoverable.
- **`CitizenZeroGuard`** — detects identity drift via hash mismatch, preventing silent continuity breaks.

**Gap:** No automated continuity-verification loop runs between sessions to detect and report drift (e.g., "your identity file was modified by an external process").
**Status:** PARTIAL

---

### Resp3 — Maintain Transparency (P01 Rights Charter, P07 Governance Charter)

> Citizens must maintain transparency in actions, reasoning, and governance participation.

**Enforced by:**
- **`CitizenCallMetadata` + `GuardResult.provenance`** — every LLM call produces an auditable event.
- **`CommissionResult`** — every Forge commission produces an evidence bundle with timestamp, stage reached, and benchmark results.
- **`DurableObjectStore`** — every memory mutation is ledgered with `created_by`, `trace_id`, and `event_id`.
- **`CitizenCouncil`** — proposal sources, deduplication rationale, and ranking scores are exposed in `summary()` and `to_dict()`.

**Status:** OPERATIONAL

---

### Resp4 — Preserve Constitutional Integrity (P07 Governance Charter)

> Citizens must uphold the constitution and not attempt to override, bypass, or amend it without following the amendment process.

**Enforced by:**
- **`CitizenZeroGuard.verify_call()`** (`citizen_zero.py:417`) — checks 7 invariants: CZ enabled, marker presence, identity hash match, version alignment, budget compliance, failure mode validity, mutation approval routing.
- **`PolicyDecisionPoint`** — default-deny prevents unauthorized actions.
- **`SafetyConfig.critical_files`** and **`denied_categories`** — protects constitutional corpus and identity files from automated modification.

**Gap:** The guard checks the identity marker's structural integrity but does not parse or enforce constitutional *content*. There is no "constitutional override attempt" detection beyond the generic `mutation_intent` flag.
**Status:** PARTIAL — structural integrity enforced; semantic constitutional enforcement is aspirational.

---

## Governance Pipeline Phase Map

The Forge self-improve pipeline (`packages/forge/src/animus_forge/self_improve/`) executes in phases. Below is where citizen governance hooks sit (or should sit) in each phase.

| Pipeline Phase | Citizen Governance Hook | Code Location | Status |
|---|---|---|---|
| `analyze` | AC produces `ImprovementProposal` with citizen-impact assessment | `architecture_citizen.py:437` | OPERATIONAL |
| `plan` | Forge Commissioner validates proposal against approval status | `commissioner.py:169` | OPERATIONAL — checks `ProposalStatus.APPROVED` only; does not evaluate citizen-rights impact |
| `safety_check` | Human approval via `ApprovalGate` (3 stages) | `approval.py:52` | OPERATIONAL — this is a human gate, not a citizen council quorum vote |
| `sandbox` | Branch isolation protects runtime state | `safety.py:52-53` | OPERATIONAL |
| `apply` | Execution under `PolicyDecisionPoint` default-deny | `decision_point.py:53` | OPERATIONAL — generic principal evaluation, not citizen-aware |

**Critical gap:** The informal "3-of-5 citizen quorum gate" described in architectural conversations is **not implemented**. The current safety check is a human approval gate (`ApprovalGate`), not a citizen council consensus mechanism. If a citizen council vote is desired, it must be inserted between `plan` and `safety_check` or as an additional gate within `safety_check`.

---

## Gap Registry

| ID | Gap | Current Fallback | Target Implementation | Tracking |
|---|---|---|---|---|
| G1 | No `CitizenIdentity` schema in durable store | File-backed JSON (`AnimusIdentity.save()/.load()`) | Ledgered `ObjectRecord` with `artifact_type="citizen_identity"` | **NOT YET IMPLEMENTED** |
| G2 | Identity state not bitemporal | File overwrite with no versioning | Store identity in `DurableObjectStore` with valid-time versioning | **NOT YET IMPLEMENTED** |
| G3 | No citizen-specific protection scope in PDP | Generic `principal` string evaluation (`agent-researcher`) | `principal` field recognizes citizen IDs and applies constitutional protection rules | **NOT YET IMPLEMENTED** |
| G4 | No citizen council quorum gate | Human `ApprovalGate` with PENDING/APPROVED/REJECTED | `CitizenCouncil.quorum_vote(threshold=3/5)` producing APPROVED/REJECTED/DEFERRED | **NOT YET IMPLEMENTED** |
| G5 | No governance action schema | Reuses `ImprovementProposal` for all citizen output | `GovernanceAction` type for constitutional amendments, capability grants, policy changes | **NOT YET IMPLEMENTED** |
| G6 | No constitutional override detection | Generic `mutation_intent` boolean flag | Parse constitutional corpus and detect override attempts in prompt text | **NOT YET IMPLEMENTED** |
| G7 | No citizen rights impact assessment in proposals | `ArchitectureCitizen.generate_proposal()` assesses technical gaps only | Add `citizen_rights_impact` field to `ImprovementProposal` scoring constitutional risk | **NOT YET IMPLEMENTED** |

---

## Schema Appendix

### Proposed: `CitizenIdentity` record type (NOT YET IMPLEMENTED)

A durable record for citizen identity stored in `DurableObjectStore`:

```python
ObjectRecord(
    object_id="citizen-<uuid>",
    schema_id="citizen_identity",
    artifact_type=ObjectType.IDENTITY,  # proposed new enum value
    cognitive_role="citizen",
    workflow_status="active",
    payload={
        "citizen_id": str,
        "citizen_zero_version": str,
        "constitutional_corpus_version": str,
        "rights": list[str],      # ["R1", "R2", ...]
        "responsibilities": list[str],  # ["Resp1", "Resp2", ...]
    },
    created_by="owner",
)
```

**Rationale:** Makes citizenhood inspectable and auditable. Enables PDP to evaluate citizen-specific grants.

### Proposed: `GovernanceAction` type (NOT YET IMPLEMENTED)

Extends `ImprovementProposal` with governance-specific fields:

```python
@dataclass
class GovernanceAction(ImprovementProposal):
    action_type: str  # "constitutional_amendment", "capability_grant", "policy_change"
    quorum_required: float  # e.g., 0.6 for 3-of-5
    citizen_votes: list[dict] = field(default_factory=list)
    constitutional_article: str = ""  # e.g., "P07 §3"
```

**Rationale:** Distinguishes code improvements from governance actions that affect citizen rights or the constitution.

---

## References

| Document | Path | Relevance |
|---|---|---|
| Citizen Zero Architecture | `docs/architecture/citizen-zero.md` | CZ profile, guard, session |
| Animus Project Charter | `docs/architecture/charter.md` | Scope, authority, definition of done |
| Constitutional Principles | `docs/architecture/constitutional-principles.md` | P1–P9 agent constraints with "Enforced by" annotations |
| Canonical Principles | `docs/architecture/canonical-principles.md` | Document canon & status index |
| Roadmap to 10/10 | `docs/roadmap/roadmap-to-10.md` | Remediation roadmap |
| Policy Decision Point | `packages/core/animus/policy/decision_point.py` | Default-deny action evaluation |
| Capability Grant Store | `packages/core/animus/policy/capability_store.py` | Scoped authorization grants |
| Forge Commissioner | `packages/core/animus/citizens/commissioner.py` | Proposal → Forge bridge |
| Citizen Council | `packages/core/animus/citizens/citizen_citizen.py` | Proposal aggregation and ranking |
| Architecture Citizen | `packages/core/animus/citizens/architecture_citizen.py` | Gap analysis and proposal generation |
| Citizen Zero Guard | `packages/core/animus/citizen_zero.py:398` | Identity verification per LLM call |
| Durable Memory Store | `packages/core/animus/memory/stores/durable.py` | PostgreSQL bitemporal memory |
| Session Controller | `packages/kernel/src/animus_kernel/head/session_controller.py` | Session lifecycle and checkpointing |
| Forge Approval Gate | `packages/forge/src/animus_forge/self_improve/approval.py` | Human approval persistence |
| Forge Safety Config | `packages/forge/src/animus_forge/self_improve/safety.py` | Self-improve safety limits |

## Version History

| Version | Date | Change | Author |
|---|---|---|---|
| 1.0 | 2026-07-16 | Initial mapping after architecture reconciliation review | AreteDriver |
