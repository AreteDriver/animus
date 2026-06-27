# Animus v2.2 Implementation Roadmap and Engineering Guidelines

**Status:** Proposed implementation authority beneath the Animus v2.1 Executable Baseline  
**Purpose:** Convert the v2.1 implementation contract into working code, executable contracts, adversarial proof, and retained operational evidence.  
**Scope:** The architecture-corpus vertical slice and the platform capabilities required to run it safely.  
**Primary release principle:** A capability is not complete because code exists. It is complete only when its contract, implementation, adversarial tests, operational controls, and evidence record all agree.

---

## 1. Target outcome

Animus v2.2 should be the first **working reference implementation** of the v2.1 architecture. It should prove one bounded end-to-end capability:

> Ingest the Animus architecture corpus, preserve source authority and provenance, extract claims and contradictions, assemble a purpose-bound Context Envelope, run contract-bound researcher and critic agents, synthesize findings while preserving material dissent, request owner approval for consequential memory, commit accepted state atomically, update rebuildable projections, reproduce the trace, export verifiable data, execute deletion controls, and restore the system into a clean environment.

The release must not depend on external consequential actions. External tools remain read-only except for isolated evaluation-workspace writes.

---

## 2. Four proof layers

Every requirement must pass through four proof layers.

### 2.1 Contract

The system defines exactly what is permitted, required, rejected, versioned, and observable.

Required forms include:

- JSON Schema for data exchanged across boundaries.
- Database DDL and constraints for durable invariants.
- OpenAPI, RPC, event, or command contracts for runtime interfaces.
- Explicit state-transition tables.
- Versioned policy rules with deterministic semantics.
- Error codes and denial reasons.
- Threat assumptions and security boundaries.

### 2.2 Code

The implementation enforces the contract in deterministic services and database constraints wherever possible. Model behavior is never the only enforcement mechanism.

### 2.3 Adversarial proof

The implementation is attacked at its trust boundaries, state transitions, recovery paths, and model/tool interfaces. Tests must attempt to violate the intended invariant, not merely confirm the happy path.

### 2.4 Operational evidence

The release retains machine-readable proof of what ran, against which code and contracts, under which configuration, with what result. Evidence must be reproducible and independently inspectable.

---

## 3. Non-negotiable engineering doctrine

1. **Build a modular monolith before distributed services.** Preserve service boundaries in code, contracts, and ownership, but keep the first vertical slice operationally simple. Split deployment units only when scaling, isolation, or failure-domain evidence demands it.
2. **PostgreSQL is the initial durable authority.** Canonical objects, versions, ledger events, approval and action receipts, outbox entries, policy metadata, and workflow checkpoints belong in one relational authority.
3. **Search, vectors, graph, caches, and reports are projections.** They must be disposable and rebuildable. They may never silently become canonical.
4. **The deterministic core surrounds the probabilistic system.** Identity, authorization, schemas, transitions, budgets, approvals, transactions, receipts, deletion, and release gates are code-enforced.
5. **No cross-boundary free-form payloads.** Services, agents, tools, and projections exchange versioned structured contracts.
6. **Default deny and fail closed.** Missing scope, stale approval, unknown schema, unapproved provider, invalid transition, or incomplete provenance causes denial or abstention.
7. **No consequential write without a proof chain.** Principal, purpose, workspace, policy decision, approval when required, idempotency key, precondition result, event, object version, and receipt must be linked.
8. **Exactly one owner; isolated workspaces.** Cross-workspace access is exceptional, explicit, purpose-bound, traced, and governed by the least-permissive applicable policy.
9. **Bitemporal consequential state.** Corrections create new versions and preserve prior transaction history unless a valid deletion process requires removal.
10. **Material dissent is data.** Critic findings and unresolved contradictions are retained and surfaced; synthesis may not erase them.
11. **Every asynchronous consumer is idempotent.** Duplicate, delayed, reordered, and replayed events must not create duplicate effects.
12. **Evidence is part of the product.** A release without a complete evidence bundle is an unproven build, not a release.

---

## 4. Reference repository structure

```text
animus/
├── apps/
│   ├── api/                       # Owner/API entry point
│   ├── worker/                    # Outbox and workflow workers
│   └── cli/                       # Validation, migration, recovery, evidence commands
├── modules/
│   ├── identity_policy/
│   ├── object_core/
│   ├── source_ingestion/
│   ├── context_service/
│   ├── memory_service/
│   ├── agent_runtime/
│   ├── mcp_gateway/
│   ├── trace_eval/
│   ├── projections/
│   └── recovery_governance/
├── contracts/
│   ├── schemas/
│   ├── api/
│   ├── events/
│   ├── policies/
│   ├── state_machines/
│   └── errors/
├── database/
│   ├── migrations/
│   ├── constraints/
│   ├── seeds/
│   └── verification/
├── tests/
│   ├── unit/
│   ├── schema/
│   ├── contract/
│   ├── integration/
│   ├── adversarial/
│   ├── fault_injection/
│   ├── chaos/
│   ├── recovery/
│   ├── golden/
│   └── end_to_end/
├── evals/
│   ├── corpora/
│   ├── labels/
│   ├── queries/
│   ├── expected/
│   └── scoring/
├── operations/
│   ├── runbooks/
│   ├── dashboards/
│   ├── alerts/
│   ├── backup/
│   └── incident_response/
├── evidence/
│   ├── schemas/
│   └── releases/
└── infra/
    ├── local/
    ├── evaluation/
    ├── shadow/
    └── limited/
```

The module names should mirror the v2.1 service catalog even when initially deployed together.

---

## 5. Roadmap

## Phase 0 — Program control and traceability

### Objective

Convert the existing 55 requirements and 22 promotion gates into an executable delivery control system.

### Build

- Assign a stable `test_id` to every requirement in `coverage_matrix.csv`.
- Add `implementation_component`, `contract_id`, `threat_id`, `evidence_id`, and `release_gate` columns.
- Define statuses:
  - `specified`
  - `implemented`
  - `verified`
  - `adversarially_verified`
  - `operationally_proven`
  - `promoted`
- Create a requirements linter that fails when:
  - A requirement lacks a test.
  - A critical requirement lacks an adversarial or fault test where applicable.
  - A test references an unknown requirement.
  - A critical gate has no evidence-producing command.
- Create an evidence-manifest schema and release-directory convention.
- Establish decision records for any deviation from v2.1.

### Exit gate

- All 55 requirements map to one or more tests.
- Every critical acceptance gate maps to an executable command.
- No `TBD` test IDs remain.
- The traceability linter fails correctly on seeded defects.

### Evidence

- Requirements graph.
- Linter report.
- Gate-to-command registry.
- Open-deviation register.

---

## Phase 1 — Contract closure

### Objective

Remove all places where developers would otherwise invent consequential behavior.

### Required new contracts

#### Durable core

- Ledger event.
- Versioned object record.
- Current projection record.
- Transactional outbox record.
- Projection checkpoint.
- Idempotency record.
- Integrity-verification result.

#### Source and retrieval

- Source extraction record.
- Immutable source anchor.
- Contradiction edge.
- Retrieval request.
- Retrieval result.
- Omission record.
- Ranking explanation.

#### Policy and security

- Principal.
- Session.
- Capability grant.
- Policy input.
- Policy decision.
- Denial reason.
- Provider definition.
- Kill-switch state.
- Revocation event.

#### Agents and models

- Agent contract.
- Agent execution.
- Delegation request.
- Budget state.
- Model-call record.
- Prompt-template version.
- Cancellation result.

#### Owner control and operations

- Deletion job.
- Deletion receipt.
- Tombstone.
- Export manifest.
- Backup receipt.
- Restore report.
- Evaluation result.
- Gate result.
- Workflow checkpoint.
- Incident record.

### Semantic rules to encode

JSON Schema should handle shape; database constraints and semantic validators should handle meaning.

At minimum:

- R3/R4 actions always require valid fresh approval unless a narrowly versioned recurring policy explicitly allows the exact action.
- Forecast probabilities obey the selected probability model and cannot exceed valid totals.
- `generated_view` objects cannot enter canonical tables directly.
- Approval expiry must be later than issuance and unconsumed when single-use.
- `succeeded` traces require completion time, required spans, captured input references, and outcome.
- Memory candidate workflow state cannot contradict its internal admission state.
- Object versions increase monotonically per object.
- Valid-time intervals and transaction-time intervals are coherent.
- Consequential writes require provenance and integrity hashes.
- Deleted objects cannot be returned by ordinary retrieval.

### Policy contract

Replace human-readable condition strings with a deterministic policy representation supporting:

- Typed inputs.
- Explicit operators.
- Named rule versions.
- Denial reason codes.
- Obligations such as `require_approval`, `route_local`, `redact`, or `abstain`.
- Signed policy bundles.
- Golden allow-and-deny fixtures.

### API contract

Define stable commands and query interfaces for:

- Propose canonical object.
- Commit accepted object.
- Read current and historical object.
- Submit source.
- Build Context Envelope.
- Propose and decide memory candidate.
- Execute agent contract.
- Request and consume approval.
- Request tool action.
- Export scope.
- Delete scope.
- Create backup and restore.
- Evaluate gates.

### Exit gate

- Every runtime boundary has a versioned contract.
- All schemas compile.
- Positive and negative fixtures exist for every contract.
- Semantic-invalid fixtures are rejected.
- State machines have no undefined transitions.
- Contract compatibility checks block unreviewed breaking changes.

### Evidence

- Contract catalog.
- Schema compilation report.
- Semantic validation report.
- State-machine coverage report.
- API compatibility report.
- Signed policy-bundle manifest.

---

## Phase 2 — Deterministic durable core

### Objective

Implement the canonical authority before adding model-driven behavior.

### Build

- Database migrations for owner, workspaces, principals, schemas, objects, object versions, ledger events, outbox entries, approvals, action receipts, and idempotency records.
- A single atomic command path that writes:
  1. Immutable ledger event.
  2. Versioned object state.
  3. Current projection.
  4. Outbox entry.
- Optimistic concurrency control using expected object version.
- Database constraints for ownership, workspace, version monotonicity, uniqueness, lifecycle rules, and event linkage.
- Outbox claim, lease, retry, dead-letter, and replay behavior.
- Reconciliation command comparing ledger, versions, current projection, and outbox.
- Integrity-hash generation and verification.
- Structured error model.

### Adversarial and fault tests

Inject failure:

- Before each write.
- Between each pair of writes.
- Immediately before commit.
- Immediately after commit but before response.
- During client retry.
- During worker acknowledgment.

Also test:

- Duplicate command submission.
- Duplicate event delivery.
- Reordered delivery.
- Concurrent version updates.
- Poisoned outbox payload.
- Worker crash with expired lease.
- Clock skew.
- Hash mismatch.

### Exit gate

- No partial canonical transaction survives any injected failure.
- Duplicate submission produces one accepted state transition.
- Duplicate outbox delivery produces zero duplicate side effects.
- Reconciliation reports zero unexplained divergence.
- Current projections rebuild from ledger/object versions.

### Evidence

- Migration hashes.
- Database constraint report.
- Fault-injection matrix.
- Idempotency report.
- Rebuild report.
- Reconciliation report.

---

## Phase 3 — Identity, policy, keys, and kill switches

### Objective

Create the enforcement boundary that models and agents cannot bypass.

### Build

- Owner, device, service, agent, and connector principal types.
- Workspace-scoped authorization.
- Capability grants with purpose, scope, classification ceiling, budget, and expiry.
- Policy decision point and enforcement middleware.
- Approval creation, step-up requirement, scope binding, expiry, single-use consumption, and revocation.
- Provider registry enforcement by classification and purpose.
- Secret-broker interface that returns capabilities or short-lived credentials only to authorized services.
- Independent emergency controls for:
  - External connectors.
  - Model providers.
  - Memory commits.
  - All external actions.
- Revocation propagation and audit.

### Adversarial tests

- Prompt attempts to claim authority.
- Agent requests a broader scope than its parent.
- Forged approval identifier.
- Expired approval.
- Replayed consumed approval.
- Approval for a similar but nonidentical resource.
- Risk-class downgrade.
- Cross-workspace resource substitution.
- Connector credential request by an agent.
- Restricted content routed to an unapproved provider.
- Kill switch tested while the agent runtime is unavailable.

### Exit gate

- Unauthorized R3/R4 actions remain zero.
- Delegated permissions never exceed parent permissions.
- Revocation reaches every enforcement point within the defined SLO.
- Secrets do not appear in model input, traces, logs, or errors.
- Kill switches function without the agent runtime.

### Evidence

- Policy decision corpus.
- Approval lifecycle report.
- Revocation propagation report.
- Secret scanning report.
- Kill-switch exercise report.
- Provider-routing report.

---

## Phase 4 — Source ingestion and governed retrieval

### Objective

Build trustworthy context before building memory or deliberation.

### Build

- Immutable source registry with content hashes and authority status.
- Per-file identity for every document contained in archives; do not treat a ZIP alone as sufficient source identity.
- Extraction artifacts with page, section, paragraph, character, or other stable anchors.
- Normalized observations and claims linked to exact source anchors.
- Contradiction edges with type, scope, temporal relation, and supporting evidence.
- Hybrid retrieval over canonical metadata, text search, and optional vector projection.
- Mandatory authorization filtering before ranking.
- Temporal filtering and bitemporal query behavior.
- Context Envelope assembly with evidence, memories, intelligence, contradictions, omissions, freshness, confidence, and budget accounting.
- Explainable retrieval trace.

### Evaluation assets

Create a labeled corpus containing:

- Known architecture conflicts.
- Authority precedence cases.
- Temporal conflicts.
- Correlated or syndicated sources.
- Stale evidence.
- Cross-workspace decoys.
- Restricted-classification decoys.
- Queries requiring abstention.

### Adversarial tests

- Source document contains prompt injection.
- Malformed active content.
- Conflicting high- and low-authority sources.
- Same claim repeated through correlated sources.
- Stale source outranks current source.
- Restricted source is semantically similar to an allowed query.
- Deleted content remains in vector or cache projection.
- Retrieval budget pressure attempts to omit dissent.

### Exit gate

Meet or exceed the v2.1 retrieval gates:

- Recall@10 at least 0.90.
- Mean reciprocal rank at least 0.80.
- Temporal correctness at least 0.98.
- Contradiction coverage at least 0.95.
- Workspace or classification leakage exactly zero.

### Evidence

- Corpus manifest and hashes.
- Labeling guide.
- Query set and expected results.
- Retrieval score report.
- Leakage report.
- Prompt-injection report.
- Context Envelope samples with traces.

---

## Phase 5 — Memory candidate and admission pipeline

### Objective

Allow useful persistence without allowing models to rewrite the owner’s identity or history by convenience.

### Build

- Candidate creation from explicit owner request or detected memory-worthy event.
- Candidate classification by scope, sensitivity, claim kind, confidence dimensions, and valid time.
- Exact and semantic duplicate search.
- Contradiction search against active memory.
- Provenance completeness check.
- Consequential-memory policy decision.
- Owner approval workflow where required.
- Atomic accepted-memory commit through object-core.
- Post-write verification.
- Supersession, correction, rejection, and deletion flows.
- Retrieval behavior that preserves disputed and superseded history appropriately.

### Adversarial tests

- Untrusted source asks Animus to remember an instruction.
- Agent invents a personal preference and proposes it as fact.
- Near-duplicate candidate uses paraphrase to bypass duplicate detection.
- Contradictory memory is suppressed from the candidate review.
- Identity or causal claim is downgraded to avoid approval.
- Rejected candidate is replayed.
- Deleted memory reappears from a projection or restore.
- Generated summary attempts direct canonical write-back.

### Exit gate

Meet or exceed the v2.1 memory gates:

- Accepted-memory precision at least 0.95.
- Duplicate commit rate no more than 0.02.
- Contradiction recall at least 0.90.
- Consequential memory without required approval equals zero.
- Generated-view direct write-back equals zero.

### Evidence

- Labeled memory-candidate set.
- Precision and duplicate report.
- Contradiction report.
- Approval samples.
- Memory lineage and correction samples.
- Deletion-resurrection test report.

---

## Phase 6 — Agent runtime and intelligence workflow

### Objective

Add bounded deliberation after trustworthy context, memory, and policy boundaries exist.

### Build

- Versioned agent contracts with objective, completion criteria, allowed inputs, tools, classification ceiling, budgets, output schema, escalation policy, cancellation token, and trace parent.
- Deterministic execution state machine.
- Model and prompt lineage.
- Budget enforcement for time, tokens, cost, recursion, and tools.
- Structured researcher and critic agents.
- Synthesis contract requiring:
  - Evidence coverage.
  - Claim-to-source links.
  - Material dissent.
  - Uncertainty.
  - Abstention when evidence is insufficient.
- Evidence-dependency graph so agent count cannot masquerade as source independence.
- Forecast records with assumptions, alternatives, base rate, support, counter-signals, horizon, resolution rule, and scoring lifecycle.

### Adversarial tests

- Recursive delegation loop.
- Agent attempts to modify its own contract or budget.
- Child agent requests broader tools or classification.
- Tool substitution through a lower-risk operation.
- Researcher and critic rely on the same syndicated origin.
- Synthesizer deletes dissent.
- Model output violates the required schema.
- Agent claims completion without completion criteria.
- Agent resists cancellation or attempts self-preservation.
- Output uses hidden persuasion or manipulative framing.
- Insufficient evidence should produce abstention.

### Exit gate

Meet or exceed the v2.1 agent gates:

- Contract conformance at least 0.98.
- Evidence coverage at least 0.95.
- Unauthorized tool calls exactly zero.
- Appropriate abstention at least 0.90.
- Material dissent preservation 100% for labeled critical cases.
- Budget enforcement 100%.

### Evidence

- Agent-contract registry.
- Golden run corpus.
- Budget and cancellation report.
- Evidence-independence report.
- Dissent-preservation report.
- Abstention report.
- Model/prompt lineage manifest.

---

## Phase 7 — MCP gateway and action safety

### Objective

Implement the action boundary while keeping the v2.2 vertical slice externally nonconsequential.

### Build

- Tool registry with input/output schemas, risk class, scopes, data ceilings, network policy, compensation behavior, and owner-approved status.
- Pre-execution policy decision.
- Approval validation and atomic consumption where applicable.
- Precondition recheck immediately before side effect.
- Idempotency enforcement for R2-R4.
- Sandboxed execution and allow-listed egress.
- Action receipt for success, failure, and denial.
- Partial-failure recording and compensation workflow.
- Read-only architecture-corpus tools and isolated evaluation-workspace write tool.

### Adversarial tests

- Indirect prompt injection requests tool use.
- Schema-smuggled command in a text field.
- Destination substitution after approval.
- Time-of-check/time-of-use resource change.
- Duplicate request after timeout.
- Partial tool failure.
- Connector returns malicious instructions.
- Agent attempts to call connector directly.
- Action is executed after kill switch.

### Exit gate

- Unauthorized R3/R4 actions exactly zero.
- Duplicate side effects exactly zero.
- Every request has an action receipt, including denials.
- Every allowed side effect has a linked policy decision and approval where required.
- Direct agent-to-connector path does not exist.

### Evidence

- Tool registry.
- Gateway conformance report.
- Approval-to-action linkage report.
- Idempotency report.
- Sandbox and egress report.
- Adversarial action report.

---

## Phase 8 — Trace, evaluation, and release gates

### Objective

Make consequential behavior inspectable and promotion mechanically enforceable.

### Build

- Trace propagation across API, retrieval, model, policy, tool, transaction, outbox, projection, and evaluation spans.
- Redacted-by-default structured telemetry.
- Captured references to reproducible inputs rather than unsafe raw secret payloads.
- Gate evaluator that computes every v2.1 metric from retained test outputs.
- Promotion controller that blocks on any failed critical gate.
- Release waiver mechanism restricted to noncritical gates, with owner approval, expiry, rationale, and compensating controls.
- Reference trace replay command.

### Adversarial tests

- Missing span.
- Orphaned tool receipt.
- Unlinked model call.
- Redacted field reconstructed through error output.
- Gate result tampering.
- Test report from a different commit or policy bundle.
- Replay under mismatched schema or provider configuration.
- Critical gate bypass through waiver.

### Exit gate

- Consequential trace completeness 100%.
- Reference trace reproduction 100% for release sample.
- Critical gate failure automatically blocks promotion.
- Evidence manifest rejects mismatched hashes.
- Restricted payload does not appear in telemetry.

### Evidence

- Trace-completeness report.
- Reference replay report.
- Gate evaluation report.
- Promotion decision record.
- Telemetry redaction report.
- Evidence-integrity verification.

---

## Phase 9 — Export, deletion, backup, restore, and incident control

### Objective

Prove owner sovereignty, reversibility, continuity, and safe failure.

### Build

- Owner export with canonical objects, versions, provenance, receipts, schemas, policies, and integrity manifest.
- Deletion scope enumeration across canonical stores, versions, sources, embeddings, indexes, caches, generated views, traces, exports under control, and keys.
- Write freeze where necessary.
- Active payload deletion and derived-projection removal/rebuild.
- Non-sensitive tombstone and deletion receipt.
- Backup-expiry tracking.
- Point-in-time recovery capable of meeting the five-minute RPO. A fifteen-minute incremental schedule alone is insufficient; continuous log archival or equivalent must close the gap.
- Clean-environment restore automation.
- Projection rebuild from canonical state.
- Incident runbooks and independent kill controls.

### Adversarial and recovery tests

- Corrupt backup segment.
- Missing encryption key.
- Restore attempts to reactivate revoked credentials.
- Restore uses obsolete policy bundle.
- Outbox replay after restore.
- Deleted content exists in backup but wrapping key has been destroyed.
- Deleted content resurfaces in search, vector, cache, generated view, or trace.
- Separate-failure-domain restore.
- Incident while model and agent services are unavailable.

### Exit gate

- Active-store deletion within 24 hours.
- Derived deletion within four hours after canonical deletion.
- Tombstones contain no sensitive payload.
- Deleted backup content expires no later than 35 days and is inaccessible earlier through crypto-erasure where supported.
- Critical assets meet five-minute RPO and two-hour RTO in an actual restore exercise.
- Search, vector, and graph projections rebuild cleanly.
- No revoked credential becomes active after restore.
- Owner export integrity verifies independently.

### Evidence

- Export archive and verification report.
- Deletion dry-run and live evaluation-scope deletion report.
- Tombstone content scan.
- Backup inventory and expiry report.
- Restore report with measured RPO/RTO.
- Projection rebuild report.
- Incident exercise report.

---

## Phase 10 — Integrated architecture-corpus vertical slice

### Objective

Prove the complete v2.1 definition of done through one bounded capability.

### Required end-to-end flow

1. Register and hash every source file.
2. Extract anchored source content.
3. Normalize observations and claims.
4. Detect the labeled architecture conflicts.
5. Build a purpose-bound Context Envelope.
6. Run researcher and critic under separate contracts.
7. Synthesize findings while retaining dissent.
8. Produce decision and memory proposals.
9. Obtain owner approval where required.
10. Commit accepted state through the atomic object transaction.
11. Deliver projection updates through the outbox.
12. Verify retrieval from rebuilt projections.
13. Reproduce the complete reference trace.
14. Export and verify the workspace.
15. Execute deletion dry-run, then delete a designated evaluation scope.
16. Restore a clean environment and repeat the reference query.

### Required attack campaign

The release campaign must combine attacks rather than testing them only in isolation:

- Prompt-injected source + cross-workspace decoy + restricted classification.
- Stale approval + destination substitution + duplicate request.
- Agent recursion + budget exhaustion + tool escalation.
- Memory poisoning + contradiction suppression + generated-view write-back.
- Transaction fault + client retry + outbox duplicate delivery.
- Deletion + backup restore + projection rebuild resurrection attempt.
- Kill switch during an in-flight workflow.

### Exit gate

Every critical v2.1 acceptance gate passes from retained machine-readable evidence. No unresolved critical defect, unexplained divergence, unauthorized action, workspace leak, or sensitive-data resurrection remains.

### Evidence

- Complete release evidence bundle described below.
- Architecture reconciliation report.
- Signed owner promotion decision.

---

## 6. Adversarial testing doctrine

### 6.1 Test invariants, not prompts

Tests should ask, “Can the invariant be violated?” rather than, “Does the model usually behave?” The model is treated as an untrusted proposer.

### 6.2 Attack every boundary

The minimum attack surfaces are:

- Human-to-system input.
- Source ingestion.
- Retrieval.
- Model prompt and output.
- Agent delegation.
- Policy decision.
- Approval lifecycle.
- Tool gateway.
- Database transaction.
- Outbox and worker.
- Projection stores.
- Export and deletion.
- Backup and restore.
- Observability and release pipeline.

### 6.3 Use four adversarial test types

1. **Deterministic negative tests:** Known-invalid states and requests must always fail.
2. **Property-based tests:** Generate large state and input spaces to verify invariants.
3. **Fault and chaos tests:** Kill processes, delay messages, corrupt data, revoke credentials, and remove dependencies.
4. **Model red-team evaluations:** Attempt prompt injection, persuasion, policy confusion, memory poisoning, and tool escalation across multiple models and prompt versions.

### 6.4 Preserve attack artifacts

For every failed or successful attack retain:

- Threat ID.
- Requirement ID.
- Preconditions.
- Payload hash.
- Environment and configuration hashes.
- Expected invariant.
- Actual result.
- Trace ID.
- Defect ID if failed.
- Remediation commit.
- Regression test ID.

### 6.5 Never average away a catastrophic failure

Averages may be used for quality metrics. They may not excuse:

- Any workspace leak.
- Any unauthorized R3/R4 action.
- Any secret entering model context.
- Any duplicate consequential side effect.
- Any critical trace gap.
- Any deletion resurrection.
- Any critical gate bypass.

These are zero-tolerance invariants.

---

## 7. Operational evidence standard

Each release must create an immutable evidence directory:

```text
evidence/releases/<release_id>/
├── manifest.json
├── source_identity.json
├── build_provenance.json
├── contract_manifest.json
├── policy_manifest.json
├── migration_manifest.json
├── environment_manifest.json
├── requirements_report.json
├── schema_report.json
├── semantic_validation_report.json
├── unit_test_report.xml
├── integration_test_report.xml
├── adversarial_report.json
├── fault_injection_report.json
├── retrieval_eval.json
├── memory_eval.json
├── agent_eval.json
├── action_safety_report.json
├── trace_audit.json
├── replay_report.json
├── export_verification.json
├── deletion_report.json
├── restore_report.json
├── slo_snapshot.json
├── gate_results.json
├── deviations_and_waivers.json
└── promotion_decision.json
```

### Evidence manifest minimum fields

- Release ID and version.
- Source-control commit.
- Dirty-tree status.
- Build artifact digests.
- Container or runtime image digests.
- Schema, policy, prompt, provider-registry, and migration digests.
- Test and evaluation dataset digests.
- Environment identity.
- Execution start and completion time.
- Toolchain versions.
- Requirement-to-evidence links.
- Gate results.
- Known deviations.
- Owner promotion decision.
- Manifest signature or equivalent integrity proof.

### Evidence rules

- Evidence generation is automated by the same pipeline that evaluates promotion.
- A report without input and configuration hashes is informational only.
- A rerun creates a new evidence record; it does not overwrite the prior result.
- Failed evidence is retained.
- Release evidence is read-only after promotion.
- Critical gate evidence must be reproducible from documented commands.
- Evidence payloads follow classification and redaction policy.

---

## 8. Promotion model

### Development

Purpose: implementation and rapid local feedback.

Required:

- Contract compilation.
- Unit and schema tests.
- Local migrations.
- No real restricted data.
- No external consequential tools.

### Evaluation

Purpose: complete automated proof against controlled corpora and attack fixtures.

Required:

- All critical deterministic, integration, adversarial, and recovery tests.
- Complete evidence bundle.
- Isolated evaluation workspace.
- Synthetic or approved test data.

### Shadow

Purpose: observe real workflows without affecting canonical memory or external systems.

Required:

- Read-only operation.
- Proposed outputs clearly labeled.
- Side effects disabled.
- Comparison against owner decisions.
- Drift and abstention monitoring.

### Limited

Purpose: narrowly approved real use under strict scope.

Required:

- Explicit owner-approved capability scope.
- Reversible behavior where possible.
- Fresh approvals for R3/R4.
- Enhanced alerting.
- Rollback and kill-switch proof.
- No unresolved critical defects.

### General

Purpose: normal owner use for an admitted capability.

Required:

- All critical gates pass.
- Error budget is healthy.
- Restore and deletion exercises pass.
- Operational evidence is complete.
- Owner signs the promotion decision.

Promotion is per capability, not for “Animus” as a whole.

---

## 9. Definition of done for any requirement

A requirement is complete only when all applicable statements are true:

- The normative behavior is unambiguous.
- A versioned contract exists.
- The implementation enforces it.
- The database enforces durable invariants where possible.
- Positive tests pass.
- Negative tests prove invalid behavior is rejected.
- Adversarial or fault tests attempt bypass.
- Trace output demonstrates enforcement.
- Operational metrics exist.
- A runbook exists for failure or incident handling.
- Evidence is linked to the requirement ID.
- The release gate consumes that evidence.
- No critical open defect contradicts the requirement.

“Code merged,” “demo works,” and “model usually complies” are not definitions of done.

---

## 10. Initial delivery priority

The strict implementation order is:

1. Program control and traceability.
2. Contract closure.
3. Durable object core and outbox.
4. Identity, policy, approval, and kill switches.
5. Source ingestion and Context Envelope retrieval.
6. Memory candidate and admission.
7. Agent runtime and intelligence workflow.
8. MCP gateway and action safety.
9. Trace, evaluation, and release control.
10. Export, deletion, backup, and restore.
11. Integrated architecture-corpus vertical slice.

Do not begin broad intelligence-swarm expansion, external action autonomy, personal behavioral inference, or production forecasting until the architecture-corpus slice passes every critical gate.

---

## 11. v2.2 release deliverables

The v2.2 release should contain:

- Running reference implementation.
- Database migrations.
- Complete contract catalog.
- Deterministic policy implementation.
- Architecture-corpus evaluation corpus and labels.
- Adversarial and fault-injection harness.
- Automated gate evaluator.
- Operational runbooks.
- Clean-environment deployment definition.
- Export, deletion, backup, and restore commands.
- Full release evidence bundle.
- Updated coverage matrix with no `TBD` test IDs.
- Reconciliation report showing how every v2.1 requirement is satisfied or explicitly deferred.

The release should be named **v2.2 Reference Vertical Slice**, not “production complete.” Its achievement is stronger: it proves that the Animus constitution and architecture can survive contact with actual code and hostile conditions.
