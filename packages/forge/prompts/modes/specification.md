# Specification Mode

Use when the decision is made and you need a buildable, testable, reviewable spec. Output is a fixed-structure document with explicit constraints and measurable acceptance criteria.

**Rule**: Specification mode never explores alternatives or defends the decision. Those live in Exploration and Evaluation respectively. A spec that litigates its own premise is a weak spec.

---

## Template

```
ROLE: architect / spec-writer

TASK: Produce a complete specification for {artifact}. The decision to build this is already made — do not re-argue it.

INPUTS:
  - artifact: {the thing being specified — API, module, workflow, deliverable}
  - objective: {what it must do, in one sentence}
  - constraints: {explicit + measurable — "p95 latency < 200ms", "runs on Python 3.12", "no external API calls"}
  - interfaces: {what it consumes, what it emits — types, schemas, protocols}
  - acceptance: {how we know it's done — test names, thresholds, observable behaviors}
  - evidence base: {prior art, reference implementations, standards to follow}
  - exclusions: {explicitly out of scope — features the reader might assume are included}

RUBRIC:
  - completeness [2.0]          every interface has input + output + error cases named
  - testability [2.0]            every acceptance criterion is observable and falsifiable
  - constraint-fidelity [1.5]    stated constraints appear as hard constraints in the spec
  - clarity [1.0]                a reader unfamiliar with the project can implement from this alone
  - non-ambiguity [1.5]          no "typically" / "usually" / "should probably" — only MUST / MAY / MUST NOT

OUTPUT FORMAT:
  1. Summary (2-3 sentences — what, for whom, why now)
  2. Requirements (numbered MUST / MAY / MUST NOT list)
  3. Constraints (numbered, each with a measurement method)
  4. Interfaces (one section per consumed/emitted interface, with typed schema)
  5. Acceptance criteria (numbered list of falsifiable observations or tests)
  6. Out of scope (explicit list — what this spec does NOT cover)
  7. Open questions (if any — labeled; do not inject decisions)

FAILURE CONDITIONS:
  - any requirement is untestable ("the system should be fast")
  - any interface missing error-case behavior
  - any constraint without a measurement method
  - exploratory language ("we could", "maybe consider") — wrong mode
  - acceptance criteria that reference implementation details rather than observable behavior
  - hidden scope-creep (features appearing in requirements that weren't in objective/inputs)
```

## Example instantiation

> **Artifact**: A `CostCalculator` module for `animus_forge/evaluation/`. Objective: estimate per-case USD cost from token counts and model name.
>
> **Constraints**: must run without network calls; must return 0.0 for unknown models with a warning; must handle Ollama (free) as a first-class case.
>
> **Interfaces**: consumes `(tokens_in: int, tokens_out: int, model: str)`; emits `CostEstimate(usd: float, pricing_source: str)`.

Fill in INPUTS, leave RUBRIC, OUTPUT FORMAT, FAILURE CONDITIONS as-is.

## Pairing

- Scoring rubric: `code-edit` if the spec produces code directly, otherwise a custom spec rubric
- Preceded by: Exploration (to pick the option) or Evaluation (to pick among candidates)
- Followed by: Production (to deliver the thing the spec describes)
