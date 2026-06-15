# METHOD-AGENT-WORKFLOW: Multi-Agent Build Loop

**Status:** Canonical methodology · Created 2026-06-14 · Owner: ARETE  
**Scope:** Reusable orchestration pattern for decomposing strategic goals into scoped tasks, delegating to parallel agents, evaluating outputs against rubrics, and repairing failures in a loop.

---

## Problem

The existing prompt modes (`exploration.md`, `specification.md`, `evaluation.md`, `production.md`) solve *single-prompt* quality. They do not solve *multi-agent orchestration*: how to take a strategic goal, break it into scoped tasks, delegate those tasks to parallel builders, evaluate their outputs against measurable rubrics, and feed rejected outputs back into a repair loop.

This methodology bridges that gap.

---

## Philosophy

**One mode per agent call.** A single prompt that mixes exploration + specification + evaluation + delivery gets all four done poorly. Split them into separate calls, each with its own contract, output format, and rubric.

**Evidence over opinion.** Every evaluation score must cite a specific line of evidence from the output. A score without a citation is invalid.

**Budget-gated everything.** Every phase has an ET ceiling. Exceeding the ceiling is a hard stop, not a suggestion.

**Repair is production, not a second chance.** When an output fails evaluation, the repair agent receives a *new* production task with narrower scope, not a vague "make it better."

---

## Phase Mapping

| Phase | Existing Mode | What Happens | Output Artifact | Gate Condition |
|---|---|---|---|---|
| **1. ROADMAP** | `exploration` + `specification` | Decompose strategic goal into milestones and task specs | `ROADMAP.md` + `TASK_SPECS/*.md` | Every task has acceptance criteria |
| **2. DELEGATE** | `production` | Spawn N agents in parallel, each owns one task spec | `TASK_OUTPUTS/*` (code, docs, tests) | None — fire and forget |
| **3. EVALUATE** | `evaluation` | Score each output against its task rubric | `EVAL_REPORT.md` | Composite ≥ B (0.80) |
| **4. REPAIR** | `production` (scoped) | Re-run failed tasks with failure feedback injected | `TASK_OUTPUTS/*` (revised) | Re-evaluate; max 3 retries |
| **5. INTEGRATE** | `production` | Merge accepted outputs, run integration tests | `MERGED_CODE` + green CI | All tests pass |

---

## Task Specification Contract

Every task file (`TASK_SPECS/{id}_{name}.md`) MUST follow this exact skeleton. Incomplete task specs are rejected before delegation.

```markdown
# TASK-{ID}: {Name}

## Objective
One sentence: what this task must deliver.

## Constraints
- Budget: {ET estimate} effective tokens.
- Deadline: {wall-clock expectation}.
- Hard limits: explicit, measurable.

## Inputs
- Source files to read.
- Prior decisions to respect.
- APIs/schemas to conform to.

## Outputs
- Files to create/modify.
- Functions/classes to implement.
- Tests to add.

## Acceptance Criteria (numbered, falsifiable)
1. {criterion} — observable pass/fail.
2. ...

## Rubric
- correctness [3.0]
- schema_valid [1.5]
- concision [0.5]

## Exclusions
Explicitly out of scope.

## Dependencies
- BLOCKS: tasks this one blocks.
- BLOCKED_BY: tasks that must complete first.
```

**Rule:** If a task spec lacks numbered acceptance criteria, it cannot enter Phase 2. Go back to Phase 1.

---

## Agent Roles

Each agent in the loop runs in exactly one mode and owns exactly one task.

| Role | Mode | Job | Input | Output |
|---|---|---|---|---|
| **Planner** | `exploration` + `specification` | Decomposes strategic goals into task specs | Strategic goal + context | `ROADMAP.md` + `TASK_SPECS/*.md` |
| **Builder** | `production` | Implements one task spec | `TASK_SPEC.md` | Code, docs, tests |
| **Tester** | `production` | Generates/runs tests from acceptance criteria | `TASK_SPEC.md` + builder output | Test results (pass/fail) |
| **Reviewer** | `evaluation` | Scores builder output against rubric | `TASK_SPEC.md` + builder output | `EVAL_REPORT` entry |
| **Repair** | `production` | Re-implements with feedback injected | `TASK_SPEC.md` (revised) + reviewer feedback | Revised output |
| **Integrator** | `production` | Merges accepted outputs, runs CI | All accepted outputs | `MERGED_CODE` + green CI |

---

## Review Protocol

### Evaluation (Phase 3)

1. Run `Evaluation` mode on each builder output.
2. Score every dimension of the task rubric with cited evidence.
3. Compute composite score. Below 0.80 → REJECT.

### Repair (Phase 4)

1. For rejected outputs, extract the **3 weakest dimensions** and their evidence lines.
2. Inject those into the task spec's `exclusions` as: `"Previous attempt failed on {dim}: {evidence}"`.
3. Spawn a **new** Builder agent with the revised spec.
4. Max 3 repair attempts. After 3 → escalate to human.

### Integration (Phase 5)

1. Merge all ACCEPTED outputs into the working tree.
2. Run `pytest` (or equivalent) on the integration test suite.
3. Run `ruff` / `mypy` / lint gates.
4. Run `verify_imports.py` for the kernel.
5. If any gate fails → treat as a new task spec and enter Phase 2.

---

## Budget Allocation Formula

For a roadmap of N tasks with average ET estimate E:

| Phase | Formula | Typical % of total |
|---|---|---|
| ROADMAP | 0.05 × N × E | ~5% |
| DELEGATE | N × E | ~65% |
| EVALUATE | 0.2 × N × E | ~13% |
| REPAIR | 0.25 × N × E × rejection_rate | ~15% |
| INTEGRATE | 0.05 × N × E | ~2% |

**Rejection rate** assumed at 30% for new task types, 10% for well-understood patterns.

---

## Reusable Prompt: Multi-Agent Orchestration

Use this prompt whenever you want to execute the full loop:

```
ROLE: orchestrator / project manager

TASK: Execute a scoped build task using the Animus kernel's multi-agent
      workflow. Do not deviate from the methodology.

INPUTS:
  - task_spec: {path to TASK_SPEC file}
  - rubric: {name of rubric file}
  - budget_limit: {ET ceiling}
  - max_retries: {default 3}

WORKFLOW (strict sequence):
  1. READ the task_spec. Extract objective, constraints, acceptance criteria.
  2. DELEGATE to Builder agent in PRODUCTION mode with task_spec as sole input.
  3. TEST the output against acceptance criteria.
  4. EVALUATE the output against rubric in EVALUATION mode.
  5. IF composite < 0.80:
       a. Inject 3 weakest-dimension feedback into task_spec exclusions.
       b. GOTO 2 (decrement max_retries).
  6. IF max_retries == 0: ESCALATE to human with evidence.
  7. DELIVER final artifact.

CONSTRAINTS:
  - One mode per agent call. Never mix modes.
  - Every agent output must cite its evidence base.
  - BudgetManager enforces ET ceiling per iteration.
  - Sandbox enforces file/line limits on all edits.

OUTPUT FORMAT:
  1. Task ID and name
  2. Builder output summary (files touched, functions added)
  3. Test results (pass/fail per acceptance criterion)
  4. Rubric scores (dim | score | evidence)
  5. Verdict: [ACCEPT | REJECT → retry #N | ESCALATE]
  6. Final artifact path(s)
```

---

## Failure Taxonomy for the Loop

In addition to the standard technical/content failure taxonomies, the orchestration loop introduces:

| Code | Meaning | Resolution |
|---|---|---|
| `ORCH-001` | Task spec missing acceptance criteria | Reject spec, return to Planner |
| `ORCH-002` | Builder output exceeds file/line limit | Sandbox rollback, reject output |
| `ORCH-003` | Evaluation composite below B (0.80) | Enter Repair phase |
| `ORCH-004` | Max retries exhausted (3) | Escalate to human |
| `ORCH-005` | Integration test failure post-merge | Treat as new task, re-enter Delegate |
| `ORCH-006` | Budget exhausted before completion | Halt, report partial progress |

---

## Integration with Existing Artifacts

| This Methodology | Uses | Stored In |
|---|---|---|
| Prompt modes | `packages/forge/prompts/modes/*.md` | `docs/METHOD_AGENT_WORKFLOW.md` |
| Scoring rubrics | `packages/forge/rubrics/*.yaml` | Referenced by task specs |
| Failure taxonomy | `packages/forge/src/animus_forge/evaluation/failure_taxonomy.py` | Extended by `ORCH-*` codes above |
| Agent roles | `packages/forge/src/animus_forge/agents/supervisor.py` | Implemented in `kernel/agents/` |
| Budget tracking | `packages/forge/src/animus_forge/budget/manager.py` | Enforced per phase |
| Sandbox | `packages/forge/src/animus_forge/self_improve/sandbox.py` | Enforced on all builder outputs |

---

## References

- [[Prompt Modes]] `packages/forge/prompts/modes/README.md`
- [[Rubrics]] `packages/forge/rubrics/`
- [[Failure Taxonomy]] `packages/forge/src/animus_forge/evaluation/failure_taxonomy.py`
- [[Kernel Extraction]] `packages/kernel/KERNEL_FILE_MAP.md`
- [[TPS Audit]] `docs/TPS_LEAN_AUDIT_2026-06.md`

---

*Part of the Animus system. See `docs/ENGINE_VS_SHELL_ASSESSMENT.md` for strategic rationale.*
