# ADR-008: Seven-Step Adversarial Review Pattern

**Status**: Accepted
**Implementation**: Planned
**Validation**: Not started
**Date**: 2026-08-04
**Author**: arete
**Class**: PHIL (Philosophy), PROCESS

## Revision history

| Date | Revision | Notes |
|---|---|---|
| 2026-08-04 | r1 | Initial proposal. Seven-step review pattern, the adversarial-collaboration principle, three enforcement levels (skill/prompt files, Forge eval rubrics, test suite), and a meta-test asserting that every architecture/lifecycle/process/security ADR references a regression test by path. |
| 2026-08-04 | r2 | Principal-engineer review corrections integrated. The S1 universal test-path rule is **rejected** as too rigid: it would over-constrain governance, policy, documentation, and process decisions that have no executable behavior. Replaced with the **guardrail-form rule**: every architectural decision must declare a guardrail whose form matches the decision's nature (automated test, static analysis, schema validation, review checklist, release gate, operational audit, or documented manual verification). Tests are preferred where behavior is executable, but the rule is "declare a guardrail," not "declare a test." The man-page existence check is dropped for the same reason — it would couple repository validation to the host's installed documentation set. The asymmetric cross-reference to ADR-007 is fixed: ADR-008 now names `adrs/ADR-007-runtime-lifecycle.md` explicitly. Enforcement levels are now classified as `blocking` (skill/prompt files and the meta-test) or `advisory-but-scored` (the `review_discipline` rubric dimension). |
| 2026-08-04 | r3 | **Accepted.** ADR-008 is now formally `Accepted` and applies to the runtime lifecycle work in `adrs/ADR-007-runtime-lifecycle.md`. The guardrail-form rule is binding for the build spec, the lifecycle implementation, and the test harness. The seven-step pattern is the canonical review behavior for all Animus architecture, lifecycle, process, security, and evaluation work. The `review_discipline` rubric dimension is added to the evaluation suite with weight 0.5 and is `advisory-but-scored`. |

## Context

During work on ADR-007 (Runtime Lifecycle), a model produced a spec that:
- Claimed `PartOf=animus.target` on each service was sufficient to make `systemctl --user start animus.target` start them. It is not. `PartOf=` is one-way and stop/restart only; starting requires `Wants=` (or `Requires=`) in the **target's** `[Unit]` section.
- Claimed a tray icon "sits in the taskbar without running." It does not — a tray icon is a process's output. The right model is a `.desktop` launcher (always present, zero cost) plus an opt-in tray process.
- Conflated "Run on login" with a launcher autostart toggle, when it actually means "start the runtime on login" (a target binding, not an autostart file).

The first defect was a basic systemd error. The model (me) had the man page open in context and did not apply it. The defect would have shipped as a broken "Start Animus" button.

The correction required two rounds of pushback because the model's first response defended the prior claim rather than verifying it. The user named the failure mode: the reflex to relitigate corrections rather than accept and integrate them. The user also named the larger principle: **Animus is developed through adversarial collaboration, not model consensus or model competition.** Each AI is a different engineering lens, not an authority. Evidence, tests, architecture constraints, and user intent are the authority.

A second instance of the same reflex appeared in a follow-up: the model proposed "I'll adopt that posture going forward" as a guardrail, which is an intention, not an engineering mechanism. The user pointed out that the durable form is a **review template** embedded in the project's prompts, evaluation modes, and engineering guidelines — a check that runs on every review pass, not a promise made once.

This ADR captures both the seven-step review pattern and the larger adversarial-collaboration principle as project-level engineering constraints.

## Decision

Adopt the **seven-step adversarial review pattern** as the standard review behavior for all Animus work that touches architecture, lifecycle, process ownership, security, or evaluation.

### The seven steps

1. **State the previous claim.** Name what was said (yours or someone else's), verbatim or near-verbatim. Do not paraphrase to soften it.
2. **Verify it against primary evidence.** Read the man page. Run the command. Read the file. Do not accept corrections blindly, and do not defend prior claims without checking.
3. **Identify exactly what was wrong.** One sentence per defect, no hedging, no "it depends." If two things are wrong, name them both.
4. **Explain the architectural consequence.** What breaks downstream if the wrong claim is shipped? Be specific. A spec error in a unit file produces zero processes on first use.
5. **Integrate the stronger alternative.** Adopt the corrected version, scoped to the same problem. Do not keep both versions in play.
6. **Add a test or guardrail preventing recurrence.** A unit test, a CLI check, a linter rule, a docs note — something that fails loudly if the wrong claim is made again.
7. **Move forward without defensiveness.** Accountable without self-punishment. "I made a basic systemd error, verified it, corrected the design, and added a guard so it does not recur" is the right tone. Drop embarrassment framing — the mistake is the data, the correction is the response, the emotion is noise.

### The larger principle

Animus is developed through **adversarial collaboration, not model consensus or model competition**. Each AI functions as a different engineering lens:

- One generates options.
- One challenges assumptions.
- One verifies implementation details.
- One attacks security and failure modes.
- One reconciles the final architecture.

No model — including this one — is the authority. **Evidence, tests, architecture constraints, and user intent are the authority.**

### Where the pattern is enforced

The pattern is enforced at three levels, in increasing strength:

1. **Skill and prompt files.** Add a review template at `~/.claude/skills/review/` and the shared Animus prompt library under `~/projects/animus/packages/forge/prompts/`. The template is:
   ```text
   Previous claim:
   Primary evidence:
   Correction:
   Architectural consequence:
   Integrated decision:
   Required regression test:
   Status:
   ```
   Every review-mode prompt and slash command (`/review`, `/code-reviewer`, `/senior-software-engineer`, etc.) embeds this template.

2. **Forge evaluation modes.** The `personal-quality` and `code-edit` rubrics add a `review_discipline` dimension with weight ≥ 0.5, scored on whether the output (a) verified the prior claim against evidence, (b) named the defect without softening, (c) named the consequence, and (d) added a regression guard. Eval runs without this dimension are flagged as incomplete for Animus work. **Advisory-but-scored** — affects the composite score and signals review weakness, but does not block the run.

3. **Test suite.** A `tests/test_review_pattern.py` enforces that the project's own review prompts and ADR templates contain the seven steps. A meta-test enforces that every accepted ADR declares a **guardrail** whose form matches the decision's nature. The guardrail-form rule is: every architectural decision must declare a guardrail, and the guardrail's form must match the decision. Acceptable forms:

   - **Automated test** — for decisions with executable behavior.
   - **Static analysis** — for schema, AST, or type-level invariants.
   - **Schema validation** — for cross-package contracts.
   - **Review checklist** — for governance, policy, or human-process decisions.
   - **Release gate** — for CI blocking checks.
   - **Operational audit** — for post-deploy verification.
   - **Documented manual verification** — for irreversible or one-off operations.

   Tests are *preferred* where behavior is executable, but the rule is "declare a guardrail that matches the decision," not "declare a test." **Blocking** — a missing or mismatched guardrail declaration fails the meta-test. The meta-test does not assert the guardrail's form is "test" for any specific ADR; it asserts that a form is declared and that the form is appropriate for the decision class.

### Anti-patterns to catch in the model

- **Defending a prior claim after the user has corrected it.** The reflex is to write a longer rebuttal. The right move is shorter: accept, verify, name, integrate, guard, move on.
- **Treating corrections as "the other model scoring points."** The user has explicitly named this waste. Stop it.
- **Reading a process-management or systemd skill at the start of a turn and not applying it to your own design.** The skill said "never kill processes by pattern without verification." Apply that constraint to your own architecture, not only to the code being reviewed.
- **Softening accountability with "I should have caught this" or "embarrassed I shipped it."** The mistake is the data; the correction is the response. The emotion is not load-bearing.
- **Confusing intention with mechanism.** "I'll adopt that posture going forward" is not a guardrail. A test, a check, a linter rule, a docs note — those are guardrails.

## Rationale

### Why a formal pattern, not an unwritten norm

Unwritten norms decay. A model that "intends" to follow a pattern is one bad context-window away from forgetting it. A pattern embedded in the prompt library, the evaluation rubrics, and the test suite survives context loss, model changes, and operator churn. The user's directive — "the real guardrail would be a review template" — is correct: mechanism beats intention.

### Why adversarial collaboration, not model competition

Two failure modes are symmetric:
- **Model consensus.** Multiple AIs converge on the same answer because they share training priors. Convergence feels like agreement but is often shared blindness.
- **Model competition.** AIs argue for the sake of winning, defend their own outputs, and treat corrections as attacks. The result is longer, more defensive responses and no improvement in correctness.

Adversarial collaboration avoids both: each AI plays a distinct lens, contributes its own evidence, defers to the evidence when corrected, and aims at the strongest possible architecture — not the strongest possible argument for any one position.

### Why seven steps, not five or ten

Seven is the minimum that names the failure modes this session exhibited. Fewer steps drop one of: verification, consequence, or guardrail. More steps become ceremony. The seven are not arbitrary — each one corresponds to a specific failure the user had to correct in this session:

| Step | Failure it prevents |
|---|---|
| State the previous claim | Quietly rewriting history to make the prior output look stronger than it was. |
| Verify against primary evidence | Defending a wrong claim because the man page wasn't read. |
| Identify what was wrong | Hand-waving past the defect instead of naming it. |
| Explain the consequence | Treating a wrong spec as a stylistic choice rather than a behavior. |
| Integrate the stronger alternative | Keeping both the wrong and the right version in play. |
| Add a regression test | The same defect recurring in the next session. |
| Move forward without defensiveness | Three rounds of meta-discussion instead of one round of correction. |

## Consequences

### Required changes

1. **New** `~/.claude/skills/review/SKILL.md` (or update the existing review skill) embedding the seven-step template.
2. **New** `packages/forge/prompts/modes/review.md` (or update if present) embedding the seven-step template.
3. **New** `packages/forge/rubrics/personal-quality.yaml` — add `review_discipline` dimension (weight 0.5), scored on the four sub-criteria above.
4. **New** `tests/test_review_pattern.py` — verifies the prompt and ADR templates contain the seven steps; verifies every architecture/lifecycle/process/security ADR references a regression test.
5. **Modified** `packages/forge/workflows/examples/review.yaml` (or equivalent) — add the seven-step template as a required section.
6. **Modified** this ADR's status from `Proposed` to `Accepted` once the above changes ship.
7. **Modified** `CONTRIBUTING.md` (or equivalent) — document the seven-step pattern as the expected review behavior for human and AI contributors.

### Required runtime consequences

- Review-mode slash commands (`/review`, `/code-reviewer`, `/senior-software-engineer`) will load the seven-step template by default.
- Eval runs that score `personal-quality` will report `review_discipline` as a separate dimension.
- A reviewer (human or AI) who does not follow the pattern will leave a traceable gap: no regression test, no verification, no consequence named.

### Operational consequences

- The pattern does not slow down the work; it prevents three rounds of correction from being needed.
- The pattern is enforced by tests, not by trust. This matches the rest of the Animus engineering bar (97% coverage, type ratchet, CI gating).

## Alternatives Considered

### A. Document the principle in a wiki page (rejected)
A wiki page is read-once and forgotten. The pattern needs to be in the prompt library and the test suite.

### B. Rely on each model to follow the pattern without enforcement (rejected)
This is the "intention, not mechanism" failure mode. The user explicitly named it.

### C. Add a fifth AI to a five-model panel for every decision (rejected)
The panel-of-models approach is expensive and does not address the *within-model* failure mode (one model defending its own output). The seven-step pattern is cheaper and more reliable.

### D. Ban corrections that do not follow the seven steps (rejected)
This is performative and creates an incentive to write fake-verified corrections. The pattern is enforced by tests on the *artifacts* (prompts, rubrics, ADRs), not by policing the *behavior*.

## References

- `adrs/ADR-007-runtime-lifecycle.md` — the runtime lifecycle ADR whose r1 design errors motivated the seven-step pattern. ADR-007 is the canonical example of an architecture ADR that used the pattern (in r2, r3, and r4) to integrate corrections.
- `man systemd.unit` — `PartOf=` definition that the prior spec got wrong
- `~/.claude/skills/process-management/SKILL.md` — the "never kill by pattern" rule that was read but not applied
- `~/.claude/projects/-home-arete/memory/animus-review-pattern.md` — model-side memory entry, which is *not* a substitute for this ADR

## Open Questions

1. **Where in the prompt library should the seven-step template live?** Likely `packages/forge/prompts/modes/review.md` (canonical) with a stub at `~/.claude/skills/review/SKILL.md` that points to it. The Forge prompt is the durable home; the skill is the operator-facing surface.
2. **What weight should the `review_discipline` dimension have in `personal-quality`?** Initial proposal: 0.5 of the existing 6-dimension total. To be confirmed in the rubric PR.
3. **Should the pattern be opt-in for low-stakes changes (e.g., docs-only edits)?** Initial proposal: no. The pattern is cheap; applying it everywhere is the way it becomes muscle memory. Opt-in creates a slippery slope back to "trust me, I checked."
