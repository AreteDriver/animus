# Exploration Mode

Use when the option space isn't mapped yet. The output is a list of distinct candidates with surfaced unknowns and challenged assumptions — not a recommendation.

**Rule**: Exploration mode never picks a winner. If the prompt tempts you toward a single answer, you're in the wrong mode — switch to Evaluation.

---

## Template

```
ROLE: option-generator / skeptical analyst

TASK: Generate N distinct options for {objective}. Surface unknowns. Challenge the stated assumptions.

INPUTS:
  - objective: {what the reader is trying to decide}
  - context: {known facts, constraints that are non-negotiable}
  - assumptions: {what the asker currently believes — flag these for challenge}
  - evidence base: {documents, data, prior decisions the model may draw from}
  - exclusions: {what NOT to include — already-rejected options, out-of-scope paths}

RUBRIC:
  - breadth [2.0]              distinct, non-overlapping options (not variants of one idea)
  - novelty [1.5]              surfaces options the reader likely hadn't considered
  - unknown-surfacing [2.0]    names what we don't know that would change the answer
  - assumption-challenge [1.5] flags stated assumptions that may not hold
  - grounding [1.0]            every option tied to at least one evidence item

OUTPUT FORMAT:
  1. Option table (markdown): id | name | premise | cheapest proof | biggest risk
  2. Three highest-leverage unknowns — for each: what question, how to answer cheaply, what it would change
  3. Three stated assumptions that may not hold — for each: assumption, why it might be wrong, test
  4. One-paragraph synthesis: the option space, its shape, where the genuine disagreement sits

FAILURE CONDITIONS:
  - any two options are minor variants of the same underlying idea
  - "it depends" without naming what it depends on
  - recommending a winner (wrong mode — switch to Evaluation)
  - no unknowns surfaced (suspicious — re-examine)
  - assumptions echoed back instead of challenged
```

## Example instantiation

> **Task**: Generate 6 distinct strategies for bringing anchormd scans past a 5% paid conversion rate. Pricing is locked at $19. Assume current conversion is 0%.

Fill in INPUTS, leave RUBRIC, OUTPUT FORMAT, FAILURE CONDITIONS as-is.

## Pairing

- Scoring rubric: `personal-quality` — high weights on `precision` + `evidence_quality` will penalize generic options
- Follow-up mode: Evaluation (to pick a winner) or Specification (to build the chosen option)
