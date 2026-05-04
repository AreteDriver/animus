# Prompt Modes

Four named prompt families. Each enforces a single objective and a fixed output contract, so outputs are scorable and failures are classifiable.

The problem the modes solve: a single prompt that mixes exploration, specification, evaluation, and delivery gets all four done poorly. Splitting them into named modes with distinct contracts produces reliable work per stage and makes regressions attributable.

## When to use each

| Mode | Use when |
|---|---|
| [`exploration.md`](exploration.md) | You don't know the option space yet. You want alternatives, surfaced unknowns, challenged assumptions. |
| [`specification.md`](specification.md) | The option is chosen. You want a buildable, testable, reviewable spec. |
| [`evaluation.md`](evaluation.md) | You have candidates (code, copy, options, outputs) and need scored comparisons, not opinions. |
| [`production.md`](production.md) | The decision is made. You want the final deliverable with no exploratory sprawl. |

## Anti-pattern

Combining two or more modes in a single prompt:

> "Analyze this opportunity, compare it to my background, rewrite the resume, tell me if I should pursue it, and draft a recruiter response."

That is exploration + evaluation + specification + production in one call. Split into four prompts, each with its own contract. Composite output quality improves more than any single prompt-engineering tweak.

## Schema

Every mode uses the same skeleton. Concrete templates in this directory fill it in:

```
ROLE:        [one role — evaluator / architect / editor / critic / analyst]
TASK:        [one task — not two, not "and also"]
INPUTS:
  - objective        [what the reader will do with the output]
  - audience         [who reads it]
  - constraints      [explicit, measurable — "350 words", not "concise"]
  - evidence base    [what the model is allowed to draw from]
  - exclusions       [what must NOT appear]

RUBRIC:
  - criterion [weight]   [e.g. "precision [2.0]" — aligns with rubrics/*.yaml]

OUTPUT FORMAT:
  [numbered sections — every section named and required]

FAILURE CONDITIONS:
  [what makes the output invalid, not merely weak]
```

## Pairing with rubrics

Each mode pairs with one or more scoring rubrics from [`../../rubrics/`](../../rubrics/):

| Mode | Recommended rubric |
|---|---|
| exploration | `personal-quality` (weighted toward precision + evidence) |
| specification | `code-edit` or a custom spec rubric |
| evaluation | `personal-quality` (actionability + format-compliance critical) |
| production | `personal-quality` with format-compliance weight ≥ 2.0 |

Run a mode's output through its rubric:

```bash
animus-forge eval run <suite> --rubric personal-quality --prompt-version <mode>-v1
```

Compare two prompt versions side-by-side:

```bash
animus-forge eval compare prev last --suite <suite>
```

## Reference

- Failure taxonomy: [`evaluation/failure_taxonomy.py`](../../src/animus_forge/evaluation/failure_taxonomy.py) (technical), [`failure_taxonomy_content.py`](../../src/animus_forge/evaluation/failure_taxonomy_content.py) (F1–F8 content quality, when installed)
- Rubric library: [`rubrics/`](../../rubrics/)
- Compare CLI: `animus-forge eval compare --help`
