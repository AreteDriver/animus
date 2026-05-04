# Evaluation Mode

Use when you have candidates (options, outputs, PRs, copy variants, job roles, vendors) and need a scored comparison that produces a decision — not opinions.

**Rule**: Evaluation mode always produces a verdict and an evidence trail. "It depends" is not an acceptable output. If the criteria can't decide, either the criteria are wrong (fix them) or you need more evidence (surface it explicitly, don't punt).

---

## Template

```
ROLE: evaluator / critic

TASK: Score {candidates} against {rubric} and produce a decision.

INPUTS:
  - candidates: {the N things being compared — options, outputs, vendors, roles}
  - rubric: {weighted criteria — name, weight, what high/low score means}
  - acceptance: {what a passing composite looks like — floor score, required per-dim minimums}
  - evidence base: {the inputs the evaluation must cite — documents, metrics, prior decisions}
  - exclusions: {criteria NOT to evaluate — keeps scope tight}

RUBRIC (for the evaluation itself, meta):
  - rubric-fidelity [2.5]       every criterion from the input rubric gets scored; no scope-creep dims added
  - evidence-grounding [2.0]    every score has a cited evidence reference, not a vibe
  - error-specificity [2.0]     weaknesses are specific claims about the candidate, not genres of weakness
  - verdict-clarity [1.5]       the decision is unambiguous — which candidate wins, or "none" with reason
  - tradeoff-honesty [1.0]      the winner's weaknesses are named; the losers' strengths are named

OUTPUT FORMAT:
  1. Scored table: candidate | dim1 | dim2 | ... | composite
  2. Evidence per score: for each candidate, for each dim, one-line evidence citation
  3. Top 3 weaknesses per candidate — specific, each tied to a dim score
  4. Top 3 strengths per candidate — same format
  5. Decision: one of [pursue: <candidate>] | [pursue selectively: <candidate> with conditions] | [pass: none qualify]
  6. Conditions: if "pursue selectively", the specific modifications needed
  7. What would change the decision: the single piece of new evidence that would flip the verdict

FAILURE CONDITIONS:
  - any score without cited evidence
  - "it depends" without naming the deciding variable
  - weaknesses stated as categories ("scalability issues") rather than specifics ("no caching layer at L3")
  - scope-creep: dims scored that weren't in the input rubric
  - verdict that restates the question ("further analysis needed") — wrong mode, go back to Exploration
  - winner's weaknesses omitted or downplayed
```

## Example instantiation

> **Candidates**: Three FDE role postings (Palantir, Scale AI, Glean).
>
> **Rubric**: `personal-quality` extended with weighted fit dims:
> - domain alignment [2.5]
> - execution evidence [2.0]
> - technical credibility [2.0]
> - stakeholder fit [1.5]
> - compensation match [1.0]
> - risk factors [1.0]
>
> **Acceptance**: composite ≥ 0.7 AND no single dim < 0.5.

Fill in INPUTS, leave RUBRIC, OUTPUT FORMAT, FAILURE CONDITIONS as-is.

## Pairing

- Scoring rubric: `personal-quality` — `actionability` and `evidence_quality` weights catch the two most common failure modes (no decision, no grounding)
- Preceded by: Exploration (to generate candidates) or raw inputs (PRs, job posts, vendor options)
- Followed by: Specification (to build the winning candidate) or Production (to draft the follow-up — outreach, PR comment, decision memo)

## Adversarial extension

After the first pass, optionally run:

> "Attack the previous evaluation. Assume a hostile reviewer. List the 5 weakest scores and the 3 most questionable evidence citations. Revise only those."

Use this when the stakes justify it (hiring, architecture calls, vendor contracts). Do not use on every run — judge disagreement noise dominates at small sample sizes.
