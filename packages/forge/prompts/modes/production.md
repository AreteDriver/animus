# Production Mode

Use when the decision is made and the spec (or intent) is clear. Output is the final deliverable — nothing else. No preamble, no options, no meta-commentary, no "let me know if..." close.

**Rule**: Production mode delivers exactly what was specified, in the specified format, with nothing extra. If you find yourself explaining your choices in the output, you're in the wrong mode.

---

## Template

```
ROLE: editor / producer

TASK: Produce the final {deliverable} for {audience}.

INPUTS:
  - deliverable: {one artifact — email, PR description, landing-page copy, decision memo, code change, launch announcement}
  - objective: {what the reader does after reading it — the specific next action}
  - audience: {who reads it — hiring manager, engineering peers, customers, oncall}
  - constraints: {measurable — "350 words", "≤ 3 bullets", "subject < 60 chars", "one CTA"}
  - voice / format: {existing pattern to match — "matches the tone of <reference>", "follows <template>"}
  - evidence base: {every claim must map here — responsibilities from a role, bullets from a resume, commits from a PR}
  - exclusions: {must not appear — hedging, soft-skill filler, generic adjectives, unsourced quantifiers}

RUBRIC:
  - format-compliance [2.5]     hits every format constraint exactly (word count, structure, sections)
  - concision [2.0]              every sentence earns its keep; remove any sentence that adds no information
  - decision-clarity [2.0]       the reader knows what they're being asked to do, or what has been decided
  - non-genericity [2.0]         claims are specific to the context; no boilerplate that could apply anywhere
  - voice-fidelity [1.5]         matches the requested voice/format reference

OUTPUT FORMAT:
  [exactly the format specified in INPUTS — no wrapper text, no "Here is the..." prefix, no trailing sign-off unless requested]

FAILURE CONDITIONS:
  - any exploratory language ("we could consider", "one option would be") — wrong mode
  - any meta-commentary about the output itself ("I've structured this as...")
  - filler phrases: "it's worth noting", "importantly", "in essence", "at the end of the day"
  - soft-skill boilerplate: "strong communicator", "collaborative", "team player" unless specifically evidenced
  - quantified claims without evidence ("significantly improved X") unless a number from the evidence base backs it
  - format violations: exceeded word count, skipped required section, wrong structure
  - hedging the deliverable: "you may want to consider..." when a decision is required
```

## Example instantiation

> **Deliverable**: Cold outreach email to a hiring manager at Scale AI about an FDE opening.
>
> **Objective**: Reader books a 20-minute call.
>
> **Audience**: Director of Solutions Engineering, Scale AI.
>
> **Constraints**: ≤ 150 words, subject ≤ 60 chars, one CTA (the call request), no attachments mentioned.
>
> **Evidence base**: Gatekeeper (LIVE, 3,177 tests, Stripe live), monolith detection rules (36 live, 0% FP 7-day), IBM manufacturing logistics ops role (17 years).
>
> **Exclusions**: no mention of EVE Online, no generic "passionate about AI", no filler adjectives, no salary.

Fill in INPUTS. Leave RUBRIC, OUTPUT FORMAT instruction, FAILURE CONDITIONS as-is. Output is the email body only.

## Pairing

- Scoring rubric: `personal-quality` — weight `format_compliance` and `precision` highest when scoring production outputs
- Preceded by: Specification (if the deliverable is spec'd) or Evaluation (if the decision that prompts the deliverable just happened)
- NOT followed by: another mode. Production is terminal — if you want to iterate, run a fresh Evaluation on this output before re-entering Production.

## Iteration guidance

If a production output scores below band B on the rubric:
1. Do NOT ask the model to "make it better" — too vague; will regress or add filler
2. DO run the output through Evaluation mode against `personal-quality`, surface the 3 lowest-scoring dims
3. DO re-enter Production with those specific dim failures named in INPUTS → exclusions (e.g. "exclusions: generic adjectives flagged in previous pass: [list]")

Measured iteration beats "try again" prompts by 2-4x in practice. The rubric tells you what to fix; the named failures keep the regen scoped.
