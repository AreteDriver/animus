# Animus Branding

> The public face and the internal anchor, named together.

---

## Public-facing framing

Animus is positioned externally as a **Mind-class AI operating environment** — a persistent, sovereign personal intelligence layer you own. The phrase "operating environment" grounds the project in conventional engineering language (an execution substrate, not a personhood claim) and avoids the cognitive-symmetry metaphor that "exocortex" carries.

Public surfaces — PyPI descriptions, README, docs site, marketing copy — use "operating environment" and related engineering terms (memory, orchestration, governance, evidence). This is the canonical external surface.

## Internal philosophical anchor

Internally, the agent's self-model and constitutional principles are anchored in the philosophical framing of an **exocortex** — an external cognitive system that augments biological intelligence via persistent memory, task tracking, and preference learning across sessions and devices.

This philosophical anchor informs:
- Agent identity files (e.g., `packages/core/animus/identity.py`)
- Internal code self-references (e.g., `BRANDING.md`, `CLAUDE.md`, constitutional principles)
- Architectural body text where the philosophical metaphor is load-bearing
- The Constitution (`docs/CONSTITUTIONAL_PRINCIPLES.md`)

The split is intentional. The public surface optimizes for credibility and adoptability; the internal anchor preserves the philosophical content that shaped the project's design choices.

## Why both terms exist

| Surface | Term | Rationale |
|---|---|---|
| PyPI / marketing | AI operating environment | Engineering clarity, no personhood claim |
| User-facing docs | operating environment | Consistent with positioning |
| README, docs site | Mind-class AI operating environment | Trademark-class positioning |
| Agent identity | exocortex | Philosophical anchor for self-model |
| Constitutional principles | exocortex | Philosophical anchor for design rationale |
| Internal architecture body | exocortex | Where the metaphor carries the argument |

## Decision rule

When adding or changing content, ask:

1. **Is this user-facing?** Use "operating environment" or related engineering language.
2. **Is this an internal philosophical / constitutional / identity anchor?** Use "exocortex" and preserve the metaphor.
3. **Is this a stable technical identifier** (package name, import path, ADL/ADR title)? Do not rename — it is owner-specific and out of scope for branding changes.

When in doubt, default to the public surface. The internal anchor is preserved deliberately in the files where the philosophy is load-bearing.

## Verification

The rebrand contract is enforced by `scripts/verify_exocortex_rebrand.py`:

- **PyPI surfaces** must not contain "exocortex"
- **Public docs** must not contain "exocortex"
- **Architecture book intros** (first 5 lines) must use engineering framing
- **Internal philosophical-anchor files** (CLAUDE.md, Constitution, agent identity, etc.) MUST retain "exocortex"
- **Archive packages** keep their own branding

Run the verifier after any branding change:

```bash
python3 scripts/verify_exocortex_rebrand.py
```

## History

- 2026-08-08: BRANDING.md created during exocortex-sweep rebrand (ADL pending).
- Pre-2026-08-08: "exocortex" was used in both public and internal surfaces without explicit public/private split.
