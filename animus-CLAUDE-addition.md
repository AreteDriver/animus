# Animus / Forge / Quorum — Protocol Invariants
> Add this content to the relevant CLAUDE.md alongside your existing file.
> Apply to whichever layer you're working in.

## Architecture Layers

- **Core (Animus)** — exocortex UI, identity, memory
- **Forge** — agent orchestration (was Gorgon)
- **Quorum** — stigmergic coordination protocol (was Convergent)

## Quorum Protocol Invariants

IntentNode schema (canonical, do not modify without migration plan):
- `id`, `agent_id`, `intent_type`, `payload`, `timestamp`, `signature`, `stability_score`

Signing: ed25519. Never change the signing algorithm — all historical nodes would fail verification.

StabilityScorer inputs: recent consensus rate, alignment score, separation distance. This is the pheromone analog — it reinforces convergent paths.

IntentResolver reads top-N stable nodes. N is configurable, default 5. Do not hardcode N.

Three boid rules govern coordination:
- **Separation** — no conflicting intents from the same agent
- **Alignment** — shared directional intent across the swarm
- **Cohesion** — stay within mission scope

## Forge Orchestration

- Never couple agent identity to a single model version — model IDs are metadata, not architecture
- Task dispatch is the ChainLog integration point: log before dispatch, log after completion
- Checkpoint/resume is a first-class feature — assume any long-running task may be interrupted

## What This Project Is

Three-layer agent infrastructure. Quorum is the coordination layer that differentiates this from a simple orchestration framework — stigmergic coordination means agents converge on stable solutions without central direction, the same way ant colonies find optimal paths.

This is simultaneously internal infrastructure (runs BenchGoblins scoring, Gorgon Media content pipeline) and a future product (Bittensor subnet candidate, enterprise AI orchestration layer).
