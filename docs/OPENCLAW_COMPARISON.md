# OpenClaw vs Animus — Feature Comparison

> Competitive analysis: what OpenClaw does well, where Animus already leads, and what to learn.

## Feature Matrix

| Capability | OpenClaw | Animus (Current) | Gap |
|---|---|---|---|
| **Self-hosted / local** | ✅ Node.js + Ollama | ✅ Your machine + Ollama | None |
| **Shell execution** | ✅ Direct system access | 🟡 Via Claude Code, not native | **Need Forge to execute** |
| **Multi-channel chat** | ✅ WhatsApp/Telegram/Discord/Slack | ❌ CLI only planned | Medium priority |
| **Persistent memory** | ✅ Markdown conversation files | 🟡 Designed but not implemented | **Core Phase 1b blocker** |
| **Skills marketplace** | ✅ 565+ community skills | ❌ | Not needed — ours are bespoke |
| **Multi-model routing** | ✅ Claude/GPT/Ollama hot-swap | 🟡 Designed for Ollama + Claude | Small gap |
| **Multi-agent orchestration** | 🟡 Basic swarms | ✅ Forge + Quorum (designed) | **Our advantage** |
| **Intent-based coordination** | ❌ No intent graph | ✅ Quorum consensus model | **Our differentiator** |
| **Self-improvement loop** | ❌ | 🟡 Phase 1b spec exists | Unique to Animus |
| **Gateway/distributed arch** | ✅ Mature | ❌ | Gap, but different design philosophy |

## Honest Assessment

OpenClaw's architecture treats each component as an independent service communicating through well-defined protocols — that's solid engineering, but it's a generic orchestration framework. 147K stars, 400K lines of code, and security concerns significant enough that NanoClaw was built specifically to address them.

Animus has two things OpenClaw doesn't and can't easily add:

1. **Intent-based coordination** — Quorum's consensus model where agents publish decisions to a shared intent graph
2. **Self-improvement** — the system modifying its own identity files based on reflection

Those aren't features — they're architectural commitments that are hard to bolt on after the fact.

## What to Steal from OpenClaw's Playbook

- **Markdown-based conversation persistence** — simple, portable, AI-readable
- **Chat-channel-as-interface pattern** — WhatsApp/Telegram reach > CLI
- **The "skills" modularity concept** — maps cleanly to Forge workflows

## What NOT to Copy

- **The 400K-line codebase sprawl** — complexity for complexity's sake
- **Permissionless execution** — their biggest security liability
- **Community-first development** — we're building a sovereign system, not a platform

## Strategic Takeaway

Ship DOSSIER and BenchGoblins, then build Animus with OpenClaw's UX lessons but our own Forge/Quorum mechanics. The intent graph and self-improvement loop are what make Animus worth building instead of just installing OpenClaw.
