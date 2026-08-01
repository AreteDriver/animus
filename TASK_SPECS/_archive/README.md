# Archived Task Specifications

These task specs were created in June 2026 for the Hermes/Animus integration roadmap.
They are now **obsolete** because:

1. **Path references are stale** — they point to `kernel/agents/prompts/hermes/`,
   `kernel/builder/terminal_agent.py`, etc., which have been reorganized during
   the boundary refactoring (July 2026).
2. **Architecture has shifted** — the TerminalAgent concept was subsumed by
   Animus Forge's workflow executor. The Hermes prompt work was deprioritized
   in favor of the local model stack.
3. **Some work shipped differently** — the FastAPI endpoint and mobile UI
   became the Bootstrap dashboard + PWA instead of a separate Kernel server.

If reviving any of these concepts, use them as reference only and rewrite
against current filesystem reality.

**Archived**: 2026-07-30
