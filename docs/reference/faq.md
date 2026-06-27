# Frequently Asked Questions

> Common questions about Animus.

---

## General

**Q: What is Animus?**
A: A Mind-class AI exocortex — persistent memory, multi-agent orchestration, and autonomous improvement. It remembers conversations, learns preferences, and coordinates AI agents across complex workflows.

**Q: Is Animus a chatbot?**
A: No. It is a persistent intelligence layer that accumulates knowledge about you over time. Conversations are one interface among many (CLI, dashboard, API, MCP server).

**Q: Does it require cloud services?**
A: No. Animus is local-first by design. It works fully offline with Ollama after initial install. Optional cloud providers (Anthropic, OpenAI) can be configured but are not required.

## Installation

**Q: Which Python version do I need?**
A: Core requires ≥3.10, Bootstrap ≥3.11, Forge ≥3.12. If in doubt, use Python 3.12 for everything.

**Q: Can I install just one package?**
A: Yes. Each package is independently installable. Install `packages/types/` first, then any combination.

## Privacy

**Q: Where is my data stored?**
A: Locally on your machine. Memory defaults to SQLite at `~/.local/share/animus/memory.db`. No telemetry by default.

**Q: Can Animus modify its own identity files?**
A: Small changes (<20% of file size) are written directly. Larger changes go through an approval gate in the dashboard. `CORE_VALUES.md` is immutable by design.

## Development

**Q: How do I run tests?**
A: Per-package:
```bash
pytest packages/core/tests/ -v
pytest packages/forge/tests/ -v
pytest packages/bootstrap/tests/ -v
pytest packages/quorum/tests/ -v
```

**Q: Why is CI failing?**
A: GitHub Actions may be blocked by billing limits. Verify locally with `pytest` and `ruff` before merging.

---

## See Also

- [Getting Started → Quickstart](../getting-started/quickstart.md)
- [Contributing → Setup](../contributing/setup.md)
- [Operators → Troubleshooting](../operators/troubleshooting.md)
