# Development Setup

> How to set up your environment to contribute to Animus.

---

## Prerequisites

- Python 3.10+ (Core), 3.11+ (Bootstrap), or 3.12+ (Forge)
- Git
- [Ollama](https://ollama.com) for local LLM inference

## Clone

```bash
git clone https://github.com/your-org/animus.git
cd animus
```

## Install Dependencies

```bash
# Install shared types first
pip install -e packages/types/

# Install the packages you want to work on
pip install -e "packages/core/[dev,api]"
pip install -e "packages/forge/[dev]"
pip install -e "packages/quorum/[dev]"
pip install -e "packages/bootstrap/[dev]"
```

## Verify

```bash
# Run tests for the package you're working on
pytest packages/core/tests/ -v
pytest packages/forge/tests/ -v
pytest packages/quorum/tests/ -v
pytest packages/bootstrap/tests/ -v
pytest packages/contracts/tests/ -v

# Check code style
ruff check packages/
ruff format --check packages/

# Type-check ratchet (must pass before pushing)
python scripts/mypy-ratchet.py core kernel forge bootstrap
```

## Pre-commit Hooks

Install pre-commit hooks to catch issues before they reach CI:

```bash
pip install pre-commit
pre-commit install
```

Run manually across all files:

```bash
pre-commit run --all-files
```

## Local LLM

Install Ollama and pull a model:

```bash
ollama pull qwen2.5:14b
```

Animus will auto-detect Ollama and use it as the default provider.

---

## See Also

- [Workflow](workflow.md) — Git flow, PR process, CI
- [Debugging](debugging.md) — Common issues
- [Architecture → Standards](../architecture/standards.md) — Code standards
