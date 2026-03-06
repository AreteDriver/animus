# Animus Forge

Multi-agent orchestration framework for production AI workflows.

## Features

- **Workflow executor** — YAML-defined multi-agent pipelines with mixins (AI, MCP, queue, graph)
- **Provider abstraction** — Anthropic, OpenAI, Ollama, and more
- **Self-improvement** — Analyze codebase, generate improvements, test in sandbox, create PRs
- **Budget management** — Persistent token/cost tracking per workflow
- **Eval framework** — Benchmark and score agent outputs
- **MCP tool execution** — Bridge to Model Context Protocol servers
- **CLI + API + TUI** — Multiple interfaces for workflow management

## Install

```bash
pip install animus-forge
```

## Quick Start

```bash
# Run a workflow
gorgon run workflows/examples/build_task.yaml

# Self-improve a codebase
gorgon self-improve run --provider ollama --path /my/project

# Analyze without making changes
gorgon self-improve analyze --focus security
```

## Self-Improvement Pipeline

The self-improve orchestrator runs a 10-stage workflow:

1. **Analyze** — Static analysis identifies improvement opportunities
2. **Plan** — AI generates an improvement plan from suggestions
3. **Safety check** — Validates against protected files and change limits
4. **Snapshot** — Creates rollback point before any changes
5. **Implement** — AI generates code changes
6. **Sandbox test** — Applies changes to temp copy, runs tests and lint
7. **Apply** — Writes changes to the actual codebase
8. **Create PR** — Creates a branch and pull request
9. **Human approval** — Waits for merge approval (skippable with `--auto-approve`)
10. **Rollback** — Automatic rollback if tests fail at any stage

## Part of the Animus Monorepo

- [Animus Core](https://pypi.org/project/animus-core/) — exocortex engine
- [Animus Quorum](https://pypi.org/project/convergentAI/) — coordination protocol
- [Animus Bootstrap](https://github.com/AreteDriver/animus/tree/main/packages/bootstrap) — system daemon

## License

MIT
