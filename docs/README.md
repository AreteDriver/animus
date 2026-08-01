# Animus Documentation

> **What is Animus?** A Mind-class AI exocortex — persistent memory, multi-agent orchestration, and autonomous improvement.
> **Version**: 2.3.0 (migrating to v2.1 baseline) · **Tests**: 16,178+ · **License**: MIT

---

## Quickstart

New here? Start with one of these paths based on your goal:

| If you want to... | Go to |
|---|---|
| **Install and run Animus** | [Getting Started → Quickstart](getting-started/quickstart.md) |
| **Understand the system** | [Architecture → Overview](architecture/overview.md) |
| **Contribute code or docs** | [Contributing → Setup](contributing/setup.md) |
| **Deploy or operate Animus** | [Operators → Deployment](operators/deployment.md) |
| **Read about the v2.1 architecture** | [Architecture → Decisions](architecture/decisions/README.md) |

---

## Documentation Map

### Getting Started

- [Quickstart](getting-started/quickstart.md) — Install, configure, and run in under 10 minutes
- [Installation](getting-started/installation.md) — Per-package install instructions
- [Concepts](getting-started/concepts.md) — Core mental models: exocortex, forge, quorum, kernel

### Architecture

- [Overview](architecture/overview.md) — System context, four-layer stack, data flow
- [Packages](architecture/packages.md) — Dependency map and package responsibilities
- [Decisions](architecture/decisions/README.md) — ADRs and decision logs (ADL)
- [Standards](architecture/standards.md) — Documentation, code quality, and commit conventions

### Packages

Each package has its own documentation lane:

- [Core](packages/core/README.md) — Personal AI exocortex (`import animus`)
- [Forge](packages/forge/README.md) — Multi-agent orchestration (`import animus_forge`)
- [Bootstrap](packages/bootstrap/README.md) — System daemon and onboarding (`import animus_bootstrap`)
- [Quorum](packages/quorum/README.md) — Agent coordination protocol (`import animus_quorum`)
- [Kernel](packages/kernel/README.md) — Autonomous builder engine (standalone)
- [Types](packages/types/README.md) — Shared schema types
- [PWA](packages/pwa/README.md) — Progressive web app interface
- [Contracts](packages/contracts/README.md) — Canonical JSON schemas

### Contributing

- [Setup](contributing/setup.md) — Dev environment, Python versions, local LLM
- [Guidelines](contributing/guidelines.md) — Code style, PR process, testing
- [Workflow](contributing/workflow.md) — Git flow, CI expectations, release process
- [Debugging](contributing/debugging.md) — Common issues and diagnostics

### Operators

- [Deployment](operators/deployment.md) — Production deployment patterns
- [Configuration](operators/configuration.md) — Config files, secrets, environment variables
- [Monitoring](operators/monitoring.md) — Health checks, metrics, alerts
- [Troubleshooting](operators/troubleshooting.md) — Runbook for common failures

### Reference

- [Glossary](reference/glossary.md) — Domain terms and definitions
- [FAQ](reference/faq.md) — Frequently asked questions
- [Changelog](reference/changelog.md) — Release history
- [Security](reference/security.md) — Threat model, security layer, best practices
- [Schema Usage](reference/schema-usage.md) — JSON Schema contracts, Pydantic models, CI pipeline
- [Whitepapers](reference/whitepapers/README.md) — Architecture whitepapers

### Roadmap

- [Current](roadmap/current.md) — Active quarter priorities
- [Index](roadmap/README.md) — All roadmap documents

---

## Repository Entry Points

| File | Purpose |
|---|---|
| [Root README](https://github.com/your-org/animus/blob/main/README.md) | Project elevator pitch and badges |
| [CLAUDE.md](https://github.com/your-org/animus/blob/main/CLAUDE.md) | Session instructions for Claude Code |
| [Project Charter](https://github.com/your-org/animus/blob/main/PROJECT_CHARTER.md) | v2.1 scope, success criteria, risks |

---

## Status

- **Build**: Clean (0 warnings, 0 errors) — verified 2026-07-06
- **Deployment**: Blocked — repo is private, so GitHub Pages requires Pro. Cloudflare Pages workflow is ready (`.github/workflows/docs-deploy-cloudflare.yml`); set `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` secrets to enable.
- **Reorganization**: Active (see [docs/planning/documentation-roadmap.md](planning/documentation-roadmap.md)). If a link is broken or a section is missing, file an issue or submit a PR.
