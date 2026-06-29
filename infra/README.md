# infra/

Deployment infrastructure for Animus — local development, CI, and production (self-hosted).

## What Goes Here

- **Docker Compose** (`docker-compose.yml`) — PostgreSQL, Redis, Ollama, Animus services
- **systemd units** (`systemd/`) — service definitions for daemon deployment on Linux workstations
- **Terraform / OpenTofu** (`terraform/`) — cloud resources (optional, for public split)
- **Nix** (`nix/`) — reproducible development shells (optional)
- **Monitoring configs** (`prometheus/`, `grafana/`) — dashboards and alerts

## Boundary vs deploy/

`deploy/` contains **runtime binaries and systemd unit files** for the current bootstrap daemon.
`infra/` contains **infrastructure-as-code** — Docker, Terraform, monitoring, the full stack definition.

## Owner

AreteDriver

## Status

Scaffolded — no manifests yet. Will be populated during Phase 2 (Durable Core) and Phase 4 (Public/Private Split) of the roadmap.
