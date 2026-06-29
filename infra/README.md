# infra/

Deployment infrastructure for Animus — local development, CI, and production (self-hosted).

## Quick Start

```bash
cd infra
cp .env.example .env      # edit with your credentials
docker compose up -d
```

This starts PostgreSQL.  
To run migrations:
```bash
cd database && alembic upgrade head
```

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

Docker Compose for PostgreSQL added during Phase 2 (Durable Core). Remaining manifests (systemd, Terraform, Nix, monitoring) scheduled for Phase 4 (Public/Private Split).
