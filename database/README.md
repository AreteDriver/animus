# database/

Database schemas, migrations, and seed data for the Animus durable core.

## What Goes Here

- **Alembic migrations** (`migrations/versions/`) — PostgreSQL schema evolution
- **Schema DDL** (`schema/`) — CREATE TABLE scripts for object registry, event ledger, bitemporal projections
- **Seed data** (`seeds/`) — reference data, demo fixtures
- **Connection utilities** (`pool.py`, `session.py`) — SQLAlchemy or asyncpg helpers

## Technology

- **Primary**: PostgreSQL 15+ (durable authority)
- **Projections**: SQLite (lightweight, offline-capable views)
- **Migration tool**: Alembic

## Boundary vs packages/

`packages/kernel/` and `packages/core/` contain the **ORM models and query logic**.
`database/` contains the **migration definitions and raw DDL** — this is the schema-of-record for operators and DBAs.

## Owner

AreteDriver

## Status

Scaffolded — no migrations yet. Will be populated during Phase 2 (Durable Core) of the roadmap.
