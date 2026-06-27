# Development Workflow

> How we work: branches, commits, PRs, and releases.

---

## Branches

Use these prefixes:

| Prefix | Use for |
|---|---|
| `feat/` | New features |
| `fix/` | Bug fixes |
| `docs/` | Documentation changes |
| `refactor/` | Code restructuring |
| `chore/` | Maintenance, deps, CI |

Example: `feat/memory-tiering`, `docs/api-reference`, `fix/budget-leak`

## Commits

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
docs: update architecture overview for v2.1
feat(forge): add resume-from-checkpoint
fix(core): handle missing config gracefully
refactor(bootstrap): split dashboard routes
```

## Pull Requests

1. Create a feature branch
2. Make focused changes (one concern per PR)
3. Run tests locally: `pytest packages/<pkg>/tests/ -v`
4. Run lint: `ruff check packages/ && ruff format --check packages/`
5. Update docs if your change affects behavior
6. Open PR with clear description
7. Wait for review (even if CI is blocked by billing)

## CI

The CI runs:
- Lint (`ruff`)
- Tests per package (Python 3.10–3.12)
- Security scan (`gitleaks`)
- Docs validation (link checker, trailing whitespace)

If CI is blocked by GitHub billing, verify locally before merging.

---

## See Also

- [Setup](setup.md) — Dev environment
- [Debugging](debugging.md) — Common issues
- [Architecture → Standards](../architecture/standards.md) — Detailed standards
