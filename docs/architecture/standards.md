# Standards

> Documentation, code, and commit conventions for the Animus project.

---

## Documentation Standards

### Markdown Style

- **Headers**: Start with `# Title`. Use `##` for sections.
- **Dates**: Add dates to time-sensitive claims: `(verified 2026-06-27)`
- **Links**: Relative paths for internal, full URLs for external
- **Code blocks**: Always specify language: ````python`, ````bash`
- **Line length**: Soft wrap (no hard limit for prose)

### Freshness

- Documents older than 90 days get a review-needed banner
- Quarterly full-site link validation
- Per-release changelog update

## Code Standards

### Python

- **Linter**: `ruff` (configured in `pyproject.toml`)
- **Line length**: 100 characters
- **Type hints**: Required for public APIs
- **Docstrings**: Google style

```toml
[tool.ruff]
target-version = "py310"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]
```

### Commits

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
docs: update architecture overview
feat(forge): add checkpoint resume
fix(core): handle missing config
refactor(bootstrap): split routes
chore: update dependencies
```

### PR Checklist

- [ ] Tests pass (`pytest`)
- [ ] Code style passes (`ruff`)
- [ ] Documentation updated
- [ ] No trailing whitespace in `.md` files
- [ ] Internal links resolve correctly
- [ ] Commit message follows conventional commits

---

## See Also

- [Contributing → Workflow](../contributing/workflow.md)
- [Contributing → Setup](../contributing/setup.md)
