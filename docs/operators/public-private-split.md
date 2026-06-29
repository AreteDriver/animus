# Public / Private Repository Split

**Status**: Planned — awaiting repo publicization or private repo creation  
**Owner**: AreteDriver  
**Last updated**: 2026-06-29

---

## Rationale

The Animus monorepo mixes three categories of content:

1. **Public-safe**: Source code, tests, documentation, JSON schemas, synthetic fixtures
2. **Owner-private**: Personal memory dumps, API keys, local config, eval results containing PII
3. **Infrastructure**: Deployment configs, CI/CD, Docker Compose (public-safe but may contain hostnames)

A single private repo blocks:
- External contributors (can't see code)
- Free GitHub Pages hosting (requires public repo or Pro)
- Community trust (opaque development)

The split solves this by isolating owner-private data into a separate repo while making the code public.

---

## Two-Repo Model

### Public Repo (`github.com/AreteDriver/animus`)

**What stays here**:

| Category | Examples | Rationale |
|---|---|---|
| Source code | All `packages/*/` | Core value, needs community eyes |
| Tests | `packages/*/tests/` | Confidence in correctness |
| Documentation | `docs/`, READMEs, ADRs | Transparency |
| JSON schemas | `packages/contracts/*.schema.json` | Interop standard |
| Synthetic fixtures | `packages/core/tests/fixtures/` | Deterministic, no PII |
| Infrastructure code | `infra/`, `database/` | Reproducible deployments |
| CI/CD configs | `.github/workflows/` | Public audit trail |

**What gets REMOVED before publicizing**:

| Category | Examples | Mitigation |
|---|---|---|
| Real user data | `user_data/`, `personal/` | Already `.gitignore`d, verify no history |
| API keys / secrets | `.env`, `secrets.yaml`, `token.json` | Already `.gitignore`d, rotate keys |
| Eval results with PII | Local eval dumps | Move to private repo |
| Private worktrees | `.claude/worktrees/` | Already `.gitignore`d |

### Private Repo (`github.com/AreteDriver/animus-private`)

**What goes here**:

| Category | Examples |
|---|---|
| Owner-specific config | `local.yaml`, `.env` overrides |
| Real memory dumps | SQLite/ChromaDB exports |
| Eval results with PII | Conversation logs, judge outputs |
| Production secrets | TLS certs, DB passwords, API keys |
| Personal dashboards | HTMX templates with owner data |
| Backup snapshots | `evidence/releases/` (optional — can stay public) |

---

## Boundary Rules

### Rule 1: No PII in public

If a file contains any of the following, it belongs in the private repo:
- Real names, addresses, emails
- Conversation transcripts
- API keys, tokens, passwords
- Production database dumps
- Memory contents from actual usage

### Rule 2: Fixtures must be synthetic

All fixtures in the public repo must be:
- Hand-crafted (not scraped from real usage)
- Clearly labeled with `_meta.purpose` and `synthetic: true`
- Reviewable in full (no large binary blobs)

### Rule 3: Config templates in public, values in private

Public repo contains `.env.example`, `config/default.yaml`.  
Private repo contains `.env` (real values) and `config/local.yaml`.

### Rule 4: Evidence bundles are public

Evidence bundles (`evidence/releases/`) contain test output and metadata — no PII. They stay in the public repo as proof of build quality.

---

## Git Hygiene Before Publicizing

```bash
# 1. Verify no secrets in history
git log --all --full-history -- .env
git log --all --full-history -- secrets.yaml
git log --all --full-history -- token.json

# 2. Check for credential patterns
git log --all -p | grep -E 'sk-|ghp_|api_key|password'

# 3. If secrets found in history, use BFG or filter-repo to purge
git filter-repo --path secrets.yaml --invert-paths

# 4. Verify .gitignore blocks common secret files
grep -E '\.env|secrets|token|key' .gitignore
```

---

## Options for Docs Deployment

Since the repo is currently private, GitHub Pages is unavailable without a Pro plan.

| Option | Effort | Cost | Drawbacks |
|---|---|---|---|
| **Make repo public** | Low | Free | All history exposed (must audit first) |
| **GitHub Pro** | Low | $4/mo | Simplest if keeping private |
| **Netlify** | Medium | Free tier | Separate build pipeline |
| **Cloudflare Pages** | Medium | Free tier | Git integration required |
| **Self-host (Nginx)** | High | Server cost | Maintenance burden |

**Recommendation**: Make repo public after completing the hygiene checklist above. This is the fastest path to free Pages hosting and external contributions.

---

## Migration Steps (When Ready)

1. **Audit** — Run the hygiene checklist, purge secrets from history if needed
2. **Branch** — Create `public/` branch with private data removed
3. **Filter** — Use `git filter-repo` to strip `user_data/`, `personal/`, and any secret files from history
4. **Make public** — Change repo visibility in GitHub settings
5. **Create private repo** — `animus-private` for owner data
6. **Submodule** (optional) — Link private repo into public repo at `private/` if needed
7. **Deploy docs** — Enable GitHub Pages on `main` branch `/ (root)`

---

## See Also

- [Migration Guide](migration-guide.md) — Step-by-step split procedure
- `COMPATIBILITY_MATRIX.md` — Package dependency graph
- `truth-baseline.toml` — Automated verification that public repo has no private data
