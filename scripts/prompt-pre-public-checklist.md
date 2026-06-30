# Senior Engineer Prompt — Pre-Public Checklist

> **Context**: Animus monorepo is functionally wired (PWA ↔ Bootstrap, Contracts runtime validation, PostgreSQL auto-backend). One config bug (`extra="ignore"`) is fixed. Before making the repo public, execute the following checklist with zero regressions.

---

## 1. Secrets Audit (Final Pass)

**Goal**: Confirm no credentials, tokens, or placeholders leaked into the git history.

**Steps**:
1. Run `git log --all --full-history --source -S 'sk-ant' -S 'ghp_' -S 'sk-' -- '*.md' '*.py' '*.toml' '*.ini' '*.yaml' '*.yml'`
2. Run `git log --all -p | grep -iE 'password|secret|token|api_key|private_key' | head -50`
3. Run `gitleaks detect --source . --verbose` (if installed; if not, skip with note).
4. Check `alembic.ini` still has no real credentials (only the placeholder comment we added).
5. Check `.env` files are in `.gitignore` and never committed.
6. Check `secrets.yaml` / `secrets.env` are in `.gitignore`.

**Deliverable**: A one-line verdict — "Clean" or a list of files/commits to purge.

---

## 2. Truth Baseline Run

**Goal**: Confirm the full test suite is green (or document expected failures).

**Steps**:
1. Run `pytest` from repo root across all packages, or package-by-package:
   - `packages/core/tests/`
   - `packages/kernel/tests/`
   - `packages/bootstrap/tests/`
   - `packages/forge/tests/`
   - `packages/quorum/tests/`
2. Capture the summary: `X passed, Y failed, Z skipped`.
3. For any failure, determine if it is:
   - A regression from the wiring work (fix it)
   - A pre-existing flake (document it)
   - An expected failure (e.g. version_alignment mismatch — document it)

**Deliverable**: Test summary with verdict.

---

## 3. End-to-End Smoke Tests

**Goal**: Verify the three wired gaps actually work in a running system.

### 3a. PWA Static Files + API
1. Ensure `packages/pwa/dist/` exists and is current (`npm run build` in `packages/pwa/`).
2. Start Bootstrap: `animus-bootstrap serve` (or `python -m animus_bootstrap.dashboard.app` for dev).
3. `curl -s http://localhost:7700/pwa/ | head -5` → should return the built `index.html`.
4. `curl -s http://localhost:7700/api/health` → should return JSON with `status: ok`.
5. `curl -s -X POST http://localhost:7700/api/conversations/messages -H 'Content-Type: application/json' -d '{"text":"hello"}'` → should return `{"text":...}`.

### 3b. PostgreSQL Auto-Backend
1. Start PostgreSQL: `cd infra && docker compose up -d` (ensure `.env` exists with credentials).
2. Export `ANIMUS_DATABASE_URL` (see `infra/.env.example` for format).
3. Run `python scripts/setup_postgres.py` → should pass all checks.
4. Start Bootstrap with the env var set.
5. Verify in logs that `MemoryLayer` initialized with `DurableMemoryStore` (not `LocalMemoryStore`).
6. Stop PostgreSQL container, unset env var, restart Bootstrap → should fall back to `LocalMemoryStore` with a warning log.

### 3c. Contracts Runtime Validation
1. Install contracts package: `pip install -e packages/contracts/`.
2. Start Bootstrap with contracts installed.
3. `curl -s -X POST http://localhost:7700/api/capture -H 'Content-Type: application/json' -d '{"text":""}'` → should return 400 (empty capture, not schema error).
4. Temporarily decorate `/api/capture` with `@validate_contract("action")` in a local test file, POST `{"text":"hi"}` → should return 422 with schema errors.
5. Remove the temporary decorator.

**Deliverable**: A markdown table with test case, command, expected result, actual result.

---

## 4. README Refresh

**Goal**: The public README is accurate, honest, and gets a new user from zero to running in < 10 minutes.

**Required updates**:
1. **Platform scope**: State "Linux-only for public open-source launch. macOS support is on the roadmap; Windows is out of scope." (per ADL-20260629-002).
2. **Prerequisites**: List Python 3.11+, Node 18+, Docker (optional, for PostgreSQL), Ollama (optional, for local LLM).
3. **Quick start**:
   ```bash
   git clone https://github.com/AreteDriver/Animus.git
   cd Animus
   pip install -e packages/bootstrap/ -e packages/kernel/ -e packages/core/ -e packages/contracts/
   cd packages/pwa && npm install && npm run build && cd ../..
   animus-bootstrap serve
   # Open http://localhost:7700/pwa/ on your phone (same Wi-Fi)
   ```
4. **PostgreSQL (optional, recommended for production)**:
   ```bash
   cp infra/.env.example infra/.env   # fill in credentials
   docker compose -f infra/docker-compose.yml up -d
   export ANIMUS_DATABASE_URL=$(cat infra/.env | grep URL)  # or set manually
   python scripts/setup_postgres.py
   ```
5. **Architecture overview**: One ASCII diagram or bullet list showing the 8 packages and how they connect (Bootstrap → Kernel → Core, PWA → Bootstrap API, etc.).
6. **What works / what is experimental**: Be honest. Mark PWA as "functional but early", Web Push as "scaffolded", Forge as "active development".
7. **Remove any outdated claims**: Check for stale version numbers, wrong package names, or references to deleted features.

**Deliverable**: Updated `README.md` committed with `docs:` prefix.

---

## 5. Repo Settings & Public Flip

**Goal**: Make the repo public without leaking anything.

**Steps**:
1. Confirm the `.gitignore` covers: `.env`, `*.env`, `secrets.*`, `dist/` (only if we don't want to commit built PWA; but we may want to for GitHub Pages — decide and document).
2. Confirm no GitHub Actions secrets are in the repo (check `.github/workflows/` for hardcoded tokens).
3. Go to GitHub repo Settings → General → Danger Zone → Change visibility → Public.
4. After flip, verify GitHub Pages deploys from `gh-pages` branch (or `main` if using Actions) and the MkDocs site is live.
5. If Pages deploy fails because repo is still private, this step is the fix.

**Deliverable**: Repo is public, GitHub Pages URL is confirmed working.

---

## Execution Rules

- **Do not commit until the user says "commit"**.
- **Fix regressions immediately** — do not skip a failing test.
- **Document expected failures** inline with a comment like `# expected: version_alignment mismatch`.
- **Stop and ask** if a smoke test reveals a wiring gap we missed.
- **Update MEMORY.md** and `README.md` wins board when done.
