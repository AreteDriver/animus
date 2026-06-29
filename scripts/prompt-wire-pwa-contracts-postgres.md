# Senior Engineer Prompt: Wire PWA, Contracts, and PostgreSQL

## Objective

Make the Animus monorepo **public-ready** by wiring three critical gaps:

1. **PWA ↔ Bootstrap Backend**: The TypeScript PWA (`packages/pwa/`) has API client code calling `/api/...` endpoints, but the FastAPI backend (`packages/bootstrap/dashboard/`) serves routes at `/` with no `/api` prefix. Wire them together with a CORS-enabled proxy or prefix.
2. **Contracts Runtime Validation**: 20 JSON schemas exist in `packages/contracts/` and Pydantic models exist in `packages/types/`, but there's no runtime gate that validates incoming/outgoing data against schemas.
3. **PostgreSQL Connection**: `DurableMemoryStore` works with SQLite in tests but needs a real PostgreSQL setup path with Docker Compose, env var configuration, and a health-check script.

## Deliverables

### 1. PWA Integration (3 tasks)

**Task 1.1: API Prefix Alignment**

The PWA's `api.ts` uses `const BASE = "/api"`. The Bootstrap dashboard's FastAPI app registers routers at `/`. Options:

- **Option A**: Mount all Bootstrap routers under `/api` prefix in `app.py`
- **Option B**: Change PWA `BASE` to `""` and serve PWA static files from Bootstrap
- **Option C**: Vite proxy in dev mode + nginx rewrite in production

**Recommended**: Option A. Add an `/api` router prefix in `packages/bootstrap/src/animus_bootstrap/dashboard/app.py`, then serve the built PWA static files from `packages/pwa/dist/` via FastAPI's `StaticFiles`.

**Task 1.2: CORS Configuration**

Add CORS middleware to the Bootstrap dashboard so the PWA (running on `localhost:5173` in dev, or served from `/` in production) can call the API.

**Task 1.3: PWA Build Integration**

Add a build step that:
1. Builds the PWA (`npm run build` in `packages/pwa/`)
2. Copies `packages/pwa/dist/` to `packages/bootstrap/src/animus_bootstrap/dashboard/static/pwa/`
3. Serves it from FastAPI at `/pwa/` or `/`

Verify: Opening `http://localhost:7700/pwa/` shows the login screen and can authenticate.

### 2. Contracts Validation Layer (2 tasks)

**Task 2.1: Schema Validator Module**

Create `packages/contracts/src/animus_contracts/validator.py` that:
- Loads all `*.schema.json` files from the package
- Validates a Python dict against a named schema using `jsonschema`
- Raises `ValidationError` with structured details on failure

**Task 2.2: Integration Gate**

Wire the validator into the Bootstrap API layer as middleware or decorator:
- `POST /api/capture` validates against `action.schema.json` before processing
- `POST /api/conversations/messages` validates against `event.schema.json`
- Return HTTP 400 with the validation error details

Add tests in `packages/contracts/tests/test_validator.py` covering:
- Valid data passes
- Invalid data fails with correct schema name in error
- Missing required field is caught

### 3. PostgreSQL Wiring (3 tasks)

**Task 3.1: Environment Configuration**

Create `scripts/setup_postgres.py` that:
- Reads `DATABASE_URL` from env (or uses `infra/.env`)
- Verifies PostgreSQL is reachable via `pg_isready` or SQLAlchemy ping
- Runs `alembic upgrade head` in `database/`
- Prints connection status and schema version

**Task 3.2: Docker Compose Health Check**

Update `infra/docker-compose.yml` to:
- Start PostgreSQL with a named volume that persists across restarts
- Add a `healthcheck:` block using `pg_isready`
- Add a `wait-for-db.sh` style init so migrations don't race the DB startup

**Task 3.3: MemoryLayer Backend Selection**

Update `packages/kernel/src/animus_kernel/memory/layer.py` so that:
- `backend="durable"` connects to PostgreSQL via `DurableMemoryStore`
- The bootstrap config can set `memory.backend: durable` in YAML
- If PostgreSQL is unreachable, it falls back to `LocalMemoryStore` with a warning

Verify: Run `scripts/setup_postgres.py`, then run `python -c "from animus_kernel.memory import MemoryLayer; ml = MemoryLayer('/tmp/animus', backend='durable')"` successfully connects.

## Verification Checklist

- [ ] PWA loads at `http://localhost:7700/pwa/` and API calls succeed
- [ ] `POST /api/capture` with invalid JSON returns HTTP 400 with schema validation error
- [ ] PostgreSQL is running in Docker, migrations applied, kernel connects
- [ ] Truth baseline updated with new checks
- [ ] All new code has tests (aim for >80% coverage on new modules)

## Context

**PWA source**: `packages/pwa/src/` — React + Vite + TypeScript. Already has `api.ts`, `auth.ts`, and views for Login, Chat, Personas, Status.

**Bootstrap backend**: `packages/bootstrap/src/animus_bootstrap/dashboard/app.py` — FastAPI with HTMX templates and static file serving. Routers exist for conversations, capture, personas, etc.

**Contracts**: `packages/contracts/*.schema.json` — 20 JSON schemas. `packages/types/src/animus_types/*.py` — 24 generated Pydantic models from those schemas.

**PostgreSQL**: `database/migrations/versions/001_initial_schema.py` has `object_registry`, `event_ledger`, `traceability`. `infra/docker-compose.yml` has a PostgreSQL service. `DurableMemoryStore` in `packages/kernel/src/animus_kernel/memory/stores/durable.py` implements the SQLAlchemy ORM.

## Anti-Patterns to Avoid

- Don't hardcode credentials in any file committed to git
- Don't make PostgreSQL mandatory for local development — SQLite fallback must remain
- Don't break existing HTMX dashboard routes (the PWA is additive, not replacing)
- Don't add heavy frontend dependencies — keep the PWA lightweight

## Expected Output

A single commit with:
- `feat(pwa): integrate PWA with Bootstrap dashboard`
- `feat(contracts): add runtime JSON Schema validation`
- `feat(database): PostgreSQL setup script and Docker health checks`

And an updated `truth-baseline.toml` with checks for:
- `packages/contracts/src/animus_contracts/validator.py` exists
- `scripts/setup_postgres.py` exists
- PWA build step in CI or documented
