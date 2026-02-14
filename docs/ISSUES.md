# Animus Known Issues & Gaps

Last updated: 2026-01-27 (v0.6.0)

## Testing Gaps

### ~~HIGH: No Offline/Mock LLM Testing~~ RESOLVED

**Status:** Resolved (v0.6.0) — `MockModel` added to `animus/cognitive.py`

**Solution implemented:**
- `MockModel(ModelInterface)` with configurable default response and response map
- `ModelConfig.mock()` classmethod for easy construction
- `MOCK` provider in `ModelProvider` enum + `create_model()` factory
- Call history recording (`calls: list[dict]`) for test assertions
- Tests in `tests/test_cognitive_mock.py` covering MockModel, CognitiveLayer, and DecisionFramework

---

### ~~MEDIUM: API Server Test Coverage~~ RESOLVED

**Status:** Resolved (v0.6.0) — `tests/test_api.py` added

**Solution implemented:**
- 30 tests covering all endpoints (status, chat, memory CRUD, tools, tasks CRUD, decisions, brief, integrations, learning, auth)
- Uses MockModel backend for deterministic responses
- Skips gracefully when FastAPI not installed

---

### ~~MEDIUM: Asyncio Deprecation Warnings~~ RESOLVED

**Status:** Resolved (v0.6.0) — All `asyncio.get_event_loop().run_until_complete()` replaced with `asyncio.run()`

**Files fixed:**
- `animus/__main__.py` — 11 call sites
- `tests/test_phase4.py` — 7 call sites
- All 6 previously failing tests now pass

---

### ~~LOW: Voice Interface Testing~~ RESOLVED

**Status:** Resolved — `tests/test_voice.py` added with `MockWhisperModel` and `MockSoundDevice`

**Solution implemented:**
- `MockWhisperModel` with configurable transcriptions and call history
- `MockSoundDevice` for recording/playback without hardware
- 15+ tests covering VoiceInput, VoiceOutput, and VoiceInterface
- Tests run in CI without audio dependencies

---

## API Inconsistencies

### ~~Memory.create() Missing~~ RESOLVED

**Status:** Resolved (v0.6.0) — `Memory.create()` classmethod added to `animus/memory.py`

**Solution:** Factory method auto-generates `id` (UUID), `created_at`, `updated_at`. All other fields have sensible defaults. Tests in `tests/test_core.py::TestMemoryCreate`.

---

### Inconsistent Method Names

**Problem:** Similar operations have different names across modules.

| Module | Add | Complete/Apply |
|--------|-----|----------------|
| TaskTracker | `add()` | `complete()` |
| MemoryStore | `store()` | N/A |
| LearningLayer | N/A | `approve_learning()` |

**Proposed:** Standardize on `add`/`remove` or document conventions.

**Priority:** Low - Documentation

---

## Feature Gaps

### ~~No Web Dashboard~~ RESOLVED

**Status:** Resolved — `animus/dashboard.py` + `/dashboard` endpoint added

**Solution implemented:**
- Single-page HTML dashboard with embedded CSS/JS served at `/dashboard`
- Shows: total learned, pending approvals, events today, guardrail violations
- Category and confidence distribution bar charts
- Tables for learned items, guardrails, and recent event history
- Auto-refreshes every 30 seconds

---

### ~~No Scheduled Tasks~~ RESOLVED

**Status:** Resolved (v0.6.0) — Background scheduler added to `LearningLayer`

**Solution:** `threading.Timer`-based scheduler with `start_auto_scan(interval_hours)`, `stop_auto_scan()`, and `auto_scan_running` property. Daemon thread, no external dependencies. Tests in `tests/test_phase5.py::TestAutoScanScheduler`.

---

### ~~No Data Export~~ RESOLVED

**Status:** Resolved — CLI commands and API endpoints added

**Solution implemented:**
- CLI: `/export-learnings [path]`, `/export-decisions [path]`, `/export-tasks [path]`
- API: `GET /export/memories`, `GET /export/learnings`, `GET /export/decisions`, `GET /export/tasks`
- All exports in JSON format with full metadata

---

## Performance

### ChromaDB Cold Start

**Problem:** First memory operation downloads 79MB embedding model.

**Impact:** ~3s delay on first use

**Mitigation:** Pre-download during install or first-run setup

**Priority:** Low - One-time cost

---

## Security

### ~~No Input Sanitization Audit~~ RESOLVED

**Status:** Resolved (v0.6.0) — Security hardening applied to `animus/tools.py`

**Changes:**
- `_validate_command()`: Whitespace normalization to prevent bypass; blocks `$()`, backtick subshells, `| sh`, `| bash`
- `_tool_read_file()`: Path resolution via `.resolve()` before I/O (prevents symlink traversal)
- `_tool_list_files()`: Same `.resolve()` fix
- `_tool_web_search()`: Control character stripping, 500-char query limit
- Tests in `tests/test_security.py::TestShellInjection` and `TestWebSearchSanitization`

---

## Next Actions

1. [x] Add MockModel for testing without LLM
2. [x] Add API integration tests
3. [x] Fix asyncio deprecation warnings
4. [x] Add Memory.create() factory method
5. [x] Implement background scheduler for auto-scan
6. [x] Security audit of tool inputs
7. [x] Encrypt integration credentials at rest
8. [x] Implement memory consolidation
9. [x] Implement incremental sync (since_version)
10. [x] Add data export (learnings, decisions, tasks)
11. [x] Build web dashboard for learning transparency
12. [x] Add voice testing infrastructure
13. [x] Add screenshots to README
