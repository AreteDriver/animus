# Animus Known Issues & Gaps

Last updated: 2026-01-13 (v0.5.0)

## Testing Gaps

### HIGH: No Offline/Mock LLM Testing

**Problem:** Core cognitive features require an LLM backend (Ollama or API key) to test.

**Affected Components:**
- `CognitiveLayer.think()` - Cannot unit test reasoning
- `CognitiveLayer.think_with_tools()` - Cannot test agentic loop
- `DecisionFramework.analyze()` - Requires model for analysis
- Full CLI conversation loop

**Proposed Solution:**
- Add `MockModel` implementation in `cognitive.py`
- Enable via `ANIMUS_MOCK_MODEL=true` or config flag
- Return deterministic responses for testing

**Priority:** High - Blocks CI/CD testing

---

### MEDIUM: API Server Test Coverage

**Problem:** API server (`api.py`) not covered by automated tests.

**Current State:**
- Manual testing only
- No integration tests for endpoints

**Proposed Solution:**
- Add `tests/test_api.py` using `httpx` + `TestClient`
- Test all endpoints with mocked backends
- Add to CI pipeline

**Priority:** Medium

---

### MEDIUM: Asyncio Deprecation Warnings

**Problem:** `asyncio.get_event_loop()` deprecation warnings in:
- `__main__.py:365` - Integration reconnect
- `test_phase4.py` - Filesystem integration tests

**Fix:**
```python
# Replace:
asyncio.get_event_loop().run_until_complete(coro)

# With:
asyncio.run(coro)
# Or use asyncio.new_event_loop() explicitly
```

**Priority:** Medium - Will break in Python 3.12+

---

### LOW: Voice Interface Testing

**Problem:** Voice interface requires audio hardware, cannot test in CI.

**Affected Components:**
- `VoiceInput` - Microphone capture
- `VoiceOutput` - Audio playback
- Whisper transcription

**Proposed Solution:**
- Add audio file input mode for testing
- Mock `sounddevice` for unit tests
- Integration test with pre-recorded audio files

**Priority:** Low - Hardware dependent

---

## API Inconsistencies

### Memory.create() Missing

**Problem:** `Memory` dataclass has no `create()` factory method unlike other dataclasses.

**Current:** Must specify all fields manually including `id`, `created_at`, `updated_at`

**Fix:** Add factory method:
```python
@classmethod
def create(cls, content: str, memory_type: MemoryType, ...) -> "Memory":
    now = datetime.now()
    return cls(
        id=str(uuid.uuid4()),
        created_at=now,
        updated_at=now,
        ...
    )
```

**Priority:** Low - Convenience improvement

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

### No Web Dashboard

**Problem:** Learning transparency data has no visual interface.

**Current:** CLI-only via `/learning` command

**Proposed:**
- Simple HTML dashboard served by API
- Or separate React/Vue frontend
- Real-time updates via WebSocket

**Priority:** Medium - Usability

---

### No Scheduled Tasks

**Problem:** Auto-scan for learning patterns requires manual trigger.

**Config exists:** `auto_scan_interval_hours: 24`

**Missing:** Background scheduler to trigger `learning.scan_and_learn()`

**Proposed:** Add `apscheduler` or simple threading timer

**Priority:** Medium - Core feature incomplete

---

### No Data Export

**Problem:** No way to export memories, learnings, or decisions.

**Proposed:**
- `/export memories` - JSON/CSV export
- `/export learnings` - Learning history
- `/export decisions` - Decision log
- API endpoints for same

**Priority:** Low - Nice to have

---

## Performance

### ChromaDB Cold Start

**Problem:** First memory operation downloads 79MB embedding model.

**Impact:** ~3s delay on first use

**Mitigation:** Pre-download during install or first-run setup

**Priority:** Low - One-time cost

---

## Security

### No Input Sanitization Audit

**Problem:** Tool inputs (file paths, commands) not audited for injection.

**At Risk:**
- `run_command` tool - Shell injection
- `read_file` tool - Path traversal
- `web_search` tool - URL injection

**Proposed:** Security audit of all tool implementations

**Priority:** High - Security

---

## Next Actions

1. [ ] Add MockModel for testing without LLM
2. [ ] Add API integration tests
3. [ ] Fix asyncio deprecation warnings
4. [ ] Add Memory.create() factory method
5. [ ] Implement background scheduler for auto-scan
6. [ ] Security audit of tool inputs
