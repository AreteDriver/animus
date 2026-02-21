# Animus Forge Self-Improvement Subsystem: Deep Review

**Date:** 2026-02-21
**Reviewer:** Claude Opus 4.6
**Scope:** `packages/forge/src/animus_forge/self_improve/`

---

## 1. ARCHITECTURE: Approval Flow

### Traced Path: Detection to Application

The orchestrator (`orchestrator.py`) defines a 10-stage linear workflow. Here is the actual flow as implemented in `SelfImproveOrchestrator.run()`:

```
                         +-------------------+
                         |   IDLE (start)    |
                         +--------+----------+
                                  |
                                  v
                   +-----------------------------+
                   |  Stage 1: ANALYZING          |
                   |  CodebaseAnalyzer.analyze()   |
                   |  Static checks only (regex)   |
                   |  (AI analysis NOT called here)|
                   +-------------+---------------+
                                 |
                      no suggestions?
                       /          \
                     yes           no
                      |             |
                 +----v----+        |
                 | COMPLETE |        |
                 |(no work) |        |
                 +----------+        |
                                     v
                   +-----------------------------+
                   |  Stage 2: PLANNING            |
                   |  _create_plan(top 5 items)    |
                   |                               |
                   |  SafetyChecker.check_changes() |
                   |  (file protection, limits,     |
                   |   category checks)             |
                   +-------------+---------------+
                                 |
                        blocking violations?
                         /           \
                       yes            no
                        |              |
                   +----v----+         |
                   | FAILED   |         |
                   |(safety)  |         |
                   +----------+         |
                                        v
                   +-----------------------------+
                   |  Stage 3: PLAN APPROVAL       |
                   |  (if config.human_approval_   |
                   |   plan)                       |
                   |  ApprovalGate                  |
                   |    .request_approval()         |
                   |                               |
                   |  NOT ASYNC - returns           |
                   |  immediately with              |
                   |  "Plan approval required"      |
                   +-------------+---------------+
                                 |
                                 v
                   +-----------------------------+
                   |  Stage 4: IMPLEMENTING        |
                   |  *** STUB ***                 |
                   |  logger.info("would happen")  |
                   |  No actual code generation    |
                   +-------------+---------------+
                                 |
                                 v
                   +-----------------------------+
                   |  Stage 5: TESTING             |
                   |  *** HARDCODED SUCCESS ***    |
                   |  SandboxResult(SUCCESS,       |
                   |    tests_passed=True,         |
                   |    lint_passed=True)           |
                   |  Sandbox class NOT used here  |
                   +-------------+---------------+
                                 |
                                 v
                   +-----------------------------+
                   |  Stage 6: APPLY APPROVAL      |
                   |  (if config.human_approval_   |
                   |   apply)                      |
                   |  Same pattern as Stage 3      |
                   +-------------+---------------+
                                 |
                                 v
                   +-----------------------------+
                   |  Stage 7: SNAPSHOT            |
                   |  RollbackManager              |
                   |    .create_snapshot()          |
                   |  Reads file contents to disk  |
                   +-------------+---------------+
                                 |
                                 v
                   +-----------------------------+
                   |  Stage 8: APPLYING            |
                   |  *** STUB ***                 |
                   |  "Would actually apply here"  |
                   |  No file writes happen        |
                   +-------------+---------------+
                                 |
                                 v
                   +-----------------------------+
                   |  Stage 9: CREATE PR           |
                   |  PRManager.create_branch()    |
                   |  PRManager.create_pr()         |
                   |  Real git commands (checkout   |
                   |  -b, gh pr create)             |
                   |  But no files were changed!   |
                   +-------------+---------------+
                                 |
                                 v
                   +-----------------------------+
                   |  Stage 10: MERGE APPROVAL     |
                   |  (if config.human_approval_   |
                   |   merge)                      |
                   |  Returns with PR URL          |
                   +-----------------------------+
                                 |
                                 v
                         +-------------------+
                         |     COMPLETE       |
                         +-------------------+
```

### Critical Observation

**Stages 4, 5, and 8 are stubs.** The orchestrator logs "Implementation would happen here" at Stage 4 and "Would actually apply changes here" at Stage 8. Stage 5 constructs a hardcoded `SandboxResult(status=SUCCESS, tests_passed=True, lint_passed=True)` without invoking the `Sandbox` class. The `analyze_with_ai()` method exists on `CodebaseAnalyzer` but the orchestrator calls `self.analyzer.analyze()` (the non-AI variant) instead.

The entire middle of the pipeline -- the part that actually generates and applies code changes -- does not exist. The safety infrastructure is built around a hole where the actual self-modification engine should be.

---

## 2. SANDBOX: Containment Analysis

### How It Works

`sandbox.py` creates an isolated copy of the codebase in a temporary directory:

**Filesystem scope:**
- Uses `tempfile.mkdtemp(prefix="gorgon_sandbox_")` in `/tmp`
- Copies entire source tree via `shutil.copytree`, excluding `.git`, `.venv`, `__pycache__`, `*.pyc`, `.mypy_cache`, `.pytest_cache`, `node_modules`, `dist`, `build`
- Supports context manager with `cleanup_on_exit` flag
- Cleanup uses `shutil.rmtree(ignore_errors=True)`

**Process scope:**
- Runs commands via `asyncio.create_subprocess_exec` with `cwd` set to sandbox path
- Captures stdout/stderr via `PIPE`
- Enforces timeout via `asyncio.wait_for` (default 300 seconds)
- Kills process on timeout

**Network scope:**
- **NONE.** Zero network isolation.

### What Escapes the Sandbox

1. **Network access** -- Full network. No Docker, no `unshare`, no `seccomp`, no `firejail`.
2. **Filesystem beyond sandbox** -- Full filesystem namespace. Can read `~/.ssh`, `~/.aws`, `/etc/passwd`.
3. **Process namespace** -- Can see/signal other processes, spawn background daemons.
4. **Environment variables** -- Inherits `ANTHROPIC_API_KEY`, `GITHUB_TOKEN`, etc.
5. **IPC** -- Shared memory, Unix sockets, D-Bus all accessible.

**Verdict:** Filesystem-copy isolation only. No actual security containment.

---

## 3. ROLLBACK: Mechanism and Failure Modes

### What It Is

File-content snapshot system:
1. **Snapshot creation**: Reads text content of specified files, saves to `.gorgon/snapshots/{id}/`. Filenames flattened (`/` to `_`). `files.json` index + global `index.json`.
2. **Rollback**: Reads stored content, writes back to codebase. Creates parent dirs if needed.
3. **Cleanup**: Auto-removes snapshots beyond `max_snapshots` (default 10), FIFO.

### What It Is NOT

- **Not Git-based.** No `git stash/checkout/reset`.
- **Not process-state aware.** No service restarts, module reloads, cache invalidation.
- **Not atomic.** Files restored one at a time. Crash mid-rollback = partial state.

### Failure Modes

1. **Partial rollback on crash**: Some files written, others not. No two-phase commit.
2. **New files not tracked**: Rollback only restores snapshot files. New files created by self-improvement are not deleted.
3. **Binary files not supported**: `read_text()`/`write_text()` only. Binary files silently skipped.
4. **Snapshot storage corruption**: Corrupted `index.json` = all metadata lost, silently continues with empty list.
5. **Race conditions**: No file locking.
6. **Filename collision**: `src/foo/bar.py` and `src/foo_bar.py` both flatten to `src_foo_bar.py`.

---

## 4. SAFETY CONSTRAINTS: Config Analysis

### Rules from `config/self_improve_safety.yaml`

**Protected (critical) files -- CANNOT be modified:**
- `src/animus_forge/auth/**`, `src/animus_forge/security/**`
- `src/animus_forge/self_improve/**` (recursive self-modification blocked)
- `.env*`, `config/self_improve_safety.yaml`
- `**/credentials*`, `**/secrets*`, `**/*.key`, `**/*.pem`

**Sensitive files -- require extra review (warning, not blocking):**
- `src/animus_forge/api.py`, `src/animus_forge/config.py`
- `pyproject.toml`, `Dockerfile`, `docker-compose*.yml`
- `.github/workflows/**`

**Quantitative limits:**
- Max 10 files per PR, 500 lines changed, 0 deleted files, 5 new files

**Approval gates:** Human approval required at all 3 stages (plan, apply, merge).

**Allowed categories:** refactoring, bug_fixes, documentation, test_coverage, performance, code_quality.

**Denied categories:** security_changes, auth_changes, config_schema_changes, database_migrations.

### Identity File Protection Gap

**FLAG: Identity files are NOT protected.** The safety config does not protect:
- Agent behavior files (`agents/supervisor.py`, `agents/provider_wrapper.py`)
- Workflow and skill definitions (`workflows/**`, `skills/**`)
- CLI entry points (`cli/**`)
- Persona/identity configuration in the broader monorepo
- `pyproject.toml` is only "sensitive" (warning), not "critical" (blocking)

The analyzer could categorize a modification to `supervisor.py` as "refactoring" and it would pass all safety checks.

---

## 5. GAPS: Production Readiness

### Showstoppers

1. **No code generation engine.** Stages 4, 5, 8 are stubs. The system can detect problems and plan fixes but cannot modify a single line of code.
2. **Sandbox provides no security.** Full user privileges, full network, full filesystem.
3. **Approval gate has no waiting mechanism.** Returns immediately. No polling, webhook, or notification integration.
4. **Rollback doesn't handle new files.** Created files persist after rollback.

### Significant Gaps

5. **Analyzer uses regex only in default mode.** `analyze_with_ai()` exists but is never called by orchestrator.
6. **File-level snapshot naming collision.** Path flattening can cause silent data loss.
7. **PR creation without changes.** Stage 9 creates empty PRs since Stage 8 is stubbed.
8. **`auto_approve` bypasses all human gates.** No enforcement preventing production use.
9. **Outer `try/except Exception` masks bugs.** All errors caught, logged, and swallowed.
10. **No rate limiting.** No cooldown, daily limits, or cost tracking.
11. **No diff content review.** Safety checker operates on file paths and line counts, not actual diff content.
12. **AI analysis truncates files.** Only first 5000 chars read for AI review.
13. **No persistent audit log.** Approval history is in-memory only, lost on restart.
14. **Environment variable inheritance in sandbox.** Secrets visible to sandboxed code.

---

## 6. INTEGRATION POINT: Core Learning vs. Forge Self-Improve

### What Each System Does

| Dimension | Core Learning | Forge Self-Improve |
|-----------|--------------|-------------------|
| **Target** | User model (preferences, patterns) | Source code files |
| **Input** | Episodic/semantic memory | Python AST, file content |
| **Output** | Learned items, preferences | Code changes, PRs |
| **Rollback** | Unlearn items by ID | Restore file contents |
| **Storage** | JSON files | File snapshots + Git |
| **Approval** | Category-based (auto/notify/confirm/approve) | Stage-based (plan/apply/merge) |
| **Guardrails** | Content-based (no harm, no exfiltrate) | File-based (protected paths, line limits) |

### Relationship: Complementary Layers, Not Parallel

These are **complementary systems** operating at different levels:
- **Core Learning** learns about the USER (preferences, patterns, corrections)
- **Forge Self-Improve** modifies the CODEBASE (code quality, refactoring, PRs)

### Integration Gaps

1. Core Learning preferences don't feed into Forge analyzer priorities
2. Forge improvement results don't reinforce Core learned patterns
3. Core's `CORE_GUARDRAILS` and Forge's `SafetyConfig` are disconnected
4. Both have `ApprovalRequest`/`ApprovalStatus` but incompatible schemas
5. Both have `RollbackManager` but fundamentally different operations (logical vs physical)

### Natural Integration

Core Learning drives the "what" (what should be improved). Forge Self-Improve drives the "how" (how to implement it). The missing piece is a shared intent/suggestion protocol between the two packages.

---

## Summary

| Area | Status | Assessment |
|------|--------|------------|
| Architecture | Designed, partially implemented | 3 of 10 stages are stubs -- the core implementation/testing/application stages |
| Sandbox | Code exists, usable | No real security isolation -- temp dir copy only, full user privileges |
| Rollback | Functional for what it does | File-content snapshots work, but no atomicity, no new-file cleanup |
| Safety Config | Well-designed | Good auth/secrets/self-modify protection. Gap: identity files not protected |
| Production Readiness | Not ready | No code generation, no real approval waiting, no audit persistence |
| Core Integration | Not connected | Complementary designs with overlapping concepts but zero wiring |

**Bottom line:** The self-improvement subsystem is a well-structured skeleton. The safety and approval infrastructure is thoughtfully designed with defense-in-depth. But the actual engine that generates, tests, and applies code changes does not exist. The system can detect problems, plan fixes, and create empty PRs -- but it cannot modify a single line of code autonomously.
