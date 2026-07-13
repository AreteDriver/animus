# Workflow Spec: `animus.workflows.ingest`

> **Artifact**: Thin deterministic orchestration module composing Lugh → Ogma → Memory tagging into a single callable pipeline and CLI command.  
> **For**: Animus users and daemon jobs who want one-shot content ingestion without manual stage chaining.  
> **Why now**: Every ingestion currently requires `yt-dlp` + manual `synthesize()` + manual memory tagging. The gap between fetch and knowledge capture is leakage.

---

## 1. Requirements

1. **MUST** expose a programmatic entry point `ingest(url: str, *, synthesize: bool = False, tag: bool = False, model: ModelInterface | None = None, min_relevance: float = 0.5) -> IngestResult`.
2. **MUST** expose a CLI entry point `animus ingest <url> [--synthesize] [--tag]`.
3. **MUST** call `animus.lugh.sources.resolve_source(url)` to determine fetcher, then `fetcher.fetch(url)` to produce a `SourceItem`.
4. **MUST** cache the fetched `SourceItem` via `animus.lugh.cache.Cache.store(item)` before any downstream stage runs.
5. **MUST** call `animus.ogma.read.synthesize(item, ...)` only when `synthesize=True`.
6. **MUST** forward `min_relevance` to Ogma synthesis; forward optional `model` to Ogma's `_resolve_provider()`.
7. **MUST** push structured concepts to Animus semantic memory only when `tag=True` and `synthesis` is non-None.
8. **MUST NOT** import or reference `quorum`, `convergent`, or any multi-agent coordination types.
9. **MUST** default Ogma provider to local Ollama (`ModelConfig.ollama()`); remote OpenRouter tier is opt-in via `ANIMUS_OGMA_REMOTE` env var (preserving ADL-20260511-001 behavior).
10. **MAY** support batch ingestion (`list[str]` urls) by iterating serially; parallelization is out of scope.
11. **MUST NOT** fork, vendor, or modify Lugh or Ogma internals; only compose their public interfaces.

---

## 2. Constraints

| # | Constraint | Measurement Method |
|---|-----------|-------------------|
| C1 | Python 3.12+ only | `python --version` ≥ 3.12 in CI runner |
| C2 | No new external dependencies beyond existing Animus core set | `pip freeze` diff against `packages/core/pyproject.toml` |
| C3 | No Rust / PyO3 compilation required | `cargo` not invoked during install; `pip install -e packages/core` succeeds without Rust toolchain |
| C4 | Lugh-only failure must still return `IngestResult` with `success=False` and populated `errors` | Unit test: mock Lugh to raise `SourceError`; assert `IngestResult.success is False` and `errors[0].stage == "lugh"` |
| C5 | Ogma failure must not invalidate Lugh cache | Unit test: mock Ogma to raise `OgmaSynthesisError`; assert cached file still exists in `~/.animus/lugh_raw/` |
| C6 | CLI exit code 0 on success, 1 on any fatal failure (Lugh fetch impossible) | Shell test: `animus ingest bad-url` returns exit code 1 |
| C7 | CLI exit code 0 on partial success (Lugh OK, Ogma failed) with stderr warning | Shell test: mock Ogma failure; assert exit code 0 and stderr contains "ogma_failed" |

---

## 3. Interfaces

### 3.1 Consumed — Lugh

```python
from animus.lugh.sources.base import SourceItem, resolve_source
from animus.lugh.cache import Cache
```

- `resolve_source(url: str) -> SourceFetcher` — returns a fetcher capable of handling the URL scheme (YouTube, arXiv, RSS, etc.)
- `SourceFetcher.fetch(url: str) -> SourceItem` — performs network I/O, produces structured item
- `Cache.store(item: SourceItem) -> None` — writes to `~/.animus/lugh_raw/<source_type>/`

**Error cases:**
- `resolve_source` raises `ValueError` → unresolvable URL scheme (fatal, `success=False`)
- `SourceFetcher.fetch` raises network error → fatal, `success=False`
- `Cache.store` raises disk error → fatal, `success=False`

### 3.2 Consumed — Ogma

```python
from animus.ogma.read import synthesize
from animus.cognitive import ModelInterface
```

- `synthesize(item: SourceItem, *, model: ModelInterface | None = None, repo_root: Path | None = None, min_relevance: float = DEFAULT_MIN_RELEVANCE, relevance_score: float | None = None, concept_hint: str | None = None, output_dir: Path | None = None) -> OgmaOutput | None`

**Error cases:**
- `OgmaSynthesisError` → non-fatal; Lugh data is preserved, error recorded in `IngestResult.errors`
- Returns `None` (relevance gated out) → non-fatal; `synthesis` field is `None`

### 3.3 Consumed — Memory (concept tagging)

```python
from animus.memory import tag_concepts  # or equivalent MCP/memory API
```

- Accepts `OgmaOutput` or parsed concepts → writes semantic memories
- **Error cases:** Memory write failure → non-fatal; error recorded in `IngestResult.errors`

### 3.4 Emitted — `IngestResult`

```python
from dataclasses import dataclass, field
from typing import Literal

@dataclass(frozen=True)
class WorkflowError:
    stage: Literal["lugh", "ogma", "memory"]
    error_type: str
    message: str

@dataclass
class IngestResult:
    item: SourceItem | None
    synthesis: OgmaOutput | None = None
    memory_tags: list[str] | None = None  # list of memory IDs
    errors: list[WorkflowError] = field(default_factory=list)
    success: bool = False
```

**Behavioral contract:**
- `item` is `None` only when Lugh fetch failed (fatal).
- `synthesis` is `None` when `synthesize=False` or Ogma failed or gated out.
- `memory_tags` is `None` when `tag=False` or memory write failed.
- `success` is `True` when Lugh fetch and cache succeeded, regardless of downstream stages.
- `errors` is empty when all requested stages succeeded.

---

## 4. Acceptance Criteria

| # | Test | How to Verify |
|---|------|-------------|
| A1 | `test_ingest_youtube_full_pipeline` | Pass a real YouTube URL with `synthesize=True, tag=True`. Assert `IngestResult.item` is non-None. Assert `IngestResult.item.raw_text` contains transcript text. Assert cache file exists in `~/.animus/lugh_raw/youtube/`. Assert `IngestResult.success is True`. |
| A2 | `test_ingest_lugh_only` | Call with `synthesize=False, tag=False`. Assert `IngestResult.synthesis is None`, `IngestResult.memory_tags is None`, `IngestResult.success is True`. |
| A3 | `test_ingest_ogma_failure_is_partial` | Mock Ogma to raise `OgmaSynthesisError`. Assert `IngestResult.success is True` (Lugh won). Assert `len(IngestResult.errors) == 1` with `stage="ogma"`. Assert Lugh cache file still exists on disk. |
| A4 | `test_ingest_cli_invocation` | Invoke `animus ingest <url> --synthesize --tag` as subprocess. Assert exit code 0. Assert stdout (or written file) contains expected result fields. |
| A5 | `test_ingest_reuses_cached_item` | Call `ingest()` twice on same URL. Assert second call skips network fetch (cache hit) — verify via mock or filesystem mtime. |
| A6 | `test_ingest_invalid_url` | Pass malformed/unresolvable URL. Assert graceful failure with `IngestResult.success is False` and `errors[0].stage == "lugh"`. |
| A7 | `test_ingest_cli_partial_exit_code` | Mock Ogma failure. Run CLI with `--synthesize`. Assert exit code 0 and stderr contains warning text indicating Ogma failure. |
| A8 | `test_ingest_memory_tag_failure_is_partial` | Mock memory tagging to raise. Assert `IngestResult.success is True`, `IngestResult.memory_tags is None`, `errors[0].stage == "memory"`. |

---

## 5. Out of Scope

- **Quorum integration** — no multi-agent consensus, voting, or intent graph writes.
- **Distributed execution** — runs local-only; no remote worker dispatch.
- **Web UI / HTTP API** — CLI + programmatic API only.
- **Real-time streaming** — Ogma synthesis output is buffered, not streamed.
- **Automatic URL discovery / watchlist polling** — Lugh's `watchlist.py` already handles this; not duplicated.
- **Changes to Ogma markdown contract** — Ogma's `PERSONA_SYSTEM_PROMPT` and `OgmaOutput` schema are untouched.
- **Changes to Lugh cache format** — `Cache.store()` behavior is not modified.
- **Parallel batch processing** — out of scope for v1; serial iteration acceptable.
- **Re-synthesis on cache hit** — if item already cached and `synthesize=True`, Ogma still runs; no "skip Ogma if synthesis exists" logic (that is a future enhancement).

---

## 6. Open Questions

| # | Question | Status |
|---|---------|--------|
| Q1 | Should `IngestResult` include a `cached: bool` field to indicate cache hits? | **Deferred** — add if observability needs it after first week of usage. |
| Q2 | Should the CLI support `--output-dir` for Ogma synthesis results? | **Deferred** — use Ogma default (`~/projects/notes/ogma/`) unless user feedback demands override. |
| Q3 | Which exact memory tagging API to call (`animus.memory.tag_concepts` vs. `mcp__animus__animus_remember`)? | **Decision needed** — depends on whether this runs inside Claude Code session (MCP available) or as standalone daemon (must use Python SDK). Default to Python SDK for daemon compatibility. |

---

## 7. File Layout (Proposed)

```
packages/core/animus/
├── workflows/
│   ├── __init__.py          # exports IngestResult, ingest, WorkflowError
│   └── ingest.py            # core implementation
├── cli.py                   # add `animus ingest` subcommand here
└── ... (existing modules)
```

No new packages or repos. Pure composition of existing `lugh`, `ogma`, `memory` surfaces.

---

*Spec version: 1.0*  
*Generated: 2026-06-29*  
*Decision basis: ADL-20260621-001 (hybrid pre-build workflow), Ogma PR #83 (provider resolution), Lugh canonical corpus v1.0*

---

## 8. Media Pipeline Extension (Research Guild Integration)

> Added: 2026-07-13 · ADL-20260713-001

The `animus_media_pipeline` MCP tool extends the basic ingest flow with conditional downstream analysis via the Research Guild citizen pipeline.

### 8.1 Flow

```
Lugh Harvest → Ogma Synthesize → MechanismCard extraction
                                    ↓
                           [OgmaOutput.animus_gap]
                                    ↓
                         ┌──────────┼──────────┐
                        NONE     PARTIAL      FULL
                         ↓          ↓          ↓
                    Store only   Store +    Store + full RG
                               PatternCit.   (Pattern → FP → Arch)
                                    ↓          ↓
                                              ProposalQueue
```

### 8.2 Gating Rules

| `animus_gap` | Downstream stages | ProposalQueue? |
|---|---|---|
| `NONE` | Store Ogma + MechanismCards only | No |
| `PARTIAL` | Store + PatternCitizen (media-tuned dedup) | No |
| `FULL` | Store + Pattern → FP → Architecture → `ImprovementProposal` | Yes (priority 3) |

Override: `run_research_guild=True` forces full pipeline regardless of gap status.

### 8.3 MCP Tool

```python
# Via MCP server (animus/mcp_server.py)
animus_media_pipeline(
    url: str,                          # YouTube playlist/channel URL
    source_type: str = "auto",        # "auto" | "youtube_playlist" | "youtube_channel" | "podcast"
    run_research_guild: bool = False,  # Force full RG pipeline
    store_outputs: bool = True,        # Persist to semantic memory
)
```

Returns a `MediaPipelineReport` markdown summary with gap status, stage results, and (when applicable) the generated proposal ID.

### 8.4 Daemon Integration

The P3 daemon's `TaskScheduler` supports recurring media scans via cron expressions:

```python
from animus.daemon.scheduler import TaskScheduler
from animus.citizens.media import MediaPipelineOrchestrator

scheduler = TaskScheduler(persistence_dir="/var/lib/animus/tasks")
task = MediaPipelineOrchestrator.schedule_scan(
    scheduler=scheduler,
    url="https://youtube.com/playlist?list=PLabc",
    source_type="youtube_playlist",
    cron_expression="0 9 * * 1",   # Mondays at 9 AM
    run_research_guild=False,
    list_limit=25,
)
```

The daemon's `_execute_task_background()` dispatches `task_type == "media_pipeline"` by calling `MediaPipelineOrchestrator.run()` with the task metadata.

### 8.5 Key Files

| File | Role |
|---|---|
| `animus/citizens/media.py` | `MediaPipelineOrchestrator`, `MediaPipelineReport`, `MediaHarvester`, `MediaSynthesizer`, `MediaAbstractionAdapter` |
| `animus/daemon/core.py` | Daemon dispatch for `"media_pipeline"` task type |
| `animus/mcp_server.py` | `animus_media_pipeline` MCP tool registration |
| `tests/test_media_pipeline.py` | 32 tests covering orchestration, gating, dedup, ProposalQueue wiring, and daemon scheduler wiring |
