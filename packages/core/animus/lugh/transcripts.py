"""
Claude Code JSONL transcript harvester.

Parses `~/.claude/projects/<mangled-dir>/<uuid>.jsonl` and subagent
sidechains into Turn + SessionSummary objects for Animus memory
budgeting, Forge orchestration tuning, and Drift//Monitor ingestion.

Stdlib-only (sqlite3 via optional TranscriptCache). Zero API calls.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path

from animus.lugh.cache import TranscriptCache
from animus.lugh.models import SessionSummary, Turn

logger = logging.getLogger(__name__)

# Anthropic pricing (USD per 1M tokens). Longest-match-first via sorted keys.
MODEL_PRICING_LAST_UPDATED = "2026-04-17"
MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-7": {"input": 15.0, "output": 75.0, "cache_read": 1.50, "cache_write": 18.75},
    "claude-opus-4-6": {"input": 15.0, "output": 75.0, "cache_read": 1.50, "cache_write": 18.75},
    "claude-opus-4": {"input": 15.0, "output": 75.0, "cache_read": 1.50, "cache_write": 18.75},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0, "cache_read": 0.30, "cache_write": 3.75},
    "claude-sonnet-4-5": {"input": 3.0, "output": 15.0, "cache_read": 0.30, "cache_write": 3.75},
    "claude-sonnet-4": {"input": 3.0, "output": 15.0, "cache_read": 0.30, "cache_write": 3.75},
    "claude-haiku-4-5": {"input": 0.80, "output": 4.0, "cache_read": 0.08, "cache_write": 1.00},
    "claude-haiku-4": {"input": 0.80, "output": 4.0, "cache_read": 0.08, "cache_write": 1.00},
    "default": {"input": 3.0, "output": 15.0, "cache_read": 0.30, "cache_write": 3.75},
}

# Known JSONL line types — anything else surfaces as a schema-drift signal.
KNOWN_LINE_TYPES = {
    "user",
    "assistant",
    "attachment",
    "system",
    "file-history-snapshot",
    "queue-operation",
    "permission-mode",
    "last-prompt",
}
TURN_TYPES = {"user", "assistant"}

# Tool-driven activity labels (strong signal)
TOOL_ACTIVITY: list[tuple[str, tuple[str, ...]]] = [
    ("coding", ("Edit", "Write", "NotebookEdit", "str_replace")),
    ("file_read", ("Read",)),
    ("search", ("Grep", "Glob", "WebSearch", "WebFetch")),
    ("mcp_call", ("mcp__",)),
    ("planning", ("TodoWrite", "ExitPlanMode", "TaskCreate", "TaskUpdate")),
    ("exploration", ("Agent",)),
    ("shell", ("Bash",)),
]
# Text fallbacks — only used when no tools and no meaningful context.
TEXT_ACTIVITY: list[tuple[str, tuple[str, ...]]] = [
    ("debugging", ("traceback", "TypeError", "ValueError", "RuntimeError", "Exception")),
    ("review", ("reviewed", "audited", "evaluated")),
]

# Duration gap above this = user-away idle, excluded from active_seconds.
IDLE_GAP_SECONDS = 5 * 60

# Sequence-aware activity thresholds (applied in a second pass across turns).
DEBUG_LOOP_WINDOW = 5  # look back this many turns
DEBUG_LOOP_ERROR_HITS = 1  # need this many Bash error hits in window
EXPLORATION_READ_BURST = 5  # N Reads in a row ⇒ exploration (demote if pure)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _decode_project_dir(mangled: str) -> str:
    """'-home-arete-projects' -> '/home/arete/projects'."""
    if mangled.startswith("-"):
        return "/" + mangled.lstrip("-").replace("-", "/")
    return mangled


def _normalize_model(raw: str | None) -> tuple[str, str | None]:
    """Return (normalized_key, unknown_original_or_None).

    If the model string doesn't match any entry in MODEL_PRICING, the original
    string is returned as the second element so the caller can track it.
    """
    if not raw:
        return "default", None
    rm = raw.lower()
    for key in sorted(MODEL_PRICING, key=len, reverse=True):
        if key == "default":
            continue
        if key in rm:
            return key, None
    return "default", raw


def _parse_ts(raw) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None


def _tool_calls_from_content(content) -> list[str]:
    if not isinstance(content, list):
        return []
    return [
        b.get("name", "unknown")
        for b in content
        if isinstance(b, dict) and b.get("type") == "tool_use"
    ]


def _tool_results_from_content(content) -> list[dict]:
    """Extract tool_result blocks from user content (responses to prior tool calls)."""
    if not isinstance(content, list):
        return []
    return [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]


def _text_from_content(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    chunks = []
    for b in content:
        if isinstance(b, dict) and b.get("type") == "text":
            chunks.append(b.get("text", ""))
    return "\n".join(chunks)


def _base_activity(tools: list[str], text: str) -> str:
    """First-pass activity from tools alone, with text fallback."""
    for activity, prefixes in TOOL_ACTIVITY:
        if any(any(t.startswith(p) or t == p for p in prefixes) for t in tools):
            return activity
    blob = text.lower()
    for activity, keywords in TEXT_ACTIVITY:
        if any(k.lower() in blob for k in keywords):
            return activity
    return "conversation"


def _turn_cost(model: str, input_t: int, output_t: int, cache_r: int, cache_w: int) -> float:
    p = MODEL_PRICING.get(model, MODEL_PRICING["default"])
    return (
        input_t * p["input"] / 1_000_000
        + output_t * p["output"] / 1_000_000
        + cache_r * p["cache_read"] / 1_000_000
        + cache_w * p["cache_write"] / 1_000_000
    )


def _reclassify_sequences(turns: list[Turn], tool_errors: list[bool]) -> None:
    """Second-pass activity labels driven by surrounding turns.

    - `debugging`: Bash (shell) turn followed by a tool_result marked as error within
      DEBUG_LOOP_WINDOW turns of another Bash — the classic debug loop.
    - `exploration`: EXPLORATION_READ_BURST consecutive file_read turns upgrade to exploration.
    """
    # Debug loop detection: look for shell + error neighbors
    for i, t in enumerate(turns):
        if t.activity != "shell":
            continue
        lo = max(0, i - DEBUG_LOOP_WINDOW)
        hits = sum(
            1 for j in range(lo, min(i + DEBUG_LOOP_WINDOW + 1, len(tool_errors))) if tool_errors[j]
        )
        if hits >= DEBUG_LOOP_ERROR_HITS:
            t.activity = "debugging"

    # Exploration burst: EXPLORATION_READ_BURST file_read turns in close succession
    # upgrade to exploration. Conversation turns (including user tool_result turns)
    # are transparent — they don't break the run. Any non-read, non-conversation
    # tool activity resets the run.
    read_indices: list[int] = []
    for i, t in enumerate(turns):
        if t.activity == "file_read":
            read_indices.append(i)
        elif t.activity != "conversation":
            if len(read_indices) >= EXPLORATION_READ_BURST:
                for k in read_indices:
                    turns[k].activity = "exploration"
            read_indices = []
    # Flush trailing run
    if len(read_indices) >= EXPLORATION_READ_BURST:
        for k in read_indices:
            turns[k].activity = "exploration"


def _active_seconds(timestamps: list[datetime]) -> float | None:
    """Sum of inter-turn gaps below IDLE_GAP_SECONDS.

    Wall-clock (`duration_seconds`) still reports max-min; `active_seconds`
    ignores long idle stretches so "user walked away" doesn't inflate work time.
    """
    if len(timestamps) < 2:
        return None
    ts = sorted(timestamps)
    total = 0.0
    for a, b in zip(ts, ts[1:]):
        gap = (b - a).total_seconds()
        if 0 < gap <= IDLE_GAP_SECONDS:
            total += gap
    return total


# ---------------------------------------------------------------------------
# core parse
# ---------------------------------------------------------------------------


def parse_session(path: Path, claude_base: Path | None = None) -> SessionSummary | None:
    """Parse a single Claude Code JSONL session file (main or subagent).

    Args:
        path: Path to the .jsonl file.
        claude_base: Base projects dir (for computing relative session_file).

    Returns:
        SessionSummary, or None if no conversational turns were parseable.
    """
    base = claude_base or (Path.home() / ".claude" / "projects")
    try:
        rel = path.relative_to(base)
    except ValueError:
        rel = path
    project_dir = rel.parts[0] if rel.parts else "unknown"
    is_sidechain = "subagents" in path.parts
    agent_id = path.stem.replace("agent-", "") if is_sidechain else None

    turns: list[Turn] = []
    timestamps: list[datetime] = []
    cwd_counts: dict[str, int] = defaultdict(int)
    session_id: str | None = None
    unknown_line_types: set[str] = set()
    unknown_models: set[str] = set()
    tool_errors: list[bool] = []  # per-turn: did this turn carry a tool_result error?

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as e:
        logger.warning("read failed %s: %s", path, e)
        return None

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue

        if session_id is None:
            session_id = raw.get("sessionId")

        ltype = raw.get("type", "unknown")
        if ltype not in KNOWN_LINE_TYPES:
            unknown_line_types.add(ltype)
        if ltype not in TURN_TYPES:
            continue

        msg = raw.get("message") or {}
        usage = msg.get("usage") or {}
        ts = _parse_ts(raw.get("timestamp"))
        if ts:
            timestamps.append(ts)

        cwd = raw.get("cwd")
        if cwd:
            cwd_counts[cwd] += 1

        content = msg.get("content", [])
        tools = _tool_calls_from_content(content)
        text = _text_from_content(content)
        tool_results = _tool_results_from_content(content)
        tool_errors.append(any(tr.get("is_error") for tr in tool_results))
        mcp = [t for t in tools if t.startswith("mcp__")]
        activity = _base_activity(tools, text)

        model, unknown = _normalize_model(msg.get("model"))
        if unknown:
            unknown_models.add(unknown)

        inp = int(usage.get("input_tokens") or 0)
        out = int(usage.get("output_tokens") or 0)
        cr = int(usage.get("cache_read_input_tokens") or 0)
        cw = int(usage.get("cache_creation_input_tokens") or 0)

        turns.append(
            Turn(
                session_id=session_id or path.stem,
                turn_index=i,
                timestamp=ts,
                role=ltype,
                is_sidechain=bool(raw.get("isSidechain", is_sidechain)),
                agent_id=raw.get("agentId", agent_id),
                cwd=cwd,
                model=model,
                input_tokens=inp,
                output_tokens=out,
                cache_read_tokens=cr,
                cache_write_tokens=cw,
                tool_calls=tools,
                mcp_servers=mcp,
                activity=activity,
                cost_usd=_turn_cost(model, inp, out, cr, cw),
            )
        )

    if not turns:
        return None

    # Second pass: sequence-aware activity reclassification
    _reclassify_sequences(turns, tool_errors)

    activity_counts: dict[str, int] = defaultdict(int)
    model_dist: dict[str, int] = defaultdict(int)
    tot_in = tot_out = tot_cr = tot_cw = tot_tools = tot_mcp = 0
    tot_cost = 0.0
    for t in turns:
        activity_counts[t.activity] += 1
        model_dist[t.model] += 1
        tot_in += t.input_tokens
        tot_out += t.output_tokens
        tot_cr += t.cache_read_tokens
        tot_cw += t.cache_write_tokens
        tot_tools += len(t.tool_calls)
        tot_mcp += len(t.mcp_servers)
        tot_cost += t.cost_usd

    n = len(turns)
    duration = (max(timestamps) - min(timestamps)).total_seconds() if len(timestamps) >= 2 else None
    active = _active_seconds(timestamps)
    primary_cwd = max(cwd_counts.items(), key=lambda kv: kv[1])[0] if cwd_counts else None

    if unknown_line_types:
        logger.info("unknown line types in %s: %s", path.name, sorted(unknown_line_types))
    if unknown_models:
        logger.warning(
            "unpriced model(s) in %s (using default pricing): %s  "
            "— update MODEL_PRICING (last updated %s)",
            path.name,
            sorted(unknown_models),
            MODEL_PRICING_LAST_UPDATED,
        )

    return SessionSummary(
        session_id=session_id or path.stem,
        session_file=str(rel),
        is_sidechain=is_sidechain,
        agent_id=agent_id,
        project_dir=project_dir,
        primary_cwd=primary_cwd,
        date=min(timestamps).strftime("%Y-%m-%d") if timestamps else None,
        duration_seconds=duration,
        active_seconds=active,
        turn_count=n,
        tool_call_count=tot_tools,
        mcp_call_count=tot_mcp,
        model_distribution=dict(model_dist),
        total_input_tokens=tot_in,
        total_output_tokens=tot_out,
        total_cache_read_tokens=tot_cr,
        total_cache_write_tokens=tot_cw,
        total_cost_usd=round(tot_cost, 6),
        activity_breakdown={k: round(v / n * 100, 2) for k, v in activity_counts.items()},
        cwd_breakdown=dict(cwd_counts),
        unknown_line_types=sorted(unknown_line_types),
        unknown_models=sorted(unknown_models),
        turns=turns,
    )


# ---------------------------------------------------------------------------
# harvest (with cache)
# ---------------------------------------------------------------------------


def harvest_transcripts(
    claude_dir: Path | None = None,
    *,
    project: str | None = None,
    since: datetime | None = None,
    include_sidechains: bool = True,
    cache: TranscriptCache | None = None,
    use_cache: bool = True,
) -> Iterator[SessionSummary]:
    """Walk Claude Code transcripts and yield SessionSummary objects.

    Args:
        claude_dir: Override default ~/.claude/projects path.
        project: Substring filter on the file path (e.g., "monolith").
        since: Only yield sessions whose date is >= this datetime.
        include_sidechains: If False, skip subagent sidechain transcripts.
        cache: Optional TranscriptCache instance. If None and use_cache is True,
               a default cache at ~/.animus/lugh_cache.sqlite is used.
        use_cache: Disable the cache entirely (re-parse every file).
    """
    base = claude_dir or (Path.home() / ".claude" / "projects")
    if not base.exists():
        logger.warning("claude dir missing: %s", base)
        return

    active_cache: TranscriptCache | None = None
    if use_cache:
        active_cache = cache or TranscriptCache()

    for f in sorted(base.glob("**/*.jsonl")):
        if not include_sidechains and "subagents" in f.parts:
            continue
        if project and project not in str(f):
            continue

        rel = str(f.relative_to(base)) if f.is_relative_to(base) else str(f)
        try:
            mtime_ns = f.stat().st_mtime_ns
        except OSError as e:
            logger.warning("stat failed %s: %s", f, e)
            continue

        s: SessionSummary | None = None
        if active_cache is not None:
            s = active_cache.get(rel, mtime_ns)

        if s is None:
            s = parse_session(f, claude_base=base)
            if s is None:
                continue
            if active_cache is not None:
                active_cache.put(rel, mtime_ns, s)

        if since and s.date:
            s_dt = datetime.strptime(s.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if s_dt < since:
                continue
        yield s


def rollup_by_cwd(sessions: Iterable[SessionSummary]) -> dict[str, dict]:
    """Aggregate per-turn cwd → cost/tokens/turns.

    A single session can span multiple cwds; this attributes cost to the
    cwd active when each turn ran.
    """
    agg: dict[str, dict] = defaultdict(
        lambda: {
            "cost": 0.0,
            "turns": 0,
            "tools": 0,
            "mcp": 0,
            "input": 0,
            "output": 0,
            "cache_read": 0,
            "cache_write": 0,
            "sessions": set(),
        }
    )
    for s in sessions:
        for t in s.turns:
            cwd = t.cwd or "unknown"
            a = agg[cwd]
            a["cost"] += t.cost_usd
            a["turns"] += 1
            a["tools"] += len(t.tool_calls)
            a["mcp"] += len(t.mcp_servers)
            a["input"] += t.input_tokens
            a["output"] += t.output_tokens
            a["cache_read"] += t.cache_read_tokens
            a["cache_write"] += t.cache_write_tokens
            a["sessions"].add(t.session_id)
    return {
        cwd: {**v, "session_count": len(v["sessions"]), "sessions": None} for cwd, v in agg.items()
    }


def drift_signals(summary: SessionSummary) -> dict:
    """Drift//Monitor-compatible payload for a single SessionSummary."""
    n = summary.turn_count or 1
    conv_pct = summary.activity_breakdown.get("conversation", 0.0)
    coding_pct = summary.activity_breakdown.get("coding", 0.0)
    debug_pct = summary.activity_breakdown.get("debugging", 0.0)
    return {
        "source": "claude_transcripts",
        "session_id": summary.session_id,
        "session_file": summary.session_file,
        "is_sidechain": summary.is_sidechain,
        "primary_cwd": summary.primary_cwd,
        "date": summary.date,
        "signals": {
            "efficiency_drift": conv_pct,
            "coding_ratio": coding_pct,
            "debugging_ratio": debug_pct,
            "tool_use_ratio": round(summary.tool_call_count / n, 3),
            "mcp_call_count": summary.mcp_call_count,
            "total_cost_usd": summary.total_cost_usd,
            "model_distribution": summary.model_distribution,
            "turn_count": n,
            "wall_seconds": summary.duration_seconds,
            "active_seconds": summary.active_seconds,
            "cache_read_tokens": summary.total_cache_read_tokens,
            "cache_write_tokens": summary.total_cache_write_tokens,
            "unknown_line_types": summary.unknown_line_types,
            "unknown_models": summary.unknown_models,
        },
        "activity_breakdown": summary.activity_breakdown,
    }
