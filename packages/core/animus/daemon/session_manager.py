"""SessionManager: maintains warm sessions with persistence and context replay.

Enables daemon to resume interrupted work and maintain continuity
across restarts. Each warm session preserves conversation history,
tool registry state, and Meta-Thinker context.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from animus.logging import get_logger

logger = get_logger("daemon.session_manager")


@dataclass
class WarmSession:
    """A persisted session that can be resumed by the daemon.

    Captures the conversational and strategic state needed to continue
    work without starting from scratch.
    """

    session_id: str
    user_id: str = "default"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_active: str = field(default_factory=lambda: datetime.now().isoformat())

    # Conversation state
    messages: list[dict[str, Any]] = field(default_factory=list)
    current_iteration: int = 0
    original_prompt: str = ""
    accumulated_response: str = ""

    # Tool state snapshot
    tool_history: dict[str, list[dict]] = field(default_factory=dict)
    last_tools_used: list[str] = field(default_factory=list)

    # Meta-Thinker state
    meta_events: list[dict[str, Any]] = field(default_factory=list)
    pending_signals: list[dict[str, Any]] = field(default_factory=list)

    # Execution metadata
    total_tokens_used: int = 0
    total_iterations: int = 0
    is_complete: bool = False
    priority: str = "normal"  # normal, high, critical

    def touch(self) -> None:
        """Update last_active timestamp."""
        self.last_active = datetime.now().isoformat()

    def to_dict(self) -> dict:
        """Serialize to dict for JSON persistence."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "messages": self.messages,
            "current_iteration": self.current_iteration,
            "original_prompt": self.original_prompt,
            "accumulated_response": self.accumulated_response,
            "tool_history": self.tool_history,
            "last_tools_used": self.last_tools_used,
            "meta_events": self.meta_events,
            "pending_signals": self.pending_signals,
            "total_tokens_used": self.total_tokens_used,
            "total_iterations": self.total_iterations,
            "is_complete": self.is_complete,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WarmSession":
        """Deserialize from dict."""
        return cls(
            session_id=data["session_id"],
            user_id=data.get("user_id", "default"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            last_active=data.get("last_active", datetime.now().isoformat()),
            messages=data.get("messages", []),
            current_iteration=data.get("current_iteration", 0),
            original_prompt=data.get("original_prompt", ""),
            accumulated_response=data.get("accumulated_response", ""),
            tool_history=data.get("tool_history", {}),
            last_tools_used=data.get("last_tools_used", []),
            meta_events=data.get("meta_events", []),
            pending_signals=data.get("pending_signals", []),
            total_tokens_used=data.get("total_tokens_used", 0),
            total_iterations=data.get("total_iterations", 0),
            is_complete=data.get("is_complete", False),
            priority=data.get("priority", "normal"),
        )


class SessionManager:
    """Manages warm sessions for daemon operation.

    Provides session lifecycle, persistence, and replay for
    maintaining continuity across daemon restarts.
    """

    def __init__(self, persistence_dir: str | Path | None = None, max_sessions: int = 50):
        self.persistence_dir = Path(persistence_dir or "~/.animus/sessions").expanduser()
        self.persistence_dir.mkdir(parents=True, exist_ok=True)
        self.max_sessions = max_sessions
        self._sessions: dict[str, WarmSession] = {}
        self._load_existing()

    def _session_path(self, session_id: str) -> Path:
        return self.persistence_dir / f"{session_id}.json"

    def _load_existing(self) -> None:
        """Load all persisted sessions on startup."""
        if not self.persistence_dir.exists():
            return
        for path in self.persistence_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                session = WarmSession.from_dict(data)
                self._sessions[session.session_id] = session
                logger.debug(f"Loaded warm session: {session.session_id}")
            except Exception as e:
                logger.warning(f"Failed to load session from {path}: {e}")

    def create(self, user_id: str = "default", priority: str = "normal") -> WarmSession:
        """Create a new warm session."""
        session_id = f"session-{int(time.time()*1000)}-{user_id}"
        session = WarmSession(session_id=session_id, user_id=user_id, priority=priority)
        self._sessions[session_id] = session
        self._persist(session)
        logger.info(f"Created warm session: {session_id}")
        return session

    def get(self, session_id: str) -> WarmSession | None:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    def update(self, session: WarmSession) -> None:
        """Update and persist a session."""
        session.touch()
        self._sessions[session.session_id] = session
        self._persist(session)

    def _persist(self, session: WarmSession) -> None:
        """Write session to disk."""
        try:
            path = self._session_path(session.session_id)
            path.write_text(json.dumps(session.to_dict(), indent=2))
        except Exception as e:
            logger.error(f"Failed to persist session {session.session_id}: {e}")

    def complete(self, session_id: str) -> None:
        """Mark a session as complete."""
        session = self._sessions.get(session_id)
        if session:
            session.is_complete = True
            session.touch()
            self._persist(session)
            logger.info(f"Session completed: {session_id}")

    def close(self, session_id: str) -> None:
        """Close and optionally archive a session."""
        session = self._sessions.pop(session_id, None)
        if session:
            session.touch()
            self._persist(session)
            logger.info(f"Session closed: {session_id}")

    def prune_old_sessions(self, max_age_hours: int = 168) -> int:
        """Remove sessions older than max_age_hours.

        Returns number of pruned sessions.
        """
        from datetime import datetime, timedelta

        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        to_remove: list[str] = []
        for sid, session in self._sessions.items():
            try:
                last = datetime.fromisoformat(session.last_active)
                if last < cutoff:
                    to_remove.append(sid)
            except ValueError:
                to_remove.append(sid)

        for sid in to_remove:
            session = self._sessions.pop(sid, None)
            if session:
                path = self._session_path(sid)
                if path.exists():
                    path.unlink()
                logger.debug(f"Pruned old session: {sid}")

        # Also enforce max_sessions cap
        if len(self._sessions) > self.max_sessions:
            # Sort by last_active ascending, remove oldest
            sorted_sessions = sorted(
                self._sessions.items(),
                key=lambda x: x[1].last_active,
            )
            excess = len(sorted_sessions) - self.max_sessions
            for sid, _ in sorted_sessions[:excess]:
                self._sessions.pop(sid, None)
                path = self._session_path(sid)
                if path.exists():
                    path.unlink()
                logger.debug(f"Pruned session (cap): {sid}")

        return len(to_remove)

    def list_sessions(self, user_id: str | None = None, active_only: bool = False) -> list[WarmSession]:
        """List sessions, optionally filtered."""
        sessions = list(self._sessions.values())
        if user_id:
            sessions = [s for s in sessions if s.user_id == user_id]
        if active_only:
            sessions = [s for s in sessions if not s.is_complete]
        return sorted(sessions, key=lambda s: s.last_active, reverse=True)

    @property
    def active_count(self) -> int:
        return sum(1 for s in self._sessions.values() if not s.is_complete)

    @property
    def total_count(self) -> int:
        return len(self._sessions)