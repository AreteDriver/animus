"""JSON session persistence for the TUI."""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

HISTORY_DIR = Path.home() / ".gorgon" / "history"

# Restrictive permissions for the history directory and session files.
_DIR_MODE = 0o700  # rwx------
_FILE_MODE = 0o600  # rw-------


class TUISession:
    """Persistent chat session stored as JSON."""

    def __init__(
        self,
        session_id: str | None = None,
        title: str = "untitled",
        provider: str = "",
        model: str = "",
        system_prompt: str | None = None,
        messages: list[dict] | None = None,
        file_context: list[str] | None = None,
        total_tokens: dict | None = None,
        created_at: str | None = None,
        updated_at: str | None = None,
    ):
        self.id = session_id or str(uuid.uuid4())
        self.title = title
        self.provider = provider
        self.model = model
        self.system_prompt = system_prompt
        self.messages = messages or []
        self.file_context = file_context or []
        self.total_tokens = total_tokens or {"input": 0, "output": 0}
        now = datetime.now(UTC).isoformat()
        self.created_at = created_at or now
        self.updated_at = updated_at or now

    @classmethod
    def create(
        cls,
        provider: str = "",
        model: str = "",
        system_prompt: str | None = None,
        messages: list[dict] | None = None,
    ) -> TUISession:
        """Create a new session."""
        title = _auto_title(messages)
        return cls(
            provider=provider,
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            title=title,
        )

    @property
    def filepath(self) -> Path:
        slug = _slugify(self.title)
        date = self.created_at[:10]
        return HISTORY_DIR / f"{date}_{slug}.json"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "provider": self.provider,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "messages": self.messages,
            "file_context": self.file_context,
            "total_tokens": self.total_tokens,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def save(self) -> Path:
        """Save session to JSON file with restrictive permissions."""
        self.updated_at = datetime.now(UTC).isoformat()
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        # Ensure the directory is only accessible by the owner.
        try:
            os.chmod(HISTORY_DIR, _DIR_MODE)
        except OSError:
            pass  # Best-effort cleanup: chmod may fail on non-POSIX filesystems

        path = self.filepath
        path.write_text(json.dumps(self.to_dict(), indent=2))
        # Restrict the file so only the owner can read/write it.
        try:
            os.chmod(path, _FILE_MODE)
        except OSError:
            pass  # Best-effort cleanup: chmod may fail on non-POSIX filesystems
        return path

    @classmethod
    def load(cls, path: Path) -> TUISession:
        """Load session from JSON file."""
        data = json.loads(path.read_text())
        return cls(
            session_id=data.get("id"),
            title=data.get("title", "untitled"),
            provider=data.get("provider", ""),
            model=data.get("model", ""),
            system_prompt=data.get("system_prompt"),
            messages=data.get("messages", []),
            file_context=data.get("file_context", []),
            total_tokens=data.get("total_tokens", {"input": 0, "output": 0}),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )

    @classmethod
    def list_sessions(cls) -> list[Path]:
        """List all saved session files, sorted by name (date prefix)."""
        if not HISTORY_DIR.exists():
            return []
        return sorted(HISTORY_DIR.glob("*.json"))


def _slugify(text: str) -> str:
    """Convert text to filesystem-safe slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower().strip())
    return slug[:50].strip("-") or "session"


def _auto_title(messages: list[dict] | None) -> str:
    """Generate a title from the first user message."""
    if not messages:
        return "untitled"
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            title = content[:60].replace("\n", " ").strip()
            return title if title else "untitled"
    return "untitled"
