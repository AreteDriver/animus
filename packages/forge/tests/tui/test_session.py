"""Tests for TUI session persistence."""

from __future__ import annotations

import json

from animus_forge.tui.session import (
    HISTORY_DIR,
    TUISession,
    _auto_title,
    _slugify,
)


class TestSlugify:
    def test_basic(self):
        assert _slugify("Hello World") == "hello-world"

    def test_special_chars(self):
        assert _slugify("Test!@#$%Workflow") == "test-workflow"

    def test_truncation(self):
        result = _slugify("a" * 100)
        assert len(result) <= 50

    def test_empty(self):
        assert _slugify("") == "session"

    def test_only_special_chars(self):
        assert _slugify("!!!") == "session"

    def test_leading_trailing_dashes(self):
        result = _slugify("--hello--")
        assert not result.startswith("-")
        assert not result.endswith("-")


class TestAutoTitle:
    def test_from_first_user_message(self):
        msgs = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a language."},
        ]
        assert _auto_title(msgs) == "What is Python?"

    def test_truncates_long_content(self):
        msgs = [{"role": "user", "content": "x" * 100}]
        assert len(_auto_title(msgs)) <= 60

    def test_strips_newlines(self):
        msgs = [{"role": "user", "content": "line1\nline2\nline3"}]
        result = _auto_title(msgs)
        assert "\n" not in result

    def test_no_messages(self):
        assert _auto_title(None) == "untitled"
        assert _auto_title([]) == "untitled"

    def test_no_user_message(self):
        msgs = [{"role": "assistant", "content": "Hi"}]
        assert _auto_title(msgs) == "untitled"

    def test_empty_user_content(self):
        msgs = [{"role": "user", "content": ""}]
        assert _auto_title(msgs) == "untitled"


class TestTUISession:
    def test_create(self):
        session = TUISession.create(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert session.provider == "anthropic"
        assert session.model == "claude-sonnet-4-20250514"
        assert session.title == "Hello"
        assert len(session.id) > 0
        assert session.created_at is not None

    def test_create_no_messages(self):
        session = TUISession.create()
        assert session.title == "untitled"
        assert session.messages == []

    def test_to_dict(self):
        session = TUISession(
            session_id="test-id",
            title="test",
            provider="openai",
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
        )
        d = session.to_dict()
        assert d["id"] == "test-id"
        assert d["title"] == "test"
        assert d["provider"] == "openai"
        assert d["model"] == "gpt-4o"
        assert len(d["messages"]) == 1
        assert "created_at" in d
        assert "updated_at" in d

    def test_filepath(self):
        session = TUISession(title="my test session")
        fp = session.filepath
        assert fp.parent == HISTORY_DIR
        assert fp.suffix == ".json"
        assert "my-test-session" in fp.name

    def test_save_and_load(self, tmp_path, monkeypatch):
        monkeypatch.setattr("animus_forge.tui.session.HISTORY_DIR", tmp_path)
        session = TUISession(
            session_id="save-test",
            title="save test",
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "hello"}],
            system_prompt="Be helpful.",
        )
        path = session.save()
        assert path.exists()

        # Verify JSON content
        data = json.loads(path.read_text())
        assert data["id"] == "save-test"
        assert data["title"] == "save test"
        assert data["system_prompt"] == "Be helpful."

        # Load it back
        loaded = TUISession.load(path)
        assert loaded.id == "save-test"
        assert loaded.title == "save test"
        assert loaded.provider == "anthropic"
        assert loaded.system_prompt == "Be helpful."
        assert len(loaded.messages) == 1

    def test_save_updates_timestamp(self, tmp_path, monkeypatch):
        monkeypatch.setattr("animus_forge.tui.session.HISTORY_DIR", tmp_path)
        session = TUISession(title="ts test")
        old_updated = session.updated_at
        session.save()
        assert session.updated_at >= old_updated

    def test_save_creates_directory(self, tmp_path, monkeypatch):
        subdir = tmp_path / "nested" / "history"
        monkeypatch.setattr("animus_forge.tui.session.HISTORY_DIR", subdir)
        session = TUISession(title="mkdir test")
        path = session.save()
        assert subdir.exists()
        assert path.exists()

    def test_list_sessions_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr("animus_forge.tui.session.HISTORY_DIR", tmp_path)
        assert TUISession.list_sessions() == []

    def test_list_sessions(self, tmp_path, monkeypatch):
        monkeypatch.setattr("animus_forge.tui.session.HISTORY_DIR", tmp_path)
        # Create a few sessions
        for i in range(3):
            s = TUISession(title=f"session {i}")
            s.save()
        sessions = TUISession.list_sessions()
        assert len(sessions) == 3
        # Should be sorted
        names = [p.name for p in sessions]
        assert names == sorted(names)

    def test_list_sessions_nonexistent_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("animus_forge.tui.session.HISTORY_DIR", tmp_path / "does_not_exist")
        assert TUISession.list_sessions() == []

    def test_file_permissions(self, tmp_path, monkeypatch):
        monkeypatch.setattr("animus_forge.tui.session.HISTORY_DIR", tmp_path)
        session = TUISession(title="perms test")
        path = session.save()
        # File should be owner-only read/write
        mode = oct(path.stat().st_mode & 0o777)
        assert mode == "0o600"

    def test_total_tokens_default(self):
        session = TUISession()
        assert session.total_tokens == {"input": 0, "output": 0}

    def test_file_context_field(self):
        session = TUISession(file_context=["/tmp/a.py", "/tmp/b.py"])
        d = session.to_dict()
        assert d["file_context"] == ["/tmp/a.py", "/tmp/b.py"]
