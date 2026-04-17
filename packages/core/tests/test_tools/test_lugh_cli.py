"""Tests for animus.lugh.cli entry point."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from animus.lugh.cli import main


def _write_session(tmp_path: Path) -> Path:
    base = tmp_path / "projects"
    d = base / "-home-arete-projects"
    d.mkdir(parents=True)
    (d / "sess-1.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "user",
                        "sessionId": "sess-1",
                        "timestamp": "2026-04-10T09:00:00Z",
                        "cwd": "/home/arete/projects/monolith",
                        "message": {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "sessionId": "sess-1",
                        "timestamp": "2026-04-10T09:00:05Z",
                        "cwd": "/home/arete/projects/monolith",
                        "message": {
                            "role": "assistant",
                            "model": "claude-opus-4-6",
                            "content": [{"type": "text", "text": "ok"}],
                            "usage": {"input_tokens": 1, "output_tokens": 1},
                        },
                    }
                ),
            ]
        )
        + "\n"
    )
    return base


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    """Redirect Path.home() so the CLI's default cache + claude_dir land in tmp."""
    fake_home = tmp_path / "home"
    (fake_home / ".animus").mkdir(parents=True, exist_ok=True)
    (fake_home / ".claude" / "projects").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(Path, "home", lambda: fake_home)


class TestCLI:
    def test_summary_runs_when_no_sessions(self, capsys):
        rc = main([])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Sessions:" in out

    def test_json_output(self, tmp_path, capsys):
        # Write a session directly into the fake ~/.claude/projects tree
        fake_home = Path.home()
        projects = fake_home / ".claude" / "projects" / "-home-arete-projects"
        projects.mkdir(parents=True, exist_ok=True)
        (projects / "sess-1.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "user",
                            "sessionId": "sess-1",
                            "timestamp": "2026-04-10T09:00:00Z",
                            "cwd": "/home/arete/projects/monolith",
                            "message": {
                                "role": "user",
                                "content": [{"type": "text", "text": "hi"}],
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "type": "assistant",
                            "sessionId": "sess-1",
                            "timestamp": "2026-04-10T09:00:05Z",
                            "cwd": "/home/arete/projects/monolith",
                            "message": {
                                "role": "assistant",
                                "model": "claude-opus-4-6",
                                "content": [{"type": "text", "text": "ok"}],
                                "usage": {"input_tokens": 1, "output_tokens": 1},
                            },
                        }
                    ),
                ]
            )
            + "\n"
        )
        rc = main(["--json", "--no-cache"])
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["sessions"] == 1
        assert "activity_pct" in data

    def test_session_not_found(self, capsys):
        rc = main(["--session", "nonexistent-uuid", "--no-cache"])
        assert rc == 1
        assert "no session" in capsys.readouterr().out.lower()

    def test_rollup_flag_accepts(self, capsys):
        rc = main(["--rollup", "--no-cache"])
        assert rc == 0

    def test_invalid_since_passed_through(self, capsys):
        # Argparse does not validate the date; main parses it — bad date raises
        with pytest.raises(ValueError):
            main(["--since", "not-a-date"])
