"""Tests for SessionController and session lifecycle management.

Covers SessionPolicy resolution, limit detection, graceful finalize,
and restart logic without requiring a live Ollama instance.
"""

from __future__ import annotations

import tempfile
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from animus_kernel.head.checkpoint import HeadCheckpoint, HeadCheckpointStore
from animus_kernel.head.context_manager import HeadContextManager
from animus_kernel.head.repl import HeadREPL
from animus_kernel.head.session_controller import (
    SessionController,
    SessionLifecycleEvent,
    SessionPolicy,
)

# ------------------------------------------------------------------
# SessionPolicy
# ------------------------------------------------------------------


class TestSessionPolicy:
    def test_defaults(self) -> None:
        policy = SessionPolicy()
        assert policy.wrapup_threshold == 0.96
        assert policy.session_timer == timedelta(minutes=30)
        assert policy.auto_restart is True
        assert "concise summary" in policy.wrapup_prompt

    def test_token_wrapup_enabled(self) -> None:
        assert SessionPolicy(wrapup_threshold=0.96).token_wrapup_enabled is True
        assert SessionPolicy(wrapup_threshold=1.0).token_wrapup_enabled is False

    def test_timer_enabled(self) -> None:
        assert SessionPolicy(session_timer=timedelta(minutes=30)).timer_enabled is True
        assert SessionPolicy(session_timer=None).timer_enabled is False
        assert SessionPolicy(session_timer=timedelta()).timer_enabled is False

    def test_resolve_for_model(self) -> None:
        override = SessionPolicy(wrapup_threshold=0.80, session_timer=timedelta(minutes=15))
        base = SessionPolicy(
            wrapup_threshold=0.96,
            session_timer=timedelta(minutes=30),
            model_overrides={"qwen2.5-coder:14b": override},
        )

        resolved = base.resolve_for_model("qwen2.5-coder:14b")
        assert resolved.wrapup_threshold == 0.80
        assert resolved.session_timer == timedelta(minutes=15)
        assert resolved.wrapup_prompt == base.wrapup_prompt

        default = base.resolve_for_model("llama3.1:8b")
        assert default.wrapup_threshold == 0.96


# ------------------------------------------------------------------
# SessionController
# ------------------------------------------------------------------


class TestSessionController:
    def test_check_limits_token_breach(self) -> None:
        policy = SessionPolicy(wrapup_threshold=0.96)
        ctrl = SessionController(policy=policy)

        breached, reason = ctrl.check_limits("sess-1", 96.0, 10.0, 5)
        assert breached is True
        assert "96.0%" in reason

        breached, reason = ctrl.check_limits("sess-1", 95.9, 10.0, 5)
        assert breached is False
        assert reason == ""

    def test_check_limits_timer_breach(self) -> None:
        policy = SessionPolicy(session_timer=timedelta(minutes=30))
        ctrl = SessionController(policy=policy)

        breached, reason = ctrl.check_limits("sess-1", 50.0, 1801.0, 5)
        assert breached is True
        assert "timer expired" in reason

        breached, reason = ctrl.check_limits("sess-1", 50.0, 1799.0, 5)
        assert breached is False

    def test_check_limits_disabled(self) -> None:
        policy = SessionPolicy(wrapup_threshold=1.0, session_timer=None)
        ctrl = SessionController(policy=policy)

        breached, reason = ctrl.check_limits("sess-1", 99.9, 9999.0, 5)
        assert breached is False

    def test_should_finalize_alias(self) -> None:
        ctrl = SessionController(policy=SessionPolicy(wrapup_threshold=0.90))
        assert ctrl.should_finalize("s", 91.0, 0.0, 0)[0] is True
        assert ctrl.should_finalize("s", 89.0, 0.0, 0)[0] is False

    def test_log_event_and_telemetry(self) -> None:
        ctrl = SessionController()
        ctrl.log_event("sess-a", SessionLifecycleEvent.RUNNING, 10.0, 0.0, 0)
        ctrl.log_event("sess-a", SessionLifecycleEvent.WRAPPING_UP, 96.0, 120.0, 5)
        ctrl.log_event("sess-b", SessionLifecycleEvent.RESTARTING, 0.0, 0.0, 0)

        assert len(ctrl.get_telemetry()) == 3
        assert len(ctrl.get_telemetry("sess-a")) == 2
        assert len(ctrl.get_telemetry("sess-b")) == 1

    def test_summary_stats(self) -> None:
        ctrl = SessionController()
        assert ctrl.get_summary_stats() == {}

        ctrl.log_event("s1", SessionLifecycleEvent.WRAPPING_UP, 96.0, 120.0, 5)
        ctrl.log_event("s2", SessionLifecycleEvent.WRAPPING_UP, 94.0, 1800.0, 10)
        ctrl.log_event("s2", SessionLifecycleEvent.RESTARTING, 0.0, 0.0, 0)

        stats = ctrl.get_summary_stats()
        assert stats["total_sessions"] == 2
        assert stats["total_wrapups"] == 2
        assert stats["total_restarts"] == 1
        assert stats["avg_utilization_at_wrapup"] == 95.0
        assert stats["avg_elapsed_seconds"] == 960.0


# ------------------------------------------------------------------
# HeadContextManager.graceful_finalize_summary
# ------------------------------------------------------------------


class TestHeadContextManagerFinalizeSummary:
    def test_finalize_summary_empty(self) -> None:
        ctx = HeadContextManager(model="qwen2.5:14b")
        assert ctx.graceful_finalize_summary() == ""

    def test_finalize_summary_with_messages(self) -> None:
        ctx = HeadContextManager(model="qwen2.5:14b")
        ctx.add_message({"role": "system", "content": "You are helpful."})
        ctx.add_message({"role": "user", "content": "Hello there"})
        ctx.add_message({"role": "assistant", "content": "Hi! How can I help?"})

        summary = ctx.graceful_finalize_summary()
        assert "## Recent Exchanges" in summary
        assert "Hello there" in summary
        assert "Hi! How can I help?" in summary

    def test_finalize_summary_preserves_existing_summary(self) -> None:
        ctx = HeadContextManager(model="qwen2.5:14b")
        ctx.set_summary("Earlier we discussed architecture.")
        ctx.add_message({"role": "user", "content": "What next?"})

        summary = ctx.graceful_finalize_summary()
        assert "## Previous Session Summary" in summary
        assert "Earlier we discussed architecture." in summary


# ------------------------------------------------------------------
# HeadREPL session lifecycle
# ------------------------------------------------------------------


class TestHeadREPLSessionLifecycle:
    def test_repl_session_fields_default(self) -> None:
        """When no session policy args are passed, controller should be None."""
        with patch.object(HeadREPL, "_generate_session_id", return_value="test-id"):
            with patch("animus_kernel.head.repl.OllamaProvider") as mock_provider:
                mock_provider.return_value.is_configured.return_value = True
                repl = HeadREPL(model="qwen2.5:14b")
                assert repl._session_controller is None
                assert repl._session_timer is None
                assert repl._wrapup_threshold == 1.0

    def test_repl_session_fields_with_policy(self) -> None:
        with patch.object(HeadREPL, "_generate_session_id", return_value="test-id"):
            with patch("animus_kernel.head.repl.OllamaProvider") as mock_provider:
                mock_provider.return_value.is_configured.return_value = True
                repl = HeadREPL(
                    model="qwen2.5:14b",
                    session_timer=timedelta(minutes=30),
                    wrapup_threshold=0.96,
                )
                assert repl._session_controller is not None
                assert repl._session_controller.policy.wrapup_threshold == 0.96
                assert repl._session_controller.policy.session_timer == timedelta(minutes=30)

    def test_check_session_limits_no_controller(self) -> None:
        with patch.object(HeadREPL, "_generate_session_id", return_value="test-id"):
            with patch("animus_kernel.head.repl.OllamaProvider") as mock_provider:
                mock_provider.return_value.is_configured.return_value = True
                repl = HeadREPL(model="qwen2.5:14b")
                assert repl._check_session_limits() is False

    def test_check_session_limits_breach(self) -> None:
        with patch.object(HeadREPL, "_generate_session_id", return_value="test-id"):
            with patch("animus_kernel.head.repl.OllamaProvider") as mock_provider:
                mock_provider.return_value.is_configured.return_value = True
                repl = HeadREPL(
                    model="qwen2.5:14b",
                    wrapup_threshold=0.50,
                )
                repl._session_started_at = repl._session_started_at or MagicMock()
                # Mock context stats to report 60% utilization
                repl.context.get_stats = MagicMock(return_value=MagicMock(utilization_percent=60.0))

                with patch.object(repl, "_graceful_finalize") as mock_finalize:
                    assert repl._check_session_limits() is True
                    mock_finalize.assert_called_once()

    def test_graceful_finalize_sets_summary(self) -> None:
        with patch.object(HeadREPL, "_generate_session_id", return_value="test-id"):
            with patch("animus_kernel.head.repl.OllamaProvider") as mock_provider:
                mock_provider.return_value.is_configured.return_value = True
                repl = HeadREPL(
                    model="qwen2.5:14b",
                    wrapup_threshold=0.96,
                )
                repl._session_started_at = repl._session_started_at or MagicMock()
                repl.context.add_message({"role": "system", "content": "test"})

                # Build a mock response with a real string content attribute
                mock_response = MagicMock()
                mock_response.content = "Summary of work done."
                with patch.object(repl, "_call_model", return_value=mock_response):
                    with patch.object(repl, "_checkpoint"):
                        with patch.object(repl, "_restart_session"):
                            repl._graceful_finalize()

                assert "Summary of work done." in repl.context._summary
                assert repl._session_wrapped_up is True

    def test_restart_session_generates_new_id(self) -> None:
        with patch.object(HeadREPL, "_generate_session_id", side_effect=["old-id", "new-id"]):
            with patch("animus_kernel.head.repl.OllamaProvider") as mock_provider:
                mock_provider.return_value.is_configured.return_value = True
                repl = HeadREPL(model="qwen2.5:14b")
                repl._session_started_at = repl._session_started_at or MagicMock()
                old_id = repl.session_id
                repl.context.set_summary("Previous summary.")

                with patch.object(repl, "bootstrap"):
                    repl._restart_session()

                assert repl.session_id != old_id
                assert repl.turns == 0
                assert repl._session_wrapped_up is False
                assert repl.context._summary == "Previous summary."

    def test_restart_session_clears_messages(self) -> None:
        with patch.object(HeadREPL, "_generate_session_id", return_value="test-id"):
            with patch("animus_kernel.head.repl.OllamaProvider") as mock_provider:
                mock_provider.return_value.is_configured.return_value = True
                repl = HeadREPL(model="qwen2.5:14b")
                repl._session_started_at = repl._session_started_at or MagicMock()
                repl.context.add_message({"role": "user", "content": "hello"})
                repl.context.set_summary("Keep me.")

                with patch.object(repl, "bootstrap"):
                    repl._restart_session()

                assert len(repl.context._messages) == 0
                assert repl.context._summary == "Keep me."


# ------------------------------------------------------------------
# HeadCheckpointStore integration with sessions
# ------------------------------------------------------------------


class TestCheckpointStoreSessionIntegration:
    def test_checkpoint_survives_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_head.db"
            store = HeadCheckpointStore(db_path=db_path)

            cp = HeadCheckpoint(
                session_id="sess-restart",
                started_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
                last_active_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
                messages=[{"role": "user", "content": "test"}],
                summary="test summary",
                total_tokens=42,
                turns=3,
            )
            store.save(cp)

            loaded = store.load("sess-restart")
            assert loaded is not None
            assert loaded.summary == "test summary"
            assert loaded.turns == 3

    def test_list_recent_ordering(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_head.db"
            store = HeadCheckpointStore(db_path=db_path)

            for i in range(3):
                cp = HeadCheckpoint(
                    session_id=f"sess-{i}",
                    started_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
                    last_active_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
                    summary=f"summary {i}",
                )
                store.save(cp)

            recent = store.list_recent(limit=5)
            assert len(recent) == 3
