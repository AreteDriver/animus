"""Tests for the proactive engine module."""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from animus_bootstrap.intelligence.proactive import (
    ProactiveCheck,
    ProactiveEngine,
    ScheduleParser,
    ScheduleResult,
)
from animus_bootstrap.intelligence.proactive.checks import get_builtin_checks
from animus_bootstrap.intelligence.proactive.checks.calendar import (
    calendar_reminder_checker,
    get_calendar_check,
)
from animus_bootstrap.intelligence.proactive.checks.morning_brief import (
    get_morning_brief_check,
    morning_brief_checker,
)
from animus_bootstrap.intelligence.proactive.checks.tasks import (
    get_task_nudge_check,
    task_nudge_checker,
)
from animus_bootstrap.intelligence.proactive.engine import NudgeRecord

# ---------------------------------------------------------------------------
# TestScheduleParser
# ---------------------------------------------------------------------------


class TestScheduleParser:
    """Tests for ScheduleParser."""

    def test_parse_interval_minutes(self) -> None:
        assert ScheduleParser.parse_interval("every 30m") == 1800.0

    def test_parse_interval_hours(self) -> None:
        assert ScheduleParser.parse_interval("every 2h") == 7200.0

    def test_parse_interval_seconds(self) -> None:
        assert ScheduleParser.parse_interval("every 60s") == 60.0

    def test_parse_interval_days(self) -> None:
        assert ScheduleParser.parse_interval("every 1d") == 86400.0

    def test_parse_interval_with_whitespace(self) -> None:
        assert ScheduleParser.parse_interval("  every  5m  ") == 300.0

    def test_parse_interval_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid interval spec"):
            ScheduleParser.parse_interval("bad input")

    def test_parse_interval_missing_unit_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid interval spec"):
            ScheduleParser.parse_interval("every 30")

    def test_is_interval_true(self) -> None:
        assert ScheduleParser.is_interval("every 30m") is True

    def test_is_interval_true_with_whitespace(self) -> None:
        assert ScheduleParser.is_interval("  every 5h  ") is True

    def test_is_interval_false_for_cron(self) -> None:
        assert ScheduleParser.is_interval("0 7 * * *") is False

    def test_parse_cron_simple(self) -> None:
        result = ScheduleParser.parse_cron("0 7 * * *")
        assert result["minute"] == [0]
        assert result["hour"] == [7]
        assert result["day"] == list(range(1, 32))
        assert result["month"] == list(range(1, 13))
        assert result["weekday"] == list(range(0, 7))

    def test_parse_cron_step(self) -> None:
        result = ScheduleParser.parse_cron("*/15 * * * *")
        assert result["minute"] == [0, 15, 30, 45]

    def test_parse_cron_range(self) -> None:
        result = ScheduleParser.parse_cron("0 9 * * 1-5")
        assert result["weekday"] == [1, 2, 3, 4, 5]

    def test_parse_cron_list(self) -> None:
        result = ScheduleParser.parse_cron("0 9 * * 1,3,5")
        assert result["weekday"] == [1, 3, 5]

    def test_parse_cron_hour_step(self) -> None:
        result = ScheduleParser.parse_cron("0 */2 * * *")
        assert result["hour"] == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]

    def test_parse_cron_range_with_step(self) -> None:
        result = ScheduleParser.parse_cron("0-30/10 * * * *")
        assert result["minute"] == [0, 10, 20, 30]

    def test_parse_cron_wrong_field_count_raises(self) -> None:
        with pytest.raises(ValueError, match="must have 5 fields"):
            ScheduleParser.parse_cron("0 7 * *")

    def test_next_cron_fire_finds_correct_time(self) -> None:
        # "0 7 * * *" means at 7:00 every day
        after = datetime(2026, 2, 20, 6, 30, 0, tzinfo=UTC)
        result = ScheduleParser.next_cron_fire("0 7 * * *", after=after)
        assert result.hour == 7
        assert result.minute == 0
        assert result.day == 20

    def test_next_cron_fire_rolls_to_next_day(self) -> None:
        after = datetime(2026, 2, 20, 8, 0, 0, tzinfo=UTC)
        result = ScheduleParser.next_cron_fire("0 7 * * *", after=after)
        assert result.day == 21
        assert result.hour == 7

    def test_next_cron_fire_with_weekday_filter(self) -> None:
        # Python weekday(): Mon=0, Tue=1, Wed=2, Thu=3, Fri=4, Sat=5, Sun=6
        # Feb 20, 2026 is a Friday (weekday=4)
        after = datetime(2026, 2, 20, 0, 0, 0, tzinfo=UTC)
        # Weekday 5 = Saturday in Python, which is the next day (within 48h)
        result = ScheduleParser.next_cron_fire("0 9 * * 5", after=after)
        assert result.weekday() == 5  # Saturday
        assert result.day == 21
        assert result.hour == 9

    def test_next_cron_fire_no_match_raises(self) -> None:
        # Day 31 in February will never match
        after = datetime(2026, 2, 1, 0, 0, 0, tzinfo=UTC)
        with pytest.raises(ValueError, match="No matching cron time"):
            ScheduleParser.next_cron_fire("0 0 31 2 *", after=after)

    def test_next_cron_fire_defaults_to_now(self) -> None:
        result = ScheduleParser.next_cron_fire("*/1 * * * *")
        assert result > datetime.now(UTC)


# ---------------------------------------------------------------------------
# TestScheduleResult
# ---------------------------------------------------------------------------


class TestScheduleResult:
    """Tests for ScheduleResult dataclass."""

    def test_fields(self) -> None:
        now = datetime.now(UTC)
        sr = ScheduleResult(next_fire=now, interval_seconds=60.0)
        assert sr.next_fire == now
        assert sr.interval_seconds == 60.0

    def test_interval_defaults_to_none(self) -> None:
        now = datetime.now(UTC)
        sr = ScheduleResult(next_fire=now)
        assert sr.interval_seconds is None


# ---------------------------------------------------------------------------
# TestProactiveCheck
# ---------------------------------------------------------------------------


class TestProactiveCheck:
    """Tests for ProactiveCheck dataclass."""

    def test_fields_and_defaults(self) -> None:
        checker = AsyncMock(return_value=None)
        pc = ProactiveCheck(name="test", schedule="every 5m", checker=checker)
        assert pc.name == "test"
        assert pc.schedule == "every 5m"
        assert pc.channels == []
        assert pc.priority == "normal"
        assert pc.enabled is True

    def test_enabled_default_is_true(self) -> None:
        pc = ProactiveCheck(name="x", schedule="every 1m", checker=AsyncMock(return_value=None))
        assert pc.enabled is True

    def test_priority_default_is_normal(self) -> None:
        pc = ProactiveCheck(name="x", schedule="every 1m", checker=AsyncMock(return_value=None))
        assert pc.priority == "normal"

    def test_custom_fields(self) -> None:
        checker = AsyncMock(return_value="nudge")
        pc = ProactiveCheck(
            name="custom",
            schedule="0 8 * * *",
            checker=checker,
            channels=["telegram", "webchat"],
            priority="high",
            enabled=False,
        )
        assert pc.channels == ["telegram", "webchat"]
        assert pc.priority == "high"
        assert pc.enabled is False


# ---------------------------------------------------------------------------
# TestNudgeRecord
# ---------------------------------------------------------------------------


class TestNudgeRecord:
    """Tests for NudgeRecord dataclass."""

    def test_fields(self) -> None:
        now = datetime.now(UTC)
        nr = NudgeRecord(
            id="abc",
            check_name="test",
            text="hello",
            channels=["webchat"],
            priority="normal",
            timestamp=now,
        )
        assert nr.id == "abc"
        assert nr.check_name == "test"
        assert nr.text == "hello"
        assert nr.channels == ["webchat"]
        assert nr.priority == "normal"
        assert nr.timestamp == now

    def test_delivered_default_is_false(self) -> None:
        nr = NudgeRecord(
            id="x",
            check_name="t",
            text="t",
            channels=[],
            priority="low",
            timestamp=datetime.now(UTC),
        )
        assert nr.delivered is False


# ---------------------------------------------------------------------------
# TestProactiveEngine
# ---------------------------------------------------------------------------


class TestProactiveEngine:
    """Tests for ProactiveEngine."""

    def _make_engine(self, tmp_path, **kwargs):
        db = tmp_path / "proactive.db"
        return ProactiveEngine(db_path=db, **kwargs)

    def _make_check(self, name="test_check", text="nudge text", **kwargs):
        checker = AsyncMock(return_value=text)
        return ProactiveCheck(
            name=name,
            schedule="every 5m",
            checker=checker,
            channels=["webchat"],
            **kwargs,
        )

    def test_register_and_list(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        check = self._make_check()
        engine.register_check(check)
        checks = engine.list_checks()
        assert len(checks) == 1
        assert checks[0].name == "test_check"
        engine.close()

    def test_register_overwrites_existing(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        check1 = self._make_check(text="first")
        check2 = self._make_check(text="second")
        engine.register_check(check1)
        engine.register_check(check2)
        checks = engine.list_checks()
        assert len(checks) == 1
        engine.close()

    def test_unregister_removes_check(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        check = self._make_check()
        engine.register_check(check)
        engine.unregister_check("test_check")
        assert engine.list_checks() == []
        engine.close()

    def test_unregister_nonexistent_is_noop(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        engine.unregister_check("nonexistent")  # Should not raise
        engine.close()

    def test_is_quiet_hours_during_quiet(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path, quiet_hours=("22:00", "07:00"))
        # 23:00 is within 22:00-07:00
        late = datetime(2026, 2, 20, 23, 0, 0, tzinfo=UTC)
        assert engine.is_quiet_hours(late) is True
        engine.close()

    def test_is_quiet_hours_outside_quiet(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path, quiet_hours=("22:00", "07:00"))
        # 12:00 is outside 22:00-07:00
        noon = datetime(2026, 2, 20, 12, 0, 0, tzinfo=UTC)
        assert engine.is_quiet_hours(noon) is False
        engine.close()

    def test_is_quiet_hours_overnight_span_early_morning(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path, quiet_hours=("22:00", "07:00"))
        # 03:00 is within overnight span
        early = datetime(2026, 2, 20, 3, 0, 0, tzinfo=UTC)
        assert engine.is_quiet_hours(early) is True
        engine.close()

    def test_is_quiet_hours_same_day_span(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path, quiet_hours=("09:00", "17:00"))
        inside = datetime(2026, 2, 20, 12, 0, 0, tzinfo=UTC)
        outside = datetime(2026, 2, 20, 20, 0, 0, tzinfo=UTC)
        assert engine.is_quiet_hours(inside) is True
        assert engine.is_quiet_hours(outside) is False
        engine.close()

    @pytest.mark.asyncio()
    async def test_run_check_returns_nudge_when_text(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        check = self._make_check()
        engine.register_check(check)
        nudge = await engine.run_check("test_check")
        assert nudge is not None
        assert nudge.text == "nudge text"
        assert nudge.check_name == "test_check"
        assert nudge.channels == ["webchat"]
        assert nudge.priority == "normal"
        engine.close()

    @pytest.mark.asyncio()
    async def test_run_check_returns_none_when_checker_returns_none(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        check = self._make_check(text=None)
        engine.register_check(check)
        nudge = await engine.run_check("test_check")
        assert nudge is None
        engine.close()

    @pytest.mark.asyncio()
    async def test_run_check_logs_to_db(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        check = self._make_check()
        engine.register_check(check)
        await engine.run_check("test_check")
        history = engine.get_nudge_history()
        assert len(history) == 1
        assert history[0].text == "nudge text"
        engine.close()

    @pytest.mark.asyncio()
    async def test_run_check_calls_send_callback(self, tmp_path) -> None:
        callback = AsyncMock()
        engine = self._make_engine(tmp_path, send_callback=callback)
        check = self._make_check()
        engine.register_check(check)
        # Ensure we're outside quiet hours
        with patch.object(engine, "is_quiet_hours", return_value=False):
            nudge = await engine.run_check("test_check")
        assert nudge is not None
        assert nudge.delivered is True
        callback.assert_awaited_once_with("nudge text", ["webchat"])
        engine.close()

    @pytest.mark.asyncio()
    async def test_run_check_skips_send_during_quiet_hours(self, tmp_path) -> None:
        callback = AsyncMock()
        engine = self._make_engine(tmp_path, send_callback=callback)
        check = self._make_check()
        engine.register_check(check)
        with patch.object(engine, "is_quiet_hours", return_value=True):
            nudge = await engine.run_check("test_check")
        assert nudge is not None
        assert nudge.delivered is False
        callback.assert_not_awaited()
        engine.close()

    @pytest.mark.asyncio()
    async def test_run_check_unknown_raises_key_error(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        with pytest.raises(KeyError, match="Check not found"):
            await engine.run_check("nonexistent")
        engine.close()

    @pytest.mark.asyncio()
    async def test_run_check_handles_checker_exception(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        checker = AsyncMock(side_effect=RuntimeError("boom"))
        check = ProactiveCheck(
            name="failing",
            schedule="every 5m",
            checker=checker,
            channels=["webchat"],
        )
        engine.register_check(check)
        nudge = await engine.run_check("failing")
        assert nudge is None
        engine.close()

    @pytest.mark.asyncio()
    async def test_run_check_handles_callback_exception(self, tmp_path) -> None:
        callback = AsyncMock(side_effect=RuntimeError("delivery failed"))
        engine = self._make_engine(tmp_path, send_callback=callback)
        check = self._make_check()
        engine.register_check(check)
        with patch.object(engine, "is_quiet_hours", return_value=False):
            nudge = await engine.run_check("test_check")
        assert nudge is not None
        assert nudge.delivered is False  # Delivery failed, so not marked delivered
        engine.close()

    def test_get_nudge_history_returns_empty(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        assert engine.get_nudge_history() == []
        engine.close()

    @pytest.mark.asyncio()
    async def test_get_nudge_history_returns_logged_nudges(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        check = self._make_check()
        engine.register_check(check)
        await engine.run_check("test_check")
        await engine.run_check("test_check")
        history = engine.get_nudge_history()
        assert len(history) == 2
        engine.close()

    @pytest.mark.asyncio()
    async def test_get_nudge_history_respects_limit(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        check = self._make_check()
        engine.register_check(check)
        for _ in range(5):
            await engine.run_check("test_check")
        history = engine.get_nudge_history(limit=3)
        assert len(history) == 3
        engine.close()

    @pytest.mark.asyncio()
    async def test_clear_history_removes_all(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        check = self._make_check()
        engine.register_check(check)
        await engine.run_check("test_check")
        count = engine.clear_history()
        assert count == 1
        assert engine.get_nudge_history() == []
        engine.close()

    @pytest.mark.asyncio()
    async def test_clear_history_with_before_datetime(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        check = self._make_check()
        engine.register_check(check)
        await engine.run_check("test_check")
        # Clear records before far future — should remove all
        future = datetime.now(UTC) + timedelta(days=1)
        count = engine.clear_history(before=future)
        assert count == 1
        assert engine.get_nudge_history() == []
        engine.close()

    @pytest.mark.asyncio()
    async def test_clear_history_with_before_past(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        check = self._make_check()
        engine.register_check(check)
        await engine.run_check("test_check")
        # Clear records before now (shouldn't remove anything since nudge was just created)
        past = datetime.now(UTC) - timedelta(days=1)
        count = engine.clear_history(before=past)
        assert count == 0
        assert len(engine.get_nudge_history()) == 1
        engine.close()

    @pytest.mark.asyncio()
    async def test_start_and_stop_lifecycle(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        assert engine.running is False
        await engine.start()
        assert engine.running is True
        assert engine._task is not None
        await engine.stop()
        assert engine.running is False
        assert engine._task is None
        engine.close()

    def test_running_property(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        assert engine.running is False
        engine.close()

    def test_close_closes_db(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        engine.close()
        # After close, DB operations should fail
        with pytest.raises(sqlite3.ProgrammingError):
            engine._conn.execute("SELECT 1")

    @pytest.mark.asyncio()
    async def test_scheduler_loop_runs_due_check(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        check = self._make_check()
        engine.register_check(check)
        # Set next_fire to the past so it fires immediately
        engine._next_fires["test_check"] = datetime.now(UTC) - timedelta(seconds=10)
        await engine.start()
        # Give the loop a moment to execute
        await asyncio.sleep(0.1)
        await engine.stop()
        # Check was called at least once
        check.checker.assert_awaited()
        engine.close()

    @pytest.mark.asyncio()
    async def test_scheduler_skips_disabled_check(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        check = self._make_check(enabled=False)
        engine.register_check(check)
        engine._next_fires["test_check"] = datetime.now(UTC) - timedelta(seconds=10)
        await engine.start()
        await asyncio.sleep(0.1)
        await engine.stop()
        check.checker.assert_not_awaited()
        engine.close()

    @pytest.mark.asyncio()
    async def test_stop_without_start_is_safe(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        await engine.stop()  # Should not raise
        engine.close()

    def test_is_quiet_hours_defaults_to_now(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path, quiet_hours=("00:00", "00:01"))
        # Should not raise, just returns a bool
        result = engine.is_quiet_hours()
        assert isinstance(result, bool)
        engine.close()


# ---------------------------------------------------------------------------
# TestBuiltinChecks
# ---------------------------------------------------------------------------


class TestBuiltinChecks:
    """Tests for built-in proactive checks."""

    @pytest.mark.asyncio()
    async def test_morning_brief_checker_returns_greeting(self) -> None:
        result = await morning_brief_checker()
        assert result is not None
        assert "Good morning" in result
        now = datetime.now(UTC)
        assert now.strftime("%A") in result

    @pytest.mark.asyncio()
    async def test_task_nudge_checker_returns_none(self) -> None:
        result = await task_nudge_checker()
        assert result is None

    @pytest.mark.asyncio()
    async def test_calendar_reminder_checker_returns_none(self) -> None:
        result = await calendar_reminder_checker()
        assert result is None

    def test_get_morning_brief_check_returns_proactive_check(self) -> None:
        check = get_morning_brief_check()
        assert isinstance(check, ProactiveCheck)
        assert check.name == "morning_brief"
        assert check.schedule == "0 7 * * *"
        assert check.channels == ["webchat"]
        assert check.enabled is True

    def test_get_task_nudge_check_returns_proactive_check(self) -> None:
        check = get_task_nudge_check()
        assert isinstance(check, ProactiveCheck)
        assert check.name == "task_nudge"
        assert check.schedule == "0 */2 * * *"
        assert check.priority == "low"

    def test_get_calendar_check_returns_disabled(self) -> None:
        check = get_calendar_check()
        assert isinstance(check, ProactiveCheck)
        assert check.name == "calendar_reminder"
        assert check.schedule == "every 15m"
        assert check.enabled is False

    def test_get_builtin_checks_returns_list(self) -> None:
        checks = get_builtin_checks()
        assert isinstance(checks, list)
        assert len(checks) == 4
        names = {c.name for c in checks}
        assert names == {"morning_brief", "task_nudge", "calendar_reminder", "reflection"}

    def test_get_builtin_checks_all_are_proactive_checks(self) -> None:
        checks = get_builtin_checks()
        for check in checks:
            assert isinstance(check, ProactiveCheck)
