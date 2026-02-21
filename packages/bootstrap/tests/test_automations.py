"""Tests for the automation pipeline module."""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from animus_bootstrap.gateway.models import GatewayMessage, create_message
from animus_bootstrap.intelligence.automations.actions import execute_action
from animus_bootstrap.intelligence.automations.conditions import (
    evaluate_condition,
    evaluate_conditions,
)
from animus_bootstrap.intelligence.automations.engine import AutomationEngine
from animus_bootstrap.intelligence.automations.models import (
    ActionConfig,
    AutomationResult,
    AutomationRule,
    Condition,
    TriggerConfig,
)
from animus_bootstrap.intelligence.automations.triggers import evaluate_trigger

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _msg(
    text: str = "hello world",
    channel: str = "telegram",
    sender_id: str = "user1",
    sender_name: str = "Alice",
) -> GatewayMessage:
    return create_message(
        channel=channel,
        sender_id=sender_id,
        sender_name=sender_name,
        text=text,
    )


# ===========================================================================
# TestModels
# ===========================================================================


class TestModels:
    """Test dataclass defaults and field generation."""

    def test_trigger_config_defaults(self) -> None:
        tc = TriggerConfig(type="message")
        assert tc.type == "message"
        assert tc.params == {}

    def test_condition_defaults(self) -> None:
        c = Condition(type="contains")
        assert c.type == "contains"
        assert c.params == {}

    def test_action_config_defaults(self) -> None:
        ac = ActionConfig(type="reply")
        assert ac.type == "reply"
        assert ac.params == {}

    def test_automation_rule_defaults_and_generated_id(self) -> None:
        r = AutomationRule()
        assert r.id  # non-empty UUID string
        assert r.name == ""
        assert r.enabled is True
        assert r.trigger.type == "message"
        assert r.conditions == []
        assert r.actions == []
        assert r.cooldown_seconds == 0
        assert isinstance(r.created_at, datetime)
        assert r.last_fired is None

    def test_automation_rule_unique_ids(self) -> None:
        r1 = AutomationRule()
        r2 = AutomationRule()
        assert r1.id != r2.id

    def test_automation_result_fields(self) -> None:
        res = AutomationResult(
            rule_id="r1",
            rule_name="test",
            triggered=True,
            actions_executed=["Replied on telegram"],
        )
        assert res.rule_id == "r1"
        assert res.rule_name == "test"
        assert res.triggered is True
        assert res.actions_executed == ["Replied on telegram"]
        assert isinstance(res.timestamp, datetime)
        assert res.error is None

    def test_automation_result_with_error(self) -> None:
        res = AutomationResult(
            rule_id="r1",
            rule_name="test",
            triggered=False,
            actions_executed=[],
            error="kaboom",
        )
        assert res.error == "kaboom"


# ===========================================================================
# TestTriggers
# ===========================================================================


class TestTriggers:
    """Test trigger evaluation."""

    def test_message_trigger_matches_keywords(self) -> None:
        trigger = TriggerConfig(type="message", params={"keywords": ["hello", "hi"]})
        assert evaluate_trigger(trigger, message=_msg("Hello there!")) is True

    def test_message_trigger_keywords_case_insensitive(self) -> None:
        trigger = TriggerConfig(type="message", params={"keywords": ["HELLO"]})
        assert evaluate_trigger(trigger, message=_msg("hello")) is True

    def test_message_trigger_keywords_no_match(self) -> None:
        trigger = TriggerConfig(type="message", params={"keywords": ["goodbye"]})
        assert evaluate_trigger(trigger, message=_msg("hello")) is False

    def test_message_trigger_matches_regex(self) -> None:
        trigger = TriggerConfig(type="message", params={"regex": r"\d{3}-\d{4}"})
        assert evaluate_trigger(trigger, message=_msg("Call 555-1234")) is True

    def test_message_trigger_regex_no_match(self) -> None:
        trigger = TriggerConfig(type="message", params={"regex": r"^\d+$"})
        assert evaluate_trigger(trigger, message=_msg("no digits here")) is False

    def test_message_trigger_matches_sender_id(self) -> None:
        trigger = TriggerConfig(type="message", params={"sender_id": "user1"})
        assert evaluate_trigger(trigger, message=_msg(sender_id="user1")) is True

    def test_message_trigger_sender_id_no_match(self) -> None:
        trigger = TriggerConfig(type="message", params={"sender_id": "user99"})
        assert evaluate_trigger(trigger, message=_msg(sender_id="user1")) is False

    def test_message_trigger_matches_channel(self) -> None:
        trigger = TriggerConfig(type="message", params={"channel": "telegram"})
        assert evaluate_trigger(trigger, message=_msg(channel="telegram")) is True

    def test_message_trigger_channel_no_match(self) -> None:
        trigger = TriggerConfig(type="message", params={"channel": "discord"})
        assert evaluate_trigger(trigger, message=_msg(channel="telegram")) is False

    def test_message_trigger_no_params_matches_any(self) -> None:
        trigger = TriggerConfig(type="message")
        assert evaluate_trigger(trigger, message=_msg()) is True

    def test_message_trigger_no_message_returns_false(self) -> None:
        trigger = TriggerConfig(type="message")
        assert evaluate_trigger(trigger, message=None) is False

    def test_event_trigger_matches_event_type(self) -> None:
        trigger = TriggerConfig(type="event", params={"event_type": "user_joined"})
        assert evaluate_trigger(trigger, event={"type": "user_joined"}) is True

    def test_event_trigger_no_match(self) -> None:
        trigger = TriggerConfig(type="event", params={"event_type": "user_joined"})
        assert evaluate_trigger(trigger, event={"type": "user_left"}) is False

    def test_event_trigger_no_event_returns_false(self) -> None:
        trigger = TriggerConfig(type="event", params={"event_type": "x"})
        assert evaluate_trigger(trigger, event=None) is False

    def test_event_trigger_no_event_type_param_matches_any(self) -> None:
        trigger = TriggerConfig(type="event")
        assert evaluate_trigger(trigger, event={"type": "anything"}) is True

    def test_webhook_trigger_always_matches(self) -> None:
        trigger = TriggerConfig(type="webhook")
        assert evaluate_trigger(trigger, event={"data": "payload"}) is True

    def test_webhook_trigger_no_event_returns_false(self) -> None:
        trigger = TriggerConfig(type="webhook")
        assert evaluate_trigger(trigger, event=None) is False

    def test_unknown_trigger_type_returns_false(self) -> None:
        trigger = TriggerConfig(type="alien_signal")
        assert evaluate_trigger(trigger, message=_msg()) is False

    def test_schedule_trigger_returns_false(self) -> None:
        trigger = TriggerConfig(type="schedule")
        assert evaluate_trigger(trigger) is False


# ===========================================================================
# TestConditions
# ===========================================================================


class TestConditions:
    """Test condition evaluation."""

    def test_empty_conditions_returns_true(self) -> None:
        assert evaluate_conditions([]) is True

    def test_contains_matches_case_insensitive(self) -> None:
        cond = Condition(type="contains", params={"text": "HELLO"})
        assert evaluate_condition(cond, message=_msg("Hello World")) is True

    def test_contains_no_match(self) -> None:
        cond = Condition(type="contains", params={"text": "goodbye"})
        assert evaluate_condition(cond, message=_msg("Hello")) is False

    def test_contains_no_message(self) -> None:
        cond = Condition(type="contains", params={"text": "hello"})
        assert evaluate_condition(cond, message=None) is False

    def test_from_channel_matches(self) -> None:
        cond = Condition(type="from_channel", params={"channel": "telegram"})
        assert evaluate_condition(cond, message=_msg(channel="telegram")) is True

    def test_from_channel_no_match(self) -> None:
        cond = Condition(type="from_channel", params={"channel": "discord"})
        assert evaluate_condition(cond, message=_msg(channel="telegram")) is False

    def test_from_channel_no_message(self) -> None:
        cond = Condition(type="from_channel", params={"channel": "x"})
        assert evaluate_condition(cond, message=None) is False

    def test_regex_matches(self) -> None:
        cond = Condition(type="regex", params={"pattern": r"^\d+$"})
        assert evaluate_condition(cond, message=_msg("12345")) is True

    def test_regex_no_match(self) -> None:
        cond = Condition(type="regex", params={"pattern": r"^\d+$"})
        assert evaluate_condition(cond, message=_msg("abc")) is False

    def test_regex_no_message(self) -> None:
        cond = Condition(type="regex", params={"pattern": r"."})
        assert evaluate_condition(cond, message=None) is False

    def test_sender_is_matches(self) -> None:
        cond = Condition(type="sender_is", params={"sender_id": "user1"})
        assert evaluate_condition(cond, message=_msg(sender_id="user1")) is True

    def test_sender_is_no_match(self) -> None:
        cond = Condition(type="sender_is", params={"sender_id": "user99"})
        assert evaluate_condition(cond, message=_msg(sender_id="user1")) is False

    def test_sender_is_no_message(self) -> None:
        cond = Condition(type="sender_is", params={"sender_id": "x"})
        assert evaluate_condition(cond, message=None) is False

    def test_time_range_within(self) -> None:
        now = datetime.now(UTC)
        # Build a range that straddles current time by +/- 1 hour
        start = (now - timedelta(hours=1)).strftime("%H:%M")
        end = (now + timedelta(hours=1)).strftime("%H:%M")
        cond = Condition(type="time_range", params={"start": start, "end": end})
        assert evaluate_condition(cond) is True

    def test_time_range_outside(self) -> None:
        now = datetime.now(UTC)
        # Range fully in the past (2 hours ago to 1 hour ago)
        start = (now - timedelta(hours=3)).strftime("%H:%M")
        end = (now - timedelta(hours=2)).strftime("%H:%M")
        cond = Condition(type="time_range", params={"start": start, "end": end})
        assert evaluate_condition(cond) is False

    def test_multiple_conditions_and_logic(self) -> None:
        conds = [
            Condition(type="contains", params={"text": "hello"}),
            Condition(type="from_channel", params={"channel": "telegram"}),
        ]
        msg = _msg("hello world", channel="telegram")
        assert evaluate_conditions(conds, message=msg) is True

    def test_multiple_conditions_one_fails(self) -> None:
        conds = [
            Condition(type="contains", params={"text": "hello"}),
            Condition(type="from_channel", params={"channel": "discord"}),
        ]
        msg = _msg("hello world", channel="telegram")
        assert evaluate_conditions(conds, message=msg) is False

    def test_unknown_condition_type_returns_false(self) -> None:
        cond = Condition(type="phase_of_moon")
        assert evaluate_condition(cond, message=_msg()) is False


# ===========================================================================
# TestActions
# ===========================================================================


class TestActions:
    """Test action execution."""

    @pytest.mark.asyncio()
    async def test_reply_action_calls_callback(self) -> None:
        callback = AsyncMock()
        action = ActionConfig(type="reply", params={"text": "pong"})
        ctx = {"channel": "telegram", "send_callback": callback}
        result = await execute_action(action, ctx)
        callback.assert_awaited_once_with("telegram", "pong")
        assert "Replied on telegram" in result

    @pytest.mark.asyncio()
    async def test_reply_action_no_callback(self) -> None:
        action = ActionConfig(type="reply", params={"text": "pong"})
        result = await execute_action(action, {"channel": "telegram"})
        assert "Replied on telegram" in result

    @pytest.mark.asyncio()
    async def test_forward_action_calls_callback(self) -> None:
        callback = AsyncMock()
        action = ActionConfig(type="forward", params={"target_channel": "discord"})
        ctx = {"message_text": "forwarded msg", "send_callback": callback}
        result = await execute_action(action, ctx)
        callback.assert_awaited_once_with("discord", "forwarded msg")
        assert "Forwarded to discord" in result

    @pytest.mark.asyncio()
    async def test_forward_action_no_callback(self) -> None:
        action = ActionConfig(type="forward", params={"target_channel": "discord"})
        result = await execute_action(action, {"message_text": "hi"})
        assert "Forwarded to discord" in result

    @pytest.mark.asyncio()
    async def test_webhook_action_posts_to_url(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch(
            "animus_bootstrap.intelligence.automations.actions.httpx.AsyncClient",
            return_value=mock_client,
        ):
            action = ActionConfig(
                type="webhook",
                params={"url": "https://example.com/hook", "payload": {"k": "v"}},
            )
            result = await execute_action(action, {})
        assert "200" in result
        mock_client.post.assert_awaited_once()

    @pytest.mark.asyncio()
    async def test_webhook_action_failure(self) -> None:
        import httpx as httpx_mod

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=httpx_mod.ConnectError("unreachable"))

        with patch(
            "animus_bootstrap.intelligence.automations.actions.httpx.AsyncClient",
            return_value=mock_client,
        ):
            action = ActionConfig(type="webhook", params={"url": "https://bad.url/hook"})
            result = await execute_action(action, {})
        assert "failed" in result.lower()

    @pytest.mark.asyncio()
    async def test_store_memory_with_manager(self) -> None:
        mm = AsyncMock()
        action = ActionConfig(
            type="store_memory",
            params={"content": "important fact", "memory_type": "episodic"},
        )
        ctx = {"memory_manager": mm}
        result = await execute_action(action, ctx)
        mm.store_fact.assert_awaited_once()
        assert "Stored memory" in result

    @pytest.mark.asyncio()
    async def test_store_memory_without_manager(self) -> None:
        action = ActionConfig(type="store_memory", params={"content": "fact"})
        result = await execute_action(action, {})
        assert "No memory_manager" in result

    @pytest.mark.asyncio()
    async def test_run_tool_with_executor(self) -> None:
        executor = AsyncMock(return_value="42")
        action = ActionConfig(
            type="run_tool",
            params={"tool_name": "calculator", "arguments": {"expr": "6*7"}},
        )
        ctx = {"tool_executor": executor}
        result = await execute_action(action, ctx)
        executor.assert_awaited_once_with("calculator", {"expr": "6*7"})
        assert "42" in result

    @pytest.mark.asyncio()
    async def test_run_tool_without_executor(self) -> None:
        action = ActionConfig(type="run_tool", params={"tool_name": "calculator"})
        result = await execute_action(action, {})
        assert "No tool_executor" in result

    @pytest.mark.asyncio()
    async def test_unknown_action_type(self) -> None:
        action = ActionConfig(type="teleport")
        result = await execute_action(action, {})
        assert "Unknown action type" in result


# ===========================================================================
# TestAutomationEngine
# ===========================================================================


class TestAutomationEngine:
    """Test the full automation engine with SQLite persistence."""

    def _make_engine(self, tmp_path) -> AutomationEngine:
        return AutomationEngine(tmp_path / "auto.db")

    def _make_rule(self, **overrides) -> AutomationRule:
        defaults = {
            "name": "test_rule",
            "trigger": TriggerConfig(type="message"),
            "actions": [ActionConfig(type="reply", params={"text": "pong"})],
        }
        defaults.update(overrides)
        return AutomationRule(**defaults)

    def test_add_and_list_rules(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        rule = self._make_rule()
        engine.add_rule(rule)
        rules = engine.list_rules()
        assert len(rules) == 1
        assert rules[0].name == "test_rule"
        engine.close()

    def test_add_rule_persists_to_db(self, tmp_path) -> None:
        db = tmp_path / "auto.db"
        rule = self._make_rule(name="persisted")
        engine = AutomationEngine(db)
        engine.add_rule(rule)
        engine.close()

        # Re-open and verify
        engine2 = AutomationEngine(db)
        rules = engine2.list_rules()
        assert len(rules) == 1
        assert rules[0].name == "persisted"
        engine2.close()

    def test_remove_rule(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        rule = self._make_rule()
        engine.add_rule(rule)
        assert engine.remove_rule(rule.id) is True
        assert engine.list_rules() == []
        engine.close()

    def test_remove_nonexistent_rule(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        assert engine.remove_rule("nonexistent") is False
        engine.close()

    def test_get_rule_by_id(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        rule = self._make_rule()
        engine.add_rule(rule)
        got = engine.get_rule(rule.id)
        assert got is not None
        assert got.name == "test_rule"
        engine.close()

    def test_get_rule_not_found(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        assert engine.get_rule("nope") is None
        engine.close()

    def test_enable_rule(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        rule = self._make_rule(enabled=False)
        engine.add_rule(rule)
        assert engine.enable_rule(rule.id) is True
        assert engine.get_rule(rule.id).enabled is True
        engine.close()

    def test_enable_nonexistent_rule(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        assert engine.enable_rule("nope") is False
        engine.close()

    def test_disable_rule(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        rule = self._make_rule()
        engine.add_rule(rule)
        assert engine.disable_rule(rule.id) is True
        assert engine.get_rule(rule.id).enabled is False
        engine.close()

    def test_disable_nonexistent_rule(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        assert engine.disable_rule("nope") is False
        engine.close()

    @pytest.mark.asyncio()
    async def test_evaluate_message_fires_matching_rule(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        callback = AsyncMock()
        rule = self._make_rule(
            trigger=TriggerConfig(type="message", params={"keywords": ["hello"]}),
        )
        engine.add_rule(rule)

        results = await engine.evaluate_message(
            _msg("hello world"),
            context={"channel": "telegram", "send_callback": callback},
        )
        assert len(results) == 1
        assert results[0].triggered is True
        assert results[0].rule_name == "test_rule"
        engine.close()

    @pytest.mark.asyncio()
    async def test_evaluate_message_skips_disabled_rules(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        rule = self._make_rule(enabled=False)
        engine.add_rule(rule)

        results = await engine.evaluate_message(_msg("hello"))
        assert results == []
        engine.close()

    @pytest.mark.asyncio()
    async def test_evaluate_message_skips_non_message_triggers(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        rule = self._make_rule(trigger=TriggerConfig(type="event"))
        engine.add_rule(rule)

        results = await engine.evaluate_message(_msg("hello"))
        assert results == []
        engine.close()

    @pytest.mark.asyncio()
    async def test_evaluate_message_respects_cooldown(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        rule = self._make_rule(cooldown_seconds=3600)
        engine.add_rule(rule)

        # First fire should succeed
        results = await engine.evaluate_message(_msg("hello"))
        assert len(results) == 1

        # Second fire within cooldown should be skipped
        results = await engine.evaluate_message(_msg("hello again"))
        assert results == []
        engine.close()

    @pytest.mark.asyncio()
    async def test_evaluate_message_conditions_block(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        rule = self._make_rule(
            conditions=[Condition(type="from_channel", params={"channel": "discord"})],
        )
        engine.add_rule(rule)

        results = await engine.evaluate_message(_msg("hello", channel="telegram"))
        assert results == []
        engine.close()

    @pytest.mark.asyncio()
    async def test_evaluate_message_multiple_rules_fire(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        r1 = self._make_rule(name="rule_a")
        r2 = self._make_rule(name="rule_b")
        engine.add_rule(r1)
        engine.add_rule(r2)

        results = await engine.evaluate_message(_msg("hello"))
        assert len(results) == 2
        names = {r.rule_name for r in results}
        assert names == {"rule_a", "rule_b"}
        engine.close()

    @pytest.mark.asyncio()
    async def test_evaluate_event_fires_matching_rule(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        rule = self._make_rule(
            trigger=TriggerConfig(type="event", params={"event_type": "user_joined"}),
        )
        engine.add_rule(rule)

        results = await engine.evaluate_event({"type": "user_joined"})
        assert len(results) == 1
        assert results[0].triggered is True
        engine.close()

    @pytest.mark.asyncio()
    async def test_evaluate_event_skips_message_triggers(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        rule = self._make_rule(trigger=TriggerConfig(type="message"))
        engine.add_rule(rule)

        results = await engine.evaluate_event({"type": "user_joined"})
        assert results == []
        engine.close()

    @pytest.mark.asyncio()
    async def test_evaluate_event_webhook_trigger(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        rule = self._make_rule(
            trigger=TriggerConfig(type="webhook"),
        )
        engine.add_rule(rule)

        results = await engine.evaluate_event({"data": "payload"})
        assert len(results) == 1
        engine.close()

    def test_get_history_returns_results(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        rule = self._make_rule()
        engine.add_rule(rule)

        asyncio.get_event_loop().run_until_complete(engine.evaluate_message(_msg("hello")))

        history = engine.get_history()
        assert len(history) == 1
        assert history[0].triggered is True
        engine.close()

    def test_get_history_filtered_by_rule_id(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        r1 = self._make_rule(name="rule_a")
        r2 = self._make_rule(name="rule_b")
        engine.add_rule(r1)
        engine.add_rule(r2)

        asyncio.get_event_loop().run_until_complete(engine.evaluate_message(_msg("hello")))

        history_a = engine.get_history(rule_id=r1.id)
        assert len(history_a) == 1
        assert history_a[0].rule_name == "rule_a"
        engine.close()

    def test_clear_history(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        rule = self._make_rule()
        engine.add_rule(rule)

        asyncio.get_event_loop().run_until_complete(engine.evaluate_message(_msg("hello")))
        assert len(engine.get_history()) == 1

        engine.clear_history()
        assert len(engine.get_history()) == 0
        engine.close()

    def test_close(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        engine.close()
        # Verify connection is closed (subsequent operations would fail)
        with pytest.raises(sqlite3.ProgrammingError):
            engine._conn.execute("SELECT 1")

    @pytest.mark.asyncio()
    async def test_action_error_recorded_in_result(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        # Create a rule with an action that will error
        rule = self._make_rule(
            actions=[ActionConfig(type="run_tool", params={"tool_name": "bad"})],
        )
        engine.add_rule(rule)

        # Provide a tool_executor that raises
        executor = AsyncMock(side_effect=RuntimeError("tool exploded"))
        results = await engine.evaluate_message(
            _msg("hello"),
            context={"tool_executor": executor},
        )
        assert len(results) == 1
        assert results[0].error is not None
        assert "tool exploded" in results[0].error
        engine.close()

    @pytest.mark.asyncio()
    async def test_rule_update_via_add(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        rule = self._make_rule(name="original")
        engine.add_rule(rule)
        assert engine.get_rule(rule.id).name == "original"

        # Update the same rule
        rule.name = "updated"
        engine.add_rule(rule)
        assert len(engine.list_rules()) == 1
        assert engine.get_rule(rule.id).name == "updated"
        engine.close()

    @pytest.mark.asyncio()
    async def test_evaluate_message_with_conditions_pass(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        rule = self._make_rule(
            conditions=[
                Condition(type="contains", params={"text": "hello"}),
                Condition(type="from_channel", params={"channel": "telegram"}),
            ],
        )
        engine.add_rule(rule)

        results = await engine.evaluate_message(_msg("hello world", channel="telegram"))
        assert len(results) == 1
        engine.close()

    @pytest.mark.asyncio()
    async def test_last_fired_updated_after_fire(self, tmp_path) -> None:
        engine = self._make_engine(tmp_path)
        rule = self._make_rule()
        engine.add_rule(rule)
        assert rule.last_fired is None

        await engine.evaluate_message(_msg("hello"))
        assert rule.last_fired is not None
        engine.close()


# ===========================================================================
# TestInitExports
# ===========================================================================


class TestInitExports:
    """Verify __init__.py re-exports."""

    def test_imports(self) -> None:
        from animus_bootstrap.intelligence.automations import (
            ActionConfig,
            AutomationEngine,
            AutomationResult,
            AutomationRule,
            Condition,
            TriggerConfig,
        )

        assert AutomationEngine is not None
        assert AutomationRule is not None
        assert TriggerConfig is not None
        assert Condition is not None
        assert ActionConfig is not None
        assert AutomationResult is not None
