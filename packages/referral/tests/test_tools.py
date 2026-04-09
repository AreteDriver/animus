"""Tests for referral MCP tools."""

from __future__ import annotations

import pytest

from animus_referral.tools.referral_tools import (
    referral_link,
    referral_programs,
    referral_status,
    reset_tools_state,
)


@pytest.fixture(autouse=True)
def clean_state(tmp_data_dir, monkeypatch):
    """Reset tool singletons and point to temp dir."""
    reset_tools_state()
    monkeypatch.setenv("REFERRAL_DATA_DIR", str(tmp_data_dir))
    yield
    reset_tools_state()


class TestReferralProgramsTool:
    def test_list_programs(self):
        result = referral_programs({"action": "list"})
        assert result["success"] is True
        assert "programs" in result
        assert result["total"] > 0

    def test_list_programs_default_action(self):
        result = referral_programs({})
        assert result["success"] is True
        assert "programs" in result

    def test_list_programs_by_type(self):
        result = referral_programs({"action": "list", "program_type": "exchange"})
        assert result["success"] is True
        for p in result["programs"]:
            assert p["program_type"] == "exchange"

    def test_list_programs_include_inactive(self):
        # First add an inactive program
        referral_programs(
            {
                "action": "add",
                "name": "inactive_test",
                "platform_url": "https://x.com",
                "program_type": "other",
            }
        )
        result = referral_programs({"action": "list", "active_only": False})
        assert result["success"] is True

    def test_add_program(self):
        result = referral_programs(
            {
                "action": "add",
                "name": "new_program",
                "platform_url": "https://new.com",
                "program_type": "exchange",
                "bonus_referrer_usd": 20.0,
                "bonus_type": "flat",
                "requirements": "Must deposit",
                "referral_url_template": "https://new.com/ref/{code}",
                "chain": "eth",
            }
        )
        assert result["success"] is True
        assert result["program"]["name"] == "new_program"

    def test_add_program_missing_name(self):
        result = referral_programs(
            {
                "action": "add",
                "platform_url": "https://x.com",
            }
        )
        assert result["success"] is False
        assert "required" in result["error"]

    def test_add_program_missing_url(self):
        result = referral_programs(
            {
                "action": "add",
                "name": "test",
            }
        )
        assert result["success"] is False
        assert "required" in result["error"]

    def test_add_program_invalid_type(self):
        result = referral_programs(
            {
                "action": "add",
                "name": "test",
                "platform_url": "https://x.com",
                "program_type": "invalid_type",
            }
        )
        assert result["success"] is False
        assert "Invalid program_type" in result["error"]

    def test_add_program_invalid_bonus_type(self):
        result = referral_programs(
            {
                "action": "add",
                "name": "test",
                "platform_url": "https://x.com",
                "program_type": "other",
                "bonus_type": "invalid_bonus",
            }
        )
        assert result["success"] is False
        assert "Invalid bonus_type" in result["error"]

    def test_add_program_duplicate(self):
        referral_programs(
            {
                "action": "add",
                "name": "dup_test",
                "platform_url": "https://dup.com",
                "program_type": "other",
            }
        )
        result = referral_programs(
            {
                "action": "add",
                "name": "dup_test",
                "platform_url": "https://dup.com",
                "program_type": "other",
            }
        )
        assert result["success"] is False
        assert "already exists" in result["error"]


class TestReferralStatusTool:
    def test_status_empty(self):
        result = referral_status({})
        assert result["success"] is True
        assert "summary" in result
        assert "report" in result

    def test_status_with_days(self):
        result = referral_status({"days": 7})
        assert result["success"] is True
        assert result["report"]["period_days"] == 7


class TestReferralLinkTool:
    def test_generate_link(self):
        result = referral_link(
            {
                "program_name": "coinbase",
                "referrer_alias": "alice",
                "referral_code": "ALICE123",
            }
        )
        assert result["success"] is True
        assert "ALICE123" in result["link"]["link_url"]
        assert result["existing"] is False

    def test_get_existing_link(self):
        referral_link(
            {
                "program_name": "coinbase",
                "referrer_alias": "alice",
                "referral_code": "ALICE123",
            }
        )
        result = referral_link(
            {
                "program_name": "coinbase",
                "referrer_alias": "alice",
            }
        )
        assert result["success"] is True
        assert result["existing"] is True

    def test_missing_program_name(self):
        result = referral_link({"referrer_alias": "alice"})
        assert result["success"] is False
        assert "required" in result["error"]

    def test_missing_referrer_alias(self):
        result = referral_link({"program_name": "coinbase"})
        assert result["success"] is False
        assert "required" in result["error"]

    def test_missing_code_for_new_link(self):
        result = referral_link(
            {
                "program_name": "coinbase",
                "referrer_alias": "alice",
            }
        )
        assert result["success"] is False
        assert "referral_code" in result["error"]

    def test_unknown_program(self):
        result = referral_link(
            {
                "program_name": "nonexistent_xyz",
                "referrer_alias": "alice",
                "referral_code": "CODE",
            }
        )
        assert result["success"] is False
        assert "not found" in result["error"]
