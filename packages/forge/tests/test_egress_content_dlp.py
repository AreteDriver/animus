"""Content-aware egress DLP at the provider boundary (roadmap A4)."""

from __future__ import annotations

import pytest
from animus_types import Sensitivity

from animus_forge.network import EgressDeniedError
from animus_forge.providers.base import CompletionRequest, assert_egress_allowed

ENDPOINT = "https://api.anthropic.com"
SECRET = "ship it with sk-ant-abc1234567890ABCDEFGHIJ today"


class TestScannableText:
    def test_includes_prompt_system_and_messages(self):
        req = CompletionRequest(
            prompt="p",
            system_prompt="s",
            messages=[
                {"role": "user", "content": "m1"},
                {"role": "user", "content": [{"type": "text", "text": "m2"}]},
            ],
        )
        text = req.scannable_text()
        assert "p" in text and "s" in text and "m1" in text and "m2" in text


class TestAssertEgressAllowed:
    def test_public_request_with_secret_payload_blocked(self):
        req = CompletionRequest(prompt=SECRET, sensitivity=Sensitivity.PUBLIC)
        with pytest.raises(EgressDeniedError, match="credential"):
            assert_egress_allowed(ENDPOINT, req)

    def test_secret_in_message_history_blocked(self):
        req = CompletionRequest(
            prompt="please review",
            sensitivity=Sensitivity.PUBLIC,
            messages=[{"role": "user", "content": SECRET}],
        )
        with pytest.raises(EgressDeniedError, match="credential"):
            assert_egress_allowed(ENDPOINT, req)

    def test_clean_public_request_allowed(self):
        req = CompletionRequest(
            prompt="summarize the budget report", sensitivity=Sensitivity.PUBLIC
        )
        assert_egress_allowed(ENDPOINT, req)  # no raise

    def test_confidential_tier_still_blocked_with_clean_payload(self):
        req = CompletionRequest(prompt="internal notes", sensitivity=Sensitivity.CONFIDENTIAL)
        with pytest.raises(EgressDeniedError, match="sensitivity"):
            assert_egress_allowed(ENDPOINT, req)
