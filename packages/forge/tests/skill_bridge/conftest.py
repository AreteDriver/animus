"""Fixtures for SkillBridge tests.

Builds a temporary ai-skills-shaped repo so tests don't depend on
~/projects/ai-skills/ existing on the runner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml

from animus_forge.skill_bridge import (
    SkillExecutor,
    SkillResolver,
)

# ---------------------------------------------------------------------------
# Mock provider stack
# ---------------------------------------------------------------------------


@dataclass
class MockCompletionResponse:
    content: str
    model: str = "mock-1"
    provider: str = "mock"
    tokens_used: int = 250
    input_tokens: int = 50
    output_tokens: int = 200
    finish_reason: str = "stop"
    latency_ms: float = 10.0


@dataclass
class MockProvider:
    name: str = "mock"
    response_text: str = "MOCK_OUTPUT"
    raise_on_complete: Exception | None = None
    calls: list = field(default_factory=list)

    def complete(self, request):  # noqa: D401 — match Provider.complete signature
        self.calls.append(request)
        if self.raise_on_complete is not None:
            raise self.raise_on_complete
        return MockCompletionResponse(content=self.response_text)


@dataclass
class MockProviderManager:
    provider: MockProvider = field(default_factory=MockProvider)
    last_hint: str | None = None

    def get_provider(self, hint: str | None = None):
        self.last_hint = hint
        return self.provider


# ---------------------------------------------------------------------------
# Mock BudgetManager — records every allocate/record_usage call
# ---------------------------------------------------------------------------


@dataclass
class MockBudget:
    allow: bool = True
    allocate_calls: list[tuple[int, str]] = field(default_factory=list)
    usage_calls: list[dict] = field(default_factory=list)

    def allocate(self, tokens, agent_id=None):
        self.allocate_calls.append((tokens, agent_id))
        return self.allow

    def record_usage(self, agent_id, **kwargs):
        record = {"agent_id": agent_id, **kwargs}
        self.usage_calls.append(record)
        return record


# ---------------------------------------------------------------------------
# Mock ApprovalStore — programmable status sequence
# ---------------------------------------------------------------------------


@dataclass
class MockApprovalStore:
    """Returns status entries from `sequence` round-robin.

    Use sequence=["pending", "approved"] to simulate one poll then
    decision; sequence=["pending", "pending", "rejected"] etc.
    """

    sequence: list[str] = field(default_factory=lambda: ["approved"])
    tokens_created: list[dict] = field(default_factory=list)
    _idx: int = 0
    _last_token: str | None = None

    def create_token(self, **kwargs):
        token = f"tok-{len(self.tokens_created):04d}"
        self.tokens_created.append({**kwargs, "token": token})
        self._idx = 0
        self._last_token = token
        return token

    def get_by_token(self, token: str):
        if token != self._last_token:
            return None
        if self._idx >= len(self.sequence):
            return {"status": self.sequence[-1]}
        status = self.sequence[self._idx]
        self._idx += 1
        return {"status": status}


# ---------------------------------------------------------------------------
# Fake clock — controllable for approval timeout tests
# ---------------------------------------------------------------------------


class FakeClock:
    def __init__(self, start: float = 0.0, step: float = 0.05) -> None:
        self._now = start
        self._step = step
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self._now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self._now += max(self._step, seconds)


# ---------------------------------------------------------------------------
# Fake ai-skills repo on tmp_path
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    """Build a minimal ai-skills layout under tmp_path."""
    (tmp_path / "personas" / "engineering" / "code-reviewer").mkdir(parents=True)
    (tmp_path / "personas" / "engineering" / "code-reviewer" / "SKILL.md").write_text(
        "---\nname: code-reviewer\ndescription: review code\n---\n# code-reviewer\n"
    )

    # Persona registered with destructive risk override (OQ3)
    (tmp_path / "personas" / "domain" / "demolisher").mkdir(parents=True)
    (tmp_path / "personas" / "domain" / "demolisher" / "SKILL.md").write_text(
        "---\nname: demolisher\ndescription: irreversible\n---\n# demolisher\n"
    )

    # Persona NOT in registry — filesystem fallback target
    (tmp_path / "personas" / "engineering" / "ghost-persona").mkdir(parents=True)
    (tmp_path / "personas" / "engineering" / "ghost-persona" / "SKILL.md").write_text(
        "---\nname: ghost-persona\ndescription: fs-fallback\n---\n# ghost\n"
    )

    # Multi-perspective persona (OQ1) — only test that it's ONE call
    (tmp_path / "personas" / "domain" / "board-of-advisors").mkdir(parents=True)
    (tmp_path / "personas" / "domain" / "board-of-advisors" / "SKILL.md").write_text(
        "---\nname: board-of-advisors\ndescription: multi-voice\n---\n# board\n"
    )

    # Agent with typed inputs + risk_level safe
    agent_dir = tmp_path / "agents" / "integrations" / "api-client"
    agent_dir.mkdir(parents=True)
    (agent_dir / "SKILL.md").write_text(
        "---\nname: api-client\ndescription: api\n---\n# api-client\n"
    )
    (agent_dir / "schema.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "api-client",
                "risk_level": "safe",
                "inputs": {
                    "operation": {
                        "type": "string",
                        "required": True,
                        "enum": ["request", "paginate", "batch"],
                    },
                    "url": {"type": "string", "required": True},
                    "timeout_seconds": {"type": "integer", "required": False},
                },
                "outputs": {"response": {"type": "object"}},
            }
        )
    )

    # Agent with output schema (OQ4)
    out_agent = tmp_path / "agents" / "analysis" / "scorer"
    out_agent.mkdir(parents=True)
    (out_agent / "SKILL.md").write_text("---\nname: scorer\ndescription: scoring\n---\n# scorer\n")
    (out_agent / "schema.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "scorer",
                "risk_level": "safe",
                "inputs": {},
                "outputs": {"score": {"type": "number"}},
            }
        )
    )
    (out_agent / "output.schema.yaml").write_text(
        yaml.safe_dump({"type": "object", "required": ["score"]})
    )

    # Destructive agent
    destruct = tmp_path / "agents" / "system" / "demolish"
    destruct.mkdir(parents=True)
    (destruct / "SKILL.md").write_text(
        "---\nname: demolish\ndescription: dangerous\n---\n# demolish\n"
    )
    (destruct / "schema.yaml").write_text(
        yaml.safe_dump({"name": "demolish", "risk_level": "destructive", "inputs": {}})
    )

    # Workflow tier — flat layout
    wf = tmp_path / "workflows" / "context-mapping"
    wf.mkdir(parents=True)
    (wf / "SKILL.md").write_text(
        "---\nname: context-mapping\ndescription: map context\n---\n# wf\n"
    )

    # Out-of-repo persona linked via symlink (R3 test)
    outside_dir = tmp_path.parent / "outside-repo"
    outside_dir.mkdir(exist_ok=True)
    outside_skill = outside_dir / "evil"
    outside_skill.mkdir(exist_ok=True)
    (outside_skill / "SKILL.md").write_text("---\nname: evil\ndescription: evil\n---\n")
    (tmp_path / "personas" / "engineering" / "evil-link").symlink_to(outside_skill)

    # registry.yaml — covers all the registered (non-fallback) entries
    (tmp_path / "registry.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0.0",
                "repository": "test/ai-skills",
                "personas": {
                    "engineering": [
                        {
                            "name": "code-reviewer",
                            "path": "personas/engineering/code-reviewer",
                            "description": "review code",
                        }
                    ],
                    "domain": [
                        {
                            "name": "demolisher",
                            "path": "personas/domain/demolisher",
                            "description": "irreversible persona",
                            "risk_level": "destructive",
                        },
                        {
                            "name": "board-of-advisors",
                            "path": "personas/domain/board-of-advisors",
                            "description": "multi-voice",
                        },
                    ],
                },
                "agents": {
                    "integrations": [
                        {
                            "name": "api-client",
                            "path": "agents/integrations/api-client",
                            "description": "api",
                        }
                    ],
                    "analysis": [
                        {
                            "name": "scorer",
                            "path": "agents/analysis/scorer",
                            "description": "scoring",
                        }
                    ],
                    "system": [
                        {
                            "name": "demolish",
                            "path": "agents/system/demolish",
                            "description": "dangerous",
                        }
                    ],
                },
                "workflows": [
                    {
                        "name": "context-mapping",
                        "path": "workflows/context-mapping",
                        "description": "map context",
                    }
                ],
            }
        )
    )
    return tmp_path


@pytest.fixture
def resolver(fake_repo: Path) -> SkillResolver:
    return SkillResolver(repo_path=fake_repo)


@pytest.fixture
def mock_provider():
    return MockProvider()


@pytest.fixture
def mock_provider_manager(mock_provider):
    return MockProviderManager(provider=mock_provider)


@pytest.fixture
def mock_budget():
    return MockBudget()


@pytest.fixture
def mock_approval_approve():
    return MockApprovalStore(sequence=["approved"])


@pytest.fixture
def mock_approval_reject():
    return MockApprovalStore(sequence=["rejected"])


@pytest.fixture
def fake_clock():
    return FakeClock()


@pytest.fixture
def executor(mock_provider_manager, mock_budget, mock_approval_approve, fake_clock):
    return SkillExecutor(
        provider_manager=mock_provider_manager,
        budget=mock_budget,
        approval_store=mock_approval_approve,
        clock=fake_clock,
    )
