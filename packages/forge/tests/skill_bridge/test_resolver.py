"""Resolver tests — A2, A3, A8, A9, A11, A15, A16."""

from __future__ import annotations

import logging
import time

import pytest

from animus_forge.skill_bridge import (
    SkillResolutionError,
    SkillResolver,
)

# ---------------------------------------------------------------------------
# A2 — agent missing required input rejected (also tests resolver basic OK)
# ---------------------------------------------------------------------------


def test_resolver_returns_resolved_skill_for_registered_persona(resolver):
    skill = resolver.resolve("code-reviewer", "persona")
    assert skill.name == "code-reviewer"
    assert skill.tier == "persona"
    assert skill.risk_level == "safe"
    assert skill.from_registry is True
    assert skill.skill_md_path.exists()


def test_resolver_loads_schema_for_agent(resolver):
    skill = resolver.resolve("api-client", "agent")
    assert skill.tier == "agent"
    assert skill.schema is not None
    assert "operation" in skill.schema.inputs


# ---------------------------------------------------------------------------
# A9 — out-of-repo symlink rejected
# ---------------------------------------------------------------------------


def test_resolver_rejects_symlink_outside_repo(resolver):
    with pytest.raises(SkillResolutionError, match="out-of-repo"):
        resolver.resolve("evil-link", "persona")


# ---------------------------------------------------------------------------
# A11 — filesystem fallback emits WARN
# ---------------------------------------------------------------------------


def test_filesystem_fallback_logs_warn(resolver, caplog):
    with caplog.at_level(logging.WARNING):
        skill = resolver.resolve("ghost-persona", "persona")
    assert skill.from_registry is False
    warnings = [r for r in caplog.records if "not in registry" in r.getMessage()]
    assert len(warnings) == 1
    assert "ghost-persona" in warnings[0].getMessage()


# ---------------------------------------------------------------------------
# A15 — registry risk_level override applied
# ---------------------------------------------------------------------------


def test_persona_registry_risk_override(resolver):
    skill = resolver.resolve("demolisher", "persona")
    assert skill.risk_level == "destructive"


# ---------------------------------------------------------------------------
# A16 — fs fallback defaults to safe
# ---------------------------------------------------------------------------


def test_filesystem_fallback_defaults_safe(resolver):
    skill = resolver.resolve("ghost-persona", "persona")
    assert skill.risk_level == "safe"


# ---------------------------------------------------------------------------
# Unknown skill → SkillResolutionError
# ---------------------------------------------------------------------------


def test_unknown_skill_raises(resolver):
    with pytest.raises(SkillResolutionError, match="skill not found"):
        resolver.resolve("nonexistent-thing", "persona")


# ---------------------------------------------------------------------------
# A8 — resolution latency (registry hit + filesystem fallback)
# ---------------------------------------------------------------------------


def test_registry_hit_latency_under_5ms(resolver):
    # Warm cache
    resolver.resolve("code-reviewer", "persona")
    elapsed = []
    for _ in range(50):
        t0 = time.perf_counter()
        resolver.resolve("code-reviewer", "persona")
        elapsed.append((time.perf_counter() - t0) * 1000)
    elapsed.sort()
    p95 = elapsed[int(len(elapsed) * 0.95) - 1]
    assert p95 < 5.0, f"registry hit p95={p95:.2f}ms > 5ms target"


def test_filesystem_fallback_latency_under_50ms(resolver):
    elapsed = []
    for _ in range(20):
        t0 = time.perf_counter()
        try:
            resolver.resolve("ghost-persona", "persona")
        except SkillResolutionError:
            pass
        elapsed.append((time.perf_counter() - t0) * 1000)
    elapsed.sort()
    p95 = elapsed[int(len(elapsed) * 0.95) - 1]
    assert p95 < 50.0, f"fs fallback p95={p95:.2f}ms > 50ms target"


# ---------------------------------------------------------------------------
# Resolver builds without an explicit repo_path (env-driven path)
# ---------------------------------------------------------------------------


def test_resolver_env_var_override(fake_repo, monkeypatch):
    monkeypatch.setenv("AI_SKILLS_REPO", str(fake_repo))
    r = SkillResolver()
    skill = r.resolve("code-reviewer", "persona")
    assert skill.skill_md_path.exists()
