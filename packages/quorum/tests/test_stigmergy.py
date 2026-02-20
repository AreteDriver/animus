"""Tests for convergent.stigmergy trail markers."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest
from convergent.stigmergy import StigmergyField


class TestLeaveMarker:
    def test_leave_and_retrieve(self) -> None:
        field = StigmergyField(":memory:")
        marker = field.leave_marker(
            "agent-1", "file_modified", "src/auth.py", "Added login endpoint"
        )
        assert marker.agent_id == "agent-1"
        assert marker.marker_type == "file_modified"
        assert marker.target == "src/auth.py"
        assert marker.content == "Added login endpoint"
        assert marker.strength == 1.0
        assert marker.marker_id  # Non-empty UUID

    def test_unique_marker_ids(self) -> None:
        field = StigmergyField(":memory:")
        m1 = field.leave_marker("a", "t", "target", "c1")
        m2 = field.leave_marker("a", "t", "target", "c2")
        assert m1.marker_id != m2.marker_id

    def test_custom_strength(self) -> None:
        field = StigmergyField(":memory:")
        marker = field.leave_marker("a", "t", "target", "c", strength=0.5)
        assert marker.strength == 0.5

    def test_with_expires_at(self) -> None:
        field = StigmergyField(":memory:")
        exp = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        marker = field.leave_marker("a", "t", "target", "c", expires_at=exp)
        assert marker.expires_at == exp


class TestGetMarkers:
    def test_get_by_target(self) -> None:
        field = StigmergyField(":memory:")
        field.leave_marker("a1", "file_modified", "src/auth.py", "Changed login")
        field.leave_marker("a2", "known_issue", "src/auth.py", "Race condition")
        field.leave_marker("a1", "file_modified", "src/db.py", "Added index")
        markers = field.get_markers("src/auth.py")
        assert len(markers) == 2
        assert all(m.target == "src/auth.py" for m in markers)

    def test_get_by_target_empty(self) -> None:
        field = StigmergyField(":memory:")
        assert field.get_markers("nonexistent") == []

    def test_get_by_type(self) -> None:
        field = StigmergyField(":memory:")
        field.leave_marker("a1", "file_modified", "src/a.py", "c1")
        field.leave_marker("a1", "known_issue", "src/b.py", "c2")
        field.leave_marker("a2", "file_modified", "src/c.py", "c3")
        markers = field.get_markers_by_type("file_modified")
        assert len(markers) == 2
        assert all(m.marker_type == "file_modified" for m in markers)

    def test_get_by_type_empty(self) -> None:
        field = StigmergyField(":memory:")
        assert field.get_markers_by_type("nonexistent") == []

    def test_get_by_agent(self) -> None:
        field = StigmergyField(":memory:")
        field.leave_marker("agent-1", "file_modified", "a.py", "c1")
        field.leave_marker("agent-2", "file_modified", "b.py", "c2")
        field.leave_marker("agent-1", "known_issue", "c.py", "c3")
        markers = field.get_markers_by_agent("agent-1")
        assert len(markers) == 2
        assert all(m.agent_id == "agent-1" for m in markers)


class TestReinforce:
    def test_reinforce_increases_strength(self) -> None:
        field = StigmergyField(":memory:")
        marker = field.leave_marker("a", "t", "target", "c", strength=0.5)
        new = field.reinforce(marker.marker_id, amount=0.3)
        assert new == pytest.approx(0.8)

    def test_reinforce_caps_at_two(self) -> None:
        field = StigmergyField(":memory:")
        marker = field.leave_marker("a", "t", "target", "c", strength=1.8)
        new = field.reinforce(marker.marker_id, amount=0.5)
        assert new == pytest.approx(2.0)

    def test_reinforce_nonexistent_returns_none(self) -> None:
        field = StigmergyField(":memory:")
        assert field.reinforce("nonexistent") is None

    def test_reinforce_default_amount(self) -> None:
        field = StigmergyField(":memory:")
        marker = field.leave_marker("a", "t", "target", "c", strength=0.5)
        new = field.reinforce(marker.marker_id)
        assert new == pytest.approx(1.0)  # 0.5 + 0.5 default

    def test_reinforced_strength_persists(self) -> None:
        field = StigmergyField(":memory:")
        marker = field.leave_marker("a", "t", "target", "c", strength=0.5)
        field.reinforce(marker.marker_id, amount=0.3)
        retrieved = field.get_markers("target")
        assert len(retrieved) == 1
        assert retrieved[0].strength == pytest.approx(0.8)


class TestEvaporate:
    def test_evaporation_reduces_strength(self) -> None:
        field = StigmergyField(":memory:", evaporation_rate=0.1)
        marker = field.leave_marker("a", "t", "target", "c", strength=1.0)
        # Manually backdate the marker to simulate age
        old_time = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        field._conn.execute(
            "UPDATE stigmergy_markers SET created_at = ? WHERE marker_id = ?",
            (old_time, marker.marker_id),
        )
        field._conn.commit()
        field.evaporate()
        markers = field.get_markers("target")
        assert len(markers) == 1
        expected = 1.0 * math.exp(-0.1 * 5)
        assert markers[0].strength == pytest.approx(expected, rel=0.01)

    def test_evaporation_removes_weak_markers(self) -> None:
        field = StigmergyField(":memory:", evaporation_rate=0.5, min_strength=0.05)
        marker = field.leave_marker("a", "t", "target", "c", strength=0.1)
        # Backdate to 30 days — 0.1 * e^(-0.5*30) ≈ 0.0 (way below threshold)
        old_time = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        field._conn.execute(
            "UPDATE stigmergy_markers SET created_at = ? WHERE marker_id = ?",
            (old_time, marker.marker_id),
        )
        field._conn.commit()
        removed = field.evaporate()
        assert removed == 1
        assert field.get_markers("target") == []

    def test_evaporation_preserves_strong_markers(self) -> None:
        field = StigmergyField(":memory:", evaporation_rate=0.01)
        field.leave_marker("a", "t", "target", "c", strength=1.0)
        # Fresh marker with slow decay — should survive
        removed = field.evaporate()
        assert removed == 0
        assert len(field.get_markers("target")) == 1

    def test_evaporation_returns_zero_when_empty(self) -> None:
        field = StigmergyField(":memory:")
        assert field.evaporate() == 0


class TestGetContextForAgent:
    def test_context_includes_relevant_markers(self) -> None:
        field = StigmergyField(":memory:")
        field.leave_marker("agent-1", "file_modified", "src/auth.py", "Added login")
        field.leave_marker("agent-2", "known_issue", "src/auth.py", "Race condition")
        field.leave_marker("agent-1", "file_modified", "src/db.py", "Added index")
        ctx = field.get_context_for_agent(["src/auth.py"])
        assert "auth.py" in ctx
        assert "Added login" in ctx
        assert "Race condition" in ctx
        assert "db.py" not in ctx  # Different target

    def test_context_multiple_files(self) -> None:
        field = StigmergyField(":memory:")
        field.leave_marker("a1", "file_modified", "a.py", "Changed A")
        field.leave_marker("a2", "file_modified", "b.py", "Changed B")
        field.leave_marker("a1", "file_modified", "c.py", "Changed C")
        ctx = field.get_context_for_agent(["a.py", "b.py"])
        assert "Changed A" in ctx
        assert "Changed B" in ctx
        assert "Changed C" not in ctx

    def test_context_empty_when_no_markers(self) -> None:
        field = StigmergyField(":memory:")
        assert field.get_context_for_agent(["src/auth.py"]) == ""

    def test_context_empty_for_empty_paths(self) -> None:
        field = StigmergyField(":memory:")
        field.leave_marker("a", "t", "target", "c")
        assert field.get_context_for_agent([]) == ""

    def test_context_includes_header(self) -> None:
        field = StigmergyField(":memory:")
        field.leave_marker("a", "file_modified", "a.py", "Content")
        ctx = field.get_context_for_agent(["a.py"])
        assert "Stigmergy Context" in ctx

    def test_context_shows_strength_and_agent(self) -> None:
        field = StigmergyField(":memory:")
        field.leave_marker("agent-1", "known_issue", "a.py", "Bug here", strength=0.75)
        ctx = field.get_context_for_agent(["a.py"])
        assert "0.75" in ctx
        assert "agent-1" in ctx
        assert "known_issue" in ctx


class TestRemoveMarker:
    def test_remove_existing(self) -> None:
        field = StigmergyField(":memory:")
        marker = field.leave_marker("a", "t", "target", "c")
        assert field.remove_marker(marker.marker_id) is True
        assert field.get_markers("target") == []

    def test_remove_nonexistent(self) -> None:
        field = StigmergyField(":memory:")
        assert field.remove_marker("nonexistent") is False


class TestCount:
    def test_count_empty(self) -> None:
        field = StigmergyField(":memory:")
        assert field.count() == 0

    def test_count_after_leaving_markers(self) -> None:
        field = StigmergyField(":memory:")
        field.leave_marker("a", "t", "t1", "c1")
        field.leave_marker("a", "t", "t2", "c2")
        assert field.count() == 2

    def test_count_after_removal(self) -> None:
        field = StigmergyField(":memory:")
        m = field.leave_marker("a", "t", "target", "c")
        field.remove_marker(m.marker_id)
        assert field.count() == 0


class TestPersistence:
    def test_file_persistence(self, tmp_path: object) -> None:
        import pathlib

        db_path = str(pathlib.Path(str(tmp_path)) / "stigmergy.db")
        f1 = StigmergyField(db_path)
        f1.leave_marker("agent-1", "file_modified", "a.py", "Changed A")
        f1.close()

        f2 = StigmergyField(db_path)
        markers = f2.get_markers("a.py")
        assert len(markers) == 1
        assert markers[0].content == "Changed A"
        f2.close()


class TestClose:
    def test_close_prevents_operations(self) -> None:
        field = StigmergyField(":memory:")
        field.close()
        with pytest.raises(Exception):  # noqa: B017
            field.leave_marker("a", "t", "target", "c")


class TestPublicAPI:
    def test_import_from_convergent(self) -> None:
        import convergent

        assert hasattr(convergent, "StigmergyField")

    def test_all_exports_listed(self) -> None:
        import convergent

        assert "StigmergyField" in convergent.__all__
