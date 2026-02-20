"""Tests for the coordination health dashboard."""

from __future__ import annotations

from convergent.health import (
    CoordinationHealth,
    HealthChecker,
    IntentGraphHealth,
    ScoringHealth,
    StigmergyHealth,
    VotingHealth,
    health_report,
)
from convergent.intent import Evidence, EvidenceKind, Intent, InterfaceKind, InterfaceSpec
from convergent.resolver import IntentResolver, PythonGraphBackend
from convergent.score_store import ScoreStore
from convergent.stigmergy import StigmergyField


class TestHealthCheckerEmpty:
    """Tests with no subsystems configured."""

    def test_empty_check(self) -> None:
        checker = HealthChecker()
        health = checker.check()
        assert health.grade == "A"
        assert health.issues == []
        assert health.intent_graph.total_intents == 0
        assert health.stigmergy.total_markers == 0
        assert health.scoring.total_agents == 0
        assert health.voting.total_decisions == 0

    def test_from_bridge_duck_typing(self) -> None:
        """HealthChecker.from_bridge works with any object that has the right attrs."""

        class FakeBridge:
            def __init__(self) -> None:
                self.stigmergy = StigmergyField(":memory:")
                self._store = ScoreStore(":memory:")

        bridge = FakeBridge()
        checker = HealthChecker.from_bridge(bridge)
        health = checker.check()
        assert health.grade == "A"

    def test_from_bridge_missing_attrs(self) -> None:
        """from_bridge handles objects without expected attributes."""

        class MinimalBridge:
            pass

        checker = HealthChecker.from_bridge(MinimalBridge())
        health = checker.check()
        assert health.grade == "A"


class TestIntentGraphHealth:
    """Tests for intent graph metrics."""

    def _make_intent(
        self,
        agent_id: str,
        name: str,
        provides: list[str] | None = None,
        requires: list[str] | None = None,
    ) -> Intent:
        prov = [
            InterfaceSpec(name=n, kind=InterfaceKind.FUNCTION, signature="")
            for n in (provides or [])
        ]
        req = [
            InterfaceSpec(name=n, kind=InterfaceKind.FUNCTION, signature="")
            for n in (requires or [])
        ]
        return Intent(
            agent_id=agent_id,
            intent=name,
            provides=prov,
            requires=req,
            evidence=[Evidence(kind=EvidenceKind.TEST_PASS, description="test")],
        )

    def test_intent_graph_metrics(self) -> None:
        backend = PythonGraphBackend()
        i1 = self._make_intent("a1", "build_auth", provides=["AuthService"])
        i2 = self._make_intent("a2", "build_ui", provides=["UIService"], requires=["AuthService"])
        backend.publish(i1)
        backend.publish(i2)
        resolver = IntentResolver(backend=backend)

        checker = HealthChecker(resolver=resolver)
        health = checker.check()

        assert health.intent_graph.total_intents == 2
        assert health.intent_graph.agent_count == 2
        assert health.intent_graph.provides_count == 2
        assert health.intent_graph.requires_count == 1
        assert health.intent_graph.avg_stability > 0

    def test_conflict_detection(self) -> None:
        backend = PythonGraphBackend()
        i1 = self._make_intent("a1", "build_auth", provides=["AuthService"])
        i2 = self._make_intent("a2", "build_auth2", provides=["AuthService"])
        backend.publish(i1)
        backend.publish(i2)
        resolver = IntentResolver(backend=backend)

        checker = HealthChecker(resolver=resolver)
        health = checker.check()

        assert health.intent_graph.conflict_count >= 1

    def test_low_stability_issue(self) -> None:
        backend = PythonGraphBackend()
        # Intent with no evidence has low stability
        i = Intent(
            agent_id="a1",
            intent="weak_intent",
            provides=[InterfaceSpec(name="Foo", kind=InterfaceKind.FUNCTION, signature="")],
        )
        backend.publish(i)
        resolver = IntentResolver(backend=backend)

        checker = HealthChecker(resolver=resolver)
        health = checker.check()
        # Low stability should generate an issue
        if health.intent_graph.avg_stability < 0.3:
            assert any("stability" in issue.lower() for issue in health.issues)

    def test_empty_intent_graph(self) -> None:
        backend = PythonGraphBackend()
        resolver = IntentResolver(backend=backend)
        checker = HealthChecker(resolver=resolver)
        health = checker.check()
        assert health.intent_graph.total_intents == 0


class TestStigmergyHealth:
    """Tests for stigmergy marker metrics."""

    def test_marker_metrics(self) -> None:
        stig = StigmergyField(":memory:")
        stig.leave_marker("a1", "file_modified", "src/auth.py", "changed login")
        stig.leave_marker("a1", "known_issue", "src/db.py", "race condition")
        stig.leave_marker("a2", "file_modified", "src/auth.py", "added tests")

        checker = HealthChecker(stigmergy=stig)
        health = checker.check()

        assert health.stigmergy.total_markers == 3
        assert health.stigmergy.markers_by_type["file_modified"] == 2
        assert health.stigmergy.markers_by_type["known_issue"] == 1
        assert health.stigmergy.unique_agents == 2
        assert health.stigmergy.unique_targets == 2
        assert health.stigmergy.avg_strength > 0

    def test_empty_stigmergy(self) -> None:
        stig = StigmergyField(":memory:")
        checker = HealthChecker(stigmergy=stig)
        health = checker.check()
        assert health.stigmergy.total_markers == 0


class TestScoringHealth:
    """Tests for phi scoring metrics."""

    def test_scoring_metrics(self) -> None:
        store = ScoreStore(":memory:")
        store.record_outcome("a1", "code_review", "approved")
        store.record_outcome("a2", "testing", "approved")
        store.save_score("a1", "code_review", 0.8)
        store.save_score("a2", "testing", 0.6)

        checker = HealthChecker(store=store)
        health = checker.check()

        assert health.scoring.total_agents == 2
        assert health.scoring.total_outcomes == 2
        assert health.scoring.avg_score == 0.7
        assert health.scoring.min_score == 0.6
        assert health.scoring.max_score == 0.8

    def test_low_trust_issue(self) -> None:
        store = ScoreStore(":memory:")
        store.save_score("bad_agent", "testing", 0.2)
        store.record_outcome("bad_agent", "testing", "failed")

        checker = HealthChecker(store=store)
        health = checker.check()

        assert any("trust" in issue.lower() for issue in health.issues)

    def test_empty_scoring(self) -> None:
        store = ScoreStore(":memory:")
        checker = HealthChecker(store=store)
        health = checker.check()
        assert health.scoring.total_agents == 0


class TestVotingHealth:
    """Tests for voting metrics."""

    def test_voting_metrics(self) -> None:
        store = ScoreStore(":memory:")
        # Manually insert decisions
        store._conn.execute(
            "INSERT INTO decisions "
            "(request_id, task_id, question, outcome, decided_at, decision_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("r1", "t1", "merge?", "approved", "2025-01-01T00:00:00Z", "{}"),
        )
        store._conn.execute(
            "INSERT INTO decisions "
            "(request_id, task_id, question, outcome, decided_at, decision_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("r2", "t2", "deploy?", "rejected", "2025-01-02T00:00:00Z", "{}"),
        )
        store._conn.execute(
            "INSERT INTO vote_records "
            "(request_id, agent_id, choice, confidence, weighted_score, reasoning, voted_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("r1", "a1", "approve", 0.9, 0.85, "looks good", "2025-01-01T00:00:00Z"),
        )
        store._conn.commit()

        checker = HealthChecker(store=store)
        health = checker.check()

        assert health.voting.total_decisions == 2
        assert health.voting.approval_rate == 0.5
        assert health.voting.outcomes["approved"] == 1
        assert health.voting.outcomes["rejected"] == 1
        assert health.voting.avg_confidence == 0.9

    def test_empty_voting(self) -> None:
        store = ScoreStore(":memory:")
        checker = HealthChecker(store=store)
        health = checker.check()
        assert health.voting.total_decisions == 0


class TestGrading:
    """Tests for the health grading system."""

    def test_grade_a(self) -> None:
        checker = HealthChecker()
        health = checker.check()
        assert health.grade == "A"

    def test_grade_b(self) -> None:
        """One issue = B."""
        store = ScoreStore(":memory:")
        store.save_score("bad", "x", 0.2)
        checker = HealthChecker(store=store)
        health = checker.check()
        assert health.grade == "B"

    def test_grade_c(self) -> None:
        """Two issues = C."""
        store = ScoreStore(":memory:")
        # Low trust issue
        store.save_score("bad1", "x", 0.2)
        # Also need a voting issue for second issue
        store._conn.executemany(
            "INSERT INTO decisions "
            "(request_id, task_id, question, outcome, decided_at, decision_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                (f"r{i}", f"t{i}", "q?", "rejected", f"2025-01-0{i + 1}T00:00:00Z", "{}")
                for i in range(6)
            ],
        )
        store._conn.execute(
            "INSERT INTO decisions "
            "(request_id, task_id, question, outcome, decided_at, decision_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("ra", "ta", "q?", "approved", "2025-01-10T00:00:00Z", "{}"),
        )
        store._conn.commit()
        checker = HealthChecker(store=store)
        health = checker.check()
        # Low-trust + low-approval = 2 issues = C
        assert len(health.issues) == 2
        assert health.grade == "C"


class TestHealthReport:
    """Tests for the text report renderer."""

    def test_empty_report(self) -> None:
        health = CoordinationHealth()
        report = health_report(health)
        assert "Grade: A" in report
        assert "None detected" in report

    def test_full_report(self) -> None:
        health = CoordinationHealth(
            intent_graph=IntentGraphHealth(
                total_intents=5,
                agent_count=3,
                avg_stability=0.8,
                min_stability=0.5,
                max_stability=1.0,
                conflict_count=1,
                provides_count=4,
                requires_count=3,
            ),
            stigmergy=StigmergyHealth(
                total_markers=10,
                markers_by_type={"file_modified": 7, "known_issue": 3},
                avg_strength=0.75,
                unique_agents=3,
                unique_targets=5,
            ),
            scoring=ScoringHealth(
                total_agents=3,
                total_outcomes=15,
                avg_score=0.72,
                min_score=0.4,
                max_score=0.9,
            ),
            voting=VotingHealth(
                total_decisions=8,
                approval_rate=0.75,
                avg_confidence=0.85,
                escalation_count=1,
                outcomes={"approved": 6, "rejected": 1, "escalated": 1},
            ),
            grade="A",
            issues=[],
        )
        report = health_report(health)
        assert "Intents: 5" in report
        assert "Markers: 10" in report
        assert "Agents: 3" in report
        assert "Decisions: 8" in report
        assert "75%" in report

    def test_report_with_issues(self) -> None:
        health = CoordinationHealth(
            issues=["Low trust agents: bad_agent"],
            grade="B",
        )
        report = health_report(health)
        assert "Grade: B" in report
        assert "Low trust agents" in report

    def test_report_no_data_sections(self) -> None:
        """Sections with no data show (no data)."""
        health = CoordinationHealth()
        report = health_report(health)
        assert "(no data)" in report
