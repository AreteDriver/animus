"""Tests for v2 Pydantic models in skills/models.py."""

from animus_forge.skills.models import (
    ContractProvides,
    ContractRequires,
    EscalationRule,
    RoutingExclusion,
    SkillCapability,
    SkillContracts,
    SkillDefinition,
    SkillErrorHandling,
    SkillRouting,
    SkillVerification,
    VerificationCheckpoint,
)


class TestRoutingExclusion:
    def test_defaults(self):
        exc = RoutingExclusion(condition="Single file read")
        assert exc.condition == "Single file read"
        assert exc.instead == ""
        assert exc.reason == ""

    def test_full_construction(self):
        exc = RoutingExclusion(
            condition="Content search by pattern",
            instead="Grep tool directly",
            reason="Grep is faster",
        )
        assert exc.instead == "Grep tool directly"
        assert exc.reason == "Grep is faster"


class TestSkillRouting:
    def test_defaults(self):
        routing = SkillRouting()
        assert routing.use_when == []
        assert routing.do_not_use_when == []

    def test_full_construction(self):
        routing = SkillRouting(
            use_when=["Filesystem operations", "Batch operations"],
            do_not_use_when=[RoutingExclusion(condition="Single file read", instead="Read tool")],
        )
        assert len(routing.use_when) == 2
        assert len(routing.do_not_use_when) == 1
        assert routing.do_not_use_when[0].instead == "Read tool"


class TestVerificationCheckpoint:
    def test_construction(self):
        cp = VerificationCheckpoint(trigger="Before destructive op", action="Verify backup")
        assert cp.trigger == "Before destructive op"
        assert cp.action == "Verify backup"


class TestSkillVerification:
    def test_defaults(self):
        ver = SkillVerification()
        assert ver.pre_conditions == []
        assert ver.post_conditions == []
        assert ver.checkpoints == []
        assert ver.completion_checklist == []

    def test_full_construction(self):
        ver = SkillVerification(
            pre_conditions=["Target path exists"],
            post_conditions=["No protected paths modified"],
            checkpoints=[VerificationCheckpoint(trigger="Before delete", action="Verify backup")],
            completion_checklist=["All operations succeeded", "Backups created"],
        )
        assert len(ver.pre_conditions) == 1
        assert len(ver.checkpoints) == 1
        assert len(ver.completion_checklist) == 2


class TestEscalationRule:
    def test_defaults(self):
        rule = EscalationRule(error_class="recoverable")
        assert rule.error_class == "recoverable"
        assert rule.action == "report"
        assert rule.max_retries == 0
        assert rule.fallback == ""

    def test_full_construction(self):
        rule = EscalationRule(
            error_class="permission",
            description="Insufficient privileges",
            action="escalate",
            max_retries=3,
            fallback="Report to user",
        )
        assert rule.max_retries == 3
        assert rule.action == "escalate"


class TestSkillErrorHandling:
    def test_defaults(self):
        eh = SkillErrorHandling()
        assert eh.escalation == []
        assert eh.self_correction == []

    def test_full_construction(self):
        eh = SkillErrorHandling(
            escalation=[EscalationRule(error_class="recoverable", action="retry", max_retries=3)],
            self_correction=["Destructive op without backup: back up remaining state"],
        )
        assert len(eh.escalation) == 1
        assert eh.escalation[0].max_retries == 3
        assert len(eh.self_correction) == 1


class TestContractProvides:
    def test_defaults(self):
        cp = ContractProvides(name="file_content")
        assert cp.type == "object"
        assert cp.consumers == []
        assert cp.description == ""

    def test_full_construction(self):
        cp = ContractProvides(
            name="search_results",
            type="array",
            consumers=["context-mapper", "intent-author"],
            description="File paths matching criteria",
        )
        assert cp.type == "array"
        assert len(cp.consumers) == 2


class TestContractRequires:
    def test_defaults(self):
        cr = ContractRequires(name="target_paths")
        assert cr.type == "object"
        assert cr.provider == ""

    def test_full_construction(self):
        cr = ContractRequires(
            name="context_map",
            type="object",
            provider="context-mapper",
            description="Project context",
        )
        assert cr.provider == "context-mapper"


class TestSkillContracts:
    def test_defaults(self):
        sc = SkillContracts()
        assert sc.provides == []
        assert sc.requires == []

    def test_full_construction(self):
        sc = SkillContracts(
            provides=[ContractProvides(name="output_a")],
            requires=[ContractRequires(name="input_b", provider="other-skill")],
        )
        assert len(sc.provides) == 1
        assert len(sc.requires) == 1
        assert sc.requires[0].provider == "other-skill"


class TestSkillCapabilityV2Fields:
    def test_defaults(self):
        cap = SkillCapability(name="test_cap")
        assert cap.parallel_safe is True
        assert cap.intent_required is False
        assert cap.post_execution == []

    def test_v2_fields(self):
        cap = SkillCapability(
            name="delete_file",
            parallel_safe=False,
            intent_required=True,
            post_execution=[
                "Verify file no longer exists",
                "Confirm backup accessible",
            ],
        )
        assert cap.parallel_safe is False
        assert cap.intent_required is True
        assert len(cap.post_execution) == 2


class TestSkillDefinitionV2Fields:
    def test_defaults(self):
        skill = SkillDefinition(name="test", agent="system")
        assert skill.type == "agent"
        assert skill.category == ""
        assert skill.risk_level == "low"
        assert skill.consensus_level == "any"
        assert skill.trust == "supervised"
        assert skill.parallel_safe is False
        assert skill.tools == []
        assert skill.routing is None
        assert skill.verification is None
        assert skill.error_handling is None
        assert skill.contracts is None
        assert skill.skill_inputs == {}
        assert skill.skill_outputs == {}

    def test_v2_fields(self):
        skill = SkillDefinition(
            name="file-operations",
            agent="system",
            type="agent",
            category="system",
            risk_level="medium",
            consensus_level="adaptive",
            trust="supervised",
            parallel_safe=False,
            tools=["Read", "Write", "Bash"],
            routing=SkillRouting(
                use_when=["Filesystem operations"],
                do_not_use_when=[
                    RoutingExclusion(condition="Single file read", instead="Read tool")
                ],
            ),
            verification=SkillVerification(
                pre_conditions=["Target path exists"],
                completion_checklist=["All ops succeeded"],
            ),
            error_handling=SkillErrorHandling(
                escalation=[EscalationRule(error_class="recoverable", action="retry")]
            ),
            contracts=SkillContracts(
                provides=[ContractProvides(name="file_content", type="string")],
                requires=[ContractRequires(name="target_paths", provider="context-mapper")],
            ),
            skill_inputs={"path": {"type": "string", "required": True}},
            skill_outputs={"success": {"type": "boolean"}},
        )
        assert skill.category == "system"
        assert skill.risk_level == "medium"
        assert skill.routing is not None
        assert len(skill.routing.use_when) == 1
        assert skill.verification is not None
        assert skill.error_handling is not None
        assert len(skill.error_handling.escalation) == 1
        assert skill.contracts is not None
        assert len(skill.contracts.provides) == 1
        assert skill.skill_inputs["path"]["type"] == "string"

    def test_backward_compatible_with_v1(self):
        """v1-style construction still works without v2 fields."""
        skill = SkillDefinition(
            name="old_skill",
            agent="browser",
            version="1.0.0",
            capabilities=[SkillCapability(name="search", description="Search the web")],
            capability_names=["search"],
            protected_paths=[],
            blocked_patterns=[],
        )
        assert skill.routing is None
        assert skill.type == "agent"
        assert skill.get_capability("search") is not None
