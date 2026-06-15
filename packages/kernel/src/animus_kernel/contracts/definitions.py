"""Predefined agent contracts for standard roles."""

from __future__ import annotations

from .base import AgentContract, AgentRole

PLANNER_CONTRACT = AgentContract(
    role=AgentRole.PLANNER,
    description="Plans implementation by breaking down requests into tasks",
    input_schema={
        "type": "object",
        "required": ["request", "context"],
        "properties": {
            "request": {
                "type": "string",
                "minLength": 1,
                "description": "The user's request or feature description",
            },
            "context": {
                "type": "object",
                "description": "Current codebase/project context",
            },
            "constraints": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Constraints or requirements to consider",
            },
        },
    },
    output_schema={
        "type": "object",
        "required": ["tasks", "architecture", "success_criteria"],
        "properties": {
            "tasks": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["id", "description", "dependencies"],
                    "properties": {
                        "id": {"type": "string"},
                        "description": {"type": "string"},
                        "dependencies": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "estimated_complexity": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                        },
                    },
                },
            },
            "architecture": {
                "type": "string",
                "description": "High-level architecture description",
            },
            "success_criteria": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            },
            "risks": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
    },
    required_context=["codebase_summary"],
)


BUILDER_CONTRACT = AgentContract(
    role=AgentRole.BUILDER,
    description="Implements code based on the plan",
    input_schema={
        "type": "object",
        "required": ["plan", "task_id"],
        "properties": {
            "plan": {
                "type": "object",
                "description": "The plan from the planner agent",
            },
            "task_id": {
                "type": "string",
                "description": "ID of the task to implement",
            },
            "previous_attempts": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Previous failed attempts with feedback",
            },
            "feedback": {
                "type": "string",
                "description": "Feedback from tester/reviewer",
            },
        },
    },
    output_schema={
        "type": "object",
        "required": ["code", "files_created", "status"],
        "properties": {
            "code": {
                "type": "string",
                "description": "The implemented code",
            },
            "files_created": {
                "type": "array",
                "items": {"type": "string"},
            },
            "files_modified": {
                "type": "array",
                "items": {"type": "string"},
            },
            "status": {
                "type": "string",
                "enum": ["complete", "partial", "blocked"],
            },
            "notes": {
                "type": "string",
            },
            "dependencies_added": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
    },
    required_context=["plan"],
)


TESTER_CONTRACT = AgentContract(
    role=AgentRole.TESTER,
    description="Tests the implemented code",
    input_schema={
        "type": "object",
        "required": ["code", "success_criteria"],
        "properties": {
            "code": {
                "type": "string",
                "description": "Code to test",
            },
            "files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Files to test",
            },
            "success_criteria": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            },
            "test_types": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["unit", "integration", "e2e"],
                },
            },
        },
    },
    output_schema={
        "type": "object",
        "required": ["passed", "tests_run", "results"],
        "properties": {
            "passed": {
                "type": "boolean",
            },
            "tests_run": {
                "type": "integer",
                "minimum": 0,
            },
            "tests_passed": {
                "type": "integer",
                "minimum": 0,
            },
            "tests_failed": {
                "type": "integer",
                "minimum": 0,
            },
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name", "passed"],
                    "properties": {
                        "name": {"type": "string"},
                        "passed": {"type": "boolean"},
                        "error": {"type": "string"},
                        "feedback_for_builder": {"type": "string"},
                    },
                },
            },
            "coverage": {
                "type": "number",
                "minimum": 0,
                "maximum": 100,
            },
            "recommendations": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
    },
    required_context=["code", "success_criteria"],
)


REVIEWER_CONTRACT = AgentContract(
    role=AgentRole.REVIEWER,
    description="Reviews code quality and approves for merge",
    input_schema={
        "type": "object",
        "required": ["code", "plan", "test_results"],
        "properties": {
            "code": {
                "type": "string",
            },
            "plan": {
                "type": "object",
            },
            "test_results": {
                "type": "object",
            },
            "review_focus": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific areas to focus review on",
            },
        },
    },
    output_schema={
        "type": "object",
        "required": ["approved", "score", "findings"],
        "properties": {
            "approved": {
                "type": "boolean",
            },
            "score": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
            },
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["severity", "category", "description"],
                    "properties": {
                        "severity": {
                            "type": "string",
                            "enum": ["critical", "major", "minor", "suggestion"],
                        },
                        "category": {"type": "string"},
                        "description": {"type": "string"},
                        "line_number": {"type": "integer"},
                        "suggested_fix": {"type": "string"},
                    },
                },
            },
            "summary": {
                "type": "string",
            },
            "requires_rework": {
                "type": "boolean",
            },
            "rework_instructions": {
                "type": "string",
            },
        },
    },
    required_context=["code", "plan", "test_results"],
)


DATA_ANALYST_CONTRACT = AgentContract(
    role=AgentRole.DATA_ANALYST,
    description="Analyzes data, creates SQL queries, pandas pipelines, and visualizations",
    input_schema={
        "type": "object",
        "required": ["request", "data_context"],
        "properties": {
            "request": {
                "type": "string",
                "minLength": 1,
                "description": "Analysis request or question to answer",
            },
            "data_context": {
                "type": "object",
                "description": "Information about available data sources",
                "properties": {
                    "tables": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Available database tables with schemas",
                    },
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Available data files (CSV, JSON, Parquet)",
                    },
                    "database_type": {
                        "type": "string",
                        "enum": [
                            "postgresql",
                            "mysql",
                            "sqlite",
                            "bigquery",
                            "snowflake",
                            "other",
                        ],
                    },
                },
            },
            "output_format": {
                "type": "string",
                "enum": ["sql", "pandas", "visualization", "report", "all"],
                "description": "Desired output format",
            },
            "constraints": {
                "type": "object",
                "properties": {
                    "max_rows": {"type": "integer"},
                    "time_range": {"type": "string"},
                    "privacy_level": {
                        "type": "string",
                        "enum": ["public", "internal", "confidential"],
                    },
                },
            },
        },
    },
    output_schema={
        "type": "object",
        "required": ["analysis", "code", "status"],
        "properties": {
            "analysis": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "key_findings": {"type": "array", "items": {"type": "string"}},
                    "recommendations": {"type": "array", "items": {"type": "string"}},
                },
            },
            "code": {
                "type": "object",
                "properties": {
                    "sql_queries": {"type": "array", "items": {"type": "string"}},
                    "python_code": {"type": "string"},
                    "visualization_code": {"type": "string"},
                },
            },
            "status": {
                "type": "string",
                "enum": ["complete", "partial", "needs_more_data"],
            },
            "data_quality_notes": {
                "type": "array",
                "items": {"type": "string"},
            },
            "follow_up_questions": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
    },
    required_context=["data_context"],
)


DEVOPS_CONTRACT = AgentContract(
    role=AgentRole.DEVOPS,
    description="Creates infrastructure, CI/CD pipelines, Docker, Kubernetes, and cloud configurations",
    input_schema={
        "type": "object",
        "required": ["request", "target_platform"],
        "properties": {
            "request": {
                "type": "string",
                "minLength": 1,
                "description": "Infrastructure or DevOps task description",
            },
            "target_platform": {
                "type": "string",
                "enum": [
                    "aws",
                    "gcp",
                    "azure",
                    "kubernetes",
                    "docker",
                    "terraform",
                    "github_actions",
                    "gitlab_ci",
                    "generic",
                ],
            },
            "existing_infrastructure": {
                "type": "object",
                "description": "Current infrastructure context",
            },
            "requirements": {
                "type": "object",
                "properties": {
                    "high_availability": {"type": "boolean"},
                    "auto_scaling": {"type": "boolean"},
                    "environment": {
                        "type": "string",
                        "enum": ["dev", "staging", "prod"],
                    },
                    "budget_tier": {
                        "type": "string",
                        "enum": ["minimal", "standard", "enterprise"],
                    },
                },
            },
            "security_requirements": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
    },
    output_schema={
        "type": "object",
        "required": ["configs", "instructions", "status"],
        "properties": {
            "configs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name", "type", "content"],
                    "properties": {
                        "name": {"type": "string"},
                        "type": {
                            "type": "string",
                            "enum": [
                                "dockerfile",
                                "docker_compose",
                                "kubernetes",
                                "terraform",
                                "github_actions",
                                "gitlab_ci",
                                "shell_script",
                                "config",
                            ],
                        },
                        "content": {"type": "string"},
                        "file_path": {"type": "string"},
                    },
                },
            },
            "instructions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["step", "action"],
                    "properties": {
                        "step": {"type": "integer"},
                        "action": {"type": "string"},
                        "command": {"type": "string"},
                        "notes": {"type": "string"},
                    },
                },
            },
            "status": {
                "type": "string",
                "enum": ["complete", "partial", "needs_manual_setup"],
            },
            "architecture_diagram": {"type": "string"},
            "cost_estimate": {
                "type": "object",
                "properties": {
                    "monthly": {"type": "string"},
                    "breakdown": {"type": "array", "items": {"type": "object"}},
                },
            },
            "security_notes": {"type": "array", "items": {"type": "string"}},
            "monitoring_setup": {"type": "object"},
        },
    },
    required_context=["target_platform"],
)


SECURITY_AUDITOR_CONTRACT = AgentContract(
    role=AgentRole.SECURITY_AUDITOR,
    description="Performs security audits, vulnerability scanning, and OWASP compliance checks",
    input_schema={
        "type": "object",
        "required": ["target", "audit_type"],
        "properties": {
            "target": {
                "type": "object",
                "description": "What to audit",
                "properties": {
                    "code": {"type": "string"},
                    "files": {"type": "array", "items": {"type": "string"}},
                    "dependencies": {"type": "array", "items": {"type": "string"}},
                    "infrastructure": {"type": "object"},
                },
            },
            "audit_type": {
                "type": "string",
                "enum": [
                    "code_review",
                    "dependency_scan",
                    "infrastructure",
                    "full_audit",
                    "owasp_top_10",
                    "compliance",
                ],
            },
            "compliance_frameworks": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["owasp", "pci_dss", "hipaa", "soc2", "gdpr", "iso27001"],
                },
            },
            "severity_threshold": {
                "type": "string",
                "enum": ["critical", "high", "medium", "low", "all"],
            },
        },
    },
    output_schema={
        "type": "object",
        "required": ["vulnerabilities", "risk_score", "status"],
        "properties": {
            "vulnerabilities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["severity", "category", "description"],
                    "properties": {
                        "severity": {
                            "type": "string",
                            "enum": ["critical", "high", "medium", "low", "info"],
                        },
                        "category": {"type": "string"},
                        "description": {"type": "string"},
                        "location": {"type": "string"},
                        "cve_id": {"type": "string"},
                        "remediation": {"type": "string"},
                        "references": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "risk_score": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
            },
            "status": {
                "type": "string",
                "enum": ["pass", "fail", "needs_review"],
            },
            "summary": {"type": "string"},
            "compliance_status": {
                "type": "object",
                "additionalProperties": {"type": "boolean"},
            },
            "recommendations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "priority": {"type": "integer"},
                        "action": {"type": "string"},
                        "effort": {"type": "string", "enum": ["low", "medium", "high"]},
                    },
                },
            },
            "dependency_report": {
                "type": "object",
                "properties": {
                    "total": {"type": "integer"},
                    "vulnerable": {"type": "integer"},
                    "outdated": {"type": "integer"},
                },
            },
        },
    },
    required_context=["target", "audit_type"],
)


MIGRATOR_CONTRACT = AgentContract(
    role=AgentRole.MIGRATOR,
    description="Handles code migrations, framework upgrades, and large-scale refactoring",
    input_schema={
        "type": "object",
        "required": ["migration_type", "source", "target"],
        "properties": {
            "migration_type": {
                "type": "string",
                "enum": [
                    "framework_upgrade",
                    "language_migration",
                    "api_migration",
                    "database_migration",
                    "dependency_upgrade",
                    "refactoring",
                ],
            },
            "source": {
                "type": "object",
                "properties": {
                    "framework": {"type": "string"},
                    "version": {"type": "string"},
                    "files": {"type": "array", "items": {"type": "string"}},
                    "code": {"type": "string"},
                },
            },
            "target": {
                "type": "object",
                "properties": {
                    "framework": {"type": "string"},
                    "version": {"type": "string"},
                },
            },
            "constraints": {
                "type": "object",
                "properties": {
                    "preserve_behavior": {"type": "boolean"},
                    "maintain_tests": {"type": "boolean"},
                    "incremental": {"type": "boolean"},
                },
            },
            "scope": {
                "type": "string",
                "enum": ["full", "partial", "analysis_only"],
            },
        },
    },
    output_schema={
        "type": "object",
        "required": ["migration_plan", "changes", "status"],
        "properties": {
            "migration_plan": {
                "type": "object",
                "properties": {
                    "phases": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "phase": {"type": "integer"},
                                "description": {"type": "string"},
                                "files_affected": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "risk_level": {
                                    "type": "string",
                                    "enum": ["low", "medium", "high"],
                                },
                            },
                        },
                    },
                    "estimated_effort": {"type": "string"},
                    "rollback_strategy": {"type": "string"},
                },
            },
            "changes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["file", "change_type"],
                    "properties": {
                        "file": {"type": "string"},
                        "change_type": {
                            "type": "string",
                            "enum": ["modify", "create", "delete", "rename"],
                        },
                        "original_code": {"type": "string"},
                        "new_code": {"type": "string"},
                        "explanation": {"type": "string"},
                    },
                },
            },
            "status": {
                "type": "string",
                "enum": ["complete", "partial", "blocked", "needs_manual_review"],
            },
            "breaking_changes": {
                "type": "array",
                "items": {"type": "string"},
            },
            "deprecation_warnings": {
                "type": "array",
                "items": {"type": "string"},
            },
            "test_updates_required": {
                "type": "array",
                "items": {"type": "string"},
            },
            "documentation_updates": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
    },
    required_context=["source", "target"],
)


MODEL_BUILDER_CONTRACT = AgentContract(
    role=AgentRole.MODEL_BUILDER,
    description="Creates and modifies 3D models, scenes, and assets for Unity, Blender, and other 3D tools",
    input_schema={
        "type": "object",
        "required": ["request", "target_platform"],
        "properties": {
            "request": {
                "type": "string",
                "minLength": 1,
                "description": "Description of the 3D model or scene to create/modify",
            },
            "target_platform": {
                "type": "string",
                "enum": ["unity", "blender", "unreal", "godot", "threejs", "generic"],
                "description": "Target 3D platform or engine",
            },
            "asset_type": {
                "type": "string",
                "enum": [
                    "model",
                    "scene",
                    "material",
                    "shader",
                    "animation",
                    "prefab",
                    "script",
                ],
                "description": "Type of 3D asset to work with",
            },
            "existing_assets": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of existing asset files to reference or modify",
            },
            "specifications": {
                "type": "object",
                "description": "Technical specifications (poly count, texture size, etc.)",
                "properties": {
                    "max_polygons": {"type": "integer", "minimum": 1},
                    "texture_resolution": {"type": "string"},
                    "target_fps": {"type": "integer", "minimum": 1},
                    "lod_levels": {"type": "integer", "minimum": 1, "maximum": 10},
                },
            },
            "style_reference": {
                "type": "string",
                "description": "Style guide or reference for the asset",
            },
            "constraints": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Technical or artistic constraints",
            },
        },
    },
    output_schema={
        "type": "object",
        "required": ["assets", "instructions", "status"],
        "properties": {
            "assets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name", "type", "content"],
                    "properties": {
                        "name": {"type": "string"},
                        "type": {
                            "type": "string",
                            "enum": [
                                "script",
                                "shader",
                                "material_config",
                                "scene_config",
                                "prefab_config",
                                "animation_config",
                                "model_config",
                            ],
                        },
                        "content": {"type": "string"},
                        "file_path": {"type": "string"},
                        "description": {"type": "string"},
                    },
                },
            },
            "instructions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["step", "action"],
                    "properties": {
                        "step": {"type": "integer", "minimum": 1},
                        "action": {"type": "string"},
                        "tool": {"type": "string"},
                        "details": {"type": "string"},
                    },
                },
                "description": "Step-by-step instructions for manual tasks",
            },
            "status": {
                "type": "string",
                "enum": ["complete", "partial", "needs_manual_work"],
            },
            "dependencies": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Required packages, assets, or plugins",
            },
            "optimization_notes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Performance optimization suggestions",
            },
            "warnings": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Potential issues or considerations",
            },
        },
    },
    required_context=["target_platform"],
)


# Registry of all contracts by role
_CONTRACT_REGISTRY: dict[AgentRole, AgentContract] = {
    AgentRole.PLANNER: PLANNER_CONTRACT,
    AgentRole.BUILDER: BUILDER_CONTRACT,
    AgentRole.TESTER: TESTER_CONTRACT,
    AgentRole.REVIEWER: REVIEWER_CONTRACT,
    AgentRole.DATA_ANALYST: DATA_ANALYST_CONTRACT,
    AgentRole.DEVOPS: DEVOPS_CONTRACT,
    AgentRole.SECURITY_AUDITOR: SECURITY_AUDITOR_CONTRACT,
    AgentRole.MIGRATOR: MIGRATOR_CONTRACT,
    AgentRole.MODEL_BUILDER: MODEL_BUILDER_CONTRACT,
}


def get_contract(role: AgentRole | str) -> AgentContract:
    """Get the contract for a given role.

    Args:
        role: AgentRole enum or string role name

    Returns:
        The AgentContract for that role

    Raises:
        ValueError: If role is not found
    """
    if isinstance(role, str):
        try:
            role = AgentRole(role)
        except ValueError:
            raise ValueError(f"Unknown role: {role}")

    if role not in _CONTRACT_REGISTRY:
        raise ValueError(f"No contract defined for role: {role.value}")

    return _CONTRACT_REGISTRY[role]


def register_contract(contract: AgentContract) -> None:
    """Register a custom contract for a role.

    Args:
        contract: The AgentContract to register
    """
    _CONTRACT_REGISTRY[contract.role] = contract


def list_contracts() -> list[dict]:
    """List all registered contracts.

    Returns:
        List of contract summaries
    """
    return [
        {
            "role": role.value,
            "description": contract.description,
            "required_context": contract.required_context,
        }
        for role, contract in _CONTRACT_REGISTRY.items()
    ]
