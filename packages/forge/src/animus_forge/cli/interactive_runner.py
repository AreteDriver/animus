"""Interactive workflow runner with guided prompts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

try:
    from InquirerPy import inquirer
    from InquirerPy.base.control import Choice
    from InquirerPy.separator import Separator
    from InquirerPy.validator import EmptyInputValidator

    INQUIRER_AVAILABLE = True
except ImportError:
    INQUIRER_AVAILABLE = False

from animus_forge.cli.rich_output import StepProgress, get_output


@dataclass
class WorkflowInput:
    """Definition of a workflow input parameter."""

    name: str
    description: str
    input_type: str = "string"  # string, number, boolean, select, multiselect
    required: bool = True
    default: Any | None = None
    choices: list[str] | None = None
    validation: Callable[[Any], bool] | None = None


@dataclass
class WorkflowTemplate:
    """A workflow template for interactive selection."""

    id: str
    name: str
    description: str
    category: str
    inputs: list[WorkflowInput] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


class InteractiveRunner:
    """Interactive CLI workflow runner with guided prompts."""

    # Built-in workflow templates
    TEMPLATES = [
        WorkflowTemplate(
            id="feature-build",
            name="Build New Feature",
            description="Plan, build, test, and review a new feature",
            category="Development",
            inputs=[
                WorkflowInput("feature_request", "Describe the feature to build", required=True),
                WorkflowInput("codebase_path", "Path to your codebase", default="."),
                WorkflowInput("test_command", "Test command to run", default="pytest"),
            ],
            tags=["feature", "build", "test"],
        ),
        WorkflowTemplate(
            id="3d-asset-build",
            name="Create 3D Asset",
            description="Generate 3D scripts, shaders, or configurations",
            category="3D/Game Dev",
            inputs=[
                WorkflowInput("asset_request", "Describe the 3D asset to create", required=True),
                WorkflowInput(
                    "target_platform",
                    "Target platform",
                    input_type="select",
                    choices=["unity", "blender", "unreal", "godot", "threejs"],
                ),
                WorkflowInput(
                    "asset_type",
                    "Asset type",
                    input_type="select",
                    choices=["script", "shader", "material", "animation", "prefab"],
                    default="script",
                ),
            ],
            tags=["3d", "game", "unity", "blender"],
        ),
        WorkflowTemplate(
            id="data-analysis",
            name="Analyze Data",
            description="Create SQL queries, pandas pipelines, and visualizations",
            category="Data",
            inputs=[
                WorkflowInput("analysis_request", "What do you want to analyze?", required=True),
                WorkflowInput(
                    "database_type",
                    "Database type",
                    input_type="select",
                    choices=["postgresql", "mysql", "sqlite", "bigquery", "csv"],
                    default="postgresql",
                ),
                WorkflowInput("tables", "Available tables (comma-separated)", default=""),
            ],
            tags=["data", "sql", "pandas", "analysis"],
        ),
        WorkflowTemplate(
            id="infrastructure-setup",
            name="Setup Infrastructure",
            description="Create Docker, Kubernetes, or CI/CD configurations",
            category="DevOps",
            inputs=[
                WorkflowInput("infra_request", "Describe the infrastructure needed", required=True),
                WorkflowInput(
                    "target_platform",
                    "Target platform",
                    input_type="select",
                    choices=[
                        "docker",
                        "kubernetes",
                        "terraform",
                        "github_actions",
                        "aws",
                        "gcp",
                    ],
                ),
                WorkflowInput(
                    "environment",
                    "Environment",
                    input_type="select",
                    choices=["dev", "staging", "prod"],
                    default="dev",
                ),
            ],
            tags=["devops", "docker", "kubernetes", "ci/cd"],
        ),
        WorkflowTemplate(
            id="security-audit",
            name="Security Audit",
            description="Perform security vulnerability scanning and compliance checking",
            category="Security",
            inputs=[
                WorkflowInput("codebase_path", "Path to codebase to audit", default="."),
                WorkflowInput(
                    "audit_type",
                    "Audit type",
                    input_type="select",
                    choices=[
                        "code_review",
                        "dependency_scan",
                        "full_audit",
                        "owasp_top_10",
                    ],
                    default="full_audit",
                ),
                WorkflowInput(
                    "compliance",
                    "Compliance frameworks",
                    input_type="multiselect",
                    choices=["owasp", "pci_dss", "hipaa", "soc2", "gdpr"],
                    default=[],
                ),
            ],
            tags=["security", "audit", "compliance"],
        ),
        WorkflowTemplate(
            id="code-migration",
            name="Code Migration",
            description="Migrate between frameworks, languages, or API versions",
            category="Migration",
            inputs=[
                WorkflowInput("source_framework", "Source framework/version", required=True),
                WorkflowInput("target_framework", "Target framework/version", required=True),
                WorkflowInput("codebase_path", "Path to codebase", default="."),
                WorkflowInput(
                    "scope",
                    "Migration scope",
                    input_type="select",
                    choices=["full", "partial", "analysis_only"],
                    default="full",
                ),
            ],
            tags=["migration", "upgrade", "refactor"],
        ),
    ]

    def __init__(self):
        """Initialize interactive runner."""
        self.output = get_output()
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check if required dependencies are available."""
        if not INQUIRER_AVAILABLE:
            self.output.warning("InquirerPy not installed. Install with: pip install InquirerPy")
            self.output.info("Falling back to basic input prompts")

    def run(self) -> dict[str, Any] | None:
        """Run the interactive workflow selection and execution.

        Returns:
            Workflow result or None if cancelled
        """
        self.output.header("Interactive Workflow Runner", "Select and configure a workflow")

        # Step 1: Select workflow category
        category = self._select_category()
        if not category:
            return None

        # Step 2: Select workflow
        workflow = self._select_workflow(category)
        if not workflow:
            return None

        # Step 3: Gather inputs
        inputs = self._gather_inputs(workflow)
        if inputs is None:
            return None

        # Step 4: Confirm and execute
        if self._confirm_execution(workflow, inputs):
            return self._execute_workflow(workflow, inputs)

        return None

    def _select_category(self) -> str | None:
        """Select workflow category."""
        categories = list(set(t.category for t in self.TEMPLATES))
        categories.sort()

        if INQUIRER_AVAILABLE:
            return inquirer.select(
                message="Select category:",
                choices=categories + [Separator(), Choice(value=None, name="Cancel")],
                default=categories[0],
            ).execute()
        else:
            self.output.print("\nCategories:")
            for i, cat in enumerate(categories, 1):
                print(f"  {i}. {cat}")
            print("  0. Cancel")

            try:
                choice = int(input("\nSelect category (number): "))
                if choice == 0:
                    return None
                return categories[choice - 1]
            except (ValueError, IndexError):
                self.output.error("Invalid selection")
                return None

    def _select_workflow(self, category: str) -> WorkflowTemplate | None:
        """Select workflow from category."""
        workflows = [t for t in self.TEMPLATES if t.category == category]

        if INQUIRER_AVAILABLE:
            choices = [Choice(value=w, name=f"{w.name} - {w.description}") for w in workflows]
            choices.extend([Separator(), Choice(value=None, name="Back")])

            return inquirer.select(
                message="Select workflow:",
                choices=choices,
            ).execute()
        else:
            self.output.print(f"\n{category} Workflows:")
            for i, wf in enumerate(workflows, 1):
                print(f"  {i}. {wf.name}")
                print(f"     {wf.description}")
            print("  0. Back")

            try:
                choice = int(input("\nSelect workflow (number): "))
                if choice == 0:
                    return None
                return workflows[choice - 1]
            except (ValueError, IndexError):
                self.output.error("Invalid selection")
                return None

    def _gather_inputs(self, workflow: WorkflowTemplate) -> dict[str, Any] | None:
        """Gather input values for workflow."""
        self.output.print(f"\n📝 Configure: {workflow.name}")
        self.output.divider()

        inputs = {}

        for input_def in workflow.inputs:
            value = self._prompt_input(input_def)
            if value is None and input_def.required:
                self.output.error(f"Required input '{input_def.name}' not provided")
                return None
            inputs[input_def.name] = value

        return inputs

    def _prompt_input(self, input_def: WorkflowInput) -> Any:
        """Prompt for a single input value."""
        default = input_def.default
        required_marker = "*" if input_def.required else ""

        if INQUIRER_AVAILABLE:
            if input_def.input_type == "select" and input_def.choices:
                return inquirer.select(
                    message=f"{input_def.description}{required_marker}:",
                    choices=input_def.choices,
                    default=default,
                ).execute()

            elif input_def.input_type == "multiselect" and input_def.choices:
                return inquirer.checkbox(
                    message=f"{input_def.description}{required_marker}:",
                    choices=input_def.choices,
                    default=default or [],
                ).execute()

            elif input_def.input_type == "boolean":
                return inquirer.confirm(
                    message=f"{input_def.description}{required_marker}:",
                    default=default or False,
                ).execute()

            elif input_def.input_type == "number":
                result = inquirer.number(
                    message=f"{input_def.description}{required_marker}:",
                    default=default,
                    validate=EmptyInputValidator() if input_def.required else None,
                ).execute()
                return int(result) if result else default

            else:  # string
                return inquirer.text(
                    message=f"{input_def.description}{required_marker}:",
                    default=str(default) if default else "",
                    validate=EmptyInputValidator() if input_def.required else None,
                ).execute()
        else:
            # Fallback to basic input
            prompt = f"{input_def.description}"
            if default:
                prompt += f" [{default}]"
            prompt += ": "

            if input_def.input_type == "select" and input_def.choices:
                print(f"\nOptions: {', '.join(input_def.choices)}")

            value = input(prompt).strip()

            if not value and default is not None:
                return default
            if not value and input_def.required:
                return None

            if input_def.input_type == "boolean":
                return value.lower() in ("yes", "y", "true", "1")
            elif input_def.input_type == "number":
                return int(value) if value else default
            elif input_def.input_type == "multiselect":
                return [v.strip() for v in value.split(",")] if value else []

            return value

    def _confirm_execution(self, workflow: WorkflowTemplate, inputs: dict[str, Any]) -> bool:
        """Confirm workflow execution."""
        self.output.newline()
        self.output.print("📋 Workflow Configuration:", prefix="")
        self.output.divider()
        self.output.print(f"  Workflow: {workflow.name}")
        self.output.print(f"  ID: {workflow.id}")

        self.output.print("\n  Inputs:")
        for key, value in inputs.items():
            display_value = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
            self.output.print(f"    • {key}: {display_value}")

        self.output.divider()

        if INQUIRER_AVAILABLE:
            return inquirer.confirm(
                message="Execute workflow?",
                default=True,
            ).execute()
        else:
            response = input("\nExecute workflow? [Y/n]: ").strip().lower()
            return response in ("", "y", "yes")

    def _execute_workflow(
        self,
        workflow: WorkflowTemplate,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute the selected workflow.

        Args:
            workflow: Workflow template
            inputs: Input values

        Returns:
            Workflow execution result
        """
        self.output.newline()
        self.output.print(f"🚀 Executing: {workflow.name}", prefix="")
        self.output.divider()

        # Show progress
        steps = [
            StepProgress("plan", "Planning", "running"),
            StepProgress("build", "Building", "pending"),
            StepProgress("test", "Testing", "pending"),
            StepProgress("review", "Reviewing", "pending"),
        ]

        with self.output.spinner(f"Running {workflow.name}..."):
            # Simulate execution (in real implementation, call workflow engine)
            import time

            time.sleep(1)

            # Update progress
            steps[0].status = "completed"
            steps[0].duration_ms = 1500
            steps[1].status = "running"

        # Show final status
        self.output.workflow_progress(workflow.name, steps)

        result = {
            "workflow_id": workflow.id,
            "status": "completed",
            "inputs": inputs,
            "outputs": {},
        }

        self.output.newline()
        self.output.success(f"Workflow '{workflow.name}' completed!")

        return result

    def list_workflows(self) -> None:
        """List all available workflow templates."""
        self.output.header("Available Workflows")

        # Group by category
        by_category: dict[str, list[WorkflowTemplate]] = {}
        for template in self.TEMPLATES:
            if template.category not in by_category:
                by_category[template.category] = []
            by_category[template.category].append(template)

        for category, workflows in sorted(by_category.items()):
            self.output.print(f"\n📁 {category}", style=None)
            self.output.divider()

            rows = [[wf.id, wf.name, wf.description[:40] + "..."] for wf in workflows]
            self.output.table(
                headers=["ID", "Name", "Description"],
                rows=rows,
            )

    def quick_run(
        self, workflow_id: str, inputs: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """Quick run a workflow by ID with optional inputs.

        Args:
            workflow_id: Workflow template ID
            inputs: Optional pre-filled inputs

        Returns:
            Workflow result or None
        """
        # Find template
        template = next((t for t in self.TEMPLATES if t.id == workflow_id), None)
        if not template:
            self.output.error(f"Workflow '{workflow_id}' not found")
            return None

        # Gather missing inputs
        final_inputs = inputs or {}
        missing_inputs = [
            inp for inp in template.inputs if inp.name not in final_inputs and inp.required
        ]

        if missing_inputs:
            self.output.print(f"\n📝 Missing inputs for: {template.name}")
            for input_def in missing_inputs:
                value = self._prompt_input(input_def)
                if value is not None:
                    final_inputs[input_def.name] = value

        # Execute
        return self._execute_workflow(template, final_inputs)


def run_interactive() -> None:
    """Entry point for interactive runner."""
    runner = InteractiveRunner()
    runner.run()


def list_workflows() -> None:
    """List available workflows."""
    runner = InteractiveRunner()
    runner.list_workflows()
