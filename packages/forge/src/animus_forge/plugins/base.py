"""Plugin base classes and interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum


class PluginHook(Enum):
    """Available plugin hooks."""

    # Workflow lifecycle
    WORKFLOW_START = "workflow_start"
    WORKFLOW_END = "workflow_end"
    WORKFLOW_ERROR = "workflow_error"

    # Step lifecycle
    STEP_START = "step_start"
    STEP_END = "step_end"
    STEP_ERROR = "step_error"
    STEP_RETRY = "step_retry"

    # Data transformation
    PRE_EXECUTE = "pre_execute"  # Before step execution
    POST_EXECUTE = "post_execute"  # After step execution

    # Custom
    CUSTOM = "custom"


@dataclass
class PluginContext:
    """Context passed to plugin hooks and handlers.

    Contains information about the current workflow/step execution.
    """

    workflow_id: str | None = None
    workflow_name: str | None = None
    execution_id: str | None = None
    step_id: str | None = None
    step_type: str | None = None
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    error: Exception | None = None

    def with_step(
        self,
        step_id: str,
        step_type: str,
        inputs: dict | None = None,
    ) -> PluginContext:
        """Create a copy with step information."""
        return PluginContext(
            workflow_id=self.workflow_id,
            workflow_name=self.workflow_name,
            execution_id=self.execution_id,
            step_id=step_id,
            step_type=step_type,
            inputs=inputs or {},
            outputs={},
            metadata=self.metadata.copy(),
        )

    def with_outputs(self, outputs: dict) -> PluginContext:
        """Create a copy with outputs."""
        return PluginContext(
            workflow_id=self.workflow_id,
            workflow_name=self.workflow_name,
            execution_id=self.execution_id,
            step_id=self.step_id,
            step_type=self.step_type,
            inputs=self.inputs,
            outputs=outputs,
            metadata=self.metadata,
        )


# Type for step handler functions
StepHandler = Callable[[dict, PluginContext], dict]


class Plugin(ABC):
    """Base class for Gorgon plugins.

    Plugins can:
    - Register custom step handlers
    - Hook into workflow/step lifecycle
    - Transform inputs/outputs
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name (must be unique)."""
        pass

    @property
    def version(self) -> str:
        """Plugin version."""
        return "1.0.0"

    @property
    def description(self) -> str:
        """Plugin description."""
        return ""

    def initialize(self, config: dict | None = None) -> None:
        """Initialize the plugin with configuration.

        Called when the plugin is loaded.

        Args:
            config: Plugin-specific configuration
        """
        pass

    def shutdown(self) -> None:
        """Cleanup when plugin is unloaded."""
        pass

    def get_handlers(self) -> dict[str, StepHandler]:
        """Get custom step handlers provided by this plugin.

        Returns:
            Dict mapping step type names to handler functions
        """
        return {}

    def get_hooks(self) -> dict[PluginHook, Callable[[PluginContext], None]]:
        """Get lifecycle hooks provided by this plugin.

        Returns:
            Dict mapping hooks to callback functions
        """
        return {}

    def on_workflow_start(self, context: PluginContext) -> None:
        """Called when a workflow starts."""
        pass

    def on_workflow_end(self, context: PluginContext) -> None:
        """Called when a workflow completes."""
        pass

    def on_workflow_error(self, context: PluginContext) -> None:
        """Called when a workflow fails."""
        pass

    def on_step_start(self, context: PluginContext) -> None:
        """Called when a step starts."""
        pass

    def on_step_end(self, context: PluginContext) -> None:
        """Called when a step completes."""
        pass

    def on_step_error(self, context: PluginContext) -> None:
        """Called when a step fails."""
        pass

    def transform_input(self, step_type: str, inputs: dict) -> dict:
        """Transform step inputs before execution.

        Args:
            step_type: Type of step
            inputs: Original inputs

        Returns:
            Transformed inputs
        """
        return inputs

    def transform_output(self, step_type: str, outputs: dict) -> dict:
        """Transform step outputs after execution.

        Args:
            step_type: Type of step
            outputs: Original outputs

        Returns:
            Transformed outputs
        """
        return outputs


class SimplePlugin(Plugin):
    """Simple plugin implementation for quick customization.

    Allows creating plugins with just functions, without subclassing.
    """

    def __init__(
        self,
        name: str,
        version: str = "1.0.0",
        description: str = "",
        handlers: dict[str, StepHandler] | None = None,
        on_workflow_start: Callable[[PluginContext], None] | None = None,
        on_workflow_end: Callable[[PluginContext], None] | None = None,
        on_step_start: Callable[[PluginContext], None] | None = None,
        on_step_end: Callable[[PluginContext], None] | None = None,
    ):
        """Create a simple plugin.

        Args:
            name: Plugin name
            version: Plugin version
            description: Plugin description
            handlers: Custom step handlers
            on_workflow_start: Workflow start callback
            on_workflow_end: Workflow end callback
            on_step_start: Step start callback
            on_step_end: Step end callback
        """
        self._name = name
        self._version = version
        self._description = description
        self._handlers = handlers or {}
        self._on_workflow_start = on_workflow_start
        self._on_workflow_end = on_workflow_end
        self._on_step_start = on_step_start
        self._on_step_end = on_step_end

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @property
    def description(self) -> str:
        return self._description

    def get_handlers(self) -> dict[str, StepHandler]:
        return self._handlers

    def on_workflow_start(self, context: PluginContext) -> None:
        if self._on_workflow_start:
            self._on_workflow_start(context)

    def on_workflow_end(self, context: PluginContext) -> None:
        if self._on_workflow_end:
            self._on_workflow_end(context)

    def on_step_start(self, context: PluginContext) -> None:
        if self._on_step_start:
            self._on_step_start(context)

    def on_step_end(self, context: PluginContext) -> None:
        if self._on_step_end:
            self._on_step_end(context)
