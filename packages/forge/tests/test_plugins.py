"""Tests for plugin system."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from animus_forge.plugins.base import (
    Plugin,
    PluginContext,
    PluginHook,
    SimplePlugin,
)
from animus_forge.plugins.loader import (
    discover_plugins,
    load_plugin_from_file,
    load_plugin_from_module,
    load_plugins,
)
from animus_forge.plugins.registry import (
    PluginRegistry,
    get_registry,
)


class TestPluginContext:
    """Tests for PluginContext class."""

    def test_create_context(self):
        """Can create plugin context."""
        ctx = PluginContext(
            workflow_id="wf1",
            workflow_name="Test Workflow",
            execution_id="exec1",
        )

        assert ctx.workflow_id == "wf1"
        assert ctx.workflow_name == "Test Workflow"
        assert ctx.execution_id == "exec1"

    def test_context_defaults(self):
        """Context has sensible defaults."""
        ctx = PluginContext()

        assert ctx.workflow_id is None
        assert ctx.inputs == {}
        assert ctx.outputs == {}
        assert ctx.metadata == {}

    def test_with_step(self):
        """Can create context with step info."""
        ctx = PluginContext(workflow_id="wf1", execution_id="exec1")
        step_ctx = ctx.with_step("step1", "shell", inputs={"command": "echo"})

        assert step_ctx.workflow_id == "wf1"
        assert step_ctx.step_id == "step1"
        assert step_ctx.step_type == "shell"
        assert step_ctx.inputs == {"command": "echo"}

    def test_with_outputs(self):
        """Can create context with outputs."""
        ctx = PluginContext(
            workflow_id="wf1",
            step_id="step1",
        )
        out_ctx = ctx.with_outputs({"result": "success"})

        assert out_ctx.workflow_id == "wf1"
        assert out_ctx.step_id == "step1"
        assert out_ctx.outputs == {"result": "success"}


class TestPluginHook:
    """Tests for PluginHook enum."""

    def test_workflow_hooks(self):
        """Workflow lifecycle hooks exist."""
        assert PluginHook.WORKFLOW_START.value == "workflow_start"
        assert PluginHook.WORKFLOW_END.value == "workflow_end"
        assert PluginHook.WORKFLOW_ERROR.value == "workflow_error"

    def test_step_hooks(self):
        """Step lifecycle hooks exist."""
        assert PluginHook.STEP_START.value == "step_start"
        assert PluginHook.STEP_END.value == "step_end"
        assert PluginHook.STEP_ERROR.value == "step_error"
        assert PluginHook.STEP_RETRY.value == "step_retry"

    def test_data_hooks(self):
        """Data transformation hooks exist."""
        assert PluginHook.PRE_EXECUTE.value == "pre_execute"
        assert PluginHook.POST_EXECUTE.value == "post_execute"


class SamplePlugin(Plugin):
    """Sample plugin for testing."""

    @property
    def name(self) -> str:
        return "sample"

    @property
    def version(self) -> str:
        return "2.0.0"

    @property
    def description(self) -> str:
        return "A sample plugin"

    def __init__(self):
        self.initialized = False
        self.shutdown_called = False
        self.config = None
        self.workflow_started = False
        self.step_started = False

    def initialize(self, config=None):
        self.initialized = True
        self.config = config

    def shutdown(self):
        self.shutdown_called = True

    def get_handlers(self):
        def custom_handler(params, context):
            return {"result": "custom"}

        return {"custom": custom_handler}

    def on_workflow_start(self, context):
        self.workflow_started = True

    def on_step_start(self, context):
        self.step_started = True


class TestPlugin:
    """Tests for Plugin base class."""

    def test_plugin_abstract(self):
        """Plugin name is abstract."""
        with pytest.raises(TypeError):
            Plugin()

    def test_sample_plugin(self):
        """Sample plugin works correctly."""
        plugin = SamplePlugin()

        assert plugin.name == "sample"
        assert plugin.version == "2.0.0"
        assert plugin.description == "A sample plugin"

    def test_plugin_initialize(self):
        """Plugin can be initialized."""
        plugin = SamplePlugin()
        plugin.initialize({"key": "value"})

        assert plugin.initialized
        assert plugin.config == {"key": "value"}

    def test_plugin_shutdown(self):
        """Plugin can be shutdown."""
        plugin = SamplePlugin()
        plugin.shutdown()

        assert plugin.shutdown_called

    def test_plugin_handlers(self):
        """Plugin can provide handlers."""
        plugin = SamplePlugin()
        handlers = plugin.get_handlers()

        assert "custom" in handlers
        result = handlers["custom"]({}, PluginContext())
        assert result == {"result": "custom"}

    def test_plugin_lifecycle_hooks(self):
        """Plugin lifecycle hooks are called."""
        plugin = SamplePlugin()
        ctx = PluginContext(workflow_id="wf1")

        plugin.on_workflow_start(ctx)
        assert plugin.workflow_started

        plugin.on_step_start(ctx)
        assert plugin.step_started

    def test_plugin_transform_input(self):
        """Transform input returns inputs by default."""
        plugin = SamplePlugin()
        inputs = {"key": "value"}

        result = plugin.transform_input("shell", inputs)
        assert result == inputs

    def test_plugin_transform_output(self):
        """Transform output returns outputs by default."""
        plugin = SamplePlugin()
        outputs = {"result": "success"}

        result = plugin.transform_output("shell", outputs)
        assert result == outputs


class TestSimplePlugin:
    """Tests for SimplePlugin class."""

    def test_simple_plugin_creation(self):
        """Can create simple plugin."""
        plugin = SimplePlugin(
            name="simple",
            version="1.0.0",
            description="Simple plugin",
        )

        assert plugin.name == "simple"
        assert plugin.version == "1.0.0"
        assert plugin.description == "Simple plugin"

    def test_simple_plugin_handlers(self):
        """Simple plugin can have handlers."""

        def my_handler(params, context):
            return {"handled": True}

        plugin = SimplePlugin(name="test", handlers={"my_type": my_handler})

        handlers = plugin.get_handlers()
        assert "my_type" in handlers

    def test_simple_plugin_callbacks(self):
        """Simple plugin callbacks are invoked."""
        started = []
        ended = []

        plugin = SimplePlugin(
            name="test",
            on_workflow_start=lambda ctx: started.append(ctx.workflow_id),
            on_workflow_end=lambda ctx: ended.append(ctx.workflow_id),
        )

        ctx = PluginContext(workflow_id="wf1")
        plugin.on_workflow_start(ctx)
        plugin.on_workflow_end(ctx)

        assert "wf1" in started
        assert "wf1" in ended


class TestPluginRegistry:
    """Tests for PluginRegistry class."""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry."""
        return PluginRegistry()

    def test_register_plugin(self, registry):
        """Can register a plugin."""
        plugin = SamplePlugin()
        registry.register(plugin)

        assert registry.get_plugin("sample") is plugin

    def test_register_with_config(self, registry):
        """Plugin is initialized with config."""
        plugin = SamplePlugin()
        registry.register(plugin, config={"setting": "value"})

        assert plugin.initialized
        assert plugin.config == {"setting": "value"}

    def test_register_duplicate(self, registry):
        """Cannot register duplicate plugins."""
        plugin1 = SamplePlugin()
        registry.register(plugin1)

        plugin2 = SamplePlugin()
        with pytest.raises(ValueError, match="already registered"):
            registry.register(plugin2)

    def test_unregister_plugin(self, registry):
        """Can unregister a plugin."""
        plugin = SamplePlugin()
        registry.register(plugin)

        result = registry.unregister("sample")

        assert result is True
        assert registry.get_plugin("sample") is None
        assert plugin.shutdown_called

    def test_unregister_nonexistent(self, registry):
        """Unregistering nonexistent returns False."""
        result = registry.unregister("nonexistent")
        assert result is False

    def test_register_handler_direct(self, registry):
        """Can register handler directly."""

        def my_handler(params, context):
            return {"result": "direct"}

        registry.register_handler("direct_type", my_handler)

        handler = registry.get_handler("direct_type")
        assert handler is not None
        result = handler({}, PluginContext())
        assert result == {"result": "direct"}

    def test_get_handler_from_plugin(self, registry):
        """Can get handler from registered plugin."""
        plugin = SamplePlugin()
        registry.register(plugin)

        handler = registry.get_handler("custom")
        assert handler is not None

    def test_get_nonexistent_handler(self, registry):
        """Get nonexistent handler returns None."""
        handler = registry.get_handler("nonexistent")
        assert handler is None

    def test_has_handler(self, registry):
        """Can check if handler exists."""
        plugin = SamplePlugin()
        registry.register(plugin)

        assert registry.has_handler("custom") is True
        assert registry.has_handler("nonexistent") is False

    def test_list_handlers(self, registry):
        """Can list all handlers."""
        plugin = SamplePlugin()
        registry.register(plugin)

        handlers = registry.list_handlers()

        assert "custom" in handlers
        assert handlers["custom"] == "sample"

    def test_trigger_hook(self, registry):
        """Can trigger plugin hooks."""
        plugin = SamplePlugin()
        registry.register(plugin)

        ctx = PluginContext(workflow_id="wf1")
        registry.trigger_hook(PluginHook.WORKFLOW_START, ctx)

        assert plugin.workflow_started

    def test_trigger_hook_error_handled(self, registry):
        """Hook errors don't stop other hooks."""

        def bad_hook(ctx):
            raise RuntimeError("Hook error")

        plugin = SimplePlugin(name="bad", on_workflow_start=bad_hook)
        registry.register(plugin)

        good_called = []
        good_plugin = SimplePlugin(
            name="good",
            on_workflow_start=lambda ctx: good_called.append(True),
        )
        registry.register(good_plugin)

        ctx = PluginContext(workflow_id="wf1")
        # Should not raise
        registry.trigger_hook(PluginHook.WORKFLOW_START, ctx)

        assert len(good_called) == 1

    def test_transform_input(self, registry):
        """Can transform inputs through plugins."""

        class TransformPlugin(Plugin):
            @property
            def name(self):
                return "transform"

            def transform_input(self, step_type, inputs):
                return {**inputs, "transformed": True}

        registry.register(TransformPlugin())

        result = registry.transform_input("shell", {"original": "value"})

        assert result["original"] == "value"
        assert result["transformed"] is True

    def test_transform_output(self, registry):
        """Can transform outputs through plugins."""

        class TransformPlugin(Plugin):
            @property
            def name(self):
                return "transform"

            def transform_output(self, step_type, outputs):
                return {**outputs, "transformed": True}

        registry.register(TransformPlugin())

        result = registry.transform_output("shell", {"result": "success"})

        assert result["result"] == "success"
        assert result["transformed"] is True

    def test_list_plugins(self, registry):
        """Can list all plugins."""
        plugin = SamplePlugin()
        registry.register(plugin)

        plugins = registry.list_plugins()

        assert len(plugins) == 1
        assert plugins[0]["name"] == "sample"
        assert plugins[0]["version"] == "2.0.0"
        assert "custom" in plugins[0]["handlers"]


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_get_registry(self):
        """get_registry returns PluginRegistry."""
        registry = get_registry()
        assert isinstance(registry, PluginRegistry)

    def test_get_registry_singleton(self):
        """get_registry returns same instance."""
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2


class TestPluginLoader:
    """Tests for plugin loading functions."""

    @pytest.fixture
    def plugin_dir(self):
        """Create a temp directory with plugin files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a valid plugin file
            plugin_code = """
from animus_forge.plugins.base import Plugin

class MyPlugin(Plugin):
    @property
    def name(self):
        return "my_plugin"

    @property
    def version(self):
        return "1.0.0"

    def get_handlers(self):
        return {"my_handler": lambda p, c: {"result": "handled"}}
"""
            plugin_path = Path(tmpdir) / "my_plugin.py"
            plugin_path.write_text(plugin_code)

            # Create another plugin
            plugin2_code = """
from animus_forge.plugins.base import Plugin

class AnotherPlugin(Plugin):
    @property
    def name(self):
        return "another"
"""
            plugin2_path = Path(tmpdir) / "another.py"
            plugin2_path.write_text(plugin2_code)

            # Create an invalid file (not a plugin)
            invalid_path = Path(tmpdir) / "invalid.py"
            invalid_path.write_text("x = 1")

            # Create a private file (should be skipped)
            private_path = Path(tmpdir) / "_private.py"
            private_path.write_text("# private")

            yield tmpdir

    def test_load_plugin_from_file(self, plugin_dir):
        """Can load plugin from file."""
        registry = PluginRegistry()
        filepath = Path(plugin_dir) / "my_plugin.py"

        plugin = load_plugin_from_file(
            filepath,
            registry,
            trusted_dir=plugin_dir,
            validate_path=True,
        )

        assert plugin is not None
        assert plugin.name == "my_plugin"
        assert registry.get_plugin("my_plugin") is plugin

    def test_load_plugin_nonexistent_file(self, plugin_dir):
        """Loading nonexistent file returns None."""
        registry = PluginRegistry()

        plugin = load_plugin_from_file(
            Path(plugin_dir) / "nonexistent.py",
            registry,
            trusted_dir=plugin_dir,
            validate_path=False,
        )

        assert plugin is None

    def test_load_plugin_not_py_file(self, plugin_dir):
        """Loading non-.py file returns None."""
        registry = PluginRegistry()
        txt_file = Path(plugin_dir) / "readme.txt"
        txt_file.write_text("readme")

        plugin = load_plugin_from_file(
            txt_file,
            registry,
            trusted_dir=plugin_dir,
            validate_path=True,
        )

        assert plugin is None

    def test_load_plugin_no_plugin_class(self, plugin_dir):
        """Loading file without Plugin class returns None."""
        registry = PluginRegistry()

        plugin = load_plugin_from_file(
            Path(plugin_dir) / "invalid.py",
            registry,
            trusted_dir=plugin_dir,
            validate_path=True,
        )

        assert plugin is None

    def test_discover_plugins(self, plugin_dir):
        """Can discover plugins in directory."""
        registry = PluginRegistry()

        plugins = discover_plugins(
            plugin_dir,
            registry,
            trusted_dir=plugin_dir,
            validate_path=True,
        )

        # Should find my_plugin and another (not invalid or _private)
        names = [p.name for p in plugins]
        assert "my_plugin" in names
        assert "another" in names
        assert len(plugins) == 2

    def test_discover_plugins_nonexistent_dir(self):
        """Discovering in nonexistent dir returns empty."""
        registry = PluginRegistry()

        plugins = discover_plugins(
            "/nonexistent/path",
            registry,
            validate_path=False,
        )

        assert plugins == []

    def test_load_plugins_from_paths(self, plugin_dir):
        """Can load plugins from multiple paths."""
        registry = PluginRegistry()
        sources = [
            Path(plugin_dir) / "my_plugin.py",
            {"path": str(Path(plugin_dir) / "another.py"), "trusted_dir": plugin_dir},
        ]

        plugins = load_plugins(
            sources,
            registry,
            trusted_dir=plugin_dir,
            validate_path=True,
        )

        assert len(plugins) == 2

    def test_load_plugins_from_directory(self, plugin_dir):
        """Can load plugins from directory source."""
        registry = PluginRegistry()
        sources = [{"directory": plugin_dir, "trusted_dir": plugin_dir}]

        plugins = load_plugins(
            sources,
            registry,
            trusted_dir=plugin_dir,
            validate_path=True,
        )

        assert len(plugins) >= 2

    def test_load_plugins_with_config(self, plugin_dir):
        """Plugins receive configuration."""
        registry = PluginRegistry()
        sources = [
            {
                "path": str(Path(plugin_dir) / "my_plugin.py"),
                "config": {"setting": "value"},
                "trusted_dir": plugin_dir,
            }
        ]

        plugins = load_plugins(sources, registry, trusted_dir=plugin_dir)

        assert len(plugins) == 1

    def test_load_plugin_from_module(self):
        """Can load plugin from installed module."""
        registry = PluginRegistry()

        # Try to load a nonexistent module
        plugin = load_plugin_from_module("nonexistent_module_xyz", registry)

        assert plugin is None

    @patch("animus_forge.plugins.loader.importlib.import_module")
    def test_load_plugin_from_module_success(self, mock_import):
        """Successfully loads plugin from module."""

        class MockPlugin(Plugin):
            @property
            def name(self):
                return "mock"

        mock_module = MagicMock()
        mock_module.MockPlugin = MockPlugin
        mock_import.return_value = mock_module

        registry = PluginRegistry()
        plugin = load_plugin_from_module("test.mock_plugin", registry)

        assert plugin is not None
        assert plugin.name == "mock"


class TestPluginIntegration:
    """Integration tests for plugin system."""

    def test_full_plugin_lifecycle(self):
        """Test complete plugin lifecycle."""
        registry = PluginRegistry()
        events = []

        plugin = SimplePlugin(
            name="tracking",
            on_workflow_start=lambda ctx: events.append(("start", ctx.workflow_id)),
            on_step_start=lambda ctx: events.append(("step", ctx.step_id)),
            on_workflow_end=lambda ctx: events.append(("end", ctx.workflow_id)),
        )

        # Register
        registry.register(plugin)
        assert registry.get_plugin("tracking") is not None

        # Trigger hooks
        ctx = PluginContext(workflow_id="wf1")
        registry.trigger_hook(PluginHook.WORKFLOW_START, ctx)

        step_ctx = ctx.with_step("step1", "shell")
        registry.trigger_hook(PluginHook.STEP_START, step_ctx)

        registry.trigger_hook(PluginHook.WORKFLOW_END, ctx)

        assert events == [("start", "wf1"), ("step", "step1"), ("end", "wf1")]

        # Unregister
        registry.unregister("tracking")
        assert registry.get_plugin("tracking") is None

    def test_handler_execution(self):
        """Test handler execution flow."""
        registry = PluginRegistry()

        def custom_handler(params, context):
            return {
                "computed": params.get("value", 0) * 2,
                "context_info": context.workflow_id,
            }

        plugin = SimplePlugin(name="compute", handlers={"compute": custom_handler})
        registry.register(plugin)

        handler = registry.get_handler("compute")
        assert handler is not None

        ctx = PluginContext(workflow_id="wf1")
        result = handler({"value": 5}, ctx)

        assert result["computed"] == 10
        assert result["context_info"] == "wf1"
