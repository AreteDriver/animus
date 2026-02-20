"""Plugin registry for managing loaded plugins."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from .base import Plugin, PluginContext, PluginHook, StepHandler

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Central registry for plugins and handlers.

    Manages plugin lifecycle and provides access to handlers and hooks.
    Thread-safe for concurrent access.
    """

    def __init__(self):
        """Initialize plugin registry."""
        self._lock = threading.Lock()
        self._plugins: dict[str, Plugin] = {}
        self._handlers: dict[str, tuple[str, StepHandler]] = {}  # type -> (plugin_name, handler)
        self._hooks: dict[PluginHook, list[tuple[str, Callable]]] = {
            hook: [] for hook in PluginHook
        }

    def register(self, plugin: Plugin, config: dict | None = None) -> None:
        """Register a plugin.

        Args:
            plugin: Plugin instance
            config: Plugin configuration

        Raises:
            ValueError: If plugin with same name already registered
        """
        with self._lock:
            if plugin.name in self._plugins:
                raise ValueError(f"Plugin already registered: {plugin.name}")

            # Initialize plugin
            plugin.initialize(config)
            self._plugins[plugin.name] = plugin

            # Register handlers
            for step_type, handler in plugin.get_handlers().items():
                if step_type in self._handlers:
                    existing = self._handlers[step_type][0]
                    logger.warning(
                        f"Handler for '{step_type}' from '{plugin.name}' "
                        f"overrides handler from '{existing}'"
                    )
                self._handlers[step_type] = (plugin.name, handler)

            # Register hooks
            hooks = plugin.get_hooks()
            for hook, callback in hooks.items():
                self._hooks[hook].append((plugin.name, callback))

            # Register standard lifecycle hooks
            self._hooks[PluginHook.WORKFLOW_START].append((plugin.name, plugin.on_workflow_start))
            self._hooks[PluginHook.WORKFLOW_END].append((plugin.name, plugin.on_workflow_end))
            self._hooks[PluginHook.WORKFLOW_ERROR].append((plugin.name, plugin.on_workflow_error))
            self._hooks[PluginHook.STEP_START].append((plugin.name, plugin.on_step_start))
            self._hooks[PluginHook.STEP_END].append((plugin.name, plugin.on_step_end))
            self._hooks[PluginHook.STEP_ERROR].append((plugin.name, plugin.on_step_error))

            logger.info(f"Registered plugin: {plugin.name} v{plugin.version}")

    def unregister(self, plugin_name: str) -> bool:
        """Unregister a plugin.

        Args:
            plugin_name: Name of plugin to remove

        Returns:
            True if plugin was removed
        """
        with self._lock:
            plugin = self._plugins.pop(plugin_name, None)
            if not plugin:
                return False

            # Remove handlers
            to_remove = [
                step_type for step_type, (name, _) in self._handlers.items() if name == plugin_name
            ]
            for step_type in to_remove:
                del self._handlers[step_type]

            # Remove hooks
            for hook in PluginHook:
                self._hooks[hook] = [
                    (name, cb) for name, cb in self._hooks[hook] if name != plugin_name
                ]

            # Shutdown plugin
            plugin.shutdown()

            logger.info(f"Unregistered plugin: {plugin_name}")
            return True

    def register_handler(
        self,
        step_type: str,
        handler: StepHandler,
        plugin_name: str = "__direct__",
    ) -> None:
        """Register a step handler directly.

        Args:
            step_type: Step type name
            handler: Handler function
            plugin_name: Name for tracking
        """
        with self._lock:
            self._handlers[step_type] = (plugin_name, handler)
            logger.debug(f"Registered handler for: {step_type}")

    def get_handler(self, step_type: str) -> StepHandler | None:
        """Get handler for a step type.

        Args:
            step_type: Step type name

        Returns:
            Handler function or None
        """
        with self._lock:
            entry = self._handlers.get(step_type)
            return entry[1] if entry else None

    def has_handler(self, step_type: str) -> bool:
        """Check if a handler exists for step type."""
        with self._lock:
            return step_type in self._handlers

    def list_handlers(self) -> dict[str, str]:
        """List all registered handlers.

        Returns:
            Dict mapping step type to plugin name
        """
        with self._lock:
            return {
                step_type: plugin_name for step_type, (plugin_name, _) in self._handlers.items()
            }

    def trigger_hook(self, hook: PluginHook, context: PluginContext) -> None:
        """Trigger a plugin hook.

        Args:
            hook: Hook to trigger
            context: Context to pass to callbacks
        """
        with self._lock:
            callbacks = list(self._hooks.get(hook, []))

        for plugin_name, callback in callbacks:
            try:
                callback(context)
            except Exception as e:
                logger.error(f"Hook error in {plugin_name}.{hook.value}: {e}")

    def transform_input(
        self,
        step_type: str,
        inputs: dict,
    ) -> dict:
        """Run input transformations from all plugins.

        Args:
            step_type: Step type
            inputs: Original inputs

        Returns:
            Transformed inputs
        """
        with self._lock:
            plugins = list(self._plugins.values())

        result = inputs
        for plugin in plugins:
            try:
                result = plugin.transform_input(step_type, result)
            except Exception as e:
                logger.error(f"Transform input error in {plugin.name}: {e}")

        return result

    def transform_output(
        self,
        step_type: str,
        outputs: dict,
    ) -> dict:
        """Run output transformations from all plugins.

        Args:
            step_type: Step type
            outputs: Original outputs

        Returns:
            Transformed outputs
        """
        with self._lock:
            plugins = list(self._plugins.values())

        result = outputs
        for plugin in plugins:
            try:
                result = plugin.transform_output(step_type, result)
            except Exception as e:
                logger.error(f"Transform output error in {plugin.name}: {e}")

        return result

    def get_plugin(self, name: str) -> Plugin | None:
        """Get a plugin by name."""
        with self._lock:
            return self._plugins.get(name)

    def list_plugins(self) -> list[dict]:
        """List all registered plugins.

        Returns:
            List of plugin info dicts
        """
        with self._lock:
            return [
                {
                    "name": p.name,
                    "version": p.version,
                    "description": p.description,
                    "handlers": list(p.get_handlers().keys()),
                }
                for p in self._plugins.values()
            ]


# Global registry instance
_registry: PluginRegistry | None = None
_registry_lock = threading.Lock()


def get_registry() -> PluginRegistry:
    """Get the global plugin registry."""
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = PluginRegistry()
        return _registry


def register_plugin(plugin: Plugin, config: dict | None = None) -> None:
    """Register a plugin with the global registry."""
    get_registry().register(plugin, config)


def register_handler(step_type: str, handler: StepHandler) -> None:
    """Register a step handler with the global registry."""
    get_registry().register_handler(step_type, handler)
