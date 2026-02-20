"""Plugin System.

Provides extensibility for custom step handlers, hooks, and integrations.
Includes marketplace for discovering and installing community plugins.
"""

from .base import (
    Plugin,
    PluginContext,
    PluginHook,
    StepHandler,
)
from .installer import PluginInstaller
from .loader import (
    discover_plugins,
    load_plugin_from_file,
    load_plugins,
)
from .marketplace import PluginMarketplace
from .models import (
    PluginCategory,
    PluginInstallation,
    PluginInstallRequest,
    PluginListing,
    PluginMetadata,
    PluginRelease,
    PluginSearchResult,
    PluginSource,
    PluginUpdateRequest,
)
from .registry import (
    PluginRegistry,
    get_registry,
    register_handler,
    register_plugin,
)

__all__ = [
    # Base
    "Plugin",
    "PluginContext",
    "PluginHook",
    "StepHandler",
    # Registry
    "PluginRegistry",
    "get_registry",
    "register_plugin",
    "register_handler",
    # Loader
    "load_plugins",
    "load_plugin_from_file",
    "discover_plugins",
    # Marketplace models
    "PluginCategory",
    "PluginSource",
    "PluginMetadata",
    "PluginRelease",
    "PluginListing",
    "PluginInstallation",
    "PluginSearchResult",
    "PluginInstallRequest",
    "PluginUpdateRequest",
    # Marketplace
    "PluginMarketplace",
    "PluginInstaller",
]
