"""Plugin discovery and loading."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

from animus_forge.errors import ValidationError
from animus_forge.utils.validation import validate_safe_path

from .base import Plugin
from .registry import PluginRegistry, get_registry

logger = logging.getLogger(__name__)


def _get_plugins_dir() -> Path:
    """Get the trusted plugins directory from settings."""
    try:
        from animus_forge.config import get_settings

        return get_settings().plugins_dir
    except Exception:
        # Fallback if settings not available
        return Path(__file__).parent / "custom"


def load_plugin_from_file(
    filepath: str | Path,
    registry: PluginRegistry | None = None,
    config: dict | None = None,
    trusted_dir: str | Path | None = None,
    validate_path: bool = True,
) -> Plugin | None:
    """Load a plugin from a Python file.

    The file must define a class that inherits from Plugin.

    Security: By default, paths are validated to prevent loading arbitrary files.
    The file must be within the trusted_dir (defaults to configured plugins_dir).

    Args:
        filepath: Path to Python file (relative to trusted_dir, or absolute)
        registry: Registry to register with (default: global)
        config: Plugin configuration
        trusted_dir: Base directory for path validation (default: settings.plugins_dir)
        validate_path: If True, validate path is within trusted_dir (default: True)

    Returns:
        Loaded Plugin instance or None on failure
    """
    filepath = Path(filepath)

    # Validate path if enabled
    if validate_path:
        base_dir = Path(trusted_dir) if trusted_dir else _get_plugins_dir()
        try:
            filepath = validate_safe_path(
                filepath,
                base_dir,
                must_exist=True,
                allow_absolute=True,
            )
        except ValidationError as e:
            logger.error(f"Plugin path validation failed: {e}")
            return None
    elif not filepath.exists():
        logger.error(f"Plugin file not found: {filepath}")
        return None

    if not filepath.suffix == ".py":
        logger.error(f"Plugin must be a .py file: {filepath}")
        return None

    registry = registry or get_registry()
    module_name = f"gorgon_plugin_{filepath.stem}"

    try:
        # Load module from file
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        if not spec or not spec.loader:
            logger.error(f"Cannot load module spec from: {filepath}")
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Find Plugin subclass
        plugin_class: type[Plugin] | None = None
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and issubclass(obj, Plugin) and obj is not Plugin:
                plugin_class = obj
                break

        if not plugin_class:
            logger.error(f"No Plugin class found in: {filepath}")
            return None

        # Instantiate and register
        plugin = plugin_class()
        registry.register(plugin, config)

        logger.info(f"Loaded plugin from file: {filepath} -> {plugin.name}")
        return plugin

    except Exception as e:
        logger.error(f"Failed to load plugin from {filepath}: {e}")
        return None


def load_plugin_from_module(
    module_name: str,
    registry: PluginRegistry | None = None,
    config: dict | None = None,
) -> Plugin | None:
    """Load a plugin from an installed Python module.

    The module must define a class that inherits from Plugin.

    Args:
        module_name: Full module path (e.g., 'gorgon_plugins.my_plugin')
        registry: Registry to register with (default: global)
        config: Plugin configuration

    Returns:
        Loaded Plugin instance or None on failure
    """
    registry = registry or get_registry()

    try:
        module = importlib.import_module(module_name)

        # Find Plugin subclass
        plugin_class: type[Plugin] | None = None
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and issubclass(obj, Plugin) and obj is not Plugin:
                plugin_class = obj
                break

        if not plugin_class:
            logger.error(f"No Plugin class found in module: {module_name}")
            return None

        plugin = plugin_class()
        registry.register(plugin, config)

        logger.info(f"Loaded plugin from module: {module_name} -> {plugin.name}")
        return plugin

    except ImportError as e:
        logger.error(f"Failed to import module {module_name}: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to load plugin from {module_name}: {e}")
        return None


def discover_plugins(
    directory: str | Path,
    registry: PluginRegistry | None = None,
    config: dict | None = None,
    trusted_dir: str | Path | None = None,
    validate_path: bool = True,
) -> list[Plugin]:
    """Discover and load plugins from a directory.

    Scans for .py files and attempts to load plugins from each.

    Security: By default, the directory is validated to be within trusted_dir.

    Args:
        directory: Directory to scan (relative to trusted_dir, or absolute)
        registry: Registry to register with (default: global)
        config: Shared plugin configuration
        trusted_dir: Base directory for path validation (default: settings.plugins_dir)
        validate_path: If True, validate directory is within trusted_dir

    Returns:
        List of successfully loaded plugins
    """
    directory = Path(directory)

    # Validate directory if enabled
    if validate_path:
        base_dir = Path(trusted_dir) if trusted_dir else _get_plugins_dir()
        try:
            directory = validate_safe_path(
                directory,
                base_dir,
                must_exist=True,
                allow_absolute=True,
            )
        except ValidationError as e:
            logger.error(f"Plugin directory validation failed: {e}")
            return []
    elif not directory.exists():
        logger.warning(f"Plugin directory not found: {directory}")
        return []

    if not directory.is_dir():
        logger.error(f"Not a directory: {directory}")
        return []

    plugins = []
    for filepath in directory.glob("*.py"):
        # Skip private files
        if filepath.name.startswith("_"):
            continue

        # Files within validated directory are safe (already within trusted_dir)
        plugin = load_plugin_from_file(
            filepath,
            registry,
            config,
            trusted_dir=directory,
            validate_path=validate_path,
        )
        if plugin:
            plugins.append(plugin)

    logger.info(f"Discovered {len(plugins)} plugins in {directory}")
    return plugins


def load_plugins(
    sources: list[str | Path | dict],
    registry: PluginRegistry | None = None,
    trusted_dir: str | Path | None = None,
    validate_path: bool = True,
) -> list[Plugin]:
    """Load plugins from multiple sources.

    Security: By default, file paths are validated to be within trusted_dir.

    Args:
        sources: List of plugin sources:
            - str/Path: File path or module name
            - dict: {"path": "...", "config": {...}} or {"module": "...", "config": {...}}
        registry: Registry to register with (default: global)
        trusted_dir: Base directory for path validation (default: settings.plugins_dir)
        validate_path: If True, validate paths are within trusted_dir

    Returns:
        List of successfully loaded plugins
    """
    registry = registry or get_registry()
    plugins = []

    for source in sources:
        if isinstance(source, dict):
            config = source.get("config")
            # Allow per-source validation override
            source_validate = source.get("validate_path", validate_path)
            source_trusted = source.get("trusted_dir", trusted_dir)

            if "path" in source:
                plugin = load_plugin_from_file(
                    source["path"],
                    registry,
                    config,
                    trusted_dir=source_trusted,
                    validate_path=source_validate,
                )
            elif "module" in source:
                plugin = load_plugin_from_module(source["module"], registry, config)
            elif "directory" in source:
                plugins.extend(
                    discover_plugins(
                        source["directory"],
                        registry,
                        config,
                        trusted_dir=source_trusted,
                        validate_path=source_validate,
                    )
                )
                continue
            else:
                logger.warning(f"Invalid plugin source dict: {source}")
                continue
        elif isinstance(source, Path) or (isinstance(source, str) and "/" in source):
            path = Path(source)
            if path.is_dir():
                plugins.extend(
                    discover_plugins(
                        path,
                        registry,
                        trusted_dir=trusted_dir,
                        validate_path=validate_path,
                    )
                )
                continue
            else:
                plugin = load_plugin_from_file(
                    path, registry, trusted_dir=trusted_dir, validate_path=validate_path
                )
        else:
            # Assume module name (no path validation needed for module imports)
            plugin = load_plugin_from_module(str(source), registry)

        if plugin:
            plugins.append(plugin)

    return plugins
