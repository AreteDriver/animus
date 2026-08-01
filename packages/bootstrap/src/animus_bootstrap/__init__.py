"""Animus Bootstrap — install daemon, onboarding wizard, and local dashboard."""

try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _version

    __version__ = _version("animus-bootstrap")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+dev"
