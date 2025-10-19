"""Orion research toolkit package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __all__ = ["__version__"]
    __version__ = version("orion-research")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"
