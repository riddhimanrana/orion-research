"""Command-line interface for the Orion research toolkit."""

# Re-export main for backward compatibility
try:
    from .main import main
except ImportError:
    # Fallback during initial setup
    main = None

__all__ = ["main"]
