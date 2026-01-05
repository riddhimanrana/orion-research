"""Command-line interface for the Orion research toolkit.

Keep package import side-effect free.

Some CLI dependencies (e.g. rich) are only needed when running the `orion`
console script. Importing submodules like `orion.cli.run_tracks` should not
require the full CLI stack.
"""


def main() -> int:
	# Lazy import to avoid importing rich (and other CLI deps) when users only
	# want to import a submodule.
	from .main import main as _main

	return _main()


__all__ = ["main"]
