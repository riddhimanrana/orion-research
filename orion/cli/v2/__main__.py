"""Allow running orion.cli.v2 as a module: python -m orion.cli.v2"""

from . import main
import sys

if __name__ == "__main__":
    sys.exit(main())
