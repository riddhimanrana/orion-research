#!/usr/bin/env python3
"""Run `scripts/test_pipeline_stages.py` with a lightweight Peft shim preloaded.

This injects a minimal `transformers.integrations.peft` module into sys.modules
so that `transformers` can import `PeftAdapterMixin` without importing heavy
optional dependencies like `accelerate` during test runs.
"""
import sys
from types import ModuleType

# Create a minimal shim module
m = ModuleType("transformers.integrations.peft")
class PeftAdapterMixin:
    pass
m.PeftAdapterMixin = PeftAdapterMixin

# Inject into sys.modules before any heavy imports
sys.modules["transformers.integrations.peft"] = m

# Now run the test script as __main__
import runpy
runpy.run_path("/Users/yogeshatluru/orion-research/scripts/test_pipeline_stages.py", run_name="__main__")
