"""
Lightweight shim to provide PeftAdapterMixin for transformers.integrations.peft import.
This avoids importing heavy optional dependencies (accelerate/peft) during tests.
"""

class PeftAdapterMixin:
    """Minimal shim class to satisfy imports that reference PeftAdapterMixin.
    This intentionally does not provide full PEFT functionality.
    """
    pass
