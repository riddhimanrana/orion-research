#!/usr/bin/env python3
"""Quick test script for FastVLM model loading"""

import sys
from pathlib import Path

# Add production to path
sys.path.insert(0, str(Path(__file__).parent / "production"))

from fastvlm_wrapper import load_fastvlm

print("="*80)
print("Testing FastVLM Model Loading")
print("="*80)
print()

try:
    print("Loading model (this may take 30-60 seconds)...")
    model = load_fastvlm()
    print("\n✓ Model loaded successfully!")
    print(f"  Device: {model.device}")
    print(f"  Conv mode: {model.conv_mode}")
    print(f"  Context length: {model.context_len}")
    print("\n" + "="*80)
    print("SUCCESS - FastVLM is ready to use!")
    print("="*80)
    sys.exit(0)
except Exception as e:
    print(f"\n✗ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
