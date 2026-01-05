#!/usr/bin/env python3
"""Simple Gemini Vision test without Orion dependencies."""

import os
import sys
import json
import time
from pathlib import Path

# Ensure repo root is on sys.path so we can import Orion utilities when running as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

def load_dotenv():
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())

load_dotenv()

from orion.utils.gemini_client import GeminiClientError, get_gemini_model
from PIL import Image

try:
    model = get_gemini_model("gemini-2.0-flash")
except GeminiClientError as exc:
    print(f"ERROR: {exc}")
    sys.exit(1)

# Use existing sample frames
frames_dir = Path(__file__).parent.parent / "results" / "gemini_test" / "sample_frames"
frames = sorted(frames_dir.glob("*.jpg"))

if not frames:
    print("No sample frames found. Run: python scripts/test_gemini_comparison.py first")
    sys.exit(1)

print(f"Found {len(frames)} sample frames")
print()

for frame_path in frames[:3]:
    print(f"=== Analyzing {frame_path.name} ===")
    
    img = Image.open(frame_path)
    
    prompt = """Analyze this video frame and describe:
1. What objects are visible?
2. What is the person (if any) doing?
3. What is the environment/setting?

Be concise."""
    
    response = model.generate_content([prompt, img])
    print(response.text)
    print()
    time.sleep(1)

print("✓ Gemini Vision test complete!")
