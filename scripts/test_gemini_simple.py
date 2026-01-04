#!/usr/bin/env python3
"""Simple Gemini Vision test without Orion dependencies."""

import os
import sys
import json
import time
from pathlib import Path

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

import google.generativeai as genai
from PIL import Image

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("ERROR: GOOGLE_API_KEY not found")
    sys.exit(1)

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")

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
