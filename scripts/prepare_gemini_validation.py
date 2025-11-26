#!/usr/bin/env python3
"""
Generate a Gemini validation prompt for scene graph relations.

Usage:
  python scripts/prepare_gemini_validation.py --samples results/video_validation/graph_samples
"""

import argparse
from pathlib import Path
import json


def generate_validation_prompt(sample_dir: Path) -> str:
    """Generate a prompt for Gemini to validate scene graph relations."""
    
    txt_files = sorted(sample_dir.glob("frame_*.txt"))
    
    prompt = """I need you to validate spatial relations detected in video frames. For each frame, I'll provide:
1. The detected objects with bounding boxes
2. The detected spatial relations between objects
3. Metrics used to compute the relations

Please review each relation and classify it as:
- CORRECT: The relation accurately describes the spatial relationship
- INCORRECT: The relation is wrong
- AMBIGUOUS: The relation could be interpreted either way

Also suggest any missing obvious relations.

---

"""
    
    for txt_file in txt_files:
        frame_num = txt_file.stem.replace("frame_", "")
        prompt += f"\n## Frame {frame_num}\n\n"
        prompt += f"[See attached image: {txt_file.stem}.jpg]\n\n"
        prompt += txt_file.read_text()
        prompt += "\n**Your validation:**\n\n"
        prompt += "---\n"
    
    return prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', type=str, required=True, help='Path to graph_samples directory')
    ap.add_argument('--output', type=str, default=None, help='Output file for prompt')
    args = ap.parse_args()
    
    sample_dir = Path(args.samples)
    if not sample_dir.exists():
        print(f"ERROR: {sample_dir} not found")
        return
    
    prompt = generate_validation_prompt(sample_dir)
    
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(prompt)
        print(f"Wrote validation prompt to {output_path}")
    else:
        print(prompt)
    
    print(f"\n{'='*70}")
    print("INSTRUCTIONS:")
    print("1. Copy the prompt above")
    print("2. Attach all JPG files from the samples directory")
    print("3. Send to Gemini for validation")
    print("4. Review feedback and adjust thresholds/constraints")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
