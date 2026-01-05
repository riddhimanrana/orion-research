#!/usr/bin/env python3
"""
Gemini Comparison Test for Orion

Runs Orion perception on a video and validates/compares results using Gemini Vision API.

Usage:
    python scripts/test_gemini_comparison.py --video data/examples/test.mp4 --output results/gemini_test
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
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


def setup_gemini():
    """Initialize Gemini API (google-genai via Orion adapter)."""
    load_dotenv()

    model_name = os.environ.get("GEMINI_MODEL") or "gemini-3-flash-preview"
    try:
        from orion.utils.gemini_client import GeminiClientError, get_gemini_model
    except Exception:
        print("ERROR: Orion Gemini adapter not available")
        sys.exit(1)

    try:
        return get_gemini_model(model_name)
    except GeminiClientError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)


def extract_sample_frames(video_path: str, output_dir: Path, num_frames: int = 5):
    """Extract sample frames from video for Gemini analysis."""
    import cv2
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample frames evenly distributed
    frame_indices = [int(i * total_frames / (num_frames + 1)) for i in range(1, num_frames + 1)]
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_path = output_dir / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frames.append((idx, frame_path))
    
    cap.release()
    return frames


def run_orion_fast(video_path: str, output_dir: Path):
    """Run Orion perception in fast mode."""
    from orion.perception.config import get_fast_config
    from orion.perception.engine import PerceptionEngine
    
    print("\n=== Running Orion Perception (Fast Mode) ===")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = get_fast_config()
    engine = PerceptionEngine(config=config)
    
    start_time = time.time()
    result = engine.process_video(
        video_path,
        save_visualizations=True,
        output_dir=str(output_dir)
    )
    elapsed = time.time() - start_time
    
    print(f"✓ Orion processed {result.total_frames} frames in {elapsed:.1f}s ({result.total_frames/elapsed:.1f} fps)")
    print(f"  Unique entities: {result.unique_entities}")
    print(f"  Total detections: {result.total_detections}")
    
    return result

def load_orion_results(results_dir: Path):
    """Load existing Orion results."""
    tracks_path = results_dir / "tracks.jsonl"
    if not tracks_path.exists():
        print(f"WARNING: No tracks found at {tracks_path}")
        return None
        
    import json
    from collections import Counter
    
    tracks = []
    with open(tracks_path) as f:
        for line in f:
            tracks.append(json.loads(line))
            
    return tracks


def ask_gemini_about_frame(model, frame_path: Path, question: str) -> str:
    """Ask Gemini a question about a frame."""
    import PIL.Image

    image = PIL.Image.open(frame_path)
    
    response = model.generate_content([question, image])
    return response.text


def compare_with_gemini(model, orion_result, sample_frames, output_dir: Path):
    """Compare Orion results with Gemini Vision analysis."""
    import PIL.Image
    from collections import Counter
    
    print("\n=== Comparing with Gemini Vision ===")
    
    comparisons = []
    
    # Build global summary for context
    all_classes = []
    if isinstance(orion_result, list):
        # Loaded from file (list of track dicts)
        all_classes = [t.get('class_name', 'unknown') for t in orion_result]
    elif hasattr(orion_result, 'entities'):
        # Live result (PerceptionResult)
        all_classes = [
            e.object_class.value if hasattr(e.object_class, 'value') else str(e.object_class)
            for e in orion_result.entities
        ]
        
    global_summary = f"Orion detected (total): {dict(Counter(all_classes))}"
    
    for frame_idx, frame_path in sample_frames:
        print(f"\n  Analyzing frame {frame_idx}...")
        image = PIL.Image.open(frame_path)
        
        # Get frame-specific detections
        frame_detections = []
        if isinstance(orion_result, list):
            # Filter tracks for this frame (+/- 2 frames window)
            frame_detections = [
                t.get('class_name', 'unknown') 
                for t in orion_result 
                if abs(t.get('frame_id', -1) - frame_idx) <= 2
            ]
        
        frame_summary = f"Orion detected in this frame: {dict(Counter(frame_detections))}"
        
        # Ask Gemini what it sees
        prompt = f"""Analyze this video frame and list all visible objects.
        
For context, a perception system detected these objects in this specific frame: {frame_summary}

Please respond with JSON in this exact format:
{{
    "objects_visible": ["list", "of", "visible", "objects"],
    "object_counts": {{"object_type": count}},
    "scene_description": "brief description",
    "comparison_with_orion": "comment on whether the Orion detections seem accurate for this frame"
}}
"""
        
        try:
            response = model.generate_content([prompt, image])
            text = response.text.strip()
            
            # Clean up markdown formatting if present
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
            
            result = json.loads(text)
            result["frame_idx"] = frame_idx
            result["orion_context"] = frame_summary
            comparisons.append(result)
            
            print(f"    Gemini found: {result.get('objects_visible', [])}")
            print(f"    Comparison: {result.get('comparison_with_orion', 'N/A')[:100]}...")
            
        except Exception as e:
            print(f"    Error analyzing frame: {e}")
            comparisons.append({
                "frame_idx": frame_idx,
                "error": str(e)
            })
        
        time.sleep(1)  # Rate limiting
    
    # Save comparison results
    comparison_file = output_dir / "gemini_comparison.json"
    with open(comparison_file, "w") as f:
        json.dump({
            "orion_summary": global_summary,
            "entity_counts": dict(Counter(all_classes)),
            "frame_comparisons": comparisons,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    
    print(f"\n✓ Comparison saved to: {comparison_file}")
    return comparisons


def ask_questions_with_gemini(genai, sample_frames):
    """Interactive Q&A about frames using Gemini."""
    print("\n=== Gemini Q&A Demo ===")
    
    questions = [
        "What is the person in this image doing?",
        "Describe the environment/setting in detail.",
        "What objects could be tracked for scene understanding?",
    ]
    
    frame_idx, frame_path = sample_frames[len(sample_frames) // 2]  # Middle frame
    print(f"\nAnalyzing frame {frame_idx}:")
    
    for q in questions:
        print(f"\n  Q: {q}")
        answer = ask_gemini_about_frame(model, frame_path, q)
        print(f"  A: {answer[:200]}..." if len(answer) > 200 else f"  A: {answer}")
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description="Test Orion + Gemini comparison")
    parser.add_argument("--video", type=str, default="data/examples/test.mp4", 
                        help="Path to video file")
    parser.add_argument("--output", type=str, default="results/gemini_test",
                        help="Output directory")
    parser.add_argument("--num-frames", type=int, default=5,
                        help="Number of sample frames for Gemini analysis")
    parser.add_argument("--skip-orion", action="store_true",
                        help="Skip Orion processing (use existing results)")
    parser.add_argument("--qa", action="store_true",
                        help="Run interactive Q&A demo")
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    output_dir = Path(args.output)
    
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)
    
    # Initialize Gemini
    model = setup_gemini()
    print("✓ Gemini API initialized")
    
    # Extract sample frames
    frames_dir = output_dir / "sample_frames"
    sample_frames = extract_sample_frames(str(video_path), frames_dir, args.num_frames)
    print(f"✓ Extracted {len(sample_frames)} sample frames")
    
    # Run Orion perception
    orion_result = None
    if not args.skip_orion:
        orion_result = run_orion_fast(str(video_path), output_dir)
    else:
        orion_result = load_orion_results(output_dir)
    
    # Compare with Gemini
    if orion_result:
        compare_with_gemini(model, orion_result, sample_frames, output_dir)
    
    # Interactive Q&A demo
    if args.qa:
        ask_questions_with_gemini(genai, sample_frames)
    
    print("\n" + "="*50)
    print("✓ Gemini comparison test complete!")
    print(f"  Results: {output_dir}")


if __name__ == "__main__":
    main()
