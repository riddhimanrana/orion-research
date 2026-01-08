#!/usr/bin/env python3
"""
Video-Based Gemini Evaluation

Sends the ENTIRE video to Gemini once and compares against Orion detections.
Much faster than frame-by-frame analysis (~1-2 minutes vs 1.5 hours).

Usage:
    python scripts/eval_video_gemini.py --video data/examples/video.mp4 --episode gemini_eval_008
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional

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
    """Initialize Gemini API."""
    load_dotenv()
    
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY or GEMINI_API_KEY not found")
        sys.exit(1)
    
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        return client
    except ImportError:
        print("ERROR: google-genai not installed. Install with: pip install google-genai")
        sys.exit(1)


def load_orion_tracks(episode_dir: Path) -> Dict[str, Any]:
    """Load Orion detection results."""
    tracks_path = episode_dir / "tracks.jsonl"
    
    if not tracks_path.exists():
        return {"tracks": [], "summary": {}}
    
    tracks = []
    with open(tracks_path) as f:
        for line in f:
            if line.strip():
                tracks.append(json.loads(line))
    
    # Summarize detections
    all_classes = [t.get("class_name") for t in tracks]
    class_counts = Counter(all_classes)
    
    # Group by frame ranges
    by_frame = defaultdict(list)
    for t in tracks:
        frame_id = t.get("frame_id", 0)
        by_frame[frame_id].append(t.get("class_name"))
    
    # Create time-based summary (every 10 seconds)
    fps = 30  # approximate
    time_summary = {}
    for frame_id, classes in sorted(by_frame.items()):
        time_sec = frame_id // fps
        time_bucket = (time_sec // 10) * 10  # 10-second buckets
        key = f"{time_bucket}s-{time_bucket+10}s"
        if key not in time_summary:
            time_summary[key] = Counter()
        time_summary[key].update(classes)
    
    return {
        "tracks": tracks,
        "total_detections": len(tracks),
        "unique_classes": dict(class_counts),
        "time_summary": {k: dict(v) for k, v in time_summary.items()},
        "frame_range": (min(by_frame.keys()), max(by_frame.keys())) if by_frame else (0, 0),
    }


def validate_with_gemini_video(
    client,
    video_path: Path,
    orion_summary: Dict[str, Any],
    model_name: str = "gemini-2.5-flash",
) -> Dict[str, Any]:
    """
    Send entire video to Gemini for validation against Orion detections.
    
    Returns structured evaluation with precision/recall estimates.
    """
    from google.genai import types
    
    print(f"\n{'='*60}")
    print("GEMINI VIDEO VALIDATION")
    print(f"{'='*60}")
    print(f"Uploading video: {video_path.name}")
    
    # Upload video file
    start_time = time.time()
    
    video_file = client.files.upload(file=video_path)
    print(f"✓ Video uploaded ({time.time() - start_time:.1f}s)")
    
    # Wait for processing
    print("Waiting for video processing...")
    while video_file.state.name == "PROCESSING":
        time.sleep(2)
        video_file = client.files.get(name=video_file.name)
    
    if video_file.state.name != "ACTIVE":
        print(f"ERROR: Video processing failed: {video_file.state.name}")
        return {"error": "Video processing failed"}
    
    print(f"✓ Video ready ({time.time() - start_time:.1f}s)")
    
    # Create comprehensive prompt
    prompt = f"""Analyze this video and evaluate the object detection results from our system (Orion).

## Orion Detection Summary:
Total detections: {orion_summary['total_detections']}
Unique classes detected: {json.dumps(orion_summary['unique_classes'], indent=2)}

Time-based breakdown:
{json.dumps(orion_summary['time_summary'], indent=2)}

## Your Task:
1. Watch the entire video carefully
2. Identify all MAIN objects that appear (furniture, people, appliances, etc.)
3. Compare against Orion's detections above

## Respond with JSON:
{{
    "video_duration_seconds": <number>,
    "scene_description": "<brief description of the video content>",
    "ground_truth_objects": {{
        "object_class": <count or "present">
    }},
    "orion_correct_detections": ["list of object classes Orion correctly detected"],
    "orion_false_positives": ["list of objects Orion detected that are NOT in the video"],
    "orion_missed_objects": ["list of main objects visible but NOT detected by Orion"],
    "precision_estimate": <0.0 to 1.0 - what fraction of Orion detections are correct>,
    "recall_estimate": <0.0 to 1.0 - what fraction of actual objects Orion detected>,
    "f1_estimate": <0.0 to 1.0>,
    "overall_verdict": "EXCELLENT|GOOD|FAIR|POOR",
    "key_issues": ["list of main problems with Orion's detection"],
    "recommendations": ["specific suggestions to improve detection"]
}}

Focus on MAIN objects - ignore small items, cables, baseboards, etc.
Be specific about false positives (e.g., "detected 'refrigerator' but it's actually a door").
"""
    
    print("Sending to Gemini for analysis...")
    
    response = client.models.generate_content(
        model=model_name,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(file_uri=video_file.uri, mime_type=video_file.mime_type),
                    types.Part.from_text(text=prompt),
                ],
            ),
        ],
    )
    
    elapsed = time.time() - start_time
    print(f"✓ Analysis complete ({elapsed:.1f}s)")
    
    # Parse response
    text = response.text.strip()
    
    # Clean markdown formatting
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1])
        else:
            text = "\n".join(lines[1:])
    
    try:
        result = json.loads(text)
        result["processing_time_seconds"] = elapsed
        return result
    except json.JSONDecodeError as e:
        print(f"WARNING: Failed to parse JSON response: {e}")
        return {
            "raw_response": text,
            "error": str(e),
            "processing_time_seconds": elapsed,
        }


def print_evaluation_report(result: Dict[str, Any], orion_summary: Dict[str, Any]):
    """Print formatted evaluation report."""
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    
    if "error" in result and "raw_response" not in result:
        print(f"ERROR: {result['error']}")
        return
    
    print(f"\nScene: {result.get('scene_description', 'N/A')}")
    print(f"Video Duration: {result.get('video_duration_seconds', 'N/A')}s")
    print(f"Processing Time: {result.get('processing_time_seconds', 0):.1f}s")
    
    print(f"\n--- Detection Quality ---")
    print(f"Precision: {result.get('precision_estimate', 0)*100:.1f}%")
    print(f"Recall: {result.get('recall_estimate', 0)*100:.1f}%")
    print(f"F1 Score: {result.get('f1_estimate', 0)*100:.1f}%")
    print(f"Verdict: {result.get('overall_verdict', 'N/A')}")
    
    print(f"\n--- Ground Truth Objects ---")
    for obj, count in result.get("ground_truth_objects", {}).items():
        print(f"  {obj}: {count}")
    
    print(f"\n--- Correct Detections ---")
    for obj in result.get("orion_correct_detections", []):
        print(f"  ✓ {obj}")
    
    print(f"\n--- False Positives (Hallucinations) ---")
    for obj in result.get("orion_false_positives", []):
        print(f"  ✗ {obj}")
    
    print(f"\n--- Missed Objects ---")
    for obj in result.get("orion_missed_objects", []):
        print(f"  ! {obj}")
    
    print(f"\n--- Key Issues ---")
    for issue in result.get("key_issues", []):
        print(f"  • {issue}")
    
    print(f"\n--- Recommendations ---")
    for rec in result.get("recommendations", []):
        print(f"  → {rec}")


def main():
    parser = argparse.ArgumentParser(description="Video-based Gemini evaluation")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--episode", required=True, help="Episode ID with Orion results")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model to use")
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)
    
    # Find episode results
    from orion.config import ensure_results_dir
    episode_dir = ensure_results_dir(args.episode)
    
    # Check if Orion has run on this episode
    tracks_path = episode_dir / "tracks.jsonl"
    if not tracks_path.exists():
        print(f"No Orion results found for episode '{args.episode}'")
        print("Running full Orion pipeline with semantic filtering...")
        
        # Use run_showcase which includes semantic filtering
        import subprocess
        cmd = [
            sys.executable, "-m", "orion.cli.run_showcase",
            "--episode", args.episode,
            "--video", str(video_path),
            "--skip-overlay",  # Skip overlay for speed
        ]
        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
        if result.returncode != 0:
            print(f"ERROR: Orion pipeline failed with code {result.returncode}")
            sys.exit(1)
    
    # Load Orion results
    print(f"\nLoading Orion results from: {episode_dir}")
    orion_summary = load_orion_tracks(episode_dir)
    print(f"  Total detections: {orion_summary['total_detections']}")
    print(f"  Unique classes: {len(orion_summary['unique_classes'])}")
    
    # Setup Gemini
    client = setup_gemini()
    
    # Run video validation
    result = validate_with_gemini_video(
        client,
        video_path,
        orion_summary,
        model_name=args.model,
    )
    
    # Print report
    print_evaluation_report(result, orion_summary)
    
    # Save results
    output_path = episode_dir / "gemini_video_validation.json"
    with open(output_path, "w") as f:
        json.dump({
            "orion_summary": orion_summary,
            "gemini_result": result,
        }, f, indent=2, default=str)
    print(f"\n✓ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
