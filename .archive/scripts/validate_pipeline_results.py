#!/usr/bin/env python3
"""
Validate pipeline results using Gemini.
Compares Fixed Vocab vs Prompt Vocab quality.
"""

import json
import argparse
import cv2
import base64
import os
from pathlib import Path
from collections import Counter

try:
    from orion.utils.gemini_client import GeminiClientError, get_gemini_model
except Exception:  # pragma: no cover
    GeminiClientError = RuntimeError  # type: ignore
    get_gemini_model = None  # type: ignore


def load_tracks(results_dir: Path) -> dict:
    """Load tracks from JSONL file."""
    tracks_file = results_dir / "tracks.jsonl"
    if not tracks_file.exists():
        return {}
    
    frame_detections = {}
    with open(tracks_file) as f:
        for line in f:
            entry = json.loads(line)
            frame_idx = entry.get("frame_idx", entry.get("frame_number", 0))
            observations = entry.get("observations", [])
            frame_detections[frame_idx] = observations
    
    return frame_detections


def get_detections_for_frame(frame_detections: dict, frame_idx: int) -> list:
    """Get detections for a specific frame."""
    # Find closest frame
    if frame_idx in frame_detections:
        return frame_detections[frame_idx]
    
    # Find closest
    frames = sorted(frame_detections.keys())
    for f in frames:
        if f >= frame_idx:
            return frame_detections[f]
    return frame_detections.get(frames[-1], []) if frames else []


def frame_to_base64(frame) -> str:
    """Convert frame to base64 for Gemini."""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')


def validate_with_gemini(frame, detections_fixed: list, detections_prompt: list, model) -> dict:
    """Ask Gemini to validate both detection sets."""
    
    # Extract class names
    classes_fixed = [d.get("label", d.get("class_name", "unknown")) for d in detections_fixed]
    classes_prompt = [d.get("label", d.get("class_name", "unknown")) for d in detections_prompt]
    
    fixed_summary = dict(Counter(classes_fixed))
    prompt_summary = dict(Counter(classes_prompt))
    
    prompt = f"""Analyze this image and compare two object detection results.

SYSTEM A (Fixed Vocab - 124 classes):
Detected objects: {json.dumps(fixed_summary, indent=2)}
Total: {len(classes_fixed)} detections

SYSTEM B (Prompt Vocab - 104 classes):
Detected objects: {json.dumps(prompt_summary, indent=2)}
Total: {len(classes_prompt)} detections

Please evaluate:
1. Which system captured the objects in the scene more accurately?
2. What objects are in the image that BOTH systems missed?
3. What false positives (phantom objects) did each system detect?
4. Which system would you choose for a home inventory application?

Provide a structured response with scores (1-10) for each system on:
- Precision (avoiding false positives)
- Recall (catching real objects)
- Overall quality

End with: WINNER: A, B, or TIE"""

    # Encode image
    img_b64 = frame_to_base64(frame)
    
    response = model.generate_content([
        {"mime_type": "image/jpeg", "data": img_b64},
        prompt
    ])
    
    return {
        "fixed_detections": len(classes_fixed),
        "prompt_detections": len(classes_prompt),
        "fixed_classes": fixed_summary,
        "prompt_classes": prompt_summary,
        "gemini_analysis": response.text
    }


def main():
    parser = argparse.ArgumentParser(description="Validate pipeline results with Gemini")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--fixed-results", default="results/pipeline_test_fixed", 
                        help="Fixed vocab results directory")
    parser.add_argument("--prompt-results", default="results/pipeline_test_prompt",
                        help="Prompt vocab results directory")
    parser.add_argument("--num-frames", type=int, default=5, help="Number of frames to validate")
    parser.add_argument("--output", default="results/vocab_validation.json", help="Output file")
    args = parser.parse_args()
    
    if get_gemini_model is None:
        print("ERROR: Orion Gemini adapter not available")
        return

    try:
        model = get_gemini_model("gemini-2.0-flash")
    except GeminiClientError as exc:
        print(f"ERROR: {exc}")
        return
    
    # Load video
    video = cv2.VideoCapture(args.video)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Load both result sets
    fixed_dir = Path(args.fixed_results)
    prompt_dir = Path(args.prompt_results)
    
    fixed_tracks = load_tracks(fixed_dir)
    prompt_tracks = load_tracks(prompt_dir)
    
    print(f"Loaded {len(fixed_tracks)} frames from Fixed, {len(prompt_tracks)} from Prompt")
    
    # Sample frames evenly
    sample_indices = [int(i * total_frames / (args.num_frames + 1)) for i in range(1, args.num_frames + 1)]
    
    results = []
    fixed_wins = 0
    prompt_wins = 0
    ties = 0
    
    for i, frame_idx in enumerate(sample_indices):
        print(f"\n[{i+1}/{args.num_frames}] Validating frame {frame_idx}...")
        
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = video.read()
        if not ret:
            continue
        
        fixed_dets = get_detections_for_frame(fixed_tracks, frame_idx)
        prompt_dets = get_detections_for_frame(prompt_tracks, frame_idx)
        
        print(f"  Fixed: {len(fixed_dets)} detections, Prompt: {len(prompt_dets)} detections")
        
        try:
            result = validate_with_gemini(frame, fixed_dets, prompt_dets, model)
            result["frame_idx"] = frame_idx
            results.append(result)
            
            # Extract winner
            analysis = result["gemini_analysis"]
            if "WINNER: A" in analysis or "WINNER: Fixed" in analysis:
                fixed_wins += 1
                print(f"  → Winner: Fixed Vocab")
            elif "WINNER: B" in analysis or "WINNER: Prompt" in analysis:
                prompt_wins += 1
                print(f"  → Winner: Prompt Vocab")
            else:
                ties += 1
                print(f"  → Tie")
                
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    video.release()
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Fixed Vocab wins: {fixed_wins}")
    print(f"Prompt Vocab wins: {prompt_wins}")
    print(f"Ties: {ties}")
    
    if fixed_wins > prompt_wins:
        print(f"\n🏆 OVERALL WINNER: Fixed Vocab (v5)")
    elif prompt_wins > fixed_wins:
        print(f"\n🏆 OVERALL WINNER: Prompt Vocab")
    else:
        print(f"\n🤝 OVERALL: TIE")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "fixed_wins": fixed_wins,
        "prompt_wins": prompt_wins,
        "ties": ties,
        "frames_validated": len(results),
        "frame_results": results
    }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Print last Gemini analysis
    if results:
        print("\n" + "-"*60)
        print("SAMPLE GEMINI ANALYSIS (last frame):")
        print("-"*60)
        print(results[-1]["gemini_analysis"])


if __name__ == "__main__":
    main()
