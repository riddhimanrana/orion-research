#!/usr/bin/env python3
"""
Phase 1 Validation Script

Validates Phase 1 detection results with Gemini Vision API.
Works with new orion detect output format (tracks.jsonl).

Usage:
    python scripts/validate_phase1.py --results results/phase1_test
"""

import argparse
import json
import os
import sys
from collections import Counter
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
    """Initialize Gemini API."""
    load_dotenv()
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found in environment or .env file")
        sys.exit(1)
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        return genai
    except ImportError:
        print("ERROR: google-generativeai not installed. Run: pip install google-generativeai")
        sys.exit(1)


def load_tracks(results_dir: Path):
    """Load tracks from tracks.jsonl."""
    tracks_path = results_dir / "tracks.jsonl"
    if not tracks_path.exists():
        print(f"ERROR: tracks.jsonl not found at {tracks_path}")
        sys.exit(1)
    
    tracks = []
    with open(tracks_path) as f:
        for line in f:
            tracks.append(json.loads(line))
    
    return tracks


def load_metadata(results_dir: Path):
    """Load episode metadata."""
    meta_path = results_dir / "episode_meta.json"
    if not meta_path.exists():
        print(f"ERROR: episode_meta.json not found at {meta_path}")
        sys.exit(1)
    
    with open(meta_path) as f:
        return json.load(f)


def analyze_tracks(tracks):
    """Analyze track statistics."""
    total_detections = len(tracks)
    unique_tracks = len(set(t['track_id'] for t in tracks))
    frames = sorted(set(t['frame_id'] for t in tracks))
    labels = [t['label'] for t in tracks]
    
    return {
        'total_detections': total_detections,
        'unique_tracks': unique_tracks,
        'frames': len(frames),
        'label_distribution': dict(Counter(labels)),
        'top_labels': Counter(labels).most_common(10)
    }


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


def get_frame_detections(tracks, frame_id, window=2):
    """Get detections for a specific frame (+/- window)."""
    frame_tracks = [
        t for t in tracks 
        if abs(t['frame_id'] - frame_id) <= window
    ]
    return frame_tracks


def validate_frame_with_gemini(genai, frame_path, frame_id, frame_detections, max_retries=3):
    """Validate frame detections using Gemini Vision."""
    import PIL.Image
    import time as _time
    
    # Use gemini-3-flash-preview for best quality
    model = genai.GenerativeModel("gemini-3-flash-preview")
    image = PIL.Image.open(frame_path)
    
    labels = [t['label'] for t in frame_detections]
    label_counts = dict(Counter(labels))
    
    # Build detailed bbox info for each detection
    detection_details = []
    for t in frame_detections:
        bbox = t.get('bbox', [])
        conf = t.get('confidence', 0)
        detection_details.append(f"  - {t['label']} (conf={conf:.2f}, bbox={[int(b) for b in bbox]})")
    detection_str = "\n".join(detection_details[:30])  # Limit to 30 for prompt length
    
    prompt = f"""You are an expert computer vision evaluator. Carefully analyze this video frame and provide a DETAILED assessment of object detection quality.

## YOLO-World Detections for this frame:
{detection_str}

## Detection Summary by Class:
{json.dumps(label_counts, indent=2)}

## Your Task:
1. List EVERY visible object in the frame (be exhaustive - include furniture, electronics, decor, architectural features, people/body parts, etc.)
2. For each YOLO detection, verify if it's correct (true positive) or incorrect (false positive)
3. Identify important objects YOLO completely missed (false negatives)
4. Pay special attention to:
   - Partially visible objects at frame edges
   - Small objects that may be missed
   - Objects occluded by other things
   - Lighting/reflection artifacts that may cause false positives
   - Duplicate detections of same object

## Response Format (JSON):
{{
    "scene_description": "Detailed description of what's happening in this frame",
    "room_type": "kitchen/bedroom/office/living_room/hallway/etc",
    "objects_visible": {{
        "furniture": ["list of visible furniture items"],
        "electronics": ["list of visible electronics"],
        "decor": ["list of decorative items"],
        "architecture": ["doors, windows, walls, etc"],
        "people_body_parts": ["any visible hands, faces, people"],
        "other": ["other objects"]
    }},
    "object_counts": {{"object_type": count}},
    "true_positives": ["YOLO detections that are CORRECT"],
    "false_positives": ["YOLO detections that are WRONG - object not actually there"],
    "false_negatives": ["Important objects YOLO MISSED completely"],
    "duplicate_detections": ["Cases where YOLO detected same object multiple times"],
    "precision": 0.0-1.0,
    "recall": 0.0-1.0,
    "accuracy_score": 0.0-1.0,
    "tracking_notes": "Any notes about objects that would be hard to track (moving, similar appearance, etc)",
    "comments": "Detailed assessment of detection quality and recommendations"
}}
"""
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content([prompt, image])
            text = response.text.strip()
            
            # Clean up markdown formatting if present
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
            
            result = json.loads(text)
            result["frame_id"] = frame_id
            result["yolo_detections"] = label_counts
            return result
        
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower():
                wait_time = (attempt + 1) * 10  # 10s, 20s, 30s
                print(f"  Rate limit hit, waiting {wait_time}s (attempt {attempt+1}/{max_retries})")
                _time.sleep(wait_time)
                continue
            print(f"  ERROR validating frame {frame_id}: {e}")
            return None
    
    print(f"  ERROR validating frame {frame_id}: Max retries exceeded")
    return None


def main():
    parser = argparse.ArgumentParser(description="Validate Phase 1 detection results with Gemini")
    parser.add_argument("--results", required=True, help="Results directory (e.g., results/phase1_test)")
    parser.add_argument("--frames", type=int, default=5, help="Number of frames to validate (default: 5)")
    parser.add_argument("--output", help="Output JSON file (default: <results>/gemini_validation.json)")
    args = parser.parse_args()
    
    results_dir = Path(args.results)
    
    # Load tracks and metadata
    print("Loading tracks...")
    tracks = load_tracks(results_dir)
    metadata = load_metadata(results_dir)
    
    # Analyze tracks
    print("\nAnalyzing tracks...")
    stats = analyze_tracks(tracks)
    print(f"  Total detections: {stats['total_detections']}")
    print(f"  Unique tracks: {stats['unique_tracks']}")
    print(f"  Frames processed: {stats['frames']}")
    print(f"\n  Top 10 detected classes:")
    for label, count in stats['top_labels']:
        print(f"    • {label}: {count}")
    
    # Setup Gemini
    print("\nInitializing Gemini API...")
    genai = setup_gemini()
    
    # Extract sample frames
    video_path = metadata['video_path']
    frames_dir = results_dir / "gemini_frames"
    print(f"\nExtracting {args.frames} sample frames from {video_path}...")
    sample_frames = extract_sample_frames(video_path, frames_dir, num_frames=args.frames)
    
    # Validate with Gemini
    print("\nValidating frames with Gemini Vision...")
    validations = []
    
    for frame_id, frame_path in sample_frames:
        print(f"\n  Frame {frame_id}...")
        frame_detections = get_frame_detections(tracks, frame_id)
        
        validation = validate_frame_with_gemini(genai, frame_path, frame_id, frame_detections)
        if validation:
            validations.append(validation)
            print(f"    Accuracy: {validation.get('accuracy_score', 'N/A')}")
            print(f"    Comments: {validation.get('comments', 'N/A')[:80]}...")
    
    # Aggregate results
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    avg_accuracy = sum(v.get('accuracy_score', 0) for v in validations) / len(validations)
    print(f"\n  Average Accuracy: {avg_accuracy:.2%}")
    
    all_false_positives = []
    all_false_negatives = []
    for v in validations:
        all_false_positives.extend(v.get('false_positives', []))
        all_false_negatives.extend(v.get('false_negatives', []))
    
    if all_false_positives:
        print(f"\n  Common False Positives:")
        for item, count in Counter(all_false_positives).most_common(5):
            print(f"    • {item}: {count}")
    
    if all_false_negatives:
        print(f"\n  Common False Negatives:")
        for item, count in Counter(all_false_negatives).most_common(5):
            print(f"    • {item}: {count}")
    
    # Save results
    output_path = Path(args.output) if args.output else results_dir / "gemini_validation.json"
    output_data = {
        'metadata': metadata,
        'track_stats': stats,
        'validations': validations,
        'summary': {
            'average_accuracy': avg_accuracy,
            'frames_validated': len(validations),
            'false_positives': dict(Counter(all_false_positives)),
            'false_negatives': dict(Counter(all_false_negatives))
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n  Results saved to: {output_path}")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
