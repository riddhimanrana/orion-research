#!/usr/bin/env python3
"""
Full Pipeline Evaluation with Gemini Validation

Runs complete Orion pipeline (detection + tracking + V-JEPA2 Re-ID + HDBSCAN clustering)
and validates EVERY sampled frame using Gemini 2.0 Flash.

Usage:
    python scripts/run_full_eval.py --video data/examples/test.mp4 --episode eval_test
    python scripts/run_full_eval.py --video data/examples/room.mp4 --episode eval_room --reid-backend vjepa2

Author: Orion Research Team
Date: January 2026
"""

import argparse
import base64
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

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


@dataclass
class FrameValidation:
    """Validation result for a single frame."""
    frame_id: int
    orion_objects: Dict[str, int]  # class_name -> count
    gemini_objects: Dict[str, int]
    precision: float  # What % of Orion detections are correct
    recall: float     # What % of Gemini objects were detected
    f1_score: float
    gemini_verdict: str
    raw_response: Dict[str, Any]


def run_phase1_detection(
    video_path: str,
    episode_id: str,
    fps: float = 5.0,
    device: str = "cuda",
    detector_backend: str = "yoloworld",
    yoloworld_open_vocab: bool = False,
) -> Tuple[Path, Dict]:
    """Run Phase 1: Detection + Tracking."""
    from orion.cli.run_tracks import process_video_to_tracks
    from orion.config import ensure_results_dir
    
    print("\n" + "="*80)
    print("PHASE 1: DETECTION + TRACKING")
    print("="*80)
    
    results_dir = ensure_results_dir(episode_id)
    
    metadata = process_video_to_tracks(
        video_path=video_path,
        episode_id=episode_id,
        target_fps=fps,
        detector_backend=detector_backend,
        device=device,
        enable_3d=False,  # Skip depth for speed
        yoloworld_open_vocab=yoloworld_open_vocab,
    )
    
    return results_dir, metadata


def run_phase2_reid(
    video_path: Path,
    results_dir: Path,
    reid_backend: str = "vjepa2",
    cosine_threshold: float = 0.70,
    max_crops_per_track: int = 5,
) -> Dict:
    """Run Phase 2: Re-ID with V-JEPA2 or DINO."""
    from orion.managers.model_manager import ModelManager
    from orion.perception.reid.matcher import build_memory_from_tracks
    
    print("\n" + "="*80)
    print(f"PHASE 2: RE-ID ({reid_backend.upper()})")
    print("="*80)
    
    # Configure Re-ID backend
    mm = ModelManager.get_instance()
    mm.reid_backend = reid_backend
    
    tracks_path = results_dir / "tracks.jsonl"
    
    build_memory_from_tracks(
        episode_id=results_dir.name,
        video_path=video_path,
        tracks_path=tracks_path,
        results_dir=results_dir,
        cosine_threshold=cosine_threshold,
        max_crops_per_track=max_crops_per_track,
        class_thresholds=None,
    )
    
    memory_path = results_dir / "memory.json"
    with open(memory_path) as f:
        memory = json.load(f)
    
    return memory


def load_tracks(tracks_path: Path) -> List[Dict]:
    """Load tracks from JSONL file."""
    tracks = []
    with open(tracks_path) as f:
        for line in f:
            if line.strip():
                tracks.append(json.loads(line))
    return tracks


def group_by_frame(tracks: List[Dict]) -> Dict[int, List[Dict]]:
    """Group tracks by frame_id."""
    by_frame = defaultdict(list)
    for t in tracks:
        frame_id = t.get("frame_id")
        if frame_id is not None:
            by_frame[frame_id].append(t)
    return dict(by_frame)


def extract_frame(video_path: Path, frame_id: int) -> Optional[np.ndarray]:
    """Extract a single frame from video."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def validate_frame_with_gemini(
    genai,
    frame: np.ndarray,
    frame_id: int,
    orion_detections: List[Dict],
    model_name: str = "gemini-2.5-flash",
) -> FrameValidation:
    """Validate a single frame's detections using Gemini."""
    import PIL.Image
    
    model = genai.GenerativeModel(model_name)
    
    # Summarize Orion detections for this frame
    orion_classes = [d.get("class_name", "unknown") for d in orion_detections]
    orion_counts = dict(Counter(orion_classes))
    
    # Convert frame to PIL Image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = PIL.Image.fromarray(rgb_frame)
    
    # Comprehensive prompt for object detection validation
    prompt = f"""Analyze this video frame and identify MAIN OBJECTS ONLY.

A computer vision system (Orion) detected these objects:
{json.dumps(orion_counts, indent=2)}

Focus on major furniture, people, and appliances. Ignore minor details like power cords or baseboards.

Respond with JSON in EXACTLY this format:
{{
    "objects_detected": {{"object_class": count, ...}},
    "orion_correct": ["list of correct detections"],
    "orion_incorrect": ["list of hallucinated objects"],
    "orion_missed": ["list of main objects Orion missed"],
    "precision_estimate": 0.0 to 1.0,
    "recall_estimate": 0.0 to 1.0,
    "overall_verdict": "EXCELLENT|GOOD|FAIR|POOR",
    "notes": "brief explanation"
}}

Main categories: furniture (chair, couch, table, desk, bed, shelf, cabinet), people, appliances (TV, microwave, fridge, oven), plants, doors, windows, lamps.
"""
    
    try:
        response = model.generate_content([prompt, pil_image])
        text = response.text.strip()
        
        # Clean markdown formatting
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[-1].strip() == "```":
                text = "\n".join(lines[1:-1])
            else:
                text = "\n".join(lines[1:])
        
        result = json.loads(text)
        
        gemini_counts = result.get("objects_detected", {})
        precision = result.get("precision_estimate", 0.0)
        recall = result.get("recall_estimate", 0.0)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        verdict = result.get("overall_verdict", "UNKNOWN")
        
        return FrameValidation(
            frame_id=frame_id,
            orion_objects=orion_counts,
            gemini_objects=gemini_counts,
            precision=precision,
            recall=recall,
            f1_score=f1,
            gemini_verdict=verdict,
            raw_response=result,
        )
        
    except Exception as e:
        print(f"    WARNING: Gemini validation failed for frame {frame_id}: {e}")
        return FrameValidation(
            frame_id=frame_id,
            orion_objects=orion_counts,
            gemini_objects={},
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            gemini_verdict="ERROR",
            raw_response={"error": str(e)},
        )


def run_gemini_validation(
    genai,
    video_path: Path,
    tracks: List[Dict],
    output_dir: Path,
    sleep_seconds: float = 0.5,  # API rate limiting
    gemini_model: str = "gemini-2.5-flash",
    max_frames: Optional[int] = None,
) -> List[FrameValidation]:
    """Validate all sampled frames with Gemini."""
    print("\n" + "="*80)
    print("GEMINI VALIDATION (ALL FRAMES)")
    print("="*80)
    
    by_frame = group_by_frame(tracks)
    frame_ids = sorted(by_frame.keys())
    
    if max_frames:
        frame_ids = frame_ids[:max_frames]
    
    print(f"Validating {len(frame_ids)} frames with {gemini_model}...")
    
    validations = []
    
    for i, frame_id in enumerate(frame_ids):
        print(f"  [{i+1}/{len(frame_ids)}] Frame {frame_id}...", end=" ", flush=True)
        
        frame = extract_frame(video_path, frame_id)
        if frame is None:
            print("SKIP (frame extraction failed)")
            continue
        
        detections = by_frame[frame_id]
        validation = validate_frame_with_gemini(genai, frame, frame_id, detections, model_name=gemini_model)
        validations.append(validation)
        
        print(f"{validation.gemini_verdict} (P={validation.precision:.2f}, R={validation.recall:.2f})")
        
        # Rate limiting
        if i < len(frame_ids) - 1:
            time.sleep(sleep_seconds)
    
    return validations


def compute_aggregate_metrics(validations: List[FrameValidation]) -> Dict:
    """Compute aggregate metrics across all frames."""
    if not validations:
        return {}
    
    valid_validations = [v for v in validations if v.gemini_verdict != "ERROR"]
    
    if not valid_validations:
        return {"error": "All validations failed"}
    
    avg_precision = np.mean([v.precision for v in valid_validations])
    avg_recall = np.mean([v.recall for v in valid_validations])
    avg_f1 = np.mean([v.f1_score for v in valid_validations])
    
    verdict_counts = Counter([v.gemini_verdict for v in valid_validations])
    
    return {
        "total_frames": len(validations),
        "successful_validations": len(valid_validations),
        "avg_precision": float(avg_precision),
        "avg_recall": float(avg_recall),
        "avg_f1": float(avg_f1),
        "verdict_distribution": dict(verdict_counts),
    }


def save_results(
    output_dir: Path,
    validations: List[FrameValidation],
    metrics: Dict,
    metadata: Dict,
):
    """Save validation results to JSON."""
    results = {
        "metadata": metadata,
        "aggregate_metrics": metrics,
        "frame_validations": [
            {
                "frame_id": v.frame_id,
                "orion_objects": v.orion_objects,
                "gemini_objects": v.gemini_objects,
                "precision": v.precision,
                "recall": v.recall,
                "f1_score": v.f1_score,
                "verdict": v.gemini_verdict,
                "details": v.raw_response,
            }
            for v in validations
        ],
    }
    
    output_path = output_dir / "gemini_full_validation.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    return output_path


def print_summary(metrics: Dict, memory_stats: Dict):
    """Print summary of results."""
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    print(f"\nDetection Quality (Gemini Validation):")
    print(f"  Avg Precision:  {metrics.get('avg_precision', 0):.1%}")
    print(f"  Avg Recall:     {metrics.get('avg_recall', 0):.1%}")
    print(f"  Avg F1 Score:   {metrics.get('avg_f1', 0):.1%}")
    
    print(f"\nVerdict Distribution:")
    for verdict, count in metrics.get("verdict_distribution", {}).items():
        print(f"  {verdict}: {count}")
    
    print(f"\nRe-ID Performance:")
    print(f"  Total Tracks:      {memory_stats.get('total_tracks', 0)}")
    print(f"  Memory Objects:    {memory_stats.get('memory_objects', 0)}")
    print(f"  Compression Ratio: {memory_stats.get('compression_ratio', 0):.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Full pipeline evaluation with Gemini validation")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--episode", required=True, help="Episode ID for results")
    parser.add_argument("--fps", type=float, default=5.0, help="Processing FPS")
    parser.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    parser.add_argument("--reid-backend", default="vjepa2", choices=["dino", "vjepa2"])
    parser.add_argument("--reid-threshold", type=float, default=0.70)
    parser.add_argument("--detector", default="yoloworld", choices=["yolo", "yoloworld", "groundingdino"])
    parser.add_argument("--skip-phase1", action="store_true", help="Skip detection (use existing tracks)")
    parser.add_argument("--skip-phase2", action="store_true", help="Skip Re-ID (use existing memory)")
    parser.add_argument("--gemini-sleep", type=float, default=0.5, help="Sleep between Gemini calls")
    parser.add_argument("--gemini-model", default="gemini-2.5-flash", help="Gemini model to use")
    parser.add_argument("--yoloworld-open-vocab", action="store_true", help="Use open vocabulary for YOLO-World")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames to validate")
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)
    
    # Setup Gemini
    genai = setup_gemini()
    
    # Phase 1: Detection + Tracking
    from orion.config import ensure_results_dir
    results_dir = ensure_results_dir(args.episode)
    
    if not args.skip_phase1:
        results_dir, phase1_meta = run_phase1_detection(
            video_path=str(video_path),
            episode_id=args.episode,
            fps=args.fps,
            device=args.device,
            detector_backend=args.detector,
            yoloworld_open_vocab=args.yoloworld_open_vocab,
        )
    else:
        print("\n[Phase 1] Skipped - using existing tracks")
        phase1_meta = {}
    
    # Phase 2: Re-ID
    if not args.skip_phase2:
        memory = run_phase2_reid(
            video_path=video_path,
            results_dir=results_dir,
            reid_backend=args.reid_backend,
            cosine_threshold=args.reid_threshold,
        )
    else:
        print("\n[Phase 2] Skipped - using existing memory")
        memory_path = results_dir / "memory.json"
        if memory_path.exists():
            with open(memory_path) as f:
                memory = json.load(f)
        else:
            memory = {"objects": []}
    
    # Load tracks for validation
    tracks_path = results_dir / "tracks.jsonl"
    tracks = load_tracks(tracks_path)
    
    # Memory stats
    memory_objects = len(memory.get("objects", []))
    unique_tracks = len(set(t.get("track_id") for t in tracks if t.get("track_id") is not None))
    
    memory_stats = {
        "total_tracks": unique_tracks,
        "memory_objects": memory_objects,
        "compression_ratio": unique_tracks / max(1, memory_objects),
    }
    
    print(f"\nRe-ID Results: {unique_tracks} tracks → {memory_objects} objects ({memory_stats['compression_ratio']:.2f}x compression)")
    
    # Phase 3: Gemini Validation
    validations = run_gemini_validation(
        genai=genai,
        video_path=video_path,
        tracks=tracks,
        output_dir=results_dir,
        sleep_seconds=args.gemini_sleep,
        gemini_model=args.gemini_model,
        max_frames=args.max_frames,
    )
    
    # Compute metrics
    metrics = compute_aggregate_metrics(validations)
    
    # Save results
    save_results(
        output_dir=results_dir,
        validations=validations,
        metrics=metrics,
        metadata={
            "video": str(video_path),
            "episode": args.episode,
            "fps": args.fps,
            "reid_backend": args.reid_backend,
            "reid_threshold": args.reid_threshold,
            "detector": args.detector,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    
    # Print summary
    print_summary(metrics, memory_stats)


if __name__ == "__main__":
    main()
