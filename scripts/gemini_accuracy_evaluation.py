#!/usr/bin/env python3
"""
Gemini-based Accuracy Evaluation for Orion Detection Results

This script validates detection accuracy using Google's Gemini vision model.
It evaluates:
1. Detection correctness (are detected objects real?)
2. Label accuracy (are labels correct?)
3. Missed objects (what was not detected?)
4. Re-ID cluster quality (are merged tracks the same object?)

Usage:
    python scripts/gemini_accuracy_evaluation.py \
        --video data/examples/test.mp4 \
        --results results/lambda_hybrid_full \
        --model gemini-2.5-flash-preview-05-20 \
        --samples 15
"""

import argparse
import base64
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orion.utils.gemini_client import get_gemini_model, load_dotenv, GeminiClientError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("gemini_eval")


@dataclass
class DetectionValidation:
    """Validation result for a single detection."""
    label: str
    bbox: List[float]
    confidence: float
    verdict: str  # CORRECT, INCORRECT, WRONG_LABEL, PARTIAL
    correct_label: Optional[str] = None
    reason: str = ""


@dataclass
class FrameValidation:
    """Validation result for a single frame."""
    frame_idx: int
    timestamp: float
    scene_type: str
    total_detections: int
    correct: int = 0
    incorrect: int = 0
    wrong_label: int = 0
    partial: int = 0
    missed: int = 0
    hallucinated: int = 0
    detection_validations: List[DetectionValidation] = field(default_factory=list)
    missed_objects: List[Dict] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)


@dataclass
class ReIDValidation:
    """Validation result for Re-ID clustering."""
    memory_id: str
    class_name: str
    num_tracks: int
    verdict: str  # CORRECT_MERGE, INCORRECT_MERGE, SHOULD_MERGE
    reason: str = ""
    track_ids: List[int] = field(default_factory=list)


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    video_path: str
    results_dir: str
    model_used: str
    timestamp: str
    total_frames_sampled: int = 0
    total_detections: int = 0
    
    # Detection metrics
    detection_precision: float = 0.0
    detection_recall: float = 0.0
    detection_f1: float = 0.0
    label_accuracy: float = 0.0
    
    # Per-class accuracy
    class_accuracy: Dict[str, Dict] = field(default_factory=dict)
    
    # Re-ID metrics
    reid_precision: float = 0.0
    reid_samples_validated: int = 0
    
    # Issues
    common_false_positives: List[str] = field(default_factory=list)
    common_missed_objects: List[str] = field(default_factory=list)
    common_label_errors: List[str] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    # Raw validations
    frame_validations: List[Dict] = field(default_factory=list)
    reid_validations: List[Dict] = field(default_factory=list)


def load_tracks(tracks_path: Path) -> Tuple[Dict[int, List[Dict]], List[Dict]]:
    """Load tracks from JSONL file.
    
    Returns:
        tracks_by_frame: Dict mapping frame_idx to list of detections
        all_tracks: All track records
    """
    tracks_by_frame: Dict[int, List[Dict]] = defaultdict(list)
    all_tracks: List[Dict] = []
    
    with open(tracks_path) as f:
        for line in f:
            if not line.strip():
                continue
            track = json.loads(line)
            # Handle both frame_idx and frame_id formats
            frame_idx = track.get("frame_idx") or track.get("frame_id", 0)
            tracks_by_frame[frame_idx].append(track)
            all_tracks.append(track)
    
    return dict(tracks_by_frame), all_tracks


def load_memory(memory_path: Path) -> Optional[Dict]:
    """Load memory.json if it exists."""
    if not memory_path.exists():
        return None
    with open(memory_path) as f:
        return json.load(f)


def extract_frame(video_path: Path, frame_idx: int) -> Optional[np.ndarray]:
    """Extract a single frame from video."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def draw_detections_on_frame(frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Draw bounding boxes and labels on frame."""
    annotated = frame.copy()
    
    for i, det in enumerate(detections):
        bbox = det.get("bbox", [])
        if len(bbox) < 4:
            continue
            
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        label = det.get("class_name", det.get("category", det.get("class", "object")))
        conf = det.get("confidence", 0.0)
        track_id = det.get("track_id", i)
        
        # Color based on confidence
        color = (0, int(255 * conf), int(255 * (1 - conf)))
        
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text = f"[{track_id}] {label}: {conf:.2f}"
        
        # Background for text
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(annotated, text, (x1, y1 - 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return annotated


def frame_to_base64(frame: np.ndarray, quality: int = 85) -> str:
    """Convert frame to base64 JPEG."""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode('utf-8')


def validate_frame_with_gemini(
    model,
    frame: np.ndarray,
    detections: List[Dict],
    frame_idx: int,
    fps: float
) -> Optional[FrameValidation]:
    """Validate a single frame's detections with Gemini."""
    
    # Build detection list for prompt
    det_descriptions = []
    for det in detections:
        label = det.get("class_name", det.get("category", det.get("class", "object")))
        conf = det.get("confidence", 0.0)
        bbox = det.get("bbox", [])
        track_id = det.get("track_id", "?")
        det_descriptions.append(f"[{track_id}] {label} (conf={conf:.2f}, bbox={[int(b) for b in bbox[:4]]})")
    
    det_text = "\n".join(det_descriptions) if det_descriptions else "No detections"
    
    prompt = f"""You are an expert object detection evaluator. Analyze this video frame and validate each detection.

The frame has bounding boxes drawn with format: [track_id] label: confidence

DETECTED OBJECTS IN THIS FRAME:
{det_text}

Respond with a JSON object (no markdown, just raw JSON):
{{
    "scene_type": "brief description (e.g., home office, kitchen, living room)",
    "detection_validations": [
        {{
            "track_id": "the track ID shown",
            "detected_label": "what the system detected",
            "verdict": "CORRECT | INCORRECT | WRONG_LABEL | PARTIAL",
            "correct_label": "what it actually is (if wrong)",
            "reason": "brief explanation"
        }}
    ],
    "missed_objects": [
        {{
            "object": "clearly visible object that should have been detected",
            "location": "where in the frame (e.g., center, bottom-left)"
        }}
    ],
    "hallucinations": [
        {{
            "track_id": "track ID of false detection",
            "reason": "why this detection is wrong"
        }}
    ],
    "quality_notes": "any issues with detection quality"
}}

EVALUATION CRITERIA:
- CORRECT: Object exists AND label is semantically correct
- WRONG_LABEL: Object exists but label is significantly wrong (e.g., "book" for "laptop")
- PARTIAL: Object partially visible/occluded, label roughly correct
- INCORRECT: No such object exists at that location (hallucination)

Be strict but fair. Small variations in labels (e.g., "monitor" vs "display") are CORRECT.
Only report clearly visible objects as missed - don't count partially occluded objects."""

    try:
        # Prepare image
        annotated = draw_detections_on_frame(frame, detections)
        img_b64 = frame_to_base64(annotated)
        
        response = model.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": base64.b64decode(img_b64)}
        ])
        
        # Parse response
        text = response.text or "{}"
        # Clean markdown
        if "```" in text:
            lines = text.split("\n")
            text = "\n".join(l for l in lines if not l.strip().startswith("```"))
        
        result = json.loads(text.strip())
        
        # Build validation object
        validation = FrameValidation(
            frame_idx=frame_idx,
            timestamp=frame_idx / fps,
            scene_type=result.get("scene_type", "unknown"),
            total_detections=len(detections)
        )
        
        # Process detection validations
        for dv in result.get("detection_validations", []):
            verdict = dv.get("verdict", "").upper()
            detected_label = dv.get("detected_label", "")
            
            if verdict == "CORRECT":
                validation.correct += 1
            elif verdict == "WRONG_LABEL":
                validation.wrong_label += 1
                validation.issues.append(f"WRONG_LABEL: {detected_label} → {dv.get('correct_label', '?')}")
            elif verdict == "PARTIAL":
                validation.partial += 1
            elif verdict == "INCORRECT":
                validation.incorrect += 1
                validation.issues.append(f"FALSE_POSITIVE: {detected_label} - {dv.get('reason', '')}")
            
            validation.detection_validations.append(DetectionValidation(
                label=detected_label,
                bbox=dv.get("bbox", []),
                confidence=dv.get("confidence", 0.0),
                verdict=verdict,
                correct_label=dv.get("correct_label"),
                reason=dv.get("reason", "")
            ))
        
        # Process missed objects
        validation.missed = len(result.get("missed_objects", []))
        for mo in result.get("missed_objects", []):
            validation.missed_objects.append(mo)
            validation.issues.append(f"MISSED: {mo.get('object', '?')} at {mo.get('location', '?')}")
        
        # Process hallucinations
        validation.hallucinated = len(result.get("hallucinations", []))
        
        return validation
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Gemini response for frame {frame_idx}: {e}")
        return None
    except Exception as e:
        logger.error(f"Gemini validation failed for frame {frame_idx}: {e}")
        return None


def validate_reid_cluster_with_gemini(
    model,
    video_path: Path,
    memory_obj: Dict,
    tracks_by_frame: Dict[int, List[Dict]],
    all_tracks: List[Dict]
) -> Optional[ReIDValidation]:
    """Validate a Re-ID cluster by showing Gemini multiple crops from merged tracks."""
    
    memory_id = memory_obj.get("memory_id", "?")
    class_name = memory_obj.get("class", "unknown")
    appearance_history = memory_obj.get("appearance_history", [])
    
    if len(appearance_history) < 2:
        # Single track, nothing to validate about merging
        return None
    
    # Get sample frames from each merged track
    track_samples = []
    cap = cv2.VideoCapture(str(video_path))
    
    for appearance in appearance_history[:4]:  # Max 4 tracks to compare
        track_id = appearance.get("track_id")
        first_frame = appearance.get("first_frame", 0)
        last_frame = appearance.get("last_frame", first_frame)
        mid_frame = (first_frame + last_frame) // 2
        
        # Find the detection for this track at mid_frame
        if mid_frame in tracks_by_frame:
            for det in tracks_by_frame[mid_frame]:
                if det.get("track_id") == track_id:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
                    ret, frame = cap.read()
                    if ret:
                        bbox = det.get("bbox", [])
                        if len(bbox) >= 4:
                            x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
                            # Crop with padding
                            h, w = frame.shape[:2]
                            pad = 20
                            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                            x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
                            crop = frame[y1:y2, x1:x2]
                            if crop.size > 0:
                                track_samples.append({
                                    "track_id": track_id,
                                    "frame_idx": mid_frame,
                                    "crop": crop
                                })
                    break
    
    cap.release()
    
    if len(track_samples) < 2:
        return None
    
    # Create composite image
    crops = [s["crop"] for s in track_samples]
    max_h = max(c.shape[0] for c in crops)
    resized = []
    for c in crops:
        scale = max_h / c.shape[0]
        new_w = int(c.shape[1] * scale)
        resized.append(cv2.resize(c, (new_w, max_h)))
    
    # Add labels
    labeled = []
    for i, (r, s) in enumerate(zip(resized, track_samples)):
        cv2.putText(r, f"Track {s['track_id']}", (5, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(r, f"Frame {s['frame_idx']}", (5, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        labeled.append(r)
    
    composite = np.hstack(labeled)
    
    # Ask Gemini
    prompt = f"""You are evaluating Re-ID (Re-Identification) accuracy. 
    
The system has merged these {len(track_samples)} object crops into a single "memory object" because it thinks they're the same physical object appearing at different times.

Memory Object ID: {memory_id}
Assigned Class: {class_name}
Merged Track IDs: {[s['track_id'] for s in track_samples]}

Look at the cropped images side by side. Each shows the same physical location in the frame at different timestamps.

Respond with JSON (no markdown):
{{
    "verdict": "CORRECT_MERGE | INCORRECT_MERGE",
    "are_same_object": true/false,
    "reason": "explanation",
    "correct_class": "what class this should be (if different)"
}}

CORRECT_MERGE means: All crops show the SAME physical object (just at different times).
INCORRECT_MERGE means: The crops show DIFFERENT objects that were wrongly merged."""

    try:
        img_b64 = frame_to_base64(composite)
        
        response = model.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": base64.b64decode(img_b64)}
        ])
        
        text = response.text or "{}"
        if "```" in text:
            lines = text.split("\n")
            text = "\n".join(l for l in lines if not l.strip().startswith("```"))
        
        result = json.loads(text.strip())
        
        return ReIDValidation(
            memory_id=memory_id,
            class_name=class_name,
            num_tracks=len(track_samples),
            verdict=result.get("verdict", "UNKNOWN"),
            reason=result.get("reason", ""),
            track_ids=[s["track_id"] for s in track_samples]
        )
        
    except Exception as e:
        logger.error(f"Re-ID validation failed for {memory_id}: {e}")
        return None


def compute_metrics(frame_validations: List[FrameValidation]) -> Dict:
    """Compute aggregate metrics from frame validations."""
    
    total_correct = sum(v.correct for v in frame_validations)
    total_wrong_label = sum(v.wrong_label for v in frame_validations)
    total_partial = sum(v.partial for v in frame_validations)
    total_incorrect = sum(v.incorrect for v in frame_validations)
    total_missed = sum(v.missed for v in frame_validations)
    total_hallucinated = sum(v.hallucinated for v in frame_validations)
    
    total_detections = sum(v.total_detections for v in frame_validations)
    
    # Precision: (correct + partial*0.5) / (all detections)
    precision = (total_correct + total_partial * 0.5) / total_detections if total_detections > 0 else 0.0
    
    # Recall: (correct + partial*0.5) / (correct + partial + missed)
    recall_denom = total_correct + total_partial + total_missed
    recall = (total_correct + total_partial * 0.5) / recall_denom if recall_denom > 0 else 0.0
    
    # F1
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Label accuracy (among existing objects)
    label_denom = total_correct + total_wrong_label + total_partial
    label_accuracy = (total_correct + total_partial * 0.5) / label_denom if label_denom > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "label_accuracy": label_accuracy,
        "total_correct": total_correct,
        "total_wrong_label": total_wrong_label,
        "total_partial": total_partial,
        "total_incorrect": total_incorrect,
        "total_missed": total_missed,
        "total_hallucinated": total_hallucinated
    }


def analyze_class_accuracy(frame_validations: List[FrameValidation]) -> Dict[str, Dict]:
    """Compute per-class accuracy."""
    
    class_stats = defaultdict(lambda: {"correct": 0, "wrong": 0, "total": 0})
    
    for fv in frame_validations:
        for dv in fv.detection_validations:
            label = dv.label.lower()
            class_stats[label]["total"] += 1
            if dv.verdict == "CORRECT":
                class_stats[label]["correct"] += 1
            elif dv.verdict in ("INCORRECT", "WRONG_LABEL"):
                class_stats[label]["wrong"] += 1
    
    result = {}
    for cls, stats in class_stats.items():
        if stats["total"] >= 2:  # Only report classes with enough samples
            result[cls] = {
                "accuracy": stats["correct"] / stats["total"],
                "correct": stats["correct"],
                "wrong": stats["wrong"],
                "total": stats["total"]
            }
    
    return dict(sorted(result.items(), key=lambda x: -x[1]["total"]))


def find_common_issues(frame_validations: List[FrameValidation]) -> Tuple[List[str], List[str], List[str]]:
    """Find common false positives, missed objects, and label errors."""
    
    fp_counter = Counter()
    missed_counter = Counter()
    label_errors = Counter()
    
    for fv in frame_validations:
        for issue in fv.issues:
            if issue.startswith("FALSE_POSITIVE:"):
                # Extract label
                label = issue.split(":")[1].split("-")[0].strip()
                fp_counter[label] += 1
            elif issue.startswith("MISSED:"):
                obj = issue.split(":")[1].split("at")[0].strip()
                missed_counter[obj] += 1
            elif issue.startswith("WRONG_LABEL:"):
                error = issue.replace("WRONG_LABEL:", "").strip()
                label_errors[error] += 1
    
    return (
        [f"{k} ({v}x)" for k, v in fp_counter.most_common(10)],
        [f"{k} ({v}x)" for k, v in missed_counter.most_common(10)],
        [f"{k} ({v}x)" for k, v in label_errors.most_common(10)]
    )


def run_evaluation(
    video_path: Path,
    results_dir: Path,
    model_name: str = "gemini-2.5-flash-preview-05-20",
    num_samples: int = 15,
    validate_reid: bool = True
) -> EvaluationReport:
    """Run full Gemini-based evaluation."""
    
    # Initialize
    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise GeminiClientError("GOOGLE_API_KEY not set")
    
    model = get_gemini_model(model_name, api_key=api_key)
    logger.info(f"Using Gemini model: {model_name}")
    
    # Load data
    tracks_path = results_dir / "tracks.jsonl"
    memory_path = results_dir / "memory.json"
    
    if not tracks_path.exists():
        raise FileNotFoundError(f"tracks.jsonl not found in {results_dir}")
    
    tracks_by_frame, all_tracks = load_tracks(tracks_path)
    memory = load_memory(memory_path)
    
    logger.info(f"Loaded {len(all_tracks)} track observations across {len(tracks_by_frame)} frames")
    
    # Get video info
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Select frames to sample (evenly distributed, with detections)
    frame_indices = sorted(tracks_by_frame.keys())
    step = max(1, len(frame_indices) // num_samples)
    selected_frames = frame_indices[::step][:num_samples]
    
    logger.info(f"Sampling {len(selected_frames)} frames for validation")
    
    # Validate detection frames
    frame_validations: List[FrameValidation] = []
    
    for i, frame_idx in enumerate(selected_frames):
        logger.info(f"Validating frame {frame_idx} ({i+1}/{len(selected_frames)})...")
        
        frame = extract_frame(video_path, frame_idx)
        if frame is None:
            logger.warning(f"Could not extract frame {frame_idx}")
            continue
        
        detections = tracks_by_frame.get(frame_idx, [])
        validation = validate_frame_with_gemini(model, frame, detections, frame_idx, fps)
        
        if validation:
            frame_validations.append(validation)
            logger.info(f"  → Correct: {validation.correct}, Wrong: {validation.wrong_label + validation.incorrect}, Missed: {validation.missed}")
        
        time.sleep(0.5)  # Rate limiting
    
    # Validate Re-ID clusters
    reid_validations: List[ReIDValidation] = []
    
    if validate_reid and memory:
        objects = memory.get("objects", [])
        # Select objects with multiple merged tracks
        multi_track_objects = [o for o in objects if len(o.get("appearance_history", [])) >= 2]
        
        logger.info(f"Validating {min(10, len(multi_track_objects))} Re-ID clusters...")
        
        for obj in multi_track_objects[:10]:
            validation = validate_reid_cluster_with_gemini(
                model, video_path, obj, tracks_by_frame, all_tracks
            )
            if validation:
                reid_validations.append(validation)
                logger.info(f"  → {validation.memory_id}: {validation.verdict}")
            time.sleep(0.5)
    
    # Compute metrics
    metrics = compute_metrics(frame_validations)
    class_accuracy = analyze_class_accuracy(frame_validations)
    fps_list, missed_list, label_errors = find_common_issues(frame_validations)
    
    # Compute Re-ID precision
    reid_correct = sum(1 for v in reid_validations if v.verdict == "CORRECT_MERGE")
    reid_precision = reid_correct / len(reid_validations) if reid_validations else 0.0
    
    # Build report
    from datetime import datetime
    report = EvaluationReport(
        video_path=str(video_path),
        results_dir=str(results_dir),
        model_used=model_name,
        timestamp=datetime.now().isoformat(),
        total_frames_sampled=len(frame_validations),
        total_detections=sum(v.total_detections for v in frame_validations),
        detection_precision=metrics["precision"],
        detection_recall=metrics["recall"],
        detection_f1=metrics["f1"],
        label_accuracy=metrics["label_accuracy"],
        class_accuracy=class_accuracy,
        reid_precision=reid_precision,
        reid_samples_validated=len(reid_validations),
        common_false_positives=fps_list,
        common_missed_objects=missed_list,
        common_label_errors=label_errors,
        frame_validations=[asdict(v) for v in frame_validations],
        reid_validations=[asdict(v) for v in reid_validations]
    )
    
    # Generate recommendations
    if metrics["precision"] < 0.7:
        report.recommendations.append("Consider increasing detection confidence threshold to reduce false positives")
    if metrics["recall"] < 0.5:
        report.recommendations.append("Detection recall is low - consider using a more sensitive detector or lower threshold")
    if metrics["label_accuracy"] < 0.8:
        report.recommendations.append("Label accuracy needs improvement - consider fine-tuning or using better prompts")
    if reid_precision < 0.7 and reid_validations:
        report.recommendations.append("Re-ID clustering has issues - consider tightening similarity threshold")
    
    return report


def print_report(report: EvaluationReport):
    """Print a human-readable report."""
    
    print("\n" + "="*80)
    print("GEMINI-VALIDATED DETECTION ACCURACY REPORT")
    print("="*80)
    print(f"Video: {report.video_path}")
    print(f"Results: {report.results_dir}")
    print(f"Model: {report.model_used}")
    print(f"Frames sampled: {report.total_frames_sampled}")
    print(f"Total detections evaluated: {report.total_detections}")
    
    print("\n" + "-"*40)
    print("DETECTION METRICS")
    print("-"*40)
    print(f"Precision:      {report.detection_precision:.1%}")
    print(f"Recall:         {report.detection_recall:.1%}")
    print(f"F1 Score:       {report.detection_f1:.1%}")
    print(f"Label Accuracy: {report.label_accuracy:.1%}")
    
    if report.reid_samples_validated > 0:
        print("\n" + "-"*40)
        print("RE-ID METRICS")
        print("-"*40)
        print(f"Clusters validated: {report.reid_samples_validated}")
        print(f"Re-ID Precision:    {report.reid_precision:.1%}")
    
    print("\n" + "-"*40)
    print("PER-CLASS ACCURACY (top 15)")
    print("-"*40)
    for i, (cls, stats) in enumerate(list(report.class_accuracy.items())[:15]):
        print(f"  {cls:<30} {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})")
    
    if report.common_false_positives:
        print("\n" + "-"*40)
        print("COMMON FALSE POSITIVES")
        print("-"*40)
        for fp in report.common_false_positives:
            print(f"  • {fp}")
    
    if report.common_missed_objects:
        print("\n" + "-"*40)
        print("COMMONLY MISSED OBJECTS")
        print("-"*40)
        for mo in report.common_missed_objects:
            print(f"  • {mo}")
    
    if report.common_label_errors:
        print("\n" + "-"*40)
        print("COMMON LABEL ERRORS")
        print("-"*40)
        for le in report.common_label_errors:
            print(f"  • {le}")
    
    if report.recommendations:
        print("\n" + "-"*40)
        print("RECOMMENDATIONS")
        print("-"*40)
        for rec in report.recommendations:
            print(f"  → {rec}")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Gemini-based detection accuracy evaluation")
    parser.add_argument("--video", required=True, help="Path to source video")
    parser.add_argument("--results", required=True, help="Path to results directory with tracks.jsonl")
    parser.add_argument("--model", default="gemini-2.5-flash-preview-05-20", help="Gemini model to use")
    parser.add_argument("--samples", type=int, default=15, help="Number of frames to sample")
    parser.add_argument("--no-reid", action="store_true", help="Skip Re-ID validation")
    parser.add_argument("--output", help="Output JSON file for full report")
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    results_dir = Path(args.results)
    
    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        sys.exit(1)
    
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        sys.exit(1)
    
    report = run_evaluation(
        video_path=video_path,
        results_dir=results_dir,
        model_name=args.model,
        num_samples=args.samples,
        validate_reid=not args.no_reid
    )
    
    print_report(report)
    
    # Save full report
    output_path = Path(args.output) if args.output else results_dir / "gemini_evaluation.json"
    with open(output_path, "w") as f:
        json.dump(asdict(report), f, indent=2)
    logger.info(f"Full report saved to: {output_path}")


if __name__ == "__main__":
    main()
