#!/usr/bin/env python3
"""
Full Evaluation with Gemini API Validation

This script:
1. Runs the perception pipeline on test videos
2. Extracts sample frames with detections
3. Uses Gemini API to validate detection accuracy
4. Generates a detailed report with improvement suggestions

Usage:
    python scripts/full_gemini_evaluation.py --videos data/examples/test.mp4 data/examples/video.mp4
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orion.perception.engine import PerceptionEngine
from orion.perception.config import PerceptionConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("gemini_eval")


@dataclass
class DetectionSample:
    """A frame sample for Gemini validation."""
    frame_idx: int
    frame_path: Path
    detections: List[Dict[str, Any]]
    timestamp: float


@dataclass
class GeminiValidation:
    """Result of Gemini validation for a sample."""
    frame_idx: int
    correct: int = 0
    incorrect: int = 0
    missed: int = 0
    hallucinated: int = 0
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class EvaluationResult:
    """Full evaluation result for a video."""
    video_path: str
    total_frames: int
    sampled_frames: int
    total_detections: int
    unique_entities: int
    processing_time: float
    validations: List[GeminiValidation] = field(default_factory=list)
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    common_issues: List[str] = field(default_factory=list)


def extract_sample_frames(
    video_path: Path,
    tracks_path: Path,
    output_dir: Path,
    num_samples: int = 10,
    min_detections: int = 1
) -> List[DetectionSample]:
    """Extract frames with detections for Gemini validation."""
    
    # Load tracks
    tracks_by_frame: Dict[int, List[Dict]] = {}
    with open(tracks_path) as f:
        for line in f:
            if not line.strip():
                continue
            track = json.loads(line)
            frame_idx = track.get("frame_idx", 0)
            if frame_idx not in tracks_by_frame:
                tracks_by_frame[frame_idx] = []
            tracks_by_frame[frame_idx].append(track)
    
    # Find frames with enough detections
    valid_frames = [
        (idx, dets) for idx, dets in tracks_by_frame.items()
        if len(dets) >= min_detections
    ]
    
    if not valid_frames:
        logger.warning("No frames with detections found")
        return []
    
    # Sample evenly across video
    valid_frames.sort(key=lambda x: x[0])
    step = max(1, len(valid_frames) // num_samples)
    selected = valid_frames[::step][:num_samples]
    
    # Extract frames
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    
    for frame_idx, detections in selected:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Draw bounding boxes on frame
        annotated = frame.copy()
        for det in detections:
            bbox = det.get("bbox", [])
            if len(bbox) >= 4:
                x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
                label = det.get("category", det.get("class", "object"))
                conf = det.get("confidence", 0.0)
                
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{label}: {conf:.2f}"
                cv2.putText(annotated, text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save frame
        frame_path = output_dir / f"frame_{frame_idx:05d}.jpg"
        cv2.imwrite(str(frame_path), annotated)
        
        # Save detection info
        info_path = output_dir / f"frame_{frame_idx:05d}.json"
        with open(info_path, "w") as f:
            json.dump({
                "frame_idx": frame_idx,
                "timestamp": frame_idx / fps,
                "detections": detections
            }, f, indent=2)
        
        samples.append(DetectionSample(
            frame_idx=frame_idx,
            frame_path=frame_path,
            detections=detections,
            timestamp=frame_idx / fps
        ))
    
    cap.release()
    logger.info(f"Extracted {len(samples)} sample frames to {output_dir}")
    return samples


def validate_with_gemini(
    samples: List[DetectionSample],
    model_name: str = "gemini-2.0-flash",
    api_key: Optional[str] = None
) -> List[GeminiValidation]:
    """Use Gemini to validate detection accuracy."""
    
    try:
        from orion.utils.gemini_client import get_gemini_model, load_dotenv
        load_dotenv()
        api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.error("GOOGLE_API_KEY not set, skipping Gemini validation")
            return []
        
        model = get_gemini_model(model_name, api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize Gemini: {e}")
        return []
    
    validations = []
    
    for sample in samples:
        logger.info(f"Validating frame {sample.frame_idx}...")
        
        # Build prompt
        det_list = []
        for det in sample.detections:
            label = det.get("category", det.get("class", "object"))
            conf = det.get("confidence", 0.0)
            bbox = det.get("bbox", [])
            det_list.append(f"- {label} (confidence: {conf:.2f}, bbox: {bbox})")
        
        det_text = "\n".join(det_list) if det_list else "No detections"
        
        prompt = f"""Analyze this video frame and validate the object detections.

DETECTED OBJECTS:
{det_text}

Please evaluate each detection and respond with a JSON object:
{{
    "frame_analysis": {{
        "scene_type": "string describing the scene (kitchen, office, living room, etc.)",
        "lighting": "good/poor/mixed",
        "motion_blur": true/false
    }},
    "detection_validation": [
        {{
            "label": "detected label",
            "verdict": "CORRECT" | "INCORRECT" | "WRONG_LABEL" | "PARTIAL",
            "correct_label": "what it should be (if incorrect)",
            "reason": "explanation"
        }}
    ],
    "missed_objects": [
        {{
            "label": "object that should have been detected",
            "approximate_location": "description of where in frame"
        }}
    ],
    "hallucinated_objects": [
        {{
            "label": "falsely detected object",
            "reason": "why this is wrong"
        }}
    ],
    "quality_issues": [
        "list of detection quality problems"
    ],
    "improvement_suggestions": [
        "specific suggestions for improving detection"
    ]
}}

Be thorough but fair. Objects partially visible at frame edges are acceptable. Focus on clear mistakes."""

        try:
            # Read image
            with open(sample.frame_path, "rb") as f:
                img_data = f.read()
            
            response = model.generate_content([
                prompt,
                {"mime_type": "image/jpeg", "data": img_data}
            ])
            
            # Parse response
            text = response.text or "{}"
            # Clean markdown code blocks
            if "```" in text:
                lines = text.split("\n")
                text = "\n".join(
                    line for line in lines 
                    if not line.strip().startswith("```")
                )
            
            result = json.loads(text.strip())
            
            # Count verdicts
            validation = GeminiValidation(frame_idx=sample.frame_idx)
            for det_val in result.get("detection_validation", []):
                verdict = det_val.get("verdict", "").upper()
                if verdict == "CORRECT":
                    validation.correct += 1
                elif verdict in ("INCORRECT", "WRONG_LABEL"):
                    validation.incorrect += 1
                    validation.issues.append(
                        f"{det_val.get('label')} → should be {det_val.get('correct_label', 'unknown')}: {det_val.get('reason', '')}"
                    )
                elif verdict == "PARTIAL":
                    validation.correct += 0.5
                    validation.incorrect += 0.5
            
            validation.missed = len(result.get("missed_objects", []))
            validation.hallucinated = len(result.get("hallucinated_objects", []))
            
            for missed in result.get("missed_objects", []):
                validation.issues.append(f"MISSED: {missed.get('label')} at {missed.get('approximate_location', 'unknown')}")
            
            for halluc in result.get("hallucinated_objects", []):
                validation.issues.append(f"HALLUCINATED: {halluc.get('label')} - {halluc.get('reason', '')}")
            
            validation.suggestions = result.get("improvement_suggestions", [])
            validations.append(validation)
            
            # Rate limiting
            time.sleep(1.0)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response for frame {sample.frame_idx}: {e}")
            logger.error(f"Response was: {text[:500]}")
        except Exception as e:
            logger.error(f"Gemini validation failed for frame {sample.frame_idx}: {e}")
    
    return validations


def compute_metrics(validations: List[GeminiValidation]) -> Tuple[float, float, float]:
    """Compute precision, recall, F1 from validations."""
    
    total_correct = sum(v.correct for v in validations)
    total_incorrect = sum(v.incorrect for v in validations)
    total_missed = sum(v.missed for v in validations)
    total_hallucinated = sum(v.hallucinated for v in validations)
    
    # Precision: correct / (correct + incorrect + hallucinated)
    precision_denom = total_correct + total_incorrect + total_hallucinated
    precision = total_correct / precision_denom if precision_denom > 0 else 0.0
    
    # Recall: correct / (correct + missed)
    recall_denom = total_correct + total_missed
    recall = total_correct / recall_denom if recall_denom > 0 else 0.0
    
    # F1
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def aggregate_issues(validations: List[GeminiValidation]) -> List[str]:
    """Find common issues across validations."""
    
    issue_counts: Dict[str, int] = {}
    for v in validations:
        for issue in v.issues:
            # Normalize issue
            key = issue.split(":")[0].strip() if ":" in issue else issue[:50]
            issue_counts[key] = issue_counts.get(key, 0) + 1
    
    # Sort by frequency
    sorted_issues = sorted(issue_counts.items(), key=lambda x: -x[1])
    return [f"{issue} (x{count})" for issue, count in sorted_issues[:10]]


def run_perception(
    video_path: Path,
    output_dir: Path,
    mode: str = "balanced"
) -> Tuple[Any, float]:
    """Run perception pipeline on video."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create config - no mode parameter, just use defaults
    config = PerceptionConfig()
    
    engine = PerceptionEngine(config=config, verbose=True)
    
    start_time = time.time()
    result = engine.process_video(
        video_path=str(video_path),
        save_visualizations=True,
        output_dir=str(output_dir),
    )
    elapsed = time.time() - start_time
    
    # Save observations as tracks.jsonl for Gemini validation
    tracks_path = output_dir / "tracks.jsonl"
    with open(tracks_path, "w") as f:
        for obs in result.raw_observations:
            track = {
                "frame_idx": obs.frame_number,
                "category": obs.object_class.value if hasattr(obs.object_class, 'value') else str(obs.object_class),
                "confidence": obs.confidence,
                "bbox": [obs.bounding_box.x1, obs.bounding_box.y1, obs.bounding_box.x2, obs.bounding_box.y2],
                "track_id": obs.entity_id,
            }
            f.write(json.dumps(track) + "\n")
    logger.info(f"Saved {len(result.raw_observations)} observations to {tracks_path}")
    
    return result, elapsed


def run_full_evaluation(
    video_paths: List[Path],
    output_base: Path,
    num_samples: int = 10,
    mode: str = "balanced",
    skip_gemini: bool = False
) -> List[EvaluationResult]:
    """Run full evaluation on all videos."""
    
    results = []
    
    for video_path in video_paths:
        logger.info(f"\n{'='*60}")
        logger.info(f"EVALUATING: {video_path.name}")
        logger.info(f"{'='*60}\n")
        
        episode_name = f"eval_{video_path.stem}"
        output_dir = output_base / episode_name
        
        # Run perception
        logger.info("Running perception pipeline...")
        perception_result, elapsed = run_perception(video_path, output_dir, mode)
        
        # Extract sample frames
        tracks_path = output_dir / "tracks.jsonl"
        samples_dir = output_dir / "gemini_samples"
        
        if tracks_path.exists():
            samples = extract_sample_frames(
                video_path, tracks_path, samples_dir, num_samples
            )
        else:
            logger.warning(f"No tracks.jsonl found at {tracks_path}")
            samples = []
        
        # Validate with Gemini
        validations = []
        if not skip_gemini and samples:
            logger.info("Validating with Gemini API...")
            validations = validate_with_gemini(samples)
        
        # Compute metrics
        precision, recall, f1 = compute_metrics(validations) if validations else (0, 0, 0)
        common_issues = aggregate_issues(validations) if validations else []
        
        eval_result = EvaluationResult(
            video_path=str(video_path),
            total_frames=perception_result.total_frames if perception_result else 0,
            sampled_frames=len(set(obs.frame_number for obs in perception_result.raw_observations)) if perception_result else 0,
            total_detections=perception_result.total_detections if perception_result else 0,
            unique_entities=perception_result.unique_entities if perception_result else 0,
            processing_time=elapsed,
            validations=validations,
            precision=precision,
            recall=recall,
            f1_score=f1,
            common_issues=common_issues
        )
        
        results.append(eval_result)
        
        # Log summary
        logger.info(f"\n--- {video_path.name} Summary ---")
        logger.info(f"  Detections: {eval_result.total_detections}")
        logger.info(f"  Entities: {eval_result.unique_entities}")
        logger.info(f"  Time: {elapsed:.1f}s")
        if validations:
            logger.info(f"  Precision: {precision:.2%}")
            logger.info(f"  Recall: {recall:.2%}")
            logger.info(f"  F1: {f1:.2%}")
            if common_issues:
                logger.info(f"  Top Issues:")
                for issue in common_issues[:5]:
                    logger.info(f"    - {issue}")
    
    return results


def generate_report(results: List[EvaluationResult], output_path: Path) -> None:
    """Generate markdown evaluation report."""
    
    lines = [
        "# Full Evaluation Report with Gemini Validation",
        "",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Video | Detections | Entities | Precision | Recall | F1 | Time |",
        "|-------|-----------|----------|-----------|--------|-----|------|",
    ]
    
    for r in results:
        video_name = Path(r.video_path).name
        lines.append(
            f"| {video_name} | {r.total_detections} | {r.unique_entities} | "
            f"{r.precision:.1%} | {r.recall:.1%} | {r.f1_score:.1%} | {r.processing_time:.1f}s |"
        )
    
    lines.extend(["", "---", "", "## Detailed Results", ""])
    
    for r in results:
        video_name = Path(r.video_path).name
        lines.extend([
            f"### {video_name}",
            "",
            f"- **Total Frames:** {r.total_frames}",
            f"- **Sampled Frames:** {r.sampled_frames}",
            f"- **Total Detections:** {r.total_detections}",
            f"- **Unique Entities:** {r.unique_entities}",
            f"- **Processing Time:** {r.processing_time:.1f}s",
            "",
        ])
        
        if r.validations:
            lines.extend([
                "#### Gemini Validation Results",
                "",
                "| Frame | Correct | Incorrect | Missed | Hallucinated |",
                "|-------|---------|-----------|--------|--------------|",
            ])
            for v in r.validations:
                lines.append(f"| {v.frame_idx} | {v.correct} | {v.incorrect} | {v.missed} | {v.hallucinated} |")
            lines.append("")
        
        if r.common_issues:
            lines.extend(["#### Common Issues", ""])
            for issue in r.common_issues:
                lines.append(f"- {issue}")
            lines.append("")
        
        # Aggregate suggestions
        all_suggestions = []
        for v in r.validations:
            all_suggestions.extend(v.suggestions)
        
        if all_suggestions:
            lines.extend(["#### Improvement Suggestions", ""])
            seen = set()
            for s in all_suggestions:
                if s not in seen:
                    lines.append(f"- {s}")
                    seen.add(s)
            lines.append("")
    
    lines.extend(["", "---", "", "## Recommended Actions", ""])
    
    # Aggregate all issues across videos
    all_issues = []
    all_suggestions = []
    for r in results:
        all_issues.extend(r.common_issues)
        for v in r.validations:
            all_suggestions.extend(v.suggestions)
    
    if all_issues:
        lines.append("### Priority Issues to Fix")
        lines.append("")
        seen = set()
        for issue in all_issues[:10]:
            if issue not in seen:
                lines.append(f"1. {issue}")
                seen.add(issue)
        lines.append("")
    
    if all_suggestions:
        lines.append("### Implementation Suggestions")
        lines.append("")
        seen = set()
        for s in all_suggestions[:10]:
            if s not in seen:
                lines.append(f"1. {s}")
                seen.add(s)
        lines.append("")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    logger.info(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Full Gemini Evaluation")
    parser.add_argument(
        "--videos",
        nargs="+",
        default=["data/examples/test.mp4", "data/examples/video.mp4"],
        help="Video files to evaluate"
    )
    parser.add_argument(
        "--output",
        default="results/gemini_evaluation",
        help="Output directory"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of sample frames per video"
    )
    parser.add_argument(
        "--mode",
        choices=["fast", "balanced", "accurate"],
        default="balanced",
        help="Perception mode"
    )
    parser.add_argument(
        "--skip-gemini",
        action="store_true",
        help="Skip Gemini validation (just run perception)"
    )
    
    args = parser.parse_args()
    
    video_paths = [Path(v) for v in args.videos]
    for vp in video_paths:
        if not vp.exists():
            logger.error(f"Video not found: {vp}")
            sys.exit(1)
    
    output_base = Path(args.output)
    
    results = run_full_evaluation(
        video_paths,
        output_base,
        num_samples=args.samples,
        mode=args.mode,
        skip_gemini=args.skip_gemini
    )
    
    # Generate report
    report_path = output_base / "evaluation_report.md"
    generate_report(results, report_path)
    
    # Save raw results
    raw_results_path = output_base / "evaluation_results.json"
    with open(raw_results_path, "w") as f:
        json.dump([{
            "video_path": r.video_path,
            "total_frames": r.total_frames,
            "sampled_frames": r.sampled_frames,
            "total_detections": r.total_detections,
            "unique_entities": r.unique_entities,
            "processing_time": r.processing_time,
            "precision": r.precision,
            "recall": r.recall,
            "f1_score": r.f1_score,
            "common_issues": r.common_issues,
            "validations": [
                {
                    "frame_idx": v.frame_idx,
                    "correct": v.correct,
                    "incorrect": v.incorrect,
                    "missed": v.missed,
                    "hallucinated": v.hallucinated,
                    "issues": v.issues,
                    "suggestions": v.suggestions
                }
                for v in r.validations
            ]
        } for r in results], f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info("EVALUATION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Report: {report_path}")
    logger.info(f"Raw results: {raw_results_path}")


if __name__ == "__main__":
    main()
