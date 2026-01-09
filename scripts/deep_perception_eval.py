#!/usr/bin/env python3
"""
Deep Perception Pipeline Evaluation with Gemini Analysis

This script runs a comprehensive evaluation of the improved perception pipeline:
1. Detection (YOLO11x with improved thresholds)
2. Classification (DINOv3 full-image context)
3. Filtering (FastVLM verification)
4. Re-ID (V-JEPA2 tracking)

Uses Gemini 2.0 Flash for deep analysis of:
- Why specific failures occur
- Track ID switches and fragmentation
- Root causes of misclassifications
- Actionable improvement suggestions

Usage:
    python scripts/deep_perception_eval.py --video data/examples/test.mp4
"""

import argparse
import base64
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import traceback

import cv2
import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("deep_eval")


# =====================================================================
# GEMINI DEEP ANALYSIS
# =====================================================================

def get_gemini():
    """Initialize Gemini client."""
    try:
        import google.generativeai as genai
        
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("No Gemini API key - set GEMINI_API_KEY")
            return None
        
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.0-flash-exp")
    except Exception as e:
        logger.warning(f"Failed to init Gemini: {e}")
        return None


def deep_analyze_frame(
    gemini,
    frame: np.ndarray,
    detections: List[Dict],
    frame_idx: int,
    analysis_type: str = "detection"
) -> Dict:
    """Run deep Gemini analysis on a frame."""
    
    if gemini is None:
        return {"error": "Gemini not available"}
    
    # Convert frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    # Resize for API
    max_size = 1024
    if max(pil_image.size) > max_size:
        ratio = max_size / max(pil_image.size)
        new_size = (int(pil_image.width * ratio), int(pil_image.height * ratio))
        pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
    
    # Format detections
    det_text = ""
    for i, det in enumerate(detections):
        bbox = det.get("bbox") or det.get("bbox_2d", [])
        conf = det.get("confidence", 0.0)
        cls = det.get("class_name") or det.get("refined_class", "unknown")
        original_cls = det.get("class_name", cls)
        refined_cls = det.get("refined_class", cls)
        
        det_text += f"{i+1}. Original: '{original_cls}' → Refined: '{refined_cls}' (conf={conf:.2f}) bbox={bbox}\n"
    
    if not det_text:
        det_text = "No detections"
    
    # Build analysis prompt based on type
    if analysis_type == "detection":
        prompt = f"""Deeply analyze the DETECTION accuracy for frame {frame_idx}.

Detections:
{det_text}

For this analysis:
1. Identify every REAL object visible in the frame
2. For each detection, classify as:
   - CORRECT: Detection matches a real object
   - MISCLASSIFIED: Real object detected but wrong class
   - HALLUCINATION: No corresponding real object
3. List MISSED objects (visible but not detected)

For each error, provide ROOT CAUSE analysis:
- Why did the model make this mistake?
- What visual features caused confusion?
- What contextual clues were ignored?

Output JSON:
{{
  "total_real_objects": <count of actual objects in scene>,
  "correct_detections": [
    {{"index": 1, "class": "...", "notes": "..."}}
  ],
  "misclassifications": [
    {{"index": 2, "detected": "...", "actual": "...", "root_cause": "...", "visual_confusion": "..."}}
  ],
  "hallucinations": [
    {{"index": 3, "detected": "...", "actual_content": "...", "root_cause": "..."}}
  ],
  "missed_objects": [
    {{"object": "...", "location": "...", "why_missed": "..."}}
  ],
  "scene_context": "brief description of scene",
  "improvement_suggestions": ["suggestion1", "suggestion2"]
}}
"""
    
    elif analysis_type == "classification":
        prompt = f"""Deeply analyze the CLASSIFICATION accuracy for frame {frame_idx}.

Detections (showing original → refined class):
{det_text}

Evaluate the FINE-GRAINED classification:
1. For each detection, is the REFINED class better than the ORIGINAL?
2. What is the most specific correct class for each object?
3. Why did classification succeed or fail?

Consider scene context:
- Is this an office, bedroom, kitchen, etc.?
- Does the refined class make sense in context?

Output JSON:
{{
  "scene_type": "office/bedroom/kitchen/etc",
  "classification_results": [
    {{
      "index": 1,
      "original_class": "...",
      "refined_class": "...",
      "correct_class": "...",
      "classification_correct": true/false,
      "refinement_helped": true/false,
      "reasoning": "..."
    }}
  ],
  "context_errors": [
    {{"issue": "...", "why": "..."}}
  ],
  "improvement_suggestions": ["..."]
}}
"""
    
    elif analysis_type == "tracking":
        prompt = f"""Deeply analyze the TRACKING for frame {frame_idx}.

Tracked objects:
{det_text}

Evaluate tracking quality:
1. Are track IDs consistent? (Same object should have same ID)
2. Any ID SWITCHES? (Same object, different IDs across frames)
3. Any ID MERGES? (Different objects, same ID)
4. Any track FRAGMENTATION? (Track breaks unnecessarily)

Look for:
- Occlusion handling
- Fast motion blur
- Similar-looking objects
- Scene transitions

Output JSON:
{{
  "unique_objects_visible": <count>,
  "track_quality": "excellent/good/fair/poor",
  "id_switches": [
    {{"object": "...", "issue": "..."}}
  ],
  "id_merges": [
    {{"objects": "...", "issue": "..."}}
  ],
  "fragmentation": [
    {{"object": "...", "cause": "..."}}
  ],
  "recommendations": ["..."]
}}
"""
    
    else:
        prompt = f"Analyze frame {frame_idx} detections: {det_text}"
    
    try:
        response = gemini.generate_content([prompt, pil_image])
        text = response.text
        
        # Extract JSON
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        return json.loads(text)
    except Exception as e:
        logger.error(f"Gemini analysis failed: {e}")
        return {"error": str(e)}


def analyze_tracks_across_video(
    gemini,
    track_data: List[Dict],
    sample_frames: List[np.ndarray],
    sample_indices: List[int]
) -> Dict:
    """Analyze tracking consistency across video using multiple frames."""
    
    if gemini is None:
        return {"error": "Gemini not available"}
    
    # Build track summary
    tracks_by_id = defaultdict(list)
    for obs in track_data:
        track_id = obs.get("track_id") or obs.get("id")
        tracks_by_id[track_id].append(obs)
    
    track_summary = []
    for track_id, observations in tracks_by_id.items():
        classes = list(set(o.get("class_name", "unknown") for o in observations))
        frames = [o.get("frame_idx", 0) for o in observations]
        track_summary.append({
            "track_id": track_id,
            "classes": classes,
            "frame_range": [min(frames), max(frames)],
            "num_observations": len(observations)
        })
    
    # Create composite image from samples
    composite_images = []
    for i, (frame, idx) in enumerate(zip(sample_frames[:3], sample_indices[:3])):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)
        pil_frame = pil_frame.resize((512, 384))
        composite_images.append(pil_frame)
    
    prompt = f"""Analyze TRACKING CONSISTENCY across this video.

Track Summary (total {len(track_summary)} tracks):
{json.dumps(track_summary[:20], indent=2)}

Review the sample frames and track data for:
1. IDENTITY CONSISTENCY: Does each track_id represent ONE unique object?
2. CLASS CONSISTENCY: If a track has multiple classes, is that an error?
3. TRACK LIFESPAN: Are tracks too short (fragmentation)?
4. OVERCOUNTING: Multiple tracks for same object?

Provide deep analysis:
- What causes tracking failures in this video?
- Are there specific object types that fail more?
- Scene transitions or occlusions causing issues?

Output JSON:
{{
  "total_tracks": <number>,
  "estimated_real_objects": <number>,
  "tracking_quality_score": 0-100,
  "consistency_issues": [
    {{"track_id": ..., "issue": "...", "root_cause": "..."}}
  ],
  "class_drift_issues": [
    {{"track_id": ..., "classes": [...], "likely_correct": "..."}}
  ],
  "fragmentation_issues": [
    {{"description": "...", "cause": "..."}}
  ],
  "overcounting": {{"detected": true/false, "details": "..."}},
  "improvement_priorities": ["priority1", "priority2", "priority3"]
}}
"""
    
    try:
        content = [prompt] + composite_images
        response = gemini.generate_content(content)
        text = response.text
        
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        return json.loads(text)
    except Exception as e:
        logger.error(f"Track analysis failed: {e}")
        return {"error": str(e)}


# =====================================================================
# PIPELINE COMPONENTS
# =====================================================================

def run_improved_detection(
    video_path: Path,
    sample_frames: int = 20,
    conf_threshold: float = 0.25
) -> Tuple[List[Dict], List[np.ndarray], List[int]]:
    """Run improved YOLO11x detection."""
    
    logger.info("Running YOLO11x detection...")
    
    from ultralytics import YOLO
    
    model = YOLO("yolo11x.pt")
    model.to("mps")
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)
    
    all_detections = []
    frames = []
    indices = []
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        results = model(frame, conf=conf_threshold, verbose=False)
        
        frame_dets = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                crop = frame[int(y1):int(y2), int(x1):int(x2)]
                crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)) if crop.size > 0 else None
                
                frame_dets.append({
                    "bbox": [x1, y1, x2, y2],
                    "class_name": r.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "frame_idx": int(frame_idx),
                    "crop": crop_pil
                })
        
        all_detections.append({
            "frame_idx": int(frame_idx),
            "detections": frame_dets
        })
        frames.append(frame)
        indices.append(int(frame_idx))
    
    cap.release()
    
    logger.info(f"Detection complete: {sum(len(d['detections']) for d in all_detections)} detections across {len(all_detections)} frames")
    
    return all_detections, frames, indices


def run_dino_classification(
    detection_results: List[Dict],
    frames: List[np.ndarray]
) -> List[Dict]:
    """Run DINOv3 full-image classification."""
    
    logger.info("Running DINOv3 classification...")
    
    try:
        from orion.backends.dino_classifier import DINOv3Classifier, SceneTypeDetector
        
        classifier = DINOv3Classifier(device="mps", scene_aware=True)
        scene_detector = SceneTypeDetector()
        
        classified_results = []
        
        for det_result, frame in zip(detection_results, frames):
            detections = det_result["detections"]
            
            if not detections:
                classified_results.append(det_result)
                continue
            
            # Detect scene type
            classes = [d["class_name"] for d in detections]
            scene_type, _ = scene_detector.detect_scene_type(classes)
            
            # Classify with context
            refined = classifier.classify_frame_detections(frame, detections, scene_type)
            
            classified_results.append({
                "frame_idx": det_result["frame_idx"],
                "scene_type": scene_type,
                "detections": refined
            })
        
        logger.info("DINOv3 classification complete")
        return classified_results
        
    except Exception as e:
        logger.error(f"DINOv3 classification failed: {e}")
        traceback.print_exc()
        return detection_results


def run_vlm_filtering(
    classified_results: List[Dict]
) -> List[Dict]:
    """Run FastVLM verification filtering."""
    
    logger.info("Running FastVLM filtering...")
    
    try:
        from orion.perception.fastvlm_filter import FastVLMFilter
        
        vlm_filter = FastVLMFilter(enable_vlm=True)
        
        filtered_results = []
        total_input = 0
        total_output = 0
        
        for result in classified_results:
            detections = result["detections"]
            scene_type = result.get("scene_type", "unknown")
            
            total_input += len(detections)
            
            filtered, stats = vlm_filter.filter_detections(detections, scene_type)
            
            total_output += len(filtered)
            
            filtered_results.append({
                "frame_idx": result["frame_idx"],
                "scene_type": scene_type,
                "detections": filtered,
                "filter_stats": {
                    "input": stats.total_input,
                    "kept": stats.kept,
                    "removed_vlm": stats.removed_vlm,
                    "removed_threshold": stats.removed_threshold
                }
            })
        
        logger.info(f"VLM filtering complete: {total_input} → {total_output} detections")
        return filtered_results
        
    except Exception as e:
        logger.error(f"VLM filtering failed: {e}")
        traceback.print_exc()
        return classified_results


# =====================================================================
# MAIN EVALUATION
# =====================================================================

def run_deep_evaluation(
    video_path: Path,
    output_dir: Path,
    sample_frames: int = 20,
    analyze_detection: bool = True,
    analyze_classification: bool = True,
    analyze_tracking: bool = True
) -> Dict:
    """Run comprehensive deep evaluation."""
    
    logger.info("=" * 70)
    logger.info("DEEP PERCEPTION PIPELINE EVALUATION")
    logger.info("=" * 70)
    logger.info(f"Video: {video_path}")
    logger.info(f"Output: {output_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Gemini
    gemini = get_gemini()
    if gemini:
        logger.info("✓ Gemini API initialized")
    else:
        logger.warning("⚠ Running without Gemini analysis")
    
    results = {
        "video": str(video_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stages": {}
    }
    
    # Stage 1: Detection
    logger.info("\n" + "=" * 50)
    logger.info("STAGE 1: DETECTION (YOLO11x)")
    logger.info("=" * 50)
    
    start_time = time.time()
    detection_results, frames, indices = run_improved_detection(
        video_path, sample_frames=sample_frames
    )
    detection_time = time.time() - start_time
    
    results["stages"]["detection"] = {
        "time": detection_time,
        "frames_processed": len(frames),
        "total_detections": sum(len(d["detections"]) for d in detection_results)
    }
    
    # Gemini detection analysis
    if gemini and analyze_detection:
        logger.info("Running Gemini detection analysis...")
        detection_analyses = []
        
        for i in range(min(5, len(frames))):
            analysis = deep_analyze_frame(
                gemini, frames[i], detection_results[i]["detections"],
                indices[i], analysis_type="detection"
            )
            detection_analyses.append({
                "frame_idx": indices[i],
                "analysis": analysis
            })
            time.sleep(1)  # Rate limiting
        
        results["stages"]["detection"]["gemini_analysis"] = detection_analyses
    
    # Stage 2: Classification
    logger.info("\n" + "=" * 50)
    logger.info("STAGE 2: CLASSIFICATION (DINOv3)")
    logger.info("=" * 50)
    
    start_time = time.time()
    classified_results = run_dino_classification(detection_results, frames)
    classification_time = time.time() - start_time
    
    results["stages"]["classification"] = {
        "time": classification_time
    }
    
    # Gemini classification analysis
    if gemini and analyze_classification:
        logger.info("Running Gemini classification analysis...")
        classification_analyses = []
        
        for i in range(min(5, len(frames))):
            analysis = deep_analyze_frame(
                gemini, frames[i], classified_results[i]["detections"],
                indices[i], analysis_type="classification"
            )
            classification_analyses.append({
                "frame_idx": indices[i],
                "scene_type": classified_results[i].get("scene_type", "unknown"),
                "analysis": analysis
            })
            time.sleep(1)
        
        results["stages"]["classification"]["gemini_analysis"] = classification_analyses
    
    # Stage 3: Filtering
    logger.info("\n" + "=" * 50)
    logger.info("STAGE 3: FILTERING (FastVLM)")
    logger.info("=" * 50)
    
    start_time = time.time()
    filtered_results = run_vlm_filtering(classified_results)
    filtering_time = time.time() - start_time
    
    results["stages"]["filtering"] = {
        "time": filtering_time,
        "total_before": sum(len(r["detections"]) for r in classified_results),
        "total_after": sum(len(r["detections"]) for r in filtered_results),
        "removal_rate": 1 - (sum(len(r["detections"]) for r in filtered_results) / 
                           max(1, sum(len(r["detections"]) for r in classified_results)))
    }
    
    # Stage 4: Tracking Analysis
    if analyze_tracking:
        logger.info("\n" + "=" * 50)
        logger.info("STAGE 4: TRACKING ANALYSIS (V-JEPA2)")
        logger.info("=" * 50)
        
        # Compile all detections as pseudo-tracks
        all_tracks = []
        for i, result in enumerate(filtered_results):
            for det in result["detections"]:
                track_entry = {
                    "track_id": f"temp_{hash(str(det.get('bbox', [])))}",
                    "frame_idx": result["frame_idx"],
                    **det
                }
                all_tracks.append(track_entry)
        
        if gemini:
            logger.info("Running Gemini tracking analysis...")
            tracking_analysis = analyze_tracks_across_video(
                gemini, all_tracks, frames, indices
            )
            results["stages"]["tracking"] = {
                "gemini_analysis": tracking_analysis
            }
    
    # Generate summary report
    logger.info("\n" + "=" * 70)
    logger.info("GENERATING SUMMARY REPORT")
    logger.info("=" * 70)
    
    report_lines = [
        "# Deep Perception Pipeline Evaluation Report",
        f"\n**Video:** {video_path}",
        f"\n**Timestamp:** {results['timestamp']}",
        "\n---\n",
        "## Processing Times\n",
        f"- Detection (YOLO11x): {detection_time:.2f}s",
        f"- Classification (DINOv3): {classification_time:.2f}s",
        f"- Filtering (FastVLM): {filtering_time:.2f}s",
        f"- **Total:** {detection_time + classification_time + filtering_time:.2f}s",
        "\n## Detection Summary\n",
        f"- Frames sampled: {len(frames)}",
        f"- Total detections: {results['stages']['detection']['total_detections']}",
        "\n## Filtering Summary\n",
        f"- Before filtering: {results['stages']['filtering']['total_before']}",
        f"- After filtering: {results['stages']['filtering']['total_after']}",
        f"- Removal rate: {results['stages']['filtering']['removal_rate']:.1%}",
    ]
    
    # Add Gemini findings
    if "gemini_analysis" in results["stages"].get("detection", {}):
        report_lines.append("\n## Detection Analysis (Gemini)\n")
        for analysis in results["stages"]["detection"]["gemini_analysis"][:3]:
            a = analysis.get("analysis", {})
            if "error" not in a:
                report_lines.append(f"\n### Frame {analysis['frame_idx']}")
                report_lines.append(f"- Scene: {a.get('scene_context', 'N/A')}")
                report_lines.append(f"- Correct: {len(a.get('correct_detections', []))}")
                report_lines.append(f"- Misclassified: {len(a.get('misclassifications', []))}")
                report_lines.append(f"- Hallucinations: {len(a.get('hallucinations', []))}")
                report_lines.append(f"- Missed: {len(a.get('missed_objects', []))}")
                
                if a.get("improvement_suggestions"):
                    report_lines.append("\n**Suggestions:**")
                    for s in a["improvement_suggestions"][:3]:
                        report_lines.append(f"- {s}")
    
    if "gemini_analysis" in results["stages"].get("classification", {}):
        report_lines.append("\n## Classification Analysis (Gemini)\n")
        for analysis in results["stages"]["classification"]["gemini_analysis"][:3]:
            a = analysis.get("analysis", {})
            if "error" not in a:
                report_lines.append(f"\n### Frame {analysis['frame_idx']} (Scene: {analysis.get('scene_type', 'N/A')})")
                
                if a.get("classification_results"):
                    for cr in a["classification_results"][:5]:
                        status = "✓" if cr.get("classification_correct") else "✗"
                        report_lines.append(
                            f"- {status} {cr.get('original_class')} → {cr.get('refined_class')} "
                            f"(should be: {cr.get('correct_class')})"
                        )
    
    if "gemini_analysis" in results["stages"].get("tracking", {}):
        report_lines.append("\n## Tracking Analysis (Gemini)\n")
        ta = results["stages"]["tracking"]["gemini_analysis"]
        if "error" not in ta:
            report_lines.append(f"- Total tracks: {ta.get('total_tracks', 'N/A')}")
            report_lines.append(f"- Estimated real objects: {ta.get('estimated_real_objects', 'N/A')}")
            report_lines.append(f"- Quality score: {ta.get('tracking_quality_score', 'N/A')}/100")
            
            if ta.get("improvement_priorities"):
                report_lines.append("\n**Top Priorities:**")
                for p in ta["improvement_priorities"]:
                    report_lines.append(f"1. {p}")
    
    report_text = "\n".join(report_lines)
    
    # Save outputs
    report_path = output_dir / "deep_evaluation_report.md"
    report_path.write_text(report_text)
    logger.info(f"Report saved: {report_path}")
    
    json_path = output_dir / "deep_evaluation_results.json"
    
    # Remove non-serializable items
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items() if k != "crop"}
        elif isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.float64)):
            return float(obj)
        else:
            return obj
    
    clean_results = clean_for_json(results)
    with open(json_path, "w") as f:
        json.dump(clean_results, f, indent=2, default=str)
    logger.info(f"Results saved: {json_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print(report_text)
    print("=" * 70)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Deep perception pipeline evaluation")
    parser.add_argument("--video", type=str, required=True, help="Path to video")
    parser.add_argument("--output", type=str, default="results/deep_eval", help="Output directory")
    parser.add_argument("--frames", type=int, default=20, help="Number of frames to sample")
    parser.add_argument("--skip-detection", action="store_true", help="Skip detection analysis")
    parser.add_argument("--skip-classification", action="store_true", help="Skip classification analysis")
    parser.add_argument("--skip-tracking", action="store_true", help="Skip tracking analysis")
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        sys.exit(1)
    
    run_deep_evaluation(
        video_path=video_path,
        output_dir=Path(args.output),
        sample_frames=args.frames,
        analyze_detection=not args.skip_detection,
        analyze_classification=not args.skip_classification,
        analyze_tracking=not args.skip_tracking
    )


if __name__ == "__main__":
    main()
