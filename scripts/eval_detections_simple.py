#!/usr/bin/env python3
"""
Simple Detection Evaluation Script - No External API Required
Compares YOLO11m vs YOLO11x detection outputs and visualizes results.
"""

import argparse
import cv2
import json
import logging
import numpy as np
import time
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_detector(model_name: str, video_path: str, sample_frames: int = 30, conf_threshold: float = 0.25):
    """Run detection on sampled frames and collect statistics."""
    import torch
    from ultralytics import YOLO
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {model_name}")
    logger.info(f"{'='*60}")
    
    # Load model
    model = YOLO(f"{model_name}.pt")
    
    # Auto-detect device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model.to(device)
    logger.info(f"Model loaded on {device}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Video: {total_frames} frames @ {fps:.1f} FPS ({width}x{height})")
    
    # Sample frames uniformly
    frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)
    
    # Statistics
    all_detections = []
    class_counts = defaultdict(int)
    confidence_by_class = defaultdict(list)
    processing_times = []
    bbox_sizes = []
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Run detection
        start = time.time()
        results = model(frame, conf=conf_threshold, verbose=False)
        elapsed = time.time() - start
        processing_times.append(elapsed)
        
        frame_detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                
                # Calculate bbox size
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                bbox_area = bbox_w * bbox_h
                rel_area = bbox_area / (width * height)
                
                detection = {
                    "frame_idx": int(frame_idx),
                    "class": cls_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                    "bbox_size": [bbox_w, bbox_h],
                    "relative_area": rel_area
                }
                frame_detections.append(detection)
                
                class_counts[cls_name] += 1
                confidence_by_class[cls_name].append(conf)
                bbox_sizes.append(rel_area)
        
        all_detections.append({
            "frame_idx": int(frame_idx),
            "timestamp": frame_idx / fps,
            "detections": frame_detections,
            "detection_count": len(frame_detections)
        })
        
        if (i + 1) % 10 == 0:
            logger.info(f"  Processed {i+1}/{sample_frames} frames...")
    
    cap.release()
    
    # Compute statistics
    total_detections = sum(class_counts.values())
    avg_time = np.mean(processing_times) if processing_times else 0
    avg_detections_per_frame = total_detections / sample_frames
    
    # Class-level stats
    class_stats = {}
    for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        confs = confidence_by_class[cls_name]
        class_stats[cls_name] = {
            "count": count,
            "avg_confidence": float(np.mean(confs)),
            "min_confidence": float(np.min(confs)),
            "max_confidence": float(np.max(confs)),
            "std_confidence": float(np.std(confs)) if len(confs) > 1 else 0
        }
    
    results = {
        "model": model_name,
        "video": video_path,
        "sample_frames": sample_frames,
        "conf_threshold": conf_threshold,
        "device": device,
        "stats": {
            "total_detections": total_detections,
            "avg_detections_per_frame": avg_detections_per_frame,
            "unique_classes": len(class_counts),
            "avg_processing_time": float(avg_time),
            "fps": 1.0 / avg_time if avg_time > 0 else 0,
            "avg_bbox_relative_area": float(np.mean(bbox_sizes)) if bbox_sizes else 0,
            "median_bbox_relative_area": float(np.median(bbox_sizes)) if bbox_sizes else 0,
        },
        "class_stats": class_stats,
        "frames": all_detections
    }
    
    return results


def compare_detectors(results_list: list, output_dir: Path):
    """Compare multiple detector results."""
    
    logger.info("\n" + "="*70)
    logger.info("COMPARISON SUMMARY")
    logger.info("="*70)
    
    comparison = {
        "models": [],
        "comparison": {}
    }
    
    # Print summary table
    print("\n{:<15} {:>12} {:>12} {:>10} {:>10} {:>12}".format(
        "Model", "Detections", "Avg/Frame", "Classes", "FPS", "Avg Conf"))
    print("-" * 75)
    
    for r in results_list:
        model = r["model"]
        stats = r["stats"]
        
        # Calculate overall average confidence
        total_conf = 0
        total_count = 0
        for cls_stats in r["class_stats"].values():
            total_conf += cls_stats["avg_confidence"] * cls_stats["count"]
            total_count += cls_stats["count"]
        avg_conf = total_conf / total_count if total_count > 0 else 0
        
        print("{:<15} {:>12} {:>12.1f} {:>10} {:>10.1f} {:>12.2f}".format(
            model,
            stats["total_detections"],
            stats["avg_detections_per_frame"],
            stats["unique_classes"],
            stats["fps"],
            avg_conf
        ))
        
        comparison["models"].append({
            "name": model,
            "total_detections": stats["total_detections"],
            "avg_detections_per_frame": stats["avg_detections_per_frame"],
            "unique_classes": stats["unique_classes"],
            "fps": stats["fps"],
            "avg_confidence": avg_conf
        })
    
    # Class-by-class comparison
    print("\n\n{:<25} ".format("Class") + " ".join(["{:>15}".format(r["model"]) for r in results_list]))
    print("-" * (25 + 16 * len(results_list)))
    
    all_classes = set()
    for r in results_list:
        all_classes.update(r["class_stats"].keys())
    
    class_comparison = {}
    for cls_name in sorted(all_classes):
        counts = []
        for r in results_list:
            count = r["class_stats"].get(cls_name, {}).get("count", 0)
            counts.append(count)
        
        if any(c > 0 for c in counts):
            print("{:<25} ".format(cls_name[:25]) + " ".join(["{:>15}".format(c) for c in counts]))
            class_comparison[cls_name] = {r["model"]: c for r, c in zip(results_list, counts)}
    
    comparison["class_comparison"] = class_comparison
    
    # Confidence distribution by model
    print("\n\nCONFIDENCE ANALYSIS:")
    print("-" * 70)
    for r in results_list:
        print(f"\n{r['model']}:")
        for cls_name, stats in sorted(r["class_stats"].items(), key=lambda x: -x[1]["count"])[:10]:
            print(f"  {cls_name:20}: n={stats['count']:4}, conf={stats['avg_confidence']:.3f} ± {stats['std_confidence']:.3f}")
    
    # Look for potential issues
    print("\n\nPOTENTIAL ISSUES:")
    print("-" * 70)
    
    issues = []
    
    # Check for classes with low confidence
    for r in results_list:
        for cls_name, stats in r["class_stats"].items():
            if stats["avg_confidence"] < 0.4:
                issue = f"[{r['model']}] Low confidence for '{cls_name}': {stats['avg_confidence']:.3f}"
                issues.append(issue)
                print(f"  ⚠ {issue}")
    
    # Check for detection count variance
    if len(results_list) == 2:
        counts1 = results_list[0]["stats"]["total_detections"]
        counts2 = results_list[1]["stats"]["total_detections"]
        diff_pct = abs(counts1 - counts2) / max(counts1, counts2) * 100
        if diff_pct > 20:
            issue = f"Detection count differs by {diff_pct:.1f}% between models"
            issues.append(issue)
            print(f"  ⚠ {issue}")
    
    # Check for high-count low-confidence (potential false positives)
    for r in results_list:
        for cls_name, stats in r["class_stats"].items():
            if stats["count"] > 10 and stats["avg_confidence"] < 0.35:
                issue = f"[{r['model']}] Potential false positives: '{cls_name}' (n={stats['count']}, conf={stats['avg_confidence']:.3f})"
                issues.append(issue)
                print(f"  ⚠ {issue}")
    
    comparison["issues"] = issues
    
    return comparison


def visualize_detections(results: dict, video_path: str, output_dir: Path, max_frames: int = 5):
    """Save visualization frames with detections."""
    
    viz_dir = output_dir / f"{results['model']}_viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    
    # Pick frames with most detections
    frames_sorted = sorted(results["frames"], key=lambda x: -x["detection_count"])[:max_frames]
    
    colors = {}
    
    for frame_data in frames_sorted:
        frame_idx = frame_data["frame_idx"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        for det in frame_data["detections"]:
            cls_name = det["class"]
            if cls_name not in colors:
                # Generate random color
                np.random.seed(hash(cls_name) % 2**32)
                colors[cls_name] = tuple(map(int, np.random.randint(0, 255, 3)))
            
            x1, y1, x2, y2 = map(int, det["bbox"])
            color = colors[cls_name]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name}: {det['confidence']:.2f}"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add frame info
        cv2.putText(frame, f"Frame {frame_idx} | {len(frame_data['detections'])} detections",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        output_path = viz_dir / f"frame_{frame_idx:05d}.jpg"
        cv2.imwrite(str(output_path), frame)
    
    cap.release()
    logger.info(f"Saved {len(frames_sorted)} visualization frames to {viz_dir}")


def main():
    parser = argparse.ArgumentParser(description="Simple Detection Evaluation")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--models", default="yolo11m,yolo11x", help="Comma-separated model names")
    parser.add_argument("--sample-frames", type=int, default=30, help="Number of frames to sample")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--output-dir", default="results/detection_eval", help="Output directory")
    parser.add_argument("--visualize", action="store_true", help="Save visualization frames")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models = [m.strip() for m in args.models.split(",")]
    
    logger.info(f"Evaluating models: {models}")
    logger.info(f"Video: {args.video}")
    logger.info(f"Sample frames: {args.sample_frames}")
    logger.info(f"Confidence threshold: {args.conf}")
    
    all_results = []
    
    for model_name in models:
        try:
            results = evaluate_detector(
                model_name=model_name,
                video_path=args.video,
                sample_frames=args.sample_frames,
                conf_threshold=args.conf
            )
            all_results.append(results)
            
            # Save individual results
            output_file = output_dir / f"{model_name}_results.json"
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved results to {output_file}")
            
            # Optionally visualize
            if args.visualize:
                visualize_detections(results, args.video, output_dir)
                
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Compare if multiple models
    if len(all_results) > 1:
        comparison = compare_detectors(all_results, output_dir)
        
        comparison_file = output_dir / "comparison.json"
        with open(comparison_file, "w") as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"\nComparison saved to {comparison_file}")
    
    logger.info("\n" + "="*70)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*70)


if __name__ == "__main__":
    main()
