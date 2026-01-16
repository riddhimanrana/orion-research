#!/usr/bin/env python3
"""
Semantic Filtering Evaluation
Compares threshold-based filtering vs VLM-backed verification
"""

import argparse
import cv2
import json
import logging
import numpy as np
import time
from collections import defaultdict
from pathlib import Path
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Suspicious labels that often cause false positives
SUSPICIOUS_LABELS = {
    "potted plant": 0.45,  # Often detects any green object
    "vase": 0.50,          # Bottles/containers confused
    "clock": 0.50,         # Round objects confused
    "book": 0.45,          # Rectangular objects confused
    "cell phone": 0.45,    # Small rectangles
    "remote": 0.50,        # Small objects
    "mouse": 0.55,         # Small objects
    "keyboard": 0.50,      # Pattern matching issues
    "sports ball": 0.55,   # Round objects
    "frisbee": 0.60,       # Round objects
    "bowl": 0.45,          # Container confusion
    "cup": 0.45,           # Container confusion
    "banana": 0.50,        # Yellow objects
    "apple": 0.50,         # Round objects
    "orange": 0.50,        # Round/orange objects
    "donut": 0.55,         # Round objects
    "cake": 0.50,          # Food misdetection
    "bird": 0.45,          # Flying objects/patterns
    "cat": 0.40,           # Animal confusion
    "dog": 0.40,           # Animal confusion
    "teddy bear": 0.50,    # Plush objects
    "handbag": 0.45,       # Bag-like objects
    "tie": 0.50,           # Long narrow objects
    "scissors": 0.55,      # Tool confusion
    "toothbrush": 0.55,    # Small objects
}

# High-confidence labels that rarely need verification
HIGH_CONFIDENCE_LABELS = {
    "person", "car", "truck", "bus", "train", "motorcycle", "bicycle",
    "chair", "couch", "bed", "dining table", "tv", "laptop", "refrigerator",
    "oven", "microwave", "sink", "toilet"
}


def get_device():
    """Auto-detect best available device."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def evaluate_threshold_filtering(base_detections: list):
    """Evaluate threshold-based semantic filtering."""
    logger.info("\n" + "="*60)
    logger.info("THRESHOLD-BASED FILTERING EVALUATION")
    logger.info("="*60)
    
    results = {
        "method": "threshold",
        "kept": [],
        "filtered": [],
        "stats_by_class": defaultdict(lambda: {"kept": 0, "filtered": 0})
    }
    
    for frame_data in base_detections:
        for det in frame_data.get("detections", []):
            cls_name = det["class"]
            conf = det["confidence"]
            
            # Apply class-specific thresholds
            if cls_name in SUSPICIOUS_LABELS:
                threshold = SUSPICIOUS_LABELS[cls_name]
            elif cls_name in HIGH_CONFIDENCE_LABELS:
                threshold = 0.30  # Lower threshold for reliable classes
            else:
                threshold = 0.40  # Default threshold
            
            if conf >= threshold:
                results["kept"].append({
                    "class": cls_name,
                    "confidence": conf,
                    "threshold": threshold,
                    "frame_idx": frame_data["frame_idx"]
                })
                results["stats_by_class"][cls_name]["kept"] += 1
            else:
                results["filtered"].append({
                    "class": cls_name,
                    "confidence": conf,
                    "threshold": threshold,
                    "frame_idx": frame_data["frame_idx"]
                })
                results["stats_by_class"][cls_name]["filtered"] += 1
    
    total = len(results["kept"]) + len(results["filtered"])
    results["stats"] = {
        "total_detections": total,
        "kept": len(results["kept"]),
        "filtered": len(results["filtered"]),
        "filter_rate": len(results["filtered"]) / total if total > 0 else 0
    }
    results["stats_by_class"] = dict(results["stats_by_class"])
    
    return results


def evaluate_confidence_only_filtering(base_detections: list, threshold: float = 0.35):
    """Evaluate simple confidence-only filtering (baseline)."""
    logger.info("\n" + "="*60)
    logger.info(f"CONFIDENCE-ONLY FILTERING (threshold={threshold})")
    logger.info("="*60)
    
    results = {
        "method": f"confidence_{threshold}",
        "kept": [],
        "filtered": [],
        "stats_by_class": defaultdict(lambda: {"kept": 0, "filtered": 0})
    }
    
    for frame_data in base_detections:
        for det in frame_data.get("detections", []):
            cls_name = det["class"]
            conf = det["confidence"]
            
            if conf >= threshold:
                results["kept"].append({
                    "class": cls_name,
                    "confidence": conf,
                    "frame_idx": frame_data["frame_idx"]
                })
                results["stats_by_class"][cls_name]["kept"] += 1
            else:
                results["filtered"].append({
                    "class": cls_name,
                    "confidence": conf,
                    "frame_idx": frame_data["frame_idx"]
                })
                results["stats_by_class"][cls_name]["filtered"] += 1
    
    total = len(results["kept"]) + len(results["filtered"])
    results["stats"] = {
        "total_detections": total,
        "kept": len(results["kept"]),
        "filtered": len(results["filtered"]),
        "filter_rate": len(results["filtered"]) / total if total > 0 else 0
    }
    results["stats_by_class"] = dict(results["stats_by_class"])
    
    return results


def evaluate_vlm_filtering(video_path: str, base_detections: list, sample_limit: int = 50):
    """Evaluate VLM-based filtering using CLIP for verification."""
    import torch
    from transformers import CLIPProcessor, CLIPModel
    
    logger.info("\n" + "="*60)
    logger.info("CLIP-BASED VLM FILTERING EVALUATION")
    logger.info("="*60)
    
    device = get_device()
    
    # Load CLIP model
    logger.info("Loading CLIP model...")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = clip_model.to(device)
    clip_model.eval()
    logger.info(f"CLIP loaded on {device}")
    
    cap = cv2.VideoCapture(video_path)
    
    results = {
        "method": "vlm_clip",
        "kept": [],
        "filtered": [],
        "stats_by_class": defaultdict(lambda: {"kept": 0, "filtered": 0, "verified": 0, "rejected": 0}),
        "timing": []
    }
    
    # Only verify suspicious detections
    suspicious_detections = []
    for frame_data in base_detections:
        for det in frame_data.get("detections", []):
            cls_name = det["class"]
            conf = det["confidence"]
            
            # Only verify if suspicious or low confidence
            if cls_name in SUSPICIOUS_LABELS or conf < 0.45:
                suspicious_detections.append({
                    **det,
                    "frame_idx": frame_data["frame_idx"]
                })
            else:
                # Auto-keep high confidence reliable classes
                results["kept"].append({
                    "class": cls_name,
                    "confidence": conf,
                    "frame_idx": frame_data["frame_idx"],
                    "verified": False
                })
                results["stats_by_class"][cls_name]["kept"] += 1
    
    # Sample if too many
    if len(suspicious_detections) > sample_limit:
        np.random.seed(42)
        suspicious_detections = list(np.random.choice(suspicious_detections, sample_limit, replace=False))
    
    logger.info(f"Verifying {len(suspicious_detections)} suspicious detections with CLIP...")
    
    for i, det in enumerate(suspicious_detections):
        frame_idx = det["frame_idx"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        x1, y1, x2, y2 = map(int, det["bbox"])
        crop = frame[max(0,y1):y2, max(0,x1):x2]
        if crop.size == 0:
            continue
        
        start = time.time()
        
        try:
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            cls_name = det["class"]
            
            # Create positive and negative prompts
            positive_prompts = [
                f"a photo of a {cls_name}",
                f"a {cls_name}",
                f"an image containing a {cls_name}"
            ]
            negative_prompts = [
                "an empty image",
                "background",
                "nothing specific"
            ]
            
            all_prompts = positive_prompts + negative_prompts
            
            # Run CLIP
            inputs = clip_processor(text=all_prompts, images=crop_pil, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            
            # Aggregate positive vs negative
            positive_score = float(probs[0, :len(positive_prompts)].sum())
            negative_score = float(probs[0, len(positive_prompts):].sum())
            
            elapsed = time.time() - start
            results["timing"].append(elapsed)
            
            # Verification decision
            verified = positive_score > 0.6 and positive_score > negative_score * 1.5
            
            detection_result = {
                "class": cls_name,
                "confidence": det["confidence"],
                "frame_idx": frame_idx,
                "clip_positive_score": positive_score,
                "clip_negative_score": negative_score,
                "verified": verified
            }
            
            if verified:
                results["kept"].append(detection_result)
                results["stats_by_class"][cls_name]["kept"] += 1
                results["stats_by_class"][cls_name]["verified"] += 1
            else:
                results["filtered"].append(detection_result)
                results["stats_by_class"][cls_name]["filtered"] += 1
                results["stats_by_class"][cls_name]["rejected"] += 1
            
        except Exception as e:
            logger.debug(f"CLIP verification error: {e}")
            continue
        
        if (i + 1) % 10 == 0:
            logger.info(f"  Processed {i+1}/{len(suspicious_detections)} verifications...")
    
    cap.release()
    
    total = len(results["kept"]) + len(results["filtered"])
    results["stats"] = {
        "total_detections": total,
        "kept": len(results["kept"]),
        "filtered": len(results["filtered"]),
        "filter_rate": len(results["filtered"]) / total if total > 0 else 0,
        "avg_verification_time": float(np.mean(results["timing"])) if results["timing"] else 0,
        "suspicious_verified": len(suspicious_detections)
    }
    results["stats_by_class"] = dict(results["stats_by_class"])
    
    return results


def compare_filtering_methods(results_list: list):
    """Compare filtering results across methods."""
    logger.info("\n" + "="*70)
    logger.info("FILTERING METHOD COMPARISON")
    logger.info("="*70)
    
    print("\n{:<25} {:>12} {:>12} {:>12} {:>12}".format(
        "Method", "Total", "Kept", "Filtered", "Filter Rate"))
    print("-" * 75)
    
    for r in results_list:
        stats = r["stats"]
        print("{:<25} {:>12} {:>12} {:>12} {:>11.1%}".format(
            r["method"],
            stats["total_detections"],
            stats["kept"],
            stats["filtered"],
            stats["filter_rate"]
        ))
    
    # Class-by-class comparison
    print("\n\nCLASS-LEVEL FILTERING ANALYSIS:")
    print("-" * 80)
    
    all_classes = set()
    for r in results_list:
        all_classes.update(r["stats_by_class"].keys())
    
    print("{:<20} ".format("Class") + " ".join(["{:>20}".format(r["method"][:20]) for r in results_list]))
    print("-" * (20 + 21 * len(results_list)))
    
    for cls_name in sorted(all_classes):
        counts = []
        for r in results_list:
            stats = r["stats_by_class"].get(cls_name, {"kept": 0, "filtered": 0})
            kept = stats["kept"]
            filtered = stats["filtered"]
            counts.append(f"{kept}k/{filtered}f")
        
        print("{:<20} ".format(cls_name[:20]) + " ".join(["{:>20}".format(c) for c in counts]))
    
    # Analysis of what each method catches
    print("\n\nKEY INSIGHTS:")
    print("-" * 70)
    
    # Find classes where methods differ significantly
    for cls_name in all_classes:
        filter_rates = []
        for r in results_list:
            stats = r["stats_by_class"].get(cls_name, {"kept": 0, "filtered": 0})
            total = stats["kept"] + stats["filtered"]
            if total > 0:
                filter_rates.append((r["method"], stats["filtered"] / total))
            else:
                filter_rates.append((r["method"], 0))
        
        if len(filter_rates) > 1:
            rates = [fr[1] for fr in filter_rates]
            if max(rates) - min(rates) > 0.2:  # Significant difference
                print(f"  '{cls_name}' shows method variance:")
                for method, rate in filter_rates:
                    print(f"    {method}: {rate:.1%} filtered")
    
    return {"methods": [r["method"] for r in results_list]}


def main():
    parser = argparse.ArgumentParser(description="Semantic Filtering Evaluation")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--detections", help="Path to base detections JSON")
    parser.add_argument("--output-dir", default="results/filtering_eval", help="Output directory")
    parser.add_argument("--skip-vlm", action="store_true", help="Skip VLM-based filtering")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load base detections
    if args.detections and Path(args.detections).exists():
        logger.info(f"Loading base detections from {args.detections}")
        with open(args.detections) as f:
            base_data = json.load(f)
            base_detections = base_data.get("frames", [])
    else:
        logger.error("Base detections required. Run eval_detections_simple.py first.")
        return
    
    logger.info(f"Loaded {len(base_detections)} frames with detections")
    
    all_results = []
    
    # 1. Confidence-only baseline (0.35)
    conf_results = evaluate_confidence_only_filtering(base_detections, threshold=0.35)
    all_results.append(conf_results)
    with open(output_dir / "confidence_filtering.json", "w") as f:
        json.dump(conf_results, f, indent=2)
    
    # 2. Threshold-based filtering
    threshold_results = evaluate_threshold_filtering(base_detections)
    all_results.append(threshold_results)
    with open(output_dir / "threshold_filtering.json", "w") as f:
        json.dump(threshold_results, f, indent=2)
    
    # 3. VLM-based filtering (optional)
    if not args.skip_vlm:
        try:
            vlm_results = evaluate_vlm_filtering(args.video, base_detections)
            all_results.append(vlm_results)
            with open(output_dir / "vlm_filtering.json", "w") as f:
                json.dump(vlm_results, f, indent=2)
        except Exception as e:
            logger.error(f"VLM filtering failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Compare all methods
    if len(all_results) > 1:
        compare_filtering_methods(all_results)
    
    logger.info("\n" + "="*70)
    logger.info("FILTERING EVALUATION COMPLETE")
    logger.info(f"Results saved to {output_dir}")
    logger.info("="*70)


if __name__ == "__main__":
    main()
