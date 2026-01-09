#!/usr/bin/env python3
"""
Classification Evaluation - YOLO-World vs DINOv3
Compares crop-based classification (YOLO-World) vs full-image context (DINOv3)
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


LABEL_REFINEMENTS = {
    "couch": ["sofa", "loveseat", "sectional", "daybed", "futon"],
    "chair": ["office chair", "dining chair", "armchair", "stool", "recliner", "desk chair"],
    "potted plant": ["houseplant", "fern", "succulent", "orchid", "palm", "plant"],
    "tv": ["television", "monitor", "screen", "display", "smart tv"],
    "bed": ["mattress", "bedframe", "king bed", "queen bed", "twin bed"],
    "dining table": ["table", "desk", "coffee table", "kitchen table"],
    "bottle": ["water bottle", "wine bottle", "plastic bottle", "glass bottle"],
    "vase": ["flower vase", "decorative vase", "ceramic vase"],
    "book": ["books", "notebook", "magazine", "textbook"],
    "clock": ["wall clock", "digital clock", "alarm clock"],
    "person": ["man", "woman", "child", "human", "figure"],
}


def get_device():
    """Auto-detect best available device."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def evaluate_yoloworld(video_path: str, base_detections: list, sample_frames: set):
    """Evaluate YOLO-World classification on detection crops."""
    import torch
    from ultralytics import YOLO
    
    logger.info("\n" + "="*60)
    logger.info("YOLO-World Classification Evaluation")
    logger.info("="*60)
    
    device = get_device()
    
    yoloworld = YOLO("yolov8x-worldv2.pt")
    yoloworld.to(device)
    logger.info(f"YOLO-World loaded on {device}")
    
    cap = cv2.VideoCapture(video_path)
    
    results = {
        "method": "YOLO-World",
        "classifications": [],
        "timing": [],
        "class_distribution": defaultdict(int)
    }
    
    base_classes = set()
    for fd in base_detections:
        for det in fd.get("detections", []):
            base_classes.add(det["class"])
    
    vocab = list(base_classes)
    for cls in base_classes:
        if cls in LABEL_REFINEMENTS:
            vocab.extend(LABEL_REFINEMENTS[cls])
    vocab = list(set(vocab))
    
    logger.info(f"Vocabulary: {len(vocab)} classes")
    yoloworld.set_classes(vocab)
    
    processed = 0
    for frame_data in base_detections:
        frame_idx = frame_data["frame_idx"]
        if frame_idx not in sample_frames:
            continue
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        for det in frame_data.get("detections", []):
            x1, y1, x2, y2 = map(int, det["bbox"])
            crop = frame[max(0,y1):y2, max(0,x1):x2]
            if crop.size == 0:
                continue
            
            start = time.time()
            try:
                yw_results = yoloworld(crop, conf=0.1, verbose=False)
                elapsed = time.time() - start
                results["timing"].append(elapsed)
                
                refined_class = det["class"]
                refined_conf = det["confidence"]
                
                for r in yw_results:
                    if len(r.boxes) > 0:
                        best_idx = int(r.boxes.conf.argmax())
                        yw_class = yoloworld.names[int(r.boxes.cls[best_idx])]
                        yw_conf = float(r.boxes.conf[best_idx])
                        
                        if yw_conf > 0.3:
                            refined_class = yw_class
                            refined_conf = yw_conf
                
                results["classifications"].append({
                    "original_class": det["class"],
                    "original_conf": det["confidence"],
                    "refined_class": refined_class,
                    "refined_conf": refined_conf,
                    "bbox": det["bbox"],
                    "frame_idx": frame_idx
                })
                
                results["class_distribution"][refined_class] += 1
                
            except Exception as e:
                logger.debug(f"YOLO-World error: {e}")
                continue
        
        processed += 1
        if processed % 5 == 0:
            logger.info(f"  Processed {processed} frames...")
    
    cap.release()
    
    total = len(results["classifications"])
    results["stats"] = {
        "total_classifications": total,
        "avg_time_per_crop": float(np.mean(results["timing"])) if results["timing"] else 0,
        "unique_refined_classes": len(results["class_distribution"]),
        "refinement_rate": sum(1 for c in results["classifications"] if c["refined_class"] != c["original_class"]) / total if total > 0 else 0
    }
    
    results["class_distribution"] = dict(results["class_distribution"])
    return results


def evaluate_dino_classification(video_path: str, base_detections: list, sample_frames: set):
    """Evaluate DINOv3 with full-image context classification."""
    import torch
    from transformers import AutoModel, AutoProcessor
    from sentence_transformers import SentenceTransformer
    
    logger.info("\n" + "="*60)
    logger.info("DINOv2 + Semantic Classification Evaluation")
    logger.info("="*60)
    
    device = get_device()
    
    logger.info("Loading DINOv2-large...")
    dino_processor = AutoProcessor.from_pretrained("facebook/dinov2-large")
    dino_model = AutoModel.from_pretrained("facebook/dinov2-large")
    dino_model = dino_model.to(device)
    dino_model.eval()
    logger.info(f"DINOv2 loaded on {device}")
    
    logger.info("Loading SentenceTransformer...")
    st_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    base_classes = set()
    for fd in base_detections:
        for det in fd.get("detections", []):
            base_classes.add(det["class"])
    
    candidates = []
    for cls in base_classes:
        candidates.append(cls)
        if cls in LABEL_REFINEMENTS:
            candidates.extend(LABEL_REFINEMENTS[cls])
    candidates = list(set(candidates))
    
    logger.info(f"Encoding {len(candidates)} candidate labels...")
    label_embeddings = st_model.encode(candidates, convert_to_tensor=True)
    
    cap = cv2.VideoCapture(video_path)
    
    results = {
        "method": "DINOv2+Semantic",
        "classifications": [],
        "timing": [],
        "class_distribution": defaultdict(int)
    }
    
    processed = 0
    for frame_data in base_detections:
        frame_idx = frame_data["frame_idx"]
        if frame_idx not in sample_frames:
            continue
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        for det in frame_data.get("detections", []):
            x1, y1, x2, y2 = map(int, det["bbox"])
            crop = frame[max(0,y1):y2, max(0,x1):x2]
            if crop.size == 0:
                continue
            
            start = time.time()
            
            try:
                crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                
                inputs = dino_processor(images=crop_pil, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = dino_model(**inputs)
                    dino_embedding = outputs.last_hidden_state.mean(dim=1)
                
                description = f"A {det['class']} in a scene"
                desc_embedding = st_model.encode([description], convert_to_tensor=True)
                
                from torch.nn.functional import cosine_similarity
                similarities = cosine_similarity(desc_embedding, label_embeddings)
                best_idx = int(similarities.argmax().item())
                best_similarity = float(similarities[0, best_idx])
                
                elapsed = time.time() - start
                results["timing"].append(elapsed)
                
                refined_class = candidates[best_idx] if best_similarity > 0.5 else det["class"]
                
                top3_indices = similarities[0].topk(3).indices.tolist()
                results["classifications"].append({
                    "original_class": det["class"],
                    "original_conf": det["confidence"],
                    "refined_class": refined_class,
                    "refined_conf": best_similarity,
                    "bbox": det["bbox"],
                    "frame_idx": frame_idx,
                    "top3_candidates": [(candidates[i], float(similarities[0, i])) for i in top3_indices]
                })
                
                results["class_distribution"][refined_class] += 1
                
            except Exception as e:
                logger.debug(f"DINO classification error: {e}")
                continue
        
        processed += 1
        if processed % 5 == 0:
            logger.info(f"  Processed {processed} frames...")
    
    cap.release()
    
    total = len(results["classifications"])
    results["stats"] = {
        "total_classifications": total,
        "avg_time_per_crop": float(np.mean(results["timing"])) if results["timing"] else 0,
        "unique_refined_classes": len(results["class_distribution"]),
        "refinement_rate": sum(1 for c in results["classifications"] if c["refined_class"] != c["original_class"]) / total if total > 0 else 0
    }
    
    results["class_distribution"] = dict(results["class_distribution"])
    return results


def compare_classifications(yoloworld_results: dict, dino_results: dict):
    """Compare classification results between methods."""
    
    logger.info("\n" + "="*70)
    logger.info("CLASSIFICATION COMPARISON")
    logger.info("="*70)
    
    print("\n{:<25} {:>15} {:>15}".format("Metric", "YOLO-World", "DINOv2+Semantic"))
    print("-" * 60)
    
    yw_stats = yoloworld_results["stats"]
    dino_stats = dino_results["stats"]
    
    print("{:<25} {:>15} {:>15}".format("Total Classifications", yw_stats["total_classifications"], dino_stats["total_classifications"]))
    print("{:<25} {:>15.3f} {:>15.3f}".format("Avg Time/Crop (s)", yw_stats["avg_time_per_crop"], dino_stats["avg_time_per_crop"]))
    print("{:<25} {:>15} {:>15}".format("Unique Classes", yw_stats["unique_refined_classes"], dino_stats["unique_refined_classes"]))
    print("{:<25} {:>14.1%} {:>14.1%}".format("Refinement Rate", yw_stats["refinement_rate"], dino_stats["refinement_rate"]))
    
    print("\n\nCLASS DISTRIBUTION:")
    print("-" * 70)
    
    all_classes = set(yoloworld_results["class_distribution"].keys()) | set(dino_results["class_distribution"].keys())
    
    print("{:<30} {:>15} {:>15}".format("Class", "YOLO-World", "DINOv2"))
    print("-" * 60)
    
    for cls in sorted(all_classes):
        yw_count = yoloworld_results["class_distribution"].get(cls, 0)
        dino_count = dino_results["class_distribution"].get(cls, 0)
        if yw_count > 0 or dino_count > 0:
            print("{:<30} {:>15} {:>15}".format(cls[:30], yw_count, dino_count))
    
    print("\n\nREFINEMENT ANALYSIS:")
    print("-" * 70)
    
    yw_refined = sum(1 for c in yoloworld_results["classifications"] if c["refined_class"] != c["original_class"])
    dino_refined = sum(1 for c in dino_results["classifications"] if c["refined_class"] != c["original_class"])
    
    print(f"YOLO-World refined {yw_refined}/{len(yoloworld_results['classifications'])} detections")
    print(f"DINOv2+Semantic refined {dino_refined}/{len(dino_results['classifications'])} detections")
    
    print("\nExample YOLO-World Refinements:")
    shown = 0
    for c in yoloworld_results["classifications"]:
        if c["refined_class"] != c["original_class"] and shown < 5:
            print(f"  {c['original_class']} -> {c['refined_class']} (conf: {c['refined_conf']:.3f})")
            shown += 1
    
    print("\nExample DINOv2 Refinements:")
    shown = 0
    for c in dino_results["classifications"]:
        if c["refined_class"] != c["original_class"] and shown < 5:
            print(f"  {c['original_class']} -> {c['refined_class']} (conf: {c['refined_conf']:.3f})")
            if "top3_candidates" in c:
                print(f"    Top 3: {c['top3_candidates']}")
            shown += 1
    
    return {"yoloworld_stats": yw_stats, "dino_stats": dino_stats, "yoloworld_refinements": yw_refined, "dino_refinements": dino_refined}


def main():
    parser = argparse.ArgumentParser(description="Classification Evaluation: YOLO-World vs DINOv2")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--detections", help="Path to base detections JSON")
    parser.add_argument("--sample-frames", type=int, default=20, help="Number of frames to sample")
    parser.add_argument("--output-dir", default="results/classification_eval", help="Output directory")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.detections and Path(args.detections).exists():
        logger.info(f"Loading base detections from {args.detections}")
        with open(args.detections) as f:
            base_data = json.load(f)
            base_detections = base_data.get("frames", [])
    else:
        logger.info("Generating base detections with YOLO11x...")
        import torch
        from ultralytics import YOLO
        
        detector = YOLO("yolo11x.pt")
        device = get_device()
        detector.to(device)
        
        cap = cv2.VideoCapture(args.video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, args.sample_frames * 2, dtype=int)
        
        base_detections = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            results = detector(frame, conf=0.25, verbose=False)
            detections = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append({
                        "class": detector.names[int(box.cls[0])],
                        "confidence": float(box.conf[0]),
                        "bbox": [x1, y1, x2, y2]
                    })
            
            base_detections.append({"frame_idx": int(frame_idx), "detections": detections})
        
        cap.release()
        logger.info(f"Generated {len(base_detections)} frame detections")
    
    sample_frames = set(fd["frame_idx"] for fd in base_detections[:args.sample_frames])
    logger.info(f"Evaluating on {len(sample_frames)} frames")
    
    yoloworld_results = None
    dino_results = None
    
    try:
        yoloworld_results = evaluate_yoloworld(args.video, base_detections, sample_frames)
        with open(output_dir / "yoloworld_classification.json", "w") as f:
            json.dump(yoloworld_results, f, indent=2)
        logger.info(f"YOLO-World results saved")
    except Exception as e:
        logger.error(f"YOLO-World evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        dino_results = evaluate_dino_classification(args.video, base_detections, sample_frames)
        with open(output_dir / "dino_classification.json", "w") as f:
            json.dump(dino_results, f, indent=2)
        logger.info(f"DINOv2 results saved")
    except Exception as e:
        logger.error(f"DINOv2 evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    
    if yoloworld_results and dino_results:
        comparison = compare_classifications(yoloworld_results, dino_results)
        with open(output_dir / "comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)
    
    logger.info("\n" + "="*70)
    logger.info("CLASSIFICATION EVALUATION COMPLETE")
    logger.info(f"Results saved to {output_dir}")
    logger.info("="*70)


if __name__ == "__main__":
    main()
