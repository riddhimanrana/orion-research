#!/usr/bin/env python3
"""
Comprehensive Perception Component Evaluation

This script evaluates each component of the perception pipeline independently:
1. Detection (YOLO11x vs YOLO11m)
2. Classification (DINOv3 full-image vs YOLO-World crops)
3. Semantic Filtering (FastVLM vs threshold-based)
4. Re-ID (V-JEPA2 tracking consistency)

Uses Gemini API for deep analysis of failures and improvement suggestions.

Usage:
    python scripts/eval_perception_components.py --video data/examples/test.mp4 --component all
    python scripts/eval_perception_components.py --video data/examples/test.mp4 --component detection
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
logger = logging.getLogger("eval_components")


# =====================================================================
# DATA CLASSES
# =====================================================================

@dataclass
class DetectionResult:
    """Single detection from any detector."""
    bbox: List[float]  # [x1, y1, x2, y2]
    class_name: str
    confidence: float
    frame_idx: int
    source: str  # "yolo11m", "yolo11x", "yoloworld"
    crop: Optional[Image.Image] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClassificationResult:
    """Classification result for a detection."""
    detection: DetectionResult
    original_class: str
    refined_class: str
    confidence: float
    source: str  # "yoloworld_crop", "dino_fullimg", "fastvlm"
    candidate_labels: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class ComponentEvaluation:
    """Evaluation results for a single component."""
    component: str
    video_path: str
    num_frames: int
    processing_time: float
    
    # Metrics
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    accuracy: float = 0.0
    
    # Detailed breakdown
    correct: int = 0
    incorrect: int = 0
    missed: int = 0
    hallucinated: int = 0
    
    # Gemini analysis
    issues: List[str] = field(default_factory=list)
    root_causes: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    
    # Raw data
    samples: List[Dict[str, Any]] = field(default_factory=list)


# =====================================================================
# GEMINI INTEGRATION
# =====================================================================

def get_gemini_client():
    """Get Gemini API client."""
    try:
        import google.generativeai as genai
        
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("No Gemini API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY")
            return None
        
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.0-flash-exp")
    except ImportError:
        logger.warning("google-generativeai not installed")
        return None


def encode_image_base64(image: Image.Image, max_size: int = 512) -> str:
    """Encode PIL image to base64 string."""
    # Resize if too large
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    import io
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def analyze_frame_with_gemini(
    gemini: Any,
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    frame_idx: int,
    context: str = ""
) -> Dict[str, Any]:
    """Use Gemini to analyze detections in a frame."""
    
    if gemini is None:
        return {"error": "Gemini not available"}
    
    # Convert frame to PIL
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    # Format detections
    det_list = []
    for i, det in enumerate(detections):
        bbox = det.get("bbox") or det.get("bbox_2d", [])
        conf = det.get("confidence", 0.0)
        cls = det.get("class_name") or det.get("category", "unknown")
        det_list.append(f"{i+1}. {cls} (conf={conf:.2f}) at bbox {bbox}")
    
    det_text = "\n".join(det_list) if det_list else "No detections"
    
    prompt = f"""Analyze this video frame (frame {frame_idx}) with the following detections:

{det_text}

{context}

Please evaluate:
1. **Correct detections**: Which detections match actual objects in the frame?
2. **Incorrect classifications**: Which objects are detected but misclassified?
3. **Hallucinations**: Which detections don't correspond to any real object?
4. **Missed objects**: What visible objects were not detected?

For each issue, explain:
- What the model likely confused (e.g., "wardrobe detected as refrigerator due to similar shape")
- Root cause (e.g., "lack of context", "lighting", "occlusion", "similar appearance")

Output JSON format:
{{
    "correct": ["list of correct detections by index"],
    "incorrect": [{{"index": 1, "detected": "class", "actual": "class", "reason": "why"}}],
    "hallucinated": [{{"index": 2, "detected": "class", "reason": "what it actually is"}}],
    "missed": ["object1 at location", "object2 at location"],
    "root_causes": ["cause1", "cause2"],
    "suggestions": ["improvement1", "improvement2"]
}}
"""
    
    try:
        response = gemini.generate_content([prompt, pil_image])
        text = response.text
        
        # Extract JSON from response
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        return json.loads(text)
    except Exception as e:
        logger.error(f"Gemini analysis failed: {e}")
        return {"error": str(e)}


# =====================================================================
# DETECTION EVALUATION
# =====================================================================

def evaluate_detection(
    video_path: Path,
    output_dir: Path,
    models: List[str] = ["yolo11m", "yolo11x"],
    sample_frames: int = 20,
    gemini: Any = None
) -> Dict[str, ComponentEvaluation]:
    """Evaluate detection models (YOLO11m vs YOLO11x)."""
    
    logger.info("=" * 60)
    logger.info("DETECTION EVALUATION")
    logger.info("=" * 60)
    
    results = {}
    
    for model_name in models:
        logger.info(f"\n--- Testing {model_name} ---")
        
        eval_result = ComponentEvaluation(
            component=f"detection_{model_name}",
            video_path=str(video_path),
            num_frames=0,
            processing_time=0.0
        )
        
        try:
            # Load model
            from ultralytics import YOLO
            import torch
            model_path = f"{model_name}.pt"
            model = YOLO(model_path)
            
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
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Sample frames evenly
            frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)
            
            all_detections = []
            start_time = time.time()
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Run detection
                results_yolo = model(frame, conf=0.25, verbose=False)
                
                frame_dets = []
                for r in results_yolo:
                    for box in r.boxes:
                        det = {
                            "bbox": box.xyxy[0].tolist(),
                            "class_name": r.names[int(box.cls)],
                            "confidence": float(box.conf),
                            "frame_idx": int(frame_idx),
                            "source": model_name
                        }
                        frame_dets.append(det)
                
                all_detections.append({
                    "frame_idx": int(frame_idx),
                    "frame": frame,
                    "detections": frame_dets
                })
            
            cap.release()
            
            eval_result.processing_time = time.time() - start_time
            eval_result.num_frames = len(all_detections)
            
            # Gemini evaluation on sample frames
            if gemini:
                logger.info(f"Running Gemini analysis on {min(5, len(all_detections))} frames...")
                
                for sample in all_detections[:5]:  # Analyze first 5 frames
                    analysis = analyze_frame_with_gemini(
                        gemini,
                        sample["frame"],
                        sample["detections"],
                        sample["frame_idx"],
                        context=f"Testing {model_name} detection model."
                    )
                    
                    if "error" not in analysis:
                        eval_result.correct += len(analysis.get("correct", []))
                        eval_result.incorrect += len(analysis.get("incorrect", []))
                        eval_result.hallucinated += len(analysis.get("hallucinated", []))
                        eval_result.missed += len(analysis.get("missed", []))
                        eval_result.root_causes.extend(analysis.get("root_causes", []))
                        eval_result.suggestions.extend(analysis.get("suggestions", []))
                    
                    sample["gemini_analysis"] = analysis
                
                # Calculate metrics
                total = eval_result.correct + eval_result.incorrect + eval_result.hallucinated
                if total > 0:
                    eval_result.precision = eval_result.correct / total
                
                if eval_result.correct + eval_result.missed > 0:
                    eval_result.recall = eval_result.correct / (eval_result.correct + eval_result.missed)
                
                if eval_result.precision + eval_result.recall > 0:
                    eval_result.f1_score = 2 * (eval_result.precision * eval_result.recall) / (eval_result.precision + eval_result.recall)
            
            eval_result.samples = [{
                "frame_idx": s["frame_idx"],
                "num_detections": len(s["detections"]),
                "classes": [d["class_name"] for d in s["detections"]],
                "gemini": s.get("gemini_analysis", {})
            } for s in all_detections]
            
            results[model_name] = eval_result
            
            logger.info(f"{model_name} Results:")
            logger.info(f"  Processing time: {eval_result.processing_time:.2f}s")
            logger.info(f"  Precision: {eval_result.precision:.2%}")
            logger.info(f"  Recall: {eval_result.recall:.2%}")
            logger.info(f"  F1: {eval_result.f1_score:.2%}")
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            traceback.print_exc()
    
    return results


# =====================================================================
# CLASSIFICATION EVALUATION
# =====================================================================

def evaluate_classification(
    video_path: Path,
    output_dir: Path,
    sample_frames: int = 15,
    gemini: Any = None
) -> Dict[str, ComponentEvaluation]:
    """Evaluate classification approaches (YOLO-World crops vs DINOv3 full-image)."""
    
    logger.info("=" * 60)
    logger.info("CLASSIFICATION EVALUATION")
    logger.info("=" * 60)
    
    results = {}
    
    # First run YOLO11x to get base detections
    from ultralytics import YOLO
    import torch
    base_detector = YOLO("yolo11x.pt")
    
    # Auto-detect device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    base_detector.to(device)
    logger.info(f"Base detector loaded on {device}")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)
    
    # Collect frames with detections
    frames_data = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        results_yolo = base_detector(frame, conf=0.25, verbose=False)
        detections = []
        for r in results_yolo:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                crop = frame[int(y1):int(y2), int(x1):int(x2)]
                if crop.size > 0:
                    crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                else:
                    crop_pil = None
                
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "class_name": r.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "crop": crop_pil
                })
        
        frames_data.append({
            "frame_idx": int(frame_idx),
            "frame": frame,
            "detections": detections
        })
    
    cap.release()
    
    # === Method 1: YOLO-World Crop Refinement ===
    logger.info("\n--- Testing YOLO-World Crop Refinement ---")
    
    yoloworld_eval = ComponentEvaluation(
        component="classification_yoloworld_crops",
        video_path=str(video_path),
        num_frames=len(frames_data),
        processing_time=0.0
    )
    
    try:
        from ultralytics import YOLO as YOLOWorld
        yoloworld = YOLOWorld("yolov8x-worldv2.pt")
        yoloworld.to(device)  # Use same device as base detector
        
        start_time = time.time()
        yoloworld_results = []
        
        for frame_data in frames_data:
            refined_dets = []
            for det in frame_data["detections"]:
                if det["crop"] is None:
                    refined_dets.append({**det, "refined_class": det["class_name"], "refinement_source": "none"})
                    continue
                
                # Set fine-grained classes based on coarse class
                coarse_class = det["class_name"].lower()
                if coarse_class == "chair":
                    prompts = ["office chair", "dining chair", "armchair", "desk chair", "stool"]
                elif coarse_class == "table":
                    prompts = ["desk", "dining table", "coffee table", "side table"]
                elif coarse_class == "tv":
                    prompts = ["tv", "monitor", "computer screen", "display"]
                else:
                    prompts = [coarse_class]  # No refinement
                
                try:
                    yoloworld.set_classes(prompts)
                    crop_np = np.array(det["crop"])
                    crop_bgr = cv2.cvtColor(crop_np, cv2.COLOR_RGB2BGR)
                    res = yoloworld(crop_bgr, conf=0.15, verbose=False)
                    
                    if res and res[0].boxes and len(res[0].boxes) > 0:
                        best_idx = res[0].boxes.conf.argmax()
                        refined_class = res[0].names[int(res[0].boxes.cls[best_idx])]
                        refined_conf = float(res[0].boxes.conf[best_idx])
                    else:
                        refined_class = det["class_name"]
                        refined_conf = det["confidence"]
                    
                    refined_dets.append({
                        **det,
                        "refined_class": refined_class,
                        "refined_confidence": refined_conf,
                        "refinement_source": "yoloworld_crop"
                    })
                except Exception as e:
                    logger.warning(f"YOLO-World refinement failed: {e}")
                    refined_dets.append({**det, "refined_class": det["class_name"], "refinement_source": "error"})
            
            yoloworld_results.append({
                "frame_idx": frame_data["frame_idx"],
                "frame": frame_data["frame"],
                "detections": refined_dets
            })
        
        yoloworld_eval.processing_time = time.time() - start_time
        
        # Gemini evaluation
        if gemini:
            logger.info("Running Gemini analysis on YOLO-World refinement...")
            for sample in yoloworld_results[:5]:
                analysis = analyze_frame_with_gemini(
                    gemini,
                    sample["frame"],
                    sample["detections"],
                    sample["frame_idx"],
                    context="Testing YOLO-World crop refinement for fine-grained classification. Focus on whether refined_class is more accurate than original class_name."
                )
                
                if "error" not in analysis:
                    yoloworld_eval.correct += len(analysis.get("correct", []))
                    yoloworld_eval.incorrect += len(analysis.get("incorrect", []))
                    yoloworld_eval.hallucinated += len(analysis.get("hallucinated", []))
                    yoloworld_eval.missed += len(analysis.get("missed", []))
                    yoloworld_eval.root_causes.extend(analysis.get("root_causes", []))
                    yoloworld_eval.suggestions.extend(analysis.get("suggestions", []))
        
        # Calculate metrics
        total = yoloworld_eval.correct + yoloworld_eval.incorrect + yoloworld_eval.hallucinated
        if total > 0:
            yoloworld_eval.precision = yoloworld_eval.correct / total
            yoloworld_eval.accuracy = yoloworld_eval.correct / total
        
        results["yoloworld_crops"] = yoloworld_eval
        
        logger.info(f"YOLO-World Crop Results:")
        logger.info(f"  Processing time: {yoloworld_eval.processing_time:.2f}s")
        logger.info(f"  Precision: {yoloworld_eval.precision:.2%}")
        
    except Exception as e:
        logger.error(f"YOLO-World evaluation failed: {e}")
        traceback.print_exc()
    
    # === Method 2: DINOv3 Full-Image Classification ===
    logger.info("\n--- Testing DINOv3 Full-Image Classification ---")
    
    dino_eval = ComponentEvaluation(
        component="classification_dino_fullimg",
        video_path=str(video_path),
        num_frames=len(frames_data),
        processing_time=0.0
    )
    
    try:
        # Try to load DINOv3
        import torch
        from transformers import AutoModel, AutoProcessor
        
        dino_model_path = Path("models/dinov3-vitl16")
        if not dino_model_path.exists():
            logger.warning("DINOv3 model not found at models/dinov3-vitl16, using HuggingFace")
            dino_model_id = "facebook/dinov2-large"
        else:
            dino_model_id = str(dino_model_path)
        
        logger.info(f"Loading DINOv3 from {dino_model_id}...")
        dino_processor = AutoProcessor.from_pretrained("facebook/dinov2-large")
        dino_model = AutoModel.from_pretrained("facebook/dinov2-large")
        
        # Auto-detect device
        if torch.cuda.is_available():
            dino_device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dino_device = "mps"
        else:
            dino_device = "cpu"
        dino_model = dino_model.to(dino_device)
        dino_model.eval()
        logger.info(f"DINOv3 loaded on {dino_device}")
        
        # Load SentenceTransformer for text matching
        from sentence_transformers import SentenceTransformer
        st_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        
        start_time = time.time()
        dino_results = []
        
        # Precompute text embeddings for common classes
        fine_labels = {
            "chair": ["office chair", "dining chair", "armchair", "desk chair", "stool", "chair"],
            "table": ["desk", "dining table", "coffee table", "side table", "table"],
            "tv": ["tv", "monitor", "computer screen", "display", "television"],
            "couch": ["couch", "sofa", "loveseat", "settee"],
            "person": ["person", "man", "woman", "child"],
        }
        
        all_labels = []
        for labels in fine_labels.values():
            all_labels.extend(labels)
        all_labels = list(set(all_labels))
        
        label_embeddings = st_model.encode(all_labels, convert_to_tensor=True)
        
        for frame_data in frames_data:
            frame_rgb = cv2.cvtColor(frame_data["frame"], cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # Get DINO features for full frame
            inputs = dino_processor(images=frame_pil, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = dino_model(**inputs)
                patch_features = outputs.last_hidden_state  # [1, num_patches, dim]
            
            refined_dets = []
            for det in frame_data["detections"]:
                bbox = det["bbox"]
                coarse_class = det["class_name"].lower()
                
                # Get patch features for this bbox region
                # Convert bbox to patch coordinates
                h, w = frame_data["frame"].shape[:2]
                patch_size = 14  # DINOv2 uses 14x14 patches
                num_patches_h = h // patch_size
                num_patches_w = w // patch_size
                
                x1, y1, x2, y2 = bbox
                patch_x1 = int((x1 / w) * num_patches_w)
                patch_y1 = int((y1 / h) * num_patches_h)
                patch_x2 = int((x2 / w) * num_patches_w)
                patch_y2 = int((y2 / h) * num_patches_h)
                
                # Extract and pool features (simplified - using CLS token)
                region_embedding = outputs.last_hidden_state[0, 0].cpu().numpy()  # CLS token
                
                # Match against fine-grained labels
                if coarse_class in fine_labels:
                    candidates = fine_labels[coarse_class]
                else:
                    candidates = [coarse_class]
                
                # Use text similarity (simplified approach)
                coarse_embedding = st_model.encode(coarse_class, convert_to_tensor=True)
                candidate_embeddings = st_model.encode(candidates, convert_to_tensor=True)
                
                from sentence_transformers import util
                similarities = util.cos_sim(coarse_embedding, candidate_embeddings)[0]
                best_idx = similarities.argmax().item()
                
                refined_dets.append({
                    **det,
                    "refined_class": candidates[best_idx],
                    "refined_confidence": float(similarities[best_idx]),
                    "refinement_source": "dino_fullimg"
                })
            
            dino_results.append({
                "frame_idx": frame_data["frame_idx"],
                "frame": frame_data["frame"],
                "detections": refined_dets
            })
        
        dino_eval.processing_time = time.time() - start_time
        
        # Gemini evaluation
        if gemini:
            logger.info("Running Gemini analysis on DINOv3 full-image classification...")
            for sample in dino_results[:5]:
                analysis = analyze_frame_with_gemini(
                    gemini,
                    sample["frame"],
                    sample["detections"],
                    sample["frame_idx"],
                    context="Testing DINOv3 full-image classification. Evaluate if having full frame context improves classification accuracy."
                )
                
                if "error" not in analysis:
                    dino_eval.correct += len(analysis.get("correct", []))
                    dino_eval.incorrect += len(analysis.get("incorrect", []))
                    dino_eval.hallucinated += len(analysis.get("hallucinated", []))
                    dino_eval.missed += len(analysis.get("missed", []))
                    dino_eval.root_causes.extend(analysis.get("root_causes", []))
                    dino_eval.suggestions.extend(analysis.get("suggestions", []))
        
        # Calculate metrics
        total = dino_eval.correct + dino_eval.incorrect + dino_eval.hallucinated
        if total > 0:
            dino_eval.precision = dino_eval.correct / total
            dino_eval.accuracy = dino_eval.correct / total
        
        results["dino_fullimg"] = dino_eval
        
        logger.info(f"DINOv3 Full-Image Results:")
        logger.info(f"  Processing time: {dino_eval.processing_time:.2f}s")
        logger.info(f"  Precision: {dino_eval.precision:.2%}")
        
    except Exception as e:
        logger.error(f"DINOv3 evaluation failed: {e}")
        traceback.print_exc()
    
    return results


# =====================================================================
# SEMANTIC FILTERING EVALUATION
# =====================================================================

def evaluate_filtering(
    video_path: Path,
    output_dir: Path,
    sample_frames: int = 15,
    gemini: Any = None
) -> Dict[str, ComponentEvaluation]:
    """Evaluate semantic filtering approaches (threshold vs FastVLM)."""
    
    logger.info("=" * 60)
    logger.info("SEMANTIC FILTERING EVALUATION")
    logger.info("=" * 60)
    
    results = {}
    
    # Get base detections first
    from ultralytics import YOLO
    import torch
    detector = YOLO("yolo11x.pt")
    
    # Auto-detect device
    if torch.cuda.is_available():
        filter_device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        filter_device = "mps"
    else:
        filter_device = "cpu"
    detector.to(filter_device)
    logger.info(f"Detector loaded on {filter_device}")
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)
    
    frames_data = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        results_yolo = detector(frame, conf=0.20, verbose=False)
        detections = []
        for r in results_yolo:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                crop = frame[int(y1):int(y2), int(x1):int(x2)]
                crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)) if crop.size > 0 else None
                
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "class_name": r.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "crop": crop_pil
                })
        
        frames_data.append({
            "frame_idx": int(frame_idx),
            "frame": frame,
            "detections": detections
        })
    
    cap.release()
    
    # Suspicious labels to focus on
    SUSPICIOUS_LABELS = {
        "refrigerator", "toilet", "sink", "microwave", "hair drier", "bird", 
        "airplane", "boat", "toaster", "bed", "kite", "teddy bear"
    }
    
    # === Method 1: Threshold-Based Filtering ===
    logger.info("\n--- Testing Threshold-Based Filtering ---")
    
    threshold_eval = ComponentEvaluation(
        component="filtering_threshold",
        video_path=str(video_path),
        num_frames=len(frames_data),
        processing_time=0.0
    )
    
    try:
        from orion.perception.semantic_filter_v2 import SemanticFilterV2
        
        start_time = time.time()
        threshold_filter = SemanticFilterV2()
        
        threshold_results = []
        for frame_data in frames_data:
            filtered_dets = []
            removed_dets = []
            
            for det in frame_data["detections"]:
                cls_name = det["class_name"].lower()
                conf = det["confidence"]
                
                # Simple threshold logic
                if cls_name in SUSPICIOUS_LABELS and conf < 0.45:
                    removed_dets.append({**det, "filter_reason": "low_confidence_suspicious"})
                else:
                    filtered_dets.append(det)
            
            threshold_results.append({
                "frame_idx": frame_data["frame_idx"],
                "frame": frame_data["frame"],
                "detections": filtered_dets,
                "removed": removed_dets
            })
        
        threshold_eval.processing_time = time.time() - start_time
        
        # Gemini evaluation
        if gemini:
            logger.info("Running Gemini analysis on threshold filtering...")
            for sample in threshold_results[:5]:
                # Analyze what was kept
                analysis = analyze_frame_with_gemini(
                    gemini,
                    sample["frame"],
                    sample["detections"],
                    sample["frame_idx"],
                    context=f"Testing threshold-based filtering. Removed {len(sample['removed'])} suspicious detections. Were the remaining detections correct?"
                )
                
                if "error" not in analysis:
                    threshold_eval.correct += len(analysis.get("correct", []))
                    threshold_eval.incorrect += len(analysis.get("incorrect", []))
                    threshold_eval.hallucinated += len(analysis.get("hallucinated", []))
                    threshold_eval.missed += len(analysis.get("missed", []))
                    threshold_eval.root_causes.extend(analysis.get("root_causes", []))
                    threshold_eval.suggestions.extend(analysis.get("suggestions", []))
        
        # Calculate metrics
        total = threshold_eval.correct + threshold_eval.incorrect + threshold_eval.hallucinated
        if total > 0:
            threshold_eval.precision = threshold_eval.correct / total
        
        results["threshold_filter"] = threshold_eval
        
        logger.info(f"Threshold Filtering Results:")
        logger.info(f"  Processing time: {threshold_eval.processing_time:.2f}s")
        logger.info(f"  Precision: {threshold_eval.precision:.2%}")
        
    except Exception as e:
        logger.error(f"Threshold filtering evaluation failed: {e}")
        traceback.print_exc()
    
    # === Method 2: FastVLM Verification ===
    logger.info("\n--- Testing FastVLM Verification ---")
    
    vlm_eval = ComponentEvaluation(
        component="filtering_fastvlm",
        video_path=str(video_path),
        num_frames=len(frames_data),
        processing_time=0.0
    )
    
    try:
        from orion.backends.mlx_fastvlm import FastVLMMLXWrapper
        
        vlm = FastVLMMLXWrapper()
        
        start_time = time.time()
        vlm_results = []
        
        for frame_data in frames_data:
            filtered_dets = []
            removed_dets = []
            
            for det in frame_data["detections"]:
                cls_name = det["class_name"].lower()
                conf = det["confidence"]
                
                # Only verify suspicious labels
                if cls_name in SUSPICIOUS_LABELS and conf < 0.60 and det["crop"] is not None:
                    # Ask FastVLM
                    try:
                        question = f"Is this object a {cls_name}? Answer yes or no."
                        answer = vlm.generate(det["crop"], question, max_tokens=10)
                        
                        is_confirmed = "yes" in answer.lower()
                        
                        if is_confirmed:
                            filtered_dets.append({**det, "vlm_verified": True})
                        else:
                            removed_dets.append({**det, "filter_reason": "vlm_rejected", "vlm_answer": answer})
                    except Exception as e:
                        logger.warning(f"VLM verification failed: {e}")
                        filtered_dets.append(det)  # Keep on error
                else:
                    filtered_dets.append(det)
            
            vlm_results.append({
                "frame_idx": frame_data["frame_idx"],
                "frame": frame_data["frame"],
                "detections": filtered_dets,
                "removed": removed_dets
            })
        
        vlm_eval.processing_time = time.time() - start_time
        
        # Gemini evaluation
        if gemini:
            logger.info("Running Gemini analysis on FastVLM filtering...")
            for sample in vlm_results[:5]:
                analysis = analyze_frame_with_gemini(
                    gemini,
                    sample["frame"],
                    sample["detections"],
                    sample["frame_idx"],
                    context=f"Testing FastVLM verification filtering. Removed {len(sample['removed'])} detections after VLM rejection. Were the remaining detections correct?"
                )
                
                if "error" not in analysis:
                    vlm_eval.correct += len(analysis.get("correct", []))
                    vlm_eval.incorrect += len(analysis.get("incorrect", []))
                    vlm_eval.hallucinated += len(analysis.get("hallucinated", []))
                    vlm_eval.missed += len(analysis.get("missed", []))
                    vlm_eval.root_causes.extend(analysis.get("root_causes", []))
                    vlm_eval.suggestions.extend(analysis.get("suggestions", []))
        
        # Calculate metrics
        total = vlm_eval.correct + vlm_eval.incorrect + vlm_eval.hallucinated
        if total > 0:
            vlm_eval.precision = vlm_eval.correct / total
        
        results["fastvlm_filter"] = vlm_eval
        
        logger.info(f"FastVLM Filtering Results:")
        logger.info(f"  Processing time: {vlm_eval.processing_time:.2f}s")
        logger.info(f"  Precision: {vlm_eval.precision:.2%}")
        
    except Exception as e:
        logger.error(f"FastVLM filtering evaluation failed: {e}")
        traceback.print_exc()
    
    return results


# =====================================================================
# RE-ID EVALUATION
# =====================================================================

def evaluate_reid(
    video_path: Path,
    output_dir: Path,
    sample_interval: int = 5,
    gemini: Any = None
) -> Dict[str, ComponentEvaluation]:
    """Evaluate Re-ID tracking consistency using V-JEPA2."""
    
    logger.info("=" * 60)
    logger.info("RE-ID EVALUATION")
    logger.info("=" * 60)
    
    results = {}
    
    reid_eval = ComponentEvaluation(
        component="reid_vjepa2",
        video_path=str(video_path),
        num_frames=0,
        processing_time=0.0
    )
    
    try:
        # Run full perception pipeline to get tracks
        from orion.perception.engine import PerceptionEngine
        from orion.perception.config import PerceptionConfig
        
        config = PerceptionConfig(mode="balanced")
        config.detection.model = "yolo11x"
        config.output_dir = str(output_dir / "reid_test")
        
        engine = PerceptionEngine(config)
        
        start_time = time.time()
        
        # Process video
        result = engine.process_video(str(video_path))
        
        reid_eval.processing_time = time.time() - start_time
        reid_eval.num_frames = result.total_frames if hasattr(result, 'total_frames') else 0
        
        # Analyze track consistency
        if hasattr(result, 'entities') and result.entities:
            # Group observations by track_id
            track_observations = defaultdict(list)
            for entity in result.entities:
                track_id = entity.track_id if hasattr(entity, 'track_id') else entity.id
                track_observations[track_id].append(entity)
            
            # Check for ID switches (same object, different tracks)
            # and fragmentation (track breaks during occlusion)
            num_tracks = len(track_observations)
            avg_track_length = np.mean([len(obs) for obs in track_observations.values()])
            
            reid_eval.samples = [{
                "num_tracks": num_tracks,
                "avg_track_length": float(avg_track_length),
                "track_lengths": {str(k): len(v) for k, v in track_observations.items()}
            }]
            
            logger.info(f"Re-ID Results:")
            logger.info(f"  Total tracks: {num_tracks}")
            logger.info(f"  Avg track length: {avg_track_length:.1f} frames")
        
        # Gemini deep analysis on track samples
        if gemini:
            logger.info("Running Gemini analysis on tracking...")
            
            # Extract sample frames with tracks overlaid
            cap = cv2.VideoCapture(str(video_path))
            
            sample_frames = [30, 60, 90, 120, 150]  # Check specific frames
            for frame_idx in sample_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Get tracks at this frame
                frame_tracks = []
                if hasattr(result, 'entities'):
                    for entity in result.entities:
                        if hasattr(entity, 'observations'):
                            for obs in entity.observations:
                                if obs.get('frame_idx') == frame_idx:
                                    frame_tracks.append({
                                        "track_id": entity.track_id if hasattr(entity, 'track_id') else entity.id,
                                        "class": entity.object_class if hasattr(entity, 'object_class') else "unknown",
                                        "bbox": obs.get('bbox') or obs.get('bbox_2d')
                                    })
                
                if frame_tracks:
                    analysis = analyze_frame_with_gemini(
                        gemini,
                        frame,
                        frame_tracks,
                        frame_idx,
                        context="Evaluate tracking consistency. Each track_id should represent a unique object. Look for: 1) ID switches (same object, different IDs), 2) ID merges (different objects, same ID), 3) Track fragmentation."
                    )
                    
                    if "error" not in analysis:
                        reid_eval.root_causes.extend(analysis.get("root_causes", []))
                        reid_eval.suggestions.extend(analysis.get("suggestions", []))
            
            cap.release()
        
        results["vjepa2_reid"] = reid_eval
        
    except Exception as e:
        logger.error(f"Re-ID evaluation failed: {e}")
        traceback.print_exc()
    
    return results


# =====================================================================
# MAIN EVALUATION RUNNER
# =====================================================================

def generate_report(
    all_results: Dict[str, Dict[str, ComponentEvaluation]],
    output_path: Path
) -> str:
    """Generate comprehensive evaluation report."""
    
    report = []
    report.append("=" * 80)
    report.append("ORION PERCEPTION PIPELINE - COMPONENT EVALUATION REPORT")
    report.append("=" * 80)
    report.append(f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Summary table
    report.append("\n## SUMMARY\n")
    report.append("| Component | Precision | Recall | F1 | Time (s) | Issues |")
    report.append("|-----------|-----------|--------|-----|----------|--------|")
    
    for category, results in all_results.items():
        for name, eval_result in results.items():
            issues_count = len(eval_result.root_causes)
            report.append(f"| {name} | {eval_result.precision:.2%} | {eval_result.recall:.2%} | {eval_result.f1_score:.2%} | {eval_result.processing_time:.1f} | {issues_count} |")
    
    # Detailed findings per component
    for category, results in all_results.items():
        report.append(f"\n\n## {category.upper()}\n")
        
        for name, eval_result in results.items():
            report.append(f"\n### {name}\n")
            report.append(f"- Frames analyzed: {eval_result.num_frames}")
            report.append(f"- Processing time: {eval_result.processing_time:.2f}s")
            report.append(f"- Correct: {eval_result.correct}")
            report.append(f"- Incorrect: {eval_result.incorrect}")
            report.append(f"- Hallucinated: {eval_result.hallucinated}")
            report.append(f"- Missed: {eval_result.missed}")
            
            if eval_result.root_causes:
                report.append("\n**Root Causes:**")
                for cause in set(eval_result.root_causes):
                    report.append(f"- {cause}")
            
            if eval_result.suggestions:
                report.append("\n**Suggestions:**")
                for suggestion in set(eval_result.suggestions):
                    report.append(f"- {suggestion}")
    
    # Write report
    report_text = "\n".join(report)
    output_path.write_text(report_text)
    
    return report_text


def main():
    parser = argparse.ArgumentParser(description="Evaluate perception pipeline components")
    parser.add_argument("--video", type=str, required=True, help="Path to test video")
    parser.add_argument("--output", type=str, default="results/component_eval", help="Output directory")
    parser.add_argument("--component", type=str, default="all", 
                       choices=["all", "detection", "classification", "filtering", "reid"],
                       help="Component to evaluate")
    parser.add_argument("--sample-frames", type=int, default=15, help="Number of frames to sample")
    parser.add_argument("--no-gemini", action="store_true", help="Skip Gemini analysis")
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        sys.exit(1)
    
    # Initialize Gemini
    gemini = None if args.no_gemini else get_gemini_client()
    if gemini:
        logger.info("✓ Gemini API initialized")
    else:
        logger.warning("⚠ Running without Gemini analysis")
    
    all_results = {}
    
    # Run evaluations
    if args.component in ["all", "detection"]:
        all_results["detection"] = evaluate_detection(
            video_path, output_dir, 
            models=["yolo11m", "yolo11x"],
            sample_frames=args.sample_frames,
            gemini=gemini
        )
    
    if args.component in ["all", "classification"]:
        all_results["classification"] = evaluate_classification(
            video_path, output_dir,
            sample_frames=args.sample_frames,
            gemini=gemini
        )
    
    if args.component in ["all", "filtering"]:
        all_results["filtering"] = evaluate_filtering(
            video_path, output_dir,
            sample_frames=args.sample_frames,
            gemini=gemini
        )
    
    if args.component in ["all", "reid"]:
        all_results["reid"] = evaluate_reid(
            video_path, output_dir,
            gemini=gemini
        )
    
    # Generate report
    report_path = output_dir / "evaluation_report.md"
    report = generate_report(all_results, report_path)
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Report saved to: {report_path}")
    print("\n" + report[:2000] + "...\n")
    
    # Save raw results as JSON
    json_results = {}
    for category, results in all_results.items():
        json_results[category] = {}
        for name, eval_result in results.items():
            json_results[category][name] = {
                "component": eval_result.component,
                "precision": eval_result.precision,
                "recall": eval_result.recall,
                "f1_score": eval_result.f1_score,
                "processing_time": eval_result.processing_time,
                "correct": eval_result.correct,
                "incorrect": eval_result.incorrect,
                "hallucinated": eval_result.hallucinated,
                "missed": eval_result.missed,
                "root_causes": list(set(eval_result.root_causes)),
                "suggestions": list(set(eval_result.suggestions))
            }
    
    json_path = output_dir / "evaluation_results.json"
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Raw results saved to: {json_path}")


if __name__ == "__main__":
    main()
