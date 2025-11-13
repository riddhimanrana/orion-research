#!/usr/bin/env python3
"""
Diagnostic: Detection Quality Analysis
- Verify YOLO segmentation is working
- Compare detection thresholds
- Test yoloe-11s-seg-pf vs yolo11s-seg
- Compare with Gemini ground truth
- Identify missed objects and classification errors
"""

import cv2
import numpy as np
import os
from pathlib import Path
from collections import defaultdict
import google.generativeai as genai
from PIL import Image


def test_yolo_segmentation(video_path: str, frame_idx: int = 100):
    """Verify YOLO segmentation masks are working"""
    from ultralytics import YOLO
    
    print("="*80)
    print("DIAGNOSTIC 1: YOLO Segmentation Verification")
    print("="*80)
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"❌ Could not read frame {frame_idx}")
        return None
    
    # Test both models
    models_to_test = {
        'yolo11s-seg': 'yolo11s-seg.pt',
        'yoloe-11s-seg-pf': 'models/yoloe-11s-seg-pf.pt',
    }
    
    results_comparison = {}
    
    for model_name, model_path in models_to_test.items():
        if not Path(model_path).exists():
            print(f"\n⚠️  {model_name} not found at {model_path}, skipping")
            continue
        
        print(f"\n{'─'*80}")
        print(f"Testing: {model_name}")
        print(f"{'─'*80}")
        
        model = YOLO(model_path)
        
        # Test with different confidence thresholds
        thresholds = [0.25, 0.35, 0.50]
        
        for conf_threshold in thresholds:
            results = model(frame, conf=conf_threshold, verbose=False)
            
            if len(results) == 0 or results[0].boxes is None:
                print(f"  conf={conf_threshold}: No detections")
                continue
            
            boxes = results[0].boxes
            masks = results[0].masks
            
            num_detections = len(boxes)
            has_masks = masks is not None
            
            classes = boxes.cls.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            class_names = [model.names[int(c)] for c in classes]
            
            print(f"\n  Confidence threshold: {conf_threshold}")
            print(f"    Detections: {num_detections}")
            print(f"    Segmentation masks: {'✓ YES' if has_masks else '✗ NO'}")
            print(f"    Avg confidence: {np.mean(confs):.3f}")
            print(f"    Classes detected: {set(class_names)}")
            
            # Show per-object details
            for idx, (cls_name, conf) in enumerate(zip(class_names, confs)):
                mask_info = ""
                if has_masks and idx < len(masks):
                    mask = masks[idx].data.cpu().numpy()
                    mask_pixels = np.sum(mask > 0.5)
                    mask_info = f", mask_pixels={mask_pixels}"
                print(f"      [{idx}] {cls_name}: conf={conf:.3f}{mask_info}")
            
            # Store for comparison
            if conf_threshold not in results_comparison:
                results_comparison[conf_threshold] = {}
            results_comparison[conf_threshold][model_name] = {
                'num_detections': num_detections,
                'has_masks': has_masks,
                'avg_conf': float(np.mean(confs)),
                'classes': class_names,
            }
    
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    for threshold, models_data in sorted(results_comparison.items()):
        print(f"\nConfidence {threshold}:")
        for model_name, data in models_data.items():
            print(f"  {model_name:20s}: {data['num_detections']:2d} detections, "
                  f"masks={'✓' if data['has_masks'] else '✗'}, "
                  f"avg_conf={data['avg_conf']:.3f}")
    
    return results_comparison


def compare_with_gemini(video_path: str, frame_idx: int = 100, yolo_detections: list = None):
    """Compare YOLO detections with Gemini ground truth"""
    
    print("\n" + "="*80)
    print("DIAGNOSTIC 2: Gemini Ground Truth Comparison")
    print("="*80)
    
    # Get Gemini's view
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("⚠️  GOOGLE_API_KEY not set, skipping Gemini comparison")
        return None
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    # Convert to PIL
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    # Query Gemini with detailed prompt
    prompt = """Analyze this image and list ALL visible objects with high precision.

For each object, provide:
1. Object name (be specific: "laptop", "water bottle", "monitor", not just "electronics")
2. Location (precise: "top-left corner", "center-right", etc.)
3. Estimated size (small/medium/large)
4. Importance (1-5, where 5 is important personal item like laptop/phone, 1 is background clutter)

Format as JSON array:
[
  {"name": "laptop", "location": "center", "size": "large", "importance": 5},
  {"name": "water bottle", "location": "right", "size": "small", "importance": 4}
]

Be exhaustive - include everything you see, even small items."""
    
    try:
        response = model.generate_content([prompt, pil_image])
        gemini_text = response.text
        
        print(f"\nFrame {frame_idx} - Gemini Analysis:")
        print("─"*80)
        print(gemini_text)
        print("─"*80)
        
        # Try to extract object names (simple parsing)
        gemini_objects = []
        for line in gemini_text.lower().split('\n'):
            if '"name"' in line or 'object' in line:
                # Simple extraction - you'd use JSON parsing in production
                for word in ['laptop', 'monitor', 'keyboard', 'mouse', 'bottle', 'cup', 
                           'phone', 'book', 'pen', 'notebook', 'desk', 'chair',
                           'screen', 'computer', 'tablet', 'cable', 'charger']:
                    if word in line:
                        gemini_objects.append(word)
        
        gemini_objects = list(set(gemini_objects))
        
        print(f"\n📋 Gemini detected objects: {gemini_objects}")
        
        if yolo_detections:
            yolo_objects = list(set(yolo_detections))
            print(f"🔍 YOLO detected objects:   {yolo_objects}")
            
            # Find mismatches
            missed_by_yolo = [obj for obj in gemini_objects if obj not in ' '.join(yolo_objects)]
            false_positives = [obj for obj in yolo_objects if obj not in ' '.join(gemini_objects)]
            
            if missed_by_yolo:
                print(f"\n⚠️  Missed by YOLO: {missed_by_yolo}")
                print("   → Consider: lower confidence threshold, different model, or custom classifier")
            
            if false_positives:
                print(f"\n⚠️  False positives: {false_positives}")
                print("   → Consider: higher confidence threshold or post-filtering")
        
        return {
            'gemini_raw': gemini_text,
            'gemini_objects': gemini_objects,
            'yolo_objects': yolo_detections,
        }
        
    except Exception as e:
        print(f"❌ Gemini query failed: {e}")
        return None


def analyze_depth_context(video_path: str, frame_idx: int = 100):
    """Test depth estimation for object understanding"""
    
    print("\n" + "="*80)
    print("DIAGNOSTIC 3: Depth-Based Object Context (Planned)")
    print("="*80)
    
    print("\n⚠️  Depth model integration pending - will enable:")
    print("   - Real-world size estimation (mm)")
    print("   - Spatial zone clustering (desk, shelf, floor)")
    print("   - Classification validation (expected vs actual size)")
    print("   - SLAM-based persistent spatial memory")
    
    return None
    
    # TODO: Integrate depth model
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    print("\nEstimating depth...")
    depth_map = None  # depth_model.estimate_depth(frame)
    
    print(f"  Depth map shape: {depth_map.shape}")
    print(f"  Depth range: {depth_map.min():.2f} - {depth_map.max():.2f}")
    print(f"  Mean depth: {depth_map.mean():.2f}")
    
    # Load YOLO detections with depth context
    from ultralytics import YOLO
    yolo = YOLO("yolo11s-seg.pt")
    results = yolo(frame, conf=0.35, verbose=False)
    
    if len(results) == 0 or results[0].boxes is None:
        print("  No detections to analyze with depth")
        return None
    
    boxes = results[0].boxes
    xyxy = boxes.xyxy.cpu().numpy()
    classes = boxes.cls.cpu().numpy()
    class_names = [yolo.names[int(c)] for c in classes]
    
    print("\n📏 Object size estimation (using depth):")
    print("─"*80)
    
    # Known object sizes (COCO classes, in mm)
    TYPICAL_SIZES = {
        'laptop': {'width': 350, 'depth': 250},
        'keyboard': {'width': 450, 'depth': 150},
        'mouse': {'width': 120, 'depth': 70},
        'bottle': {'width': 70, 'height': 250},
        'cup': {'width': 80, 'height': 120},
        'cell phone': {'width': 70, 'height': 150},
        'book': {'width': 200, 'depth': 250},
        'monitor': {'width': 600, 'height': 400},
        'tv': {'width': 800, 'height': 500},
    }
    
    for idx, (box, cls_name) in enumerate(zip(xyxy, class_names)):
        x1, y1, x2, y2 = map(int, box)
        
        # Get median depth in bbox
        depth_roi = depth_map[y1:y2, x1:x2]
        median_depth = np.median(depth_roi) if depth_roi.size > 0 else 1.0
        
        # Estimate real-world size
        bbox_width_px = x2 - x1
        bbox_height_px = y2 - y1
        
        # Simple perspective scaling (rough estimate)
        # Real size ≈ pixel_size × depth / focal_length_estimate
        focal_estimate = 1000  # pixels (rough for typical camera)
        estimated_width_mm = (bbox_width_px * median_depth * 1000) / focal_estimate
        estimated_height_mm = (bbox_height_px * median_depth * 1000) / focal_estimate
        
        # Check if size matches expected
        expected = TYPICAL_SIZES.get(cls_name, {})
        size_match = "✓" if expected else "?"
        
        print(f"  [{idx}] {cls_name:15s}: depth={median_depth:.2f}m, "
              f"size={estimated_width_mm:.0f}×{estimated_height_mm:.0f}mm {size_match}")
        
        if expected:
            expected_str = f"{expected.get('width', '?')}×{expected.get('height', expected.get('depth', '?'))}mm"
            print(f"       Expected: {expected_str}")
    
    print("\n💡 Insights:")
    print("   - Use depth + size to validate classifications")
    print("   - Objects at similar depths might be in same 'zone' (desk, shelf, etc.)")
    print("   - Unusual sizes might indicate misclassification")
    
    return depth_map


def main():
    video_path = "data/examples/video.mp4"
    test_frame = 100
    
    print("\n🔬 DETECTION QUALITY DIAGNOSTIC")
    print(f"Video: {video_path}")
    print(f"Test frame: {test_frame}\n")
    
    # Diagnostic 1: YOLO segmentation verification
    yolo_comparison = test_yolo_segmentation(video_path, test_frame)
    
    # Diagnostic 2: Gemini comparison
    yolo_detections = None
    if yolo_comparison and 0.35 in yolo_comparison:
        standard_model = yolo_comparison[0.35].get('yolo11s-seg', {})
        yolo_detections = standard_model.get('classes', [])
    
    gemini_comparison = compare_with_gemini(video_path, test_frame, yolo_detections)
    
    # Diagnostic 3: Depth context
    depth_analysis = analyze_depth_context(video_path, test_frame)
    
    # Summary
    print("\n" + "="*80)
    print("🎯 RECOMMENDATIONS")
    print("="*80)
    
    print("""
1. Model Selection:
   - yolo11s-seg has segmentation masks ✓
   - Test yoloe-11s-seg-pf for better accuracy (if available)
   - Consider confidence threshold: 0.35 seems balanced

2. Missing Objects (non-COCO):
   - Use FastVLM to describe undetected regions
   - Build custom embedding-based classifier for personal items
   - Use spatial context (depth zones) to infer object types

3. Depth Integration:
   - Objects at similar depth → likely same surface (desk, shelf)
   - Use size + depth for classification validation
   - SLAM can track these depth zones across frames

4. Next Steps:
   - Implement spatial zone clustering (HDBSCAN on depth + position)
   - Add FastVLM descriptions for unknown objects
   - Build importance scorer (depth, size, persistence → priority)
""")


if __name__ == "__main__":
    main()
