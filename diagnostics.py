#!/usr/bin/env python3
"""
DIAGNOSTIC SCRIPT - Find exact failures
"""

import sys
import cv2
import numpy as np
from pathlib import Path

print("\n" + "="*100)
print("🔧 ORION SYSTEM DIAGNOSTICS")
print("="*100)

# 1. Test YOLO
print("\n[1/6] Testing YOLO...")
try:
    from ultralytics import YOLO
    yolo = YOLO("yolo11n.pt")
    
    # Test on single frame
    cap = cv2.VideoCapture("data/examples/room.mp4")
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        results = yolo(frame, conf=0.3, verbose=False)
        detections = len(results[0].boxes)
        print(f"   ✓ YOLO working: {detections} detections on frame 0")
        if detections > 0:
            for i, det in enumerate(results[0].boxes[:3]):
                cls_name = yolo.names[int(det.cls[0])]
                conf = float(det.conf[0])
                print(f"     - {cls_name}: {conf:.2f}")
    else:
        print("   ✗ Cannot read frame")
except Exception as e:
    print(f"   ✗ YOLO ERROR: {e}")

# 2. Test Scene Classifier
print("\n[2/6] Testing Scene Classifier...")
try:
    from orion.semantic.scene_classifier import SceneClassifier
    classifier = SceneClassifier()
    
    cap = cv2.VideoCapture("data/examples/room.mp4")
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        result = classifier.classify_frame(frame)
        print(f"   ✓ Scene classifier working")
        print(f"     Type: {type(result)}")
        if hasattr(result, 'scene_type'):
            print(f"     Scene type: {result.scene_type}")
        elif isinstance(result, dict):
            print(f"     Result: {result}")
        else:
            print(f"     Result: {result}")
    else:
        print("   ✗ Cannot read frame")
except Exception as e:
    print(f"   ✗ SCENE CLASSIFIER ERROR: {e}")
    import traceback
    traceback.print_exc()

# 3. Test CLIP Embeddings
print("\n[3/6] Testing CLIP Embeddings...")
try:
    from orion.graph.embeddings import create_embedding_model
    model = create_embedding_model("openai/clip-vit-base-patch32")
    
    cap = cv2.VideoCapture("data/examples/room.mp4")
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Test embedding
        embedding = model.embed_image(frame)
        print(f"   ✓ CLIP embeddings working")
        print(f"     Embedding shape: {embedding.shape if hasattr(embedding, 'shape') else type(embedding)}")
        print(f"     Embedding type: {type(embedding)}")
    else:
        print("   ✗ Cannot read frame")
except Exception as e:
    print(f"   ✗ CLIP EMBEDDING ERROR: {e}")
    import traceback
    traceback.print_exc()

# 4. Test Depth Model
print("\n[4/6] Testing Depth Model...")
try:
    import torch
    print("   Loading Depth Anything V2...")
    depth_model = torch.hub.load(
        'DepthAnything/Depth-Anything-V2',
        'dpt_small',
        pretrained=True,
        trust_repo=True
    )
    depth_model.eval()
    
    cap = cv2.VideoCapture("data/examples/room.mp4")
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        from PIL import Image
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        with torch.no_grad():
            depth = depth_model.infer_image(img)
        
        print(f"   ✓ Depth model working")
        print(f"     Output shape: {depth.shape if hasattr(depth, 'shape') else type(depth)}")
    else:
        print("   ✗ Cannot read frame")
except Exception as e:
    print(f"   ✗ DEPTH MODEL ERROR: {e}")
    import traceback
    traceback.print_exc()

# 5. Test Memgraph Connection
print("\n[5/6] Testing Memgraph...")
try:
    from orion.graph.memgraph_backend import MemgraphBackend
    mgraph = MemgraphBackend()
    print(f"   ✓ Memgraph backend imported")
    print(f"     Status: Check if connection available")
except Exception as e:
    print(f"   ✗ MEMGRAPH ERROR: {e}")

# 6. Test Camera Intrinsics
print("\n[6/6] Testing Camera Intrinsics...")
try:
    cap = cv2.VideoCapture("data/examples/room.mp4")
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        h, w = frame.shape[:2]
        print(f"   ✓ Frame dimensions: {w}x{h}")
        print(f"     Assumed focal length: {w} pixels")
        print(f"     Assumed principal point: ({w/2}, {h/2})")
        print(f"     ⚠️  NOTE: These are HARDCODED assumptions, not real calibration")
    else:
        print("   ✗ Cannot read frame")
except Exception as e:
    print(f"   ✗ CAMERA ERROR: {e}")

print("\n" + "="*100)
print("📋 SUMMARY")
print("="*100)
print("""
If any components show ✗ ERROR:
  1. Scene Classifier failing → Implementation bug
  2. Depth Model failing → torch hub cache issue
  3. CLIP Embeddings failing → Model loading issue
  4. Memgraph failing → Backend not available
  
CRITICAL: The system appears to be working at some level (getting YOLO detections),
but key components are failing silently or returning generic results.
""")

print("\n")
