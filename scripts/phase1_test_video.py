#!/usr/bin/env python3
"""
Phase 1: Test example video through YOLO11s-seg detection → CLIP embeddings → EnhancedTracker → Re-ID
"""

import cv2
import sys
import numpy as np
from pathlib import Path
from collections import Counter
import time

# Video metadata extraction
def get_video_metadata(video_path: str):
    """Extract basic video metadata"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        sys.exit(1)
    
    metadata = {
        'resolution': (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'duration_sec': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
    }
    cap.release()
    return metadata


def test_yolo_detection(video_path: str, sample_frames=[0, 100, 500, 1000, 1500]):
    """Test YOLO11s-seg detection on sample frames"""
    from ultralytics import YOLO
    
    # Load YOLO11s-seg directly (bypass model manager for testing)
    model_path = "yolo11s-seg.pt"
    print(f"\n[Step 2] YOLO11s-seg detection test")
    print(f"  Loading model: {model_path}")
    
    yolo = YOLO(model_path)
    print(f"  Model loaded ✓")
    print(f"  Testing frames: {sample_frames}")
    
    cap = cv2.VideoCapture(video_path)
    detection_stats = []
    
    for frame_idx in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"  Warning: Could not read frame {frame_idx}")
            continue
        
        # Run detection
        start = time.time()
        results = yolo(frame, verbose=False)
        elapsed = time.time() - start
        
        # Extract results
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            num_detections = len(boxes)
            classes = boxes.cls.cpu().numpy() if boxes.cls is not None else []
            confidences = boxes.conf.cpu().numpy() if boxes.conf is not None else []
            has_masks = results[0].masks is not None
            
            class_names = [yolo.names[int(c)] for c in classes] if len(classes) > 0 else []
            
            detection_stats.append({
                'frame': frame_idx,
                'num_detections': num_detections,
                'classes': class_names,
                'avg_conf': np.mean(confidences) if len(confidences) > 0 else 0.0,
                'has_masks': has_masks,
                'inference_time': elapsed,
            })
            
            print(f"\n  Frame {frame_idx}:")
            print(f"    Detections: {num_detections}")
            print(f"    Classes: {Counter(class_names).most_common(5)}")
            print(f"    Avg confidence: {np.mean(confidences):.3f}" if len(confidences) > 0 else "    No detections")
            print(f"    Segmentation masks: {'✓' if has_masks else '✗'}")
            print(f"    Inference: {elapsed*1000:.1f}ms")
        else:
            print(f"\n  Frame {frame_idx}: No detections")
            detection_stats.append({
                'frame': frame_idx,
                'num_detections': 0,
                'classes': [],
                'avg_conf': 0.0,
                'has_masks': False,
                'inference_time': elapsed,
            })
    
    cap.release()
    
    # Summary
    total_detections = sum(s['num_detections'] for s in detection_stats)
    avg_detections = total_detections / len(detection_stats) if detection_stats else 0
    avg_inference = np.mean([s['inference_time'] for s in detection_stats]) if detection_stats else 0
    all_classes = [c for s in detection_stats for c in s['classes']]
    
    print(f"\n  ✓ Detection Summary:")
    print(f"    Total detections: {total_detections} across {len(detection_stats)} frames")
    print(f"    Avg detections/frame: {avg_detections:.1f}")
    print(f"    Avg inference time: {avg_inference*1000:.1f}ms")
    print(f"    Unique classes: {len(set(all_classes))}")
    print(f"    Top classes: {Counter(all_classes).most_common(10)}")
    
    return detection_stats


def test_clip_embeddings(video_path: str, yolo, sample_frame=100):
    """Test CLIP embedding extraction for detected objects"""
    from orion.managers.model_manager import ModelManager
    
    print(f"\n[Step 3] CLIP embedding extraction test")
    
    manager = ModelManager.get_instance()
    clip = manager.clip
    print(f"  CLIP model loaded ✓")
    
    # Get a frame with detections
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, sample_frame)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"  Error: Could not read frame {sample_frame}")
        return None
    
    # Run YOLO
    results = yolo(frame, verbose=False)
    if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
        print(f"  Warning: No detections in frame {sample_frame}")
        return None
    
    boxes = results[0].boxes
    xyxy = boxes.xyxy.cpu().numpy()
    classes = boxes.cls.cpu().numpy()
    class_names = [yolo.names[int(c)] for c in classes]
    
    print(f"  Frame {sample_frame}: {len(xyxy)} detections")
    
    # Extract embeddings for each detection
    embeddings = []
    for idx, (box, cls_name) in enumerate(zip(xyxy, class_names)):
        x1, y1, x2, y2 = map(int, box)
        crop = frame[y1:y2, x1:x2]
        
        if crop.size == 0:
            print(f"    Detection {idx} ({cls_name}): Empty crop, skipping")
            continue
        
        # Get CLIP embedding
        embedding = clip.encode_image(crop)
        embeddings.append({
            'class': cls_name,
            'embedding': embedding,
            'box': box,
        })
        
        print(f"    Detection {idx} ({cls_name}): embedding shape {embedding.shape}, norm {np.linalg.norm(embedding):.3f}")
    
    # Test similarity between same-class objects
    if len(embeddings) >= 2:
        print(f"\n  Similarity matrix (first 4 detections):")
        n = min(4, len(embeddings))
        for i in range(n):
            sims = []
            for j in range(n):
                if i == j:
                    sims.append("1.000")
                else:
                    sim = np.dot(embeddings[i]['embedding'], embeddings[j]['embedding']) / (
                        np.linalg.norm(embeddings[i]['embedding']) * np.linalg.norm(embeddings[j]['embedding'])
                    )
                    sims.append(f"{sim:.3f}")
            print(f"    {embeddings[i]['class']:10s}: {' '.join(sims)}")
    
    print(f"\n  ✓ CLIP embeddings extracted for {len(embeddings)} detections")
    print(f"    Embedding dimension: {embeddings[0]['embedding'].shape if embeddings else 'N/A'}")
    
    return embeddings


def test_enhanced_tracker(video_path: str, yolo, clip, num_frames=150, skip_frames=10):
    """Test EnhancedTracker with YOLO detections and CLIP embeddings"""
    from orion.perception.enhanced_tracker import EnhancedTracker
    
    print(f"\n[Step 4] EnhancedTracker integration test")
    print(f"  Processing {num_frames} frames (skip every {skip_frames})")
    
    # Initialize tracker
    tracker = EnhancedTracker(
        max_age=30,  # Keep tracks for 30 frames without update
        min_hits=3,  # Require 3 consecutive detections to confirm track
        iou_threshold=0.3,
        appearance_threshold=0.5,  # Cosine similarity for Re-ID
        ema_alpha=0.9,  # Appearance EMA weight
        max_gallery_size=5,
    )
    
    print(f"  Tracker initialized:")
    print(f"    max_age={tracker.max_age}, min_hits={tracker.min_hits}")
    print(f"    iou_threshold={tracker.iou_threshold}, appearance_threshold={tracker.appearance_threshold}")
    
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    processed_frames = 0
    track_history = []
    
    while processed_frames < num_frames:
        # Skip frames
        for _ in range(skip_frames):
            ret = cap.grab()
            if not ret:
                break
            frame_idx += 1
        
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO
        results = yolo(frame, verbose=False)
        
        # Build detections as dicts (expected by EnhancedTracker)
        detections = []
        embeddings = []
        if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()
            
            for box, conf, cls in zip(xyxy, confs, classes):
                x1, y1, x2, y2 = box
                crop = frame[int(y1):int(y2), int(x1):int(x2)]
                
                if crop.size == 0:
                    continue
                
                # Get CLIP embedding
                embedding = clip.encode_image(crop)
                embeddings.append(embedding)
                
                # Detection dict format expected by EnhancedTracker
                w, h = x2 - x1, y2 - y1
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                detections.append({
                    'bbox_2d': box,
                    'bbox_3d': np.array([cx, cy, 1000.0, w, h, 100.0]),  # Mock 3D (assume 1m depth)
                    'class_name': yolo.names[int(cls)],
                    'confidence': float(conf),
                    'depth_mm': 1000.0,  # Mock depth
                })
        
        # Update tracker
        tracks = tracker.update(detections, embeddings if embeddings else None, camera_pose=None, frame_idx=frame_idx)
        
        track_history.append({
            'frame': frame_idx,
            'num_detections': len(detections),
            'num_tracks': len(tracks),
            'confirmed_tracks': sum(1 for t in tracks if t.hits >= tracker.min_hits),
            'track_ids': [t.id for t in tracks],
        })
        
        processed_frames += 1
        frame_idx += 1
    
    cap.release()
    
    # Analysis
    total_detections = sum(h['num_detections'] for h in track_history)
    avg_detections = total_detections / len(track_history) if track_history else 0
    max_tracks = max(h['num_tracks'] for h in track_history) if track_history else 0
    all_track_ids = set()
    for h in track_history:
        all_track_ids.update(h['track_ids'])
    
    print(f"\n  ✓ Tracking complete:")
    print(f"    Processed {len(track_history)} frames")
    print(f"    Total detections: {total_detections}")
    print(f"    Avg detections/frame: {avg_detections:.1f}")
    print(f"    Unique track IDs: {len(all_track_ids)}")
    print(f"    Max concurrent tracks: {max_tracks}")
    
    # Show sample frames
    print(f"\n  Sample tracking snapshots:")
    for i in [0, len(track_history)//4, len(track_history)//2, 3*len(track_history)//4, len(track_history)-1]:
        if i < len(track_history):
            h = track_history[i]
            print(f"    Frame {h['frame']:4d}: {h['num_detections']} detections → {h['confirmed_tracks']} confirmed tracks (IDs: {h['track_ids'][:5]}{'...' if len(h['track_ids']) > 5 else ''})")
    
    # Get tracker stats
    stats = tracker.get_statistics()
    print(f"\n  Tracker statistics:")
    for key, value in stats.items():
        print(f"    {key}: {value}")
    
    return track_history, tracker


def test_reid_gallery(video_path: str, yolo, clip, test_frames=[100, 200, 300, 400]):
    """Test Re-ID gallery functionality: track persistence and re-association"""
    from orion.perception.enhanced_tracker import EnhancedTracker
    
    print(f"\n[Step 5] Re-ID Gallery Validation Test")
    print(f"  Testing re-identification across frame gaps")
    
    # Initialize tracker with strict settings to force track breaks
    tracker = EnhancedTracker(
        max_age=10,  # Shorter lifespan to force re-ID scenarios
        min_hits=2,  # Lower threshold for testing
        iou_threshold=0.3,
        appearance_threshold=0.6,  # Higher threshold = stricter Re-ID
        max_gallery_size=10,  # More appearance history
        ema_alpha=0.8,  # Faster appearance updates
    )
    
    print(f"  Tracker config (strict for Re-ID testing):")
    print(f"    max_age={tracker.max_age} (short to force breaks)")
    print(f"    appearance_threshold={tracker.appearance_threshold} (strict matching)")
    
    cap = cv2.VideoCapture(video_path)
    
    # Phase 1: Build initial gallery by processing early frames
    print(f"\n  Phase 1: Building appearance gallery (frames 0-150)")
    for frame_idx in range(0, 150, 10):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        results = yolo(frame, verbose=False)
        detections = []
        embeddings = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()
            
            for box, conf, cls in zip(xyxy, confs, classes):
                x1, y1, x2, y2 = box
                crop = frame[int(y1):int(y2), int(x1):int(x2)]
                if crop.size == 0:
                    continue
                
                embedding = clip.encode_image(crop)
                embeddings.append(embedding)
                
                w, h = x2 - x1, y2 - y1
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                detections.append({
                    'bbox_2d': box,
                    'bbox_3d': np.array([cx, cy, 1000.0, w, h, 100.0]),
                    'class_name': yolo.names[int(cls)],
                    'confidence': float(conf),
                    'depth_mm': 1000.0,
                })
        
        tracks = tracker.update(detections, embeddings if embeddings else None, camera_pose=None, frame_idx=frame_idx)
    
    # Capture gallery state
    initial_tracks = len(tracker.tracks)
    gallery_tracks = {t.id: {
        'class': t.class_name,
        'hits': t.hits,
        'gallery_size': len(t.appearance_features) if t.appearance_features else 0,
        'avg_appearance_norm': np.linalg.norm(t.avg_appearance) if t.avg_appearance is not None else 0,
    } for t in tracker.tracks}
    
    print(f"  Gallery built: {initial_tracks} tracks with appearance features")
    for tid, info in list(gallery_tracks.items())[:5]:
        print(f"    Track {tid} ({info['class']}): {info['hits']} hits, {info['gallery_size']} embeddings, avg_norm={info['avg_appearance_norm']:.3f}")
    
    # Phase 2: Create gap (no updates) to force track aging
    print(f"\n  Phase 2: Simulating occlusion gap (skip 50 frames, no tracker updates)")
    gap_start = 200
    gap_end = 250
    
    # Phase 3: Resume tracking - test if Re-ID recovers lost tracks
    print(f"\n  Phase 3: Resume tracking after gap (frames {gap_end}-400)")
    reid_events = []
    
    for frame_idx in range(gap_end, 400, 10):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        results = yolo(frame, verbose=False)
        detections = []
        embeddings = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()
            
            for box, conf, cls in zip(xyxy, confs, classes):
                x1, y1, x2, y2 = box
                crop = frame[int(y1):int(y2), int(x1):int(x2)]
                if crop.size == 0:
                    continue
                
                embedding = clip.encode_image(crop)
                embeddings.append(embedding)
                
                w, h = x2 - x1, y2 - y1
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                detections.append({
                    'bbox_2d': box,
                    'bbox_3d': np.array([cx, cy, 1000.0, w, h, 100.0]),
                    'class_name': yolo.names[int(cls)],
                    'confidence': float(conf),
                    'depth_mm': 1000.0,
                })
        
        old_track_ids = {t.id for t in tracker.tracks}
        tracks = tracker.update(detections, embeddings if embeddings else None, camera_pose=None, frame_idx=frame_idx)
        new_track_ids = {t.id for t in tracks}
        
        # Detect re-associations (old track ID reappears)
        reactivated = old_track_ids.intersection(new_track_ids)
        if reactivated:
            for tid in reactivated:
                track = next((t for t in tracks if t.id == tid), None)
                if track:
                    reid_events.append({
                        'frame': frame_idx,
                        'track_id': tid,
                        'class': track.class_name,
                        'time_since_update': track.time_since_update,
                    })
    
    cap.release()
    
    # Analysis
    final_tracks = len(tracker.tracks)
    reid_count = len(reid_events)
    
    print(f"\n  ✓ Re-ID Gallery Test Complete:")
    print(f"    Initial tracks (with gallery): {initial_tracks}")
    print(f"    Final tracks after gap: {final_tracks}")
    print(f"    Re-ID events detected: {reid_count}")
    
    if reid_events:
        print(f"\n  Re-ID Event Examples:")
        for event in reid_events[:5]:
            print(f"    Frame {event['frame']}: Track {event['track_id']} ({event['class']}) re-identified after {event['time_since_update']} frames gap")
    else:
        print(f"    ⚠️  No Re-ID events detected (tracks may have aged out or gap too short)")
    
    # Test appearance similarity in gallery
    if len(tracker.tracks) >= 2:
        print(f"\n  Gallery appearance similarity test:")
        track_list = list(tracker.tracks)[:4]
        for i, t1 in enumerate(track_list):
            if t1.avg_appearance is None:
                continue
            sims = []
            for t2 in track_list:
                if t2.avg_appearance is None:
                    sims.append("  N/A ")
                    continue
                sim = np.dot(t1.avg_appearance, t2.avg_appearance) / (
                    np.linalg.norm(t1.avg_appearance) * np.linalg.norm(t2.avg_appearance)
                )
                sims.append(f"{sim:5.3f}")
            print(f"    Track {t1.id:2d} ({t1.class_name:10s}): {' '.join(sims)}")
    
    return reid_events, tracker


if __name__ == "__main__":
    video_path = "data/examples/video.mp4"
    
    print("="*80)
    print("PHASE 1: Example Video Pipeline Test")
    print("="*80)
    
    # Step 1: Verify video exists and extract metadata
    print(f"\n[Step 1] Video metadata for {video_path}")
    meta = get_video_metadata(video_path)
    print(f"  Resolution: {meta['resolution'][0]}x{meta['resolution'][1]}")
    print(f"  Total frames: {meta['total_frames']}")
    print(f"  FPS: {meta['fps']:.2f}")
    print(f"  Duration: {meta['duration_sec']:.2f}s")
    print(f"  ✓ Video file validated")
    
    # Step 2: Test YOLO11s-seg detection
    detection_stats = test_yolo_detection(video_path)
    
    # Step 3: Test CLIP embedding extraction
    # Reuse YOLO model from ultralytics import
    from ultralytics import YOLO
    yolo = YOLO("yolo11s-seg.pt")
    embeddings = test_clip_embeddings(video_path, yolo, sample_frame=100)
    
    # Step 4: Test EnhancedTracker
    from orion.managers.model_manager import ModelManager
    manager = ModelManager.get_instance()
    clip = manager.clip
    track_history, tracker = test_enhanced_tracker(video_path, yolo, clip, num_frames=150, skip_frames=10)
    
    # Step 5: Test Re-ID gallery functionality
    reid_events, reid_tracker = test_reid_gallery(video_path, yolo, clip)
    
    print("\n" + "="*80)
    print("PHASE 1 COMPLETE: All components validated ✓")
    print("="*80)
