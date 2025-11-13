#!/usr/bin/env python3
"""
Visual Re-ID Test with DINOv3
==============================

Visualizes object detection, tracking, and Re-ID on video frames.
Shows bounding boxes, track IDs, and embedding similarities.

Usage:
    python scripts/test_visual_reid.py --video data/examples/sample.mp4
    python scripts/test_visual_reid.py --video data/examples/sample.mp4 --use-dino
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Setup path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from orion.managers.model_manager import ModelManager


def compute_similarity_matrix(embeddings):
    """Compute cosine similarity matrix"""
    if len(embeddings) == 0:
        return np.zeros((0, 0))
    
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    normalized = embeddings / norms
    
    # Compute similarity
    return normalized @ normalized.T


def draw_detections(frame, detections, embeddings, use_dino=False):
    """Draw bounding boxes and track IDs on frame"""
    vis = frame.copy()
    h, w = vis.shape[:2]
    
    # Color palette for tracks
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128),
        (128, 128, 0), (128, 0, 128), (0, 128, 128),
    ]
    
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        label = det['label']
        conf = det['conf']
        track_id = det.get('track_id', i)
        
        # Get color for this track
        color = colors[track_id % len(colors)]
        
        # Draw box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        text = f"#{track_id} {label} {conf:.2f}"
        
        # Add embedding similarity if available
        if embeddings is not None and len(embeddings) > i:
            emb_norm = np.linalg.norm(embeddings[i])
            text += f" |e|={emb_norm:.2f}"
        
        # Background for text
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(vis, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1)
    
    # Add backend indicator
    backend_text = "DINOv3" if use_dino else "CLIP"
    cv2.putText(vis, f"Embeddings: {backend_text}", (10, h - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return vis


def visualize_similarity_matrix(sim_matrix, labels, output_path):
    """Visualize embedding similarity matrix"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(sim_matrix, cmap='RdYlGn', vmin=-1, vmax=1)
    
    # Ticks and labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels([f"#{l['track_id']} {l['label'][:8]}" for l in labels], rotation=45, ha='right')
    ax.set_yticklabels([f"#{l['track_id']} {l['label'][:8]}" for l in labels])
    
    # Colorbar
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    
    # Title
    ax.set_title('Object Embedding Similarity Matrix')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved similarity matrix: {output_path}")


def process_video(video_path, output_dir, use_dino=False, max_frames=50, skip_frames=10):
    """Process video and visualize Re-ID"""
    
    print(f"Processing: {video_path}")
    print(f"Output: {output_dir}")
    print(f"Embeddings: {'DINOv3' if use_dino else 'CLIP'}")
    print()
    
    # Initialize models
    mm = ModelManager.get_instance()
    yolo = mm.yolo
    embedder = mm.dino if use_dino else mm.clip
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {total_frames} frames @ {fps:.1f} FPS")
    print(f"Processing every {skip_frames} frames, max {max_frames}")
    print()
    
    # Track state
    all_detections = []
    all_embeddings = []
    frame_idx = 0
    processed_count = 0
    
    # Simple tracker (IoU-based)
    next_track_id = 0
    prev_detections = []
    
    while cap.isOpened() and processed_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % skip_frames != 0:
            frame_idx += 1
            continue
        
        # Run detection
        results = yolo(frame, verbose=False)
        
        if not results or len(results) == 0:
            frame_idx += 1
            continue
        
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            frame_idx += 1
            continue
        
        # Extract detections
        frame_detections = []
        frame_embeddings = []
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = yolo.names[cls]
            
            # Skip low confidence
            if conf < 0.3:
                continue
            
            # Extract crop for embedding
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            
            # Get embedding
            try:
                emb = embedder.encode_image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                frame_embeddings.append(emb)
            except Exception as e:
                print(f"  Warning: Embedding failed for {label}: {e}")
                continue
            
            det = {
                'bbox': (x1, y1, x2, y2),
                'label': label,
                'conf': conf,
                'track_id': -1,  # Will assign below
            }
            frame_detections.append(det)
        
        # Simple IoU-based tracking
        if len(prev_detections) > 0 and len(frame_detections) > 0:
            # Compute IoU matrix
            iou_matrix = np.zeros((len(frame_detections), len(prev_detections)))
            
            for i, det in enumerate(frame_detections):
                x1, y1, x2, y2 = det['bbox']
                area1 = (x2 - x1) * (y2 - y1)
                
                for j, prev_det in enumerate(prev_detections):
                    px1, py1, px2, py2 = prev_det['bbox']
                    area2 = (px2 - px1) * (py2 - py1)
                    
                    # Intersection
                    ix1 = max(x1, px1)
                    iy1 = max(y1, py1)
                    ix2 = min(x2, px2)
                    iy2 = min(y2, py2)
                    
                    if ix2 > ix1 and iy2 > iy1:
                        inter = (ix2 - ix1) * (iy2 - iy1)
                        union = area1 + area2 - inter
                        iou_matrix[i, j] = inter / (union + 1e-6)
            
            # Greedy assignment
            assigned = set()
            for i, det in enumerate(frame_detections):
                best_j = np.argmax(iou_matrix[i])
                if iou_matrix[i, best_j] > 0.3 and best_j not in assigned:
                    # Inherit track ID
                    det['track_id'] = prev_detections[best_j]['track_id']
                    assigned.add(best_j)
                else:
                    # New track
                    det['track_id'] = next_track_id
                    next_track_id += 1
        else:
            # First frame: assign new IDs
            for det in frame_detections:
                det['track_id'] = next_track_id
                next_track_id += 1
        
        prev_detections = frame_detections.copy()
        
        # Visualize frame
        vis = draw_detections(frame, frame_detections, 
                             np.array(frame_embeddings) if frame_embeddings else None,
                             use_dino=use_dino)
        
        # Save frame
        frame_out = output_dir / f"frame_{processed_count:04d}.jpg"
        cv2.imwrite(str(frame_out), vis)
        
        # Store for analysis
        all_detections.extend(frame_detections)
        all_embeddings.extend(frame_embeddings)
        
        print(f"  Frame {frame_idx:4d}: {len(frame_detections)} objects detected")
        
        processed_count += 1
        frame_idx += 1
    
    cap.release()
    
    print()
    print(f"✓ Processed {processed_count} frames")
    print(f"  Total detections: {len(all_detections)}")
    print(f"  Unique tracks: {next_track_id}")
    print()
    
    # Compute and visualize similarity matrix
    if len(all_embeddings) > 0:
        embeddings = np.array(all_embeddings)
        sim_matrix = compute_similarity_matrix(embeddings)
        
        # Save similarity matrix
        sim_out = output_dir / "similarity_matrix.png"
        visualize_similarity_matrix(sim_matrix, all_detections, sim_out)
        
        # Print statistics
        n = len(embeddings)
        print("Embedding Statistics:")
        print(f"  Dimension: {embeddings.shape[1]}")
        print(f"  Mean L2 norm: {np.linalg.norm(embeddings, axis=1).mean():.4f}")
        print()
        
        # Similarity stats (exclude diagonal)
        mask = ~np.eye(n, dtype=bool)
        if mask.sum() > 0:
            same_class_sims = []
            diff_class_sims = []
            
            for i in range(n):
                for j in range(i+1, n):
                    sim = sim_matrix[i, j]
                    if all_detections[i]['label'] == all_detections[j]['label']:
                        same_class_sims.append(sim)
                    else:
                        diff_class_sims.append(sim)
            
            print("Similarity Analysis:")
            print(f"  Same-class pairs: {len(same_class_sims)}")
            if same_class_sims:
                print(f"    Mean: {np.mean(same_class_sims):.4f}")
                print(f"    Min:  {np.min(same_class_sims):.4f}")
                print(f"    Max:  {np.max(same_class_sims):.4f}")
            
            print(f"  Different-class pairs: {len(diff_class_sims)}")
            if diff_class_sims:
                print(f"    Mean: {np.mean(diff_class_sims):.4f}")
                print(f"    Min:  {np.min(diff_class_sims):.4f}")
                print(f"    Max:  {np.max(diff_class_sims):.4f}")
    
    print()
    print(f"✓ Results saved to: {output_dir}")
    print(f"  - frame_*.jpg (annotated frames)")
    print(f"  - similarity_matrix.png")


def main():
    parser = argparse.ArgumentParser(description="Visual Re-ID test with DINOv3")
    parser.add_argument("--video", type=Path, required=True, help="Input video path")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument("--use-dino", action="store_true", help="Use DINOv3 (default: CLIP)")
    parser.add_argument("--max-frames", type=int, default=50, help="Max frames to process")
    parser.add_argument("--skip", type=int, default=10, help="Process every Nth frame")
    
    args = parser.parse_args()
    
    if not args.video.exists():
        print(f"❌ Video not found: {args.video}")
        return 1
    
    # Create output directory
    if args.output_dir is None:
        backend_name = "dinov3" if args.use_dino else "clip"
        args.output_dir = Path(f"results/visual_reid_{backend_name}")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        process_video(
            args.video,
            args.output_dir,
            use_dino=args.use_dino,
            max_frames=args.max_frames,
            skip_frames=args.skip,
        )
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
