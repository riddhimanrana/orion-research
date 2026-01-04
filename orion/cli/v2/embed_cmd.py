"""orion embed - Run V-JEPA2 or other Re-ID embedding"""

import json
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


def run_embed(args) -> int:
    """Generate Re-ID embeddings for tracked objects."""
    
    episode_dir = Path("results") / args.episode
    tracks_path = episode_dir / "tracks.jsonl"
    
    if not tracks_path.exists():
        print(f"No tracks found for episode: {args.episode}")
        print("Run: orion detect --episode <name> first")
        return 1
    
    print(f"\n  Stage 2: EMBEDDING (Re-ID)")
    print(f"  Embedder: {args.embedder}")
    print(f"  Mode: {args.mode}")
    print("  " + "─" * 60)
    
    # Load tracks
    tracks_by_id = defaultdict(list)
    all_observations = []
    
    with open(tracks_path) as f:
        for line in f:
            if line.strip():
                obs = json.loads(line)
                all_observations.append(obs)
                tracks_by_id[obs["track_id"]].append(obs)
    
    print(f"  Loaded {len(all_observations)} observations for {len(tracks_by_id)} tracks")
    
    # Load episode meta to get video path
    meta_path = episode_dir / "episode_meta.json"
    if not meta_path.exists():
        print("  ✗ Episode metadata not found")
        return 1
    
    with open(meta_path) as f:
        meta = json.load(f)
    
    video_path = meta["video_path"]
    
    # Initialize embedder
    embedder = None
    try:
        if args.embedder == "vjepa2":
            from orion.backends.vjepa2_backend import VJepa2Embedder
            embedder = VJepa2Embedder()
            print("  Loaded V-JEPA2 embedder")
        elif args.embedder == "videomae":
            from orion.backends.vjepa2_backend import VideoMAEEmbedder
            embedder = VideoMAEEmbedder()
            print("  Loaded VideoMAE embedder")
        elif args.embedder == "clip":
            from orion.perception.embedder import ClipEmbedder
            embedder = ClipEmbedder()
            print("  Loaded CLIP embedder")
        elif args.embedder == "dinov3":
            from orion.perception.embedder import DINOv3Embedder
            embedder = DINOv3Embedder()
            print("  Loaded DINOv3 embedder")
        else:
            print(f"  ✗ Unknown embedder: {args.embedder}")
            return 1
    except ImportError as e:
        print(f"  ✗ Failed to import embedder: {e}")
        return 1
    except Exception as e:
        print(f"  ✗ Failed to initialize embedder: {e}")
        return 1
    
    # Extract crops and compute embeddings
    import cv2
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ✗ Cannot open video: {video_path}")
        return 1
    
    # Build frame → observations mapping
    obs_by_frame = defaultdict(list)
    for obs in all_observations:
        obs_by_frame[obs["frame_id"]].append(obs)
    
    # Get sorted frames
    frame_ids = sorted(obs_by_frame.keys())
    
    # Storage for crops per track
    track_crops = defaultdict(list)  # track_id → [(frame_id, crop), ...]
    
    # Sample representative frames per track
    print(f"  Extracting crops from {len(frame_ids)} frames...")
    
    current_frame = 0
    for target_frame in frame_ids:
        # Seek to frame
        while current_frame < target_frame:
            cap.read()
            current_frame += 1
        
        ret, frame = cap.read()
        if not ret:
            continue
        current_frame += 1
        
        # Extract crops for observations in this frame
        for obs in obs_by_frame[target_frame]:
            bbox = obs.get("bbox")
            if not bbox:
                continue
            
            x1, y1, x2, y2 = map(int, bbox)
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue
            
            crop = frame[y1:y2, x1:x2]
            track_crops[obs["track_id"]].append((target_frame, crop))
    
    cap.release()
    
    print(f"  Extracted crops for {len(track_crops)} tracks")
    
    # Compute embeddings
    embeddings = {}
    
    for i, (track_id, crops) in enumerate(track_crops.items()):
        if (i + 1) % 10 == 0:
            print(f"  Processing track {i + 1}/{len(track_crops)}...")
        
        if args.mode == "single":
            # Use best/middle crop
            mid_idx = len(crops) // 2
            _, crop = crops[mid_idx]
            
            try:
                emb = embedder.embed_single_image(crop)
                embeddings[track_id] = emb
            except Exception as e:
                logger.warning(f"Failed to embed track {track_id}: {e}")
                
        elif args.mode == "video":
            # Use multiple crops as video sequence
            if len(crops) < 2:
                # Fall back to single
                _, crop = crops[0]
                emb = embedder.embed_single_image(crop)
            else:
                # Sample up to 8 frames
                step = max(1, len(crops) // 8)
                sampled = [crops[j][1] for j in range(0, len(crops), step)][:8]
                
                try:
                    emb = embedder.embed_video_sequence(sampled)
                    embeddings[track_id] = emb
                except Exception as e:
                    # Fall back to single
                    logger.warning(f"Video embedding failed, using single: {e}")
                    emb = embedder.embed_single_image(crops[len(crops)//2][1])
                    embeddings[track_id] = emb
    
    # Save embeddings
    embeddings_path = episode_dir / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    print(f"  Saved {len(embeddings)} embeddings to {embeddings_path}")
    
    # Cluster to create memory objects
    print(f"\n  Clustering tracks (threshold: {args.threshold})...")
    
    memory_objects = cluster_embeddings(embeddings, tracks_by_id, args.threshold)
    
    # Save memory
    memory = {
        "episode": args.episode,
        "embedder": args.embedder,
        "mode": args.mode,
        "objects": memory_objects,
    }
    
    memory_path = episode_dir / "memory.json"
    with open(memory_path, "w") as f:
        json.dump(memory, f, indent=2)
    
    print(f"  Created {len(memory_objects)} memory objects")
    print(f"  Saved to {memory_path}")
    
    # Update status
    meta["status"]["embedded"] = True
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    return 0


def cluster_embeddings(embeddings: dict, tracks_by_id: dict, threshold: float) -> list:
    """Cluster embeddings to merge same-object tracks."""
    
    track_ids = list(embeddings.keys())
    if not track_ids:
        return []
    
    # Build embedding matrix
    emb_list = []
    valid_ids = []
    
    for tid in track_ids:
        emb = embeddings[tid]
        if isinstance(emb, np.ndarray):
            emb_list.append(emb.flatten())
            valid_ids.append(tid)
    
    if not emb_list:
        # No valid embeddings, create object per track
        objects = []
        for tid in track_ids:
            obs = tracks_by_id[tid]
            label = obs[0].get("label", "object") if obs else "object"
            objects.append({
                "id": f"obj_{len(objects)}",
                "track_ids": [tid],
                "canonical_label": label,
                "total_observations": len(obs),
                "first_frame": min(o["frame_id"] for o in obs),
                "last_frame": max(o["frame_id"] for o in obs),
            })
        return objects
    
    # Normalize and compute similarity matrix
    emb_array = np.array(emb_list)
    norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
    emb_array = emb_array / (norms + 1e-8)
    sim_matrix = emb_array @ emb_array.T
    
    # Simple greedy clustering
    assigned = set()
    clusters = []
    
    for i, tid in enumerate(valid_ids):
        if tid in assigned:
            continue
        
        cluster = [tid]
        assigned.add(tid)
        
        # Find similar tracks
        for j, tid2 in enumerate(valid_ids):
            if tid2 in assigned:
                continue
            if sim_matrix[i, j] >= threshold:
                cluster.append(tid2)
                assigned.add(tid2)
        
        clusters.append(cluster)
    
    # Add unassigned tracks (no valid embedding)
    for tid in track_ids:
        if tid not in assigned:
            clusters.append([tid])
    
    # Build memory objects
    objects = []
    for cluster in clusters:
        # Collect all observations
        all_obs = []
        for tid in cluster:
            all_obs.extend(tracks_by_id.get(tid, []))
        
        if not all_obs:
            continue
        
        # Majority vote for label
        labels = [o.get("label", "object") for o in all_obs]
        label_counts = defaultdict(int)
        for l in labels:
            label_counts[l] += 1
        canonical_label = max(label_counts, key=label_counts.get)
        
        obj = {
            "id": f"obj_{len(objects)}",
            "track_ids": cluster,
            "merged_track_ids": cluster if len(cluster) > 1 else [],
            "canonical_label": canonical_label,
            "total_observations": len(all_obs),
            "first_frame": min(o["frame_id"] for o in all_obs),
            "last_frame": max(o["frame_id"] for o in all_obs),
        }
        objects.append(obj)
    
    return objects
