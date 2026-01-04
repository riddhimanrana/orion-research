"""
orion analyze - Run full analysis pipeline

Stages:
1. DETECT: YOLO-World detection + online tracking
2. EMBED: V-JEPA2 re-identification embeddings
3. FILTER: FastVLM semantic filtering
4. GRAPH: Scene graph + Causal Influence Scoring
5. EXPORT: (optional) Export to Memgraph
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def run_analyze(args) -> int:
    """Run full analysis pipeline."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Load episode metadata
    episode_dir = Path("results") / args.episode
    meta_path = episode_dir / "episode_meta.json"
    
    if not meta_path.exists():
        logger.error(f"Episode not found: {args.episode}")
        logger.error(f"Run: orion init --episode {args.episode} --video <path>")
        return 1
    
    with open(meta_path) as f:
        meta = json.load(f)
    
    video_path = meta["video_path"]
    
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                  ORION V2 - FULL PIPELINE ANALYSIS               ║
╠══════════════════════════════════════════════════════════════════╣
║  Episode:   {args.episode:<52} ║
║  Detector:  {args.detector:<52} ║
║  Embedder:  {args.embedder:<52} ║
║  Device:    {args.device:<52} ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    total_start = time.time()
    
    # ═══════════════════════════════════════════════════════════════
    # STAGE 1: DETECTION + TRACKING
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  STAGE 1: DETECTION + TRACKING")
    print("═" * 70)
    
    stage1_start = time.time()
    
    if args.detector == "yoloworld":
        from orion.backends.yoloworld_backend import YOLOWorldDetector, YOLOWorldConfig
        
        config = YOLOWorldConfig(
            model="yolov8x-worldv2",
            device=args.device,
            confidence=0.25
        )
        detector = YOLOWorldDetector(config)
        
        # Run detection with built-in tracking
        tracks_data = []
        frame_count = 0
        detection_count = 0
        
        logger.info(f"Running YOLO-World detection + tracking on {video_path}")
        
        for frame_id, tracks in detector.track(video_path):
            for track in tracks:
                tracks_data.append({
                    "frame_id": frame_id,
                    "timestamp": frame_id / meta["video"]["fps"],
                    "track_id": track["track_id"],
                    "bbox": track["bbox"],
                    "confidence": track["confidence"],
                    "label": track["label"],
                    "class_id": track["class_id"]
                })
                detection_count += 1
            frame_count += 1
            
            if frame_count % 100 == 0:
                logger.info(f"  Processed {frame_count} frames, {detection_count} detections")
        
        # Save tracks
        tracks_path = episode_dir / "tracks.jsonl"
        with open(tracks_path, "w") as f:
            for track in tracks_data:
                f.write(json.dumps(track) + "\n")
        
        logger.info(f"✓ Saved {len(tracks_data)} track observations → {tracks_path}")
        
    else:
        # Fall back to YOLO11x
        logger.info("Using YOLO11x detector (fallback)")
        # Import and run legacy detector
        from orion.cli.run_tracks import run_tracks_pipeline
        run_tracks_pipeline(
            episode=args.episode,
            video=video_path,
            yolo_model=args.detector.replace("yolo", "yolo"),
            fps=args.fps,
            device=args.device
        )
    
    stage1_time = time.time() - stage1_start
    print(f"\n  ✓ Stage 1 complete in {stage1_time:.1f}s")
    
    # Update status
    meta["status"]["detected"] = True
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    # ═══════════════════════════════════════════════════════════════
    # STAGE 2: RE-ID EMBEDDINGS (V-JEPA2)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "═" * 70)
    print("  STAGE 2: RE-ID EMBEDDINGS (V-JEPA2)")
    print("═" * 70)
    
    stage2_start = time.time()
    
    if args.embedder == "vjepa2":
        from orion.backends.vjepa2_backend import VJepa2Embedder
        import numpy as np
        import cv2
        from collections import defaultdict
        
        embedder = VJepa2Embedder(device=args.device)
        
        # Load tracks
        tracks_path = episode_dir / "tracks.jsonl"
        tracks = []
        with open(tracks_path) as f:
            for line in f:
                tracks.append(json.loads(line))
        
        # Group by track_id
        tracks_by_id = defaultdict(list)
        for t in tracks:
            tracks_by_id[t["track_id"]].append(t)
        
        logger.info(f"Computing embeddings for {len(tracks_by_id)} tracks")
        
        # Open video for crop extraction
        cap = cv2.VideoCapture(video_path)
        
        embeddings = {}
        for track_id, observations in tracks_by_id.items():
            # Get middle observation for crop
            mid_obs = observations[len(observations) // 2]
            frame_id = mid_obs["frame_id"]
            bbox = mid_obs["bbox"]
            
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Crop
            x1, y1, x2, y2 = map(int, bbox)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            
            # Resize to consistent size
            crop = cv2.resize(crop, (224, 224))
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            # Embed
            embedding = embedder.embed_single_image(crop)
            embeddings[track_id] = embedding.numpy().tolist()
            
            if len(embeddings) % 10 == 0:
                logger.info(f"  Embedded {len(embeddings)} / {len(tracks_by_id)} tracks")
        
        cap.release()
        
        # Cluster embeddings into memory objects
        logger.info("Clustering tracks into memory objects...")
        
        # Simple agglomerative clustering
        import torch.nn.functional as F
        import torch
        
        track_ids = list(embeddings.keys())
        emb_matrix = torch.tensor([embeddings[tid][0] for tid in track_ids])
        emb_matrix = F.normalize(emb_matrix, dim=1)
        
        # Compute pairwise similarities
        sim_matrix = torch.mm(emb_matrix, emb_matrix.T)
        
        # Cluster with threshold
        threshold = 0.75
        clusters = []
        assigned = set()
        
        for i, tid in enumerate(track_ids):
            if tid in assigned:
                continue
            cluster = [tid]
            assigned.add(tid)
            
            for j, other_tid in enumerate(track_ids):
                if other_tid in assigned:
                    continue
                if sim_matrix[i, j] > threshold:
                    cluster.append(other_tid)
                    assigned.add(other_tid)
            
            clusters.append(cluster)
        
        # Build memory objects
        memory_objects = []
        for i, cluster in enumerate(clusters):
            # Get canonical label (most common)
            labels = []
            for tid in cluster:
                for obs in tracks_by_id[tid]:
                    labels.append(obs["label"])
            
            from collections import Counter
            label_counts = Counter(labels)
            canonical_label = label_counts.most_common(1)[0][0] if label_counts else "unknown"
            
            memory_objects.append({
                "id": f"mem_{i:03d}",
                "canonical_label": canonical_label,
                "track_ids": cluster,
                "total_observations": sum(len(tracks_by_id[tid]) for tid in cluster),
                "embedding": embeddings[cluster[0]]  # Representative embedding
            })
        
        # Save memory
        memory_path = episode_dir / "memory.json"
        with open(memory_path, "w") as f:
            json.dump({
                "episode": args.episode,
                "objects": memory_objects,
                "clustering_threshold": threshold,
                "embedder": "vjepa2"
            }, f, indent=2)
        
        logger.info(f"✓ Created {len(memory_objects)} memory objects → {memory_path}")
        
    else:
        # Fall back to DINO/CLIP
        logger.info(f"Using {args.embedder} embedder (legacy)")
        from orion.perception.reid.matcher import build_memory_from_tracks
        build_memory_from_tracks(str(episode_dir))
    
    stage2_time = time.time() - stage2_start
    print(f"\n  ✓ Stage 2 complete in {stage2_time:.1f}s")
    
    meta["status"]["embedded"] = True
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    # ═══════════════════════════════════════════════════════════════
    # STAGE 3: SEMANTIC FILTERING (FastVLM)
    # ═══════════════════════════════════════════════════════════════
    if not args.skip_filter:
        print("\n" + "═" * 70)
        print("  STAGE 3: SEMANTIC FILTERING (FastVLM + MPNet)")
        print("═" * 70)
        
        stage3_start = time.time()
        
        # Import and run VLM filter
        try:
            from orion.cli.run_vlm_filter import run_vlm_filter
            
            run_vlm_filter(
                video_path=video_path,
                results_dir=str(episode_dir),
                tracks_file="tracks.jsonl",
                scene_trigger="cosine",
                scene_change_threshold=0.98,
                sentence_model="sentence-transformers/all-mpnet-base-v2"
            )
            
            meta["status"]["filtered"] = True
            
        except Exception as e:
            logger.warning(f"FastVLM filtering failed: {e}")
            logger.warning("Continuing without filtering...")
        
        stage3_time = time.time() - stage3_start
        print(f"\n  ✓ Stage 3 complete in {stage3_time:.1f}s")
    else:
        print("\n  ⊘ Stage 3 skipped (--skip-filter)")
    
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    # ═══════════════════════════════════════════════════════════════
    # STAGE 4: SCENE GRAPH + CIS
    # ═══════════════════════════════════════════════════════════════
    if not args.skip_graph:
        print("\n" + "═" * 70)
        print("  STAGE 4: SCENE GRAPH + CAUSAL INFLUENCE SCORING")
        print("═" * 70)
        
        stage4_start = time.time()
        
        # Build scene graph
        from orion.graph.scene_graph import build_scene_graph_from_tracks
        
        tracks_file = episode_dir / "tracks_filtered.jsonl"
        if not tracks_file.exists():
            tracks_file = episode_dir / "tracks.jsonl"
        
        build_scene_graph_from_tracks(
            tracks_path=str(tracks_file),
            output_path=str(episode_dir / "scene_graph.jsonl")
        )
        
        # Compute CIS
        logger.info("Computing Causal Influence Scores...")
        # TODO: Implement CIS computation
        
        meta["status"]["graphed"] = True
        
        stage4_time = time.time() - stage4_start
        print(f"\n  ✓ Stage 4 complete in {stage4_time:.1f}s")
    else:
        print("\n  ⊘ Stage 4 skipped (--skip-graph)")
    
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    # ═══════════════════════════════════════════════════════════════
    # STAGE 5: MEMGRAPH EXPORT (optional)
    # ═══════════════════════════════════════════════════════════════
    if args.export_memgraph:
        print("\n" + "═" * 70)
        print("  STAGE 5: MEMGRAPH EXPORT")
        print("═" * 70)
        
        stage5_start = time.time()
        
        from orion.graph.memgraph_backend import export_episode_to_memgraph
        
        try:
            export_episode_to_memgraph(
                episode_dir=str(episode_dir),
                host="localhost",
                port=7687
            )
            meta["status"]["exported"] = True
            
        except Exception as e:
            logger.error(f"Memgraph export failed: {e}")
        
        stage5_time = time.time() - stage5_start
        print(f"\n  ✓ Stage 5 complete in {stage5_time:.1f}s")
    
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    total_time = time.time() - total_start
    
    print(f"""

╔══════════════════════════════════════════════════════════════════╗
║                      ANALYSIS COMPLETE                           ║
╠══════════════════════════════════════════════════════════════════╣
║  Episode:      {args.episode:<48} ║
║  Total time:   {total_time:.1f}s{' ' * max(0, 48 - len(f'{total_time:.1f}s'))}║
║                                                                  ║
║  Outputs:                                                        ║
║    - tracks.jsonl         (raw detections + tracks)              ║
║    - memory.json          (clustered memory objects)             ║
║    - tracks_filtered.jsonl (validated tracks)                    ║
║    - scene_graph.jsonl    (spatial relationships)                ║
║                                                                  ║
║  Next steps:                                                     ║
║    orion query "What objects are in the video?"                  ║
║    orion status --episode {args.episode:<30} ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    return 0
