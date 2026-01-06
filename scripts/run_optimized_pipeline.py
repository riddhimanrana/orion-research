#!/usr/bin/env python3
"""Run optimized Orion v2 pipeline with YOLO-World + V-JEPA2.

Usage:
    python scripts/run_optimized_pipeline.py --video data/examples/test.mp4 --episode test_showcase
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orion.perception.config import get_yoloworld_coarse_config
from orion.perception.engine import PerceptionEngine
from orion.config import ensure_results_dir
from orion.perception.reid.matcher import build_memory_from_tracks

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pipeline(video_path: str, episode_id: str, device: str = "cuda") -> dict:
    """Run the optimized YOLO-World + V-JEPA2 pipeline."""
    
    logger.info("=" * 80)
    logger.info("ORION v2 OPTIMIZED PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Video: {video_path}")
    logger.info(f"Episode: {episode_id}")
    logger.info(f"Device: {device}")
    
    start_time = time.time()
    
    # Get optimized config
    config = get_yoloworld_coarse_config()
    
    # Override device
    config.embedding.device = device
    
    logger.info("\n--- Configuration ---")
    logger.info(f"Detection backend: {config.detection.backend}")
    logger.info(f"Confidence threshold: {config.detection.confidence_threshold}")
    logger.info(f"NMS IoU threshold: {config.detection.iou_threshold}")
    logger.info(f"Target FPS: {config.target_fps}")
    logger.info(f"Prompt preset: {config.detection.yoloworld_prompt_preset}")
    
    # Initialize engine
    logger.info("\n[1/4] Initializing PerceptionEngine...")
    engine = PerceptionEngine(config=config, verbose=True)
    
    # Ensure results directory
    results_dir = ensure_results_dir(episode_id)
    
    # Save episode metadata for overlay renderer
    episode_meta = {
        "episode_id": episode_id,
        "video_path": str(Path(video_path).resolve()),
    }
    (results_dir / "episode_meta.json").write_text(json.dumps(episode_meta, indent=2))
    
    # Process video (Phase 1: Detection + Tracking)
    logger.info("\n[2/4] Running detection + tracking (Phase 1)...")
    result = engine.process_video(
        video_path=video_path,
        save_visualizations=True,
        output_dir=str(results_dir),
    )
    
    # Tracks are already saved by the engine to tracks.jsonl
    tracks_path = results_dir / "tracks.jsonl"
    
    # Count unique track IDs from entities
    unique_tracks = len(result.entities)
    total_detections = len(result.raw_observations)
    
    phase1_time = time.time() - start_time
    logger.info(f"\n✓ Phase 1 complete: {total_detections} detections, {unique_tracks} unique entities")
    logger.info(f"  Time: {phase1_time:.2f}s")
    
    # Phase 2: Re-ID + Memory
    logger.info("\n[3/4] Building memory (Phase 2: V-JEPA2 Re-ID)...")
    phase2_start = time.time()
    
    build_memory_from_tracks(
        episode_id=episode_id,
        video_path=Path(video_path),
        tracks_path=tracks_path,
        results_dir=results_dir,
        cosine_threshold=0.60,  # Use balanced threshold
        max_crops_per_track=5,
        class_thresholds=None,
    )
    
    # Load and count memory objects
    memory_path = results_dir / "memory.json"
    memory_data = json.loads(memory_path.read_text())
    memory_objects = len(memory_data.get("objects", []))
    
    phase2_time = time.time() - phase2_start
    logger.info(f"\n✓ Phase 2 complete: {memory_objects} memory objects")
    logger.info(f"  Time: {phase2_time:.2f}s")
    
    # Phase 3: Scene Graph (optional, using graph module)
    logger.info("\n[4/4] Building scene graph (Phase 3)...")
    phase3_start = time.time()
    
    try:
        from orion.graph import (
            build_graph_summary,
            build_scene_graphs,
            load_memory,
            load_tracks,
            save_graph_summary,
            save_scene_graphs,
        )
        
        memory = load_memory(memory_path)
        tracks = load_tracks(tracks_path)
        
        graphs = build_scene_graphs(
            memory,
            tracks,
            relations=["near", "on", "held_by"],
            enable_class_filtering=True,
        )
        
        graph_path = results_dir / "scene_graph.jsonl"
        save_scene_graphs(graphs, graph_path)
        
        summary = build_graph_summary(graphs)
        summary_path = results_dir / "graph_summary.json"
        save_graph_summary(summary, summary_path)
        
        phase3_time = time.time() - phase3_start
        logger.info(f"\n✓ Phase 3 complete: {summary.get('total_frames', 0)} frames, {summary.get('avg_edges_per_frame', 0):.1f} edges/frame")
        logger.info(f"  Time: {phase3_time:.2f}s")
        
    except Exception as e:
        logger.warning(f"Scene graph stage failed: {e}")
        phase3_time = 0
    
    total_time = time.time() - start_time
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total detections: {total_detections}")
    logger.info(f"Unique entities: {unique_tracks}")
    logger.info(f"Memory objects: {memory_objects}")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Results saved to: {results_dir}")
    
    return {
        "total_detections": total_detections,
        "unique_entities": unique_tracks,
        "memory_objects": memory_objects,
        "total_time": total_time,
        "results_dir": str(results_dir),
    }


def main():
    parser = argparse.ArgumentParser(description="Run optimized Orion v2 pipeline")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--episode", required=True, help="Episode ID for results")
    parser.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"], help="Inference device")
    
    args = parser.parse_args()
    
    if not Path(args.video).exists():
        parser.error(f"Video file not found: {args.video}")
    
    run_pipeline(args.video, args.episode, args.device)


if __name__ == "__main__":
    main()
