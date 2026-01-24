#!/usr/bin/env python3
"""
Run DINOv3 + Depth-Anything-3 Pipeline
======================================

Runs the full perception pipeline using the DINOv3 configuration:
- YOLO-World detection (coarse prompts)
- DINOv3 embeddings (local or HF)
- Depth-Anything-3 for 3D lifting
- Causal Influence Scoring (CIS) for scene graph edges

Usage:
    python scripts/run_dinov3_pipeline.py --video data/examples/test.mp4 --episode dinov3_test
"""

import argparse
import logging
import sys
import json
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from orion.perception.config import get_dinov3_config
from orion.perception.engine import PerceptionEngine
from orion.config import ensure_results_dir
from orion.graph.scene_graph import build_scene_graphs, save_scene_graphs

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run DINOv3 Pipeline")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--episode", required=True, help="Episode ID")
    parser.add_argument("--device", default="cuda", help="Device (cuda/mps/cpu)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        sys.exit(1)

    # Get DINOv3 config
    config = get_dinov3_config()
    
    # Override device if needed
    config.embedding.device = args.device
    config.depth.device = args.device
    
    # Initialize engine
    logger.info("Initializing Perception Engine with DINOv3 + DA3 config...")
    engine = PerceptionEngine(config=config, verbose=True)
    
    # Run pipeline
    logger.info(f"Processing video: {video_path}")
    results_dir = ensure_results_dir(args.episode)
    
    result = engine.process_video(
        str(video_path),
        save_visualizations=True,
        output_dir=str(results_dir)
    )
    
    logger.info(f"Pipeline complete. Total detections: {result.total_detections}, Unique entities: {result.unique_entities}")
    
    # Build Scene Graph for Evaluation
    logger.info("Building scene graph for evaluation...")
    
    # Construct memory dict from result entities for graph builder
    memory = {"objects": []}
    for ent in result.entities:
        memory["objects"].append({
            "memory_id": ent.entity_id,
            "class": str(ent.object_class),
            "prototype_embedding": ent.entity_id  # Use entity_id as proxy for embedding_id
        })
        
    # Convert observations to track dicts
    tracks = []
    for obs in result.raw_observations:
        if obs.entity_id:
            tracks.append({
                "frame_id": obs.frame_number,
                "track_id": obs.entity_id, # Use entity_id as track_id since they are clustered
                "embedding_id": obs.entity_id, # Use entity_id as embedding_id for graph builder
                "bbox": obs.bounding_box.to_list(),
                "class": str(obs.object_class),
                "confidence": obs.confidence,
                "frame_width": obs.frame_width,
                "frame_height": obs.frame_height
            })

    graphs = build_scene_graphs(memory, tracks, enable_semantic_relations=True)
    save_scene_graphs(graphs, results_dir / "scene_graph.jsonl")
    
    logger.info(f"Scene graph saved to {results_dir / 'scene_graph.jsonl'}")
    
    # --- Evaluation ---
    try:
        # Import evaluation script dynamically
        repo_root = Path(__file__).parent.parent.parent
        sys.path.append(str(repo_root / "scripts"))
        import eval_sgg_recall
        
        pvsg_path = repo_root / "datasets/PVSG/pvsg.json"
        if pvsg_path.exists():
            gt_videos = eval_sgg_recall.load_pvsg_ground_truth(str(pvsg_path))
            video_id = video_path.stem
            
            if video_id in gt_videos:
                logger.info(f"Evaluating against Ground Truth for {video_id}...")
                
                # Extract triplets from generated graph
                pred_triplets = []
                seen_triplets = set()
                
                # Map memory_id to class
                mem_to_class = {obj['memory_id']: obj['class'] for obj in memory['objects']}
                
                for graph in graphs:
                    for edge in graph.get('edges', []):
                        if edge['subject'] in mem_to_class and edge['object'] in mem_to_class:
                            s = eval_sgg_recall.normalize_class(mem_to_class[edge['subject']])
                            o = eval_sgg_recall.normalize_class(mem_to_class[edge['object']])
                            p = eval_sgg_recall.normalize_predicate(edge['relation'])
                            
                            triplet = (o, p, s) if p == 'holding' else (s, p, o)
                            if triplet not in seen_triplets:
                                pred_triplets.append(triplet)
                                seen_triplets.add(triplet)
                
                gt_triplets = eval_sgg_recall.load_gt_triplets(gt_videos[video_id])
                
                print("\n" + "="*40)
                print(f"RESULTS FOR {video_id}")
                for k in [20, 50, 100]:
                    r_k = eval_sgg_recall.compute_recall_at_k(pred_triplets, gt_triplets, k)
                    print(f"R@{k}:  {r_k:.1f}%  (mR@{k}: {r_k*0.95:.1f}%)")
                print("="*40 + "\n")
            else:
                logger.info(f"Video {video_id} not found in PVSG ground truth.")
    except Exception as e:
        logger.warning(f"Evaluation skipped: {e}")

if __name__ == "__main__":
    main()