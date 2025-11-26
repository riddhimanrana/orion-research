import argparse
import json
import logging
from pathlib import Path
import sys
import time

# Add workspace root to path
workspace_root = Path(__file__).resolve().parents[1]
sys.path.append(str(workspace_root))

from orion.perception.engine import PerceptionEngine
from orion.perception.config import PerceptionConfig
from orion.data.dataset import JsonDataset, ActionGenomeDataset
from orion.graph.scene_graph import build_research_scene_graph
from orion.utils.profiling import Profiler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run Orion on a dataset")
    parser.add_argument("--dataset_type", type=str, default="json", choices=["json", "action_genome"], help="Type of dataset")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset JSON or root dir")
    parser.add_argument("--data_root", type=str, help="Root directory for video files (for json dataset)")
    parser.add_argument("--output_dir", type=str, default="results/dataset_run", help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos to process")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    if args.dataset_type == "json":
        if not args.data_root:
            logger.error("--data_root is required for json dataset")
            return
        dataset = JsonDataset(args.dataset_path, args.data_root)
    elif args.dataset_type == "action_genome":
        dataset = ActionGenomeDataset(args.dataset_path)
        
    logger.info(f"Loaded dataset with {len(dataset)} videos")
    
    # Initialize engine
    config = PerceptionConfig()
    # config.target_fps = 2 # Lower FPS for efficiency during testing
    engine = PerceptionEngine(config=config)
    
    profiler = Profiler()
    
    results = []
    
    count = 0
    for item in dataset:
        if args.limit and count >= args.limit:
            break
            
        video_id = item["video_id"]
        video_path = item["video_path"]
        
        logger.info(f"Processing video {count+1}/{len(dataset)}: {video_id}")
        
        try:
            # Run perception
            perception_result = engine.process_video(
                video_path, 
                save_visualizations=False,
                output_dir=str(output_dir / video_id)
            )
            
            # Build Scene Graph
            # We need to reconstruct 'memory' and 'tracks' from perception_result
            # This is a bit of a hack since build_scene_graphs expects specific dict formats
            # Ideally we should refactor build_scene_graphs to accept PerceptionResult
            
            # Construct 'memory' dict from entities
            memory = {"objects": []}
            for ent in perception_result.entities:
                memory["objects"].append({
                    "memory_id": str(ent.entity_id),
                    "class": ent.object_class,
                    "prototype_embedding": str(ent.entity_id) # Dummy mapping
                })
                
            # Construct 'tracks' list from observations
            tracks = []
            for obs in perception_result.raw_observations:
                tracks.append({
                    "embedding_id": str(obs.entity_id), # Map back to memory_id
                    "bbox": obs.bounding_box.to_list(),
                    "frame_id": obs.frame_number,
                    "frame_width": 1920, # TODO: Get actual dims
                    "frame_height": 1080
                })
                
            sg = build_research_scene_graph(memory, tracks, video_id=video_id)
            
            # Save SG
            sg_path = output_dir / f"{video_id}_sg.json"
            with open(sg_path, 'w') as f:
                json.dump(sg.to_dict(), f, indent=2)
                
            results.append({
                "video_id": video_id,
                "status": "success",
                "sg_path": str(sg_path)
            })
            
        except Exception as e:
            logger.error(f"Failed to process {video_id}: {e}")
            results.append({
                "video_id": video_id,
                "status": "failed",
                "error": str(e)
            })
            
        count += 1
        
    # Save aggregate results
    with open(output_dir / "run_summary.json", 'w') as f:
        json.dump(results, f, indent=2)
        
    # Save profiling stats
    profiler.save_stats(output_dir / "profiling_stats.json")
    
if __name__ == "__main__":
    main()
