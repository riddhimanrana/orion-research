import json
import sys
from pathlib import Path
import logging

# Add workspace root to path
workspace_root = Path(__file__).resolve().parents[1]
sys.path.append(str(workspace_root))

from orion.query.index import VideoIndex, EntityObservation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import argparse

def populate_index():
    parser = argparse.ArgumentParser(description="Populate video index from pipeline output")
    parser.add_argument("--results_dir", type=Path, default=workspace_root / "results/full_video_analysis", help="Directory containing pipeline_output.json")
    parser.add_argument("--video_path", type=Path, default=workspace_root / "data/examples/video.mp4", help="Path to the video file")
    args = parser.parse_args()

    results_dir = args.results_dir
    json_path = results_dir / "pipeline_output.json"
    index_path = results_dir / "video_index.db"
    video_path = args.video_path
    
    if not json_path.exists():
        logger.error(f"Results not found at {json_path}")
        return

    logger.info(f"Loading results from {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Initialize index
    if index_path.exists():
        index_path.unlink()
    
    index = VideoIndex(index_path, video_path)
    index.create_schema()
    
    entities = data.get("entities", [])
    count = 0
    
    logger.info(f"Indexing {len(entities)} entities...")
    
    for entity in entities:
        # Parse entity ID (e.g., "merged_entity_18" -> 18)
        try:
            entity_id_str = entity.get("entity_id", "0")
            if "merged_entity_" in entity_id_str:
                entity_id = int(entity_id_str.replace("merged_entity_", ""))
            elif "entity_" in entity_id_str:
                entity_id = int(entity_id_str.replace("entity_", ""))
            else:
                entity_id = hash(entity_id_str) % 100000
        except:
            entity_id = 0
            
        class_name = entity.get("object_class", "unknown")
        description = entity.get("description", "")
        
        # Add observations
        for obs in entity.get("observations", []):
            # Map observation data to EntityObservation
            # Note: JSON output might have different keys than EntityObservation expects
            # We need to adapt
            
            bbox = obs.get("bbox", [0,0,0,0])
            
            observation = EntityObservation(
                entity_id=entity_id,
                frame_idx=obs.get("frame", 0),
                timestamp=obs.get("timestamp", 0.0),
                class_name=class_name,
                confidence=obs.get("confidence", 0.0),
                bbox=bbox,
                zone_id=obs.get("spatial_zone"), # Might be None
                pose=None, # 3D pose if available
                caption=description if obs.get("frame") == entity.get("description_frame") else None
            )
            
            index.add_observation(observation)
            count += 1
            
    index.conn.commit()
    logger.info(f"Successfully indexed {count} observations.")
    logger.info(f"Index saved to {index_path}")

if __name__ == "__main__":
    populate_index()
