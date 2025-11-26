import sys
import os
import json
import logging
from pathlib import Path
import numpy as np

# Add workspace root to path
workspace_root = Path(__file__).resolve().parents[1]
sys.path.append(str(workspace_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from orion.perception.engine import PerceptionEngine
from orion.perception.config import PerceptionConfig

def run_pipeline():
    video_path = workspace_root / "data/examples/video.mp4"
    output_dir = workspace_root / "results/memgraph_run"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        print(f"Error: Video not found at {video_path}")
        return

    print(f"Processing video: {video_path}")
    print("Note: This script requires a running Memgraph instance on localhost:7687")
    
    # Initialize engine with Memgraph enabled
    config = PerceptionConfig()
    config.enable_3d = True
    config.enable_depth = True
    config.enable_tracking = True
    config.use_memgraph = True # Enable Memgraph sync
    config.memgraph_host = "127.0.0.1"
    config.memgraph_port = 7687
    
    try:
        engine = PerceptionEngine(config=config, verbose=True)
        
        if engine.memgraph is None:
            print("Warning: Memgraph backend failed to initialize. Check if Memgraph is running.")
            print("Continuing without Memgraph sync...")
        
        # Process video
        result = engine.process_video(str(video_path), save_visualizations=True, output_dir=str(output_dir))
        
        # Serialize results to JSON for analysis
        output_data = {
            "video_path": str(video_path),
            "total_frames": result.total_frames,
            "fps": result.fps,
            "duration": result.duration_seconds,
            "entities": [e.to_dict() for e in result.entities],
            "metrics": result.metrics
        }
        
        json_path = output_dir / "pipeline_output.json"
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        print(f"Results saved to {json_path}")
        print(f"Found {len(result.entities)} unique entities.")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")

if __name__ == "__main__":
    run_pipeline()
