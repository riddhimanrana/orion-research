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
    video_path = workspace_root / "data/examples/test.mp4"
    output_dir = workspace_root / "results/test_run"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        print(f"Error: Video not found at {video_path}")
        return

    print(f"Processing video: {video_path}")
    
    # Initialize engine with 3D and SLAM enabled
    config = PerceptionConfig()
    config.enable_3d = True
    config.enable_depth = True
    config.enable_tracking = True
    # config.enable_slam = True # Assuming this is part of enable_3d or separate, checking config
    
    engine = PerceptionEngine(config=config, verbose=True)
    
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
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
        
    print(f"Results saved to {json_path}")

if __name__ == "__main__":
    # Enable MPS fallback for DepthAnything3
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    run_pipeline()
