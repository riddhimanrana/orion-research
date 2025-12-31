import argparse
import sys
import os
import json
import logging
from pathlib import Path
from dataclasses import asdict
import numpy as np

# Add workspace root to path
workspace_root = Path(__file__).resolve().parents[1]
sys.path.append(str(workspace_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from orion.perception.engine import PerceptionEngine
from orion.perception.config import get_accurate_config, get_balanced_config, get_fast_config, PerceptionConfig

def get_config_by_name(name: str) -> PerceptionConfig:
    """Returns a perception config by its preset name."""
    if name == "accurate":
        return get_accurate_config()
    if name == "balanced":
        return get_balanced_config()
    if name == "fast":
        return get_fast_config()
    raise ValueError(f"Unknown config name: {name}")

def run_pipeline(video_path_str: str, output_dir_str: str, config_name: str, save_tracks: bool):
    """
    Main function to run the perception pipeline.
    """
    video_path = Path(video_path_str)
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        logging.error(f"Error: Video not found at {video_path}")
        return

    logging.info("Note: This script can connect to a running Memgraph instance on localhost:7687 if use_memgraph is enabled in the config.")
    
    # Initialize engine with the specified config
    try:
        config = get_config_by_name(config_name)
        config.enable_tracking = True
        config.use_memgraph = True  # Attempt to use Memgraph
        
        logging.info(f"Running pipeline with '{config_name}' configuration.")
        
        engine = PerceptionEngine(config=config)
        
        if config.use_memgraph and engine.memgraph is None:
            logging.warning("Memgraph backend failed to initialize. Check if Memgraph is running.")
            logging.warning("Continuing without Memgraph sync...")
        
        # Process video
        result = engine.process_video(str(video_path), save_visualizations=save_tracks, output_dir=str(output_dir))
        
        # Serialize results to JSON for analysis
        output_data = {
            "video_path": str(video_path),
            "config": asdict(config),
            "total_frames": result.total_frames,
            "fps": result.fps,
            "duration": result.duration_seconds,
            "entities": [asdict(e) for e in result.entities],
        }
        
        # Save the main results file
        output_json_path = output_dir / "perception_run.json"
        with open(output_json_path, "w") as f:
            json.dump(output_data, f, indent=2, cls=NumpyEncoder)

        # Also save the config separately for easy reference
        config_json_path = output_dir / "pipeline_output.json"
        with open(config_json_path, "w") as f:
            json.dump({"config": asdict(config)}, f, indent=2)
            
        logging.info(f"✅ Perception run complete. Results saved to {output_dir}")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        if isinstance(obj, (np.bool_)):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Orion perception pipeline with Memgraph integration.")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory to save results.")
    parser.add_argument("--config-name", type=str, default="accurate", choices=["fast", "balanced", "accurate"], help="Name of the config to use.")
    parser.add_argument("--save-tracks", action="store_true", help="Save visualization of tracks.")
    
    args = parser.parse_args()
    
    run_pipeline(
        video_path_str=args.video,
        output_dir_str=args.results_dir,
        config_name=args.config_name,
        save_tracks=args.save_tracks
    )
