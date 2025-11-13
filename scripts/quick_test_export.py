"""Quick test to verify export functionality"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orion.perception.config import get_accurate_config
from orion.perception.engine import PerceptionEngine

config = get_accurate_config()
config.target_fps = 0.5  # Very sparse sampling for speed
config.enable_3d = True
config.enable_tracking = True

engine = PerceptionEngine(config)
result = engine.process_video("data/examples/video.mp4", save_visualizations=True, output_dir="results")

print(f"\nDone: {result.unique_entities} entities, {result.total_detections} detections")
print("Check results/ for exported files")
