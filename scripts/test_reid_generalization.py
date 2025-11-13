"""Test Re-ID on test.mp4 to verify we're not overfitting"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orion.perception.config import get_accurate_config
from orion.perception.engine import PerceptionEngine
import json
from collections import Counter

print("="*80)
print("TESTING RE-ID ON test.mp4 (GENERALIZATION TEST)")
print("="*80)

config = get_accurate_config()
config.target_fps = 0.5  # Sparse sampling for speed
config.enable_3d = True
config.enable_tracking = True

engine = PerceptionEngine(config)
result = engine.process_video("data/examples/test.mp4", save_visualizations=True, output_dir="results/test_video")

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"Total detections: {result.total_detections}")
print(f"Unique entities: {result.unique_entities}")
print(f"Reduction: {100*(1-result.unique_entities/result.total_detections):.1f}%")

# Load and analyze entities
with open("results/test_video/entities.json") as f:
    data = json.load(f)

class_counts = Counter([e['class'] for e in data['entities']])
print("\nEntity breakdown by class:")
for cls in sorted(class_counts.keys()):
    count = class_counts[cls]
    entities = [e for e in data['entities'] if e['class'] == cls]
    print(f"  {cls}: {count}")
    if count > 1:
        for e in entities:
            print(f"    - entity {e['id']}: {e['observation_count']} obs, frames {e['first_frame']}-{e['last_frame']}")

print("="*80)
