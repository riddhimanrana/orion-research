"""Test full pipeline: Perception + Semantic + CLIP class correction"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orion.perception.config import get_accurate_config
from orion.perception.engine import PerceptionEngine
import json
from collections import Counter

print("="*80)
print("FULL PIPELINE TEST: Perception + Semantic + CLIP Class Correction")
print("="*80)

# Test video
video_path = "data/examples/room.mp4"
output_dir = "results/full_pipeline"

print(f"\nVideo: {video_path}")
print(f"Output: {output_dir}\n")

# ====================================================================
# STEP 1: Perception (with CLIP class correction)
# ====================================================================
print("STEP 1: Running Perception Engine...")
print("-" * 80)

config = get_accurate_config()
config.target_fps = 0.25  # Process 1 frame every 4 seconds
config.enable_3d = False  # Skip 3D for faster processing
config.enable_tracking = True
config.yolo_model = "yolo11n"  # Use faster nano model
# Use CLIP backend for embeddings (needed for CLIP class correction)
config.embedding.backend = "clip"

engine = PerceptionEngine(config)
perception_result = engine.process_video(video_path, save_visualizations=True, output_dir=output_dir)

print(f"\n✓ Perception complete:")
print(f"  Detections: {perception_result.total_detections}")
print(f"  Entities: {perception_result.unique_entities}")

# ====================================================================
# STEP 2: CLIP Class Correction
# ====================================================================
print("\n" + "="*80)
print("STEP 2: Class Correction Summary")
print("-" * 80)

class_metrics = (perception_result.metrics or {}).get("class_correction") if perception_result.metrics else None
if class_metrics:
    print(
        f"✓ Class correction updated {class_metrics['corrected_entities']} entities "
        f"(min similarity {class_metrics['min_similarity']:.2f})"
    )
else:
    print("⚠ Class correction disabled or no updates recorded")

# ====================================================================
# STEP 3: CIS Summary (Perception-integrated)
# ====================================================================
print("\n" + "="*80)
print("STEP 3: Causal Influence Summary")
print("-" * 80)

cis_metrics = (perception_result.metrics or {}).get("cis") if perception_result.metrics else None
if cis_metrics and cis_metrics.get("links"):
    print(f"✓ CIS enabled: {cis_metrics['link_count']} links scored")
    preview = cis_metrics["links"][:5]
    for idx, link in enumerate(preview, start=1):
        print(
            f"  {idx}. {link['agent_id']} → {link['patient_id']} (score={link['influence_score']:.2f})"
        )
        print(f"     justification: {link['justification']}")
    if cis_metrics["link_count"] > len(preview):
        print(f"  … {cis_metrics['link_count'] - len(preview)} more links in metrics")
else:
    print("⚠ No CIS links were generated (check enable_cis flag or entity count)")

# ====================================================================
# STEP 4: Summary & Export
# ====================================================================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

# ====================================================================
# FINAL SUMMARY
# ====================================================================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

print("\nEntity Class Distribution (after CLIP correction):")
class_counts = Counter()
for entity in perception_result.entities:
    class_counts[entity.display_class()] += 1

for cls in sorted(class_counts.keys()):
    print(f"  {cls}: {class_counts[cls]}")

print("\n" + "="*80)
print("✓ Full pipeline test complete!")
print(f"  Results saved to: {output_dir}/")
print("="*80)
