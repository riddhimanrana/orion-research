#!/usr/bin/env python3
"""
STEP 1: Prepare Action Genome ground truth for Orion evaluation.
Loads Action Genome benchmark and converts to GroundTruthGraph.
"""

import json
import logging
import os
import sys
from dataclasses import asdict
from typing import Dict, Any
import numpy as np

sys.path.insert(0, ".")

from orion.evaluation.benchmarks.action_genome_loader import ActionGenomeBenchmark
from orion.evaluation.ag_adapter import ActionGenomeAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AG_DATASET_ROOT = "dataset/ag"
OUTPUT_DIR = "data/ag_50"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def main():
    print("=" * 70)
    print("STEP 1: Prepare Action Genome Ground Truth")
    print("=" * 70)

    # Check dataset exists
    if not os.path.exists(AG_DATASET_ROOT):
        print(f"\n❌ Action Genome dataset not found at: {AG_DATASET_ROOT}")
        print(f"\n   Expected structure:")
        print(f"   {AG_DATASET_ROOT}/")
        print(f"   ├── videos/           # Charades 480p videos")
        print(f"   ├── annotations/      # AG annotation files (.pkl, .txt)")
        print(f"   └── frames/           # Extracted frames")
        return False

    print(f"\n1. Loading Action Genome benchmark...")
    try:
        benchmark = ActionGenomeBenchmark(AG_DATASET_ROOT)
        print(f"   ✓ Loaded {len(benchmark.clips)} clips")
    except Exception as e:
        print(f"   ❌ Error loading benchmark: {e}")
        return False

    print(f"\n2. Converting to GroundTruthGraphs (first 50 clips)...")
    adapter = ActionGenomeAdapter(fps=30.0)

    ground_truth_graphs = {}

    for i, (clip_id, ag_dataset) in enumerate(list(benchmark.clips.items())[:50]):
        try:
            gt_graph = adapter.convert_to_ground_truth(ag_dataset)
            # Convert dataclass to dict
            ground_truth_graphs[clip_id] = asdict(gt_graph)

            if (i + 1) % 10 == 0:
                print(f"   Converted {i + 1} clips...")
        except Exception as e:
            logger.warning(f"Failed to convert clip {clip_id}: {e}")
            continue

    print(f"   ✓ Converted {len(ground_truth_graphs)} clips successfully")

    print(f"\n3. Saving ground truth...")
    gt_file = f"{OUTPUT_DIR}/ground_truth_graphs.json"
    with open(gt_file, "w") as f:
        json.dump(ground_truth_graphs, f, indent=2, cls=NumpyEncoder)

    # Summary stats
    if len(ground_truth_graphs) > 0:
        total_entities = sum(len(gt.get("entities", {})) for gt in ground_truth_graphs.values())
        total_relationships = sum(len(gt.get("relationships", [])) for gt in ground_truth_graphs.values())
        total_events = sum(len(gt.get("events", [])) for gt in ground_truth_graphs.values())

        print(f"\n" + "=" * 70)
        print(f"STEP 1 COMPLETE")
        print(f"=" * 70)
        print(f"""
✓ Ground truth saved to: {gt_file}

Action Genome Ground Truth Summary:
  Clips: {len(ground_truth_graphs)}
  Total entities: {total_entities}
  Total relationships: {total_relationships}
  Total events: {total_events}
  Avg entities/clip: {total_entities / len(ground_truth_graphs):.1f}
  Avg relationships/clip: {total_relationships / len(ground_truth_graphs):.1f}

Next: Run Orion pipeline on frames
   python scripts/3_run_orion_ag_eval.py
""")
    else:
        print(f"\n" + "=" * 70)
        print(f"STEP 1 COMPLETE (WARNING)")
        print(f"=" * 70)
        print(f"""
⚠️  No clips were successfully converted!

Ground truth saved to: {gt_file} (empty)

Please check:
  1. Are the annotation pickle files correctly formatted?
  2. Do the video files exist in {AG_DATASET_ROOT}/videos/?
  3. Check the logs above for any errors during conversion.
""")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
