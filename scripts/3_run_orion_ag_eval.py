#!/usr/bin/env python3
"""
STEP 3: Run Orion pipeline on Action Genome clips.
Runs full perception + semantic graph generation on AG video frames.

Requirements:
- Action Genome videos in dataset/ag/videos/
- Ground truth from step 1: data/ag_50/ground_truth_graphs.json
- Neo4j running (optional, can use dummy credentials)
- Ollama running with required models

Note: This script uses the original Action Genome videos directly.
If Neo4j is not needed for your evaluation, dummy credentials are passed.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import tempfile
import subprocess

sys.path.insert(0, ".")

from orion.run_pipeline import run_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AG_DATASET_ROOT = "data/ag_50"
AG_SOURCE_ROOT = "dataset/ag"
VIDEOS_DIR = os.path.join(AG_SOURCE_ROOT, "videos")
GROUND_TRUTH_FILE = os.path.join(AG_DATASET_ROOT, "ground_truth_graphs.json")
OUTPUT_DIR = os.path.join(AG_DATASET_ROOT, "results")
PREDICTIONS_FILE = os.path.join(OUTPUT_DIR, "predictions.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "intermediate"), exist_ok=True)


def main():
    print("=" * 70)
    print("STEP 3: Run Orion Pipeline on Action Genome Clips")
    print("=" * 70)

    # Load ground truth
    if not os.path.exists(GROUND_TRUTH_FILE):
        print(f"\n❌ Ground truth not found: {GROUND_TRUTH_FILE}")
        print(f"   Run: python scripts/1_prepare_ag_data.py")
        return False

    print(f"\n1. Loading ground truth...")
    with open(GROUND_TRUTH_FILE, "r") as f:
        ground_truth_graphs = json.load(f)

    print(f"   ✓ Loaded {len(ground_truth_graphs)} clips")

    # Check for videos
    if not os.path.exists(VIDEOS_DIR):
        print(f"❌ Videos directory not found: {VIDEOS_DIR}")
        return False

    print(f"\n2. Running Orion pipeline on {min(50, len(ground_truth_graphs))} clips...")
    print(f"   (Full perception + semantic graph generation)")
    print(f"   Videos from: {VIDEOS_DIR}")
    print(f"   Note: Using dummy Neo4j credentials for evaluation")

    predictions = {}
    processed_count = 0
    failed_clips = []

    clips_to_process = list(ground_truth_graphs.keys())[:50]

    for i, clip_id in enumerate(clips_to_process):
        try:
            if (i + 1) % 10 == 0:
                print(f"   Processing clip {i + 1}/{len(clips_to_process)}...")

            # Use existing video file
            video_path = os.path.join(VIDEOS_DIR, clip_id)
            if not os.path.exists(video_path):
                logger.warning(f"Video not found for {clip_id}")
                failed_clips.append(clip_id)
                continue

            try:
                logger.info(f"Processing: {clip_id}")

                # Run Orion pipeline
                output_graph = run_pipeline(
                    video_path=video_path,
                    output_dir=os.path.join(OUTPUT_DIR, "intermediate", clip_id),
                    neo4j_uri="bolt://localhost:7687",
                    neo4j_user="neo4j",
                    neo4j_password="orion123",
                    clear_db=False,
                    part1_config="balanced",
                    part2_config="balanced",
                    skip_part1=False,
                    skip_part2=False,
                    verbose=False,
                    use_progress_ui=False,
                )

                # Extract results
                if isinstance(output_graph, dict):
                    pred_graph = {
                        "entities": output_graph.get("part2", {}).get("entities", {}),
                        "relationships": output_graph.get("part2", {}).get("relationships", []),
                        "events": output_graph.get("part2", {}).get("events", []),
                        "causal_links": output_graph.get("part2", {}).get("causal_links", []),
                    }
                else:
                    pred_graph = output_graph if isinstance(output_graph, dict) else {}

                predictions[clip_id] = pred_graph
                processed_count += 1
                logger.info(f"✓ Processed {clip_id}")

            except Exception as e:
                logger.warning(f"Pipeline failed for {clip_id}: {e}")
                failed_clips.append(clip_id)
                continue

        except Exception as e:
            logger.error(f"Error processing {clip_id}: {e}")
            failed_clips.append(clip_id)

    print(f"   ✓ Successfully processed {processed_count}/{len(clips_to_process)} clips")

    if failed_clips:
        print(f"   ⚠️  Failed: {len(failed_clips)} clips")

    print(f"\n3. Saving predictions...")
    with open(PREDICTIONS_FILE, "w") as f:
        json.dump(predictions, f, indent=2)

    # Summary stats
    total_entities = sum(len(p.get("entities", {})) for p in predictions.values())
    total_relationships = sum(len(p.get("relationships", [])) for p in predictions.values())
    total_events = sum(len(p.get("events", [])) for p in predictions.values())

    print(f"\n" + "=" * 70)
    print(f"STEP 3 COMPLETE")
    print(f"=" * 70)
    print(f"""
✓ Predictions saved to: {PREDICTIONS_FILE}

Orion Pipeline Results:
  Clips processed: {processed_count}
  Total entities: {total_entities}
  Total relationships: {total_relationships}
  Total events: {total_events}
  Avg entities/clip: {total_entities / processed_count if processed_count else 0:.1f}
  Avg relationships/clip: {total_relationships / processed_count if processed_count else 0:.1f}

Next: Evaluate predictions against ground truth
   python scripts/4_evaluate_ag_predictions.py
""")

    return processed_count > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
