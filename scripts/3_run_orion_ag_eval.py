#!/usr/bin/env python3
"""
STEP 3: Run Orion pipeline on Action Genome clips.
Runs full perception + semantic graph generation on 50 AG video clips.
"""

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
import subprocess

sys.path.insert(0, '.')

from orion.run_pipeline import run_pipeline
from orion.evaluation.orion_adapter import OrionKGAdapter
from orion.config_manager import ConfigManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AG_DATASET_ROOT = 'dataset/ag'
VIDEOS_DIR = f'{AG_DATASET_ROOT}/videos'
FRAMES_DIR = f'{AG_DATASET_ROOT}/frames'
GROUND_TRUTH_FILE = 'data/ag_50/ground_truth_graphs.json'
OUTPUT_DIR = 'data/ag_50/results'
PREDICTIONS_FILE = f'{OUTPUT_DIR}/predictions.json'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/intermediate', exist_ok=True)


def find_video_for_clip(clip_id: str) -> Optional[str]:
    """Find video file for a clip ID."""
    # Try common video extensions
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        video_path = f'{VIDEOS_DIR}/{clip_id}{ext}'
        if os.path.exists(video_path):
            return video_path
    
    # Try looking in subdirectories (e.g., Charades/)
    for root, dirs, files in os.walk(VIDEOS_DIR):
        for file in files:
            if file.startswith(clip_id) and file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                return os.path.join(root, file)
    
    return None


def create_video_from_frames(clip_id: str, frame_dir: str, output_video: str, fps: float = 30.0) -> bool:
    """
    Create a temporary video from frame sequence using ffmpeg.
    
    Args:
        clip_id: Clip identifier
        frame_dir: Directory containing frames
        output_video: Output video file path
        fps: Frames per second
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Find frame files
        frame_pattern = f'{frame_dir}/{clip_id}_*.jpg'
        
        # Use ffmpeg to create video from image sequence
        cmd = [
            'ffmpeg',
            '-framerate', str(fps),
            '-pattern_type', 'glob',
            '-i', frame_pattern,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-y',  # Overwrite output file
            output_video
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        
        if result.returncode == 0 and os.path.exists(output_video):
            logger.info(f"Created video for {clip_id}: {output_video}")
            return True
        else:
            logger.warning(f"Failed to create video for {clip_id}: {result.stderr.decode()}")
            return False
    
    except Exception as e:
        logger.error(f"Error creating video from frames for {clip_id}: {e}")
        return False


def main():
    print("="*70)
    print("STEP 3: Run Orion Pipeline on Action Genome Clips")
    print("="*70)
    
    # Load ground truth
    if not os.path.exists(GROUND_TRUTH_FILE):
        print(f"\n❌ Ground truth not found: {GROUND_TRUTH_FILE}")
        print(f"   Run: python scripts/1_prepare_ag_data.py")
        return False
    
    print(f"\n1. Loading ground truth...")
    with open(GROUND_TRUTH_FILE, 'r') as f:
        ground_truth_graphs = json.load(f)
    
    print(f"   ✓ Loaded {len(ground_truth_graphs)} clips")
    
    # Get Neo4j config for adapter
    config = ConfigManager.get_config()
    
    print(f"\n2. Running Orion pipeline on first 50 clips...")
    print(f"   (Full perception + semantic graph generation)")
    print(f"   Videos from: {VIDEOS_DIR}")
    print(f"   Frames from: {FRAMES_DIR}")
    
    predictions = {}
    processed_count = 0
    failed_clips = []
    
    clips_to_process = list(ground_truth_graphs.items())[:50]
    
    for i, (clip_id, gt_graph) in enumerate(clips_to_process):
        try:
            if (i + 1) % 10 == 0:
                print(f"   Processing clip {i+1}/50...")
            
            # Strategy 1: Try to find existing video file
            video_path = find_video_for_clip(clip_id)
            
            # Strategy 2: Try to create video from frame sequence
            if not video_path:
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                    temp_video = tmp.name
                
                if create_video_from_frames(clip_id, FRAMES_DIR, temp_video):
                    video_path = temp_video
                else:
                    logger.warning(f"Could not find or create video for {clip_id}")
                    failed_clips.append(clip_id)
                    continue
            
            # Run Orion pipeline on video
            try:
                print(f"      🎬 Processing: {clip_id}")
                
                output_graph = run_pipeline(
                    video_path=video_path,
                    output_dir=f'{OUTPUT_DIR}/intermediate/{clip_id}',
                    part1_config='balanced',  # Perception config
                    part2_config='balanced',  # Semantic uplift config
                    skip_part1=False,
                    skip_part2=False,
                    verbose=False,
                    use_progress_ui=False
                )
                
                # Extract results from output
                if isinstance(output_graph, dict):
                    # Convert from run_pipeline output format
                    pred_graph = {
                        "entities": output_graph.get("part2", {}).get("entities", {}),
                        "relationships": output_graph.get("part2", {}).get("relationships", []),
                        "events": output_graph.get("part2", {}).get("events", []),
                        "causal_links": output_graph.get("part2", {}).get("causal_links", []),
                    }
                else:
                    pred_graph = output_graph
                
                predictions[clip_id] = pred_graph
                processed_count += 1
                
                logger.info(f"✓ Processed {clip_id}")
                
            except Exception as e:
                logger.warning(f"Pipeline execution failed for {clip_id}: {e}")
                failed_clips.append(clip_id)
                continue
            
            finally:
                # Clean up temp video if created
                if video_path and video_path.startswith(tempfile.gettempdir()):
                    try:
                        os.remove(video_path)
                    except:
                        pass
        
        except Exception as e:
            logger.error(f"Error processing clip {clip_id}: {e}")
            failed_clips.append(clip_id)
            continue
    
    print(f"   ✓ Successfully processed {processed_count}/50 clips")
    
    if failed_clips:
        print(f"   ⚠️  Failed to process {len(failed_clips)} clips:")
        for clip_id in failed_clips[:5]:
            print(f"      - {clip_id}")
        if len(failed_clips) > 5:
            print(f"      ... and {len(failed_clips) - 5} more")
    
    print(f"\n3. Saving predictions...")
    with open(PREDICTIONS_FILE, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    # Summary stats
    total_entities = sum(len(p.get('entities', {})) for p in predictions.values())
    total_relationships = sum(len(p.get('relationships', [])) for p in predictions.values())
    total_events = sum(len(p.get('events', [])) for p in predictions.values())
    total_causal = sum(len(p.get('causal_links', [])) for p in predictions.values())
    
    print(f"\n" + "="*70)
    print(f"STEP 3 COMPLETE")
    print(f"="*70)
    print(f"""
✓ Predictions saved to: {PREDICTIONS_FILE}

Orion Pipeline Results:
  Clips processed: {processed_count}
  Total entities detected: {total_entities}
  Total relationships inferred: {total_relationships}
  Total events identified: {total_events}
  Total causal links: {total_causal}
  Avg entities/clip: {total_entities/processed_count if processed_count else 0:.1f}
  Avg relationships/clip: {total_relationships/processed_count if processed_count else 0:.1f}
  Avg events/clip: {total_events/processed_count if processed_count else 0:.1f}

Next: Evaluate predictions against ground truth
   python scripts/4_evaluate_ag_predictions.py
""")
    
    return processed_count > 0

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
