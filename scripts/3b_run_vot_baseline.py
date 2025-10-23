#!/usr/bin/env python3
"""
STEP 3B: Run VoT Baseline on Action Genome Clips.
Generates predictions using Video-of-Thought style baseline for comparison with Orion.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import tempfile
import subprocess

sys.path.insert(0, '.')

from orion.baselines import VOTBaseline, VOTConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AG_DATASET_ROOT = 'data/ag_50'
FRAMES_DIR = os.path.join(AG_DATASET_ROOT, 'frames')
GROUND_TRUTH_FILE = os.path.join(AG_DATASET_ROOT, 'ground_truth_graphs.json')
OUTPUT_DIR = os.path.join(AG_DATASET_ROOT, 'results')
VOT_PREDICTIONS_FILE = os.path.join(OUTPUT_DIR, 'vot_predictions.json')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'intermediate'), exist_ok=True)


def create_video_from_frames(clip_id: str, frame_dir: str, fps: float = 30.0) -> Optional[str]:
    """
    Create temporary video from frame sequence using ffmpeg.
    
    Returns:
        Path to created video, or None if failed
    """
    try:
        frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
        if not frame_files:
            logger.warning(f"No frames found for {clip_id}")
            return None
        
        # Create temp video
        temp_video = os.path.join(tempfile.gettempdir(), f'{clip_id}_vot_temp.mp4')
        
        # Use ffmpeg to create video from image sequence
        frame_pattern = os.path.join(frame_dir, 'frame%04d.jpg')
        
        cmd = [
            'ffmpeg',
            '-framerate', str(fps),
            '-i', frame_pattern,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-y',
            temp_video
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        
        if result.returncode == 0 and os.path.exists(temp_video):
            logger.info(f"Created video for {clip_id}")
            return temp_video
        else:
            logger.warning(f"Failed to create video: {result.stderr.decode()[:200]}")
            return None
    
    except Exception as e:
        logger.error(f"Error creating video: {e}")
        return None


def normalize_vot_output(raw_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize VoT output to match Orion's prediction format for fair comparison.
    
    Args:
        raw_output: Raw output from VOTBaseline.process_video()
        
    Returns:
        Normalized output compatible with evaluation metrics
    """
    # VoT typically has lower-quality extractions, so we normalize carefully
    entities = raw_output.get('entities', {})
    relationships = raw_output.get('relationships', [])
    events = raw_output.get('events', [])
    causal_links = raw_output.get('causal_links', [])
    
    normalized = {
        "entities": entities,
        "relationships": relationships,
        "events": events,
        "causal_links": causal_links,
        "pipeline": "vot_baseline",
        "num_captions": raw_output.get('num_captions', 0),
        "num_scenes": raw_output.get('num_scenes', 0),
        "fps_sampled": raw_output.get('fps_sampled', 0.5),
    }
    
    return normalized


def main():
    print("="*70)
    print("STEP 3B: Run VoT Baseline on Action Genome Clips")
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
    
    if not os.path.exists(FRAMES_DIR):
        print(f"❌ Frames directory not found: {FRAMES_DIR}")
        return False
    
    print(f"\n2. Running VoT baseline on {min(50, len(ground_truth_graphs))} clips...")
    print(f"   (LLM-only captions + Gemma3 reasoning)")
    print(f"   Frames from: {FRAMES_DIR}")
    
    # Initialize VoT baseline
    config = VOTConfig(
        fps=0.5,  # Sample at 0.5 FPS as per paper
        description_model="fastvlm",
        llm_model="gemma3:4b",
        scene_window_seconds=5.0,
    )
    
    baseline = VOTBaseline(config)
    
    predictions = {}
    processed_count = 0
    failed_clips = []
    
    clips_to_process = list(ground_truth_graphs.keys())[:50]
    
    for i, clip_id in enumerate(clips_to_process):
        try:
            if (i + 1) % 10 == 0:
                print(f"   Processing clip {i+1}/{len(clips_to_process)}...")
            
            clip_frames_dir = os.path.join(FRAMES_DIR, clip_id)
            if not os.path.exists(clip_frames_dir):
                logger.warning(f"Frames not found for {clip_id}")
                failed_clips.append(clip_id)
                continue
            
            # Create video from frames
            video_path = create_video_from_frames(clip_id, clip_frames_dir, fps=30.0)
            if not video_path:
                logger.warning(f"Failed to create video for {clip_id}")
                failed_clips.append(clip_id)
                continue
            
            try:
                logger.info(f"Processing VoT: {clip_id}")
                
                # Run VoT baseline
                raw_output = baseline.process_video(video_path)
                
                # Normalize output for comparison
                pred_graph = normalize_vot_output(raw_output)
                
                predictions[clip_id] = pred_graph
                processed_count += 1
                logger.info(f"✓ Processed {clip_id}")
                
            except Exception as e:
                logger.warning(f"VoT pipeline failed for {clip_id}: {e}")
                failed_clips.append(clip_id)
                continue
            
            finally:
                # Clean up temp video
                try:
                    if video_path and os.path.exists(video_path):
                        os.remove(video_path)
                except:
                    pass
        
        except Exception as e:
            logger.error(f"Error processing {clip_id}: {e}")
            failed_clips.append(clip_id)
    
    print(f"   ✓ Successfully processed {processed_count}/{len(clips_to_process)} clips")
    
    if failed_clips:
        print(f"   ⚠️  Failed: {len(failed_clips)} clips")
    
    print(f"\n3. Saving predictions...")
    with open(VOT_PREDICTIONS_FILE, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    # Summary stats
    total_entities = sum(len(p.get('entities', {})) for p in predictions.values())
    total_relationships = sum(len(p.get('relationships', [])) for p in predictions.values())
    total_events = sum(len(p.get('events', [])) for p in predictions.values())
    
    print(f"\n" + "="*70)
    print(f"STEP 3B COMPLETE: VoT Baseline")
    print(f"="*70)
    print(f"""
✓ VoT predictions saved to: {VOT_PREDICTIONS_FILE}

VoT Baseline Results (LLM-only Captions):
  Clips processed: {processed_count}
  Total entities: {total_entities}
  Total relationships: {total_relationships}
  Total events: {total_events}
  Avg entities/clip: {total_entities/processed_count if processed_count else 0:.1f}
  Avg relationships/clip: {total_relationships/processed_count if processed_count else 0:.1f}

Note: This baseline uses free-form caption reasoning without:
  - Structured embeddings
  - Entity tracking/clustering
  - Explicit causal inference
  
This demonstrates the limitations of pure caption-based reasoning.

Next: Compare VoT against Orion
   python scripts/4_evaluate_ag_predictions.py --include-baseline
""")
    
    return processed_count > 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
