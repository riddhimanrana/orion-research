#!/usr/bin/env python3
"""
Run Orion Perception Pipeline on VSGR Dataset
==============================================

Processes VSGR videos through Orion's perception pipeline to generate:
- Agent candidates (tracked entities with motion data)
- State changes (detected events)
- CIS component scores (temporal, spatial, motion, semantic)

This provides the predictions needed for CIS optimization.

Usage:
    # Process all videos in training split
    python scripts/run_orion_on_vsgr.py \
        --vsgr-root data/vsgr_aspire \
        --output data/orion_predictions/vsgr/ \
        --split train
    
    # Process specific videos for testing
    python scripts/run_orion_on_vsgr.py \
        --vsgr-root data/vsgr_aspire \
        --output data/orion_predictions/vsgr_test/ \
        --video-ids video_001 video_002 \
        --max-videos 5

Author: Orion Research Team
Date: October 2025
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orion.config import OrionConfig
from orion.tracking_engine import ObservationCollector, EntityTracker
from orion.causal_inference import CausalInferenceEngine, AgentCandidate, StateChange

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OrionVSGRRunner:
    """
    Run Orion perception pipeline on VSGR videos.
    
    Generates predictions for CIS optimization and evaluation.
    """
    
    def __init__(
        self,
        config: OrionConfig,
        output_dir: Path
    ):
        """
        Args:
            config: Orion configuration
            output_dir: Directory to save predictions
        """
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_video(
        self,
        video_path: Path,
        video_id: str
    ) -> Dict:
        """
        Process a single video through Orion pipeline.
        
        Args:
            video_path: Path to video file
            video_id: Unique identifier for video
            
        Returns:
            Dictionary with agent candidates and state changes
        """
        logger.info(f"\nProcessing video: {video_id}")
        logger.info(f"  Path: {video_path}")
        
        # Initialize perception components
        observer = ObservationCollector(self.config)
        tracker = EntityTracker(self.config)
        
        # Process video frame by frame
        logger.info("  Running perception pipeline...")
        
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                logger.error(f"  Failed to open video: {video_path}")
                return self._empty_result(video_id)
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"  Video: {total_frames} frames @ {fps:.2f} FPS")
            
            frame_idx = 0
            observations = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Collect observations (YOLO detection + CLIP embedding)
                obs = observer.process_frame(frame, frame_idx, frame_idx / fps)
                observations.extend(obs)
                
                frame_idx += 1
                
                if frame_idx % 100 == 0:
                    logger.info(f"  Processed {frame_idx}/{total_frames} frames")
            
            cap.release()
            
            logger.info(f"  Collected {len(observations)} observations")
            
            # Cluster into entities
            logger.info("  Clustering into entities...")
            entities = tracker.cluster_observations(observations)
            logger.info(f"  Found {len(entities)} unique entities")
            
            # Detect state changes
            logger.info("  Detecting state changes...")
            state_changes = self._detect_state_changes(entities, fps)
            logger.info(f"  Detected {len(state_changes)} state changes")
            
            # Generate agent candidates
            logger.info("  Generating agent candidates...")
            agent_candidates = self._generate_agent_candidates(
                entities,
                state_changes,
                fps
            )
            logger.info(f"  Generated {len(agent_candidates)} agent candidates")
            
            # Save results
            result = {
                "video_id": video_id,
                "metadata": {
                    "total_frames": total_frames,
                    "fps": fps,
                    "num_observations": len(observations),
                    "num_entities": len(entities),
                    "num_state_changes": len(state_changes),
                    "num_agent_candidates": len(agent_candidates)
                },
                "agent_candidates": [self._serialize_agent(a) for a in agent_candidates],
                "state_changes": [self._serialize_state_change(s) for s in state_changes]
            }
            
            return result
            
        except Exception as e:
            logger.error(f"  Error processing video: {e}")
            import traceback
            traceback.print_exc()
            return self._empty_result(video_id)
    
    def _detect_state_changes(
        self,
        entities: List,
        fps: float
    ) -> List[StateChange]:
        """
        Detect state changes in entity tracks.
        
        Simple heuristic: significant motion changes or appearance changes.
        """
        state_changes = []
        
        for entity in entities:
            # Get observations for this entity
            obs_list = sorted(entity.observations, key=lambda x: x.timestamp)
            
            if len(obs_list) < 2:
                continue
            
            # Detect motion changes (simplified)
            for i in range(1, len(obs_list)):
                prev_obs = obs_list[i-1]
                curr_obs = obs_list[i]
                
                # Calculate motion
                dt = curr_obs.timestamp - prev_obs.timestamp
                if dt == 0:
                    continue
                
                dx = curr_obs.centroid[0] - prev_obs.centroid[0]
                dy = curr_obs.centroid[1] - prev_obs.centroid[1]
                speed = np.sqrt(dx*dx + dy*dy) / dt
                
                # Significant motion change = potential state change
                if speed > 50.0:  # pixels/second
                    state_change = StateChange(
                        entity_id=entity.entity_id,
                        timestamp=curr_obs.timestamp,
                        frame_number=curr_obs.frame_number,
                        centroid=curr_obs.centroid,
                        change_type="motion_change",
                        description=f"Entity {entity.entity_id} motion change"
                    )
                    state_changes.append(state_change)
        
        return state_changes
    
    def _generate_agent_candidates(
        self,
        entities: List,
        state_changes: List[StateChange],
        fps: float
    ) -> List[AgentCandidate]:
        """
        Generate agent candidates for each state change.
        
        Agent candidates are entities that could have caused the state change.
        """
        candidates = []
        
        for state_change in state_changes:
            # Find entities present around the time of state change
            temporal_window = 2.0  # seconds
            
            for entity in entities:
                # Check if entity was present around state change time
                entity_times = [obs.timestamp for obs in entity.observations]
                
                if not entity_times:
                    continue
                
                min_time = min(entity_times)
                max_time = max(entity_times)
                
                # Entity overlaps with state change window
                if (min_time <= state_change.timestamp <= max_time or
                    abs(min_time - state_change.timestamp) <= temporal_window or
                    abs(max_time - state_change.timestamp) <= temporal_window):
                    
                    # Find observation closest to state change
                    closest_obs = min(
                        entity.observations,
                        key=lambda obs: abs(obs.timestamp - state_change.timestamp)
                    )
                    
                    candidate = AgentCandidate(
                        entity_id=entity.entity_id,
                        timestamp=closest_obs.timestamp,
                        centroid=closest_obs.centroid,
                        motion_data=None,  # TODO: Add motion data
                        embedding=closest_obs.embedding,
                        class_label=closest_obs.class_label,
                        confidence=closest_obs.confidence
                    )
                    
                    candidates.append(candidate)
        
        return candidates
    
    def _serialize_agent(self, agent: AgentCandidate) -> Dict:
        """Serialize AgentCandidate to JSON-compatible dict."""
        return {
            "entity_id": agent.entity_id,
            "timestamp": float(agent.timestamp),
            "centroid": [float(x) for x in agent.centroid],
            "class_label": agent.class_label,
            "confidence": float(agent.confidence),
            "embedding": [float(x) for x in agent.embedding] if agent.embedding is not None else None
        }
    
    def _serialize_state_change(self, state_change: StateChange) -> Dict:
        """Serialize StateChange to JSON-compatible dict."""
        return {
            "entity_id": state_change.entity_id,
            "timestamp": float(state_change.timestamp),
            "frame_number": int(state_change.frame_number),
            "centroid": [float(x) for x in state_change.centroid],
            "change_type": state_change.change_type,
            "description": state_change.description
        }
    
    def _empty_result(self, video_id: str) -> Dict:
        """Return empty result for failed videos."""
        return {
            "video_id": video_id,
            "metadata": {
                "error": "Failed to process video"
            },
            "agent_candidates": [],
            "state_changes": []
        }


def main():
    parser = argparse.ArgumentParser(
        description='Run Orion perception on VSGR videos'
    )
    parser.add_argument(
        '--vsgr-root',
        type=str,
        required=True,
        help='Path to VSGR dataset root directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/orion_predictions/vsgr/',
        help='Output directory for predictions'
    )
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'val', 'test', 'all'],
        default='train',
        help='Dataset split to process'
    )
    parser.add_argument(
        '--video-ids',
        nargs='+',
        help='Specific video IDs to process'
    )
    parser.add_argument(
        '--max-videos',
        type=int,
        help='Maximum number of videos to process'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to Orion config file (optional)'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("ORION PERCEPTION ON VSGR DATASET")
    logger.info("="*80)
    logger.info(f"VSGR root: {args.vsgr_root}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Split: {args.split}")
    
    # Load configuration
    if args.config:
        config = OrionConfig.from_yaml(args.config)
    else:
        config = OrionConfig()
    
    logger.info(f"Using config: {config}")
    
    # Find videos to process
    vsgr_root = Path(args.vsgr_root)
    videos_dir = vsgr_root / "videos"
    
    if not videos_dir.exists():
        logger.error(f"Videos directory not found: {videos_dir}")
        logger.error("Make sure VSGR dataset is downloaded")
        return 1
    
    # Collect video files
    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mov']:
        video_files.extend(videos_dir.glob(f"**/{ext}"))
    
    if not video_files:
        logger.error(f"No videos found in: {videos_dir}")
        return 1
    
    logger.info(f"\nFound {len(video_files)} total videos")
    
    # Filter by video IDs if specified
    if args.video_ids:
        video_files = [
            v for v in video_files
            if any(vid in str(v) for vid in args.video_ids)
        ]
        logger.info(f"Filtered to {len(video_files)} videos matching IDs")
    
    # Limit if max_videos specified
    if args.max_videos:
        video_files = video_files[:args.max_videos]
        logger.info(f"Limited to {len(video_files)} videos (max: {args.max_videos})")
    
    # Process videos
    runner = OrionVSGRRunner(config, output_dir)
    
    results = []
    for i, video_path in enumerate(video_files, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"[{i}/{len(video_files)}] Processing: {video_path.name}")
        logger.info(f"{'='*80}")
        
        video_id = video_path.stem
        result = runner.process_video(video_path, video_id)
        results.append(result)
        
        # Save individual result
        result_path = output_dir / f"{video_id}.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"✓ Saved predictions to: {result_path}")
    
    # Save summary
    summary_path = output_dir / "summary.json"
    summary = {
        "total_videos": len(results),
        "successful": len([r for r in results if "error" not in r["metadata"]]),
        "failed": len([r for r in results if "error" in r["metadata"]]),
        "total_agent_candidates": sum(r["metadata"].get("num_agent_candidates", 0) for r in results),
        "total_state_changes": sum(r["metadata"].get("num_state_changes", 0) for r in results),
        "video_ids": [r["video_id"] for r in results]
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("\n" + "="*80)
    logger.info("PROCESSING COMPLETE")
    logger.info("="*80)
    logger.info(f"Total videos: {summary['total_videos']}")
    logger.info(f"Successful: {summary['successful']}")
    logger.info(f"Failed: {summary['failed']}")
    logger.info(f"Total agent candidates: {summary['total_agent_candidates']}")
    logger.info(f"Total state changes: {summary['total_state_changes']}")
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"Summary saved to: {summary_path}")
    
    logger.info("\nNext step:")
    logger.info(f"  python scripts/test_cis_optimization.py \\")
    logger.info(f"    --ground-truth data/ground_truth/vsgr_aspire_train.json \\")
    logger.info(f"    --predictions {output_dir} \\")
    logger.info(f"    --trials 100")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
