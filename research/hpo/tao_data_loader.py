"""
TAO-Amodal Data Loader for CIS Training
========================================

Converts TAO-Amodal bounding box annotations into AgentCandidate objects
for CIS hyperparameter optimization without requiring actual video files.

This enables training on VSGR ground truth annotations using the spatial
and temporal information from TAO-Amodal tracking data.

Author: Orion Research Team
Date: October 2025
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from ..causal_inference import AgentCandidate, StateChange
    from ..motion_tracker import MotionData
except ImportError:
    from causal_inference import AgentCandidate, StateChange  # type: ignore
    from motion_tracker import MotionData  # type: ignore

logger = logging.getLogger("HPO.TAOLoader")


@dataclass
class TAOTrack:
    """Represents a tracked object across frames"""
    track_id: int
    category_id: int
    category_name: str
    video_id: int
    annotations: List[Dict]  # Frame-level bboxes


@dataclass
class VSGRGroundTruth:
    """VSGR ground truth causal annotation"""
    video_id: int
    frame_range: List[int]
    agent_id: str
    patient_id: str
    agent_class: str
    patient_class: str
    interaction_type: str
    is_causal: bool
    confidence: float
    annotation_source: str
    reasoning: str


class TAODataLoader:
    """
    Loads TAO-Amodal annotations and converts them to Orion data structures.
    
    This bypasses the need for actual video files by synthesizing
    AgentCandidate objects directly from bounding box tracks.
    """
    
    def __init__(
        self,
        tao_json_path: Path,
        vsgr_json_path: Path,
        fps: float = 30.0
    ):
        """
        Args:
            tao_json_path: Path to TAO-Amodal annotations (aspire_train.json)
            vsgr_json_path: Path to VSGR ground truth (vsgr_aspire_train.json)
            fps: Assumed frames per second (default: 30)
        """
        self.tao_json_path = tao_json_path
        self.vsgr_json_path = vsgr_json_path
        self.fps = fps
        
        # Load data
        logger.info(f"Loading TAO-Amodal from {tao_json_path}")
        with open(tao_json_path) as f:
            self.tao_data = json.load(f)
        
        logger.info(f"Loading VSGR ground truth from {vsgr_json_path}")
        with open(vsgr_json_path) as f:
            vsgr_raw = json.load(f)
            self.vsgr_data = vsgr_raw.get("causal_pairs", [])
        
        # Build lookup tables
        self._build_lookups()
        
        logger.info(f"Loaded {len(self.tracks)} tracks from {len(self.videos)} videos")
        logger.info(f"Loaded {len(self.vsgr_data)} VSGR ground truth pairs")
    
    def _build_lookups(self):
        """Build lookup tables for fast access"""
        
        # Category lookup
        self.categories = {cat["id"]: cat["name"] for cat in self.tao_data["categories"]}
        
        # Video lookup
        self.videos = {vid["id"]: vid for vid in self.tao_data["videos"]}
        
        # Image lookup
        self.images = {img["id"]: img for img in self.tao_data["images"]}
        
        # Track lookup
        self.tracks = {track["id"]: track for track in self.tao_data["tracks"]}
        
        # Group annotations by track_id
        self.track_annotations: Dict[int, List[Dict]] = defaultdict(list)
        for ann in self.tao_data["annotations"]:
            self.track_annotations[ann["track_id"]].append(ann)
        
        # Sort annotations by frame for each track
        for track_id in self.track_annotations:
            self.track_annotations[track_id].sort(
                key=lambda a: self.images[a["image_id"]]["frame_index"]
            )
    
    def get_video_tracks(self, video_id: int) -> List[TAOTrack]:
        """Get all tracks for a specific video"""
        video_tracks = []
        
        for track_id, track_info in self.tracks.items():
            if track_info["video_id"] != video_id:
                continue
            
            annotations = self.track_annotations.get(track_id, [])
            if not annotations:
                continue
            
            category_name = self.categories.get(track_info["category_id"], "unknown")
            
            video_tracks.append(TAOTrack(
                track_id=track_id,
                category_id=track_info["category_id"],
                category_name=category_name,
                video_id=video_id,
                annotations=annotations
            ))
        
        return video_tracks
    
    def track_to_agent_candidates(
        self,
        track: TAOTrack,
        frame_range: Optional[Tuple[int, int]] = None
    ) -> List[AgentCandidate]:
        """
        Convert a TAO track to AgentCandidate objects.
        
        Args:
            track: TAO track object
            frame_range: Optional (start_frame, end_frame) to filter
            
        Returns:
            List of AgentCandidate objects (one per frame)
        """
        agents = []
        
        for i, ann in enumerate(track.annotations):
            image_id = ann["image_id"]
            image_info = self.images[image_id]
            frame_idx = image_info["frame_index"]
            
            # Filter by frame range if specified
            if frame_range:
                if frame_idx < frame_range[0] or frame_idx > frame_range[1]:
                    continue
            
            # Extract bbox [x, y, width, height]
            bbox = ann["bbox"]
            x, y, w, h = bbox
            centroid = (x + w / 2, y + h / 2)
            
            # Calculate motion from previous frame
            motion_data = None
            if i > 0:
                prev_ann = track.annotations[i - 1]
                prev_bbox = prev_ann["bbox"]
                prev_x, prev_y, prev_w, prev_h = prev_bbox
                prev_centroid = (prev_x + prev_w / 2, prev_y + prev_h / 2)
                
                # Calculate velocity (pixels per frame)
                dx = centroid[0] - prev_centroid[0]
                dy = centroid[1] - prev_centroid[1]
                
                prev_frame_idx = self.images[prev_ann["image_id"]]["frame_index"]
                frame_diff = frame_idx - prev_frame_idx
                if frame_diff > 0:
                    vx = dx / frame_diff
                    vy = dy / frame_diff
                    speed = np.sqrt(vx**2 + vy**2)
                    
                    motion_data = MotionData(
                        centroid=centroid,
                        velocity=(vx, vy),
                        speed=speed,
                        direction=np.arctan2(vy, vx),
                        timestamp=frame_idx / self.fps
                    )
            
            # Create agent candidate
            agent = AgentCandidate(
                entity_id=f"track_{track.track_id}",
                temp_id=f"track_{track.track_id}_f{frame_idx}",
                timestamp=frame_idx / self.fps,
                centroid=centroid,
                bounding_box=[int(x), int(y), int(w), int(h)],
                motion_data=motion_data,
                visual_embedding=[0.0] * 512,  # Placeholder (not available from TAO)
                object_class=track.category_name,
                description=track.category_name
            )
            
            agents.append(agent)
        
        return agents
    
    def get_vsgr_for_video(self, video_id: int) -> List[VSGRGroundTruth]:
        """Get all VSGR ground truth annotations for a video"""
        gt_list = []
        
        for item in self.vsgr_data:
            if item["video_id"] != video_id:
                continue
            
            gt_list.append(VSGRGroundTruth(
                video_id=item["video_id"],
                frame_range=item["frame_range"],
                agent_id=item["agent_id"],
                patient_id=item["patient_id"],
                agent_class=item["agent_class"],
                patient_class=item["patient_class"],
                interaction_type=item["interaction_type"],
                is_causal=item["is_causal"],
                confidence=item["confidence"],
                annotation_source=item["annotation_source"],
                reasoning=item.get("reasoning", "")
            ))
        
        return gt_list
    
    def create_state_change_from_gt(
        self,
        gt: VSGRGroundTruth,
        patient_track: TAOTrack
    ) -> Optional[StateChange]:
        """
        Create a StateChange object from VSGR ground truth.
        
        Args:
            gt: VSGR ground truth annotation
            patient_track: The TAO track for the patient entity
            
        Returns:
            StateChange object or None if frame not found
        """
        # The frame_range in VSGR refers to the annotation index, not absolute frame number
        # We need to find the closest annotation in the patient track
        target_frame = gt.frame_range[1]  # Use end of frame range
        
        # Find the annotation closest to this index in the track
        if target_frame >= len(patient_track.annotations):
            # Use the last annotation
            patient_ann = patient_track.annotations[-1]
        else:
            patient_ann = patient_track.annotations[target_frame] if target_frame < len(patient_track.annotations) else patient_track.annotations[-1]
        
        if not patient_ann:
            logger.warning(
                f"Could not find patient annotation for "
                f"track {patient_track.track_id}"
            )
            return None
        
        # Get the actual frame index from the image
        image_id = patient_ann["image_id"]
        frame_idx = self.images[image_id]["frame_index"]
        
        # Extract bbox
        bbox = patient_ann["bbox"]
        x, y, w, h = bbox
        centroid = (x + w / 2, y + h / 2)
        
        # Create state change
        # Use interaction_type as state description
        old_desc = gt.patient_class
        new_desc = f"{gt.patient_class} ({gt.interaction_type})"
        
        return StateChange(
            entity_id=gt.patient_id,
            timestamp=frame_idx / self.fps,
            frame_number=frame_idx,
            old_description=old_desc,
            new_description=new_desc,
            centroid=centroid,
            bounding_box=[int(x), int(y), int(w), int(h)]
        )
    
    def prepare_training_data(
        self,
        video_ids: Optional[List[int]] = None,
        max_videos: Optional[int] = None
    ) -> Tuple[List[AgentCandidate], List[StateChange], List[Dict]]:
        """
        Prepare complete training dataset for CIS optimization.
        
        Args:
            video_ids: Specific video IDs to use (None = all with VSGR annotations)
            max_videos: Maximum number of videos to process
            
        Returns:
            Tuple of (agent_candidates, state_changes, ground_truth_pairs)
        """
        logger.info("Preparing training data from TAO-Amodal + VSGR...")
        
        # Find all videos with VSGR annotations
        if video_ids is None:
            video_ids = list(set(item["video_id"] for item in self.vsgr_data))
        
        if max_videos:
            video_ids = video_ids[:max_videos]
        
        logger.info(f"Processing {len(video_ids)} videos...")
        
        all_agents = []
        all_state_changes = []
        all_ground_truth = []
        
        for video_id in video_ids:
            logger.info(f"Processing video {video_id}...")
            
            # Get tracks for this video
            tracks = self.get_video_tracks(video_id)
            track_map = {f"track_{t.track_id}": t for t in tracks}
            
            # Get VSGR ground truth
            gt_list = self.get_vsgr_for_video(video_id)
            
            if not gt_list:
                logger.warning(f"No VSGR annotations for video {video_id}, skipping")
                continue
            
            # Convert tracks to agent candidates
            for track in tracks:
                agents = self.track_to_agent_candidates(track)
                all_agents.extend(agents)
            
            # Create state changes and ground truth pairs
            for gt in gt_list:
                # Find patient track
                patient_track = track_map.get(gt.patient_id)
                if not patient_track:
                    logger.warning(
                        f"Patient track {gt.patient_id} not found for video {video_id}"
                    )
                    continue
                
                # Create state change
                state_change = self.create_state_change_from_gt(gt, patient_track)
                if state_change:
                    all_state_changes.append(state_change)
                    
                    # Add ground truth pair
                    # Use the actual frame number from the state change
                    all_ground_truth.append({
                        "agent_id": gt.agent_id,
                        "patient_id": gt.patient_id,
                        "state_change_frame": state_change.frame_number,
                        "is_causal": gt.is_causal,
                        "confidence": gt.confidence,
                        "annotation_source": gt.annotation_source
                    })
        
        logger.info(f"Prepared {len(all_agents)} agent candidates")
        logger.info(f"Prepared {len(all_state_changes)} state changes")
        logger.info(f"Prepared {len(all_ground_truth)} ground truth pairs")
        
        return all_agents, all_state_changes, all_ground_truth


def load_tao_training_data(
    tao_json_path: str,
    vsgr_json_path: str,
    max_videos: Optional[int] = None
) -> Tuple[List[AgentCandidate], List[StateChange], List[Dict]]:
    """
    Convenience function to load TAO-Amodal training data.
    
    Args:
        tao_json_path: Path to TAO annotations
        vsgr_json_path: Path to VSGR ground truth
        max_videos: Maximum videos to process (None = all)
        
    Returns:
        Tuple of (agent_candidates, state_changes, ground_truth_pairs)
    """
    loader = TAODataLoader(
        Path(tao_json_path),
        Path(vsgr_json_path)
    )
    
    return loader.prepare_training_data(max_videos=max_videos)
