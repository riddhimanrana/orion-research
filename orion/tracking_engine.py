"""
Tracking-Based Perception Engine
=================================

This module implements a smart object tracking system that:
1. Detects and embeds ALL observations across the entire video (Phase 1)
2. Clusters observations into unique entities using embeddings (Phase 2)
3. Describes each unique entity ONCE from its best frame (Phase 3)
4. Builds a rich temporal knowledge graph with relationships (Phase 4)

Key Innovation: "Track First, Describe Once, Link Always"
- Only generate description when we FIRST see a unique object
- Reuse description for all subsequent appearances of same object
- Only re-describe if significant visual state change detected

Author: Orion Research Team
Date: October 2025
"""

import logging
import os
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

try:
    from .model_manager import ModelManager
    from .config import OrionConfig
except ImportError:
    from model_manager import ModelManager  # type: ignore
    from config import OrionConfig  # type: ignore

try:
    from .models import AssetManager
except ImportError:
    from models import ModelManager as AssetManager  # type: ignore

try:
    from .runtime import get_active_backend, select_backend
except ImportError:
    from runtime import get_active_backend, select_backend  # type: ignore

try:
    from .motion_tracker import MotionTracker, MotionData
except ImportError:
    from motion_tracker import MotionTracker, MotionData  # type: ignore

# Optional imports
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    hdbscan = None
    HDBSCAN_AVAILABLE = False
    print("Warning: hdbscan not available. Install with: pip install hdbscan")

warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logger = logging.getLogger('TrackingEngine')

# For backward compatibility - will be removed in future
# Use OrionConfig directly instead
Config = None  # Deprecated - use OrionConfig


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Observation:
    """Single detection of an object in a specific frame"""
    frame_number: int
    timestamp: float
    bbox: List[int]  # [x1, y1, x2, y2]
    class_name: str
    confidence: float
    embedding: np.ndarray
    crop: np.ndarray  # Cropped image for description
    frame_width: int
    frame_height: int
    motion: Optional[MotionData] = None
    
    def get_bbox_area(self) -> float:
        """Calculate bounding box area"""
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])
    
    def get_bbox_center(self) -> Tuple[float, float]:
        """Calculate bounding box center"""
        x = (self.bbox[0] + self.bbox[2]) / 2.0
        y = (self.bbox[1] + self.bbox[3]) / 2.0
        return (x, y)
    
    def get_centrality_score(self) -> float:
        """How centered is this detection? (0-1)"""
        center_x, _ = self.get_bbox_center()
        frame_center = self.frame_width / 2.0
        distance_from_center = abs(center_x - frame_center)
        return 1.0 - (distance_from_center / frame_center)


@dataclass
class Entity:
    """A unique tracked object across multiple frames"""
    id: str
    class_name: str
    observations: List[Observation] = field(default_factory=list)
    description: Optional[str] = None
    described_from_frame: Optional[int] = None
    state_changes: List[Dict[str, Any]] = field(default_factory=list)
    motion_history: List[MotionData] = field(default_factory=list)
    
    @property
    def first_seen(self) -> float:
        return min(obs.timestamp for obs in self.observations) if self.observations else 0.0
    
    @property
    def last_seen(self) -> float:
        return max(obs.timestamp for obs in self.observations) if self.observations else 0.0
    
    @property
    def duration(self) -> float:
        return self.last_seen - self.first_seen
    
    @property
    def appearance_count(self) -> int:
        return len(self.observations)
    
    @property
    def average_embedding(self) -> np.ndarray:
        """Calculate average embedding across all observations"""
        embeddings = np.array([obs.embedding for obs in self.observations])
        avg = np.mean(embeddings, axis=0)
        return avg / np.linalg.norm(avg)  # L2 normalize
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        centroid_history: List[Dict[str, Any]] = []
        observations_payload: List[Dict[str, Any]] = []
        frame_numbers: List[int] = []

        for obs in self.observations:
            centroid = obs.get_bbox_center()
            centroid_history.append({
                'frame_number': obs.frame_number,
                'timestamp': obs.timestamp,
                'centroid': [float(centroid[0]), float(centroid[1])],
                'frame_width': obs.frame_width,
                'frame_height': obs.frame_height,
            })

            obs_payload: Dict[str, Any] = {
                'frame_number': obs.frame_number,
                'timestamp': obs.timestamp,
                'bbox': [int(v) for v in obs.bbox],
                'centroid': [float(centroid[0]), float(centroid[1])],
                'confidence': obs.confidence,
                'class_name': obs.class_name,
                'frame_width': obs.frame_width,
                'frame_height': obs.frame_height,
            }

            motion_dict = motion_to_dict(obs.motion)
            if motion_dict:
                obs_payload['motion'] = motion_dict

            observations_payload.append(obs_payload)
            frame_numbers.append(obs.frame_number)

        if centroid_history:
            avg_x = float(np.mean([c['centroid'][0] for c in centroid_history]))
            avg_y = float(np.mean([c['centroid'][1] for c in centroid_history]))
            average_centroid = [avg_x, avg_y]
        else:
            average_centroid = [0.0, 0.0]

        motion_history_payload = [
            md for md in (motion_to_dict(motion) for motion in self.motion_history)
            if md is not None
        ]

        return {
            'entity_id': self.id,
            'class_name': self.class_name,
            'description': self.description,
            'first_seen': self.first_seen,
            'last_seen': self.last_seen,
            'duration': self.duration,
            'appearance_count': self.appearance_count,
            'described_from_frame': self.described_from_frame,
            'state_changes': self.state_changes,
            'average_embedding': self.average_embedding.tolist(),
            'frame_numbers': frame_numbers,
            'observations': observations_payload,
            'centroid_history': centroid_history,
            'average_centroid': average_centroid,
            'motion_history': motion_history_payload,
        }


def motion_to_dict(motion: Optional[MotionData]) -> Optional[Dict[str, Any]]:
    """Serialize MotionData into plain Python primitives."""
    if motion is None:
        return None
    vx, vy = motion.velocity
    cx, cy = motion.centroid
    return {
        'timestamp': motion.timestamp,
        'centroid': [float(cx), float(cy)],
        'velocity': [float(vx), float(vy)],
        'speed': float(motion.speed),
        'direction': float(motion.direction),
    }


# ============================================================================
# PHASE 1: OBSERVATION COLLECTION
# ============================================================================

class ObservationCollector:
    """Collects all detections with embeddings from entire video"""
    
    def __init__(self, config: Optional[OrionConfig] = None):
        self.config = config or OrionConfig()
        self.asset_manager = AssetManager()
        self.model_manager = ModelManager.get_instance()
        self.observations: List[Observation] = []
        
    def load_models(self):
        """Load models via ModelManager (lazy loading)"""
        logger.info("Loading models...")
        
        # Models are lazy-loaded automatically when accessed
        # Just trigger loading and log
        _ = self.model_manager.yolo
        logger.info("✓ YOLO11x loaded")
        
        _ = self.model_manager.clip
        logger.info(f"✓ CLIP loaded ({self.config.embedding.embedding_dim}-dim embeddings)")
        if self.config.embedding.use_text_conditioning:
            logger.info("  Mode: Multimodal (vision + text conditioning)")
        else:
            logger.info("  Mode: Vision only")
    
    def process_video(self, video_path: str) -> List[Observation]:
        """Process entire video and collect observations"""
        logger.info("="*80)
        logger.info("PHASE 1: OBSERVATION COLLECTION")
        logger.info("="*80)
        
        self.load_models()
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video: {video_path}")
        logger.info(f"  FPS: {fps:.2f}, Total frames: {total_frames}")
        logger.info(f"  Resolution: {frame_width}x{frame_height}")
        
        # Calculate frame sampling
        frame_interval = int(fps / self.config.video.target_fps)
        frame_interval = max(1, frame_interval)
        
        logger.info(f"Sampling every {frame_interval} frames (target {self.config.video.target_fps} FPS)")
        
        self.observations = []
        frame_count = 0
        
        pbar = tqdm(total=total_frames, desc="Collecting observations", 
                   disable=not self.config.logging.show_progress)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames
                if frame_count % frame_interval == 0:
                    timestamp = frame_count / fps
                    
                    # Detect objects
                    detections = self._detect_objects(frame)
                    
                    # Process each detection
                    for detection in detections:
                        # Crop object
                        crop, padded_bbox = self._crop_object(frame, detection['bbox'])
                        
                        # Generate embedding (pass class_name for multimodal conditioning)
                        embedding = self._generate_embedding(crop, detection['class_name'])
                        
                        # Create observation
                        obs = Observation(
                            frame_number=frame_count,
                            timestamp=timestamp,
                            bbox=detection['bbox'],
                            class_name=detection['class_name'],
                            confidence=detection['confidence'],
                            embedding=embedding,
                            crop=crop,
                            frame_width=frame_width,
                            frame_height=frame_height
                        )
                        
                        self.observations.append(obs)
                
                frame_count += 1
                pbar.update(1)
        
        finally:
            cap.release()
            pbar.close()
        
        logger.info(f"\n✓ Collected {len(self.observations)} observations from {frame_count} frames")
        logger.info(f"  Average: {len(self.observations) / max(frame_count // frame_interval, 1):.1f} detections per sampled frame")
        logger.info("="*80 + "\n")
        
        return self.observations
    
    def _detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects in frame using YOLO"""
        results = self.model_manager.yolo(
            frame,
            conf=self.config.detection.confidence_threshold,
            iou=self.config.detection.iou_threshold,
            verbose=False
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                x1, y1, x2, y2 = bbox
                
                # Filter by size
                width = x2 - x1
                height = y2 - y1
                if width < self.config.detection.min_object_size or height < self.config.detection.min_object_size:
                    continue
                
                detections.append({
                    'bbox': bbox,
                    'confidence': float(boxes.conf[i]),
                    'class_id': int(boxes.cls[i]),
                    'class_name': result.names[int(boxes.cls[i])]
                })
        
        return detections
    
    def _crop_object(self, frame: np.ndarray, bbox: List[int]) -> Tuple[np.ndarray, List[int]]:
        """Crop object from frame with padding"""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        
        # Add padding
        padding = self.config.detection.bbox_padding_percent
        width = x2 - x1
        height = y2 - y1
        
        x1_padded = max(0, int(x1 - width * padding))
        y1_padded = max(0, int(y1 - height * padding))
        x2_padded = min(w, int(x2 + width * padding))
        y2_padded = min(h, int(y2 + height * padding))
        
        crop = frame[y1_padded:y2_padded, x1_padded:x2_padded]
        return crop, [x1_padded, y1_padded, x2_padded, y2_padded]
    
    def _generate_embedding(self, crop: np.ndarray, class_name: Optional[str] = None) -> np.ndarray:
        """Generate embedding for crop using CLIP"""
        # Convert BGR to RGB
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_crop)
        
        # Use CLIP for embeddings
        if self.config.embedding.use_text_conditioning and class_name:
            # Multimodal: condition on YOLO class
            # This helps catch misclassifications - if image doesn't match class,
            # embedding will be different from other instances of that class
            embedding = self.model_manager.clip.encode_multimodal(
                pil_image,
                f"a {class_name}",
                normalize=True
            )
        else:
            # Vision only
            embedding = self.model_manager.clip.encode_image(
                pil_image,
                normalize=True
            )
        
        return embedding


# ============================================================================
# PHASE 2: ENTITY CLUSTERING
# ============================================================================

class EntityTracker:
    """Clusters observations into unique tracked entities"""
    
    def __init__(self, config: Optional[OrionConfig] = None):
        self.config = config or OrionConfig()
        self.entities: List[Entity] = []
    
    def cluster_observations(self, observations: List[Observation]) -> List[Entity]:
        """Cluster observations using HDBSCAN on embeddings"""
        logger.info("="*80)
        logger.info("PHASE 2: ENTITY CLUSTERING")
        logger.info("="*80)
        
        if not HDBSCAN_AVAILABLE:
            logger.error("HDBSCAN not available - cannot cluster entities")
            return self._fallback_clustering(observations)
        
        logger.info(f"Clustering {len(observations)} observations...")
        
        # Extract embeddings
        embeddings = np.array([obs.embedding for obs in observations])
        logger.info(f"Embedding shape: {embeddings.shape}")
        logger.info(f"Embedding dtype: {embeddings.dtype}")
        
        # Check if embeddings are normalized
        norms = np.linalg.norm(embeddings, axis=1)
        logger.info(f"Embedding norms - min: {norms.min():.4f}, max: {norms.max():.4f}, mean: {norms.mean():.4f}")
        
        # Sample some pairwise distances to diagnose
        sample_euc_dists = []
        if len(embeddings) > 1:
            sample_cos_dists = []
            for i in range(min(10, len(embeddings))):
                for j in range(i+1, min(10, len(embeddings))):
                    # Cosine distance for normalized vectors
                    cos_sim = np.dot(embeddings[i], embeddings[j])
                    cos_dist = 1 - cos_sim
                    sample_cos_dists.append(cos_dist)
                    
                    # Euclidean distance
                    euc_dist = np.linalg.norm(embeddings[i] - embeddings[j])
                    sample_euc_dists.append(euc_dist)
            
            if sample_cos_dists:
                logger.info(f"Sample cosine distances - min: {min(sample_cos_dists):.4f}, "
                           f"max: {max(sample_cos_dists):.4f}, mean: {np.mean(sample_cos_dists):.4f}")
                logger.info(f"Sample euclidean distances - min: {min(sample_euc_dists):.4f}, "
                           f"max: {max(sample_euc_dists):.4f}, mean: {np.mean(sample_euc_dists):.4f}")
                logger.info(f"→ Current CLUSTER_SELECTION_EPSILON = {self.config.clustering.cluster_selection_epsilon}")
                logger.info(f"→ Mean euclidean distance is {np.mean(sample_euc_dists):.2f}, "
                           f"which is {np.mean(sample_euc_dists)/self.config.clustering.cluster_selection_epsilon:.1f}x larger than epsilon")
        
        # For normalized embeddings, euclidean distance is monotonically related to cosine distance
        # Since ||a|| = ||b|| = 1, we have: ||a-b||^2 = 2(1 - a·b) = 2 * cosine_distance
        # So we can use euclidean metric on normalized embeddings
        logger.info("Using euclidean metric on normalized embeddings")
        logger.info(f"Clustering params: min_cluster_size={self.config.clustering.min_cluster_size}, "
                   f"min_samples={self.config.clustering.min_samples}, epsilon={self.config.clustering.cluster_selection_epsilon}")
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.config.clustering.min_cluster_size,
            min_samples=self.config.clustering.min_samples,
            metric=self.config.clustering.metric,
            cluster_selection_method='eom',  # Excess of Mass (same as semantic_uplift.py)
            cluster_selection_epsilon=self.config.clustering.cluster_selection_epsilon
        )
        
        labels = clusterer.fit_predict(embeddings)
        
        # Log statistics
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        logger.info(f"Clustering results:")
        logger.info(f"  Unique entities (clusters): {n_clusters}")
        logger.info(f"  Singleton objects (noise): {n_noise}")
        logger.info(f"  Total unique objects: {n_clusters + n_noise}")
        
        # CRITICAL CHECK: If everything is noise, clustering failed!
        if n_clusters == 0 and n_noise == len(observations):
            logger.error(f"Clustering FAILED: All {len(observations)} observations marked as noise!")
            logger.error(f"This means HDBSCAN couldn't find ANY clusters.")
            logger.error(f"Diagnosis:")
            logger.error(f"  - Current epsilon: {self.config.clustering.cluster_selection_epsilon}")
            if sample_euc_dists:
                logger.error(f"  - Mean euclidean distance: {np.mean(sample_euc_dists):.4f}")
                logger.error(f"  - Epsilon is {self.config.clustering.cluster_selection_epsilon / np.mean(sample_euc_dists):.2f}x the mean distance")
            logger.error(f"Solution: DECREASE epsilon (try 0.3 or lower) OR decrease min_cluster_size")
            logger.warning("Falling back to simple class-based grouping...")
            return self._class_based_fallback(observations)
        
        # Group observations by cluster
        entity_map = defaultdict(list)
        unique_counter = 0
        
        for obs, label in zip(observations, labels):
            if label == -1:
                # Noise - unique entity
                entity_id = f"unique_{unique_counter:05d}"
                unique_counter += 1
            else:
                entity_id = f"entity_{label:04d}"
            
            entity_map[entity_id].append(obs)
        
        # Create Entity objects
        self.entities = []
        for entity_id, obs_list in entity_map.items():
            entity = Entity(
                id=entity_id,
                class_name=obs_list[0].class_name,  # Use first observation's class
                observations=sorted(obs_list, key=lambda x: x.timestamp)
            )
            self.entities.append(entity)
        
        # Sort by first appearance
        self.entities.sort(key=lambda e: e.first_seen)
        
        logger.info(f"\n✓ Created {len(self.entities)} tracked entities")
        logger.info(f"  Average appearances per entity: {len(observations) / len(self.entities):.1f}")
        logger.info("="*80 + "\n")

        self._attach_motion_history(self.entities)

        return self.entities
    
    def _fallback_clustering(self, observations: List[Observation]) -> List[Entity]:
        """Fallback: treat each observation as unique entity"""
        logger.warning("Using fallback clustering (each observation = unique entity)")
        
        entities = []
        for i, obs in enumerate(observations):
            entity = Entity(
                id=f"obs_{i:05d}",
                class_name=obs.class_name,
                observations=[obs]
            )
            entities.append(entity)
        
        self._attach_motion_history(entities)
        return entities
    
    def _class_based_fallback(self, observations: List[Observation]) -> List[Entity]:
        """
        Smarter fallback: Group observations by object class
        Better than treating each as unique, but not as good as embedding clustering
        """
        logger.warning("Using class-based fallback clustering")
        logger.info("Grouping observations by object class (e.g., all 'keyboard' together)")
        
        # Group by class
        class_groups = defaultdict(list)
        for obs in observations:
            class_groups[obs.class_name].append(obs)
        
        # Create entities
        entities = []
        for class_name, obs_list in sorted(class_groups.items()):
            entity = Entity(
                id=f"class_{class_name}_{len(entities):03d}",
                class_name=class_name,
                observations=sorted(obs_list, key=lambda x: x.timestamp)
            )
            entities.append(entity)
        
        logger.info(f"Created {len(entities)} entities from {len(class_groups)} object classes")
        for entity in entities:
            logger.info(f"  {entity.id}: {len(entity.observations)} appearances")
        
        self._attach_motion_history(entities)
        return entities

    def _attach_motion_history(self, entities: List[Entity]) -> None:
        """Compute per-observation motion vectors for each entity."""
        for entity in entities:
            tracker = MotionTracker(smoothing_window=3)
            motion_records: List[MotionData] = []
            for obs in entity.observations:
                motion = tracker.update(entity.id, obs.timestamp, obs.bbox)
                if motion is None:
                    continue
                obs.motion = motion
                motion_records.append(motion)
            entity.motion_history = motion_records


# ============================================================================
# PHASE 3: SMART DESCRIPTION
# ============================================================================

class SmartDescriber:
    """Describes each entity ONCE from its best frame"""
    
    def __init__(self, config: Optional[OrionConfig] = None):
        self.config = config or OrionConfig()
        self.model_manager = ModelManager.get_instance()
    
    def describe_entities(self, entities: List[Entity]) -> List[Entity]:
        """Describe each entity once from best frame"""
        logger.info("="*80)
        logger.info("PHASE 3: SMART DESCRIPTION GENERATION")
        logger.info("="*80)
        
        # Load FastVLM via ModelManager
        _ = self.model_manager.fastvlm
        logger.info("✓ FastVLM loaded")
        
        logger.info(f"Generating descriptions for {len(entities)} unique entities...")
        logger.info(f"(Only describing each object ONCE from its best frame)")
        
        described_count = 0
        skipped_low_conf = 0
        skipped_errors = 0
        
        for entity in tqdm(entities, desc="Describing entities", disable=not self.config.logging.show_progress):
            try:
                # Select best observation
                best_obs = self._select_best_observation(entity)
                
                # Check confidence - skip low-confidence entities
                if best_obs.confidence < self.config.detection.low_confidence_threshold:
                    entity.description = (f"Low confidence detection (conf={best_obs.confidence:.2f}). "
                                        f"Likely false positive - skipped description.")
                    entity.described_from_frame = best_obs.frame_number
                    skipped_low_conf += 1
                    continue
                
                # Generate description
                description = self._generate_description(entity, best_obs)
                
                entity.description = description
                entity.described_from_frame = best_obs.frame_number
                described_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to describe {entity.id}: {e}")
                entity.description = f"A {entity.class_name} (description failed)"
                skipped_errors += 1
        
        logger.info(f"\n✓ Described {described_count} entities")
        if skipped_low_conf > 0:
            logger.info(f"  Skipped {skipped_low_conf} low-confidence entities (< {self.config.detection.low_confidence_threshold})")
        if skipped_errors > 0:
            logger.warning(f"  Skipped {skipped_errors} entities due to errors")
        logger.info("="*80 + "\n")
        
        return entities
    
    def _select_best_observation(self, entity: Entity) -> Observation:
        """Select best observation for description"""
        if len(entity.observations) == 1:
            return entity.observations[0]
        
        scored_obs = []
        
        for obs in entity.observations:
            # Calculate scores
            size_score = obs.get_bbox_area()
            centrality_score = obs.get_centrality_score()
            confidence_score = obs.confidence
            
            # Weighted combination
            total_score = (
                self.config.description.size_weight * size_score +
                self.config.description.centrality_weight * centrality_score +
                self.config.description.confidence_weight * confidence_score
            )
            
            scored_obs.append((total_score, obs))
        
        # Return observation with highest score
        return max(scored_obs, key=lambda x: x[0])[1]
    
    def _generate_description(self, entity: Entity, observation: Observation) -> str:
        """
        Generate unbiased description with YOLO verification
        
        This uses a two-stage approach:
        1. First, ask FastVLM what it sees (no bias from YOLO class)
        2. If FastVLM disagrees with YOLO, flag for review
        """
        # Convert crop to PIL Image
        rgb_crop = cv2.cvtColor(observation.crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_crop)
        
        # STAGE 1: Unbiased open-ended description
        # Don't mention the YOLO class - let FastVLM decide what it sees
        open_prompt = f"""What do you see in this image? Provide a detailed description.

Focus on:
- What type of object this is
- Its appearance, color, and shape
- Any distinguishing features or characteristics
- The context or setting if visible

Be objective and describe exactly what you observe."""

        # Generate initial description
        description = self.model_manager.fastvlm.generate_description(
            image=pil_image,
            prompt=open_prompt,
            max_tokens=self.config.description.max_tokens,
            temperature=self.config.description.temperature
        )
        
        # STAGE 2: Quick sanity check - does FastVLM agree with YOLO?
        # Check if the YOLO class appears in the description
        yolo_class_lower = entity.class_name.lower()
        description_lower = description.lower()
        
        # Common synonyms for classes
        synonyms = {
            'tv': ['monitor', 'screen', 'display', 'television'],
            'monitor': ['tv', 'screen', 'display', 'television'],
            'laptop': ['computer', 'notebook'],
            'cell phone': ['phone', 'smartphone', 'mobile'],
            'mouse': ['computer mouse', 'peripheral'],
            'keyboard': ['keys', 'typing'],
        }
        
        # Check if YOLO class or its synonyms appear in description
        class_mentioned = yolo_class_lower in description_lower
        if not class_mentioned and yolo_class_lower in synonyms:
            class_mentioned = any(syn in description_lower for syn in synonyms[yolo_class_lower])
        
        if not class_mentioned and observation.confidence < 0.7:
            # FastVLM doesn't see what YOLO claimed, and YOLO wasn't confident
            # Flag this as potentially misclassified
            logger.warning(f"Potential misclassification: YOLO said '{entity.class_name}' "
                          f"(conf={observation.confidence:.2f}), but FastVLM sees: "
                          f"{description[:100]}...")
        
        return description.strip()
    
    def detect_state_changes(self, entities: List[Entity]) -> List[Entity]:
        """
        Detect significant state changes using embedding similarity
        
        EFFICIENT VERSION: We DON'T re-describe here - that's too expensive.
        We just mark which observations show significant visual changes.
        The causal inference engine can analyze these later if needed.
        """
        logger.info("Detecting state changes using embedding similarity...")
        
        total_changes = 0
        total_comparisons = 0
        
        for entity in entities:
            if len(entity.observations) < 2:
                entity.state_changes = []
                continue
            
            changes = []
            observations = entity.observations
            
            # Compare consecutive observations
            for i in range(len(observations) - 1):
                curr_emb = observations[i].embedding
                next_emb = observations[i + 1].embedding
                
                # Calculate cosine similarity (embeddings are normalized)
                similarity = float(np.dot(curr_emb, next_emb))
                total_comparisons += 1
                
                if similarity < self.config.clustering.state_change_threshold:
                    # Significant visual change detected
                    change_info = {
                        'from_frame': observations[i].frame_number,
                        'to_frame': observations[i + 1].frame_number,
                        'from_time': observations[i].timestamp,
                        'to_time': observations[i + 1].timestamp,
                        'similarity': similarity,
                        'change_magnitude': 1.0 - similarity,
                        # NOTE: No new_description - we don't re-run FastVLM here
                        # The causal inference engine will analyze this if needed
                    }
                    changes.append(change_info)
                    total_changes += 1
            
            entity.state_changes = changes
        
        logger.info(f"✓ Completed {total_comparisons} embedding comparisons")
        if total_changes > 0:
            logger.info(f"✓ Detected {total_changes} state changes (similarity < {self.config.clustering.state_change_threshold})")
        else:
            logger.info("✓ No significant state changes detected")
        
        return entities


# ============================================================================
# MAIN TRACKING ENGINE
# ============================================================================

def run_tracking_engine(video_path: str, config: Optional[OrionConfig] = None) -> Tuple[List[Entity], List[Observation]]:
    """
    Main tracking-based perception pipeline
    
    Args:
        video_path: Path to video file
        config: Optional configuration (uses default if not provided)
    
    Returns:
        Tuple of (entities, observations)
    """
    if config is None:
        config = OrionConfig()
    
    logger.info("\n" + "="*80)
    logger.info("TRACKING-BASED PERCEPTION ENGINE")
    logger.info("Track First, Describe Once, Link Always")
    logger.info("="*80 + "\n")
    
    start_time = time.time()
    
    # Phase 1: Collect observations
    collector = ObservationCollector(config)
    observations = collector.process_video(video_path)
    
    # Phase 2: Cluster into entities
    tracker = EntityTracker(config)
    entities = tracker.cluster_observations(observations)
    
    # Phase 3: Describe each entity once
    describer = SmartDescriber(config)
    entities = describer.describe_entities(entities)
    
    # Phase 3b: Detect state changes
    entities = describer.detect_state_changes(entities)
    
    elapsed_time = time.time() - start_time
    
    logger.info("\n" + "="*80)
    logger.info("TRACKING ENGINE COMPLETE")
    logger.info("="*80)
    logger.info(f"Total observations: {len(observations)}")
    logger.info(f"Unique entities: {len(entities)}")
    logger.info(f"Descriptions generated: {len(entities) + sum(len(e.state_changes) for e in entities)}")
    logger.info(f"Total time: {elapsed_time:.2f}s")
    logger.info(f"Efficiency: {len(observations) / max(len(entities), 1):.1f}x fewer descriptions than detections")
    logger.info("="*80 + "\n")
    
    return entities, observations


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import json
    
    # Setup logging
    config = OrionConfig()
    logging.basicConfig(
        level=config.logging.level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    VIDEO_PATH = "data/examples/video1.mp4"
    OUTPUT_DIR = Path("data/testing")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not Path(VIDEO_PATH).exists():
        logger.error(f"Video not found: {VIDEO_PATH}")
        sys.exit(1)
    
    try:
        # Run tracking engine
        entities, observations = run_tracking_engine(VIDEO_PATH, config)
        
        # Save results
        output = {
            'entities': [e.to_dict() for e in entities],
            'stats': {
                'total_observations': len(observations),
                'unique_entities': len(entities),
                'efficiency_ratio': len(observations) / max(len(entities), 1)
            }
        }
        
        output_path = OUTPUT_DIR / 'tracking_results.json'
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Tracking engine failed: {e}", exc_info=True)
        sys.exit(1)
