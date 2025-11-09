"""
Zone Manager - Spatial Zone Detection and Management

Implements HDBSCAN-based clustering to discover and maintain spatial zones
(rooms, subzones, outdoor areas) from 3D entity observations.

Author: Orion Research Team
Date: November 2025
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import time


class ZoneType(Enum):
    """Types of spatial zones"""
    INDOOR_ROOM = "indoor_room"
    SUBZONE = "subzone"
    OUTDOOR_ZONE = "outdoor_zone"
    UNKNOWN = "unknown"


@dataclass
class Zone:
    """
    Represents a discovered spatial zone (room, subzone, outdoor area).
    """
    zone_id: str
    type: ZoneType
    centroid_3d_mm: np.ndarray  # (x, y, z) in mm
    footprint_2d: Optional[np.ndarray] = None  # Convex hull polygon (N, 2)
    bounding_volume: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (min, max)
    
    # Semantic information
    label: str = "unknown"  # kitchen, bedroom, outdoor, etc.
    embedding: Optional[np.ndarray] = None  # CLIP embedding of representative view
    confidence: float = 0.0
    
    # Temporal tracking
    first_seen: float = 0.0  # timestamp
    last_seen: float = 0.0  # timestamp
    persistence_score: float = 0.0  # how stable is this zone
    
    # Membership
    members: List[Tuple[str, float]] = field(default_factory=list)  # [(entity_id, timestamp), ...]
    entity_count: int = 0
    
    # Visualization
    representative_frames: List[Tuple[int, float, str]] = field(default_factory=list)  # [(frame_idx, timestamp, img_path), ...]
    color: Tuple[int, int, int] = (128, 128, 128)  # BGR for visualization


@dataclass
class ZoneObservation:
    """Single observation to be clustered into zones"""
    entity_id: str
    timestamp: float
    centroid_3d_mm: np.ndarray  # (x, y, z)
    embedding: Optional[np.ndarray] = None
    class_label: str = "unknown"
    frame_idx: int = 0


class ZoneManager:
    """
    Manages spatial zone detection and maintenance using HDBSCAN clustering.
    
    Modes:
    - DENSE (indoor): Full persistent zones with merging
    - SPARSE (outdoor): Sliding window, recent zones only
    """
    
    def __init__(
        self,
        mode: str = "dense",  # dense=indoor (permanent), sparse=outdoor (sliding window)
        min_cluster_size: int = 30,  # Increased for room-scale clustering
        min_samples: int = 5,
        merge_distance_mm: float = 3000.0,  # 3 meters for room-scale merging (reduced from 5m)
        min_observations: int = 50,  # Reduced to detect zones faster
        embedding_weight: float = 0.3,
        spatial_weight: float = 1.0,
        temporal_weight: float = 0.01,  # Drastically reduced - time shouldn't split zones
        temporal_scale: float = 100.0,  # Much larger scale to minimize temporal effect
        sliding_window_seconds: float = 300.0,  # 5 minutes for sparse mode
        recent_window_seconds: float = 30.0,  # Only use recent 30s for zone detection
    ):
        """
        Initialize zone manager.
        
        Args:
            mode: "dense" (indoor, permanent) or "sparse" (outdoor, sliding window)
            min_cluster_size: HDBSCAN min cluster size (larger for room-scale)
            min_samples: HDBSCAN min samples
            merge_distance_mm: Max distance to merge zones (room-scale)
            min_observations: Minimum observations before clustering starts
            embedding_weight: Weight for semantic similarity in clustering
            spatial_weight: Weight for 3D spatial distance
            temporal_weight: Weight for temporal proximity (minimal)
            temporal_scale: Scale factor for time → spatial units
            sliding_window_seconds: History retention for sparse mode
            recent_window_seconds: Window for zone detection (focus on recent observations)
        """
        self.mode = mode
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.merge_distance_mm = merge_distance_mm
        self.min_observations = min_observations
        self.embedding_weight = embedding_weight
        self.spatial_weight = spatial_weight
        self.temporal_weight = temporal_weight
        self.temporal_scale = temporal_scale
        self.sliding_window_seconds = sliding_window_seconds
        self.recent_window_seconds = recent_window_seconds
        
        # Zone storage
        self.zones: Dict[str, Zone] = {}
        self.next_zone_id = 0
        
        # Observation buffer for clustering
        self.observation_buffer: deque = deque(maxlen=2000)  # Increased buffer
        
        # Adjacency graph (zone connections)
        self.adjacency: Dict[str, List[Tuple[str, float]]] = {}  # zone_id → [(neighbor_id, weight), ...]
        
        # Scene classifier (for labeling zones)
        self.scene_classifier = None
        """
        Initialize zone manager.
        
        Args:
            mode: "dense" (indoor, permanent) or "sparse" (outdoor, sliding window)
            min_cluster_size: HDBSCAN min cluster size
            min_samples: HDBSCAN min samples
            merge_distance_mm: Max distance to merge zones
            embedding_weight: Weight for semantic similarity in clustering
            spatial_weight: Weight for 3D spatial distance
            temporal_weight: Weight for temporal proximity
            temporal_scale: Scale factor for time → spatial units
            sliding_window_seconds: History retention for sparse mode
        """
        self.mode = mode
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.merge_distance_mm = merge_distance_mm
        self.embedding_weight = embedding_weight
        self.spatial_weight = spatial_weight
        self.temporal_weight = temporal_weight
        self.temporal_scale = temporal_scale
        self.sliding_window_seconds = sliding_window_seconds
        
        # Zone storage
        self.zones: Dict[str, Zone] = {}
        self.next_zone_id = 0
        
        # Observation buffer for clustering
        self.observation_buffer: deque = deque(maxlen=1000)
        
        # Adjacency graph (zone connections)
        self.adjacency: Dict[str, List[Tuple[str, float]]] = {}  # zone_id → [(neighbor_id, weight), ...]
        
        # Scene classifier (for labeling zones)
        self.scene_classifier = None
    
    def add_observation(
        self,
        entity_id: str,
        timestamp: float,
        centroid_3d_mm: np.ndarray,
        embedding: Optional[np.ndarray] = None,
        class_label: str = "unknown",
        frame_idx: int = 0
    ):
        """
        Add entity observation to buffer for clustering.
        
        Args:
            entity_id: Entity ID
            timestamp: Observation timestamp
            centroid_3d_mm: 3D position in mm (x, y, z)
            embedding: Optional CLIP embedding
            class_label: Entity class
            frame_idx: Frame index
        """
        obs = ZoneObservation(
            entity_id=entity_id,
            timestamp=timestamp,
            centroid_3d_mm=centroid_3d_mm,
            embedding=embedding,
            class_label=class_label,
            frame_idx=frame_idx
        )
        self.observation_buffer.append(obs)
    
    def update_zones(
        self,
        current_timestamp: float,
        frame: Optional[np.ndarray] = None
    ):
        """
        Run clustering and update zones.
        
        Args:
            current_timestamp: Current time
            frame: Optional frame for scene classification
        """
        # Check minimum observations threshold
        if len(self.observation_buffer) < self.min_observations:
            return
        
        # Filter to recent observations only (last 30 seconds)
        # This allows detecting new rooms as camera moves
        cutoff_time = current_timestamp - self.recent_window_seconds
        recent_observations = [
            obs for obs in self.observation_buffer
            if obs.timestamp >= cutoff_time
        ]
        
        if len(recent_observations) < 10:  # Need at least 10 recent observations
            recent_observations = list(self.observation_buffer)[-50:]  # Use last 50 if not enough recent
        
        # Filter observations by mode
        if self.mode == "sparse":
            # Only keep recent observations
            cutoff_time = current_timestamp - self.sliding_window_seconds
            observations = [
                obs for obs in recent_observations
                if obs.timestamp >= cutoff_time
            ]
        else:
            # Dense mode: use recent observations
            observations = recent_observations
        
        if len(observations) < self.min_observations:
            return  # Not enough data yet
        
        # Aggregate observations by entity (CRITICAL FIX)
        # Don't cluster individual observations, cluster entity centroids!
        entity_data = self._aggregate_observations_by_entity(observations)
        
        if len(entity_data) < 3:  # Need at least 3 entities to form a zone
            return
        
        # Run clustering on aggregated entity positions
        new_zones = self._cluster_entity_centroids(entity_data, current_timestamp)
        
        # Match new zones with existing zones (RE-IDENTIFICATION)
        # This prevents creating duplicate zones when returning to same room
        matched_zones, unmatched_new = self._match_zones_with_history(new_zones, current_timestamp)
        
        # Update matched zones
        for existing_zone_id, new_zone in matched_zones:
            self.zones[existing_zone_id].last_seen = current_timestamp
            self.zones[existing_zone_id].centroid_3d_mm = new_zone.centroid_3d_mm
            self.zones[existing_zone_id].members.extend(new_zone.members)
            self.zones[existing_zone_id].entity_count = len(set([m[0] for m in self.zones[existing_zone_id].members]))
            self.zones[existing_zone_id].persistence_score = min(2.0, self.zones[existing_zone_id].persistence_score + 0.1)
        
        # Add unmatched new zones
        for new_zone in unmatched_new:
            self.zones[new_zone.zone_id] = new_zone
        
        # Label zones using scene classification
        if frame is not None and self.scene_classifier is None:
            from orion.semantic.scene_classifier import SceneClassifier
            self.scene_classifier = SceneClassifier()
        
        if frame is not None and self.scene_classifier:
            self._label_zones(frame, current_timestamp)
        
        # Update persistence scores
        self._update_persistence_scores(current_timestamp)
        
        # Clean up old zones (sparse mode only)
        if self.mode == "sparse":
            self._cleanup_old_zones(current_timestamp)
    
    def _aggregate_observations_by_entity(
        self,
        observations: List[ZoneObservation]
    ) -> Dict[str, Dict]:
        """
        Aggregate observations by entity ID.
        Computes mean position, embedding, and class for each unique entity.
        
        This is the KEY FIX: Instead of clustering 641 observations,
        cluster 24 entity centroids (one per unique entity).
        
        Args:
            observations: Raw observations
        
        Returns:
            Dict mapping entity_id → {centroid_3d, embedding, class_label, count}
        """
        entity_data = {}
        
        for obs in observations:
            if obs.entity_id not in entity_data:
                entity_data[obs.entity_id] = {
                    'positions': [],
                    'embeddings': [],
                    'class_labels': [],
                    'timestamps': []
                }
            
            entity_data[obs.entity_id]['positions'].append(obs.centroid_3d_mm)
            entity_data[obs.entity_id]['timestamps'].append(obs.timestamp)
            entity_data[obs.entity_id]['class_labels'].append(obs.class_label)
            if obs.embedding is not None:
                entity_data[obs.entity_id]['embeddings'].append(obs.embedding)
        
        # Compute aggregates
        aggregated = {}
        for entity_id, data in entity_data.items():
            positions = np.array(data['positions'])
            aggregated[entity_id] = {
                'centroid_3d_mm': np.mean(positions, axis=0),
                'std_3d_mm': np.std(positions, axis=0),
                'embedding': np.mean(data['embeddings'], axis=0) if data['embeddings'] else None,
                'class_label': max(set(data['class_labels']), key=data['class_labels'].count),
                'observation_count': len(positions),
                'latest_timestamp': max(data['timestamps'])
            }
        
        return aggregated
    
    def _match_zones_with_history(
        self,
        new_zones: List[Zone],
        current_timestamp: float
    ) -> Tuple[List[Tuple[str, Zone]], List[Zone]]:
        """
        Match newly detected zones with existing zones (zone re-identification).
        
        This prevents creating duplicate zones when camera returns to a previously seen room.
        Uses spatial proximity and semantic similarity.
        
        Args:
            new_zones: Newly detected zones from current clustering
            current_timestamp: Current timestamp
        
        Returns:
            Tuple of (matched_zones, unmatched_new_zones)
            - matched_zones: [(existing_zone_id, new_zone), ...]
            - unmatched_new_zones: [Zone, ...]
        """
        if not self.zones:
            # No existing zones, all new zones are unmatched
            return [], new_zones
        
        matched = []
        unmatched = []
        used_existing_ids = set()
        
        # Match threshold: 4m spatial distance (increased for room re-ID across time)
        # We need looser matching since camera might be at different position in same room
        spatial_threshold_mm = 4000.0  # 4 meters
        semantic_threshold = 0.6
        
        for new_zone in new_zones:
            best_match_id = None
            best_match_score = 0.0
            
            for existing_id, existing_zone in self.zones.items():
                if existing_id in used_existing_ids:
                    continue  # Already matched
                
                # Spatial similarity (inverse distance)
                spatial_dist = np.linalg.norm(
                    new_zone.centroid_3d_mm - existing_zone.centroid_3d_mm
                )
                spatial_score = max(0.0, 1.0 - (spatial_dist / spatial_threshold_mm))
                
                # Semantic similarity (embedding)
                semantic_score = 0.0
                if (new_zone.embedding is not None and 
                    existing_zone.embedding is not None):
                    semantic_score = np.dot(new_zone.embedding, existing_zone.embedding)
                    semantic_score = max(0.0, min(1.0, (semantic_score + 1.0) / 2.0))  # Normalize to [0, 1]
                
                # Combined score (weighted - spatial more important for room re-ID)
                combined_score = 0.8 * spatial_score + 0.2 * semantic_score
                
                # Lower threshold for matching (0.4 instead of 0.5)
                # This allows matching even when camera is at different angle in same room
                if combined_score > best_match_score and combined_score > 0.4:
                    best_match_score = combined_score
                    best_match_id = existing_id
            
            if best_match_id is not None:
                matched.append((best_match_id, new_zone))
                used_existing_ids.add(best_match_id)
            else:
                unmatched.append(new_zone)
        
        return matched, unmatched
    
    def _cluster_entity_centroids(
        self,
        entity_data: Dict[str, Dict],
        current_timestamp: float
    ) -> List[Zone]:
        """
        Cluster entity centroids (not raw observations).
        This produces room-scale zones instead of over-segmentation.
        
        Args:
            entity_data: Aggregated entity data
            current_timestamp: Current timestamp
        
        Returns:
            List of discovered zones
        """
        entity_ids = list(entity_data.keys())
        centroids = np.array([entity_data[eid]['centroid_3d_mm'] for eid in entity_ids])
        embeddings = [entity_data[eid]['embedding'] for eid in entity_ids]
        
        # Build feature matrix
        features = []
        for i, eid in enumerate(entity_ids):
            # Spatial features (dominant) - convert to meters
            spatial_feat = centroids[i][:2] / 1000.0  # x, y in meters
            spatial_feat = spatial_feat * self.spatial_weight
            
            # Semantic feature (embedding, optional)
            if embeddings[i] is not None and self.embedding_weight > 0:
                # Use first few dims of embedding
                semantic_feat = embeddings[i][:16] * self.embedding_weight
                feat = np.concatenate([spatial_feat, semantic_feat])
            else:
                feat = spatial_feat
            
            features.append(feat)
        
        features = np.array(features)
        
        # Use simple DBSCAN for room-scale clustering
        # For indoor: entities within 2-3m should be same zone (room-scale)
        # Lower eps to distinguish separate rooms
        from sklearn.cluster import DBSCAN
        
        # eps in feature space: 2.5m for dense (detect separate rooms)
        # Higher min_samples would merge more, but we want room separation
        eps = 2.5 if self.mode == "dense" else 8.0
        clusterer = DBSCAN(eps=eps, min_samples=2, metric='euclidean')
        
        labels = clusterer.fit_predict(features)
        
        # Build zones from clusters
        zones = []
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise label
        
        for cluster_id in unique_labels:
            cluster_mask = labels == cluster_id
            cluster_entity_ids = [eid for eid, mask in zip(entity_ids, cluster_mask) if mask]
            cluster_centroids = centroids[cluster_mask]
            cluster_embeddings = [emb for emb, mask in zip(embeddings, cluster_mask) if mask and emb is not None]
            
            if len(cluster_entity_ids) < 2:
                continue
            
            # Compute zone properties
            centroid_mean = np.mean(cluster_centroids, axis=0)
            
            # Convex hull for footprint (2D)
            points_2d = cluster_centroids[:, :2]  # x, y
            footprint = self._compute_convex_hull(points_2d)
            
            # Bounding volume (3D)
            bbox_min = np.min(cluster_centroids, axis=0)
            bbox_max = np.max(cluster_centroids, axis=0)
            bounding_volume = {
                'min': bbox_min.tolist(),
                'max': bbox_max.tolist(),
                'size': (bbox_max - bbox_min).tolist()
            }
            
            # Mean embedding
            if cluster_embeddings:
                zone_embedding = np.mean(cluster_embeddings, axis=0)
            else:
                zone_embedding = None
            
            # Create zone
            zone_id = f"zone_{self.next_zone_id}"
            self.next_zone_id += 1
            
            zone = Zone(
                zone_id=zone_id,
                type=ZoneType.INDOOR_ROOM,
                centroid_3d_mm=centroid_mean,
                footprint_2d=footprint,
                bounding_volume=(bbox_min, bbox_max),
                label="unknown",
                embedding=zone_embedding,
                confidence=0.8,
                first_seen=current_timestamp,
                last_seen=current_timestamp,
                persistence_score=1.0,
                members=[(eid, current_timestamp) for eid in cluster_entity_ids],
                entity_count=len(cluster_entity_ids)
            )
            
            zones.append(zone)
        
        return zones
    
    def _cluster_observations(
        self,
        observations: List[ZoneObservation],
        current_timestamp: float
    ) -> List[Zone]:
        """
        OLD METHOD - Kept for compatibility but should not be used.
        Use _cluster_entity_centroids instead.
        
        Cluster observations using HDBSCAN.
        
        Args:
            observations: List of observations to cluster
            current_timestamp: Current timestamp
        
        Returns:
            List of discovered zones
        """
        # Build feature matrix
        features = []
        for obs in observations:
            # Spatial features (dominant)
            spatial_feat = obs.centroid_3d_mm[:2] / 1000.0  # Convert to meters (x, y)
            spatial_feat = spatial_feat * self.spatial_weight
            
            # Temporal feature (optional, small weight)
            temporal_feat = [(current_timestamp - obs.timestamp) / self.temporal_scale]
            temporal_feat = [temporal_feat[0] * self.temporal_weight]
            
            # Semantic feature (embedding, optional)
            if obs.embedding is not None and self.embedding_weight > 0:
                # Use first few dims of embedding
                semantic_feat = obs.embedding[:16] * self.embedding_weight
                feat = np.concatenate([spatial_feat, temporal_feat, semantic_feat])
            else:
                feat = np.concatenate([spatial_feat, temporal_feat])
            
            features.append(feat)
        
        features = np.array(features)
        
        # Run HDBSCAN
        try:
            import hdbscan
        except ImportError:
            print("⚠️  hdbscan not installed, falling back to simple spatial clustering")
            return self._fallback_spatial_clustering(observations)
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        
        labels = clusterer.fit_predict(features)
        
        # Build zones from clusters
        zones = []
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise label
        
        for cluster_id in unique_labels:
            cluster_mask = labels == cluster_id
            cluster_obs = [obs for obs, mask in zip(observations, cluster_mask) if mask]
            
            if len(cluster_obs) < self.min_cluster_size:
                continue
            
            # Compute zone properties
            centroids = np.array([obs.centroid_3d_mm for obs in cluster_obs])
            centroid_mean = np.mean(centroids, axis=0)
            
            # Convex hull for footprint (2D)
            points_2d = centroids[:, :2]  # x, y
            footprint = self._compute_convex_hull(points_2d)
            
            # Bounding volume (3D)
            bbox_min = np.min(centroids, axis=0)
            bbox_max = np.max(centroids, axis=0)
            
            # Representative embedding
            embeddings = [obs.embedding for obs in cluster_obs if obs.embedding is not None]
            if embeddings:
                embedding_mean = np.mean(embeddings, axis=0)
            else:
                embedding_mean = None
            
            # Assign color
            color = self._generate_zone_color(len(zones))
            
            zone = Zone(
                zone_id=f"temp_{cluster_id}",  # Temporary ID
                type=ZoneType.INDOOR_ROOM if self.mode == "dense" else ZoneType.OUTDOOR_ZONE,
                centroid_3d_mm=centroid_mean,
                footprint_2d=footprint,
                bounding_volume=(bbox_min, bbox_max),
                embedding=embedding_mean,
                confidence=0.5,
                first_seen=min(obs.timestamp for obs in cluster_obs),
                last_seen=max(obs.timestamp for obs in cluster_obs),
                members=[(obs.entity_id, obs.timestamp) for obs in cluster_obs],
                entity_count=len(cluster_obs),
                color=color
            )
            
            zones.append(zone)
        
        return zones
    
    def _fallback_spatial_clustering(
        self,
        observations: List[ZoneObservation]
    ) -> List[Zone]:
        """
        Simple spatial clustering fallback if HDBSCAN not available.
        Uses K-means or DBSCAN from sklearn.
        """
        from sklearn.cluster import DBSCAN
        
        # Use only spatial features
        positions = np.array([obs.centroid_3d_mm[:2] / 1000.0 for obs in observations])
        
        clusterer = DBSCAN(
            eps=self.merge_distance_mm / 1000.0,  # Convert to meters
            min_samples=self.min_samples
        )
        
        labels = clusterer.fit_predict(positions)
        
        # Build zones (same logic as above, simplified)
        zones = []
        unique_labels = set(labels)
        unique_labels.discard(-1)
        
        for cluster_id in unique_labels:
            cluster_mask = labels == cluster_id
            cluster_obs = [obs for obs, mask in zip(observations, cluster_mask) if mask]
            
            if len(cluster_obs) < self.min_cluster_size:
                continue
            
            centroids = np.array([obs.centroid_3d_mm for obs in cluster_obs])
            centroid_mean = np.mean(centroids, axis=0)
            
            zone = Zone(
                zone_id=f"temp_{cluster_id}",
                type=ZoneType.UNKNOWN,
                centroid_3d_mm=centroid_mean,
                first_seen=min(obs.timestamp for obs in cluster_obs),
                last_seen=max(obs.timestamp for obs in cluster_obs),
                members=[(obs.entity_id, obs.timestamp) for obs in cluster_obs],
                entity_count=len(cluster_obs),
                color=self._generate_zone_color(len(zones))
            )
            
            zones.append(zone)
        
        return zones
    
    def _merge_or_create_zones(
        self,
        new_zones: List[Zone],
        current_timestamp: float
    ):
        """
        Merge new zones with existing zones or create new ones.
        
        Args:
            new_zones: Newly discovered zones
            current_timestamp: Current timestamp
        """
        for new_zone in new_zones:
            # Find best matching existing zone
            best_match_id = None
            best_score = 0.0
            
            for zone_id, existing_zone in self.zones.items():
                score = self._zone_similarity(new_zone, existing_zone)
                if score > best_score:
                    best_score = score
                    best_match_id = zone_id
            
            # Merge or create
            if best_score > 0.7 and best_match_id is not None:
                # Merge with existing zone
                self._merge_zones(best_match_id, new_zone, current_timestamp)
            else:
                # Create new zone
                zone_id = f"zone_{self.next_zone_id}"
                self.next_zone_id += 1
                
                new_zone.zone_id = zone_id
                new_zone.last_seen = current_timestamp
                self.zones[zone_id] = new_zone
    
    def _zone_similarity(self, zone_a: Zone, zone_b: Zone) -> float:
        """
        Compute similarity between two zones.
        
        Args:
            zone_a: First zone
            zone_b: Second zone
        
        Returns:
            Similarity score [0, 1]
        """
        # Spatial distance
        dist_3d = np.linalg.norm(zone_a.centroid_3d_mm - zone_b.centroid_3d_mm)
        spatial_score = max(0, 1 - (dist_3d / self.merge_distance_mm) ** 2)
        
        # Embedding similarity
        if zone_a.embedding is not None and zone_b.embedding is not None:
            emb_sim = np.dot(zone_a.embedding, zone_b.embedding) / (
                np.linalg.norm(zone_a.embedding) * np.linalg.norm(zone_b.embedding) + 1e-6
            )
            emb_score = (emb_sim + 1) / 2  # Map [-1, 1] → [0, 1]
        else:
            emb_score = 0.5
        
        # Combined score
        score = 0.7 * spatial_score + 0.3 * emb_score
        
        return float(score)
    
    def _merge_zones(
        self,
        existing_zone_id: str,
        new_zone: Zone,
        current_timestamp: float
    ):
        """
        Merge new zone into existing zone.
        
        Args:
            existing_zone_id: ID of existing zone
            new_zone: New zone to merge
            current_timestamp: Current timestamp
        """
        existing = self.zones[existing_zone_id]
        
        # Update centroid (weighted average)
        weight_existing = existing.entity_count
        weight_new = new_zone.entity_count
        total_weight = weight_existing + weight_new
        
        existing.centroid_3d_mm = (
            existing.centroid_3d_mm * weight_existing +
            new_zone.centroid_3d_mm * weight_new
        ) / total_weight
        
        # Update embedding
        if existing.embedding is not None and new_zone.embedding is not None:
            existing.embedding = (
                existing.embedding * weight_existing +
                new_zone.embedding * weight_new
            ) / total_weight
        elif new_zone.embedding is not None:
            existing.embedding = new_zone.embedding
        
        # Update members
        existing.members.extend(new_zone.members)
        existing.entity_count = len(existing.members)
        
        # Update timestamps
        existing.last_seen = current_timestamp
        
        # Increase persistence
        existing.persistence_score = min(1.0, existing.persistence_score + 0.1)
    
    def _label_zones(self, frame: np.ndarray, current_timestamp: float):
        """
        Label zones using scene classification.
        
        Args:
            frame: Current video frame
            current_timestamp: Current timestamp
        """
        if not self.scene_classifier:
            return
        
        # Classify scene
        scene_type, confidence = self.scene_classifier.classify(frame)
        
        # Apply to all zones (simple heuristic: all zones get same label for now)
        # TODO: Improve with per-zone classification using representative frames
        for zone in self.zones.values():
            if zone.label == "unknown" or zone.confidence < confidence:
                zone.label = scene_type.value
                zone.confidence = confidence
    
    def _update_persistence_scores(self, current_timestamp: float):
        """
        Update persistence scores for all zones.
        
        Args:
            current_timestamp: Current timestamp
        """
        for zone in self.zones.values():
            time_since_seen = current_timestamp - zone.last_seen
            
            # Decay persistence if not seen recently
            if time_since_seen > 30.0:  # 30 seconds
                zone.persistence_score *= 0.95
            else:
                # Boost persistence if actively seen
                zone.persistence_score = min(1.0, zone.persistence_score + 0.05)
    
    def _cleanup_old_zones(self, current_timestamp: float):
        """
        Remove old zones (sparse mode only).
        
        Args:
            current_timestamp: Current timestamp
        """
        zones_to_remove = []
        
        for zone_id, zone in self.zones.items():
            time_since_seen = current_timestamp - zone.last_seen
            
            if time_since_seen > self.sliding_window_seconds:
                zones_to_remove.append(zone_id)
        
        for zone_id in zones_to_remove:
            del self.zones[zone_id]
            if zone_id in self.adjacency:
                del self.adjacency[zone_id]
    
    def query_zone_by_point(
        self,
        point_3d_mm: np.ndarray
    ) -> Optional[str]:
        """
        Find which zone contains a given 3D point.
        
        Args:
            point_3d_mm: 3D point (x, y, z) in mm
        
        Returns:
            Zone ID or None
        """
        best_zone_id = None
        min_distance = float('inf')
        
        for zone_id, zone in self.zones.items():
            dist = np.linalg.norm(point_3d_mm - zone.centroid_3d_mm)
            
            # Check if within bounding volume
            if zone.bounding_volume:
                bbox_min, bbox_max = zone.bounding_volume
                if np.all(point_3d_mm >= bbox_min) and np.all(point_3d_mm <= bbox_max):
                    if dist < min_distance:
                        min_distance = dist
                        best_zone_id = zone_id
        
        return best_zone_id
    
    def get_zone_statistics(self) -> dict:
        """Get statistics about current zones."""
        return {
            'total_zones': len(self.zones),
            'zone_types': {
                zone_type.value: sum(1 for z in self.zones.values() if z.type == zone_type)
                for zone_type in ZoneType
            },
            'total_observations': len(self.observation_buffer),
            'mode': self.mode,
        }
    
    def _compute_convex_hull(self, points_2d: np.ndarray) -> Optional[np.ndarray]:
        """Compute convex hull of 2D points."""
        if len(points_2d) < 3:
            return None
        
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(points_2d)
            return points_2d[hull.vertices]
        except:
            return None
    
    def _generate_zone_color(self, zone_index: int) -> Tuple[int, int, int]:
        """Generate unique color for zone visualization."""
        import colorsys
        golden_ratio = 0.618033988749895
        hue = (zone_index * golden_ratio) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
        # Convert to BGR for OpenCV
        return (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
