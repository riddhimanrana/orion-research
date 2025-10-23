"""
Spatial Co-Location Zone System
================================

Tracks regions where multiple entities co-occur over time, enabling queries like:
"Which entities were together in location X during time period Y?"

This is distinct from simple position classification (left/right/center).

Author: Orion Research Team  
Date: October 22, 2024
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
from sklearn.cluster import DBSCAN

logger = logging.getLogger('orion.spatial_colocation')


@dataclass
class BoundingBox:
    """Simple bounding box representation"""
    x1: float
    y1: float
    x2: float
    y2: float
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get centroid of bbox"""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def area(self) -> float:
        """Get area of bbox"""
        return (self.x2 - self.x1) * (self.y2 - self.y1)
    
    def iou(self, other: 'BoundingBox') -> float:
        """Compute IoU with another bbox"""
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        
        if x1 >= x2 or y1 >= y2:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def distance_to(self, other: 'BoundingBox') -> float:
        """Euclidean distance between centers"""
        cx1, cy1 = self.center
        cx2, cy2 = other.center
        return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)


@dataclass
class EntityAppearance:
    """Single appearance of an entity in a frame"""
    entity_id: str
    frame_idx: int
    timestamp_ms: float
    bbox: BoundingBox
    class_name: str
    confidence: float


@dataclass
class SpatialCoLocationZone:
    """
    Represents a spatial region where multiple entities co-occur over time.
    
    This is what we actually want: tracking which entities were together in which
    locations, not just classifying individual entity positions.
    """
    zone_id: str
    
    # Spatial extent
    spatial_bounds: BoundingBox  # Bounding box of the zone
    frame_width: int
    frame_height: int
    
    # Temporal extent
    frame_range: Tuple[int, int]  # (start_frame, end_frame)
    duration_ms: float
    
    # Entities present in this zone
    entity_ids: Set[str]  # All entities that appeared here
    entity_appearances: Dict[str, List[int]]  # entity_id -> list of frame indices
    
    # Semantic description
    location_descriptor: str  # "left area", "center region", etc.
    activity_summary: Optional[str] = None  # "person and cup interact"
    
    # Statistics
    total_frames: int = 0  # Number of frames this zone was active
    max_concurrent_entities: int = 0  # Peak number of entities at once
    
    def __post_init__(self):
        """Calculate derived statistics"""
        self.total_frames = self.frame_range[1] - self.frame_range[0] + 1
        
        # Calculate max concurrent entities
        frame_counts: Dict[int, int] = defaultdict(int)
        for entity_id, frames in self.entity_appearances.items():
            for frame_idx in frames:
                frame_counts[frame_idx] += 1
        
        self.max_concurrent_entities = max(frame_counts.values()) if frame_counts else 0
    
    def get_location_descriptor(self) -> str:
        """Generate human-readable location descriptor"""
        cx, cy = self.spatial_bounds.center
        
        # Normalize to 0-1
        nx = cx / self.frame_width
        ny = cy / self.frame_height
        
        # Horizontal position
        if nx < 0.33:
            h_pos = "left"
        elif nx > 0.66:
            h_pos = "right"
        else:
            h_pos = "center"
        
        # Vertical position
        if ny < 0.33:
            v_pos = "upper"
        elif ny > 0.66:
            v_pos = "lower"
        else:
            v_pos = "middle"
        
        # Combine
        if v_pos == "middle" and h_pos == "center":
            return "center area"
        elif v_pos == "middle":
            return f"{h_pos} side"
        elif h_pos == "center":
            return f"{v_pos} area"
        else:
            return f"{v_pos} {h_pos} area"


class SpatialCoLocationAnalyzer:
    """
    Analyzes entity positions over time to identify spatial zones where
    multiple entities co-occur.
    """
    
    def __init__(
        self,
        proximity_threshold_pixels: float = 200.0,
        min_frames_for_zone: int = 5,
        temporal_merge_gap: int = 30  # Merge zones within 30 frames
    ):
        """
        Initialize analyzer.
        
        Args:
            proximity_threshold_pixels: Max distance to consider entities "together"
            min_frames_for_zone: Minimum frames for a zone to be considered valid
            temporal_merge_gap: Max frame gap to merge temporally adjacent zones
        """
        self.proximity_threshold = proximity_threshold_pixels
        self.min_frames = min_frames_for_zone
        self.temporal_merge_gap = temporal_merge_gap
    
    def analyze_colocation_zones(
        self,
        entity_appearances: List[EntityAppearance],
        frame_width: int,
        frame_height: int
    ) -> List[SpatialCoLocationZone]:
        """
        Analyze entity appearances to identify spatial co-location zones.
        
        Args:
            entity_appearances: All entity appearances across frames
            frame_width: Video frame width
            frame_height: Video frame height
        
        Returns:
            List of identified co-location zones
        """
        logger.info(f"Analyzing {len(entity_appearances)} entity appearances for co-location")
        
        # Group appearances by frame
        frames: Dict[int, List[EntityAppearance]] = defaultdict(list)
        for appearance in entity_appearances:
            frames[appearance.frame_idx].append(appearance)
        
        # Identify spatial clusters in each frame
        frame_clusters = {}
        for frame_idx, appearances in frames.items():
            clusters = self._cluster_entities_in_frame(appearances, frame_width, frame_height)
            if clusters:
                frame_clusters[frame_idx] = clusters
        
        # Track clusters across time to form zones
        zones = self._merge_temporal_clusters(frame_clusters, frame_width, frame_height)
        
        # Filter out zones that are too short
        valid_zones = [z for z in zones if z.total_frames >= self.min_frames]
        
        logger.info(f"Identified {len(valid_zones)} co-location zones (filtered from {len(zones)})")
        return valid_zones
    
    def _cluster_entities_in_frame(
        self,
        appearances: List[EntityAppearance],
        frame_width: int,
        frame_height: int
    ) -> List[Dict]:
        """
        Cluster entities that are spatially close in a single frame.
        
        CRITICAL: Only create clusters where multiple DIFFERENT entities co-occur.
        We don't want 1:1 mapping (445 zones = 445 observations bug).
        
        Returns:
            List of clusters, each containing entity IDs and spatial bounds
        """
        if len(appearances) < 2:
            # Need at least 2 entities to form a co-location
            return []
        
        # CRITICAL FIX: Check if we have multiple UNIQUE entities
        unique_entities = set(app.entity_id for app in appearances)
        if len(unique_entities) < 2:
            # All appearances are from same entity - no co-location
            return []
        
        # Extract positions
        positions = np.array([app.bbox.center for app in appearances])
        
        # Cluster using DBSCAN with pixel-based distance
        # eps is the maximum distance between two samples to be considered in same neighborhood
        clustering = DBSCAN(eps=self.proximity_threshold, min_samples=2, metric='euclidean').fit(positions)
        
        # Group by cluster
        clusters_dict: Dict[int, List[EntityAppearance]] = defaultdict(list)
        for idx, label in enumerate(clustering.labels_):
            if label != -1:  # Ignore noise points
                clusters_dict[label].append(appearances[idx])
        
        # Convert to cluster format
        clusters = []
        for cluster_id, cluster_apps in clusters_dict.items():
            # CRITICAL FIX: Verify this cluster has multiple unique entities
            unique_entities_in_cluster = set(app.entity_id for app in cluster_apps)
            if len(unique_entities_in_cluster) < 2:
                # Skip single-entity clusters
                continue
            
            # Compute bounding box that encompasses all entities
            all_x1 = [app.bbox.x1 for app in cluster_apps]
            all_y1 = [app.bbox.y1 for app in cluster_apps]
            all_x2 = [app.bbox.x2 for app in cluster_apps]
            all_y2 = [app.bbox.y2 for app in cluster_apps]
            
            cluster_bbox = BoundingBox(
                x1=min(all_x1),
                y1=min(all_y1),
                x2=max(all_x2),
                y2=max(all_y2)
            )
            
            clusters.append({
                'bbox': cluster_bbox,
                'entities': {app.entity_id for app in cluster_apps},
                'appearances': cluster_apps
            })
        
        return clusters
    
    def _merge_temporal_clusters(
        self,
        frame_clusters: Dict[int, List[Dict]],
        frame_width: int,
        frame_height: int
    ) -> List[SpatialCoLocationZone]:
        """
        Merge spatially and temporally adjacent clusters into zones.
        
        This tracks clusters across frames to form persistent zones.
        """
        zones: List[SpatialCoLocationZone] = []
        active_zones: List[Dict] = []  # Currently tracked zones
        
        # Sort frames
        sorted_frames = sorted(frame_clusters.keys())
        
        for frame_idx in sorted_frames:
            clusters = frame_clusters[frame_idx]
            
            # Try to match clusters to active zones
            matched_zones = set()
            matched_clusters = set()
            
            for zone_idx, zone_data in enumerate(active_zones):
                for cluster_idx, cluster in enumerate(clusters):
                    # Check spatial overlap
                    iou = zone_data['bbox'].iou(cluster['bbox'])
                    
                    # Check entity overlap
                    entity_overlap = len(zone_data['entities'] & cluster['entities'])
                    entity_overlap_ratio = entity_overlap / len(zone_data['entities'] | cluster['entities'])
                    
                    # Match if spatially overlapping OR same entities
                    if iou > 0.3 or entity_overlap_ratio > 0.5:
                        # Extend zone
                        zone_data['frame_end'] = frame_idx
                        zone_data['entities'] |= cluster['entities']
                        
                        # Update appearances
                        for app in cluster['appearances']:
                            zone_data['entity_appearances'][app.entity_id].append(frame_idx)
                        
                        # Update spatial bounds (expand if needed)
                        zone_data['bbox'] = BoundingBox(
                            x1=min(zone_data['bbox'].x1, cluster['bbox'].x1),
                            y1=min(zone_data['bbox'].y1, cluster['bbox'].y1),
                            x2=max(zone_data['bbox'].x2, cluster['bbox'].x2),
                            y2=max(zone_data['bbox'].y2, cluster['bbox'].y2)
                        )
                        
                        matched_zones.add(zone_idx)
                        matched_clusters.add(cluster_idx)
                        break  # One cluster can only match one zone per frame
            
            # Create new zones from unmatched clusters
            for cluster_idx, cluster in enumerate(clusters):
                if cluster_idx not in matched_clusters:
                    new_zone = {
                        'bbox': cluster['bbox'],
                        'frame_start': frame_idx,
                        'frame_end': frame_idx,
                        'entities': cluster['entities'].copy(),
                        'entity_appearances': defaultdict(list)
                    }
                    
                    for app in cluster['appearances']:
                        new_zone['entity_appearances'][app.entity_id].append(frame_idx)
                    
                    active_zones.append(new_zone)
            
            # Finalize zones that haven't been updated recently
            zones_to_finalize = []
            for zone_idx, zone_data in enumerate(active_zones):
                if zone_idx not in matched_zones:
                    # Check if zone has been inactive too long
                    if frame_idx - zone_data['frame_end'] > self.temporal_merge_gap:
                        zones_to_finalize.append(zone_idx)
            
            # Finalize and remove inactive zones
            for zone_idx in sorted(zones_to_finalize, reverse=True):
                zone_data = active_zones.pop(zone_idx)
                zones.append(self._create_zone_from_data(zone_data, frame_width, frame_height))
        
        # Finalize remaining active zones
        for zone_data in active_zones:
            zones.append(self._create_zone_from_data(zone_data, frame_width, frame_height))
        
        return zones
    
    def _create_zone_from_data(
        self,
        zone_data: Dict,
        frame_width: int,
        frame_height: int
    ) -> SpatialCoLocationZone:
        """Convert zone tracking data to final zone object"""
        frame_start = zone_data['frame_start']
        frame_end = zone_data['frame_end']
        duration_ms = (frame_end - frame_start) * (1000 / 30)  # Assuming 30 FPS
        
        zone_id = f"zone_{frame_start}_{frame_end}_{len(zone_data['entities'])}"
        
        zone = SpatialCoLocationZone(
            zone_id=zone_id,
            spatial_bounds=zone_data['bbox'],
            frame_width=frame_width,
            frame_height=frame_height,
            frame_range=(frame_start, frame_end),
            duration_ms=duration_ms,
            entity_ids=zone_data['entities'],
            entity_appearances=dict(zone_data['entity_appearances']),
            location_descriptor=""  # Will be set below
        )
        
        # Generate location descriptor
        zone.location_descriptor = zone.get_location_descriptor()
        
        return zone


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_appearances_from_observations(
    observations: List[Dict],
    fps: float = 30.0
) -> List[EntityAppearance]:
    """
    Convert observation log to list of entity appearances.
    
    Args:
        observations: List of observation dicts from tracking engine
        fps: Frames per second for timestamp calculation
    
    Returns:
        List of EntityAppearance objects
    """
    appearances = []
    
    for obs in observations:
        entity_id = str(obs.get('entity_id', obs.get('id', 'unknown')))
        frame_idx = obs.get('frame_idx', 0)
        bbox_data = obs.get('bbox', [0, 0, 0, 0])
        
        if len(bbox_data) >= 4:
            bbox = BoundingBox(
                x1=float(bbox_data[0]),
                y1=float(bbox_data[1]),
                x2=float(bbox_data[2]),
                y2=float(bbox_data[3])
            )
            
            appearance = EntityAppearance(
                entity_id=entity_id,
                frame_idx=frame_idx,
                timestamp_ms=frame_idx * (1000.0 / fps),
                bbox=bbox,
                class_name=obs.get('class', obs.get('class_name', 'unknown')),
                confidence=obs.get('confidence', 1.0)
            )
            
            appearances.append(appearance)
    
    return appearances
