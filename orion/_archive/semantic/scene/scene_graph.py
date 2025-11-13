"""
Scene Graph Generation
Extracts spatial relationships between objects using geometric reasoning and depth
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.spatial.distance import cdist


@dataclass
class SpatialRelation:
    """A spatial relationship between two entities"""
    subject_id: int
    subject_class: str
    predicate: str  # ON, UNDER, IN, NEAR, HOLDS, etc.
    object_id: int
    object_class: str
    confidence: float
    distance_3d: Optional[float] = None  # 3D distance in meters
    
    def __str__(self):
        dist_str = f" ({self.distance_3d:.2f}m)" if self.distance_3d else ""
        return f"{self.subject_class}({self.subject_id}) {self.predicate} {self.object_class}({self.object_id}){dist_str} [conf={self.confidence:.2f}]"


class SceneGraphGenerator:
    """
    Generates scene graphs from detections + depth
    Uses geometric rules and learned heuristics
    """
    
    def __init__(self, 
                 near_threshold: float = 0.5,  # meters
                 overlap_threshold: float = 0.3,
                 use_masks: bool = False):
        """
        Args:
            near_threshold: Distance threshold for NEAR relation (meters)
            overlap_threshold: IoU threshold for containment relations
            use_masks: Use instance masks if available (Detectron2)
        """
        self.near_threshold = near_threshold
        self.overlap_threshold = overlap_threshold
        self.use_masks = use_masks
        
        # Predicate definitions
        self.predicates = [
            "on", "under", "in", "near", "holds",
            "sitting_on", "standing_on", "leaning_against",
            "next_to", "behind", "in_front_of"
        ]
        
        # Common spatial configurations (subject can be ON object)
        self.can_support = {
            "table", "desk", "counter", "shelf", "floor",
            "chair", "sofa", "bed", "cabinet", "stand"
        }
        
        # Common containers
        self.containers = {
            "box", "bag", "basket", "container", "cup",
            "bowl", "backpack", "suitcase", "drawer"
        }
        
        # Person actions
        self.person_actions = {"person", "man", "woman", "child"}
    
    def generate(self, 
                 detections: Dict,
                 depth_map: np.ndarray,
                 camera_intrinsics: Optional[Dict] = None) -> List[SpatialRelation]:
        """
        Generate scene graph from detections
        
        Args:
            detections: Dict with boxes, classes, class_names, masks
            depth_map: (H, W) depth in meters
            camera_intrinsics: fx, fy, cx, cy for 3D projection
            
        Returns:
            List of SpatialRelation objects
        """
        if detections["num_detections"] == 0:
            return []
        
        boxes = detections["boxes"]
        classes = detections["class_names"]
        masks = detections.get("masks", None)
        
        relations = []
        
        # Extract 3D centroids
        centroids_3d = self._compute_3d_centroids(boxes, depth_map, camera_intrinsics)
        
        # Pairwise relation detection
        for i in range(len(boxes)):
            for j in range(len(boxes)):
                if i == j:
                    continue
                
                # Extract relation
                relation = self._detect_relation(
                    i, j,
                    boxes[i], boxes[j],
                    classes[i], classes[j],
                    centroids_3d[i], centroids_3d[j],
                    masks[i] if masks is not None else None,
                    masks[j] if masks is not None else None,
                    depth_map
                )
                
                if relation:
                    relations.append(relation)
        
        return relations
    
    def _compute_3d_centroids(self, 
                              boxes: np.ndarray,
                              depth_map: np.ndarray,
                              camera_intrinsics: Optional[Dict]) -> np.ndarray:
        """
        Compute 3D centroids from 2D boxes + depth
        
        Returns:
            (N, 3) array of [x, y, z] in meters
        """
        N = len(boxes)
        centroids = np.zeros((N, 3))
        
        # Default intrinsics (assume 1080p, 60deg FOV)
        fx = camera_intrinsics.get("fx", 1000.0) if camera_intrinsics else 1000.0
        fy = camera_intrinsics.get("fy", 1000.0) if camera_intrinsics else 1000.0
        cx = camera_intrinsics.get("cx", depth_map.shape[1] / 2) if camera_intrinsics else depth_map.shape[1] / 2
        cy = camera_intrinsics.get("cy", depth_map.shape[0] / 2) if camera_intrinsics else depth_map.shape[0] / 2
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)
            
            # Centroid in image
            cx_img = (x1 + x2) / 2
            cy_img = (y1 + y2) / 2
            
            # Sample depth at centroid
            cy_int = int(np.clip(cy_img, 0, depth_map.shape[0] - 1))
            cx_int = int(np.clip(cx_img, 0, depth_map.shape[1] - 1))
            depth = depth_map[cy_int, cx_int]
            
            # Project to 3D
            x_3d = (cx_img - cx) * depth / fx
            y_3d = (cy_img - cy) * depth / fy
            z_3d = depth
            
            centroids[i] = [x_3d, y_3d, z_3d]
        
        return centroids
    
    def _detect_relation(self,
                         idx_i: int, idx_j: int,
                         box_i: np.ndarray, box_j: np.ndarray,
                         class_i: str, class_j: str,
                         pos_3d_i: np.ndarray, pos_3d_j: np.ndarray,
                         mask_i: Optional[np.ndarray],
                         mask_j: Optional[np.ndarray],
                         depth_map: np.ndarray) -> Optional[SpatialRelation]:
        """
        Detect spatial relation between two objects
        
        Returns:
            SpatialRelation if relation detected, None otherwise
        """
        # Compute 3D distance
        distance_3d = np.linalg.norm(pos_3d_i - pos_3d_j)
        
        # NEAR relation (universal)
        if distance_3d < self.near_threshold:
            return SpatialRelation(
                subject_id=idx_i,
                subject_class=class_i,
                predicate="near",
                object_id=idx_j,
                object_class=class_j,
                confidence=1.0 - distance_3d / self.near_threshold,
                distance_3d=distance_3d
            )
        
        # ON relation (object i is ON object j)
        if class_j in self.can_support:
            # Check vertical alignment + depth ordering
            x1_i, y1_i, x2_i, y2_i = box_i
            x1_j, y1_j, x2_j, y2_j = box_j
            
            # Horizontal overlap
            x_overlap = min(x2_i, x2_j) - max(x1_i, x1_j)
            x_union = max(x2_i, x2_j) - min(x1_i, x1_j)
            
            if x_overlap > 0 and x_overlap / x_union > 0.3:
                # Check if i is above j
                if y2_i < y1_j and pos_3d_i[1] < pos_3d_j[1]:  # i above j in image and 3D
                    if abs(pos_3d_i[2] - pos_3d_j[2]) < 0.3:  # Similar depth (contact)
                        return SpatialRelation(
                            subject_id=idx_i,
                            subject_class=class_i,
                            predicate="on",
                            object_id=idx_j,
                            object_class=class_j,
                            confidence=0.8,
                            distance_3d=distance_3d
                        )
        
        # IN relation (object i is IN object j)
        if class_j in self.containers:
            # Check containment via IoU
            iou = self._compute_iou(box_i, box_j)
            if iou > self.overlap_threshold:
                return SpatialRelation(
                    subject_id=idx_i,
                    subject_class=class_i,
                    predicate="in",
                    object_id=idx_j,
                    object_class=class_j,
                    confidence=iou,
                    distance_3d=distance_3d
                )
        
        # HOLDS relation (person holding object)
        if class_i in self.person_actions:
            # Check if object j is small and near person's hands
            box_j_area = (box_j[2] - box_j[0]) * (box_j[3] - box_j[1])
            if box_j_area < 10000 and distance_3d < 0.5:  # Small object, close
                # Check if j is in front of person (similar depth or closer)
                if pos_3d_j[2] <= pos_3d_i[2] + 0.2:
                    return SpatialRelation(
                        subject_id=idx_i,
                        subject_class=class_i,
                        predicate="holds",
                        object_id=idx_j,
                        object_class=class_j,
                        confidence=0.7,
                        distance_3d=distance_3d
                    )
        
        # No relation detected
        return None
    
    def _compute_iou(self, box_a: np.ndarray, box_b: np.ndarray) -> float:
        """Compute IoU between two boxes"""
        x1_a, y1_a, x2_a, y2_a = box_a
        x1_b, y1_b, x2_b, y2_b = box_b
        
        # Intersection
        x1_i = max(x1_a, x1_b)
        y1_i = max(y1_a, y1_b)
        x2_i = min(x2_a, x2_b)
        y2_i = min(y2_a, y2_b)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area_a = (x2_a - x1_a) * (y2_a - y1_a)
        area_b = (x2_b - x1_b) * (y2_b - y1_b)
        union = area_a + area_b - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def format_scene_graph(self, relations: List[SpatialRelation]) -> str:
        """Format scene graph as human-readable text"""
        if not relations:
            return "No spatial relations detected"
        
        lines = ["Scene Graph:"]
        for rel in relations:
            lines.append(f"  • {rel}")
        
        return "\n".join(lines)
    
    def export_to_memgraph(self, relations: List[SpatialRelation]) -> List[str]:
        """
        Export scene graph as Cypher queries for Memgraph
        
        Returns:
            List of Cypher CREATE statements
        """
        queries = []
        
        for rel in relations:
            query = f"""
            MATCH (a:Entity {{id: {rel.subject_id}}}), (b:Entity {{id: {rel.object_id}}})
            CREATE (a)-[:SPATIAL_RELATION {{
                predicate: '{rel.predicate}',
                confidence: {rel.confidence},
                distance_3d: {rel.distance_3d if rel.distance_3d else 'null'}
            }}]->(b)
            """
            queries.append(query.strip())
        
        return queries
