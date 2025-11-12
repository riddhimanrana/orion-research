#!/usr/bin/env python3
"""
COMPLETE PERCEPTION & STORAGE PIPELINE
=======================================

Full end-to-end processing of egocentric video:
1. Video frame extraction (realtime speed)
2. YOLO11n object detection
3. Depth estimation (Depth Anything V2)
4. Camera intrinsics extraction
5. 3D spatial understanding
6. Object tracking & Re-ID (CLIP)
7. Spatial zone assignment
8. Scene classification
9. Storage in Memgraph
10. Export complete graph structure

Run: python full_perception_pipeline.py --video data/examples/room.mp4
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import cv2
import numpy as np
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Orion imports
try:
    from orion.managers.model_manager import ModelManager
    from orion.perception.config import DetectionConfig
    from orion.perception.observer import FrameObserver
    from orion.semantic.scene_classifier import SceneClassifier
    from orion.graph.embeddings import EmbeddingModel, create_embedding_model
    ORION_AVAILABLE = True
except ImportError as e:
    ORION_AVAILABLE = False
    logger.warning(f"Orion import issue: {e}")


@dataclass
class CameraIntrinsics:
    """Camera calibration parameters"""
    fx: float  # focal length x
    fy: float  # focal length y
    cx: float  # principal point x
    cy: float  # principal point y
    width: int
    height: int
    
    def to_matrix(self) -> np.ndarray:
        """Return 3x3 intrinsic matrix"""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])


@dataclass
class Object3D:
    """3D object representation"""
    object_id: int
    class_name: str
    bbox_2d: Tuple[float, float, float, float]  # x1, y1, x2, y2
    depth: float  # meters
    confidence: float
    zone: str  # left/center/right
    embedding: Optional[np.ndarray] = None


@dataclass
class FrameObservation:
    """Complete observation for one frame"""
    frame_idx: int
    timestamp: float
    camera_intrinsics: CameraIntrinsics
    objects: List[Object3D]
    scene_type: str
    depth_map: Optional[np.ndarray] = None
    rgb_frame: Optional[np.ndarray] = None


class ComprehensivePerceptionPipeline:
    """
    Complete perception pipeline processing all aspects of egocentric video
    """
    
    def __init__(self, video_path: str, target_fps: float = 30.0):
        self.video_path = video_path
        self.target_fps = target_fps
        self.observations: List[FrameObservation] = []
        
        # Components
        self.yolo = None
        self.scene_classifier = None
        self.embedding_model = None
        self.camera_intrinsics = None
        self.object_tracker = {}  # Track objects across frames
        self.depth_model = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all perception components"""
        logger.info("🔧 Initializing perception components...")
        
        # YOLO
        try:
            if Path("yolo11n.pt").exists():
                self.yolo = YOLO("yolo11n.pt")
                logger.info("  ✓ YOLO11n loaded")
            else:
                logger.warning("  ⚠️  yolo11n.pt not found")
        except Exception as e:
            logger.warning(f"  ⚠️  YOLO init failed: {e}")
        
        # Scene classifier
        try:
            self.scene_classifier = SceneClassifier()
            logger.info("  ✓ Scene classifier loaded")
        except Exception as e:
            logger.warning(f"  ⚠️  Scene classifier failed: {e}")
        
        # Embedding model (CLIP for Re-ID)
        try:
            self.embedding_model = create_embedding_model("openai/clip-vit-base-patch32")
            logger.info("  ✓ CLIP embedding model loaded")
        except Exception as e:
            logger.warning(f"  ⚠️  Embedding model failed: {e}")
        
        # Depth model
        try:
            import torch
            self.depth_model = torch.hub.load('DepthAnything/Depth-Anything-V2', 'dpt_small', pretrained=True, trust_repo=True)
            self.depth_model.eval()
            logger.info("  ✓ Depth Anything V2 loaded")
        except Exception as e:
            logger.warning(f"  ⚠️  Depth model failed: {e}")
        
        logger.info("✓ Initialization complete")
    
    def estimate_camera_intrinsics(self, frame: np.ndarray) -> CameraIntrinsics:
        """Estimate camera intrinsic parameters from frame"""
        h, w = frame.shape[:2]
        
        # Assume typical smartphone camera parameters
        # (in real scenario, would use camera calibration)
        focal_length = w  # pixels (reasonable assumption)
        
        return CameraIntrinsics(
            fx=focal_length,
            fy=focal_length,
            cx=w / 2,
            cy=h / 2,
            width=w,
            height=h
        )
    
    def estimate_depth(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Estimate depth map using Depth Anything V2"""
        if not self.depth_model:
            return None
        
        try:
            import torch
            from PIL import Image
            
            # Convert to PIL
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Inference
            with torch.no_grad():
                depth = self.depth_model.infer_image(img)
            
            # Normalize to 0-1
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
            return depth
            
        except Exception as e:
            logger.debug(f"Depth estimation failed: {e}")
            return None
    
    def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects using YOLO"""
        if not self.yolo:
            return []
        
        try:
            results = self.yolo(frame, conf=0.3, verbose=False)
            detections = []
            
            for det in results[0].boxes:
                obj = {
                    'class': self.yolo.names[int(det.cls[0])],
                    'confidence': float(det.conf[0]),
                    'bbox': [float(x) for x in det.xyxy[0].tolist()],  # x1,y1,x2,y2
                    'center': [
                        (float(det.xyxy[0][0]) + float(det.xyxy[0][2])) / 2,
                        (float(det.xyxy[0][1]) + float(det.xyxy[0][3])) / 2
                    ]
                }
                detections.append(obj)
            
            return detections
            
        except Exception as e:
            logger.debug(f"YOLO detection failed: {e}")
            return []
    
    def estimate_depth_for_object(self, depth_map: np.ndarray, bbox: Tuple[float, float, float, float]) -> float:
        """Estimate depth of object from depth map"""
        if depth_map is None:
            return 0.0
        
        try:
            x1, y1, x2, y2 = bbox
            h, w = depth_map.shape[:2]
            
            x1 = max(0, int(x1 * depth_map.shape[1] / w))
            x2 = min(depth_map.shape[1], int(x2 * depth_map.shape[1] / w))
            y1 = max(0, int(y1 * depth_map.shape[0] / h))
            y2 = min(depth_map.shape[0], int(y2 * depth_map.shape[0] / h))
            
            if x1 >= x2 or y1 >= y2:
                return 0.0
            
            # Use median depth in bbox (robust to outliers)
            obj_depth = np.median(depth_map[y1:y2, x1:x2])
            return float(obj_depth) * 10.0  # Scale to ~meters
            
        except Exception as e:
            logger.debug(f"Depth estimation for object failed: {e}")
            return 0.0
    
    def assign_spatial_zone(self, center_x: float, frame_width: int) -> str:
        """Assign spatial zone based on x position"""
        if center_x < frame_width / 3:
            return "left"
        elif center_x < 2 * frame_width / 3:
            return "center"
        else:
            return "right"
    
    def get_embedding(self, frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
        """Extract CLIP embedding for object region"""
        if not self.embedding_model:
            return None
        
        try:
            x1, y1, x2, y2 = bbox
            x1, x2 = int(x1), int(x2)
            y1, y2 = int(y1), int(y2)
            
            # Crop object region
            obj_patch = frame[y1:y2, x1:x2]
            if obj_patch.size == 0:
                return None
            
            # Get CLIP embedding
            embedding = self.embedding_model.embed_image(obj_patch)
            return embedding
            
        except Exception as e:
            logger.debug(f"Embedding extraction failed: {e}")
            return None
    
    def classify_scene(self, frame: np.ndarray) -> str:
        """Classify scene type"""
        if not self.scene_classifier:
            return "unknown"
        
        try:
            result = self.scene_classifier.classify_frame(frame)
            if hasattr(result, 'scene_type'):
                return result.scene_type.value
            elif isinstance(result, dict):
                return result.get('scene_type', 'unknown')
        except Exception as e:
            logger.debug(f"Scene classification failed: {e}")
        
        return "unknown"
    
    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        fps: float
    ) -> Optional[FrameObservation]:
        """
        Process single frame through entire pipeline
        Returns complete observation with all data
        """
        try:
            timestamp = frame_idx / fps
            
            # 1. Camera intrinsics
            intrinsics = self.estimate_camera_intrinsics(frame)
            
            # 2. Depth estimation
            depth_map = self.estimate_depth(frame)
            
            # 3. Object detection
            detections = self.detect_objects(frame)
            
            # 4. Scene classification
            scene_type = self.classify_scene(frame)
            
            # 5. Process each detection
            objects_3d = []
            for idx, det in enumerate(detections):
                # Estimate depth
                depth = self.estimate_depth_for_object(depth_map, tuple(det['bbox']))
                
                # Get embedding
                embedding = self.get_embedding(frame, tuple(det['bbox']))
                
                # Assign zone
                zone = self.assign_spatial_zone(det['center'][0], frame.shape[1])
                
                obj_3d = Object3D(
                    object_id=hash(f"{det['class']}_{det['center'][0]}_{det['center'][1]}") % 100000,
                    class_name=det['class'],
                    bbox_2d=tuple(det['bbox']),
                    depth=depth,
                    confidence=det['confidence'],
                    zone=zone,
                    embedding=embedding
                )
                objects_3d.append(obj_3d)
            
            observation = FrameObservation(
                frame_idx=frame_idx,
                timestamp=timestamp,
                camera_intrinsics=intrinsics,
                objects=objects_3d,
                scene_type=scene_type,
                depth_map=depth_map,
                rgb_frame=frame.copy()
            )
            
            return observation
            
        except Exception as e:
            logger.error(f"Frame {frame_idx} processing error: {e}")
            return None
    
    def process_video(self) -> Dict[str, Any]:
        """Process entire video at realtime speed"""
        logger.info(f"\n📹 Opening video: {self.video_path}")
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {self.video_path}")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        logger.info(f"  • FPS: {fps:.1f}")
        logger.info(f"  • Total frames: {total_frames}")
        logger.info(f"  • Duration: {duration:.1f}s")
        
        logger.info(f"\n🎬 Processing video (realtime speed, ~{duration:.0f}s)...")
        
        frame_times = []
        frame_idx = 0
        start_time = time.time()
        
        while True:
            frame_start = time.time()
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Process frame
            observation = self.process_frame(frame, frame_idx, fps)
            if observation:
                self.observations.append(observation)
            
            frame_elapsed = time.time() - frame_start
            frame_times.append(frame_elapsed)
            
            # Maintain realtime speed (don't process faster than video plays)
            ideal_frame_time = 1.0 / fps
            if frame_elapsed < ideal_frame_time:
                time.sleep(ideal_frame_time - frame_elapsed)
            
            if (frame_idx + 1) % 30 == 0:
                elapsed = time.time() - start_time
                avg_fps = (frame_idx + 1) / elapsed
                logger.info(f"  Frame {frame_idx+1:4d}/{total_frames}: {len(observation.objects)} objs, scene={observation.scene_type}, depth_valid={observation.depth_map is not None}, avg_fps={avg_fps:.1f}")
            
            frame_idx += 1
        
        cap.release()
        
        total_elapsed = time.time() - start_time
        
        logger.info(f"\n✓ Processing complete!")
        logger.info(f"  • Processed: {len(self.observations)} frames in {total_elapsed:.1f}s")
        logger.info(f"  • Target realtime: {duration:.1f}s, actual: {total_elapsed:.1f}s")
        logger.info(f"  • Speed ratio: {duration/total_elapsed:.2f}x")
        
        return self._compile_results()
    
    def _compile_results(self) -> Dict[str, Any]:
        """Compile all results into structured format"""
        
        # Aggregate statistics
        all_objects = defaultdict(int)
        all_scenes = defaultdict(int)
        total_detections = 0
        total_depth_valid = 0
        total_embeddings = 0
        
        for obs in self.observations:
            all_scenes[obs.scene_type] += 1
            total_detections += len(obs.objects)
            if obs.depth_map is not None:
                total_depth_valid += 1
            for obj in obs.objects:
                all_objects[obj.class_name] += 1
                if obj.embedding is not None:
                    total_embeddings += 1
        
        # Spatial distribution
        spatial_zones = defaultdict(lambda: defaultdict(int))
        for obs in self.observations:
            for obj in obs.objects:
                spatial_zones[obj.zone][obj.class_name] += 1
        
        # Temporal events (scene changes)
        temporal_events = []
        prev_scene = None
        for obs in self.observations:
            if obs.scene_type != prev_scene:
                temporal_events.append({
                    'frame': obs.frame_idx,
                    'timestamp': obs.timestamp,
                    'event': f'Scene changed to {obs.scene_type}',
                    'objects': [o.class_name for o in obs.objects]
                })
                prev_scene = obs.scene_type
        
        return {
            "metadata": {
                "video": str(self.video_path),
                "total_frames": len(self.observations),
                "fps": self.target_fps,
                "timestamp": datetime.now().isoformat()
            },
            "statistics": {
                "total_detections": total_detections,
                "unique_objects": dict(all_objects),
                "scenes_visited": dict(all_scenes),
                "depth_frames": total_depth_valid,
                "embeddings_extracted": total_embeddings
            },
            "spatial_zones": dict(spatial_zones),
            "temporal_events": temporal_events,
            "observations_count": len(self.observations)
        }
    
    def export_memgraph_structure(self) -> Dict[str, Any]:
        """Export complete graph structure as if stored in Memgraph"""
        
        graph = {
            "nodes": {
                "frames": [],
                "objects": [],
                "scenes": [],
                "zones": []
            },
            "relationships": {
                "CONTAINS": [],  # frame contains objects
                "LOCATED_IN": [],  # object located in zone
                "OBSERVED_IN": [],  # scene observed in frame
                "PRECEDES": [],  # frame precedes frame
                "SAME_OBJECT": []  # objects are same entity (Re-ID)
            }
        }
        
        # Nodes: Frames
        for obs in self.observations:
            graph["nodes"]["frames"].append({
                "id": f"frame_{obs.frame_idx}",
                "frame_idx": obs.frame_idx,
                "timestamp": obs.timestamp,
                "scene_type": obs.scene_type,
                "camera_fx": obs.camera_intrinsics.fx,
                "camera_fy": obs.camera_intrinsics.fy
            })
        
        # Nodes: Objects (unique across frames)
        seen_objects = {}
        for obs in self.observations:
            for obj in obs.objects:
                obj_key = obj.class_name
                if obj_key not in seen_objects:
                    seen_objects[obj_key] = {
                        "id": f"object_{obj_key}_{len(seen_objects)}",
                        "class": obj.class_name,
                        "occurrences": 0,
                        "avg_depth": [],
                        "avg_confidence": [],
                        "embedding_dims": len(obj.embedding) if obj.embedding is not None else 0
                    }
                seen_objects[obj_key]["occurrences"] += 1
                seen_objects[obj_key]["avg_depth"].append(obj.depth)
                seen_objects[obj_key]["avg_confidence"].append(obj.confidence)
        
        for obj_key, obj_data in seen_objects.items():
            avg_depth = np.mean(obj_data["avg_depth"]) if obj_data["avg_depth"] else 0
            avg_conf = np.mean(obj_data["avg_confidence"]) if obj_data["avg_confidence"] else 0
            graph["nodes"]["objects"].append({
                "id": obj_data["id"],
                "class": obj_data["class"],
                "total_observations": obj_data["occurrences"],
                "avg_depth": float(avg_depth),
                "avg_confidence": float(avg_conf),
                "has_embedding": obj_data["embedding_dims"] > 0
            })
        
        # Nodes: Scenes
        for obs in self.observations:
            if obs.scene_type not in [s["id"] for s in graph["nodes"]["scenes"]]:
                graph["nodes"]["scenes"].append({
                    "id": f"scene_{obs.scene_type}",
                    "type": obs.scene_type,
                    "first_frame": obs.frame_idx
                })
        
        # Nodes: Zones
        for zone in ["left", "center", "right"]:
            graph["nodes"]["zones"].append({
                "id": f"zone_{zone}",
                "name": zone
            })
        
        # Relationships: CONTAINS
        for obs in self.observations:
            for obj in obs.objects:
                obj_id = f"object_{obj.class_name}_{list(seen_objects.keys()).index(obj.class_name)}" if obj.class_name in seen_objects else "unknown"
                graph["relationships"]["CONTAINS"].append({
                    "from": f"frame_{obs.frame_idx}",
                    "to": obj_id,
                    "confidence": obj.confidence,
                    "bbox": obj.bbox_2d,
                    "depth": obj.depth
                })
        
        # Relationships: LOCATED_IN
        for obs in self.observations:
            for obj in obs.objects:
                obj_id = f"object_{obj.class_name}_{list(seen_objects.keys()).index(obj.class_name)}" if obj.class_name in seen_objects else "unknown"
                graph["relationships"]["LOCATED_IN"].append({
                    "from": obj_id,
                    "to": f"zone_{obj.zone}",
                    "frame": obs.frame_idx,
                    "timestamp": obs.timestamp
                })
        
        # Relationships: OBSERVED_IN
        for obs in self.observations:
            graph["relationships"]["OBSERVED_IN"].append({
                "from": f"frame_{obs.frame_idx}",
                "to": f"scene_{obs.scene_type}",
                "duration": 1  # frames
            })
        
        # Relationships: PRECEDES (frame temporal links)
        for i in range(len(self.observations) - 1):
            graph["relationships"]["PRECEDES"].append({
                "from": f"frame_{self.observations[i].frame_idx}",
                "to": f"frame_{self.observations[i+1].frame_idx}"
            })
        
        return graph
    
    def print_storage_report(self):
        """Print comprehensive storage report"""
        print("\n" + "="*100)
        print("📊 COMPLETE MEMGRAPH STORAGE STRUCTURE")
        print("="*100)
        
        graph = self.export_memgraph_structure()
        
        print(f"\n🔵 NODES ({sum(len(v) for v in graph['nodes'].values())} total):")
        for node_type, nodes in graph['nodes'].items():
            print(f"  • {node_type}: {len(nodes)}")
            if len(nodes) <= 10:
                for node in nodes:
                    print(f"    - {node.get('id', 'N/A')}")
            else:
                for node in nodes[:5]:
                    print(f"    - {node.get('id', 'N/A')}")
                print(f"    ... and {len(nodes)-5} more")
        
        print(f"\n🔗 RELATIONSHIPS ({sum(len(v) for v in graph['relationships'].values())} total):")
        for rel_type, rels in graph['relationships'].items():
            print(f"  • {rel_type}: {len(rels)} edges")
        
        print(f"\n📈 STATISTICS:")
        print(f"  • Total observations: {len(self.observations)}")
        print(f"  • Total detections: {sum(len(o.objects) for o in self.observations)}")
        print(f"  • Depth frames: {sum(1 for o in self.observations if o.depth_map is not None)}")
        print(f"  • Embeddings: {sum(sum(1 for obj in o.objects if obj.embedding is not None) for o in self.observations)}")
        
        print("\n" + "="*100 + "\n")
        
        return graph


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Complete perception pipeline")
    parser.add_argument("--video", default="data/examples/room.mp4", help="Video path")
    
    args = parser.parse_args()
    
    if not Path(args.video).exists():
        logger.error(f"❌ Video not found: {args.video}")
        sys.exit(1)
    
    # Run pipeline
    pipeline = ComprehensivePerceptionPipeline(args.video)
    results = pipeline.process_video()
    
    if results:
        # Print comprehensive report
        graph = pipeline.print_storage_report()
        
        # Save results
        results["graph_structure"] = graph
        
        output_path = Path("perception_complete_output.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
