"""
Fast Video Processing - Real-time optimized
============================================

Process video at real-time speed by:
- Skipping FastVLM during processing
- Saving to VideoIndex for later queries
- Maximum parallelization

Target: <66s for 66s video

Author: Orion Research Team
Date: November 2025
"""

import argparse
import cv2
import time
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orion.managers.model_manager import ModelManager
from orion.perception.depth import DepthEstimator
from orion.slam.slam_engine import OpenCVSLAM, SLAMConfig
from orion.perception.tracking import EntityTracker3D, TrackingConfig
from orion.zones.scene_classifier import SceneClassifier
from orion.zones.zone_manager import ZoneManager
from orion.query.index import VideoIndex, EntityObservation, SpatialZone


class FastVideoProcessor:
    """
    Ultra-fast video processor that skips FastVLM.
    Saves all data to VideoIndex for query-time enrichment.
    """
    
    def __init__(
        self,
        video_path: str,
        output_index: str,
        skip_frames: int = 20,
        yolo_model: str = 'yolo11s'
    ):
        self.video_path = video_path
        self.output_index = Path(output_index)
        self.skip_frames = skip_frames
        
        print("="*80)
        print("FAST VIDEO PROCESSOR (Real-time Optimized)")
        print("="*80)
        
        # Open video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open: {video_path}")
        
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.video_fps
        
        print(f"\n📹 Video: {self.frame_width}x{self.frame_height} @ {self.video_fps:.1f} FPS")
        print(f"⏱️  Duration: {self.duration:.1f}s ({self.total_frames} frames)")
        print(f"⏭️  Skip: Every {skip_frames} frames → ~{self.video_fps/skip_frames:.1f} FPS processing")
        print(f"🎯 Target: <{self.duration:.0f}s processing time\n")
        
        # Initialize models
        print("🔧 Loading models...")
        self.model_manager = ModelManager.get_instance()
        self.model_manager.yolo_model_name = yolo_model
        self.yolo_model = self.model_manager.yolo
        self.clip_model = self.model_manager.clip
        self.depth_estimator = DepthEstimator(model_name="midas", device="mps")
        print(f"  ✓ {yolo_model.upper()} + CLIP + Depth loaded")
        
        # YOLO classes
        self.yolo_classes = list(self.yolo_model.names.values()) if hasattr(self.yolo_model, 'names') else []
        
        # Initialize SLAM
        print("\n🗺️  Initializing SLAM...")
        slam_config = SLAMConfig(num_features=2000, match_ratio_test=0.75, min_matches=6)
        self.slam = OpenCVSLAM(config=slam_config)
        print("  ✓ Visual SLAM")
        
        # Initialize tracker
        print("\n👁️  Initializing tracker...")
        tracking_config = TrackingConfig(
            max_distance_pixels=150.0,
            max_distance_3d_mm=1500.0,
            ttl_frames=30,
            reid_window_frames=90,
            reid_similarity_threshold=0.85
        )
        self.tracker = EntityTracker3D(config=tracking_config, yolo_classes=self.yolo_classes)
        print("  ✓ 3D tracker with Re-ID")
        
        # Initialize scene classifier and zones
        print("\n🏠 Initializing spatial zones...")
        self.scene_classifier = SceneClassifier(clip_model=self.clip_model)
        self.zone_manager = ZoneManager(mode='dense', min_cluster_size=8, min_samples=3, merge_distance_mm=2500.0)
        print("  ✓ Scene classifier + Zone manager")
        
        # Create video index
        print(f"\n💾 Creating index: {self.output_index}")
        self.index = VideoIndex(self.output_index, Path(video_path))
        self.index.create_schema()
        print("  ✓ SQLite index created")
        
        # Stats
        self.frame_count = 0
        self.processed_count = 0
        self.start_time = None
        
    def process(self):
        """Process video and save to index"""
        print("\n" + "="*80)
        print("PROCESSING")
        print("="*80 + "\n")
        
        self.start_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            
            # Skip frames
            if self.frame_count % self.skip_frames != 0:
                continue
            
            self.processed_count += 1
            timestamp = self.frame_count / self.video_fps
            
            # Progress
            if self.processed_count % 10 == 0:
                elapsed = time.time() - self.start_time
                fps = self.processed_count / elapsed
                eta = (self.total_frames / self.skip_frames - self.processed_count) / fps
                progress = (self.frame_count / self.total_frames) * 100
                print(f"[Frame {self.frame_count:4d}/{self.total_frames}] "
                      f"Progress: {progress:5.1f}% | "
                      f"FPS: {fps:.2f} | "
                      f"ETA: {eta:.0f}s")
            
            # === YOLO Detection ===
            results = self.yolo_model(frame, conf=0.25, verbose=False)
            detections = []
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    class_name = self.yolo_classes[cls] if cls < len(self.yolo_classes) else "unknown"
                    
                    # Extract CLIP embedding
                    crop = frame[int(y1):int(y2), int(x1):int(x2)]
                    if crop.size > 0:
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        from PIL import Image
                        crop_pil = Image.fromarray(crop_rgb)
                        embedding = self.clip_model.encode_image(crop_pil).flatten().tolist()
                    else:
                        embedding = None
                    
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': conf,
                        'class': class_name,
                        'embedding': embedding
                    })
            
            # === Depth Estimation ===
            depth_map = self.depth_estimator.estimate(frame)
            
            # === SLAM ===
            slam_result = None
            if self.processed_count > 1:
                slam_result = self.slam.process_frame(frame, depth_map)
            
            camera_pose = None
            if slam_result and slam_result.success:
                t = slam_result.translation
                camera_pose = [float(t[0]), float(t[1]), float(t[2])]
            
            # === Tracking ===
            tracks = self.tracker.update(
                frame=frame,
                detections=detections,
                depth_map=depth_map,
                camera_pose=camera_pose,
                frame_number=self.frame_count
            )
            
            # === Scene Classification ===
            scene_result = self.scene_classifier.classify(frame)
            
            # === Zone Assignment ===
            for track in tracks:
                if camera_pose and track.position_3d:
                    zone_id = self.zone_manager.assign_zone(
                        entity_id=track.entity_id,
                        position_3d=track.position_3d,
                        camera_pose=camera_pose,
                        scene_type=scene_result.scene_type if scene_result else 'unknown',
                        is_indoor=scene_result.is_indoor if scene_result else True,
                        frame_number=self.frame_count
                    )
                    track.zone_id = zone_id
            
            # === Save to Index ===
            for track in tracks:
                obs = EntityObservation(
                    entity_id=track.entity_id,
                    frame_idx=self.frame_count,
                    timestamp=timestamp,
                    class_name=track.most_likely_class,
                    confidence=track.confidence,
                    bbox=track.bbox,
                    zone_id=track.zone_id if hasattr(track, 'zone_id') else None,
                    pose=camera_pose,
                    clip_embedding=track.appearance.flatten().tolist() if hasattr(track, 'appearance') and track.appearance is not None else None,
                    caption=None  # Will be filled on query
                )
                self.index.add_observation(obs)
        
        # Save zones
        for zone_id, zone in self.zone_manager.zones.items():
            zone_data = SpatialZone(
                zone_id=zone_id,
                zone_type=zone.zone_type,
                frame_indices=zone.frame_numbers,
                entity_ids=list(zone.entity_ids),
                centroid=zone.centroid.tolist() if zone.centroid is not None else None
            )
            self.index.add_zone(zone_data)
        
        self.index.commit()
        self.index.close()
        self.cap.release()
        
        # Final stats
        elapsed = time.time() - self.start_time
        fps = self.processed_count / elapsed
        
        print("\n" + "="*80)
        print("PROCESSING COMPLETE")
        print("="*80)
        print(f"⏱️  Total time: {elapsed:.1f}s (target: <{self.duration:.0f}s)")
        print(f"🎥 Avg FPS: {fps:.2f}")
        print(f"⚡ Processed: {self.processed_count}/{self.total_frames} frames")
        print(f"📊 Entities: {len(self.tracker.entity_history)}")
        print(f"🗺️  Zones: {len(self.zone_manager.zones)}")
        print(f"💾 Index: {self.output_index}")
        
        if elapsed < self.duration:
            speedup = self.duration / elapsed
            print(f"\n✅ REAL-TIME ACHIEVED! ({speedup:.2f}x faster)")
        else:
            slowdown = elapsed / self.duration
            print(f"\n⚠️  {slowdown:.2f}x slower than real-time")
        
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Fast Video Processing (Real-time)')
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--output', type=str, default=None, help='Output index path (default: video_name.db)')
    parser.add_argument('--skip', type=int, default=20, help='Frame skip (default: 20)')
    parser.add_argument('--yolo-model', type=str, default='yolo11s', 
                        choices=['yolo11n', 'yolo11s', 'yolo11m', 'yolo11x'])
    
    args = parser.parse_args()
    
    # Default output path
    if args.output is None:
        video_path = Path(args.video)
        args.output = video_path.parent / f"{video_path.stem}_index.db"
    
    processor = FastVideoProcessor(
        video_path=args.video,
        output_index=args.output,
        skip_frames=args.skip,
        yolo_model=args.yolo_model
    )
    
    processor.process()


if __name__ == '__main__':
    main()
