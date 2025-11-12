#!/usr/bin/env python3
"""
Real-Time Egocentric Spatial Understanding Pipeline
====================================================

Production-ready real-time system with:
- Adaptive FPS (1-30 FPS based on performance)
- Semantic scale recovery
- Robust temporal tracking with re-ID
- Ground plane detection
- Visual caching for static scenes
- Dynamic resource budgets
- Scene memory & room recognition

Target: 10-15 FPS on M1 Mac with full pipeline

Author: Orion Research
Date: November 11, 2025
"""

import time
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

# Core perception
from orion.perception.depth_anything import DepthAnythingV2Estimator
from orion.perception.semantic_scale import (
    SemanticScaleRecovery,
    GroundPlaneDetector,
    TemporalDepthFusion
)
from orion.perception.temporal_tracker import TemporalTracker
from orion.perception.clip_reid import CLIPReIDExtractor
from orion.perception.slam_fusion import create_slam_fusion, DepthSLAMFusion

# Adaptive processing
from orion.perception.realtime_adaptive import (
    AdaptiveFrameSelector,
    DynamicResourceBudget,
    VisualCache,
    PerformanceMetrics,
    get_performance_stats
)

# Semantic understanding
from orion.semantic.enhanced_spatial_reasoning import EnhancedSpatialReasoning
from orion.semantic.scene_graph import SceneGraphGenerator


@dataclass
class RealTimeConfig:
    """Configuration for real-time pipeline"""
    # Performance targets
    target_fps: float = 10.0
    max_frame_time_ms: float = 100.0
    
    # Adaptive features
    enable_adaptive_fps: bool = True
    enable_visual_caching: bool = True
    enable_dynamic_budgets: bool = True
    
    # Quality vs speed tradeoffs
    depth_model_size: str = 'small'  # small/base/large
    use_temporal_fusion: bool = True
    use_semantic_scale: bool = True
    use_ground_plane: bool = True
    
    # Tracking
    use_clip_reid: bool = True
    tracker_iou_threshold: float = 0.3
    tracker_max_age: int = 30
    
    # SLAM
    use_slam: bool = True
    point_cloud_stride: int = 8  # Higher = faster
    
    # Visualization
    enable_rerun: bool = True
    show_performance_overlay: bool = True
    
    # Research/Debug
    verbose: bool = True  # Detailed logging for research


class RealTimePipeline:
    """
    Real-time spatial understanding pipeline
    
    Designed for:
    - Live video streaming
    - Egocentric mobile capture
    - Interactive querying
    - LLM contextual understanding
    """
    
    def __init__(self, config: RealTimeConfig):
        """Initialize pipeline components"""
        self.config = config
        
        print("\n" + "="*70)
        print("🚀 REAL-TIME SPATIAL UNDERSTANDING PIPELINE")
        print("="*70)
        print(f"Target FPS: {config.target_fps}")
        print(f"Adaptive: {config.enable_adaptive_fps}")
        print(f"Visual Caching: {config.enable_visual_caching}")
        print("="*70)
        
        # Initialize components
        self._init_detection()
        self._init_depth()
        self._init_tracking()
        self._init_slam()
        self._init_semantic()
        self._init_adaptive()
        
        print("="*70)
        print("✅ Pipeline initialized and ready")
        print("="*70 + "\n")
        
    def _init_detection(self):
        """Initialize object detector"""
        print("\n📦 Detection:")
        from ultralytics import YOLO
        self.detector = YOLO('yolo11m.pt')
        print("  ✓ YOLO11m loaded")
        
    def _init_depth(self):
        """Initialize depth estimation"""
        print("\n🌊 Depth Estimation:")
        
        # Depth Anything V2
        self.depth_estimator = DepthAnythingV2Estimator(
            model_size=self.config.depth_model_size,
            device='mps'
        )
        print(f"  ✓ Depth Anything V2 ({self.config.depth_model_size})")
        
        # Semantic scale recovery
        if self.config.use_semantic_scale:
            self.scale_recovery = SemanticScaleRecovery(confidence_threshold=0.6)
            print("  ✓ Semantic scale recovery")
        
        # Ground plane detection
        if self.config.use_ground_plane:
            self.ground_detector = GroundPlaneDetector(camera_height_prior=1.5)
            print("  ✓ Ground plane detection")
        
        # Temporal fusion
        if self.config.use_temporal_fusion:
            self.depth_fusion = TemporalDepthFusion(window_size=5)
            print("  ✓ Temporal depth fusion (5 frames)")
        
    def _init_tracking(self):
        """Initialize object tracking"""
        print("\n🎯 Temporal Tracking:")
        
        self.tracker = TemporalTracker(
            iou_threshold=self.config.tracker_iou_threshold,
            embedding_threshold=0.7,
            max_age=self.config.tracker_max_age
        )
        
        # CLIP re-ID
        if self.config.use_clip_reid:
            self.reid_extractor = CLIPReIDExtractor(device='mps')
            print("  ✓ CLIP re-ID embeddings")
        
    def _init_slam(self):
        """Initialize SLAM"""
        if not self.config.use_slam:
            self.slam = None
            return
        
        print("\n🗺️  SLAM:")
        # Will be initialized on first frame with video dimensions
        self.slam = None
        self.slam_ready = False
        print("  ⏳ SLAM will initialize on first frame")
        
    def _init_semantic(self):
        """Initialize semantic understanding"""
        print("\n🧠 Semantic Understanding:")
        
        self.spatial_reasoner = EnhancedSpatialReasoning()
        print("  ✓ Spatial reasoning")
        
        self.scene_graph = SceneGraphGenerator()
        print("  ✓ Scene graph generation")
        
    def _init_adaptive(self):
        """Initialize adaptive processing"""
        print("\n⚡ Adaptive Processing:")
        
        if self.config.enable_adaptive_fps:
            self.frame_selector = AdaptiveFrameSelector(
                target_fps=self.config.target_fps,
                motion_threshold=0.05,
                similarity_threshold=0.95
            )
            print(f"  ✓ Adaptive frame selection (target: {self.config.target_fps} FPS)")
        
        if self.config.enable_dynamic_budgets:
            self.resource_budget = DynamicResourceBudget(
                target_frame_time_ms=self.config.max_frame_time_ms
            )
            print(f"  ✓ Dynamic resource budgets ({self.config.max_frame_time_ms}ms/frame)")
        
        if self.config.enable_visual_caching:
            self.visual_cache = VisualCache(cache_size=100)
            print("  ✓ Visual feature caching")
        
        # Performance tracking
        self.metrics = PerformanceMetrics()
        self.frame_count = 0
        
    def _log(self, message: str, level: str = "INFO"):
        """Log message if verbose enabled"""
        if self.config.verbose:
            emoji = {
                "INFO": "ℹ️",
                "DETECT": "🔍",
                "DEPTH": "📏",
                "TRACK": "🎯",
                "REID": "👤",
                "SLAM": "🗺️",
                "SCALE": "📐",
                "CACHE": "💾",
                "SKIP": "⏭️"
            }.get(level, "•")
            print(f"{emoji}  {message}")
        
    def process_frame(self,
                     frame: np.ndarray,
                     frame_idx: int,
                     force_keyframe: bool = False) -> Dict:
        """
        Process single frame
        
        Args:
            frame: RGB frame (H, W, 3)
            frame_idx: Frame index
            force_keyframe: Force full processing
        
        Returns:
            results: Dict with detections, tracks, depth, pose, etc.
        """
        frame_start = time.time()
        
        if self.config.verbose:
            print(f"\n{'='*70}")
            print(f"🎬 FRAME {frame_idx}")
            print(f"{'='*70}")
        
        # Check if we should process this frame
        if self.config.enable_adaptive_fps and not force_keyframe:
            should_process, selector_stats = self.frame_selector.should_process(frame)
            
            if not should_process:
                self._log(f"Skipping frame {frame_idx} - {selector_stats.get('reason', 'low motion/similar')}", "SKIP")
                return {
                    'skipped': True,
                    'reason': selector_stats.get('reason'),
                    'frame_idx': frame_idx
                }
        
        # Check visual cache
        if self.config.enable_visual_caching and not force_keyframe:
            cached = self.visual_cache.get(frame)
            if cached is not None:
                self._log(f"Using cached results for frame {frame_idx} - static scene detected", "CACHE")
                return {
                    'cached': True,
                    'frame_idx': frame_idx,
                    **cached
                }
        
        # Initialize SLAM on first frame
        if self.config.use_slam and not self.slam_ready:
            h, w = frame.shape[:2]
            self.slam = create_slam_fusion(
                frame_width=w,
                frame_height=h,
                depth_model=self.depth_estimator,
                fov_degrees=60.0,
                slam_type='vo',
                point_cloud_stride=self.config.point_cloud_stride
            )
            self.slam_ready = True
        
        results = {}
        elapsed_ms = 0.0
        
        # 1. DETECTION (always run)
        t0 = time.time()
        detections = self._run_detection(frame)
        detection_time = (time.time() - t0) * 1000
        elapsed_ms += detection_time
        
        if self.config.enable_dynamic_budgets:
            self.resource_budget.record_time('detection', detection_time)
        
        results['detections'] = detections
        results['n_detections'] = len(detections)
        
        # 2. DEPTH ESTIMATION (adaptive)
        run_depth = True
        if self.config.enable_dynamic_budgets:
            run_depth = self.resource_budget.should_run_component('depth', elapsed_ms)
            if not run_depth and self.config.verbose:
                self._log("Depth skipped - over time budget", "DEPTH")
        
        if run_depth:
            t0 = time.time()
            depth_map = self._run_depth(frame, detections)
            depth_time = (time.time() - t0) * 1000
            elapsed_ms += depth_time
            
            if self.config.enable_dynamic_budgets:
                self.resource_budget.record_time('depth', depth_time)
            
            results['depth_map'] = depth_map
        else:
            results['depth_map'] = None
            depth_time = 0.0
        
        # 3. TRACKING (high priority)
        t0 = time.time()
        tracks = self._run_tracking(frame, detections)
        tracking_time = (time.time() - t0) * 1000
        elapsed_ms += tracking_time
        
        if self.config.enable_dynamic_budgets:
            self.resource_budget.record_time('tracking', tracking_time)
        
        results['tracks'] = tracks
        results['n_tracks'] = len(tracks)
        
        # 4. SLAM (adaptive - skip if over budget)
        run_slam = self.config.use_slam and self.slam_ready
        if self.config.enable_dynamic_budgets and run_slam:
            run_slam = self.resource_budget.should_run_component('slam', elapsed_ms)
        
        if run_slam and results['depth_map'] is not None:
            t0 = time.time()
            slam_results = self._run_slam(frame, results['depth_map'], detections)
            slam_time = (time.time() - t0) * 1000
            elapsed_ms += slam_time
            
            if self.config.enable_dynamic_budgets:
                self.resource_budget.record_time('slam', slam_time)
            
            results['camera_pose'] = slam_results['camera_pose']
            results['point_cloud'] = slam_results['point_cloud']
        else:
            results['camera_pose'] = None
            results['point_cloud'] = None
            slam_time = 0.0
        
        # 5. SEMANTIC UNDERSTANDING (lightweight)
        scene_graph = self._run_semantic(detections, results.get('depth_map'))
        results['scene_graph'] = scene_graph
        
        # Update metrics
        frame_time = (time.time() - frame_start) * 1000
        self._update_metrics(
            frame_time, detection_time, depth_time,
            tracking_time, slam_time, len(detections)
        )
        
        results['metrics'] = self.metrics
        results['frame_idx'] = frame_idx
        
        # Cache results for static scenes
        if self.config.enable_visual_caching and self.metrics.motion_score < 0.05:
            self.visual_cache.put(frame, results)
        
        # Update adaptive systems
        if self.config.enable_adaptive_fps:
            self.frame_selector.update_performance(frame_time)
        
        if self.config.enable_dynamic_budgets:
            self.resource_budget.adjust_budget()
        
        self.frame_count += 1
        
        return results
    
    def _run_detection(self, frame: np.ndarray) -> List[Dict]:
        """Run object detection"""
        results = self.detector(frame, conf=0.3, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = self.detector.names[cls_id]
            
            detections.append({
                'bbox': xyxy,
                'score': conf,
                'class': cls_name,
                'class_id': cls_id
            })
        
        if self.config.verbose and detections:
            det_str = ', '.join([f"{d['class']} ({d['score']:.2f})" for d in detections[:5]])
            if len(detections) > 5:
                det_str += ' ...'
            self._log(f"Detected {len(detections)} objects: {det_str}", "DETECT")
        
        return detections
    
    def _run_depth(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Run depth estimation with scale recovery"""
        # Estimate depth
        depth_raw, confidence = self.depth_estimator.estimate(frame, input_size=518)
        
        if self.config.verbose:
            depth_stats = f"range: {depth_raw.min():.2f}m - {depth_raw.max():.2f}m, mean: {depth_raw.mean():.2f}m"
            self._log(f"Depth estimated - {depth_stats}", "DEPTH")
        
        # Temporal fusion
        if self.config.use_temporal_fusion:
            depth_raw = self.depth_fusion.fuse(depth_raw, confidence)
            if self.config.verbose:
                self._log(f"Applied temporal depth fusion (5 frame window)", "DEPTH")
        
        # Semantic scale recovery
        if self.config.use_semantic_scale and detections:
            h, w = frame.shape[:2]
            intrinsics = self.spatial_reasoner.estimate_camera_intrinsics(w, h)
            depth_corrected, scale_stats = self.scale_recovery.correct_depth(
                detections, depth_raw, intrinsics
            )
            
            if self.config.verbose and scale_stats.get('scale_factor'):
                scale_factor = scale_stats['scale_factor']
                n_objects = scale_stats.get('n_anchors', 0)
                self._log(f"Scale correction: {scale_factor:.3f}x using {n_objects} objects", "SCALE")
        else:
            depth_corrected = depth_raw
        
        # Ground plane detection (for future scale refinement)
        if self.config.use_ground_plane:
            ground_plane = self.ground_detector.detect(depth_corrected, downsample=4)
            if self.config.verbose and ground_plane is not None:
                self._log(f"Ground plane detected at {ground_plane.get('distance', 0):.2f}m", "DEPTH")
        
        return depth_corrected
    
    def _run_tracking(self, frame: np.ndarray, detections: List[Dict]) -> List:
        """Run temporal tracking with re-ID"""
        # Extract CLIP embeddings
        embeddings = None
        if self.config.use_clip_reid and detections:
            embeddings = self.reid_extractor.extract(frame, detections)
            if self.config.verbose:
                n_valid = sum(1 for e in embeddings if e is not None)
                self._log(f"Extracted CLIP embeddings for {n_valid}/{len(detections)} objects", "REID")
        
        # Update tracker
        tracks_before = len(self.tracker.tracks)
        tracks = self.tracker.update(detections, embeddings)
        
        if self.config.verbose and tracks:
            # Count new, confirmed, lost tracks
            confirmed = sum(1 for t in tracks if hasattr(t, 'is_confirmed') and t.is_confirmed)
            new_tracks = len(self.tracker.tracks) - tracks_before
            
            track_info = f"{len(tracks)} active ({confirmed} confirmed)"
            if new_tracks > 0:
                track_info += f", +{new_tracks} new"
            elif new_tracks < 0:
                track_info += f", {abs(new_tracks)} lost"
            
            self._log(f"Tracking: {track_info}", "TRACK")
            
            # Show individual track updates
            for track in tracks:
                if hasattr(track, 'track_id') and hasattr(track, 'is_confirmed'):
                    if track.is_confirmed and len(track.bbox_history) <= 3:
                        # Recently confirmed
                        self._log(f"  Track #{track.track_id} confirmed", "TRACK")
        
        return tracks
    
    def _run_slam(self,
                 frame: np.ndarray,
                 depth_map: np.ndarray,
                 detections: List[Dict]) -> Dict:
        """Run SLAM + depth fusion"""
        timestamp = self.frame_count / 30.0
        
        camera_pose, depth_metric, frame_points = self.slam.process_frame(
            frame, timestamp, detections
        )
        
        if self.config.verbose and camera_pose is not None:
            # Extract camera position
            pos = camera_pose[:3, 3]
            self._log(f"Camera pose: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]m", "SLAM")
            
            if frame_points is not None and len(frame_points) > 0:
                self._log(f"Point cloud: {len(frame_points)} points added", "SLAM")
        
        return {
            'camera_pose': camera_pose,
            'point_cloud': frame_points
        }
    
    def _run_semantic(self,
                     detections: List[Dict],
                     depth_map: Optional[np.ndarray]) -> List[Dict]:
        """Run semantic scene understanding"""
        # Convert list of detections to dict format expected by scene graph
        if not detections:
            return []
        
        detection_dict = {
            'num_detections': len(detections),
            'boxes': np.array([det['bbox'] for det in detections]),
            'class_names': [det['class'] for det in detections],
            'scores': np.array([det['score'] for det in detections])
        }
        
        # Generate scene graph relationships
        try:
            relationships = self.scene_graph.generate(
                detection_dict,
                depth_map=depth_map
            )
            return [vars(r) if hasattr(r, '__dict__') else r for r in relationships]
        except Exception as e:
            # Scene graph is optional - don't fail if it errors
            return []
    
    def _update_metrics(self,
                       frame_time: float,
                       detection_time: float,
                       depth_time: float,
                       tracking_time: float,
                       slam_time: float,
                       n_objects: int):
        """Update performance metrics"""
        self.metrics.frame_time_ms = frame_time
        self.metrics.fps = 1000.0 / (frame_time + 1e-6)
        self.metrics.detection_time_ms = detection_time
        self.metrics.depth_time_ms = depth_time
        self.metrics.tracking_time_ms = tracking_time
        self.metrics.slam_time_ms = slam_time
        self.metrics.n_objects = n_objects
        
        # Update motion score from frame selector
        if hasattr(self, 'frame_selector') and self.frame_selector.motion_history:
            self.metrics.motion_score = np.mean(list(self.frame_selector.motion_history))
    
    def get_stats(self) -> str:
        """Get formatted performance statistics"""
        return get_performance_stats(self.metrics)
