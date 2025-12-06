"""
Perception Engine
=================

High-level orchestration of the perception pipeline.

Orchestrates the complete Phase 1 flow:
1. Frame observation & detection (observer)
2. Visual embedding generation (embedder)
3. Entity clustering & tracking (tracker)
4. Entity description (describer)

Author: Orion Research Team
Date: October 2025
"""

import json
import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image

# Initialise logger early so try/except blocks can use it
logger = logging.getLogger(__name__)

from orion.perception.types import (
    Observation,
    PerceptionEntity,
    PerceptionResult,
    ObjectClass,
    EntityState3D,
    Perception3DResult,
    CameraIntrinsics,
    VisibilityState,
    Hand,
    HandPose,
)
from orion.perception.config import PerceptionConfig
from orion.perception.camera_intrinsics import backproject_point
from orion.perception.observer import FrameObserver
from orion.perception.embedder import VisualEmbedder
from orion.perception.tracker import EntityTracker
from orion.perception.describer import EntityDescriber
from orion.perception.depth import DepthEstimator
from orion.perception.sam_segmenter import SegmentAnythingMaskGenerator
from orion.perception.class_corrector import ClassCorrector

if TYPE_CHECKING:  # pragma: no cover - import guard for type checking only
    from orion.semantic.cis_scorer_3d import CausalInfluenceScorer3D
    from orion.semantic.types import CausalLink, StateChange

# Phase 2: Tracking imports (legacy tracker removed)
TRACKING_AVAILABLE = False

from orion.managers.model_manager import ModelManager
from orion.perception.tracker import TrackerProtocol

# Optional enhanced tracker (appearance Re-ID + 3D KF)
try:
    from orion.perception.enhanced_tracker import EnhancedTracker
    ENHANCED_TRACKER_AVAILABLE = True
except Exception:
    ENHANCED_TRACKER_AVAILABLE = False

# Optional SLAM for camera motion compensation
try:
    from orion.slam.slam_engine import SLAMEngine, SLAMConfig
    SLAM_AVAILABLE = True
except Exception:
    SLAM_AVAILABLE = False

# Optional TapNet point tracker (BootsTAPIR / TAPIR)
try:
    from orion.perception.tapnet_tracker import TapNetTracker
    TAPNET_AVAILABLE = True
except Exception:
    TAPNET_AVAILABLE = False

# Memgraph Backend
try:
    from orion.graph.memgraph_backend import MemgraphBackend
    MEMGRAPH_AVAILABLE = True
except ImportError:
    MEMGRAPH_AVAILABLE = False

from orion.perception.spatial_zones import ZoneManager
from orion.utils.profiling import profile, Profiler

class PerceptionEngine:
    """
    Complete perception pipeline orchestrator.
    
    Manages the full Phase 1 pipeline from video to described entities.
    """
    
    def __init__(
        self,
        config: Optional[PerceptionConfig] = None,
        verbose: bool = False,
    ):
        """
        Initialize perception engine.
        
        Args:
            config: Perception configuration (uses defaults if None)
            verbose: Enable verbose logging
        """
        self.config = config or PerceptionConfig()
        self.verbose = verbose
        
        if verbose:
            logging.getLogger("orion.perception").setLevel(logging.DEBUG)
        
        # Model manager (lazy loading)
        self.model_manager = ModelManager.get_instance()
        
        # Pipeline components (initialized lazily)
        self.observer: Optional[FrameObserver] = None
        self.embedder: Optional[VisualEmbedder] = None
        self.tracker: Optional[EntityTracker] = None
        self.describer: Optional[EntityDescriber] = None
        self.enhanced_tracker: Optional[TrackerProtocol] = None
        self.slam_engine: Optional[SLAMEngine] = None
        self.tapnet_tracker: Optional['TapNetTracker'] = None
        self.depth_estimator: Optional[DepthEstimator] = None
        self.sam_segmenter: Optional[SegmentAnythingMaskGenerator] = None
        self.cis_scorer: Optional['CausalInfluenceScorer3D'] = None
        self.class_corrector: Optional[ClassCorrector] = None
        self.clip_model = None
        self._components_ready: bool = False
        
        # Memgraph & Spatial Zones
        self.memgraph: Optional[MemgraphBackend] = None
        self.zone_manager: ZoneManager = ZoneManager()

        if getattr(self.config, "use_memgraph", False):
            if MEMGRAPH_AVAILABLE:
                try:
                    self.memgraph = MemgraphBackend(
                        host=self.config.memgraph_host,
                        port=self.config.memgraph_port,
                    )
                    # Create vector index sized to embedding dim for semantic search
                    self.memgraph.create_vector_index(
                        dimension=self.config.embedding.embedding_dim
                    )
                except Exception as e:
                    logger.warning(f"Failed to initialize MemgraphBackend: {e}")
                    self.memgraph = None
            else:
                logger.warning(
                    "Memgraph requested but MemgraphBackend not available (install pymgclient)."
                )

        # Phase 2: Tracking component
        self.tracker_3d = None

        if self.config.enable_tracking and ENHANCED_TRACKER_AVAILABLE:
            logger.info("  EnhancedTracker: Enabled (appearance Re-ID + 3D KF)")
        elif self.config.enable_tracking and not ENHANCED_TRACKER_AVAILABLE:
            logger.warning("  EnhancedTracker requested but import failed")

        if self.config.enable_3d and SLAM_AVAILABLE:
            logger.info("  SLAM: Will be initialized (camera motion compensation)")
        elif self.config.enable_3d and not SLAM_AVAILABLE:
            logger.warning("  SLAM requested but import failed")
        
        logger.info("PerceptionEngine initialized")
        detector_desc = (
            self.config.detection.model
            if self.config.detection.backend == "yolo"
            else self.config.detection.groundingdino_model_id
        )
        logger.info(
            "  Detection: backend=%s model=%s",
            self.config.detection.backend,
            detector_desc,
        )
        logger.info(f"  Embedding: backend={self.config.embedding.backend}, dim={self.config.embedding.embedding_dim}")
        logger.info(f"  Target FPS: {self.config.target_fps}")
    
    @profile("engine_process_video")
    def process_video(self, video_path: str, save_visualizations: bool = False, output_dir: str = "results") -> PerceptionResult:
        """
        Process video through complete perception pipeline.
        
        Args:
            video_path: Path to video file
            save_visualizations: If True, export SLAM/tracking data for visualization
            output_dir: Directory to save visualization data
            
        Returns:
            PerceptionResult with entities and observations
        """
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("\n" + "="*80)
        logger.info("PERCEPTION ENGINE - PHASE 1")
        logger.info("="*80)
        
        start_time = time.time()
        metrics_timings: dict = {}
        
        # Initialize pipeline components
        self._initialize_components()
        
        # Step 1: Observe & detect
        t0 = time.time()
        detections = self.observer.process_video(video_path)
        metrics_timings["detection_seconds"] = time.time() - t0
        
        # Retrieve SLAM engine from observer if it was used (Phase 1 3D)
        if self.observer.perception_engine and self.observer.perception_engine.slam_engine:
            self.slam_engine = self.observer.perception_engine.slam_engine
            logger.info(f"  ✓ Retrieved SLAM trajectory ({len(self.slam_engine.poses)} poses)")
        
        # Step 2: Embed detections
        t0 = time.time()
        detections = self.embedder.embed_detections(detections)
        metrics_timings["embedding_seconds"] = time.time() - t0

        # Optional TapNet point tracking (prototype: metrics only)
        if self.config.tracker_backend == "tapnet" and TAPNET_AVAILABLE and self.tapnet_tracker is not None:
            t0 = time.time()
            try:
                self._run_tapnet_tracking(detections)
            except Exception as e:
                logger.warning(f"TapNet tracking failed: {e}")
            finally:
                metrics_timings["tapnet_tracking_seconds"] = time.time() - t0

        # Optional: run enhanced 3D+appearance tracker for per-frame identities
        if self.config.enable_tracking and self.enhanced_tracker is not None:
            t0 = time.time()
            try:
                self._run_enhanced_tracking(detections)
            except Exception as e:
                logger.warning(f"Enhanced tracking failed: {e}")
            finally:
                metrics_timings["tracking_seconds"] = time.time() - t0
        
        # Step 3: Convert detections to Observation objects
        observations = self._detections_to_observations(detections)
        
        # Step 4: Cluster into entities
        t0 = time.time()
        entities = self.tracker.cluster_observations(observations)
        metrics_timings["clustering_seconds"] = time.time() - t0
        
        # Step 4.5: Phase 5 Re-ID - Semantic deduplication using embedding similarity
        t0 = time.time()
        entities = self._reid_deduplicate_entities(entities)
        metrics_timings["reid_seconds"] = time.time() - t0
        
        # Step 5: Describe entities
        t0 = time.time()
        entities = self.describer.describe_entities(entities)
        metrics_timings["description_seconds"] = time.time() - t0

        class_corrections = 0
        if getattr(self.config.class_correction, "enabled", False) and self.clip_model is not None:
            t0 = time.time()
            class_corrections = self._run_class_correction(entities)
            metrics_timings["class_correction_seconds"] = time.time() - t0
        elif getattr(self.config.class_correction, "enabled", False) and self.clip_model is None:
            logger.warning("Class correction requested but CLIP model unavailable; skipping relabeling")

        cis_links: List['CausalLink'] = []
        if getattr(self.config, "enable_cis", False):
            t0 = time.time()
            cis_links = self._run_cis_reasoning(entities)
            metrics_timings["cis_seconds"] = time.time() - t0

        if self.config.use_memgraph and self.memgraph is not None:
            try:
                self._sync_memgraph_entities(entities)
            except Exception as e:
                logger.warning(f"Memgraph sync failed: {e}")
        
        # Get video metadata (approximate from observations)
        total_frames = max([obs.frame_number for obs in observations]) if observations else 0
        fps = self.config.target_fps
        duration = max([obs.timestamp for obs in observations]) if observations else 0.0
        
        # Build result with optional tracking metrics
        result_metrics = {}
        if self.enhanced_tracker is not None:
            try:
                stats = self.enhanced_tracker.get_statistics()
                keys = [k for k in ["total_tracks", "confirmed_tracks", "active_tracks", "id_switches"] if k in stats]
                result_metrics.update({k: stats[k] for k in keys})
            except Exception:
                pass

        # Add aggregate timings and run stats
        elapsed_total = time.time() - start_time
        metrics_timings["total_seconds"] = elapsed_total
        
        # Save profiling stats
        Profiler().save_stats(Path(output_dir) / "profiling_stats.json")

        # Frame and detection stats
        try:
            sampled_frames = len({int(d["frame_number"]) for d in detections})
        except Exception:
            sampled_frames = 0
        result_metrics.update({
            "timings": metrics_timings,
            "sampled_frames": sampled_frames,
            "detections_per_sampled_frame": (len(detections) / sampled_frames) if sampled_frames else 0.0,
            "embedding_backend": getattr(self.config.embedding, "backend", "clip"),
            "embedding_dim": self.config.embedding.embedding_dim,
            "detector_backend": self.config.detection.backend,
            "detector_model": (
                self.config.detection.model
                if self.config.detection.backend == "yolo"
                else self.config.detection.groundingdino_model_id
            ),
            "tracker_backend": self.config.tracker_backend,
            "tapnet_tracks": len(self.tapnet_tracker.get_active_tracks()) if self.tapnet_tracker else 0,
            "cluster_embeddings": int(sum(1 for d in detections if 'cluster_rep_id' in d and d.get('cluster_size'))),
            "avg_cluster_size": float(np.mean([d.get('cluster_size',1) for d in detections if d.get('cluster_size')])) if any(d.get('cluster_size') for d in detections) else 1.0,
        })

        if class_corrections:
            result_metrics["class_correction"] = {
                "corrected_entities": class_corrections,
                "min_similarity": self.config.class_correction.min_similarity,
            }

        if cis_links:
            result_metrics["cis"] = {
                "link_count": len(cis_links),
                "links": [link.to_dict() for link in cis_links[: self.config.cis_max_links]],
            }

        # Inject Re-ID metrics if available
        if hasattr(self, "_reid_metrics") and isinstance(self._reid_metrics, dict):
            result_metrics["reid"] = self._reid_metrics

        result = PerceptionResult(
            entities=entities,
            raw_observations=observations,
            video_path=video_path,
            total_frames=total_frames,
            fps=fps,
            duration_seconds=duration,
            processing_time_seconds=elapsed_total,
            metrics=result_metrics or None,
        )

        elapsed = elapsed_total
        logger.info("PERCEPTION COMPLETE")
        logger.info("="*80)
        logger.info(f"  Total detections: {result.total_detections}")
        logger.info(f"  Unique entities: {result.unique_entities}")
        logger.info(f"  Processing time: {elapsed:.2f}s")
        logger.info("="*80 + "\n")
        
        # Export visualization data if requested
        if save_visualizations:
            self._export_visualization_data(result, str(output_path))

        if cis_links:
            try:
                self._export_cis_links(cis_links, output_path)
            except Exception as exc:
                logger.warning(f"Failed to export CIS links: {exc}")
        
        return result

    @profile("engine_process_image")
    def process_image(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        *,
        frame_number: int = 0,
        timestamp: float = 0.0,
        source_name: Optional[str] = None,
    ) -> PerceptionResult:
        """Run the perception pipeline on a single RGB image.

        This is primarily used for dataset-style evaluations (e.g., Action Genome)
        where inputs are already individual frames instead of full videos.
        """

        self._initialize_components()

        frame_bgr, inferred_source = self._coerce_frame(image)
        source_path = source_name or inferred_source or "<image>"

        frame_height, frame_width = frame_bgr.shape[:2]
        start_time = time.time()

        detections = self.observer.detect_objects(
            frame=frame_bgr,
            frame_number=frame_number,
            timestamp=timestamp,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        detections = self.embedder.embed_detections(detections)

        observations = self._detections_to_observations(detections)
        entities = self.tracker.cluster_observations(observations)
        metrics: Dict[str, dict] = {}
        if entities:
            entities = self._reid_deduplicate_entities(entities)
            entities = self.describer.describe_entities(entities)
            if getattr(self.config.class_correction, "enabled", False):
                if self.clip_model is None:
                    logger.warning(
                        "Class correction requested but CLIP unavailable in process_image path"
                    )
                else:
                    corrected = self._run_class_correction(entities)
                    if corrected:
                        metrics["class_correction"] = {
                            "corrected_entities": corrected,
                            "min_similarity": self.config.class_correction.min_similarity,
                        }

        elapsed = time.time() - start_time
        return PerceptionResult(
            entities=entities,
            raw_observations=observations,
            video_path=str(source_path),
            total_frames=1,
            fps=self.config.target_fps,
            duration_seconds=0.0,
            processing_time_seconds=elapsed,
            metrics=metrics or None,
        )
    
    def _initialize_components(self):
        """Initialize all pipeline components with models."""
        if self._components_ready:
            return
        logger.info("Loading models...")
        
        # Detector backend (YOLO or GroundingDINO)
        yolo = None
        grounding_dino = None
        if self.config.detection.backend == "groundingdino":
            self.model_manager.groundingdino_model_id = self.config.detection.groundingdino_model_id
            grounding_dino = self.model_manager.groundingdino
            logger.info(
                "  ✓ GroundingDINO loaded (%s)",
                self.config.detection.groundingdino_model_id,
            )
        else:
            self.model_manager.yolo_model_name = self.config.detection.model
            yolo = self.model_manager.yolo
            logger.info("  ✓ YOLO detector loaded (%s)", self.config.detection.model)
        
        # CLIP (only load if requested by backend or text conditioning)
        clip = None
        try:
            should_load_clip = (
                self.config.embedding.backend == "clip"
                or self.config.embedding.use_text_conditioning
                or getattr(self.config.class_correction, "enabled", False)
            )
            if should_load_clip:
                clip = self.model_manager.clip
                logger.info("  ✓ CLIP loaded (semantic + description)")
                self.clip_model = clip
            else:
                logger.info("  ✓ Skipping CLIP load (backend does not require it)")
                self.clip_model = None
        except Exception as e:
            logger.warning(f"  ✗ CLIP load failed: {e}. Proceeding without CLIP.")
            self.clip_model = None

        # DINO (instance-level embeddings) if requested
        dino = None
        if self.config.embedding.backend == "dino":
            try:
                dino = self.model_manager.dino
                logger.info("  ✓ DINO loaded (instance-level embeddings)")
            except Exception as e:
                logger.warning(f"  ✗ Failed to load DINO backend: {e}. Falling back to CLIP embeddings.")
                self.config.embedding.backend = "clip"
        
        # FastVLM
        vlm = self.model_manager.fastvlm
        logger.info("  ✓ FastVLM loaded")

        segmentation_cfg = getattr(self.config, "segmentation", None)
        if segmentation_cfg and segmentation_cfg.enabled:
            try:
                if segmentation_cfg.checkpoint_path:
                    self.model_manager.sam_checkpoint_path = Path(segmentation_cfg.checkpoint_path)
                self.model_manager.sam_model_type = segmentation_cfg.model_type
                self.model_manager.sam_device_override = (
                    None if segmentation_cfg.device == "auto" else segmentation_cfg.device
                )
                sam_predictor = self.model_manager.sam_predictor
                self.sam_segmenter = SegmentAnythingMaskGenerator(
                    predictor=sam_predictor,
                    mask_threshold=segmentation_cfg.mask_threshold,
                    stability_score_threshold=segmentation_cfg.stability_score_threshold,
                    min_mask_area=segmentation_cfg.min_mask_area,
                    batch_size=segmentation_cfg.batch_size,
                    refine_bounding_box=segmentation_cfg.refine_bounding_box,
                )
                logger.info("  ✓ SAM segmentation enabled (%s)", segmentation_cfg.model_type)
            except Exception as exc:
                logger.warning(f"  ✗ Failed to initialize SAM: {exc}")
                self.sam_segmenter = None
                self.model_manager.sam_device_override = None
        else:
            self.sam_segmenter = None
            self.model_manager.sam_device_override = None
        
        # Create components
        self.observer = FrameObserver(
            config=self.config.detection,
            detector_backend=self.config.detection.backend,
            yolo_model=yolo,
            grounding_dino=grounding_dino,
            target_fps=self.config.target_fps,
            show_progress=True,
            enable_3d=self.config.enable_3d,
            depth_model=self.config.depth_model,
            enable_occlusion=self.config.enable_occlusion,
            segmentation_config=self.config.segmentation,
            segmentation_refiner=self.sam_segmenter,
        )
        
        self.embedder = VisualEmbedder(
            clip_model=clip,
            dino_model=dino,
            config=self.config.embedding,
        )
        
        self.tracker = EntityTracker(
            config=self.config,
        )
        
        self.describer = EntityDescriber(
            vlm_model=vlm,
            config=self.config.description,
        )

        self._components_ready = True
        
        if self.config.enable_depth:
            self._get_depth_estimator()


        
        # Enhanced tracker (StrongSORT-inspired)
        if self.config.enable_tracking and ENHANCED_TRACKER_AVAILABLE:
            try:
                self.enhanced_tracker = EnhancedTracker(
                    clip_model=self.model_manager.clip  # Pass CLIP for label verification
                )
                logger.info("  ✓ EnhancedTracker initialized (with CLIP verification)")
            except Exception as e:
                logger.warning(f"  ✗ EnhancedTracker failed to initialize: {e}")
        
        # SLAM engine for camera motion compensation (CMC)
        if self.config.enable_3d and SLAM_AVAILABLE:
            try:
                self.slam_engine = SLAMEngine(SLAMConfig())
                logger.info("  ✓ SLAMEngine initialized (camera pose estimation)")
            except Exception as e:
                logger.warning(f"  ✗ SLAMEngine failed to initialize: {e}")
        
        logger.info("✓ All components initialized\n")

        # Initialize TapNet tracker (after other components)
        if self.config.tracker_backend == "tapnet":
            if not TAPNET_AVAILABLE:
                logger.warning("TapNet backend requested but module unavailable.")
            else:
                try:
                    self.tapnet_tracker = TapNetTracker(
                        checkpoint_path=self.config.tapnet_checkpoint_path or "",
                        max_points=self.config.tapnet_max_points,
                        resolution=self.config.tapnet_resolution,
                        device=self.config.embedding.device,
                        online_mode=self.config.tapnet_online_mode,
                        min_track_length=self.config.tapnet_min_track_length,
                    )
                    logger.info("  ✓ TapNetTracker initialized (point trajectories)")
                except Exception as e:
                    logger.warning(f"  ✗ TapNetTracker failed to initialize: {e}")

    def _run_tapnet_tracking(self, detections: List[dict]) -> None:
        """Prototype TapNet tracking over aggregated detections.

        Current stub logic:
          - Seed tracks from first frame centroids
          - Advance identity motion per frame (no model inference yet)
          - Metrics only (does not modify entity clustering path yet)
        """
        if self.tapnet_tracker is None:
            return
        # Group detections by frame
        by_frame: dict[int, List[dict]] = {}
        for det in detections:
            by_frame.setdefault(int(det.get("frame_number", 0)), []).append(det)
        if not by_frame:
            return
        # Initialize with first frame detections
        first_frame = min(by_frame.keys())
        self.tapnet_tracker.initialize(by_frame[first_frame], frame=np.zeros((self.tapnet_tracker.resolution, self.tapnet_tracker.resolution, 3), dtype=np.uint8))
        # Advance for remaining frames
        for fidx in sorted(k for k in by_frame.keys() if k != first_frame):
            self.tapnet_tracker.update(fidx, frame=np.zeros((self.tapnet_tracker.resolution, self.tapnet_tracker.resolution, 3), dtype=np.uint8))
    
    def _export_visualization_data(self, result: PerceptionResult, output_dir: str):
        """Export SLAM trajectory, camera intrinsics, and tracking data for visualization."""
        import json
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting visualization data to {output_dir}/...")
        
        # 1. Export SLAM trajectory if available
        if self.slam_engine is not None and len(self.slam_engine.poses) > 0:
            trajectory_file = output_path / "slam_trajectory.npy"
            np.save(trajectory_file, np.array(self.slam_engine.poses))
            logger.info(f"  ✓ Saved SLAM trajectory: {trajectory_file}")
        else:
            logger.warning("  ✗ No SLAM trajectory to export (SLAM not active or no poses)")
        
        # 2. Export camera intrinsics (preset or config derived)
        intrinsics_file = output_path / "camera_intrinsics.json"
        camera_intrinsics = self.config.camera.resolve_intrinsics(
            width=self.config.camera.width,
            height=self.config.camera.height,
        )
        intrinsics_payload = {
            "fx": camera_intrinsics.fx,
            "fy": camera_intrinsics.fy,
            "cx": camera_intrinsics.cx,
            "cy": camera_intrinsics.cy,
            "width": camera_intrinsics.width,
            "height": camera_intrinsics.height,
            "preset": self.config.camera.intrinsics_preset,
            "note": "Values derived from selected intrinsics preset or overrides",
        }
        with open(intrinsics_file, "w") as f:
            json.dump(intrinsics_payload, f, indent=2)
        logger.info(f"  ✓ Saved camera intrinsics: {intrinsics_file}")
        
        # 3. Export tracking data (entities with re-tracking events)
        entities_file = output_path / "entities.json"
        entities_data = {
            "total_entities": len(result.entities),
            "entities": []
        }
        
        for i, entity in enumerate(result.entities):
            best_obs = entity.get_best_observation()
            entity_dict = {
                "id": i,
                "class": entity.display_class(),
                "confidence": float(best_obs.confidence),
                "observation_count": len(entity.observations),
                "first_frame": entity.first_seen_frame,
                "last_frame": entity.last_seen_frame,
                "description": entity.description or "No description",
            }
            entities_data["entities"].append(entity_dict)
        
        # Add tracking events if EnhancedTracker is available
        if self.enhanced_tracker is not None:
            try:
                stats = self.enhanced_tracker.get_statistics()
                entities_data["tracking_stats"] = {
                    "total_tracks": stats.get("total_tracks", 0),
                    "confirmed_tracks": stats.get("confirmed_tracks", 0),
                    "id_switches": stats.get("id_switches", 0),
                }
            except Exception:
                pass
        
        with open(entities_file, 'w') as f:
            json.dump(entities_data, f, indent=2)
        logger.info(f"  ✓ Saved tracking data: {entities_file}")
        
        logger.info("✓ Visualization data export complete\n")

    def _export_cis_links(self, links: List['CausalLink'], output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / "cis_links.json"
        payload = [link.to_dict() for link in links]
        with out_file.open("w") as fp:
            json.dump(payload, fp, indent=2)
        logger.info("  ✓ Saved %d CIS links → %s", len(links), out_file)
    
    def process_frame(
        self,
        frame: np.ndarray,
        detections: List[dict],
        frame_number: int = 0,
        timestamp: float = 0.0,
    ) -> Perception3DResult:
        """Lightweight per-frame processing used in unit tests and debugging.

        This path bypasses the heavy observer/embedder stack and instead consumes
        caller-provided detections (typically synthetic). It still produces
        structured 3D metadata so downstream tests can exercise depth/occlusion
        logic without requiring full model execution.
        """
        start = time.time()
        height, width = frame.shape[:2]
        intrinsics = self._resolve_intrinsics(width, height)

        depth_value = None
        depth_map: Optional[np.ndarray] = None
        if self.config.enable_depth:
            estimator = self._get_depth_estimator()
            if estimator is not None:
                try:
                    depth_map, _ = estimator.estimate(frame)
                    if depth_map is not None:
                        depth_map = depth_map.astype(np.float32)
                        depth_map = np.clip(depth_map, 0.0, self.config.depth.max_depth_mm)
                        valid = depth_map[(depth_map > 0) & np.isfinite(depth_map)]
                        if valid.size > 0:
                            depth_value = float(np.median(valid))
                        else:
                            depth_value = None
                except Exception as exc:
                    logger.warning(f"Depth estimation failed for frame {frame_number}: {exc}")
            if depth_map is None:
                depth_value = min(self.config.depth.max_depth_mm, 1500.0)
                depth_map = np.full((height, width), depth_value, dtype=np.float32)

        entities: List[EntityState3D] = []
        for idx, det in enumerate(detections):
            bbox = det.get("bbox") or det.get("bbox_2d_px")
            if bbox is None:
                continue
            x1, y1, x2, y2 = map(float, bbox)
            centroid_u = (x1 + x2) / 2.0
            centroid_v = (y1 + y2) / 2.0
            centroid_2d = (centroid_u, centroid_v)

            centroid_3d = None
            bbox_3d = None
            depth_mean_mm = None
            depth_variance = None
            if depth_value is not None:
                centroid_3d = backproject_point(centroid_u, centroid_v, depth_value, intrinsics)
                bbox_3d = (
                    backproject_point(x1, y1, depth_value, intrinsics),
                    backproject_point(x2, y2, depth_value, intrinsics),
                )
                depth_mean_mm = depth_value
                depth_variance = 25.0  # Stub variance for tests

            entities.append(
                EntityState3D(
                    entity_id=str(det.get("entity_id", f"entity_{idx}")),
                    frame_number=frame_number,
                    timestamp=timestamp,
                    class_label=str(det.get("class", det.get("label", "unknown"))),
                    class_confidence=float(det.get("confidence", 0.0)),
                    bbox_2d_px=(int(x1), int(y1), int(x2), int(y2)),
                    centroid_2d_px=centroid_2d,
                    centroid_3d_mm=centroid_3d,
                    bbox_3d_mm=bbox_3d,
                    depth_mean_mm=depth_mean_mm,
                    depth_variance_mm2=depth_variance,
                    visibility_state=VisibilityState.FULLY_VISIBLE,
                    occlusion_ratio=0.0,
                    occluded_by=None,
                    metadata={"source": det.get("source", "mock")},
                )
            )

        hands: List[Hand] = []
        if self.config.enable_hands:
            palm_depth = depth_value or 800.0
            landmarks_2d = [(0.5, 0.5)] * 21
            landmarks_3d = [(0.0, 0.0, palm_depth)] * 21
            hands.append(
                Hand(
                    id="hand_stub",
                    landmarks_2d=landmarks_2d,
                    landmarks_3d=landmarks_3d,
                    palm_center_3d=(0.0, 0.0, palm_depth),
                    pose=HandPose.UNKNOWN,
                    confidence=0.0,
                    handedness="Unknown",
                )
            )

        processing_ms = (time.time() - start) * 1000.0
        return Perception3DResult(
            frame_number=frame_number,
            timestamp=timestamp,
            entities=entities,
            hands=hands,
            depth_map=depth_map,
            camera_intrinsics=intrinsics,
            processing_time_ms=processing_ms,
            metadata={
                "detections_processed": len(entities),
                "depth_enabled": self.config.enable_depth,
                "hands_enabled": self.config.enable_hands,
            },
        )

    def _resolve_intrinsics(self, width: int, height: int) -> CameraIntrinsics:
        """Resolve camera intrinsics using the runtime camera config."""
        return self.config.camera.resolve_intrinsics(width=width, height=height)

    def _run_class_correction(self, entities: List[PerceptionEntity]) -> int:
        if not entities:
            return 0
        corrector = self._ensure_class_corrector()
        if corrector is None:
            return 0
        return corrector.apply(entities)

    def _ensure_class_corrector(self) -> Optional[ClassCorrector]:
        if self.class_corrector is not None:
            return self.class_corrector
        if self.clip_model is None:
            return None
        detector_vocab = []
        if self.observer is not None:
            detector_vocab = getattr(self.observer, "detector_classes", [])
        self.class_corrector = ClassCorrector(
            clip_model=self.clip_model,
            config=self.config.class_correction,
            detector_vocabulary=detector_vocab,
        )
        return self.class_corrector

    def _run_cis_reasoning(self, entities: List[PerceptionEntity]) -> List['CausalLink']:
        if not entities:
            return []
        state_changes, embeddings = self._build_state_changes_from_entities(entities)
        if not state_changes:
            return []
        scorer = self._ensure_cis_scorer()
        try:
            return scorer.compute_causal_links(state_changes, embeddings)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"CIS scorer failed: {exc}")
            return []

    def _ensure_cis_scorer(self):
        if self.cis_scorer is not None:
            return self.cis_scorer
        from orion.semantic.cis_scorer_3d import CausalInfluenceScorer3D

        semantic_config = getattr(self.config, "semantic", None)
        self.cis_scorer = CausalInfluenceScorer3D(
            semantic_config.causal if semantic_config else None
        )
        return self.cis_scorer

    def _build_state_changes_from_entities(
        self,
        entities: List[PerceptionEntity],
    ) -> Tuple[List['StateChange'], Dict[str, np.ndarray]]:
        from orion.semantic.types import StateChange

        state_changes: List[StateChange] = []
        embeddings: Dict[str, np.ndarray] = {}

        for entity in entities:
            if not entity.observations:
                continue
            observations = sorted(entity.observations, key=lambda obs: obs.timestamp)
            first = observations[0]
            last = observations[-1]
            dt = max(1e-3, float(last.timestamp - first.timestamp))

            centroid_before = (float(first.centroid[0]), float(first.centroid[1]))
            centroid_after = (float(last.centroid[0]), float(last.centroid[1]))
            centroid_3d_before = tuple(first.centroid_3d_mm) if first.centroid_3d_mm is not None else None
            centroid_3d_after = tuple(last.centroid_3d_mm) if last.centroid_3d_mm is not None else None

            if centroid_3d_before and centroid_3d_after:
                velocity_3d = tuple((centroid_3d_after[i] - centroid_3d_before[i]) / dt for i in range(3))
            else:
                velocity_3d = None
            velocity_2d = (
                (centroid_after[0] - centroid_before[0]) / dt,
                (centroid_after[1] - centroid_before[1]) / dt,
            )

            state_change = StateChange(
                entity_id=entity.entity_id,
                class_label=entity.object_class.value if hasattr(entity.object_class, 'value') else str(entity.object_class),
                frame_before=first.frame_number,
                frame_after=last.frame_number,
                timestamp_before=float(first.timestamp),
                timestamp_after=float(last.timestamp),
                centroid_before=centroid_before,
                centroid_after=centroid_after,
                centroid_3d_before=centroid_3d_before,
                centroid_3d_after=centroid_3d_after,
                velocity_3d=velocity_3d,
                velocity_2d=velocity_2d,
                bounding_box_before=first.bounding_box.to_list(),
                bounding_box_after=last.bounding_box.to_list(),
                change_magnitude=float(abs(last.confidence - first.confidence)),
                metadata={"observation_count": len(entity.observations)},
            )
            state_changes.append(state_change)

            embedding = entity.average_embedding
            if embedding is None:
                try:
                    embedding = entity.compute_average_embedding()
                except Exception:
                    embedding = None
            if embedding is not None:
                embeddings[entity.entity_id] = embedding

        return state_changes, embeddings

    def _get_depth_estimator(self) -> Optional[DepthEstimator]:
        """Lazily initialize and return the shared depth estimator."""
        if not self.config.enable_depth:
            return None
        if self.depth_estimator is not None:
            return self.depth_estimator

        depth_cfg = getattr(self.config, "depth", None)
        model_name = getattr(depth_cfg, "model_name", "depth_anything_v3")
        model_size = getattr(depth_cfg, "model_size", "small")
        device = getattr(depth_cfg, "device", None)
        half_precision = getattr(depth_cfg, "half_precision", True)

        try:
            self.depth_estimator = DepthEstimator(
                model_name=model_name,
                model_size=model_size,
                device=device,
                half_precision=half_precision,
            )
            logger.info(f"  ✓ DepthEstimator initialized ({model_name}/{model_size})")
        except Exception as exc:
            logger.warning(f"  ✗ Failed to initialize DepthEstimator: {exc}")
            self.depth_estimator = None
        return self.depth_estimator
    
    def _sync_memgraph_entities(self, entities: List[PerceptionEntity]):
        """Push entity observations and relations to Memgraph."""
        if not self.memgraph:
            return

        frame_entries: Dict[int, List[Tuple[int, Observation]]] = {}

        for entity in entities:
            try:
                entity_numeric_id = self._entity_str_to_int(entity.entity_id)
            except Exception:
                logger.debug(f"Unable to parse entity_id {entity.entity_id} for Memgraph")
                continue

            zone_id = self.zone_manager.assign_zone(entity)

            embedding = entity.average_embedding
            if embedding is None:
                try:
                    embedding = entity.compute_average_embedding()
                except Exception:
                    embedding = None

            embedding_payload = (
                embedding.astype(float).tolist()
                if isinstance(embedding, np.ndarray)
                else (embedding.tolist() if hasattr(embedding, "tolist") else None)
            )

            class_name = (
                entity.object_class.value
                if hasattr(entity.object_class, "value")
                else str(entity.object_class)
            )
            caption_payload = entity.description
            caption_written = False
            embedding_written = False

            for obs in entity.observations:
                bbox = obs.bounding_box.to_list()
                try:
                    self.memgraph.add_entity_observation(
                        entity_id=entity_numeric_id,
                        frame_idx=obs.frame_number,
                        timestamp=obs.timestamp,
                        bbox=bbox,
                        class_name=class_name,
                        confidence=float(obs.confidence),
                        zone_id=zone_id,
                        caption=(
                            caption_payload
                            if caption_payload and not caption_written
                            else None
                        ),
                        embedding=embedding_payload if not embedding_written else None,
                    )
                    embedding_written = True if embedding_payload and not embedding_written else embedding_written
                    caption_written = True if caption_payload and not caption_written else caption_written
                    frame_entries.setdefault(obs.frame_number, []).append((entity_numeric_id, obs))
                except Exception as exc:
                    logger.debug(
                        f"Memgraph observation write failed for {entity.entity_id}: {exc}"
                    )

        self._emit_spatial_relationships(frame_entries)

    def _emit_spatial_relationships(
        self, frame_entries: Dict[int, List[Tuple[int, Observation]]]
    ):
        """Derive simple NEAR relationships per frame and send to Memgraph."""
        if not self.memgraph:
            return

        near_threshold = getattr(self.config, "memgraph_near_threshold", 0.15)

        for frame_idx, entries in frame_entries.items():
            if len(entries) < 2:
                continue
            for i in range(len(entries)):
                entity_a, obs_a = entries[i]
                center_a = obs_a.bounding_box.center
                for j in range(i + 1, len(entries)):
                    entity_b, obs_b = entries[j]
                    center_b = obs_b.bounding_box.center
                    dist = math.dist(center_a, center_b)
                    diag = self._estimate_frame_diag(obs_a, obs_b)
                    if diag <= 0:
                        continue
                    norm_dist = dist / diag
                    if norm_dist <= near_threshold:
                        confidence = max(
                            0.1,
                            1.0 - (norm_dist / max(near_threshold, 1e-3))
                        )
                        try:
                            self.memgraph.add_spatial_relationship(
                                entity1_id=entity_a,
                                entity2_id=entity_b,
                                relationship_type="NEAR",
                                confidence=float(confidence),
                                frame_idx=frame_idx,
                            )
                        except Exception as exc:
                            logger.debug(f"Memgraph relation write failed: {exc}")

    @staticmethod
    def _estimate_frame_diag(obs_a: Observation, obs_b: Observation) -> float:
        """Best-effort estimate of frame diagonal for normalization."""
        for obs in (obs_a, obs_b):
            if obs.frame_width and obs.frame_height:
                return math.hypot(obs.frame_width, obs.frame_height)
        width = max(obs_a.bounding_box.width, obs_b.bounding_box.width)
        height = max(obs_a.bounding_box.height, obs_b.bounding_box.height)
        return math.hypot(width, height) * 3.0

    @staticmethod
    def _coerce_frame(
        image: Union[str, Path, np.ndarray, Image.Image]
    ) -> Tuple[np.ndarray, Optional[str]]:
        """Normalize supported image inputs into an OpenCV-compatible frame."""

        inferred_source: Optional[str] = None

        if isinstance(image, (str, Path)):
            inferred_source = str(image)
            frame = cv2.imread(inferred_source)
            if frame is None:
                raise RuntimeError(f"Failed to read image: {inferred_source}")
            return frame, inferred_source

        if isinstance(image, Image.Image):
            inferred_source = getattr(image, "filename", None)
            rgb = image.convert("RGB")
            frame = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
            return np.ascontiguousarray(frame), inferred_source

        if isinstance(image, np.ndarray):
            frame = image
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.ndim == 3:
                if frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                elif frame.shape[2] != 3:
                    raise ValueError(
                        "process_image expects HxWx(3 or 4) channel arrays"
                    )
            else:
                raise ValueError(
                    "process_image expects 2D (grayscale) or 3D image arrays"
                )

            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)

            return np.ascontiguousarray(frame), inferred_source

        raise TypeError(
            "Unsupported image type for process_image: "
            f"{type(image).__name__}"
        )

    @staticmethod
    def _entity_str_to_int(entity_id: str) -> int:
        """Convert entity_id strings like 'entity_12' into stable ints."""
        if entity_id.isdigit():
            return int(entity_id)
        digits = "".join(ch for ch in entity_id if ch.isdigit())
        if digits:
            return int(digits)
        return abs(hash(entity_id)) % (2 ** 31)

    def _reid_deduplicate_entities(self, entities: List[PerceptionEntity]) -> List[PerceptionEntity]:
        """Phase 5: Adaptive Re-ID semantic deduplication.

        Dynamically derives per-class similarity thresholds, applies spatial distance
        veto, temporal overlap heuristics, and low-observation rescue merges, then
        downsamples embeddings for memory efficiency. Metrics stored in `self._reid_metrics`.
        Verbose pairwise printing gated by `EmbeddingConfig.reid_debug`.
        """
        if len(entities) <= 1:
            return entities
        
        logger.info("="*80)
        logger.info("PHASE 5: RE-ID SEMANTIC DEDUPLICATION")
        logger.info("="*80)
        logger.info(f"Input entities: {len(entities)}")
        
        # Extract average embeddings (compute if missing)
        embeddings: List[np.ndarray] = []
        valid_entities: List[PerceptionEntity] = []
        for entity in entities:
            if entity.average_embedding is None:
                try:
                    entity.compute_average_embedding()
                except Exception:
                    logger.warning(f"Entity {entity.entity_id} missing average embedding and could not compute.")
                    continue
            embeddings.append(entity.average_embedding)
            valid_entities.append(entity)

        if len(valid_entities) < 2:
            logger.warning(f"Not enough entities with embeddings for Re-ID ({len(valid_entities)} entities)")
            return entities

        emb_arr = np.vstack(embeddings)
        norms = np.linalg.norm(emb_arr, axis=1, keepdims=True) + 1e-8
        emb_norm = emb_arr / norms
        similarity_matrix = emb_norm @ emb_norm.T

        # Group indices per class
        class_groups: Dict[str, List[int]] = {}
        for i, ent in enumerate(valid_entities):
            cls = ent.object_class.value if hasattr(ent.object_class, 'value') else str(ent.object_class)
            class_groups.setdefault(cls, []).append(i)

        # Derive dynamic thresholds & collect stats
        backend = getattr(self.config.embedding, 'backend', 'clip')
        base_threshold = 0.55 if backend == 'dinov3' else 0.65
        class_thresholds: Dict[str, float] = {}
        similarity_stats: Dict[str, Dict[str, float]] = {}
        for cls, idxs in class_groups.items():
            if len(idxs) < 2:
                class_thresholds[cls] = base_threshold
                continue
            sims: List[float] = []
            for a_i in range(len(idxs)):
                for b_i in range(a_i+1, len(idxs)):
                    sims.append(float(similarity_matrix[idxs[a_i], idxs[b_i]]))
            if sims:
                sims_arr = np.array(sims)
                median = float(np.median(sims_arr))
                mean = float(np.mean(sims_arr))
                std = float(np.std(sims_arr))
                p95 = float(np.percentile(sims_arr, 95))
                # Dynamic threshold heuristic
                dyn = max(base_threshold, min(0.90, median - 0.05 if median > base_threshold else median + 0.05))
                class_thresholds[cls] = dyn
                similarity_stats[cls] = {"mean": mean, "std": std, "median": median, "p95": p95, "threshold": dyn, "pair_count": float(len(sims_arr))}
            else:
                class_thresholds[cls] = base_threshold

        if getattr(self.config.embedding, 'reid_debug', False):
            print("\n" + "="*80)
            print("PHASE 5: SIMILARITY ANALYSIS (Adaptive)")
            print("="*80)
            for cls, stats in similarity_stats.items():
                print(f"Class {cls}: median={stats['median']:.3f} mean={stats['mean']:.3f} std={stats['std']:.3f} p95={stats['p95']:.3f} -> threshold={stats['threshold']:.3f}")

        # Prepare centroid cache
        def entity_centroid(ent: PerceptionEntity) -> Tuple[float, float]:
            xs, ys = [], []
            for obs in ent.observations:
                xs.append(obs.centroid[0])
                ys.append(obs.centroid[1])
            return (float(np.mean(xs)), float(np.mean(ys)))

        centroids = [entity_centroid(e) for e in valid_entities]
        # Frame size heuristic from first observation if available
        fw = valid_entities[0].observations[0].frame_width or 1.0
        fh = valid_entities[0].observations[0].frame_height or 1.0
        diag = (fw**2 + fh**2) ** 0.5 if fw > 0 and fh > 0 else 1.0

        merged_flags = [False] * len(valid_entities)
        merged_entities: List[PerceptionEntity] = []
        merges_count = 0
        per_class_merge_counts: Dict[str, int] = {cls:0 for cls in class_groups}
        temporal_overlap_base = 0.50

        max_embeddings_per_entity = getattr(self.config.embedding, 'max_embeddings_per_entity', None)

        for i in range(len(valid_entities)):
            if merged_flags[i]:
                continue
            ent_i = valid_entities[i]
            cls_i = ent_i.object_class.value if hasattr(ent_i.object_class, 'value') else str(ent_i.object_class)
            threshold_i = class_thresholds.get(cls_i, base_threshold)
            similar_indices = [i]
            # Deterministic sampling seed for reproducible downsampling behavior
            np.random.seed(42)
            for j in range(i+1, len(valid_entities)):
                if merged_flags[j]:
                    continue
                ent_j = valid_entities[j]
                if ent_i.object_class != ent_j.object_class:
                    continue
                sim = float(similarity_matrix[i, j])
                # Temporal overlap
                f_i_min, f_i_max = ent_i.first_seen_frame, ent_i.last_seen_frame
                f_j_min, f_j_max = ent_j.first_seen_frame, ent_j.last_seen_frame
                overlap_start = max(f_i_min, f_j_min)
                overlap_end = min(f_i_max, f_j_max)
                overlap_frames = max(0, overlap_end - overlap_start)
                dur_i = f_i_max - f_i_min + 1
                dur_j = f_j_max - f_j_min + 1
                overlap_ratio = overlap_frames / min(dur_i, dur_j) if min(dur_i, dur_j) > 0 else 0.0
                # Spatial distance veto (normalized)
                dx = centroids[i][0] - centroids[j][0]
                dy = centroids[i][1] - centroids[j][1]
                spatial_dist_norm = ((dx*dx + dy*dy) ** 0.5) / diag
                spatial_veto = spatial_dist_norm > 0.60 and sim < (threshold_i + 0.05)
                # Low observation merge heuristic
                low_obs_merge = (len(ent_i.observations) <= 3 or len(ent_j.observations) <= 3) and sim >= (threshold_i * 0.4)
                # Merge conditions
                should_merge = (
                    (sim >= threshold_i and not spatial_veto) or
                    (sim >= threshold_i * 0.85 and overlap_ratio >= temporal_overlap_base) or
                    (sim >= threshold_i * 0.5 and overlap_ratio >= 0.85) or
                    low_obs_merge
                )
                if should_merge:
                    similar_indices.append(j)
                    merged_flags[j] = True
                    merges_count += 1
                    per_class_merge_counts[cls_i] += 1
                    if getattr(self.config.embedding, 'reid_debug', False):
                        print(f"  MERGE {ent_i.entity_id} + {ent_j.entity_id} sim={sim:.3f} overlap={overlap_ratio:.2f} dist={spatial_dist_norm:.2f} th={threshold_i:.3f}")
            if len(similar_indices) == 1:
                merged_entities.append(ent_i)
                continue
            # Consolidate observations
            all_obs: List[Observation] = []
            for idx in similar_indices:
                all_obs.extend(valid_entities[idx].observations)
            merged = PerceptionEntity(
                entity_id=f"merged_{ent_i.entity_id}",
                object_class=ent_i.object_class,
                observations=all_obs,
            )
            
            # Back-populate merged entity_id to observations
            for obs in all_obs:
                obs.entity_id = merged.entity_id
                
            # Compute average embedding with optional downsampling
            obs_embeddings = [obs.visual_embedding for obs in all_obs]
            if max_embeddings_per_entity and len(obs_embeddings) > max_embeddings_per_entity:
                # Random but deterministic subset (seed could be configurable)
                subset_idx = np.random.choice(len(obs_embeddings), size=max_embeddings_per_entity, replace=False)
                obs_embeddings = [obs_embeddings[k] for k in subset_idx]
            avg_emb = np.mean(obs_embeddings, axis=0)
            norm = np.linalg.norm(avg_emb)
            merged.average_embedding = avg_emb / norm if norm > 0 else avg_emb
            merged_entities.append(merged)
            logger.info(f"  Merged {len(similar_indices)} entities into {merged.entity_id} ({cls_i})")

        reduction = len(valid_entities) - len(merged_entities)
        logger.info(f"Output entities: {len(merged_entities)} (reduced by {reduction})")
        logger.info("="*80 + "\n")

        # Store metrics for inclusion in final result
        self._reid_metrics = {
            "backend": backend,
            "base_threshold": base_threshold,
            "class_thresholds": class_thresholds,
            "similarity_stats": similarity_stats,
            "merges_total": merges_count,
            "per_class_merges": per_class_merge_counts,
            "reduction": reduction,
        }

        return merged_entities
    
    def _detections_to_observations(self, detections: List[dict]) -> List[Observation]:
        """
        Convert detection dicts to Observation objects.
        
        Args:
            detections: List of detection dicts
            
        Returns:
            List of Observation objects
        """
        observations = []
        
        for i, det in enumerate(detections):
            # Map class name to ObjectClass enum
            try:
                object_class = ObjectClass(det["object_class"])
            except ValueError:
                object_class = ObjectClass.UNKNOWN
            
            obs = Observation(
                bounding_box=det["bounding_box"],
                centroid=det["centroid"],
                object_class=object_class,
                confidence=det["confidence"],
                visual_embedding=det["embedding"],
                frame_number=det["frame_number"],
                timestamp=det["timestamp"],
                temp_id=f"obs_{i}",
                image_patch=det.get("crop"),
                spatial_zone=det.get("spatial_zone"),
                raw_yolo_class=det.get("object_class"),
                frame_width=det.get("frame_width"),
                frame_height=det.get("frame_height"),
                depth_mm=det.get("depth_mm"),
                centroid_3d_mm=det.get("centroid_3d_mm"),
                visibility_state=det.get("visibility_state"),
                occlusion_ratio=det.get("occlusion_ratio"),
                segmentation_mask=det.get("segmentation_mask"),
                features=det.get("features") or {},
            )
            
            observations.append(obs)
        
        return observations

    def _run_enhanced_tracking(self, detections: List[dict]) -> None:
        """Run EnhancedTracker over detections grouped by frame, with SLAM pose update."""
        if self.enhanced_tracker is None:
            return

        # Group detections by frame
        by_frame: dict[int, List[dict]] = {}
        for det in detections:
            by_frame.setdefault(int(det["frame_number"]), []).append(det)

        last_stats = {}
        for fidx in sorted(by_frame.keys()):
            frame_dets = by_frame[fidx]
            # Build arrays for tracker
            converted, embs = self._convert_for_enhanced_tracker(frame_dets)

            # For now, use simple identity pose (SLAM integration requires frame/depth data)
            # TODO: Properly integrate SLAM by passing frames and depth to process_frame()
            camera_pose = None
            if self.slam_engine is not None:
                # Generate simple forward-moving trajectory for visualization
                # In a full implementation, pass actual frame data to slam_engine.process_frame()
                t = np.array([0.0, 0.0, fidx * 0.05])  # Move 5cm per frame
                camera_pose = np.eye(4)
                camera_pose[:3, 3] = t
                self.slam_engine.poses.append(camera_pose)
                self.slam_engine.trajectory.append(t)

            # Update tracker with pose for camera motion compensation
            self.enhanced_tracker.update(converted, embs, camera_pose=camera_pose, frame_idx=fidx)
            last_stats = self.enhanced_tracker.get_statistics()

        if last_stats:
            logger.info(
                f"  EnhancedTracker: total_tracks={last_stats.get('total_tracks', 0)}, "
                f"confirmed={last_stats.get('confirmed_tracks', 0)}"
            )

    def _convert_for_enhanced_tracker(self, frame_dets: List[dict]) -> Tuple[List[dict], List[Optional[np.ndarray]]]:
        """Convert observer/embedder detections to EnhancedTracker format.

        Required keys by EnhancedTracker per detection:
          - bbox_3d: [x, y, z, w, h, d] (mm)
          - bbox_2d: [x1, y1, x2, y2]
          - class_name: str
          - confidence: float
          - depth_mm: float
        """
        import numpy as _np  # local import to avoid hard dependency at module load

        converted: List[dict] = []
        embeddings: List[Optional[_np.ndarray]] = []

        for det in frame_dets:
            # 2D bbox
            bb = det.get("bounding_box")
            if bb is None:
                # Fallback: skip if no bbox present
                continue
            x1, y1, x2, y2 = float(bb.x1), float(bb.y1), float(bb.x2), float(bb.y2)
            w_px = max(1.0, x2 - x1)
            h_px = max(1.0, y2 - y1)

            # 3D center (prefer centroid_3d_mm if available)
            c3d = det.get("centroid_3d_mm")
            depth_mm = float(det.get("depth_mm", 0.0) or 0.0)
            if c3d is not None and len(c3d) == 3:
                cx_mm, cy_mm, cz_mm = float(c3d[0]), float(c3d[1]), float(c3d[2])
            else:
                # Fallback: approximate using 2D centroid (pixels) and depth
                c2d = det.get("centroid", ((x1 + x2) / 2.0, (y1 + y2) / 2.0))
                cx_mm, cy_mm, cz_mm = float(c2d[0]), float(c2d[1]), depth_mm

            # Approximate size in mm using pixel extents as a proxy
            # (relative values suffice for cost scaling inside EnhancedTracker)
            size_mm = _np.array([w_px, h_px, max(w_px, h_px)], dtype=_np.float32)

            converted.append({
                'bbox_3d': _np.array([cx_mm, cy_mm, cz_mm, *size_mm.tolist()], dtype=_np.float32),
                'bbox_2d': _np.array([x1, y1, x2, y2], dtype=_np.float32),
                'class_name': str(det.get("object_class", "unknown")),
                'confidence': float(det.get("confidence", 0.0)),
                'depth_mm': depth_mm,
            })

            # Prefer CLIP embeddings for EnhancedTracker's semantic checks
            emb = det.get("clip_embedding", det.get("embedding"))
            embeddings.append(emb if emb is not None else None)

        return converted, embeddings


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def run_perception(
    video_path: str,
    config: Optional[PerceptionConfig] = None,
    verbose: bool = False,
) -> PerceptionResult:
    """
    Convenience function to run perception pipeline.
    
    Args:
        video_path: Path to video file
        config: Optional perception configuration
        verbose: Enable verbose logging
        
    Returns:
        PerceptionResult with entities and observations
        
    Example:
        >>> from orion.perception.engine import run_perception
        >>> result = run_perception("video.mp4")
        >>> print(f"Found {result.unique_entities} entities")
    """
    engine = PerceptionEngine(config=config, verbose=verbose)
    return engine.process_video(video_path)
