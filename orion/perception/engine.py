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

import logging
import time
import numpy as np
from typing import List, Optional, Tuple

# Initialise logger early so try/except blocks can use it
logger = logging.getLogger(__name__)

from orion.perception.types import Observation, PerceptionEntity, PerceptionResult, ObjectClass
from orion.perception.config import PerceptionConfig
from orion.perception.observer import FrameObserver
from orion.perception.embedder import VisualEmbedder
from orion.perception.tracker import EntityTracker
from orion.perception.describer import EntityDescriber

# Phase 2: Tracking imports (legacy tracker may be archived)
try:
    from orion.perception.tracking import EntityTracker3D, TrackingConfig, BayesianEntityBelief  # type: ignore
    TRACKING_AVAILABLE = True
except ImportError:
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
        
        # Phase 2: Tracking component
        self.tracker_3d: Optional[EntityTracker3D] = None
        if self.config.enable_tracking and TRACKING_AVAILABLE:
            logger.info("  Tracking: Enabled (Phase 2)")
        elif self.config.enable_tracking and not TRACKING_AVAILABLE:
            logger.warning("  Tracking: Requested but not available!")

        if self.config.enable_tracking and ENHANCED_TRACKER_AVAILABLE:
            logger.info("  EnhancedTracker: Enabled (appearance Re-ID + 3D KF)")
        elif self.config.enable_tracking and not ENHANCED_TRACKER_AVAILABLE:
            logger.warning("  EnhancedTracker requested but import failed")

        if self.config.enable_3d and SLAM_AVAILABLE:
            logger.info("  SLAM: Will be initialized (camera motion compensation)")
        elif self.config.enable_3d and not SLAM_AVAILABLE:
            logger.warning("  SLAM requested but import failed")
        
        logger.info("PerceptionEngine initialized")
        logger.info(f"  Detection: {self.config.detection.model}")
        logger.info(f"  Embedding: backend={self.config.embedding.backend}, dim={self.config.embedding.embedding_dim}")
        logger.info(f"  Target FPS: {self.config.target_fps}")
    
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
            "yolo_model": self.config.detection.model,
            "tracker_backend": self.config.tracker_backend,
            "tapnet_tracks": len(self.tapnet_tracker.get_active_tracks()) if self.tapnet_tracker else 0,
            "cluster_embeddings": int(sum(1 for d in detections if 'cluster_rep_id' in d and d.get('cluster_size'))),
            "avg_cluster_size": float(np.mean([d.get('cluster_size',1) for d in detections if d.get('cluster_size')])) if any(d.get('cluster_size') for d in detections) else 1.0,
        })

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
            self._export_visualization_data(result, output_dir)
        
        return result
    
    def _initialize_components(self):
        """Initialize all pipeline components with models."""
        logger.info("Loading models...")
        
        # YOLO
        yolo = self.model_manager.yolo
        logger.info("  ✓ YOLO11x loaded")
        
        # CLIP (only load if requested by backend or text conditioning)
        clip = None
        try:
            if self.config.embedding.backend == "clip" or self.config.embedding.use_text_conditioning:
                clip = self.model_manager.clip
                logger.info("  ✓ CLIP loaded (semantic + description)")
            else:
                logger.info("  ✓ Skipping CLIP load (backend does not require it)")
        except Exception as e:
            logger.warning(f"  ✗ CLIP load failed: {e}. Proceeding without CLIP.")

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
        
        # Create components
        self.observer = FrameObserver(
            yolo_model=yolo,
            config=self.config.detection,
            target_fps=self.config.target_fps,
            show_progress=True,
            enable_3d=self.config.enable_3d,
            depth_model=self.config.depth_model,
            enable_occlusion=self.config.enable_occlusion,
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
        
        # Phase 2: Initialize tracking if enabled
        if self.config.enable_tracking and TRACKING_AVAILABLE:
            # Get YOLO class names from model
            yolo_classes = list(yolo.names.values()) if hasattr(yolo, 'names') else []
            
            tracking_config = TrackingConfig(
                max_distance_pixels=self.config.tracking_max_distance_pixels,
                max_distance_3d_mm=self.config.tracking_max_distance_3d_mm,
                ttl_frames=self.config.tracking_ttl_frames,
                reid_window_frames=self.config.tracking_reid_window_frames,
                class_belief_lr=self.config.tracking_class_belief_lr,
            )
            
            self.tracker_3d = EntityTracker3D(
                config=tracking_config,
                yolo_classes=yolo_classes
            )
            logger.info("  ✓ EntityTracker3D initialized (Phase 2)")
        
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
        
        # 2. Export camera intrinsics
        intrinsics_file = output_path / "camera_intrinsics.json"
        # Default intrinsics (adjust based on your camera/video)
        intrinsics = {
            "fx": 525.0,  # Focal length X
            "fy": 525.0,  # Focal length Y
            "cx": 319.5,  # Principal point X
            "cy": 239.5,  # Principal point Y
            "width": 640,
            "height": 480,
            "note": "Default intrinsics - adjust based on actual camera calibration"
        }
        with open(intrinsics_file, 'w') as f:
            json.dump(intrinsics, f, indent=2)
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
                "class": entity.object_class.value if hasattr(entity.object_class, 'value') else str(entity.object_class),
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
