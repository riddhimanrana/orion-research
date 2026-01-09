#!/usr/bin/env python3
"""
Deep Research Evaluation Script
===============================

Runs the Orion perception pipeline on remote Lambda (Linux/CUDA) with
comparison between Baseline (YOLO11x) and Enhanced (GroundingDINO + V-JEPA2 + DINO Refinement).

Author: GitHub Copilot
Date: January 8, 2026
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from orion.perception.config import PerceptionConfig, DetectionConfig, EmbeddingConfig
from orion.perception.engine import PerceptionEngine, PerceptionResult
from orion.perception.types import Observation, PerceptionEntity, ObjectClass
from orion.backends.dino_classifier import DINOv3Classifier, SceneTypeDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("deep_research")

# Evaluation targets (Lambda NFS)
VIDEOS = [
    "/lambda/nfs/orion-core-fs/test.mp4",
    "/lambda/nfs/orion-core-fs/video.mp4"
]

RESULTS_ROOT = Path("results/deep_research")

class DeepResearchEvaluator:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.results_root = RESULTS_ROOT
        self.results_root.mkdir(parents=True, exist_ok=True)
        
        # Initialize DINO v2 (fallback as DINOv3 is assumed missing on Lambda for now)
        self.dino = DINOv3Classifier(device=device)
        self.scene_detector = SceneTypeDetector()
        
    def get_baseline_config(self) -> PerceptionConfig:
        """Config A: Baseline (YOLO11x, standard tracking, no depth)."""
        return PerceptionConfig(
            detection=DetectionConfig(
                backend="yolo",
                model="yolo11x",
                confidence_threshold=0.25,
            ),
            embedding=EmbeddingConfig(
                batch_size=16,
                device=self.device,
            ),
            target_fps=1.0,  # Speed up for quick research
            enable_tracking=True,
            enable_3d=False
        )
        
    def get_enhanced_config(self) -> PerceptionConfig:
        """Config B: Enhanced (GroundingDINO base, standard tracking, V-JEPA2)."""
        return PerceptionConfig(
            detection=DetectionConfig(
                backend="groundingdino",
                model="IDEA-Research/grounding-dino-tiny", # Switch to tiny for reliability/speed in research
                confidence_threshold=0.20,
            ),
            embedding=EmbeddingConfig(
                batch_size=8, # V-JEPA2 is heavy
                device=self.device,
            ),
            target_fps=1.0,  # Speed up for quick research
            enable_tracking=True,
            enable_3d=False
        )

    def run_eval(self, config_name: str, config: PerceptionConfig, video_path: str):
        video_name = Path(video_path).stem
        output_dir = self.results_root / config_name / video_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\n[{config_name}] Processing {video_name}...")
        
        engine = PerceptionEngine(config=config)
        
        # Instrument timings
        t_start = time.time()
        result = engine.process_video(video_path, save_visualizations=True, output_dir=str(output_dir))
        total_time = time.time() - t_start
        
        # Step: DINO Refinement (Post-process if needed)
        # Note: In a real 'deep research' scenario, we might want to see how DINO
        # improves object classification compared to GroundingDINO/YOLO.
        if config_name == "enhanced":
            self.refine_with_dino(video_path, result)
            
        # Collect Metrics
        metrics = self.collect_metrics(result, total_time, engine)
        
        # Save Metrics
        metrics_file = output_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        # Save Tracks in JSONL format as requested
        tracks_file = output_dir / "tracks.jsonl"
        self.save_jsonl_tracks(result, tracks_file)
        
        logger.info(f"[{config_name}] Done. FPS: {metrics['fps_overall']:.2f}, Tracks: {metrics['unique_tracks']}")
        return metrics

    def refine_with_dino(self, video_path: str, result: PerceptionResult):
        """Perform DINO refinement on the 'best' frame/crop of each entity."""
        logger.info("Running DINO Refinement sweep...")
        
        count = 0
        for entity in tqdm(result.entities, desc="DINO Refinement"):
            if not entity.observations:
                continue
            best_obs = max(entity.observations, key=lambda o: o.confidence)
            if best_obs.image_patch is not None:
                # Use classify_crop
                ref = self.dino.classify_crop(
                    best_obs.image_patch, 
                    str(entity.object_class)
                )
                if ref.confidence > 0.4:
                    entity.corrected_class = ref.refined_class
                    entity.correction_confidence = ref.confidence
                    count += 1
        logger.info(f"Refined {count}/{len(result.entities)} entities")

    def collect_metrics(self, result: PerceptionResult, total_time: float, engine: PerceptionEngine) -> Dict[str, Any]:
        # 1. Object Recall
        unique_tracks = len(result.entities)
        total_detections = len(result.raw_observations)
        
        # 2. Identity Consistency
        # Proxy: Average observations per entity
        obs_per_entity = np.mean([len(e.observations) for e in result.entities]) if result.entities else 0
        
        # Proxy: Number of tracks per class (high = potentially many ID switches or fragmentation)
        class_counts = {}
        for entity in result.entities:
            cls = str(entity.object_class)
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        avg_tracks_per_class = np.mean(list(class_counts.values())) if class_counts else 0
        
        # 3. Confidence Calibration
        confidences = [obs.confidence for obs in result.raw_observations]
        hist, bins = np.histogram(confidences, bins=10, range=(0, 1))
        
        # 4. Pipeline Latency
        # Extract from engine if profiled, otherwise use total_time / frames
        num_frames = result.total_frames or 1
        fps_overall = num_frames / total_time if total_time > 0 else 0
        
        return {
            "unique_tracks": unique_tracks,
            "total_detections": total_detections,
            "obs_per_entity": float(obs_per_entity),
            "avg_tracks_per_class": float(avg_tracks_per_class),
            "class_distribution": class_counts,
            "confidence_histogram": hist.tolist(),
            "confidence_bins": bins.tolist(),
            "avg_confidence": float(np.mean(confidences)) if confidences else 0,
            "processing_time_sec": total_time,
            "fps_overall": fps_overall,
            "frames_processed": num_frames
        }

    def save_jsonl_tracks(self, result: PerceptionResult, output_path: Path):
        """Save raw observations to JSONL format."""
        with open(output_path, 'w') as f:
            for obs in result.raw_observations:
                obs_dict = {
                    "frame_id": obs.frame_number,
                    "timestamp": obs.timestamp,
                    "bbox": obs.bounding_box.to_list(),
                    "class": str(obs.object_class),
                    "confidence": float(obs.confidence),
                    "track_id": obs.entity_id or -1,
                }
                # Add metadata if available
                if obs.candidate_labels:
                    obs_dict["candidate_labels"] = obs.candidate_labels
                if obs.vlm_description:
                    obs_dict["vlm_description"] = obs.vlm_description
                
                f.write(json.dumps(obs_dict) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Orion Deep Research Evaluation")
    parser.add_argument("--videos", nargs="+", default=VIDEOS, help="Videos to evaluate")
    parser.add_argument("--device", default="cuda", help="Device to use")
    args = parser.parse_args()
    
    evaluator = DeepResearchEvaluator(device=args.device)
    
    configs = [
        ("baseline", evaluator.get_baseline_config()),
        ("enhanced", evaluator.get_enhanced_config())
    ]
    
    summary = {}
    
    for config_name, config in configs:
        summary[config_name] = {}
        for video_path in args.videos:
            if not os.path.exists(video_path):
                logger.warning(f"Skipping missing video: {video_path}")
                continue
            
            metrics = evaluator.run_eval(config_name, config, video_path)
            summary[config_name][Path(video_path).name] = metrics
            
    # Save final summary
    with open(RESULTS_ROOT / "global_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("\n" + "="*40)
    logger.info("DEEP RESEARCH EVALUATION COMPLETE")
    logger.info("="*40)
    logger.info(f"Results saved to: {RESULTS_ROOT}")

if __name__ == "__main__":
    main()
