"""
Comprehensive Detection Analysis
=================================

Analyzes perception pipeline output to show:
- What objects are actually being detected
- Per-class detection counts and confidence stats
- Temporal persistence (how long entities last)
- Spatial distribution across zones
- Tracking quality metrics (ID switches, fragmentation)

Usage:
    python scripts/analyze_detections.py --video data/examples/video.mp4 --mode accurate --tracking
    python scripts/analyze_detections.py --video data/examples/video.mp4 --mode quick --save-entities

Outputs detailed statistics to console and optionally saves entity details to JSON/CSV.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List

import numpy as np

from orion.perception.config import get_fast_config, get_balanced_config, get_accurate_config
from orion.perception.engine import run_perception


def build_config(mode: str, tracking: bool, target_fps: float | None, conf: float | None):
    if mode == "quick":
        cfg = get_fast_config()
        cfg.target_fps = 0.5 if target_fps is None else target_fps
        cfg.detection.confidence_threshold = 0.5 if conf is None else conf
    elif mode == "balanced":
        cfg = get_balanced_config()
        if target_fps:
            cfg.target_fps = target_fps
        if conf:
            cfg.detection.confidence_threshold = conf
    elif mode == "accurate":
        cfg = get_accurate_config()
        if target_fps:
            cfg.target_fps = target_fps
        if conf:
            cfg.detection.confidence_threshold = conf
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    cfg.enable_tracking = tracking
    return cfg


def analyze_result(result, config):
    """Generate comprehensive analysis of perception results."""
    entities = result.entities
    observations = result.raw_observations
    metrics = result.metrics or {}
    
    # === 1. Detection Summary ===
    print("\n" + "=" * 80)
    print("DETECTION SUMMARY")
    print("=" * 80)
    print(f"Total detections: {result.total_detections}")
    print(f"Unique entities: {result.unique_entities}")
    print(f"Sampled frames: {metrics.get('sampled_frames', 'N/A')}")
    print(f"Avg detections/frame: {metrics.get('detections_per_sampled_frame', 0.0):.2f}")
    
    # === 2. Per-Class Breakdown ===
    print("\n" + "=" * 80)
    print("DETECTED OBJECTS BY CLASS")
    print("=" * 80)
    
    class_detections = Counter([obs.object_class.value for obs in observations])
    class_entities = Counter([ent.object_class.value for ent in entities])
    
    # Calculate per-class stats
    class_stats = []
    for cls in sorted(class_detections.keys()):
        det_count = class_detections[cls]
        ent_count = class_entities[cls]
        
        # Confidence stats
        cls_obs = [obs for obs in observations if obs.object_class.value == cls]
        confs = [obs.confidence for obs in cls_obs]
        avg_conf = np.mean(confs)
        std_conf = np.std(confs)
        
        # Temporal stats
        cls_ents = [ent for ent in entities if ent.object_class.value == cls]
        durations = [ent.last_seen_frame - ent.first_seen_frame for ent in cls_ents]
        avg_duration = np.mean(durations) if durations else 0
        
        class_stats.append({
            'class': cls,
            'detections': det_count,
            'entities': ent_count,
            'avg_conf': avg_conf,
            'std_conf': std_conf,
            'avg_duration_frames': avg_duration,
        })
    
    # Print table
    print(f"{'Class':<20} {'Detections':<12} {'Entities':<10} {'Avg Conf':<12} {'Avg Duration':<15}")
    print("-" * 80)
    for stat in sorted(class_stats, key=lambda x: x['detections'], reverse=True):
        print(
            f"{stat['class']:<20} {stat['detections']:<12} {stat['entities']:<10} "
            f"{stat['avg_conf']:.3f}±{stat['std_conf']:.3f}   {stat['avg_duration_frames']:.1f} frames"
        )
    
    # === 3. Entity Persistence ===
    print("\n" + "=" * 80)
    print("ENTITY PERSISTENCE (Top 10)")
    print("=" * 80)
    print(f"{'Entity ID':<15} {'Class':<20} {'Appearances':<12} {'Frames':<20} {'Duration':<10}")
    print("-" * 80)
    
    sorted_entities = sorted(entities, key=lambda e: e.appearance_count, reverse=True)
    for ent in sorted_entities[:10]:
        frame_span = f"{ent.first_seen_frame}-{ent.last_seen_frame}"
        duration = ent.last_seen_frame - ent.first_seen_frame
        print(
            f"{ent.entity_id:<15} {ent.object_class.value:<20} {ent.appearance_count:<12} "
            f"{frame_span:<20} {duration:<10}"
        )
    
    # === 4. Spatial Distribution ===
    print("\n" + "=" * 80)
    print("SPATIAL DISTRIBUTION")
    print("=" * 80)
    
    # Handle both string zones and SpatialContext objects
    zone_counts = Counter()
    for obs in observations:
        if obs.spatial_zone:
            if isinstance(obs.spatial_zone, str):
                zone_counts[obs.spatial_zone] += 1
            elif hasattr(obs.spatial_zone, 'zone_type'):
                zone_counts[obs.spatial_zone.zone_type] += 1
    
    if zone_counts:
        print(f"{'Zone':<20} {'Detections':<12} {'Percentage':<12}")
        print("-" * 50)
        total = sum(zone_counts.values())
        for zone, count in zone_counts.most_common():
            pct = 100 * count / total
            print(f"{zone:<20} {count:<12} {pct:.1f}%")
    else:
        print("(No spatial zone data)")
    
    # === 5. Tracking Metrics ===
    if metrics.get('total_tracks'):
        print("\n" + "=" * 80)
        print("TRACKING QUALITY")
        print("=" * 80)
        print(f"Total tracks: {metrics.get('total_tracks', 0)}")
        print(f"Confirmed tracks: {metrics.get('confirmed_tracks', 0)}")
        print(f"Active tracks: {metrics.get('active_tracks', 0)}")
        print(f"ID switches: {metrics.get('id_switches', 0)}")
        
        # Calculate fragmentation
        if result.unique_entities > 0:
            fragmentation = metrics.get('total_tracks', 0) / result.unique_entities
            print(f"Fragmentation ratio: {fragmentation:.2f} (tracks/entity, lower is better)")
    
    # === 6. Timing Breakdown ===
    timings = metrics.get('timings', {})
    if timings:
        print("\n" + "=" * 80)
        print("PERFORMANCE BREAKDOWN")
        print("=" * 80)
        total = timings.get('total_seconds', result.processing_time_seconds)
        
        phases = [
            ('Detection (YOLO)', timings.get('detection_seconds', 0)),
            ('Embedding (DINO/CLIP)', timings.get('embedding_seconds', 0)),
            ('Tracking', timings.get('tracking_seconds', 0)),
            ('Clustering', timings.get('clustering_seconds', 0)),
            ('Description (VLM)', timings.get('description_seconds', 0)),
        ]
        
        print(f"{'Phase':<25} {'Time (s)':<12} {'Percentage':<12}")
        print("-" * 50)
        for phase_name, phase_time in phases:
            pct = 100 * phase_time / total if total > 0 else 0
            print(f"{phase_name:<25} {phase_time:<12.3f} {pct:.1f}%")
        print("-" * 50)
        print(f"{'TOTAL':<25} {total:<12.3f} 100.0%")
    
    return class_stats


def save_entity_details(entities, output_path: Path):
    """Save per-entity details to JSON."""
    entity_data = []
    for ent in entities:
        entity_data.append({
            'entity_id': ent.entity_id,
            'class': ent.object_class.value,
            'appearances': ent.appearance_count,
            'first_frame': ent.first_seen_frame,
            'last_frame': ent.last_seen_frame,
            'duration_frames': ent.last_seen_frame - ent.first_seen_frame,
            'description': ent.description,
        })
    
    with output_path.open('w') as f:
        json.dump(entity_data, f, indent=2)
    
    print(f"\n[Analysis] Saved entity details to {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Input video path")
    ap.add_argument("--mode", default="accurate", choices=["quick", "balanced", "accurate"])
    ap.add_argument("--tracking", action="store_true", help="Enable tracking metrics")
    ap.add_argument("--target-fps", type=float, default=None)
    ap.add_argument("--conf", type=float, default=None, help="Override confidence threshold")
    ap.add_argument("--save-entities", action="store_true", help="Save entity details to JSON")
    args = ap.parse_args()
    
    # Build config
    cfg = build_config(args.mode, args.tracking, args.target_fps, args.conf)
    
    print(f"\n[Analysis] Running perception on {args.video}")
    print(f"[Analysis] Mode: {args.mode}, Tracking: {args.tracking}")
    print(f"[Analysis] Backend: {cfg.embedding.backend}, Model: {cfg.detection.model}")
    
    # Run perception
    result = run_perception(args.video, config=cfg)
    
    # Analyze
    class_stats = analyze_result(result, cfg)
    
    # Save entity details if requested
    if args.save_entities:
        out_dir = Path("results")
        out_dir.mkdir(exist_ok=True, parents=True)
        save_entity_details(result.entities, out_dir / "entities.json")
    
    print("\n" + "=" * 80)
    print(f"[Analysis] Complete! Detected {result.unique_entities} unique entities across {result.total_detections} detections")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
