#!/usr/bin/env python3
"""
Generate PVSG batch results in academic paper table format.
"""
import json
import os
from pathlib import Path
from typing import Dict, List

def load_metrics(results_dir: str, episode_id: str) -> Dict:
    """Load all metrics from result files"""
    meta_path = os.path.join(results_dir, episode_id, "run_metadata.json")
    mem_path = os.path.join(results_dir, episode_id, "memory.json")
    
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            data = json.load(f)
            stats = data.get("statistics", {})
            meta = {
                "frames": stats.get("frames_processed", 0),
                "detections": stats.get("total_detections", 0),
                "tracks": stats.get("unique_tracks", 0),
                "time": data.get("processing_time_seconds", 0),
            }
    
    mem_objects = 0
    if os.path.exists(mem_path):
        with open(mem_path) as f:
            data = json.load(f)
            mem_objects = len(data.get("objects", []))
    
    meta["objects"] = mem_objects
    
    # Compute derived metrics
    if meta.get("frames", 0) > 0:
        meta["det_per_frame"] = round(meta["detections"] / meta["frames"], 2)
        meta["tracks_per_frame"] = round(meta["tracks"] / meta["frames"], 2)
    else:
        meta["det_per_frame"] = 0
        meta["tracks_per_frame"] = 0
    
    return meta

def format_table(results_dir: str = "results", video_ids: List[str] = None):
    """Format results as academic-style table"""
    
    if video_ids is None:
        video_ids = [
            "0001_4164158586",
            "0003_3396832512",
            "0003_6141007489",
            "0005_2505076295",
            "0006_2889117240",
            "0008_6225185844",
            "0008_8890945814",
            "0018_3057666738",
            "0020_10793023296",
            "0020_5323209509",
        ]
    
    results = []
    for vid_id in video_ids:
        metrics = load_metrics(results_dir, vid_id)
        results.append((vid_id, metrics))
    
    # Print Table 1: Detection & Tracking Metrics
    print("\n" + "="*130)
    print("Table 1. Orion Scene Graph Generation on PVSG (VidOR Subset) - Detection & Tracking Metrics")
    print("="*130)
    print(f"{'Method':<20} {'Frames':<10} {'Detections':<14} {'Det/Frame':<12} {'Unique Tracks':<15} {'Tracks/Frame':<13} {'Time(s)':<10}")
    print("-"*130)
    
    total_frames = 0
    total_dets = 0
    total_tracks = 0
    total_time = 0
    
    for vid_id, metrics in results:
        print(f"{vid_id:<20} {metrics['frames']:<10} {metrics['detections']:<14} {metrics['det_per_frame']:<12.2f} {metrics['tracks']:<15} {metrics['tracks_per_frame']:<13.2f} {metrics['time']:<10.2f}")
        total_frames += metrics["frames"]
        total_dets += metrics["detections"]
        total_tracks += metrics["tracks"]
        total_time += metrics["time"]
    
    print("-"*130)
    avg_det_per_frame = total_dets / total_frames if total_frames > 0 else 0
    avg_tracks_per_frame = total_tracks / total_frames if total_frames > 0 else 0
    print(f"{'Average/Total':<20} {total_frames:<10} {total_dets:<14} {avg_det_per_frame:<12.2f} {total_tracks:<15} {avg_tracks_per_frame:<13.2f} {total_time:<10.2f}")
    print("="*130)
    
    # Print Table 2: Memory & Objects Metrics
    print("\n" + "="*130)
    print("Table 2. Orion Scene Graph Generation on PVSG (VidOR Subset) - Memory & Object Metrics")
    print("="*130)
    print(f"{'Method':<20} {'Memory Objects':<16} {'Objects/Frame':<16} {'Detections/Object':<18} {'Tracks/Object':<15}")
    print("-"*130)
    
    for vid_id, metrics in results:
        det_per_obj = metrics["detections"] / metrics["objects"] if metrics["objects"] > 0 else 0
        tracks_per_obj = metrics["tracks"] / metrics["objects"] if metrics["objects"] > 0 else 0
        obj_per_frame = metrics["objects"] / metrics["frames"] if metrics["frames"] > 0 else 0
        print(f"{vid_id:<20} {metrics['objects']:<16} {obj_per_frame:<16.3f} {det_per_obj:<18.2f} {tracks_per_obj:<15.2f}")
    
    print("-"*130)
    total_objects = sum(m["objects"] for _, m in results)
    avg_obj_per_frame = total_objects / total_frames if total_frames > 0 else 0
    avg_det_per_obj = total_dets / total_objects if total_objects > 0 else 0
    avg_tracks_per_obj = total_tracks / total_objects if total_objects > 0 else 0
    print(f"{'Average/Total':<20} {total_objects:<16} {avg_obj_per_frame:<16.3f} {avg_det_per_obj:<18.2f} {avg_tracks_per_obj:<15.2f}")
    print("="*130)

if __name__ == "__main__":
    format_table()
