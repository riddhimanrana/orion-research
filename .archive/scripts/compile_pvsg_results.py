#!/usr/bin/env python3
"""
Compile PVSG batch results into a comparison table (like PVSG/VSGR paper format).
"""
import json
import os
from pathlib import Path
from typing import Dict, List

def load_run_metadata(results_dir: str, episode_id: str) -> Dict:
    """Load tracking metrics from run_metadata.json"""
    meta_path = os.path.join(results_dir, episode_id, "run_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            data = json.load(f)
            # Extract nested statistics
            stats = data.get("statistics", {})
            return {
                "frames_processed": stats.get("frames_processed"),
                "total_detections": stats.get("total_detections"),
                "unique_tracks": stats.get("unique_tracks"),
                "processing_time": data.get("processing_time_seconds"),
            }
    return {}

def load_memory(results_dir: str, episode_id: str) -> Dict:
    """Load Re-ID clustering results from memory.json"""
    mem_path = os.path.join(results_dir, episode_id, "memory.json")
    if os.path.exists(mem_path):
        with open(mem_path) as f:
            return json.load(f)
    return {}

def count_tracks(results_dir: str, episode_id: str) -> int:
    """Count unique tracks from tracks.jsonl"""
    tracks_path = os.path.join(results_dir, episode_id, "tracks.jsonl")
    if os.path.exists(tracks_path):
        track_ids = set()
        with open(tracks_path) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if "track_id" in obj:
                        track_ids.add(obj["track_id"])
                except:
                    pass
        return len(track_ids)
    return 0

def compile_results(results_dir: str = "results", video_ids: List[str] = None):
    """Compile batch results into table format"""
    
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
    
    rows = []
    for vid_id in video_ids:
        meta = load_run_metadata(results_dir, vid_id)
        mem = load_memory(results_dir, vid_id)
        
        row = {
            "Video ID": vid_id,
            "Frames": meta.get("frames_processed", "N/A"),
            "Detections": meta.get("total_detections", "N/A"),
            "Unique Tracks": meta.get("unique_tracks", "N/A"),
            "Memory Objects": len(mem.get("objects", [])) if mem else "N/A",
            "Processing Time (s)": round(meta.get("processing_time", 0), 2),
        }
        rows.append(row)
    
    # Print table
    print("\n" + "="*120)
    print("ORION SCENE GRAPH ON PVSG (First 10 Videos - VidOR Subset)")
    print("="*120)
    print(f"{'Video ID':<20} {'Frames':<10} {'Detections':<12} {'Unique Tracks':<15} {'Memory Objects':<15} {'Time (s)':<12}")
    print("-"*120)
    
    for row in rows:
        print(f"{row['Video ID']:<20} {str(row['Frames']):<10} {str(row['Detections']):<12} {str(row['Unique Tracks']):<15} {str(row['Memory Objects']):<15} {str(row['Processing Time (s)']):<12}")
    
    print("-"*120)
    
    # Aggregate stats
    total_frames = sum(r['Frames'] if isinstance(r['Frames'], int) else 0 for r in rows)
    total_detections = sum(r['Detections'] if isinstance(r['Detections'], int) else 0 for r in rows)
    total_tracks = sum(r['Unique Tracks'] if isinstance(r['Unique Tracks'], int) else 0 for r in rows)
    total_objects = sum(r['Memory Objects'] if isinstance(r['Memory Objects'], int) else 0 for r in rows)
    total_time = sum(float(r['Processing Time (s)']) if isinstance(r['Processing Time (s)'], (int, float)) else 0 for r in rows)
    
    print(f"{'TOTAL':<20} {total_frames:<10} {total_detections:<12} {total_tracks:<15} {total_objects:<15} {total_time:<12.2f}")
    print("="*120)
    
    return rows

if __name__ == "__main__":
    compile_results()
