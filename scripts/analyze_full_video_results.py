import json
import numpy as np
from pathlib import Path
import sys

# Add workspace root to path
workspace_root = Path(__file__).resolve().parents[1]
sys.path.append(str(workspace_root))

def analyze_results():
    results_dir = workspace_root / "results/full_video_analysis"
    json_path = results_dir / "pipeline_output.json"
    
    if not json_path.exists():
        print(f"Error: Results not found at {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    entities = data.get("entities", [])
    total_frames = data.get("total_frames", 0)
    
    print(f"Total Frames: {total_frames}")
    print(f"Total Entities: {len(entities)}")
    
    # Analyze Re-ID: Check for entities with long duration or large gaps between observations
    print("\n--- Re-ID Analysis (Long-term persistence) ---")
    long_term_entities = []
    reid_candidates = []
    
    for entity in entities:
        obs_frames = [obs['frame'] for obs in entity.get('observations', [])]
        if not obs_frames:
            continue
            
        min_frame = min(obs_frames)
        max_frame = max(obs_frames)
        duration = max_frame - min_frame
        
        # Check for gaps (indicating re-id after disappearance)
        sorted_frames = sorted(obs_frames)
        max_gap = 0
        if len(sorted_frames) > 1:
            max_gap = max(np.diff(sorted_frames))
            
        entity_info = {
            "id": entity.get("entity_id"),
            "class": entity.get("object_class"),
            "original_class": entity.get("original_class"), # If available
            "min_frame": min_frame,
            "max_frame": max_frame,
            "duration": duration,
            "max_gap": max_gap,
            "num_observations": len(obs_frames)
        }
        
        # Define "long term" as spanning > 50% of video or appearing at start and end
        if duration > (total_frames * 0.5) or (min_frame < total_frames * 0.1 and max_frame > total_frames * 0.9):
            long_term_entities.append(entity_info)
            
        # Define "Re-ID" as having a significant gap (e.g., > 100 frames) where it wasn't seen
        if max_gap > 100:
            reid_candidates.append(entity_info)

    print(f"Entities spanning start to end (>50% duration or start+end): {len(long_term_entities)}")
    for e in long_term_entities:
        print(f"  - ID: {e['id']} ({e['class']}): Frames {e['min_frame']} -> {e['max_frame']} (Gap: {e['max_gap']})")

    print(f"\nEntities with significant gaps (>100 frames) - Potential Re-ID events: {len(reid_candidates)}")
    for e in reid_candidates:
        print(f"  - ID: {e['id']} ({e['class']}): Frames {e['min_frame']} -> {e['max_frame']} (Max Gap: {e['max_gap']})")

    # Analyze Spatial Consistency (if 3D data available)
    # This is harder without visualizing, but we can check if entities have 3D positions
    print("\n--- Spatial Analysis ---")
    entities_with_3d = [e for e in entities if e.get('position_3d') is not None]
    print(f"Entities with 3D positions: {len(entities_with_3d)}")
    
    # Check SLAM trajectory
    slam_path = results_dir / "slam_trajectory.npy"
    if slam_path.exists():
        trajectory = np.load(slam_path)
        print(f"SLAM Trajectory loaded: {trajectory.shape} poses")
        
        # Check for loop closure (start and end positions close)
        if len(trajectory) > 0:
            start_pose = trajectory[0] # 4x4 matrix
            end_pose = trajectory[-1]
            
            # Extract translation
            start_trans = start_pose[:3, 3]
            end_trans = end_pose[:3, 3]
            
            dist = np.linalg.norm(start_trans - end_trans)
            print(f"Distance between start and end camera pose: {dist:.4f} units")
            if dist < 1.0: # Threshold depends on scale, assuming meters usually
                print("  -> Possible loop closure detected (start and end are spatially close)")
            else:
                print("  -> Start and end positions are distinct")

if __name__ == "__main__":
    analyze_results()
