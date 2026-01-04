"""orion graph - Build scene graph with CIS"""

import json
import logging
from pathlib import Path
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


def run_graph(args) -> int:
    """Build scene graph and compute Causal Influence Scores."""
    
    episode_dir = Path("results") / args.episode
    
    # Use filtered tracks if available
    tracks_path = episode_dir / "tracks_filtered.jsonl"
    if not tracks_path.exists():
        tracks_path = episode_dir / "tracks.jsonl"
    
    if not tracks_path.exists():
        print(f"No tracks found for episode: {args.episode}")
        return 1
    
    print(f"\n  Stage 4: SCENE GRAPH")
    print(f"  Using: {tracks_path.name}")
    print(f"  Compute CIS: {args.compute_cis}")
    print("  " + "─" * 60)
    
    # Load tracks
    tracks_by_frame = defaultdict(list)
    all_observations = []
    
    with open(tracks_path) as f:
        for line in f:
            if line.strip():
                obs = json.loads(line)
                all_observations.append(obs)
                tracks_by_frame[obs["frame_id"]].append(obs)
    
    frame_ids = sorted(tracks_by_frame.keys())
    print(f"  Loaded {len(all_observations)} observations across {len(frame_ids)} frames")
    
    # Build per-frame scene graphs
    scene_graphs = []
    
    for frame_id in frame_ids:
        observations = tracks_by_frame[frame_id]
        
        # Build nodes (objects in this frame)
        nodes = []
        for obs in observations:
            node = {
                "id": obs.get("memory_object_id", obs["track_id"]),
                "label": obs.get("label", "object"),
                "bbox": obs.get("bbox"),
                "confidence": obs.get("confidence", 1.0),
            }
            nodes.append(node)
        
        # Build edges (spatial relationships)
        edges = []
        for i, obs1 in enumerate(observations):
            for j, obs2 in enumerate(observations):
                if i >= j:
                    continue
                
                bbox1 = obs1.get("bbox")
                bbox2 = obs2.get("bbox")
                
                if not bbox1 or not bbox2:
                    continue
                
                # Compute spatial relationship
                predicate, confidence = compute_spatial_relation(bbox1, bbox2)
                
                if predicate and confidence > 0.3:
                    edge = {
                        "subject": obs1.get("memory_object_id", obs1["track_id"]),
                        "predicate": predicate,
                        "object": obs2.get("memory_object_id", obs2["track_id"]),
                        "confidence": round(confidence, 3),
                    }
                    edges.append(edge)
        
        scene_graph = {
            "frame_id": frame_id,
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
        }
        scene_graphs.append(scene_graph)
    
    # Save scene graphs
    sg_path = episode_dir / "scene_graph.jsonl"
    with open(sg_path, "w") as f:
        for sg in scene_graphs:
            f.write(json.dumps(sg) + "\n")
    
    print(f"  Generated {len(scene_graphs)} frame graphs")
    print(f"  Saved to {sg_path}")
    
    # Compute aggregate statistics
    total_nodes = sum(sg["node_count"] for sg in scene_graphs)
    total_edges = sum(sg["edge_count"] for sg in scene_graphs)
    print(f"  Total nodes: {total_nodes}, edges: {total_edges}")
    
    # Compute CIS if requested
    if args.compute_cis:
        print(f"\n  Computing Causal Influence Scores...")
        cis_scores = compute_cis(all_observations, scene_graphs)
        
        # Save CIS
        cis_path = episode_dir / "cis_scores.json"
        with open(cis_path, "w") as f:
            json.dump(cis_scores, f, indent=2)
        
        print(f"  Computed CIS for {len(cis_scores['object_scores'])} objects")
        print(f"  Saved to {cis_path}")
        
        # Show top influential objects
        if cis_scores["object_scores"]:
            sorted_scores = sorted(
                cis_scores["object_scores"].items(),
                key=lambda x: x[1]["influence_score"],
                reverse=True
            )
            
            print(f"\n  Top 5 influential objects:")
            for obj_id, scores in sorted_scores[:5]:
                print(f"    - {obj_id}: {scores['influence_score']:.3f}")
    
    # Update status
    meta_path = episode_dir / "episode_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        meta["status"]["graphed"] = True
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
    
    return 0


def compute_spatial_relation(bbox1: list, bbox2: list) -> tuple:
    """Compute spatial relationship between two bboxes.
    
    Returns:
        (predicate, confidence) tuple
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Centers
    cx1, cy1 = (x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2
    cx2, cy2 = (x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2
    
    # Sizes
    w1, h1 = x2_1 - x1_1, y2_1 - y1_1
    w2, h2 = x2_2 - x1_2, y2_2 - y1_2
    
    # Distance between centers
    dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
    avg_size = (w1 + h1 + w2 + h2) / 4
    
    # Check containment
    contains_1_in_2 = (x1_1 >= x1_2 and y1_1 >= y1_2 and x2_1 <= x2_2 and y2_1 <= y2_2)
    contains_2_in_1 = (x1_2 >= x1_1 and y1_2 >= y1_1 and x2_2 <= x2_1 and y2_2 <= y2_1)
    
    if contains_1_in_2:
        return "inside", 0.9
    if contains_2_in_1:
        return "contains", 0.9
    
    # Check overlap (IoU)
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area1 = w1 * h1
        area2 = w2 * h2
        iou = inter_area / (area1 + area2 - inter_area + 1e-8)
        
        if iou > 0.5:
            return "overlapping", min(0.9, iou + 0.3)
    
    # Check vertical relationship
    if cy1 > cy2 + h2 * 0.3:  # 1 is below 2
        if abs(cx1 - cx2) < avg_size:
            return "below", 0.7
    elif cy2 > cy1 + h1 * 0.3:  # 1 is above 2
        if abs(cx1 - cx2) < avg_size:
            return "above", 0.7
    
    # Check horizontal relationship
    if cx1 > cx2 + w2 * 0.3:  # 1 is right of 2
        return "right_of", 0.6
    elif cx2 > cx1 + w1 * 0.3:  # 1 is left of 2
        return "left_of", 0.6
    
    # Check nearness
    if dist < avg_size * 2:
        return "near", max(0.3, 1 - dist / (avg_size * 2))
    
    return None, 0.0


def compute_cis(observations: list, scene_graphs: list) -> dict:
    """Compute Causal Influence Scores for objects.
    
    CIS measures how much an object's presence/state affects other objects.
    Higher CIS = more influential in the scene.
    """
    
    # Track object appearances
    object_frames = defaultdict(set)  # obj_id → set of frame_ids
    object_cooccurrences = defaultdict(lambda: defaultdict(int))  # obj1 → obj2 → count
    
    for obs in observations:
        obj_id = obs.get("memory_object_id", obs["track_id"])
        object_frames[obj_id].add(obs["frame_id"])
    
    # Count co-occurrences
    frame_objects = defaultdict(set)
    for obs in observations:
        obj_id = obs.get("memory_object_id", obs["track_id"])
        frame_objects[obs["frame_id"]].add(obj_id)
    
    for frame_id, objs in frame_objects.items():
        for obj1 in objs:
            for obj2 in objs:
                if obj1 != obj2:
                    object_cooccurrences[obj1][obj2] += 1
    
    # Count relationship participation
    edge_counts = defaultdict(int)
    subject_counts = defaultdict(int)
    object_counts = defaultdict(int)
    
    for sg in scene_graphs:
        for edge in sg.get("edges", []):
            subject_counts[edge["subject"]] += 1
            object_counts[edge["object"]] += 1
            edge_counts[edge["predicate"]] += 1
    
    # Compute CIS per object
    object_scores = {}
    total_frames = len(set(obs["frame_id"] for obs in observations))
    
    for obj_id in object_frames.keys():
        # Temporal presence (how often visible)
        temporal_score = len(object_frames[obj_id]) / max(1, total_frames)
        
        # Interaction score (how often in relationships)
        interaction_score = (subject_counts[obj_id] + object_counts[obj_id]) / max(1, len(observations))
        
        # Co-occurrence score (how many other objects appear with this one)
        cooccur_score = len(object_cooccurrences[obj_id]) / max(1, len(object_frames))
        
        # Combined influence score
        influence_score = (
            0.3 * temporal_score + 
            0.5 * interaction_score + 
            0.2 * cooccur_score
        )
        
        object_scores[obj_id] = {
            "temporal_score": round(temporal_score, 3),
            "interaction_score": round(interaction_score, 3),
            "cooccurrence_score": round(cooccur_score, 3),
            "influence_score": round(influence_score, 3),
            "frames_visible": len(object_frames[obj_id]),
            "subject_count": subject_counts[obj_id],
            "object_count": object_counts[obj_id],
        }
    
    return {
        "episode_frames": total_frames,
        "total_objects": len(object_frames),
        "edge_type_counts": dict(edge_counts),
        "object_scores": object_scores,
    }
