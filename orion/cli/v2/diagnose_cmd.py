"""orion diagnose - Diagnose Re-ID and tracking quality"""

import json
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


def run_diagnose(args) -> int:
    """Run diagnostics on Re-ID and tracking quality."""
    
    episode_dir = Path("results") / args.episode
    
    if not episode_dir.exists():
        print(f"Episode not found: {args.episode}")
        return 1
    
    tracks_path = episode_dir / "tracks.jsonl"
    memory_path = episode_dir / "memory.json"
    
    if not tracks_path.exists():
        print("No tracks found. Run: orion analyze --episode <name>")
        return 1
    
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  RE-ID DIAGNOSTICS: {args.episode:<44} ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    # Load tracks
    tracks_by_id = defaultdict(list)
    with open(tracks_path) as f:
        for line in f:
            if line.strip():
                obs = json.loads(line)
                tracks_by_id[obs["track_id"]].append(obs)
    
    # Analyze track statistics
    print("  1. TRACK STATISTICS")
    print("  " + "─" * 60)
    
    track_stats = []
    for track_id, observations in tracks_by_id.items():
        frames = [o["frame_id"] for o in observations]
        labels = set(o.get("label", "") for o in observations)
        confidences = [o.get("confidence", 0) for o in observations]
        
        track_stats.append({
            "id": track_id,
            "observations": len(observations),
            "frame_span": max(frames) - min(frames) + 1,
            "labels": labels,
            "avg_conf": sum(confidences) / len(confidences) if confidences else 0,
            "gaps": count_gaps(frames),
        })
    
    print(f"  Total tracks: {len(track_stats)}")
    print(f"  Total observations: {sum(t['observations'] for t in track_stats)}")
    
    # Sort by observation count
    track_stats.sort(key=lambda x: x["observations"], reverse=True)
    
    print(f"\n  Top 10 tracks by observation count:")
    print(f"  {'Track ID':<15} {'Obs':<8} {'Span':<10} {'Gaps':<8} {'Conf':<8} {'Labels'}")
    print(f"  {'-'*15} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*20}")
    
    for stat in track_stats[:10]:
        labels_str = ", ".join(stat["labels"])[:25]
        print(f"  {stat['id']:<15} {stat['observations']:<8} {stat['frame_span']:<10} {stat['gaps']:<8} {stat['avg_conf']:.2f}     {labels_str}")
    
    # Analyze fragmentation
    print(f"\n\n  2. FRAGMENTATION ANALYSIS")
    print("  " + "─" * 60)
    
    short_tracks = [t for t in track_stats if t["observations"] < 5]
    medium_tracks = [t for t in track_stats if 5 <= t["observations"] < 30]
    long_tracks = [t for t in track_stats if t["observations"] >= 30]
    
    print(f"  Short tracks (<5 obs):    {len(short_tracks):<5} ({100*len(short_tracks)/len(track_stats):.1f}%)")
    print(f"  Medium tracks (5-30 obs): {len(medium_tracks):<5} ({100*len(medium_tracks)/len(track_stats):.1f}%)")
    print(f"  Long tracks (>30 obs):    {len(long_tracks):<5} ({100*len(long_tracks)/len(track_stats):.1f}%)")
    
    fragmented = [t for t in track_stats if t["gaps"] > 0]
    print(f"\n  Tracks with gaps: {len(fragmented)} ({100*len(fragmented)/len(track_stats):.1f}%)")
    
    if fragmented:
        avg_gaps = sum(t["gaps"] for t in fragmented) / len(fragmented)
        print(f"  Average gaps per fragmented track: {avg_gaps:.1f}")
    
    # Analyze label consistency
    print(f"\n\n  3. LABEL CONSISTENCY")
    print("  " + "─" * 60)
    
    label_changes = [t for t in track_stats if len(t["labels"]) > 1]
    print(f"  Tracks with multiple labels: {len(label_changes)} ({100*len(label_changes)/len(track_stats):.1f}%)")
    
    if label_changes:
        print(f"\n  Examples of label inconsistency:")
        for t in label_changes[:5]:
            print(f"    Track {t['id']}: {', '.join(t['labels'])}")
    
    # Analyze memory clustering if available
    if memory_path.exists():
        print(f"\n\n  4. MEMORY CLUSTERING")
        print("  " + "─" * 60)
        
        with open(memory_path) as f:
            memory = json.load(f)
        
        objects = memory.get("objects", [])
        print(f"  Memory objects: {len(objects)}")
        print(f"  Tracks merged: {len(track_stats) - len(objects)}")
        print(f"  Merge ratio: {len(objects)/len(track_stats):.2f}")
        
        # Show merged tracks
        for obj in objects[:5]:
            merged = obj.get("merged_track_ids", [obj["id"]])
            if len(merged) > 1:
                print(f"\n    {obj['canonical_label']} merged tracks: {merged}")
    
    # Re-ID embedding analysis
    embeddings_path = episode_dir / "embeddings.npy"
    if embeddings_path.exists():
        print(f"\n\n  5. EMBEDDING ANALYSIS")
        print("  " + "─" * 60)
        
        embeddings = np.load(embeddings_path, allow_pickle=True).item()
        
        dims = set()
        for track_id, emb in embeddings.items():
            if isinstance(emb, np.ndarray):
                dims.add(emb.shape[-1])
        
        print(f"  Tracks with embeddings: {len(embeddings)}")
        print(f"  Embedding dimensions: {dims}")
        
        # Compute similarity matrix for top tracks
        if len(track_stats) >= 2 and args.similarity:
            print(f"\n  Computing pairwise similarities for top {min(10, len(track_stats))} tracks...")
            
            top_ids = [t["id"] for t in track_stats[:10]]
            top_embs = []
            valid_ids = []
            
            for tid in top_ids:
                if tid in embeddings:
                    emb = embeddings[tid]
                    if isinstance(emb, np.ndarray):
                        top_embs.append(emb.flatten())
                        valid_ids.append(tid)
            
            if len(top_embs) >= 2:
                # Normalize and compute cosine similarity
                top_embs = np.array(top_embs)
                norms = np.linalg.norm(top_embs, axis=1, keepdims=True)
                top_embs = top_embs / (norms + 1e-8)
                sim_matrix = top_embs @ top_embs.T
                
                print(f"\n  Similarity matrix (top {len(valid_ids)} tracks):")
                print(f"  {'':<12}", end="")
                for tid in valid_ids:
                    print(f" {tid[:8]:<8}", end="")
                print()
                
                for i, tid in enumerate(valid_ids):
                    print(f"  {tid[:12]:<12}", end="")
                    for j in range(len(valid_ids)):
                        if i == j:
                            print(f" {'—':<8}", end="")
                        else:
                            print(f" {sim_matrix[i,j]:.4f}  ", end="")
                    print()
    
    # Recommendations
    print(f"\n\n  6. RECOMMENDATIONS")
    print("  " + "─" * 60)
    
    issues = []
    
    if len(short_tracks) > len(track_stats) * 0.5:
        issues.append("• High fragmentation: >50% of tracks have <5 observations")
        issues.append("  → Increase IoU threshold for tracker")
        issues.append("  → Lower Re-ID similarity threshold")
    
    if len(label_changes) > len(track_stats) * 0.2:
        issues.append("• Label inconsistency: >20% of tracks have multiple labels")
        issues.append("  → Use VLM filtering to standardize labels")
        issues.append("  → Consider track-level label voting")
    
    if len(fragmented) > len(track_stats) * 0.3:
        issues.append("• Temporal gaps: >30% of tracks have frame gaps")
        issues.append("  → Increase max_age in tracker config")
        issues.append("  → Use Re-ID to reconnect fragmented tracks")
    
    if memory_path.exists():
        objects = memory.get("objects", [])
        if len(objects) > len(track_stats) * 0.9:
            issues.append("• Low merging: Memory has nearly as many objects as tracks")
            issues.append("  → V-JEPA2 embeddings may not be clustering well")
            issues.append("  → Try lowering cluster threshold")
    
    if not issues:
        issues.append("✓ No major issues detected")
        issues.append("  Track quality appears good")
    
    for issue in issues:
        print(f"  {issue}")
    
    print()
    return 0


def count_gaps(frames: list) -> int:
    """Count gaps in frame sequence."""
    if len(frames) < 2:
        return 0
    
    frames = sorted(frames)
    gaps = 0
    for i in range(1, len(frames)):
        if frames[i] - frames[i-1] > 1:
            gaps += 1
    return gaps
