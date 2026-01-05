"""
orion embed - Phase 2 V-JEPA2 Re-ID Command

Embed tracks with V-JEPA2 and cluster fragmented tracks into unified object IDs.

Usage:
    orion embed --episode phase1_test_v2
    orion embed --episode phase1_test_v2 --similarity 0.75 --device cuda
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, TYPE_CHECKING
from dataclasses import dataclass, asdict

# Defer heavy imports
if TYPE_CHECKING:
    import cv2
    import numpy as np

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()


# Object filtering taxonomy
DYNAMIC_CLASSES = {
    # People and body parts
    "person", "hand", "face", "arm",
    # Small movable objects
    "book", "notebook", "laptop", "phone", "remote", "cup", "bottle", 
    "keyboard", "mouse", "headphones", "airpods", "webcam", "microphone",
    "backpack", "box", "package", "bag", "container", "basket",
    "pen", "pencil", "paper", "document",
    # Small furniture (can move)
    "chair", "stool", "pillow", "blanket", "vase", "plant",
    "wrist rest", "mousepad", "pencil case", "figurine"
}

STATIC_CLASSES = {
    # Large furniture
    "couch", "sofa", "bed", "desk", "table", "dining table", "coffee table",
    "bookshelf", "dresser", "cabinet", "nightstand", "wardrobe", "closet",
    # Appliances
    "refrigerator", "microwave", "oven", "tv", "television", "monitor",
    # Architecture
    "wall", "ceiling", "floor", "door", "window", "staircase", "railing"
}

SUPPRESS_CLASSES = {
    "curtain", "drapes", "shadow", "reflection", "light", "baseboard",
    "doorknob", "light switch", "outlet"
}


@dataclass
class EmbedStats:
    """Statistics from embedding run."""
    tracks_before: int = 0
    tracks_embedded: int = 0
    objects_after: int = 0
    fragmentation_reduction: float = 0.0
    elapsed_seconds: float = 0.0
    avg_intra_similarity: float = 0.0
    avg_inter_similarity: float = 0.0


def load_tracks(tracks_path: Path) -> list:
    """Load tracks from JSONL file."""
    tracks = []
    with open(tracks_path) as f:
        for line in f:
            if line.strip():
                tracks.append(json.loads(line))
    return tracks


def group_tracks_by_id(tracks: list) -> dict:
    """Group track observations by track_id."""
    by_id = defaultdict(list)
    for t in tracks:
        by_id[t['track_id']].append(t)
    return dict(by_id)


def get_track_tier(label: str) -> int:
    """Determine track tier (1=dynamic/Re-ID, 2=static, 3=suppress)."""
    label_lower = label.lower()
    
    # Check suppress first
    for suppress in SUPPRESS_CLASSES:
        if suppress in label_lower:
            return 3
    
    # Check dynamic
    for dynamic in DYNAMIC_CLASSES:
        if dynamic in label_lower or label_lower in dynamic:
            return 1
    
    # Check static
    for static in STATIC_CLASSES:
        if static in label_lower or label_lower in static:
            return 2
    
    # Default to tier 1 (Re-ID) for unknown classes
    return 1


def extract_track_crops(video_path: str, tracks_by_id: dict, output_dir: Path, 
                        console: Console, max_crops_per_track: int = 10):
    """Extract image crops for each track."""
    import cv2
    import numpy as np
    
    cap = cv2.VideoCapture(video_path)
    
    # Get unique frames we need
    frame_to_tracks = defaultdict(list)
    for track_id, observations in tracks_by_id.items():
        # Sample evenly if too many observations
        if len(observations) > max_crops_per_track:
            indices = np.linspace(0, len(observations) - 1, max_crops_per_track, dtype=int)
            sampled = [observations[i] for i in indices]
        else:
            sampled = observations
        
        for obs in sampled:
            frame_to_tracks[obs['frame_id']].append((track_id, obs))
    
    # Extract crops
    crops_by_track = defaultdict(list)
    frames_needed = sorted(frame_to_tracks.keys())
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Extracting crops...", total=len(frames_needed))
        
        for frame_id in frames_needed:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if not ret:
                progress.update(task, advance=1)
                continue
            
            for track_id, obs in frame_to_tracks[frame_id]:
                bbox = obs['bbox']
                x1, y1, x2, y2 = [int(b) for b in bbox]
                
                # Clamp to frame bounds
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                
                # Convert BGR to RGB
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crops_by_track[track_id].append({
                    'frame_id': frame_id,
                    'crop': crop_rgb,
                    'bbox': bbox
                })
            
            progress.update(task, advance=1)
    
    cap.release()
    return dict(crops_by_track)


def embed_tracks_vjepa2(crops_by_track: dict, device: str, console: Console):
    """Embed all tracks using V-JEPA2."""
    from orion.backends.vjepa2_backend import VJepa2Embedder
    
    console.print("\n  Loading V-JEPA2 model...")
    embedder = VJepa2Embedder(device=device)
    
    track_embeddings = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Embedding tracks...", total=len(crops_by_track))
        
        for track_id, crop_data in crops_by_track.items():
            crops = [c['crop'] for c in crop_data]
            
            try:
                # Use single-frame mode (middle frame) for speed
                embedding = embedder.embed_track_crops(crops, mode="single")
                track_embeddings[track_id] = {
                    'embedding': embedding.numpy().flatten(),
                    'num_crops': len(crops)
                }
            except Exception as e:
                logger.warning(f"Failed to embed track {track_id}: {e}")
            
            progress.update(task, advance=1)
    
    return track_embeddings


def cluster_tracks_by_similarity(track_embeddings: dict, tracks_by_id: dict, 
                                  similarity_threshold: float = 0.75):
    """Cluster tracks by cosine similarity within same class."""
    import numpy as np
    
    # Group tracks by class
    tracks_by_class = defaultdict(list)
    for track_id in track_embeddings.keys():
        if track_id in tracks_by_id:
            label = tracks_by_id[track_id][0]['label']
            tracks_by_class[label].append(track_id)
    
    # Cluster within each class
    unified_mapping = {}  # old_track_id -> unified_id
    next_unified_id = 1
    
    all_intra_sims = []
    all_inter_sims = []
    
    for label, track_ids in tracks_by_class.items():
        if len(track_ids) == 1:
            # Single track for this class
            unified_mapping[track_ids[0]] = next_unified_id
            next_unified_id += 1
            continue
        
        # Get embeddings matrix
        embs = np.array([track_embeddings[tid]['embedding'] for tid in track_ids])
        
        # Normalize for cosine similarity
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embs_normalized = embs / norms
        
        # Compute similarity matrix
        sim_matrix = embs_normalized @ embs_normalized.T
        
        # Simple greedy clustering
        assigned = set()
        clusters = []
        
        for i, tid_i in enumerate(track_ids):
            if tid_i in assigned:
                continue
            
            cluster = [tid_i]
            assigned.add(tid_i)
            
            for j, tid_j in enumerate(track_ids):
                if tid_j in assigned:
                    continue
                
                if sim_matrix[i, j] >= similarity_threshold:
                    cluster.append(tid_j)
                    assigned.add(tid_j)
                    all_intra_sims.append(sim_matrix[i, j])
            
            clusters.append(cluster)
        
        # Assign unified IDs
        for cluster in clusters:
            for tid in cluster:
                unified_mapping[tid] = next_unified_id
            next_unified_id += 1
        
        # Collect inter-cluster similarities
        for c1_idx, c1 in enumerate(clusters):
            for c2_idx, c2 in enumerate(clusters):
                if c1_idx >= c2_idx:
                    continue
                for tid1 in c1:
                    for tid2 in c2:
                        i = track_ids.index(tid1)
                        j = track_ids.index(tid2)
                        all_inter_sims.append(sim_matrix[i, j])
    
    avg_intra = np.mean(all_intra_sims) if all_intra_sims else 0.0
    avg_inter = np.mean(all_inter_sims) if all_inter_sims else 0.0
    
    return unified_mapping, avg_intra, avg_inter


def create_memory(unified_mapping: dict, track_embeddings: dict, 
                  tracks_by_id: dict, tracks: list) -> dict:
    """Create unified object memory."""
    import numpy as np
    
    # Group by unified ID
    by_unified = defaultdict(list)
    for track_id, unified_id in unified_mapping.items():
        if track_id in tracks_by_id:
            by_unified[unified_id].extend(tracks_by_id[track_id])
    
    memory = {
        'objects': [],
        'metadata': {
            'total_tracks_before': len(tracks_by_id),
            'tracks_embedded': len(track_embeddings),
            'total_objects_after': len(by_unified),
            'fragmentation_reduction': 0.0
        }
    }
    
    if len(tracks_by_id) > 0:
        memory['metadata']['fragmentation_reduction'] = 1.0 - (len(by_unified) / len(tracks_by_id))
    
    for unified_id, observations in by_unified.items():
        # Get representative embedding
        first_track_id = observations[0]['track_id']
        embedding = None
        if first_track_id in track_embeddings:
            embedding = track_embeddings[first_track_id]['embedding'].tolist()
        
        obj = {
            'object_id': unified_id,
            'label': observations[0]['label'],
            'first_seen_frame': min(obs['frame_id'] for obs in observations),
            'last_seen_frame': max(obs['frame_id'] for obs in observations),
            'total_observations': len(observations),
            'original_track_ids': list(set(obs['track_id'] for obs in observations)),
            'avg_confidence': np.mean([obs['confidence'] for obs in observations])
        }
        
        if embedding:
            obj['embedding'] = embedding
        
        memory['objects'].append(obj)
    
    # Sort by first appearance
    memory['objects'].sort(key=lambda x: x['first_seen_frame'])
    
    return memory


def handle_embed(args, settings) -> int:
    """
    Run Phase 2: V-JEPA2 Re-ID embedding and track clustering.
    
    Outputs:
        - results/<episode>/memory.json: Unified object memory with embeddings
        - results/<episode>/embed_stats.json: Run statistics
    """
    import numpy as np
    
    episode = args.episode
    results_dir = Path("results") / episode
    
    if not results_dir.exists():
        console.print(f"[red]✗ Results directory not found: {results_dir}[/red]")
        console.print("[yellow]  Run 'orion detect' first to generate tracks.[/yellow]")
        return 1
    
    tracks_path = results_dir / "tracks.jsonl"
    meta_path = results_dir / "episode_meta.json"
    
    if not tracks_path.exists():
        console.print(f"[red]✗ tracks.jsonl not found in {results_dir}[/red]")
        return 1
    
    if not meta_path.exists():
        console.print(f"[red]✗ episode_meta.json not found in {results_dir}[/red]")
        return 1
    
    # Load metadata
    with open(meta_path) as f:
        meta = json.load(f)
    
    video_path = meta['video_path']
    device = getattr(args, 'device', 'cuda')
    similarity = getattr(args, 'similarity', 0.75)
    
    console.print(f"""
[bold green]╔══════════════════════════════════════════════════════════════════╗[/bold green]
[bold green]║[/bold green]        [bold white]ORION V2 - PHASE 2: V-JEPA2 RE-ID[/bold white]                   [bold green]║[/bold green]
[bold green]╠══════════════════════════════════════════════════════════════════╣[/bold green]
[bold green]║[/bold green]  Episode:    [cyan]{episode:<51}[/cyan] [bold green]║[/bold green]
[bold green]║[/bold green]  Device:     [cyan]{device:<51}[/cyan] [bold green]║[/bold green]
[bold green]║[/bold green]  Similarity: [cyan]{similarity:<51}[/cyan] [bold green]║[/bold green]
[bold green]╚══════════════════════════════════════════════════════════════════╝[/bold green]
""")
    
    start_time = time.time()
    
    # Step 1: Load tracks
    console.print("\n[bold]Step 1:[/bold] Loading tracks...")
    tracks = load_tracks(tracks_path)
    tracks_by_id = group_tracks_by_id(tracks)
    
    console.print(f"  Loaded {len(tracks)} observations across {len(tracks_by_id)} tracks")
    
    # Step 2: Filter by tier
    console.print("\n[bold]Step 2:[/bold] Filtering tracks by tier...")
    tier_counts = {1: 0, 2: 0, 3: 0}
    tracks_for_reid = {}
    
    for track_id, observations in tracks_by_id.items():
        label = observations[0]['label']
        tier = get_track_tier(label)
        tier_counts[tier] += 1
        
        if tier == 1:  # Dynamic - do Re-ID
            tracks_for_reid[track_id] = observations
    
    console.print(f"  Tier 1 (Dynamic/Re-ID):  {tier_counts[1]} tracks")
    console.print(f"  Tier 2 (Static):         {tier_counts[2]} tracks")
    console.print(f"  Tier 3 (Suppressed):     {tier_counts[3]} tracks")
    
    # Step 3: Extract crops
    console.print("\n[bold]Step 3:[/bold] Extracting track crops...")
    crops_dir = results_dir / "track_crops"
    crops_by_track = extract_track_crops(
        video_path, tracks_for_reid, crops_dir, console
    )
    console.print(f"  Extracted crops for {len(crops_by_track)} tracks")
    
    # Step 4: Embed with V-JEPA2
    console.print("\n[bold]Step 4:[/bold] Embedding with V-JEPA2...")
    track_embeddings = embed_tracks_vjepa2(crops_by_track, device, console)
    console.print(f"  Embedded {len(track_embeddings)} tracks")
    
    # Step 5: Cluster
    console.print("\n[bold]Step 5:[/bold] Clustering by similarity...")
    unified_mapping, avg_intra, avg_inter = cluster_tracks_by_similarity(
        track_embeddings, tracks_for_reid, similarity
    )
    num_unified = len(set(unified_mapping.values()))
    console.print(f"  Clustered {len(unified_mapping)} tracks into {num_unified} objects")
    console.print(f"  Avg intra-cluster similarity: {avg_intra:.3f}")
    console.print(f"  Avg inter-cluster similarity: {avg_inter:.3f}")
    
    # Step 6: Create memory
    console.print("\n[bold]Step 6:[/bold] Creating unified memory...")
    memory = create_memory(
        unified_mapping, track_embeddings, tracks_for_reid, tracks
    )
    
    # Add static objects (tier 2) without Re-ID clustering
    for track_id, observations in tracks_by_id.items():
        label = observations[0]['label']
        tier = get_track_tier(label)
        
        if tier == 2:
            # Add as static object without embedding
            obj = {
                'object_id': max(o['object_id'] for o in memory['objects']) + 1 if memory['objects'] else 1,
                'label': label,
                'tier': 'static',
                'first_seen_frame': min(obs['frame_id'] for obs in observations),
                'last_seen_frame': max(obs['frame_id'] for obs in observations),
                'total_observations': len(observations),
                'original_track_ids': [track_id],
                'avg_confidence': np.mean([obs['confidence'] for obs in observations])
            }
            memory['objects'].append(obj)
    
    # Save memory
    memory_path = results_dir / "memory.json"
    with open(memory_path, 'w') as f:
        json.dump(memory, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    
    elapsed = time.time() - start_time
    
    # Save stats (convert to native Python types for JSON serialization)
    stats = EmbedStats(
        tracks_before=len(tracks_by_id),
        tracks_embedded=len(track_embeddings),
        objects_after=len(memory['objects']),
        fragmentation_reduction=float(memory['metadata']['fragmentation_reduction']),
        elapsed_seconds=float(round(elapsed, 2)),
        avg_intra_similarity=float(round(float(avg_intra), 4)),
        avg_inter_similarity=float(round(float(avg_inter), 4))
    )
    
    with open(results_dir / "embed_stats.json", 'w') as f:
        json.dump(asdict(stats), f, indent=2)
    
    # Print summary
    console.print(f"""
[bold green]═══════════════════════════════════════════════════════════════════[/bold green]
  [bold green]✓ PHASE 2 COMPLETE[/bold green]
[bold green]═══════════════════════════════════════════════════════════════════[/bold green]

  [bold]Results:[/bold]
    • Tracks before:           {stats.tracks_before}
    • Tracks embedded:         {stats.tracks_embedded}
    • Unified objects:         {stats.objects_after}
    • Fragmentation reduction: [green]{stats.fragmentation_reduction:.1%}[/green]
    • Processing time:         {stats.elapsed_seconds:.1f}s

  [bold]Re-ID Quality:[/bold]
    • Avg intra-cluster sim:   {stats.avg_intra_similarity:.3f}
    • Avg inter-cluster sim:   {stats.avg_inter_similarity:.3f}
    • Separation score:        {stats.avg_intra_similarity - stats.avg_inter_similarity:.3f}

  [bold]Output files:[/bold]
    • {memory_path}
    • {results_dir / "embed_stats.json"}

  [bold]Next step:[/bold]
    orion graph --episode {episode}
""")
    
    return 0
