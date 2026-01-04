"""orion status - Show episode status"""

import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


def run_status(args) -> int:
    """Show episode status."""
    
    results_dir = Path("results")
    
    if args.episode:
        # Show specific episode
        episode_dir = results_dir / args.episode
        meta_path = episode_dir / "episode_meta.json"
        
        if not meta_path.exists():
            print(f"Episode not found: {args.episode}")
            return 1
        
        with open(meta_path) as f:
            meta = json.load(f)
        
        status = meta.get("status", {})
        video = meta.get("video", {})
        
        # Check file sizes
        def get_size(path):
            p = episode_dir / path
            if p.exists():
                size = p.stat().st_size
                if size > 1_000_000:
                    return f"{size / 1_000_000:.1f} MB"
                elif size > 1_000:
                    return f"{size / 1_000:.1f} KB"
                else:
                    return f"{size} B"
            return "—"
        
        # Count lines in JSONL
        def count_lines(path):
            p = episode_dir / path
            if p.exists():
                with open(p) as f:
                    return sum(1 for _ in f)
            return 0
        
        tracks_count = count_lines("tracks.jsonl")
        filtered_count = count_lines("tracks_filtered.jsonl")
        
        # Load memory object count
        memory_count = 0
        memory_path = episode_dir / "memory.json"
        if memory_path.exists():
            with open(memory_path) as f:
                mem = json.load(f)
                memory_count = len(mem.get("objects", []))
        
        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  EPISODE: {args.episode:<53} ║
╠══════════════════════════════════════════════════════════════════╣
║  Video                                                           ║
║    Path:       {Path(meta['video_path']).name:<48} ║
║    Duration:   {video.get('duration_sec', 0):.1f}s ({video.get('frame_count', 0)} frames @ {video.get('fps', 0):.1f} FPS){' ' * max(0, 20 - len(f"{video.get('duration_sec', 0):.1f}s"))}║
║    Resolution: {video.get('width', 0)}x{video.get('height', 0):<40} ║
╠══════════════════════════════════════════════════════════════════╣
║  Pipeline Status                                                 ║
║    [{'✓' if status.get('initialized') else ' '}] Initialized                                                   ║
║    [{'✓' if status.get('detected') else ' '}] Detected        ({tracks_count} track observations){' ' * max(0, 25 - len(str(tracks_count)))}║
║    [{'✓' if status.get('embedded') else ' '}] Embedded        ({memory_count} memory objects){' ' * max(0, 27 - len(str(memory_count)))}║
║    [{'✓' if status.get('filtered') else ' '}] Filtered        ({filtered_count} observations kept){' ' * max(0, 22 - len(str(filtered_count)))}║
║    [{'✓' if status.get('graphed') else ' '}] Graphed                                                      ║
║    [{'✓' if status.get('exported') else ' '}] Exported to Memgraph                                        ║
╠══════════════════════════════════════════════════════════════════╣
║  Files                                                           ║
║    tracks.jsonl:          {get_size('tracks.jsonl'):<36} ║
║    memory.json:           {get_size('memory.json'):<36} ║
║    tracks_filtered.jsonl: {get_size('tracks_filtered.jsonl'):<36} ║
║    scene_graph.jsonl:     {get_size('scene_graph.jsonl'):<36} ║
║    vlm_scene.jsonl:       {get_size('vlm_scene.jsonl'):<36} ║
╚══════════════════════════════════════════════════════════════════╝
""")
        
        if args.verbose:
            print("\n  Memory Objects:")
            if memory_path.exists():
                with open(memory_path) as f:
                    mem = json.load(f)
                for obj in mem.get("objects", [])[:10]:
                    print(f"    - {obj['id']}: {obj['canonical_label']} ({obj['total_observations']} obs)")
                if len(mem.get("objects", [])) > 10:
                    print(f"    ... and {len(mem['objects']) - 10} more")
        
        return 0
    
    else:
        # List all episodes
        print("\n  Available Episodes:")
        print("  " + "─" * 60)
        
        if not results_dir.exists():
            print("  No episodes found. Run: orion init --episode <name> --video <path>")
            return 0
        
        episodes = []
        for ep_dir in sorted(results_dir.iterdir()):
            if not ep_dir.is_dir():
                continue
            meta_path = ep_dir / "episode_meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                status = meta.get("status", {})
                stages = sum([
                    status.get("detected", False),
                    status.get("embedded", False),
                    status.get("filtered", False),
                    status.get("graphed", False),
                    status.get("exported", False)
                ])
                episodes.append((ep_dir.name, stages, meta.get("created_at", "")))
        
        if not episodes:
            print("  No episodes found. Run: orion init --episode <name> --video <path>")
            return 0
        
        for name, stages, created in episodes:
            status_bar = "█" * stages + "░" * (5 - stages)
            print(f"  {name:<30} [{status_bar}] {stages}/5 stages")
        
        print("\n  Use: orion status --episode <name> for details")
        return 0
