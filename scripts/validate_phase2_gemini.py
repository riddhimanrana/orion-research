#!/usr/bin/env python3
"""
Phase 2 Semantic Validation with Gemini Vision

Validates V-JEPA2 clustering quality by:
1. Sampling frames showing merged objects
2. Asking Gemini to verify if merges are semantically correct
3. Identifying false merges (different objects merged as one)
4. Identifying missed merges (same object left fragmented)
5. Computing semantic correctness metrics

Usage:
    python scripts/validate_phase2_gemini.py --episode <episode_name> [--sample-size 10]
"""

import argparse
import base64
import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from orion.utils.gemini_client import GeminiClientError, get_gemini_model

console = Console()


def load_memory(results_dir: Path) -> dict:
    """Load memory.json from Phase 2."""
    memory_path = results_dir / "memory.json"
    if not memory_path.exists():
        raise FileNotFoundError(f"No memory.json found at {memory_path}")
    
    with open(memory_path) as f:
        return json.load(f)


def load_tracks(results_dir: Path) -> list[dict]:
    """Load tracks.jsonl from Phase 1."""
    tracks_path = results_dir / "tracks.jsonl"
    if not tracks_path.exists():
        raise FileNotFoundError(f"No tracks.jsonl found at {tracks_path}")
    
    tracks = []
    with open(tracks_path) as f:
        for line in f:
            tracks.append(json.loads(line))
    return tracks


def get_video_path(episode: str) -> Path:
    """Find video for episode."""
    results_dir = Path("results") / episode
    
    # Check episode_meta.json first (from Phase 1)
    episode_meta_path = results_dir / "episode_meta.json"
    if episode_meta_path.exists():
        with open(episode_meta_path) as f:
            meta = json.load(f)
            if "video_path" in meta:
                return Path(meta["video_path"])
    
    # Check common locations
    episode_dir = Path("data/examples/episodes") / episode
    if (episode_dir / "video.mp4").exists():
        return episode_dir / "video.mp4"
    
    if (results_dir / "video.mp4").exists():
        return results_dir / "video.mp4"
    
    # Check meta.json for video_path
    meta_path = episode_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
            if "video_path" in meta:
                return Path(meta["video_path"])
    
    raise FileNotFoundError(f"No video found for episode {episode}")


def extract_frame_with_boxes(video_path: Path, frame_idx: int, boxes: list[dict]) -> np.ndarray:
    """Extract frame and draw bounding boxes with labels."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Could not read frame {frame_idx}")
    
    # Color palette for different objects
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 255),  # Orange
        (255, 128, 0),  # Light Blue
    ]
    
    for i, box in enumerate(boxes):
        color = colors[i % len(colors)]
        x1, y1, x2, y2 = map(int, box['bbox'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Label with object info
        label = f"{box['label']} (T{box['track_id']}→O{box['object_id']})"
        font_scale = 0.7
        thickness = 2
        (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # Background for text
        cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w + 5, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, (255, 255, 255), thickness)
    
    return frame


def frame_to_base64(frame: np.ndarray) -> str:
    """Convert frame to base64 for Gemini."""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buffer).decode('utf-8')


def sample_merged_objects(memory: dict, tracks: list[dict], sample_size: int = 10) -> list[dict]:
    """
    Sample objects that were merged from multiple tracks.
    These are the most interesting cases to validate.
    """
    # Group tracks by frame
    tracks_by_frame = defaultdict(list)
    for track in tracks:
        tracks_by_frame[track['frame_id']].append(track)
    
    # Find objects with multiple merged tracks
    merged_objects = [
        obj for obj in memory['objects']
        if len(obj.get('original_track_ids', [])) > 1
    ]
    
    if not merged_objects:
        console.print("[yellow]No merged objects found (all tracks kept separate)")
        return []
    
    # Sample up to sample_size merged objects
    sampled = random.sample(merged_objects, min(sample_size, len(merged_objects)))
    
    samples = []
    for obj in sampled:
        # Find a frame where multiple merged tracks appear
        track_ids = set(obj['original_track_ids'])
        
        # Find frames with tracks from this merged object
        frames_with_tracks = defaultdict(list)
        for frame_id, frame_tracks in tracks_by_frame.items():
            for t in frame_tracks:
                if t['track_id'] in track_ids:
                    frames_with_tracks[frame_id].append(t)
        
        # Find frame with most tracks from this object
        best_frame = max(frames_with_tracks.keys(), key=lambda f: len(frames_with_tracks[f]))
        
        samples.append({
            'object': obj,
            'frame_id': best_frame,
            'tracks_in_frame': frames_with_tracks[best_frame],
            'all_track_ids': list(track_ids),
        })
    
    return samples


def sample_same_class_pairs(memory: dict, tracks: list[dict], sample_size: int = 10) -> list[dict]:
    """
    Sample pairs of objects with the same class that were NOT merged.
    These could be missed merges (false negatives).
    """
    # Group objects by label
    objects_by_label = defaultdict(list)
    for obj in memory['objects']:
        objects_by_label[obj['label']].append(obj)
    
    # Group tracks by frame
    tracks_by_frame = defaultdict(list)
    for track in tracks:
        tracks_by_frame[track['frame_id']].append(track)
    
    pairs = []
    for label, objs in objects_by_label.items():
        if len(objs) < 2:
            continue
        
        # Sample pairs of different objects with same label
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                obj1, obj2 = objs[i], objs[j]
                
                # Find a frame where both objects appear (via their tracks)
                track_ids_1 = set(obj1.get('original_track_ids', [obj1.get('object_id')]))
                track_ids_2 = set(obj2.get('original_track_ids', [obj2.get('object_id')]))
                
                for frame_id, frame_tracks in tracks_by_frame.items():
                    frame_track_ids = {t['track_id'] for t in frame_tracks}
                    if frame_track_ids & track_ids_1 and frame_track_ids & track_ids_2:
                        # Found a frame with both objects
                        pairs.append({
                            'object1': obj1,
                            'object2': obj2,
                            'frame_id': frame_id,
                            'tracks_in_frame': [
                                t for t in frame_tracks 
                                if t['track_id'] in track_ids_1 or t['track_id'] in track_ids_2
                            ]
                        })
                        break
    
    # Sample
    if pairs:
        return random.sample(pairs, min(sample_size, len(pairs)))
    return []


def validate_merge_with_gemini(
    model,
    frame: np.ndarray,
    tracks_info: list[dict],
    object_info: dict,
) -> dict:
    """
    Ask Gemini to validate if merged tracks are the same object.
    """
    # Convert frame to base64
    img_data = frame_to_base64(frame)
    
    # Build prompt
    track_desc = ", ".join([f"Track {t['track_id']} (box {t['bbox']})" for t in tracks_info])
    
    prompt = f"""Analyze this video frame showing tracked objects.

The following tracks have been MERGED into a single unified object (Object ID {object_info['object_id']}, label: "{object_info['label']}"):
{track_desc}

Each track is highlighted with a different colored bounding box in the image.

Please evaluate:
1. Do all the highlighted bounding boxes show the SAME physical object?
2. Is this merge CORRECT (same real-world object) or INCORRECT (different objects merged)?

Respond in this exact JSON format:
{{
    "is_same_object": true/false,
    "merge_correct": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation",
    "object_description": "What the object actually is"
}}"""

    try:
        response = model.generate_content([
            {"mime_type": "image/jpeg", "data": img_data},
            prompt
        ])
        
        # Parse JSON from response
        text = response.text.strip()
        # Handle markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        return json.loads(text)
    
    except Exception as e:
        return {
            "is_same_object": None,
            "merge_correct": None,
            "confidence": 0.0,
            "reasoning": f"Error: {str(e)}",
            "object_description": "Unknown"
        }


def validate_separation_with_gemini(
    model,
    frame: np.ndarray,
    tracks_info: list[dict],
    obj1: dict,
    obj2: dict,
) -> dict:
    """
    Ask Gemini if two SEPARATE objects are actually the same (missed merge).
    """
    img_data = frame_to_base64(frame)
    
    prompt = f"""Analyze this video frame showing tracked objects.

Two separate objects have been identified:
- Object {obj1['object_id']} (label: "{obj1['label']}"): {len(obj1.get('original_track_ids', []))} tracks merged
- Object {obj2['object_id']} (label: "{obj2['label']}"): {len(obj2.get('original_track_ids', []))} tracks merged

These objects were kept SEPARATE (not merged together).

The bounding boxes for tracks from both objects are highlighted in the image.

Please evaluate:
1. Are these actually the SAME physical object that should have been merged?
2. Or are they correctly identified as DIFFERENT objects?

Respond in this exact JSON format:
{{
    "are_same_object": true/false,
    "separation_correct": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation",
    "object1_description": "What object 1 is",
    "object2_description": "What object 2 is"
}}"""

    try:
        response = model.generate_content([
            {"mime_type": "image/jpeg", "data": img_data},
            prompt
        ])
        
        text = response.text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        return json.loads(text)
    
    except Exception as e:
        return {
            "are_same_object": None,
            "separation_correct": None,
            "confidence": 0.0,
            "reasoning": f"Error: {str(e)}",
            "object1_description": "Unknown",
            "object2_description": "Unknown"
        }


def run_full_scene_analysis(
    model,
    video_path: Path,
    memory: dict,
) -> dict:
    """
    Ask Gemini to analyze the full video and estimate object counts.
    This provides ground truth for comparison.
    """
    console.print("\n[bold cyan]Running full scene analysis...[/bold cyan]")
    
    # Sample several frames from the video
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample 5 evenly spaced frames
    sample_frames = [int(i * total_frames / 6) for i in range(1, 6)]
    frames_base64 = []
    
    for frame_idx in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frames_base64.append(frame_to_base64(frame))
    
    cap.release()
    
    # Build comprehensive prompt
    prompt = f"""Analyze these 5 frames from a video to count unique objects.

The video has been processed by an AI system that detected:
- Total tracks: {memory['metadata'].get('total_tracks', 'unknown')}
- Unified objects: {len(memory['objects'])}

Object breakdown by label:
"""
    
    # Add label counts
    label_counts = defaultdict(int)
    for obj in memory['objects']:
        label_counts[obj['label']] += 1
    
    for label, count in sorted(label_counts.items()):
        prompt += f"  - {label}: {count}\n"
    
    prompt += """
Your task:
1. Count the TRUE number of unique physical objects you can see across all frames
2. For each object type, estimate how many distinct instances exist
3. Compare to the AI's counts above - are they over-counting (fragmented) or under-counting (merged)?

Respond in this exact JSON format:
{
    "scene_description": "Brief description of the scene",
    "ground_truth_counts": {
        "person": 2,
        "laptop": 1,
        ...
    },
    "total_unique_objects": 15,
    "ai_accuracy_assessment": "good/over-counting/under-counting",
    "specific_issues": [
        "person count seems too high - likely fragmented tracks",
        ...
    ],
    "recommendations": "Suggestions for improvement"
}"""

    try:
        # Build content with all frames
        content = []
        for i, img_data in enumerate(frames_base64):
            content.append({"mime_type": "image/jpeg", "data": img_data})
        content.append(prompt)
        
        response = model.generate_content(content)
        text = response.text.strip()
        
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        return json.loads(text)
    
    except Exception as e:
        return {
            "scene_description": "Unknown",
            "ground_truth_counts": {},
            "total_unique_objects": -1,
            "ai_accuracy_assessment": f"Error: {str(e)}",
            "specific_issues": [],
            "recommendations": ""
        }


def main():
    parser = argparse.ArgumentParser(description="Validate Phase 2 with Gemini Vision")
    parser.add_argument("--episode", required=True, help="Episode name")
    parser.add_argument("--sample-size", type=int, default=5, help="Number of merges to validate")
    parser.add_argument("--skip-merge-validation", action="store_true", help="Skip individual merge checks")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    args = parser.parse_args()
    
    # Configure Gemini
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        console.print("[red]Error: GOOGLE_API_KEY or GEMINI_API_KEY not set[/red]")
        sys.exit(1)
    
    try:
        model = get_gemini_model("gemini-2.0-flash", api_key=api_key)
    except GeminiClientError as exc:
        console.print(f"[red]Error: {exc}[/red]")
        sys.exit(1)
    
    # Load data
    results_dir = Path("results") / args.episode
    console.print(Panel(f"[bold]Phase 2 Semantic Validation: {args.episode}[/bold]"))
    
    try:
        memory = load_memory(results_dir)
        tracks = load_tracks(results_dir)
        video_path = get_video_path(args.episode)
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    
    # Summary
    console.print(f"\n[cyan]Memory loaded:[/cyan]")
    console.print(f"  Objects: {len(memory['objects'])}")
    console.print(f"  Tracks (raw): {memory['metadata'].get('total_tracks', len(tracks))}")
    console.print(f"  Video: {video_path}")
    
    results = {
        "episode": args.episode,
        "timestamp": datetime.now().isoformat(),
        "memory_summary": {
            "total_objects": len(memory['objects']),
            "total_tracks": memory['metadata'].get('total_tracks', 0),
        },
        "scene_analysis": {},
        "merge_validations": [],
        "separation_validations": [],
        "metrics": {},
    }
    
    # Step 1: Full scene analysis (ground truth)
    scene_analysis = run_full_scene_analysis(model, video_path, memory)
    results["scene_analysis"] = scene_analysis
    
    console.print("\n[bold green]Scene Analysis Results:[/bold green]")
    console.print(f"  Description: {scene_analysis.get('scene_description', 'N/A')}")
    console.print(f"  AI Assessment: {scene_analysis.get('ai_accuracy_assessment', 'N/A')}")
    
    if scene_analysis.get('ground_truth_counts'):
        table = Table(title="Ground Truth vs AI Counts")
        table.add_column("Label", style="cyan")
        table.add_column("Ground Truth", style="green")
        table.add_column("AI Count", style="yellow")
        table.add_column("Diff", style="red")
        
        label_counts = defaultdict(int)
        for obj in memory['objects']:
            label_counts[obj['label']] += 1
        
        for label, gt_count in scene_analysis['ground_truth_counts'].items():
            ai_count = label_counts.get(label, 0)
            diff = ai_count - gt_count
            diff_str = f"+{diff}" if diff > 0 else str(diff)
            table.add_row(label, str(gt_count), str(ai_count), diff_str)
        
        console.print(table)
    
    if scene_analysis.get('specific_issues'):
        console.print("\n[yellow]Issues identified:[/yellow]")
        for issue in scene_analysis['specific_issues']:
            console.print(f"  • {issue}")
    
    # Step 2: Validate merged objects (if not skipped)
    if not args.skip_merge_validation:
        console.print("\n[bold cyan]Validating merged objects...[/bold cyan]")
        
        merged_samples = sample_merged_objects(memory, tracks, args.sample_size)
        
        correct_merges = 0
        incorrect_merges = 0
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task("Validating merges...", total=len(merged_samples))
            
            for sample in merged_samples:
                obj = sample['object']
                frame = extract_frame_with_boxes(
                    video_path, 
                    sample['frame_id'], 
                    [{
                        'bbox': t['bbox'],
                        'label': t['label'],
                        'track_id': t['track_id'],
                        'object_id': obj['object_id']
                    } for t in sample['tracks_in_frame']]
                )
                
                result = validate_merge_with_gemini(
                    model, frame, sample['tracks_in_frame'], obj
                )
                
                results["merge_validations"].append({
                    "object_id": obj['object_id'],
                    "label": obj['label'],
                    "track_ids": sample['all_track_ids'],
                    "frame_id": sample['frame_id'],
                    "validation": result
                })
                
                if result.get('merge_correct'):
                    correct_merges += 1
                elif result.get('merge_correct') is False:
                    incorrect_merges += 1
                
                progress.update(task, advance=1)
        
        console.print(f"\n[green]Correct merges: {correct_merges}[/green]")
        console.print(f"[red]Incorrect merges: {incorrect_merges}[/red]")
        
        results["metrics"]["correct_merges"] = correct_merges
        results["metrics"]["incorrect_merges"] = incorrect_merges
        results["metrics"]["merge_accuracy"] = correct_merges / max(1, correct_merges + incorrect_merges)
    
    # Step 3: Validate same-class separations (check for missed merges)
    console.print("\n[bold cyan]Checking for missed merges...[/bold cyan]")
    
    separation_samples = sample_same_class_pairs(memory, tracks, args.sample_size)
    
    correct_separations = 0
    missed_merges = 0
    
    if separation_samples:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task("Checking separations...", total=len(separation_samples))
            
            for sample in separation_samples:
                frame = extract_frame_with_boxes(
                    video_path,
                    sample['frame_id'],
                    [{
                        'bbox': t['bbox'],
                        'label': t['label'],
                        'track_id': t['track_id'],
                        'object_id': sample['object1']['object_id'] if t['track_id'] in sample['object1'].get('original_track_ids', []) else sample['object2']['object_id']
                    } for t in sample['tracks_in_frame']]
                )
                
                result = validate_separation_with_gemini(
                    model, frame, sample['tracks_in_frame'],
                    sample['object1'], sample['object2']
                )
                
                results["separation_validations"].append({
                    "object1_id": sample['object1']['object_id'],
                    "object2_id": sample['object2']['object_id'],
                    "label": sample['object1']['label'],
                    "frame_id": sample['frame_id'],
                    "validation": result
                })
                
                if result.get('separation_correct'):
                    correct_separations += 1
                elif result.get('separation_correct') is False:
                    missed_merges += 1
                
                progress.update(task, advance=1)
        
        console.print(f"\n[green]Correct separations: {correct_separations}[/green]")
        console.print(f"[yellow]Missed merges: {missed_merges}[/yellow]")
        
        results["metrics"]["correct_separations"] = correct_separations
        results["metrics"]["missed_merges"] = missed_merges
    else:
        console.print("  No same-class object pairs found to validate")
    
    # Final summary
    console.print(Panel("[bold]Validation Complete[/bold]"))
    
    # Save results
    output_path = args.output or (results_dir / "phase2_validation.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    console.print(f"\n[green]Results saved to: {output_path}[/green]")
    
    # Print actionable recommendations
    if scene_analysis.get('recommendations'):
        console.print(f"\n[bold yellow]Recommendations:[/bold yellow]")
        console.print(f"  {scene_analysis['recommendations']}")


if __name__ == "__main__":
    main()
