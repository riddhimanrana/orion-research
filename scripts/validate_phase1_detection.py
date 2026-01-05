#!/usr/bin/env python3
"""
Phase 1 Detection Validation with Gemini

Validates YOLO-World detection quality by:
1. Sampling frames and comparing detections to Gemini's analysis
2. Finding missed objects (false negatives)
3. Finding incorrect detections (false positives / wrong labels)
4. Evaluating object class confusion
5. Providing recommendations for vocabulary improvements

Usage:
    python scripts/validate_phase1_detection.py --episode <episode> --sample-frames 10
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
import google.generativeai as genai
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


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
    
    raise FileNotFoundError(f"No video found for episode {episode}")


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


def get_detections_for_frame(tracks: list[dict], frame_id: int) -> list[dict]:
    """Get all detections for a specific frame."""
    return [t for t in tracks if t['frame_id'] == frame_id]


def extract_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    """Extract a single frame from video."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Could not read frame {frame_idx}")
    return frame


def draw_detections_on_frame(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw bounding boxes and labels on frame."""
    frame = frame.copy()
    
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
    ]
    
    for i, det in enumerate(detections):
        color = colors[i % len(colors)]
        x1, y1, x2, y2 = map(int, det['bbox'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        label = f"{det['label']} ({det['confidence']:.2f})"
        font_scale = 0.5
        thickness = 1
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(frame, (x1, y1 - h - 5), (x1 + w + 2, y1), color, -1)
        cv2.putText(frame, label, (x1 + 1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, (255, 255, 255), thickness)
    
    return frame


def frame_to_base64(frame: np.ndarray) -> str:
    """Convert frame to base64 for Gemini."""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buffer).decode('utf-8')


def validate_frame_detections(
    model: genai.GenerativeModel,
    frame: np.ndarray,
    detections: list[dict],
    frame_id: int,
) -> dict:
    """
    Ask Gemini to validate detections for a frame.
    Returns analysis of correct, missed, and incorrect detections.
    """
    # Convert frame to base64 (WITHOUT bounding boxes for unbiased analysis)
    img_data = frame_to_base64(frame)
    
    # Build detection summary
    det_summary = []
    for d in detections:
        x1, y1, x2, y2 = map(int, d['bbox'])
        det_summary.append(f"- {d['label']} (conf={d['confidence']:.2f}) at bbox [{x1},{y1},{x2},{y2}]")
    
    det_text = "\n".join(det_summary) if det_summary else "(no detections)"
    
    prompt = f"""Analyze this video frame to evaluate object detection quality.

The AI detector (YOLO-World) found these objects in frame {frame_id}:
{det_text}

Your task:
1. List ALL objects YOU can see in the frame (be comprehensive)
2. For each detector result, classify as:
   - CORRECT: Object exists and label is accurate
   - WRONG_LABEL: Object exists but label is wrong (specify correct label)
   - FALSE_POSITIVE: No such object exists at that location
3. List objects YOU see that the detector MISSED (false negatives)
4. Identify any systematic issues (e.g., confusion between similar classes)

Respond in this exact JSON format:
{{
    "ground_truth_objects": ["list", "of", "all", "objects", "you", "see"],
    "detection_analysis": [
        {{"label": "detected_label", "bbox": [x1,y1,x2,y2], "verdict": "CORRECT|WRONG_LABEL|FALSE_POSITIVE", "correct_label": "if wrong_label, what should it be", "notes": "optional notes"}}
    ],
    "missed_objects": [
        {{"label": "object_type", "description": "where it is in the frame", "importance": "high|medium|low"}}
    ],
    "class_confusions": [
        {{"detected_as": "wrong_class", "should_be": "correct_class", "count": 1}}
    ],
    "detection_quality_score": 0.0-1.0,
    "recommendations": ["list of vocab/prompt improvements"]
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
        console.print(f"[red]Error analyzing frame {frame_id}: {e}[/red]")
        return {
            "ground_truth_objects": [],
            "detection_analysis": [],
            "missed_objects": [],
            "class_confusions": [],
            "detection_quality_score": 0.0,
            "recommendations": [f"Error: {str(e)}"]
        }


def analyze_vocabulary_coverage(
    model: genai.GenerativeModel,
    video_path: Path,
    current_vocab: list[str],
    sample_frames: int = 5
) -> dict:
    """
    Analyze if the current vocabulary covers objects in the video.
    """
    console.print("\n[bold cyan]Analyzing vocabulary coverage...[/bold cyan]")
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample evenly spaced frames
    frame_indices = [int(i * total_frames / (sample_frames + 1)) for i in range(1, sample_frames + 1)]
    
    frames_base64 = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames_base64.append(frame_to_base64(frame))
    cap.release()
    
    prompt = f"""Analyze these video frames to evaluate vocabulary coverage for object detection.

Current detector vocabulary ({len(current_vocab)} classes):
{', '.join(current_vocab[:50])}... (and {len(current_vocab)-50} more)

Your task:
1. List ALL distinct object types you see across the frames
2. Identify objects NOT covered by the current vocabulary
3. Suggest vocabulary additions for better coverage
4. Identify ambiguous or overlapping classes in the vocabulary

Respond in JSON format:
{{
    "all_objects_seen": ["comprehensive list of all object types"],
    "covered_by_vocab": ["objects that match vocabulary"],
    "not_covered": ["objects missing from vocabulary"],
    "suggested_additions": ["new classes to add"],
    "ambiguous_classes": [
        {{"classes": ["class1", "class2"], "issue": "why they're ambiguous"}}
    ],
    "vocabulary_quality_score": 0.0-1.0,
    "improved_vocabulary": ["optimized vocabulary list"]
}}"""

    try:
        content = []
        for img_data in frames_base64:
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
            "all_objects_seen": [],
            "covered_by_vocab": [],
            "not_covered": [],
            "suggested_additions": [],
            "ambiguous_classes": [],
            "vocabulary_quality_score": 0.0,
            "improved_vocabulary": [],
            "error": str(e)
        }


def generate_improved_vocabulary(
    model: genai.GenerativeModel,
    current_vocab: list[str],
    validation_results: list[dict],
    vocab_analysis: dict
) -> list[str]:
    """
    Generate an improved vocabulary based on validation results.
    """
    console.print("\n[bold cyan]Generating improved vocabulary...[/bold cyan]")
    
    # Collect all recommendations and missed objects
    all_missed = []
    all_confusions = []
    all_recommendations = []
    
    for result in validation_results:
        all_missed.extend(result.get('missed_objects', []))
        all_confusions.extend(result.get('class_confusions', []))
        all_recommendations.extend(result.get('recommendations', []))
    
    prompt = f"""Based on detection validation results, optimize the YOLO-World vocabulary.

Current vocabulary ({len(current_vocab)} classes):
{', '.join(current_vocab)}

Issues found:
1. Missed objects: {json.dumps(all_missed[:10])}
2. Class confusions: {json.dumps(all_confusions[:10])}
3. Recommendations: {all_recommendations[:10]}
4. Vocab analysis suggestions: {vocab_analysis.get('suggested_additions', [])}
5. Not covered: {vocab_analysis.get('not_covered', [])}

Generate an IMPROVED vocabulary that:
1. Adds missing object types
2. Uses more specific class names to reduce confusion
3. Removes redundant/overlapping classes
4. Keeps total classes under 80 for efficiency
5. Prioritizes indoor/home objects for this video type

Respond in JSON format:
{{
    "improved_vocabulary": ["list", "of", "optimized", "class", "names"],
    "changes_made": [
        {{"action": "added|removed|renamed", "original": "old", "new": "new", "reason": "why"}}
    ],
    "expected_improvements": "What should improve with this vocabulary"
}}"""

    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(text)
        return result.get('improved_vocabulary', current_vocab)
    
    except Exception as e:
        console.print(f"[yellow]Could not generate improved vocabulary: {e}[/yellow]")
        return current_vocab


def main():
    parser = argparse.ArgumentParser(description="Validate Phase 1 detections with Gemini")
    parser.add_argument("--episode", required=True, help="Episode name")
    parser.add_argument("--sample-frames", type=int, default=10, help="Number of frames to validate")
    parser.add_argument("--analyze-vocab", action="store_true", help="Also analyze vocabulary coverage")
    parser.add_argument("--generate-improved", action="store_true", help="Generate improved vocabulary")
    parser.add_argument("--output", type=str, help="Output JSON file")
    args = parser.parse_args()
    
    # Configure Gemini
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        console.print("[red]Error: GOOGLE_API_KEY or GEMINI_API_KEY not set[/red]")
        sys.exit(1)
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    # Load data
    results_dir = Path("results") / args.episode
    console.print(Panel(f"[bold]Phase 1 Detection Validation: {args.episode}[/bold]"))
    
    try:
        tracks = load_tracks(results_dir)
        video_path = get_video_path(args.episode)
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    
    # Get unique frames with detections
    frames_with_detections = sorted(set(t['frame_id'] for t in tracks))
    console.print(f"\nLoaded {len(tracks)} detections across {len(frames_with_detections)} frames")
    console.print(f"Video: {video_path}")
    
    # Sample frames evenly
    if len(frames_with_detections) > args.sample_frames:
        step = len(frames_with_detections) // args.sample_frames
        sampled_frames = [frames_with_detections[i * step] for i in range(args.sample_frames)]
    else:
        sampled_frames = frames_with_detections
    
    console.print(f"Validating {len(sampled_frames)} frames...")
    
    # Get current vocabulary (inline to avoid import issues)
    current_vocab = [
        # People
        "person", "face", "hand",
        # Furniture
        "chair", "table", "desk", "couch", "sofa", "bed", "cabinet", "shelf", "drawer",
        "ottoman", "stool", "bench", "nightstand", "dresser", "wardrobe",
        # Soft furnishings
        "pillow", "blanket", "cushion", "rug", "carpet", "curtain", "mat",
        # Electronics
        "laptop", "phone", "cellphone", "tv", "television", "monitor", "screen", 
        "keyboard", "mouse", "remote", "controller", "camera",
        "speaker", "headphones", "charger", "appliance",
        # Kitchen
        "cup", "mug", "glass", "bottle", "plate", "bowl", "fork", "spoon", "knife",
        "pan", "pot", "microwave", "refrigerator", "sink", "faucet", "oven", "stove",
        "toaster", "blender", "coffee maker", "kettle", "dishwasher",
        # Food
        "food", "fruit", "vegetable", "bread", "pizza", "sandwich",
        # Tools/Items
        "book", "notebook", "pen", "pencil", "paper", "document",
        "bag", "backpack", "purse", "wallet", "key", "keys",
        "box", "container", "package", "tool", "toy", "toys",
        # Lighting
        "lamp", "light", "chandelier", "candle",
        # Decor
        "plant", "vase", "picture", "frame", "painting", "clock", "mirror", 
        # Structure
        "window", "door", "wall", "floor", "ceiling", "stairs",
    ]
    
    # Validate each frame
    results = {
        "episode": args.episode,
        "timestamp": datetime.now().isoformat(),
        "total_detections": len(tracks),
        "frames_validated": len(sampled_frames),
        "frame_results": [],
        "summary": {},
        "vocabulary_analysis": {},
        "improved_vocabulary": []
    }
    
    all_verdicts = defaultdict(int)
    all_missed = []
    all_confusions = []
    quality_scores = []
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task = progress.add_task("Validating frames...", total=len(sampled_frames))
        
        for frame_id in sampled_frames:
            detections = get_detections_for_frame(tracks, frame_id)
            frame = extract_frame(video_path, frame_id)
            
            result = validate_frame_detections(model, frame, detections, frame_id)
            result['frame_id'] = frame_id
            result['num_detections'] = len(detections)
            results['frame_results'].append(result)
            
            # Aggregate stats
            for analysis in result.get('detection_analysis', []):
                all_verdicts[analysis.get('verdict', 'UNKNOWN')] += 1
            
            all_missed.extend(result.get('missed_objects', []))
            all_confusions.extend(result.get('class_confusions', []))
            
            if result.get('detection_quality_score'):
                quality_scores.append(result['detection_quality_score'])
            
            progress.update(task, advance=1)
    
    # Compute summary
    total_analyzed = sum(all_verdicts.values())
    results['summary'] = {
        "correct_detections": all_verdicts.get('CORRECT', 0),
        "wrong_labels": all_verdicts.get('WRONG_LABEL', 0),
        "false_positives": all_verdicts.get('FALSE_POSITIVE', 0),
        "total_analyzed": total_analyzed,
        "precision": all_verdicts.get('CORRECT', 0) / max(1, total_analyzed),
        "avg_quality_score": np.mean(quality_scores) if quality_scores else 0,
        "total_missed_objects": len(all_missed),
        "total_class_confusions": len(all_confusions),
    }
    
    # Print summary
    console.print("\n[bold green]Detection Validation Summary:[/bold green]")
    table = Table(title="Detection Quality")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Correct Detections", str(results['summary']['correct_detections']))
    table.add_row("Wrong Labels", str(results['summary']['wrong_labels']))
    table.add_row("False Positives", str(results['summary']['false_positives']))
    table.add_row("Precision", f"{results['summary']['precision']:.1%}")
    table.add_row("Avg Quality Score", f"{results['summary']['avg_quality_score']:.2f}")
    table.add_row("Missed Objects", str(results['summary']['total_missed_objects']))
    console.print(table)
    
    # Show most common missed objects
    if all_missed:
        missed_counts = defaultdict(int)
        for m in all_missed:
            missed_counts[m.get('label', 'unknown')] += 1
        
        console.print("\n[yellow]Most commonly missed objects:[/yellow]")
        for label, count in sorted(missed_counts.items(), key=lambda x: -x[1])[:10]:
            console.print(f"  • {label}: {count}")
    
    # Show class confusions
    if all_confusions:
        console.print("\n[yellow]Class confusions:[/yellow]")
        for conf in all_confusions[:10]:
            console.print(f"  • {conf.get('detected_as')} → should be {conf.get('should_be')}")
    
    # Vocabulary analysis
    if args.analyze_vocab:
        vocab_analysis = analyze_vocabulary_coverage(model, video_path, current_vocab)
        results['vocabulary_analysis'] = vocab_analysis
        
        console.print("\n[bold cyan]Vocabulary Analysis:[/bold cyan]")
        console.print(f"  Quality Score: {vocab_analysis.get('vocabulary_quality_score', 0):.2f}")
        
        if vocab_analysis.get('not_covered'):
            console.print(f"\n[yellow]Objects not in vocabulary:[/yellow]")
            for obj in vocab_analysis['not_covered'][:15]:
                console.print(f"  • {obj}")
        
        if vocab_analysis.get('suggested_additions'):
            console.print(f"\n[green]Suggested additions:[/green]")
            for obj in vocab_analysis['suggested_additions'][:15]:
                console.print(f"  + {obj}")
    
    # Generate improved vocabulary
    if args.generate_improved:
        improved_vocab = generate_improved_vocabulary(
            model, current_vocab, results['frame_results'],
            results.get('vocabulary_analysis', {})
        )
        results['improved_vocabulary'] = improved_vocab
        
        console.print(f"\n[green]Generated improved vocabulary ({len(improved_vocab)} classes)[/green]")
    
    # Save results
    output_path = args.output or (results_dir / "phase1_validation.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    console.print(f"\n[green]Results saved to: {output_path}[/green]")
    
    # Print actionable recommendations
    console.print(Panel("[bold]Recommendations for Phase 1 Improvement[/bold]"))
    
    all_recommendations = []
    for result in results['frame_results']:
        all_recommendations.extend(result.get('recommendations', []))
    
    # Deduplicate recommendations
    unique_recs = list(set(all_recommendations))[:10]
    for rec in unique_recs:
        console.print(f"  • {rec}")


if __name__ == "__main__":
    main()
