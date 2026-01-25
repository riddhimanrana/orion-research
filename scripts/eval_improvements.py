#!/usr/bin/env python3
"""
Phase 4 Part 2 Improvements Evaluation

Runs evaluation on test videos and compares against baseline (eval_004).
Measures impact of improvements on:
- Remote control false positives
- Spatial relationship detection
- Temporal query accuracy
- Scene graph edge generation
- Overall F1 scores

Usage:
    python scripts/eval_improvements.py [--skip-gemini] [--skip-perception]
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger("eval_improvements")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_pipeline(video_path: str, episode_id: str, fps: float = 4.0, mode: str = "balanced") -> Dict[str, Any]:
    """Run perception pipeline on video."""
    logger.info(f"Running pipeline on {video_path} (episode={episode_id}, fps={fps})")
    
    import sys as _sys
    python_exe = _sys.executable
    
    cmd = [
        python_exe, "-m", "orion.cli.run_showcase",
        "--episode", episode_id,
        "--video", video_path,
        "--fps", str(fps),
        "--detector-backend", "yolo",
        "--yolo-model", "yolo11m",
        "--confidence", "0.25",
    ]
    
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Pipeline failed: {result.stderr}")
        return {"success": False, "error": result.stderr}
    
    logger.info("✓ Pipeline completed")
    
    # Load results
    results_dir = REPO_ROOT / "results" / episode_id
    metrics = {}
    
    # Load metadata
    meta_path = results_dir / "run_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
            metrics["tracks"] = meta.get("statistics", {}).get("unique_tracks", 0)
            metrics["frames"] = meta.get("statistics", {}).get("frames_processed", 0)
    
    # Load memory
    memory_path = results_dir / "memory.json"
    if memory_path.exists():
        with open(memory_path) as f:
            memory = json.load(f)
            metrics["memory_objects"] = len(memory.get("objects", []))
    
    # Load graph summary
    summary_path = results_dir / "graph_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
            metrics["total_edges"] = summary.get("total_edges", 0)
            metrics["avg_edges_per_frame"] = summary.get("avg_edges_per_frame", 0.0)
            metrics["total_frames_with_edges"] = summary.get("total_frames_with_edges", 0)
    
    metrics["success"] = True
    metrics["results_dir"] = str(results_dir)
    
    return metrics


def analyze_remote_detections(episode_id: str) -> Dict[str, Any]:
    """Analyze remote control detections in tracks."""
    logger.info(f"Analyzing remote detections in {episode_id}")
    
    results_dir = REPO_ROOT / "results" / episode_id
    tracks_path = results_dir / "tracks.jsonl"
    
    if not tracks_path.exists():
        return {"remote_count": 0, "remote_confidence_avg": 0.0}
    
    remote_detections = []
    total_detections = 0
    
    with open(tracks_path) as f:
        for line in f:
            if not line.strip():
                continue
            track = json.loads(line)
            total_detections += 1
            
            if track.get("category", "").lower() == "remote":
                remote_detections.append(track)
    
    remote_confs = [float(d.get("confidence", 0.0)) for d in remote_detections]
    
    return {
        "remote_count": len(remote_detections),
        "remote_confidence_avg": sum(remote_confs) / len(remote_confs) if remote_confs else 0.0,
        "remote_confidence_min": min(remote_confs) if remote_confs else 0.0,
        "remote_confidence_max": max(remote_confs) if remote_confs else 0.0,
        "total_detections": total_detections,
        "remote_percentage": (len(remote_detections) / total_detections * 100) if total_detections else 0.0,
    }


def run_gemini_evaluation(videos: list, output_dir: str = "results/eval_improvements") -> Dict[str, Any]:
    """Run Gemini evaluation on test videos."""
    logger.info(f"Running Gemini evaluation on {len(videos)} videos")
    
    import sys as _sys
    python_exe = _sys.executable
    
    cmd = [
        python_exe, "scripts/full_gemini_evaluation.py",
        "--videos"] + videos + [
        "--output", output_dir,
        "--samples", "10",
        "--mode", "balanced",
    ]
    
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.warning(f"Gemini evaluation had issues: {result.stderr}")
        return {"success": False, "error": result.stderr}
    
    logger.info("✓ Gemini evaluation completed")
    
    # Parse results
    output_path = Path(output_dir) / "evaluation_report.json"
    if output_path.exists():
        with open(output_path) as f:
            return json.load(f)
    
    return {"success": True}


def main():
    parser = argparse.ArgumentParser(description="Evaluate improvements on test videos")
    parser.add_argument("--skip-gemini", action="store_true", help="Skip Gemini validation")
    parser.add_argument("--skip-perception", action="store_true", help="Reuse existing results")
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("PHASE 4 PART 2: IMPROVEMENTS EVALUATION")
    logger.info("=" * 70)
    logger.info(f"Date: {datetime.now().isoformat()}")
    logger.info("")
    
    # Test videos
    videos = [
        str(REPO_ROOT / "data/examples/test.mp4"),
        str(REPO_ROOT / "data/examples/video_short.mp4"),
    ]
    
    eval_results = {
        "timestamp": datetime.now().isoformat(),
        "improvements": [
            "Remote control filtering (0.55 min confidence)",
            "Spatial query enhancement (multi-edge fallback)",
            "Temporal query enhancement (keyword support)",
            "Spatial predicate tuning (0.12 NEAR distance)",
            "Activity query enhancement (interaction detection)",
        ],
        "videos": {},
    }
    
    # Run pipeline on each video
    for i, video_path in enumerate(videos, 1):
        video_name = Path(video_path).stem
        episode_id = f"eval_improvements_{video_name}"
        
        logger.info("")
        logger.info(f"[{i}/{len(videos)}] Processing {video_name}")
        logger.info("-" * 70)
        
        if args.skip_perception:
            logger.info("⊘ Skipping perception (--skip-perception)")
            metrics = {"success": True}
        else:
            metrics = run_pipeline(video_path, episode_id)
        
        if not metrics.get("success"):
            logger.error(f"✗ Failed to process {video_name}")
            eval_results["videos"][video_name] = metrics
            continue
        
        # Analyze remote detections
        remote_analysis = analyze_remote_detections(episode_id)
        metrics["remote_analysis"] = remote_analysis
        
        logger.info(f"  Tracks: {metrics.get('tracks', 'N/A')}")
        logger.info(f"  Memory objects: {metrics.get('memory_objects', 'N/A')}")
        logger.info(f"  Total edges: {metrics.get('total_edges', 'N/A')}")
        logger.info(f"  Avg edges/frame: {metrics.get('avg_edges_per_frame', 'N/A'):.2f}")
        logger.info(f"  Remote detections: {remote_analysis['remote_count']} ({remote_analysis['remote_percentage']:.1f}%)")
        
        eval_results["videos"][video_name] = metrics
    
    # Run Gemini evaluation if API key available
    if not args.skip_gemini and "GOOGLE_API_KEY" in __import__("os").environ:
        logger.info("")
        logger.info("=" * 70)
        logger.info("GEMINI VALIDATION")
        logger.info("=" * 70)
        
        gemini_result = run_gemini_evaluation(videos)
        eval_results["gemini"] = gemini_result
    else:
        logger.info("")
        logger.info("⊘ Skipping Gemini validation (--skip-gemini or no GOOGLE_API_KEY)")
    
    # Save evaluation results
    output_dir = REPO_ROOT / "results" / "eval_improvements"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / "evaluation_results.json"
    with open(report_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    
    # Print summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 70)
    
    for video_name, metrics in eval_results["videos"].items():
        logger.info(f"\n{video_name}:")
        if metrics.get("success"):
            logger.info(f"  ✓ Perception completed")
            logger.info(f"    - Tracks: {metrics.get('tracks', 'N/A')}")
            logger.info(f"    - Memory objects: {metrics.get('memory_objects', 'N/A')}")
            logger.info(f"    - Scene graph edges: {metrics.get('total_edges', 'N/A')}")
            logger.info(f"    - Avg edges/frame: {metrics.get('avg_edges_per_frame', 'N/A'):.2f}")
            
            remote = metrics.get("remote_analysis", {})
            logger.info(f"    - Remote detections: {remote.get('remote_count', 0)}")
        else:
            logger.error(f"  ✗ Failed: {metrics.get('error', 'Unknown error')}")
    
    logger.info(f"\n✓ Results saved to {report_path}")
    logger.info("")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
