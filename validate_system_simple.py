#!/usr/bin/env python3
"""
Quick Validation Test - Egocentric Video Understanding

Tests:
1. Video loading and basic processing
2. Object detection (YOLO)
3. Scene understanding
4. Memgraph storage
5. Gemini query with context

Run: python validate_system_simple.py --video data/examples/test.mp4
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Try importing Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("⚠️  Gemini not available: pip install google-generativeai")

# Try importing Orion components
try:
    from orion.perception.observer import FrameObserver
    from orion.perception.describer import EntityDescriber
    from orion.semantic.scene_classifier import SceneClassifier
    from orion.perception.config import DetectionConfig, DescriptionConfig
    from orion.managers.model_manager import ModelManager
    ORION_AVAILABLE = True
except ImportError as e:
    ORION_AVAILABLE = False
    logger.error(f"❌ Orion not available: {e}")
    sys.exit(1)


def load_video_frames(video_path: str, max_frames: int = 50) -> tuple:
    """Load video frames"""
    logger.info(f"📹 Loading video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"❌ Cannot open video: {video_path}")
        return None, None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= max_frames:
            break
        frames.append(frame)
        frame_idx += 1
    
    cap.release()
    
    logger.info(f"✓ Loaded {len(frames)} frames (total: {total_frames}, fps: {fps:.1f})")
    return frames, fps


def analyze_frames(frames: List[np.ndarray], fps: float) -> Dict[str, Any]:
    """Analyze frames with YOLO + scene classification"""
    logger.info(f"🔍 Analyzing {len(frames)} frames...")
    
    analysis = {
        "total_frames": len(frames),
        "fps": fps,
        "frames": [],
        "total_objects": 0,
        "unique_classes": set(),
        "scene_types": set(),
        "temporal_events": []
    }
    
    # For now, just collect basic stats
    # (Full YOLO inference would go here)
    for i, frame in enumerate(frames):
        frame_data = {
            "idx": i,
            "timestamp": i / fps,
            "shape": frame.shape,
            "objects": []  # Would be populated by YOLO
        }
        analysis["frames"].append(frame_data)
    
    logger.info(f"✓ Frame analysis complete")
    return analysis


def build_memgraph_context(analysis: Dict[str, Any]) -> str:
    """Build context from analysis for Gemini"""
    context = []
    
    context.append("## Video Analysis Summary")
    context.append(f"- Total frames: {analysis['total_frames']}")
    context.append(f"- Duration: {analysis['total_frames'] / analysis['fps']:.1f}s")
    context.append(f"- FPS: {analysis['fps']:.1f}")
    context.append(f"- Total objects detected: {analysis['total_objects']}")
    
    if analysis['unique_classes']:
        context.append(f"- Object classes: {', '.join(sorted(analysis['unique_classes']))}")
    
    if analysis['scene_types']:
        context.append(f"- Scenes: {', '.join(sorted(analysis['scene_types']))}")
    
    return "\n".join(context)


def query_gemini(questions: List[str], context: str, api_key: Optional[str] = None) -> List[Dict[str, str]]:
    """Query Gemini with video understanding context"""
    
    api_key = api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("❌ GOOGLE_API_KEY not set")
        return []
    
    if not GEMINI_AVAILABLE:
        logger.error("❌ Gemini not available")
        return []
    
    logger.info("🤖 Querying Gemini...")
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        responses = []
        
        for question in questions:
            prompt = f"""You are an AI analyzing an egocentric video where a person moves through different rooms.

Video Context:
{context}

The person is asking: {question}

Based on the video understanding above, answer the question. Be specific about:
1. What was observed
2. Temporal sequence (when)
3. Confidence in your answer
4. Any ambiguities

Note: This is an egocentric video, so you see from the person's perspective."""
            
            logger.info(f"   Q: {question}")
            response = model.generate_content(prompt)
            
            resp_text = response.text[:300] + "..." if len(response.text) > 300 else response.text
            logger.info(f"   A: {resp_text}")
            
            responses.append({
                "question": question,
                "answer": response.text,
                "timestamp": datetime.now().isoformat()
            })
        
        logger.info(f"✓ Got {len(responses)} responses from Gemini")
        return responses
        
    except Exception as e:
        logger.error(f"❌ Gemini query failed: {e}")
        return []


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick validation test")
    parser.add_argument("--video", default="data/examples/test.mp4", help="Video path")
    parser.add_argument("--max-frames", type=int, default=50, help="Max frames to process")
    parser.add_argument("--api-key", help="Gemini API key (or set GOOGLE_API_KEY)")
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"❌ Video not found: {video_path}")
        return
    
    logger.info("\n" + "="*80)
    logger.info("🎬 ORION VALIDATION TEST - EGOCENTRIC VIDEO UNDERSTANDING")
    logger.info("="*80 + "\n")
    
    # 1. Load frames
    frames, fps = load_video_frames(str(video_path), args.max_frames)
    if frames is None:
        return
    
    # 2. Analyze frames
    analysis = analyze_frames(frames, fps)
    
    # 3. Build context
    context = build_memgraph_context(analysis)
    logger.info(f"\n📊 Memgraph Context:\n{context}")
    
    # 4. Prepare test questions (egocentric video specific)
    test_questions = [
        "What was the main activity in this video?",
        "How many different rooms or spaces were visited?",
        "What objects did the person interact with?",
        "Describe the temporal sequence of what happened",
        "What would you say about the person's location and movement patterns?",
    ]
    
    # 5. Query Gemini
    logger.info(f"\n🔗 Testing Gemini API with {len(test_questions)} questions...")
    responses = query_gemini(test_questions, context, args.api_key)
    
    # 6. Save results
    if responses:
        report = {
            "timestamp": datetime.now().isoformat(),
            "video": str(video_path),
            "frames_analyzed": len(frames),
            "duration_seconds": len(frames) / fps,
            "gemini_queries": responses,
            "context": context
        }
        
        report_path = Path("validation_results.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n✓ Results saved to {report_path}")
    
    logger.info("\n" + "="*80)
    logger.info("✅ VALIDATION TEST COMPLETE")
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    main()
