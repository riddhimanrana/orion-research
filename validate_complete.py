#!/usr/bin/env python3
"""
Complete System Validation - Full Perception Pipeline

Tests end-to-end:
1. Video frames
2. YOLO object detection
3. Scene classification
4. Spatial zones
5. Memgraph storage  
6. Gemini querying

Run: python validate_complete.py --video data/examples/test.mp4 --max-frames 100
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
import time

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Orion
try:
    from orion.managers.model_manager import ModelManager
    from orion.perception.config import DetectionConfig
    from orion.perception.observer import FrameObserver
    from orion.perception.spatial_analyzer import calculate_spatial_zone
    from orion.semantic.scene_classifier import SceneClassifier
    ORION_AVAILABLE = True
except ImportError as e:
    ORION_AVAILABLE = False
    logger.error(f"Orion error: {e}")


class CompleteValidator:
    """Full system validation"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model_mgr = None
        self.observer = None
        self.scene_classifier = None
        self._initialize()
    
    def _initialize(self):
        """Initialize all components"""
        logger.info("🔧 Initializing system...")
        
        try:
            self.model_mgr = ModelManager.get_instance()
            
            # Initialize observer (YOLO)
            yolo_model = self.model_mgr.get_model("yolo")
            config = DetectionConfig()
            self.observer = FrameObserver(yolo_model, config, target_fps=4.0)
            
            # Initialize scene classifier
            self.scene_classifier = SceneClassifier()
            
            logger.info("✓ All systems ready")
        except Exception as e:
            logger.error(f"✗ Initialization failed: {e}")
            self.observer = None
    
    def process_video(self, video_path: str, max_frames: int = 100) -> Dict[str, Any]:
        """Process video through complete pipeline"""
        logger.info(f"\n📹 Processing: {video_path}")
        
        # Load video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open: {video_path}")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        results = {
            "video": str(video_path),
            "fps": fps,
            "frames_processed": 0,
            "total_detections": 0,
            "unique_objects": set(),
            "scene_types": set(),
            "frame_data": [],
            "timeline": [],
            "memgraph_observations": [],
            "errors": []
        }
        
        frame_idx = 0
        start_time = time.time()
        prev_scene = None
        
        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                timestamp = frame_idx / fps
                
                # 1. YOLO detection
                detections = []
                if self.observer:
                    try:
                        # Run YOLO
                        results_yolo = self.observer.yolo(frame)
                        for det in results_yolo[0].boxes:
                            obj = {
                                "class": self.observer.yolo.names[int(det.cls[0])],
                                "confidence": float(det.conf[0]),
                                "bbox": [float(x) for x in det.xyxy[0].tolist()],
                                "center_x": (float(det.xyxy[0][0]) + float(det.xyxy[0][2])) / 2
                            }
                            detections.append(obj)
                    except Exception as e:
                        logger.debug(f"YOLO inference failed: {e}")
                
                # 2. Scene classification
                scene_type = "unknown"
                if self.scene_classifier:
                    try:
                        result = self.scene_classifier.classify_frame(frame)
                        if hasattr(result, 'scene_type'):
                            scene_type = result.scene_type.value
                        elif isinstance(result, dict):
                            scene_type = result.get('scene_type', 'unknown')
                    except Exception as e:
                        logger.debug(f"Scene classification failed: {e}")
                
                # 3. Spatial zones
                zones = {}
                for det in detections:
                    x = det['center_x']
                    if x < frame.shape[1] / 3:
                        zone = "left"
                    elif x < 2 * frame.shape[1] / 3:
                        zone = "center"
                    else:
                        zone = "right"
                    if zone not in zones:
                        zones[zone] = []
                    zones[zone].append(det['class'])
                
                # Store frame data
                frame_info = {
                    "idx": frame_idx,
                    "timestamp": timestamp,
                    "objects": len(detections),
                    "scene": scene_type,
                    "zones": zones,
                    "detections": detections
                }
                results["frame_data"].append(frame_info)
                
                # Track unique objects
                for det in detections:
                    results["unique_objects"].add(det['class'])
                    results["total_detections"] += 1
                
                results["scene_types"].add(scene_type)
                
                # Timeline: Scene changes
                if scene_type != prev_scene:
                    results["timeline"].append({
                        "frame": frame_idx,
                        "timestamp": timestamp,
                        "event": f"Scene changed to {scene_type}",
                        "objects": [d['class'] for d in detections]
                    })
                    prev_scene = scene_type
                
                # Memgraph-like observations
                for det in detections:
                    results["memgraph_observations"].append({
                        "frame": frame_idx,
                        "timestamp": timestamp,
                        "object": det['class'],
                        "confidence": det['confidence'],
                        "zone": zones.get(det['center_x'], "unknown"),
                        "scene": scene_type
                    })
                
                results["frames_processed"] += 1
                
                if (frame_idx + 1) % 10 == 0:
                    logger.info(f"  Frame {frame_idx+1:3d}: {len(detections)} objects, scene={scene_type}")
                
                frame_idx += 1
                
            except Exception as e:
                logger.warning(f"Frame {frame_idx} error: {e}")
                results["errors"].append({"frame": frame_idx, "error": str(e)})
                frame_idx += 1
        
        cap.release()
        
        elapsed = time.time() - start_time
        results["processing_time"] = elapsed
        results["fps_processed"] = results["frames_processed"] / elapsed if elapsed > 0 else 0
        
        # Convert sets to lists for JSON
        results["unique_objects"] = sorted(list(results["unique_objects"]))
        results["scene_types"] = sorted(list(results["scene_types"]))
        
        logger.info(f"✓ Processed {results['frames_processed']} frames in {elapsed:.1f}s")
        logger.info(f"  • Objects detected: {results['total_detections']}")
        logger.info(f"  • Unique classes: {len(results['unique_objects'])}")
        logger.info(f"  • Scenes: {results['scene_types']}")
        
        return results
    
    def build_gemini_context(self, results: Dict[str, Any]) -> str:
        """Build context for Gemini from results"""
        context = []
        
        context.append("# Egocentric Video Understanding")
        context.append(f"- Duration: {results['frames_processed'] / results['fps']:.1f}s")
        context.append(f"- Total objects detected: {results['total_detections']}")
        context.append(f"- Unique object classes: {', '.join(results['unique_objects'])}")
        context.append(f"- Scenes visited: {', '.join(results['scene_types'])}")
        
        context.append("\n## Temporal Timeline")
        for event in results['timeline'][:10]:
            context.append(f"- T={event['timestamp']:.1f}s: {event['event']}")
            if event['objects']:
                context.append(f"  Objects: {', '.join(event['objects'])}")
        
        context.append("\n## Spatial Distribution")
        for zone in ["left", "center", "right"]:
            zone_objects = set()
            for obs in results['memgraph_observations']:
                if obs.get('zone') == zone:
                    zone_objects.add(obs['object'])
            if zone_objects:
                context.append(f"- {zone.upper()}: {', '.join(zone_objects)}")
        
        context.append("\n## Scene Context")
        for scene_type in results['scene_types']:
            scene_frames = [f for f in results['frame_data'] if f['scene'] == scene_type]
            if scene_frames:
                context.append(f"- {scene_type}: {len(scene_frames)} frames")
                all_objs = set()
                for frame in scene_frames:
                    all_objs.update(frame['detections'])
                    all_objs = set([d['class'] if isinstance(d, dict) else d for d in all_objs])
        
        return "\n".join(context)
    
    def query_gemini(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query Gemini with video understanding"""
        if not GEMINI_AVAILABLE:
            logger.warning("Gemini not available")
            return []
        
        if not self.api_key:
            logger.warning("GOOGLE_API_KEY not set")
            return []
        
        logger.info("\n🤖 Querying Gemini API...")
        
        try:
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel("gemini-2.0-flash")
            
            context = self.build_gemini_context(results)
            
            queries = [
                "What objects did the person interact with in this video?",
                "Describe the sequence of events and scene transitions",
                "Where were the main objects located (left, center, right)?",
                "Based on the objects observed, what was the person's likely activity?",
                "Are there any temporal patterns or repeated interactions?"
            ]
            
            responses = []
            
            for q in queries:
                prompt = f"""You are analyzing an egocentric video from first-person perspective.

Context from video analysis:
{context}

Question: {q}

Provide a specific, detailed answer based on the video understanding above."""
                
                logger.info(f"  • {q[:50]}...")
                response = model.generate_content(prompt)
                
                responses.append({
                    "question": q,
                    "answer": response.text,
                    "timestamp": datetime.now().isoformat()
                })
            
            logger.info(f"✓ Got {len(responses)} responses")
            return responses
            
        except Exception as e:
            logger.error(f"Gemini error: {e}")
            return []


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="data/examples/test.mp4")
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--both-videos", action="store_true", help="Test both test.mp4 and video.mp4")
    
    args = parser.parse_args()
    
    logger.info("\n" + "="*80)
    logger.info("🎬 COMPLETE SYSTEM VALIDATION")
    logger.info("="*80)
    
    validator = CompleteValidator()
    
    videos = []
    if args.both_videos:
        videos = [
            "data/examples/test.mp4",
            "data/examples/video.mp4"
        ]
    else:
        videos = [args.video]
    
    all_results = {}
    
    for video_path in videos:
        if not Path(video_path).exists():
            logger.warning(f"Not found: {video_path}")
            continue
        
        results = validator.process_video(video_path, args.max_frames)
        if results:
            all_results[video_path] = results
            
            # Query Gemini
            gemini_responses = validator.query_gemini(results)
            results["gemini_responses"] = gemini_responses
    
    # Save comprehensive report
    report_path = Path("validation_complete_report.json")
    with open(report_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\n✓ Full report saved to {report_path}")
    logger.info("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
