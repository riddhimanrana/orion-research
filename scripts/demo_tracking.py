#!/usr/bin/env python3
"""
Demo: Orion Perception Pipeline with Enhanced Tracking

Shows how to use PerceptionEngine with tracking enabled.
"""

import sys
from pathlib import Path

# Add orion to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orion.perception.engine import PerceptionEngine
from orion.perception.config import PerceptionConfig, get_balanced_config


def main():
    print("="*80)
    print("Orion Perception with Enhanced Tracking Demo")
    print("="*80)
    
    # Use balanced config with tracking enabled
    config = get_balanced_config()
    config.enable_tracking = True  # Enable EnhancedTracker
    config.enable_3d = True        # Enable SLAM for CMC
    
    print(f"\nConfiguration:")
    print(f"  Detection model: {config.detection.model}")
    print(f"  Embedding dim:   {config.embedding.embedding_dim}")
    print(f"  Target FPS:      {config.target_fps}")
    print(f"  Tracking:        {config.enable_tracking}")
    print(f"  3D/SLAM:         {config.enable_3d}")
    
    # Initialize engine
    print("\nInitializing perception engine...")
    engine = PerceptionEngine(config=config, verbose=True)
    
    print("\n" + "="*80)
    print("Engine initialized successfully!")
    print("="*80)
    
    # Show what's loaded
    print("\nActive components:")
    if engine.observer:
        print("  ✓ FrameObserver (YOLO detection)")
    if engine.embedder:
        print("  ✓ VisualEmbedder (CLIP)")
    if engine.tracker:
        print("  ✓ EntityTracker (clustering)")
    if engine.describer:
        print("  ✓ EntityDescriber (FastVLM)")
    if engine.enhanced_tracker:
        print("  ✓ EnhancedTracker (3D+appearance Re-ID)")
    if engine.slam_engine:
        print("  ✓ SLAMEngine (camera pose estimation)")
    
    print("\n" + "="*80)
    print("Ready to process video!")
    print("="*80)
    print("\nUsage:")
    print("  # Process a video")
    print("  result = engine.process_video('path/to/video.mp4')")
    print("")
    print("  # Access results")
    print("  print(f'Entities: {result.unique_entities}')")
    print("  print(f'Detections: {result.total_detections}')")
    print("")
    print("  # Tracker stats (if tracking enabled)")
    print("  if engine.enhanced_tracker:")
    print("      stats = engine.enhanced_tracker.get_statistics()")
    print("      print(f'Tracks: {stats}')")
    print("="*80)


if __name__ == "__main__":
    main()
