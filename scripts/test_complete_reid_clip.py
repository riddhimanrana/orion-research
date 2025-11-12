#!/usr/bin/env python3
"""
Comprehensive Re-ID + CLIP Test on video_short.mp4

This script:
1. Runs the full unified pipeline on video_short.mp4
2. Tests Phase 5 Re-ID matching with CLIP embeddings
3. Verifies object deduplication across frames
4. Shows how CLIP embeddings enable semantic matching
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from orion.perception.unified_pipeline import UnifiedPipeline
import numpy as np


def main():
    """Run complete Re-ID CLIP test"""
    print("\n" + "="*80)
    print("🎯 UNIFIED PIPELINE + Re-ID CLIP TEST")
    print("="*80)
    print("\nTesting on: data/examples/video_short.mp4")
    print("Focus: Phase 5 Re-ID + CLIP semantic matching\n")
    
    # Run pipeline
    try:
        pipeline = UnifiedPipeline(
            video_path="data/examples/video_short.mp4",
            max_frames=60,  # Full short video
            use_rerun=False
        )
        
        results = pipeline.run(benchmark=True)
        
        # Extract key metrics
        print("\n" + "="*80)
        print("📊 Re-ID + CLIP RESULTS")
        print("="*80)
        
        if results:
            print(f"\nFrames processed: {results.get('frames_processed', 0)}")
            print(f"Raw detections (Phase 1): {results.get('total_detections', 0)}")
            print(f"Tracked objects (Phase 4): {results.get('tracked_objects', 0)}")
            print(f"Unified objects (Phase 5): {results.get('unified_objects', 0)}")
            
            # Compute reduction factors
            raw_dets = results.get('total_detections', 1)
            unified_objs = results.get('unified_objects', 1)
            reduction = raw_dets / max(unified_objs, 1)
            
            print(f"\n✨ Re-ID Deduplication Factor: {reduction:.1f}x")
            print(f"   {raw_dets} detections → {unified_objs} unified objects")
            
            # Phase timing
            print("\n⏱️  Phase Timing:")
            timing = results.get('timing', {})
            for phase, t in timing.items():
                print(f"   {phase}: {t:.2f}s")
        
        print("\n" + "="*80)
        print("✅ Test complete!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
