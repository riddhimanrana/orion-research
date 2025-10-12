#!/usr/bin/env python3
"""
Test script for Perception Engine with FastVLM integration
Tests the complete pipeline from video to rich descriptions
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add production to path
sys.path.insert(0, str(Path(__file__).parent / "production"))

from perception_engine import run_perception_engine

def test_perception_with_fastvlm():
    """Test perception engine with FastVLM descriptions"""
    
    print("="*80)
    print("PERCEPTION ENGINE + FASTVLM INTEGRATION TEST")
    print("="*80)
    print()
    
    # Check if test video exists
    test_video = Path("data/examples/video1.mp4")
    if not test_video.exists():
        print(f"⚠️  Test video not found at: {test_video}")
        print("   Looking for alternative test videos...")
        
        # Try to find any video
        video_paths = list(Path("data").rglob("*.mp4"))
        if video_paths:
            test_video = video_paths[0]
            print(f"   Using: {test_video}")
        else:
            print("   No video files found in data/ directory")
            print("   Please add a test video to data/examples/video1.mp4")
            return False
    
    print(f"Test Video: {test_video}")
    print(f"Output: Will be saved to data/testing/")
    print()
    print("-"*80)
    print("Running perception engine...")
    print("-"*80)
    print()
    
    try:
        # Run perception engine (this will use FastVLM for descriptions)
        results = run_perception_engine(str(test_video))
        
        print()
        print("="*80)
        print("✓ PERCEPTION ENGINE TEST SUCCESSFUL")
        print("="*80)
        print()
        print(f"Results Summary:")
        print(f"  - Total objects detected: {len(results)}")
        
        # Show sample results
        if results:
            print(f"\nSample detections (first 3):")
            for i, obj in enumerate(results[:3], 1):
                print(f"\n  {i}. Frame {obj.get('frame_id', 'N/A')} @ {obj.get('timestamp', 'N/A')}s")
                print(f"     Class: {obj.get('class', 'N/A')}")
                print(f"     Position: {obj.get('spatial_position', 'N/A')}")
                if 'description' in obj:
                    desc = obj['description'][:100] + "..." if len(obj['description']) > 100 else obj['description']
                    print(f"     Description: {desc}")
        
        print()
        print("-"*80)
        print("Check the output JSON in data/testing/ for full results")
        print("-"*80)
        
        return True
        
    except Exception as e:
        print()
        print("="*80)
        print("✗ PERCEPTION ENGINE TEST FAILED")
        print("="*80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_perception_with_fastvlm()
    sys.exit(0 if success else 1)
