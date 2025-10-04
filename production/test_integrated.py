#!/usr/bin/env python3
"""
Test Integrated Pipeline: End-to-End Demo
==========================================

This script demonstrates the complete integrated pipeline from video to knowledge graph.

Usage:
    python production/test_integrated.py [--video path/to/video.mp4] [--use-fastvlm] [--quick]
"""

import sys
import argparse
import logging
from pathlib import Path

# Add production directory to path
sys.path.insert(0, str(Path(__file__).parent))

from production.integrated_pipeline import (
    run_integrated_pipeline,
    check_prerequisites,
    print_results_summary
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TestIntegrated')


def create_test_video(output_path: str = "data/testing/test_video_short.mp4") -> str:
    """
    Create a short test video for demonstration.
    
    Args:
        output_path: Where to save the test video
    
    Returns:
        Path to created video
    """
    import cv2
    import numpy as np
    from pathlib import Path
    
    logger.info("Creating test video...")
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Video parameters
    width, height = 640, 480
    fps = 10
    duration_seconds = 10  # Short 10-second video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    num_frames = fps * duration_seconds
    
    for frame_num in range(num_frames):
        # Create frame with moving objects
        frame = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light gray background
        
        # Moving rectangle (simulating a car)
        x = int((frame_num / num_frames) * (width - 100))
        y = height // 2
        cv2.rectangle(frame, (x, y), (x + 100, y + 60), (0, 0, 255), -1)
        cv2.putText(frame, "CAR", (x + 20, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Stationary circle (simulating a person)
        center_x = width // 4
        center_y = height // 3
        cv2.circle(frame, (center_x, center_y), 40, (255, 0, 0), -1)
        cv2.putText(frame, "PERSON", (center_x - 50, center_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Add frame number
        cv2.putText(frame, f"Frame {frame_num+1}/{num_frames}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        writer.write(frame)
    
    writer.release()
    logger.info(f"✓ Test video created: {output_path}")
    logger.info(f"  Duration: {duration_seconds}s, Frames: {num_frames}, FPS: {fps}")
    
    return output_path


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test Integrated Pipeline")
    parser.add_argument(
        '--video',
        type=str,
        help='Path to video file (creates test video if not specified)'
    )
    parser.add_argument(
        '--use-fastvlm',
        action='store_true',
        help='Use real FastVLM model (slower but higher quality)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Use fast configurations for quick testing'
    )
    parser.add_argument(
        '--skip-part1',
        action='store_true',
        help='Skip Part 1 and use existing perception log'
    )
    parser.add_argument(
        '--skip-part2',
        action='store_true',
        help='Skip Part 2 (only run perception)'
    )
    parser.add_argument(
        '--perception-log',
        type=str,
        help='Path to existing perception log (for skip-part1)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/testing',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("INTEGRATED PIPELINE TEST: Parts 1 + 2")
    print("="*80)
    print()
    
    # Check prerequisites
    print("Checking prerequisites...")
    status = check_prerequisites()
    print()
    
    # Check critical dependencies
    critical = ['torch', 'ultralytics']
    if args.use_fastvlm:
        critical.append('transformers')
    if not args.skip_part2:
        critical.extend(['hdbscan', 'sentence_transformers', 'neo4j', 'neo4j_connection'])
    
    missing = [dep for dep in critical if not status.get(dep, False)]
    if missing:
        print(f"❌ Missing critical dependencies: {', '.join(missing)}")
        print("\nPlease install missing dependencies:")
        for dep in missing:
            if dep == 'neo4j_connection':
                print("  - Start Neo4j: docker run --name neo4j -p7474:7474 -p7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest")
            elif dep == 'hdbscan':
                print(f"  - pip install {dep}==0.8.39")
            elif dep == 'sentence_transformers':
                print(f"  - pip install {dep}==3.3.1")
            else:
                print(f"  - pip install {dep}")
        print()
        return 1
    
    # Get video path
    if args.video:
        video_path = args.video
        if not Path(video_path).exists():
            print(f"❌ Video file not found: {video_path}")
            return 1
    else:
        print("No video specified, creating test video...")
        video_path = create_test_video()
        print()
    
    # Determine configurations
    if args.quick:
        part1_config = "fast"
        part2_config = "fast"
        print("Using FAST configurations for quick testing")
    else:
        part1_config = "balanced"
        part2_config = "balanced"
        print("Using BALANCED configurations")
    
    print(f"FastVLM: {'ENABLED (real model)' if args.use_fastvlm else 'DISABLED (placeholder)'}")
    print()
    
    # Run pipeline
    print("="*80)
    print("RUNNING INTEGRATED PIPELINE")
    print("="*80)
    print()
    
    try:
        results = run_integrated_pipeline(
            video_path=video_path,
            output_dir=args.output_dir,
            use_fastvlm=args.use_fastvlm,
            part1_config=part1_config,
            part2_config=part2_config,
            skip_part1=args.skip_part1,
            skip_part2=args.skip_part2,
        )
        
        print()
        print("="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        print()
        
        print_results_summary(results)
        
        if results['success']:
            print()
            print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
            print()
            
            if not args.skip_part2 and results.get('part2'):
                print("Next steps:")
                print("  1. Open Neo4j Browser: http://localhost:7474")
                print("  2. Run sample queries:")
                print("     MATCH (e:Entity) RETURN e LIMIT 25")
                print("     MATCH (e:Entity)-[:PARTICIPATED_IN]->(ev:Event) RETURN e, ev")
                print("  3. Analyze results:")
                print(f"     python production/test_part2.py --use-part1-output")
            
            return 0
        else:
            print()
            print("❌ PIPELINE FAILED")
            print(f"Errors: {results.get('errors', [])}")
            return 1
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.exception("Pipeline failed with exception")
        print(f"\n❌ Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
