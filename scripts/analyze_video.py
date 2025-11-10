"""
Simple video analysis with real-time processing + interactive queries

This is a simplified wrapper around the SLAM pipeline optimized for:
- Real-time processing (<66s for 66s video)
- Memgraph graph database export
- Interactive natural language queries

Usage:
    python scripts/analyze_video.py video.mp4
    python scripts/analyze_video.py video.mp4 --interactive
    python scripts/analyze_video.py video.mp4 --fast --interactive
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Analyze video with real-time processing + interactive queries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/analyze_video.py video.mp4
  python scripts/analyze_video.py video.mp4 -i
  python scripts/analyze_video.py video.mp4 --fast -i
  python scripts/analyze_video.py video.mp4 --yolo-model yolo11s

What it does:
  1. Process video in real-time (<66s for 66s video)
  2. Export to Memgraph graph database
  3. (Optional) Start interactive query mode
  4. Ask questions like "What color was the book?"
        """
    )
    
    parser.add_argument("video", help="Path to video file")
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Start interactive query mode after processing"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Ultra-fast mode: YOLO11n + skip=50 (fastest, less accurate)"
    )
    parser.add_argument(
        "--yolo-model",
        choices=["yolo11n", "yolo11s", "yolo11m", "yolo11x"],
        default="yolo11n",
        help="YOLO model (n=fastest/default, s=fast, m=balanced, x=accurate)"
    )
    parser.add_argument(
        "--skip",
        type=int,
        help="Frame skip interval (auto-selected based on model if not specified)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Limit frames (for testing)"
    )
    parser.add_argument(
        "--viz",
        choices=["rerun", "none"],
        default="none",
        help="Visualization mode (none=headless/default, rerun=3D browser)"
    )
    
    args = parser.parse_args()
    
    # Check video exists
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ Video not found: {args.video}")
        sys.exit(1)
    
    # Auto-select skip based on model for optimal speed
    if args.skip is None:
        skip_values = {
            "yolo11n": 50,  # Real-time
            "yolo11s": 30,  # Fast
            "yolo11m": 20,  # Balanced
            "yolo11x": 15,  # Accurate
        }
        skip = skip_values.get(args.yolo_model, 50)
    else:
        skip = args.skip
    
    # Override for fast mode
    if args.fast:
        args.yolo_model = "yolo11n"
        skip = 50
    
    # Build SLAM command
    slam_script = Path(__file__).parent / "run_slam_complete.py"
    
    cmd = [
        sys.executable,
        str(slam_script),
        "--video", str(video_path.absolute()),
        "--yolo-model", args.yolo_model,
        "--skip", str(skip),
        "--no-adaptive",  # Always disable for predictable timing
        "--export-memgraph",  # Always export for queries
    ]
    
    if args.viz == "rerun":
        cmd.append("--rerun")
    
    if args.max_frames:
        cmd.extend(["--max-frames", str(args.max_frames)])
    
    # Display configuration
    print("\n" + "="*60)
    print("🎥 ORION VIDEO ANALYSIS")
    print("="*60)
    print(f"📹 Video: {video_path.name}")
    print(f"🤖 Model: {args.yolo_model.upper()} (skip={skip})")
    print(f"📊 Export: Memgraph graph database")
    if args.interactive:
        print(f"💬 Mode: Interactive queries after processing")
    print(f"🎯 Target: Real-time (<66s for 66s video)")
    print("="*60 + "\n")
    
    # Run SLAM pipeline
    try:
        subprocess.run(cmd, check=True)
        
        # Start interactive mode if requested
        if args.interactive:
            print("\n" + "="*60)
            print("💬 INTERACTIVE QUERY MODE")
            print("="*60)
            print("Ask questions about the video:")
            print("  • What color was the book?")
            print("  • Where was the laptop?")
            print("  • What objects were near the person?")
            print("\nType 'quit' to exit")
            print("="*60 + "\n")
            
            query_script = Path(__file__).parent / "query_memgraph.py"
            if query_script.exists():
                query_cmd = [sys.executable, str(query_script), "--interactive"]
                subprocess.run(query_cmd)
            else:
                print(f"❌ Query script not found: {query_script}")
        else:
            print("\n✅ Processing complete!")
            print("\n💡 To query the video, run:")
            print(f"   python scripts/query_memgraph.py --interactive")
            print("\n   Or re-run with: python scripts/analyze_video.py {args.video} -i\n")
            
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Processing failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
