import sys
import argparse
from pathlib import Path
import logging

# Add workspace root to path
workspace_root = Path(__file__).resolve().parents[1]
sys.path.append(str(workspace_root))

from orion.query.index import VideoIndex
from orion.query.query_engine import QueryEngine

def main():
    parser = argparse.ArgumentParser(description="Orion QA Interface")
    parser.add_argument("--query", type=str, help="Question to ask (e.g., 'Where was the book?')")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    args = parser.parse_args()
    
    results_dir = workspace_root / "results/full_video_analysis"
    index_path = results_dir / "video_index.db"
    video_path = workspace_root / "data/examples/video.mp4"
    
    if not index_path.exists():
        print(f"Error: Index not found at {index_path}. Run scripts/populate_index.py first.")
        return

    # Initialize engine
    index = VideoIndex(index_path, video_path)
    index.connect()
    
    engine = QueryEngine(index, video_path)
    
    if args.query:
        answer_query(engine, args.query)
    elif args.interactive:
        print("Orion QA Interface (Type 'exit' to quit)")
        print("-" * 40)
        while True:
            query = input("Q: ")
            if query.lower() in ('exit', 'quit'):
                break
            answer_query(engine, query)
    else:
        parser.print_help()

def import_sqlite3():
    import sqlite3
    return sqlite3

def answer_query(engine, query):
    result = engine.query(query)
    print(f"A: {result.answer}")
    if result.entities:
        print(f"   (Found {len(result.entities)} relevant observations)")
        # Show timestamps
        timestamps = sorted(list(set([f"{obs.timestamp:.1f}s" for obs in result.entities])))
        if len(timestamps) > 5:
            print(f"   Timestamps: {', '.join(timestamps[:5])} ...")
        else:
            print(f"   Timestamps: {', '.join(timestamps)}")
    print("-" * 40)

if __name__ == "__main__":
    main()
