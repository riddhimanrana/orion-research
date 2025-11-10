"""
Query CLI - Interactive visual queries
=======================================

Query processed videos with natural language.

Author: Orion Research Team
Date: November 2025
"""

import argparse
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orion.query.index import VideoIndex
from orion.query.query_engine import QueryEngine
from orion.managers.model_manager import ModelManager


def main():
    parser = argparse.ArgumentParser(
        description='Query processed videos',
        epilog='''
Examples:
  python scripts/query_video.py --index video_index.db "What color was the book?"
  python scripts/query_video.py --index video_index.db "Where did I see the laptop?"
  python scripts/query_video.py --index video_index.db "What was on the desk?"
        '''
    )
    parser.add_argument('--index', type=str, required=True, help='Path to video index (.db file)')
    parser.add_argument('--video', type=str, required=True, help='Path to original video')
    parser.add_argument('query', type=str, nargs='?', help='Query string (optional, will prompt if not provided)')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode (multiple queries)')
    
    args = parser.parse_args()
    
    # Load index
    print("Loading video index...")
    index = VideoIndex(Path(args.index), Path(args.video))
    index.conn = index.conn or __import__('sqlite3').connect(args.index)
    
    # Load FastVLM for on-demand captioning
    print("Loading FastVLM for visual descriptions...")
    model_manager = ModelManager.get_instance()
    fastvlm = model_manager.fastvlm
    print("  ✓ FastVLM loaded\n")
    
    # Create query engine
    engine = QueryEngine(index, Path(args.video), fastvlm)
    
    if args.interactive:
        # Interactive mode
        print("="*80)
        print("INTERACTIVE QUERY MODE")
        print("="*80)
        print("\nAsk questions about the video. Type 'quit' to exit.\n")
        print("Examples:")
        print("  - What color was the book?")
        print("  - Where did I see the laptop?")
        print("  - What was on the desk?\n")
        
        while True:
            try:
                query = input("❓ Query: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                if not query:
                    continue
                
                result = engine.query(query)
                print(f"\n💡 Answer: {result.answer}\n")
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
    
    elif args.query:
        # Single query mode
        print(f"❓ Query: {args.query}\n")
        result = engine.query(args.query)
        print(f"💡 Answer: {result.answer}\n")
        
        if result.entities:
            print(f"📊 Found {len(result.entities)} observations")
            for i, obs in enumerate(result.entities[:3], 1):
                timestamp = obs.timestamp
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)
                print(f"  {i}. Entity {obs.entity_id} at {minutes}:{seconds:02d} (confidence: {obs.confidence:.2f})")
    
    else:
        # Prompt for query
        query = input("❓ Query: ").strip()
        if query:
            result = engine.query(query)
            print(f"\n💡 Answer: {result.answer}\n")
    
    engine.close()
    index.close()


if __name__ == '__main__':
    main()
