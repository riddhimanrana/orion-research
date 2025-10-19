#!/usr/bin/env python3
"""
Quick test of enhanced QA with predefined questions
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from orion.video_qa import VideoQASystem

def main():
    print("="*80)
    print("VIDEO QA - QUICK TEST")
    print("="*80)
    
    qa = VideoQASystem()
    
    if not qa.connect():
        print("❌ Cannot connect to Neo4j")
        return 1
    
    # Test questions covering different types
    questions = [
        "What type of rooms appear in the video?",
        "What objects are most common in the video?",
        "Are there any bedroom scenes? What objects are in them?",
        "What objects are always found together?",
        "Describe the spatial layout of the workspace",
        "What happens in the timeline of the video?",
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{i}. Q: {question}")
        print("-" * 60)
        answer = qa.ask_question(question)
        print(f"A: {answer}")
        print("=" * 80)
    
    qa.close()
    return 0

if __name__ == "__main__":
    sys.exit(main())
