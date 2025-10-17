#!/usr/bin/env python3
"""
Test Enhanced Knowledge Graph and QA System
============================================

This script:
1. Loads tracking results
2. Builds enhanced knowledge graph with scene/spatial/causal reasoning
3. Tests the enhanced QA system

Usage:
    python scripts/test_enhanced_kg.py [tracking_results.json]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orion.enhanced_knowledge_graph import EnhancedKnowledgeGraphBuilder
from src.orion.enhanced_video_qa import EnhancedVideoQASystem

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)


def test_knowledge_graph(results_path: Path):
    """Build enhanced knowledge graph from tracking results"""
    logger.info("="*80)
    logger.info("TESTING ENHANCED KNOWLEDGE GRAPH")
    logger.info("="*80)
    
    # Load tracking results
    if not results_path.exists():
        logger.error(f"Results file not found: {results_path}")
        logger.info("Run tracking engine first:")
        logger.info("  python scripts/test_tracking.py data/examples/video1.mp4")
        return False
    
    with open(results_path) as f:
        tracking_results = json.load(f)
    
    logger.info(f"Loaded {len(tracking_results['entities'])} entities")
    logger.info(f"Video: {tracking_results['video_path']}")
    logger.info(f"Total observations: {tracking_results['total_observations']}")
    logger.info(f"Efficiency ratio: {tracking_results['efficiency_ratio']:.2f}x")
    
    # Build enhanced knowledge graph
    logger.info("\nBuilding enhanced knowledge graph...")
    builder = EnhancedKnowledgeGraphBuilder()
    
    try:
        stats = builder.build_from_tracking_results(tracking_results)
        builder.close()
        
        logger.info("\n✓ Knowledge graph built successfully!")
        logger.info(f"  Entities: {stats['entities']}")
        logger.info(f"  Scenes: {stats['scenes']}")
        logger.info(f"  Spatial Relationships: {stats['spatial_relationships']}")
        logger.info(f"  Causal Chains: {stats['causal_chains']}")
        logger.info(f"  Scene Transitions: {stats['scene_transitions']}")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ Failed to build knowledge graph: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_qa_system():
    """Test the enhanced QA system with sample questions"""
    logger.info("\n" + "="*80)
    logger.info("TESTING ENHANCED QA SYSTEM")
    logger.info("="*80)
    
    qa = EnhancedVideoQASystem()
    
    if not qa.connect():
        logger.error("✗ Cannot connect to Neo4j")
        logger.info("Make sure Neo4j is running:")
        logger.info("  docker run -d -p 7687:7687 -p 7474:7474 \\")
        logger.info("    -e NEO4J_AUTH=neo4j/orion123 \\")
        logger.info("    neo4j:latest")
        return False
    
    # Test questions
    test_questions = [
        ("What type of room or setting is this?", "scene"),
        ("What objects are in the video?", "entity"),
        ("What objects are near the laptop?", "spatial"),
        ("What happened during the video?", "temporal"),
    ]
    
    logger.info("\nTesting sample questions...\n")
    
    for question, q_type in test_questions:
        logger.info(f"Question ({q_type}): {question}")
        logger.info("-" * 60)
        
        answer = qa.ask_question(question)
        logger.info(f"Answer:\n{answer}\n")
        logger.info("=" * 60 + "\n")
    
    qa.close()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test enhanced knowledge graph and QA system"
    )
    parser.add_argument(
        "results_file",
        nargs="?",
        default="data/testing/tracking_results_save1.json",
        help="Path to tracking results JSON file"
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip knowledge graph building (use existing graph)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive QA session after tests"
    )
    
    args = parser.parse_args()
    results_path = Path(args.results_file)
    
    # Build knowledge graph
    if not args.skip_build:
        success = test_knowledge_graph(results_path)
        if not success:
            logger.error("Knowledge graph building failed")
            return 1
    else:
        logger.info("Skipping knowledge graph building (using existing)")
    
    # Test QA system
    success = test_qa_system()
    if not success:
        logger.error("QA system test failed")
        return 1
    
    # Interactive session
    if args.interactive:
        logger.info("\n" + "="*80)
        logger.info("Starting interactive QA session...")
        logger.info("="*80 + "\n")
        
        qa = EnhancedVideoQASystem()
        qa.start_interactive_session()
    
    logger.info("\n✓ All tests completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
