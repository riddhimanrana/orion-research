#!/usr/bin/env python3
"""
Complete Enhanced Pipeline Test
================================

Full pipeline with:
1. Load tracking results
2. Apply class corrections
3. Build knowledge graph
4. Test QA system with corrected classes

Usage:
    python scripts/test_complete_pipeline.py [tracking_results.json]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orion.class_correction import correct_tracking_results
from orion.knowledge_graph import KnowledgeGraphBuilder
from orion.video_qa import VideoQASystem

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Test complete enhanced pipeline with class corrections"
    )
    parser.add_argument(
        "results_file",
        nargs="?",
        default="data/testing/tracking_results_save1.json",
        help="Path to tracking results JSON file"
    )
    parser.add_argument(
        "--skip-correction",
        action="store_true",
        help="Skip class correction step"
    )
    parser.add_argument(
        "--skip-kg",
        action="store_true",
        help="Skip knowledge graph building"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive QA session"
    )
    parser.add_argument(
        "--no-llm-correction",
        action="store_true",
        help="Disable LLM-based class correction (faster)"
    )
    
    args = parser.parse_args()
    results_path = Path(args.results_file)
    
    # Load original results
    if not results_path.exists():
        logger.error(f"Results file not found: {results_path}")
        logger.info("Run tracking engine first:")
        logger.info("  python scripts/test_tracking.py data/examples/video1.mp4")
        return 1
    
    with open(results_path) as f:
        tracking_results = json.load(f)
    
    logger.info("="*80)
    logger.info("COMPLETE ENHANCED PIPELINE")
    logger.info("="*80)
    logger.info(f"Video: {tracking_results['video_path']}")
    logger.info(f"Entities: {tracking_results['total_entities']}")
    logger.info(f"Observations: {tracking_results['total_observations']}")
    
    # Step 1: Class Correction
    if not args.skip_correction:
        logger.info("\n[STEP 1/3] Applying Class Corrections...")
        tracking_results = correct_tracking_results(
            tracking_results,
            use_llm=not args.no_llm_correction,
            save_corrected=True
        )
        
        logger.info(f"✓ Corrected {tracking_results.get('num_corrections', 0)} classes")
        
        # Show corrections
        if tracking_results.get('class_corrections'):
            logger.info("\nCorrections applied:")
            for entity_id, correction in tracking_results['class_corrections'].items():
                logger.info(f"  {entity_id}: {correction}")
    else:
        logger.info("\n[STEP 1/3] Skipping class correction")
    
    # Step 2: Build Knowledge Graph
    if not args.skip_kg:
        logger.info("\n[STEP 2/3] Building Knowledge Graph...")
        builder = KnowledgeGraphBuilder()
        
        try:
            stats = builder.build_from_tracking_results(tracking_results)
            builder.close()
            
            logger.info("\n✓ Knowledge graph complete!")
            logger.info(f"  Entities: {stats['entities']}")
            logger.info(f"  Scenes: {stats['scenes']}")
            logger.info(f"  Spatial Relationships: {stats['spatial_relationships']}")
            logger.info(f"  Causal Chains: {stats['causal_chains']}")
            logger.info(f"  Scene Transitions: {stats['scene_transitions']}")
        
        except Exception as e:
            logger.error(f"✗ Knowledge graph building failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        logger.info("\n[STEP 2/3] Skipping knowledge graph building")
    
    # Step 3: Test QA
    logger.info("\n[STEP 3/3] Testing QA System...")
    
    qa = VideoQASystem()
    if not qa.connect():
        logger.error("✗ Cannot connect to Neo4j")
        return 1
    
    # Test with corrected classes
    test_questions = [
        "What type of room is this?",
        "What objects are in the video?",
        "Tell me about the hair drier",  # Should now reference corrected class
        "What's near the laptop?",
    ]
    
    logger.info("\nSample Q&A with corrected classes:")
    logger.info("-" * 80)
    
    for question in test_questions:
        logger.info(f"\nQ: {question}")
        answer = qa.ask_question(question)
        logger.info(f"A: {answer[:200]}..." if len(answer) > 200 else f"A: {answer}")
    
    qa.close()
    
    # Interactive session
    if args.interactive:
        logger.info("\n" + "="*80)
        logger.info("Starting Interactive QA Session...")
        logger.info("="*80 + "\n")
        qa = VideoQASystem()
        qa.start_interactive_session()
    
    logger.info("\n" + "="*80)
    logger.info("✓ PIPELINE COMPLETE")
    logger.info("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
