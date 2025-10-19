#!/usr/bin/env python3
"""
Test the new tracking engine on a sample video.

This script runs the complete 4-phase tracking pipeline:
1. Observation Collection (detect all objects with embeddings)
2. Entity Clustering (group observations into unique entities)
3. Smart Description (describe each entity once from best frame)
4. Temporal Graph (build knowledge graph with movements)

Usage:
    python scripts/test_tracking.py path/to/video.mp4
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orion.neo4j_manager import Neo4jManager
from orion.tracking_engine import run_tracking_engine
from orion.temporal_graph_builder import TemporalGraphBuilder


def main():
    parser = argparse.ArgumentParser(
        description="Test entity-based tracking pipeline"
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to video file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/testing",
        help="Output directory for results"
    )
    parser.add_argument(
        "--build-graph",
        action="store_true",
        help="Build Neo4j temporal knowledge graph"
    )
    parser.add_argument(
        "--neo4j-uri",
        type=str,
        default="neo4j://127.0.0.1:7687",
        help="Neo4j URI"
    )
    parser.add_argument(
        "--neo4j-user",
        type=str,
        default="neo4j",
        help="Neo4j username"
    )
    parser.add_argument(
        "--neo4j-password",
        type=str,
        default="orion123",
        help="Neo4j password"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(levelname)s | %(name)s | %(message)s'
    )
    
    logger = logging.getLogger('TestTracking')
    
    # Validate video path
    video_path = Path(args.video_path)
    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("TRACKING ENGINE TEST")
    logger.info("="*80)
    logger.info(f"Video: {video_path}")
    logger.info(f"Output: {output_dir}")
    logger.info("="*80 + "\n")
    
    try:
        # Run tracking pipeline (Phases 1-3)
        logger.info("Starting tracking pipeline...")
        entities, observations = run_tracking_engine(str(video_path))
        
        # Save results
        results_path = output_dir / "tracking_results.json"
        results = {
            'video_path': str(video_path),
            'total_observations': len(observations),
            'total_entities': len(entities),
            'efficiency_ratio': len(observations) / max(len(entities), 1),
            'entities': [
                {
                    'id': e.id,
                    'class': e.class_name,
                    'description': e.description,
                    'appearance_count': e.appearance_count,
                    'first_seen': e.first_seen,
                    'last_seen': e.last_seen,
                    'duration': e.duration,
                    'described_from_frame': e.described_from_frame,
                    'state_changes': len(e.state_changes),
                    'frame_numbers': [obs.frame_number for obs in e.observations]
                }
                for e in entities
            ]
        }
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n✓ Results saved to {results_path}")
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("TRACKING SUMMARY")
        logger.info("="*80)
        logger.info(f"Total observations: {len(observations)}")
        logger.info(f"Unique entities: {len(entities)}")
        logger.info(f"Efficiency ratio: {results['efficiency_ratio']:.1f}x")
        logger.info(f"  (system found {len(observations)} detections")
        logger.info(f"   but only described {len(entities)} unique entities)")
        logger.info("")
        
        for entity in entities[:10]:  # Show first 10
            logger.info(f"  {entity.id}: {entity.class_name}")
            logger.info(f"    Appearances: {entity.appearance_count}")
            logger.info(f"    Duration: {entity.duration:.1f}s")
            if entity.description:
                desc = entity.description[:100] + "..." if len(entity.description) > 100 else entity.description
                logger.info(f"    Description: {desc}")
            if entity.state_changes:
                logger.info(f"    State changes: {len(entity.state_changes)}")
            logger.info("")
        
        if len(entities) > 10:
            logger.info(f"  ... and {len(entities) - 10} more entities")
        
        logger.info("="*80 + "\n")
        
        # Build temporal knowledge graph if requested
        if args.build_graph:
            logger.info("Building temporal knowledge graph...")
            
            neo4j_manager = Neo4jManager(
                uri=args.neo4j_uri,
                user=args.neo4j_user,
                password=args.neo4j_password,
            )
            builder = TemporalGraphBuilder(neo4j_manager=neo4j_manager)
            
            try:
                stats = builder.build_graph(entities, observations)
                logger.info("\n✓ Knowledge graph built successfully")
                logger.info(f"  Entity nodes: {stats['entity_nodes']}")
                logger.info(f"  Frame nodes: {stats['frame_nodes']}")
                logger.info(f"  Relationships: {stats['appearance_rels'] + stats['spatial_rels'] + stats['movement_rels']}")
            finally:
                builder.close()
        
        return 0
        
    except Exception as e:
        logger.error(f"\n✗ Tracking pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
