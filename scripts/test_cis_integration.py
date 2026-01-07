#!/usr/bin/env python3
"""
Test CIS (Causal Influence Scoring) Integration with Memgraph
==============================================================

This script validates the Stage 4+5 integration:
1. Runs perception with CIS enabled
2. Exports to Memgraph
3. Queries and audits CIS relationships

Usage:
    python scripts/test_cis_integration.py --video data/examples/test.mp4 --episode cis_test
    python scripts/test_cis_integration.py --video data/examples/video.mp4 --episode cis_test_full
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("cis_test")


def run_perception_with_cis(video_path: str, episode_id: str, memgraph_host: str = "127.0.0.1"):
    """Run the full perception pipeline with CIS enabled."""
    from orion.perception.engine import PerceptionEngine
    from orion.perception.config import PerceptionConfig, DetectionConfig, EmbeddingConfig, TrackingConfig
    from orion.config import ensure_results_dir
    
    results_dir = ensure_results_dir(episode_id)
    logger.info(f"Results will be saved to: {results_dir}")
    
    # Configure with CIS enabled
    config = PerceptionConfig(
        detection=DetectionConfig(
            backend="yoloworld",
            confidence_threshold=0.25,
            yoloworld_prompt_preset="coco",
        ),
        embedding=EmbeddingConfig(
            embedding_dim=1024,
            batch_size=8,
        ),
        tracking=TrackingConfig(
            max_age=30,
            min_hits=2,
            appearance_threshold=0.60,
        ),
        target_fps=5.0,
        enable_tracking=True,
        # Stage 4: CIS configuration
        enable_cis=True,
        cis_threshold=0.45,
        cis_compute_every_n_frames=3,
        cis_temporal_buffer_size=30,
        cis_depth_gate_mm=2000.0,
        # Stage 5: Memgraph configuration
        use_memgraph=True,
        memgraph_host=memgraph_host,
        memgraph_port=7687,
        memgraph_batch_size=10,
        memgraph_sync_observations=True,
        memgraph_sync_cis=True,
    )
    
    # Initialize Memgraph backend
    try:
        from orion.graph.backends.memgraph import MemgraphBackend
        db = MemgraphBackend(host=memgraph_host, port=7687)
        db.clear_all()  # Start fresh
        logger.info(f"✓ Connected to Memgraph at {memgraph_host}:7687")
        
        # Create vector index for V-JEPA2 embeddings (1024-dim)
        db.create_vector_index(dimension=1024, metric="cosine")
    except Exception as e:
        logger.error(f"Memgraph connection failed: {e}")
        logger.error("Make sure Memgraph is running: docker-compose up -d memgraph")
        return None
    
    # Run perception
    logger.info("Running perception pipeline with CIS enabled...")
    engine = PerceptionEngine(config=config, db_manager=db, verbose=True)
    
    try:
        result = engine.process_video(
            video_path=video_path,
            save_visualizations=True,
            output_dir=str(results_dir),
        )
        logger.info(f"✓ Perception complete: {result.unique_entities} entities, {result.total_detections} detections")
        
        # Save CIS edges to JSONL for analysis
        cis_edges_path = results_dir / "cis_edges.jsonl"
        if hasattr(engine, '_cis_edges') and engine._cis_edges:
            with open(cis_edges_path, 'w') as f:
                for edge in engine._cis_edges:
                    f.write(json.dumps(edge.to_dict()) + '\n')
            logger.info(f"✓ Saved {len(engine._cis_edges)} CIS edges to {cis_edges_path}")
        
        return result, db
        
    except Exception as e:
        logger.error(f"Perception failed: {e}")
        raise
    finally:
        pass  # Keep db open for queries


def audit_memgraph_cis(db, min_score: float = 0.5):
    """Query and audit CIS relationships in Memgraph."""
    logger.info("\n" + "="*60)
    logger.info("MEMGRAPH CIS AUDIT")
    logger.info("="*60)
    
    # Get general statistics
    stats = db.get_statistics()
    logger.info(f"Graph Statistics:")
    logger.info(f"  Entities: {stats.get('entities', 0)}")
    logger.info(f"  Frames: {stats.get('frames', 0)}")
    logger.info(f"  Observations: {stats.get('observations', 0)}")
    logger.info(f"  Spatial Relationships: {stats.get('spatial_relationships', 0)}")
    
    # Get CIS statistics
    cis_stats = db.get_cis_statistics()
    logger.info(f"\nCIS Statistics:")
    logger.info(f"  INFLUENCES edges: {cis_stats.get('cis_influences', 0)}")
    logger.info(f"  GRASPS edges: {cis_stats.get('cis_grasps', 0)}")
    logger.info(f"  MOVES_WITH edges: {cis_stats.get('cis_moves_with', 0)}")
    logger.info(f"  Total CIS edges: {cis_stats.get('total_cis_edges', 0)}")
    logger.info(f"  Average CIS score: {cis_stats.get('avg_cis_score', 0):.3f}")
    logger.info(f"  Max CIS score: {cis_stats.get('max_cis_score', 0):.3f}")
    
    # Query top CIS relationships
    logger.info(f"\nTop CIS Relationships (score >= {min_score}):")
    cis_rels = db.query_cis_relationships(min_score=min_score, limit=20)
    
    if not cis_rels:
        logger.warning("  No CIS relationships found above threshold")
    else:
        for rel in cis_rels:
            logger.info(
                f"  [{rel['relationship_type']}] "
                f"{rel['agent_class']} (ID:{rel['agent_id']}) → "
                f"{rel['patient_class']} (ID:{rel['patient_id']}) "
                f"[score={rel['cis_score']:.3f}, frame={rel['frame_idx']}, type={rel['influence_type']}]"
            )
    
    # Check for GRASPS specifically (hand-object interactions)
    logger.info("\nGRASPS Interactions (hand-object):")
    grasps = db.query_cis_relationships(relationship_type="GRASPS", min_score=0.3, limit=10)
    if not grasps:
        logger.info("  No GRASPS interactions found")
    else:
        for g in grasps:
            logger.info(
                f"  {g['agent_class']} → {g['patient_class']} "
                f"[score={g['cis_score']:.3f}, frame={g['frame_idx']}]"
            )
    
    return cis_stats


def run_cypher_audit(db):
    """Run advanced Cypher queries for CIS audit."""
    logger.info("\n" + "="*60)
    logger.info("ADVANCED CYPHER AUDIT")
    logger.info("="*60)
    
    cursor = db.connection.cursor()
    
    # 1. Find objects with most interactions
    try:
        cursor.execute("""
            MATCH (e:Entity)-[r]->(other:Entity)
            WHERE type(r) IN ['INFLUENCES', 'GRASPS', 'MOVES_WITH']
            WITH e, count(r) as interaction_count
            ORDER BY interaction_count DESC
            LIMIT 5
            RETURN e.id, e.class_name, interaction_count
        """)
        logger.info("\nMost Interactive Entities:")
        for row in cursor.fetchall():
            logger.info(f"  {row[1]} (ID:{row[0]}): {row[2]} interactions")
    except Exception as e:
        logger.warning(f"  Query failed: {e}")
    
    # 2. Find person → object GRASPS chains
    try:
        cursor.execute("""
            MATCH (p:Entity)-[r:GRASPS]->(obj:Entity)
            WHERE p.class_name = 'person'
            RETURN p.id, obj.class_name, r.cis_score, r.frame_idx
            ORDER BY r.cis_score DESC
            LIMIT 10
        """)
        logger.info("\nPerson GRASPS Object Events:")
        results = cursor.fetchall()
        if not results:
            logger.info("  No person→object GRASPS found")
        for row in results:
            logger.info(f"  Person {row[0]} → {row[1]} [score={row[2]:.3f}, frame={row[3]}]")
    except Exception as e:
        logger.warning(f"  Query failed: {e}")
    
    # 3. Temporal analysis: when do interactions peak?
    try:
        cursor.execute("""
            MATCH ()-[r]->()
            WHERE type(r) IN ['INFLUENCES', 'GRASPS', 'MOVES_WITH']
              AND r.frame_idx IS NOT NULL
            WITH r.frame_idx / 30 as time_bucket, count(r) as edge_count
            ORDER BY edge_count DESC
            LIMIT 5
            RETURN time_bucket, edge_count
        """)
        logger.info("\nPeak Interaction Time Buckets (30-frame windows):")
        for row in cursor.fetchall():
            logger.info(f"  Frames {row[0]*30}-{(row[0]+1)*30}: {row[1]} edges")
    except Exception as e:
        logger.warning(f"  Query failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test CIS integration with Memgraph")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--episode", default="cis_test", help="Episode ID for results")
    parser.add_argument("--memgraph-host", default="127.0.0.1", help="Memgraph host")
    parser.add_argument("--skip-perception", action="store_true", help="Skip perception, just audit")
    parser.add_argument("--min-cis-score", type=float, default=0.5, help="Min CIS score for audit")
    args = parser.parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        sys.exit(1)
    
    db = None
    result = None
    
    if not args.skip_perception:
        perception_result = run_perception_with_cis(
            video_path=str(video_path),
            episode_id=args.episode,
            memgraph_host=args.memgraph_host,
        )
        if perception_result is not None:
            result, db = perception_result
        else:
            logger.error("Perception failed, cannot audit")
            sys.exit(1)
    else:
        # Just connect to Memgraph for audit
        try:
            from orion.graph.backends.memgraph import MemgraphBackend
            db = MemgraphBackend(host=args.memgraph_host, port=7687)
            logger.info(f"✓ Connected to Memgraph for audit")
        except Exception as e:
            logger.error(f"Memgraph connection failed: {e}")
            sys.exit(1)
    
    if db:
        audit_memgraph_cis(db, min_score=args.min_cis_score)
        run_cypher_audit(db)
        db.close()
    
    logger.info("\n✓ CIS Integration Test Complete")


if __name__ == "__main__":
    main()
