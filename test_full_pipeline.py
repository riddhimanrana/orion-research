#!/usr/bin/env python3
"""
Test script for full pipeline analysis with detailed logging.
"""
import sys
import json
from pathlib import Path

# Add orion to path
sys.path.insert(0, str(Path(__file__).parent))

from orion.pipeline import VideoPipeline, PipelineConfig
from orion.perception.config import get_fast_config
from orion.semantic.config import get_fast_semantic_config
from orion.settings import OrionSettings

def main():
    print("=" * 80)
    print("ORION FULL PIPELINE TEST")
    print("=" * 80)
    
    # Load settings and get Neo4j password
    settings = OrionSettings.load()
    neo4j_password = settings.get_neo4j_password()
    print(f"\n[CONFIG] Neo4j password decoded: {'*' * len(neo4j_password)}")
    
    # Create pipeline config
    config = PipelineConfig(
        perception_config=get_fast_config(),
        semantic_config=get_fast_semantic_config(),
        neo4j_uri="neo4j://127.0.0.1:7687",
        neo4j_user="neo4j",
        neo4j_password=neo4j_password,
        skip_graph_ingestion=False,
    )
    
    print(f"\n[INIT] Creating pipeline...")
    pipeline = VideoPipeline(config)
    
    print(f"[INIT] Neo4j manager: {pipeline.neo4j_manager}")
    if pipeline.neo4j_manager:
        print(f"[INIT] Neo4j connected: {pipeline.neo4j_manager.driver is not None}")
    
    # Process full video
    video_path = "data/examples/video.mp4"
    scene_id = "full_video_test"
    
    print(f"\n[VIDEO] Processing: {video_path}")
    print(f"[VIDEO] Scene ID: {scene_id}")
    print("-" * 80)
    
    results = pipeline.process_video(video_path, scene_id)
    
    print("\n" + "=" * 80)
    print("PERCEPTION RESULTS")
    print("=" * 80)
    
    perception = results["perception"]
    print(f"\nVideo Info:")
    print(f"  Duration: {perception.get('duration_seconds', 0):.1f}s")
    print(f"  Total frames: {perception.get('total_frames', 0)}")
    print(f"  FPS: {perception.get('fps', 0):.1f}")
    print(f"  Frames sampled: {perception.get('frames_sampled', 0)}")
    
    print(f"\nEntity Detection:")
    print(f"  Total entities: {perception.get('num_entities', 0)}")
    print(f"  Unique classes: {perception.get('num_unique_classes', 0)}")
    
    entities = perception.get("entities", [])
    print(f"\nTop 10 Entities:")
    for i, entity in enumerate(entities[:10], 1):
        print(f"  {i}. {entity['entity_id']:15s} | class={entity['class']:15s} | obs={entity['observations_count']:3d}")
    
    if len(entities) > 10:
        print(f"  ... and {len(entities) - 10} more entities")
    
    print("\n" + "=" * 80)
    print("SEMANTIC RESULTS")
    print("=" * 80)
    
    semantic = results["semantic"]
    print(f"\nEntity Processing:")
    print(f"  Entities consolidated: {semantic.get('num_entities', 0)}")
    print(f"  State changes detected: {semantic.get('state_changes', 0)}")
    
    print(f"\nSpatial Analysis:")
    print(f"  Spatial zones: {semantic.get('spatial_zones', 0)}")
    
    print(f"\nTemporal Analysis:")
    print(f"  Events detected: {semantic.get('num_events', 0)}")
    print(f"  Causal links: {semantic.get('causal_links', 0)}")
    
    # Show events if any
    events = semantic.get("events", [])
    if events:
        print(f"\nDetected Events:")
        for i, event in enumerate(events[:5], 1):
            print(f"  {i}. {event.get('description', 'No description')}")
            print(f"     Time: {event.get('start_time', 0):.1f}s - {event.get('end_time', 0):.1f}s")
    
    print("\n" + "=" * 80)
    print("GRAPH INGESTION RESULTS")
    print("=" * 80)
    
    graph = results.get("graph", {})
    print(f"\nStatus: {graph.get('status', 'unknown')}")
    print(f"Scene ID: {graph.get('scene_id', 'N/A')}")
    print(f"Entities ingested: {graph.get('entities_ingested', 0)}")
    print(f"Events ingested: {graph.get('events_ingested', 0)}")
    
    if graph.get('status') == 'failed':
        print(f"\nError: {graph.get('error', 'Unknown error')}")
    
    # Query Neo4j to verify
    if pipeline.neo4j_manager and pipeline.neo4j_manager.driver:
        print("\n" + "=" * 80)
        print("NEO4J DATABASE VERIFICATION")
        print("=" * 80)
        
        try:
            with pipeline.neo4j_manager.driver.session() as session:
                # Count nodes by label
                result = session.run("""
                    MATCH (n)
                    RETURN labels(n) as labels, count(n) as count
                    ORDER BY count DESC
                """)
                
                print("\nNodes in database:")
                total_nodes = 0
                for record in result:
                    labels = record["labels"]
                    count = record["count"]
                    total_nodes += count
                    print(f"  {labels}: {count}")
                
                if total_nodes == 0:
                    print("  (database is empty)")
                
                # Sample entities
                result = session.run("""
                    MATCH (e:Entity)
                    RETURN e.entity_id as id, e.object_class as class
                    LIMIT 5
                """)
                
                entities = list(result)
                if entities:
                    print("\nSample entities in Neo4j:")
                    for entity in entities:
                        print(f"  {entity['id']} - {entity['class']}")
                
        except Exception as e:
            print(f"\nError querying Neo4j: {e}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    
    # Save results to file
    output_file = "test_results/full_pipeline_results.json"
    Path("test_results").mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nFull results saved to: {output_file}")

if __name__ == "__main__":
    main()
