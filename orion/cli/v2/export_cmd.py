"""orion export - Export to Memgraph"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run_export(args) -> int:
    """Export episode data to Memgraph."""
    
    episode_dir = Path("results") / args.episode
    
    if not episode_dir.exists():
        print(f"Episode not found: {args.episode}")
        return 1
    
    print(f"\n  Stage 5: MEMGRAPH EXPORT")
    print(f"  Host: {args.memgraph_host}:{args.memgraph_port}")
    print("  " + "─" * 60)
    
    # Try to connect to Memgraph
    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("  ✗ neo4j package not installed")
        print("  Run: pip install neo4j")
        return 1
    
    uri = f"bolt://{args.memgraph_host}:{args.memgraph_port}"
    
    try:
        driver = GraphDatabase.driver(uri, auth=("", ""))
        driver.verify_connectivity()
        print(f"  ✓ Connected to Memgraph")
    except Exception as e:
        print(f"  ✗ Cannot connect to Memgraph: {e}")
        print("  Make sure Memgraph is running:")
        print("    docker run -d --name memgraph -p 7687:7687 memgraph/memgraph")
        return 1
    
    # Load data
    meta_path = episode_dir / "episode_meta.json"
    memory_path = episode_dir / "memory.json"
    tracks_path = episode_dir / "tracks_filtered.jsonl"
    if not tracks_path.exists():
        tracks_path = episode_dir / "tracks.jsonl"
    sg_path = episode_dir / "scene_graph.jsonl"
    cis_path = episode_dir / "cis_scores.json"
    
    # Load episode metadata
    with open(meta_path) as f:
        meta = json.load(f)
    
    # Load memory objects
    memory_objects = []
    if memory_path.exists():
        with open(memory_path) as f:
            memory = json.load(f)
            memory_objects = memory.get("filtered_objects", memory.get("objects", []))
    
    # Load observations
    observations = []
    if tracks_path.exists():
        with open(tracks_path) as f:
            for line in f:
                if line.strip():
                    observations.append(json.loads(line))
    
    # Load scene graphs
    scene_graphs = []
    if sg_path.exists():
        with open(sg_path) as f:
            for line in f:
                if line.strip():
                    scene_graphs.append(json.loads(line))
    
    # Load CIS scores
    cis_scores = {}
    if cis_path.exists():
        with open(cis_path) as f:
            cis_scores = json.load(f)
    
    print(f"  Data to export:")
    print(f"    Memory objects: {len(memory_objects)}")
    print(f"    Observations: {len(observations)}")
    print(f"    Scene graphs: {len(scene_graphs)}")
    print(f"    CIS scores: {len(cis_scores.get('object_scores', {}))}")
    
    # Clear existing data for this episode if requested
    if args.clear_existing:
        with driver.session() as session:
            session.run("""
                MATCH (n)
                WHERE n.episode = $episode
                DETACH DELETE n
            """, episode=args.episode)
            print(f"  Cleared existing data for episode: {args.episode}")
    
    # Create episode node
    with driver.session() as session:
        session.run("""
            MERGE (e:Episode {id: $episode})
            SET e.video_path = $video_path,
                e.fps = $fps,
                e.frame_count = $frame_count,
                e.duration_sec = $duration_sec,
                e.created_at = $created_at
        """, 
            episode=args.episode,
            video_path=meta.get("video_path", ""),
            fps=meta.get("video", {}).get("fps", 0),
            frame_count=meta.get("video", {}).get("frame_count", 0),
            duration_sec=meta.get("video", {}).get("duration_sec", 0),
            created_at=meta.get("created_at", "")
        )
    print(f"  Created Episode node")
    
    # Create MemoryObject nodes
    with driver.session() as session:
        for obj in memory_objects:
            cis = cis_scores.get("object_scores", {}).get(obj["id"], {})
            
            session.run("""
                MERGE (o:MemoryObject {id: $id, episode: $episode})
                SET o.label = $label,
                    o.total_observations = $total_obs,
                    o.first_frame = $first_frame,
                    o.last_frame = $last_frame,
                    o.influence_score = $influence_score,
                    o.vlm_description = $vlm_description
            """,
                id=obj["id"],
                episode=args.episode,
                label=obj.get("canonical_label", "object"),
                total_obs=obj.get("total_observations", 0),
                first_frame=obj.get("first_frame", 0),
                last_frame=obj.get("last_frame", 0),
                influence_score=cis.get("influence_score", 0),
                vlm_description=obj.get("vlm_description", "")
            )
            
            # Link to episode
            session.run("""
                MATCH (e:Episode {id: $episode})
                MATCH (o:MemoryObject {id: $id, episode: $episode})
                MERGE (e)-[:CONTAINS]->(o)
            """, episode=args.episode, id=obj["id"])
    
    print(f"  Created {len(memory_objects)} MemoryObject nodes")
    
    # Create Frame nodes (sample every 5th to reduce graph size)
    frame_ids = sorted(set(obs["frame_id"] for obs in observations))
    sampled_frames = frame_ids[::5]  # Every 5th frame
    
    with driver.session() as session:
        for frame_id in sampled_frames:
            session.run("""
                MERGE (f:Frame {id: $frame_id, episode: $episode})
                SET f.timestamp = $timestamp
            """,
                frame_id=frame_id,
                episode=args.episode,
                timestamp=frame_id / meta.get("video", {}).get("fps", 30)
            )
            
            # Link to episode
            session.run("""
                MATCH (e:Episode {id: $episode})
                MATCH (f:Frame {id: $frame_id, episode: $episode})
                MERGE (e)-[:HAS_FRAME]->(f)
            """, episode=args.episode, frame_id=frame_id)
    
    print(f"  Created {len(sampled_frames)} Frame nodes (sampled)")
    
    # Create OBSERVED_IN relationships
    with driver.session() as session:
        for obs in observations:
            if obs["frame_id"] not in sampled_frames:
                continue
            
            obj_id = obs.get("memory_object_id", obs["track_id"])
            
            session.run("""
                MATCH (o:MemoryObject {id: $obj_id, episode: $episode})
                MATCH (f:Frame {id: $frame_id, episode: $episode})
                MERGE (o)-[r:OBSERVED_IN]->(f)
                SET r.bbox = $bbox,
                    r.confidence = $confidence
            """,
                obj_id=obj_id,
                episode=args.episode,
                frame_id=obs["frame_id"],
                bbox=obs.get("bbox"),
                confidence=obs.get("confidence", 1.0)
            )
    
    # Create spatial relationships from scene graphs
    edge_count = 0
    with driver.session() as session:
        for sg in scene_graphs:
            if sg["frame_id"] not in sampled_frames:
                continue
            
            for edge in sg.get("edges", []):
                session.run("""
                    MATCH (s:MemoryObject {id: $subject, episode: $episode})
                    MATCH (o:MemoryObject {id: $object, episode: $episode})
                    MATCH (f:Frame {id: $frame_id, episode: $episode})
                    MERGE (s)-[r:RELATES {frame: $frame_id, predicate: $predicate}]->(o)
                    SET r.confidence = $confidence
                """,
                    subject=edge["subject"],
                    object=edge["object"],
                    predicate=edge["predicate"],
                    frame_id=sg["frame_id"],
                    episode=args.episode,
                    confidence=edge.get("confidence", 1.0)
                )
                edge_count += 1
    
    print(f"  Created {edge_count} RELATES edges")
    
    # Create temporal NEXT relationships between frames
    with driver.session() as session:
        for i in range(len(sampled_frames) - 1):
            session.run("""
                MATCH (f1:Frame {id: $id1, episode: $episode})
                MATCH (f2:Frame {id: $id2, episode: $episode})
                MERGE (f1)-[:NEXT]->(f2)
            """,
                id1=sampled_frames[i],
                id2=sampled_frames[i + 1],
                episode=args.episode
            )
    
    print(f"  Created temporal NEXT edges")
    
    driver.close()
    
    # Update status
    meta["status"]["exported"] = True
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"\n  ✓ Export complete")
    print(f"  Query with: orion query --episode {args.episode} --cypher 'MATCH (n) RETURN n LIMIT 10'")
    
    return 0
