#!/usr/bin/env python3
"""
Knowledge Graph Visualization and Exploration
==============================================

Interactive tool to visualize and explore the enhanced knowledge graph.

Features:
- View graph statistics
- List all scenes, entities, relationships
- Visualize spatial relationships
- Export subgraphs
- Generate reports

Usage:
    python scripts/explore_kg.py
    python scripts/explore_kg.py --export scene_graph.json
    python scripts/explore_kg.py --stats
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from neo4j import GraphDatabase

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


class KnowledgeGraphExplorer:
    """Interactive knowledge graph explorer"""
    
    def __init__(
        self,
        neo4j_uri: str = "neo4j://127.0.0.1:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "orion123"
    ):
        self.uri = neo4j_uri
        self.user = neo4j_user
        self.password = neo4j_password
        self.driver = None
    
    def connect(self) -> bool:
        """Connect to Neo4j"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("✓ Connected to Neo4j")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to connect to Neo4j: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics"""
        if not self.driver:
            return {}
        
        stats = {}
        
        with self.driver.session() as session:
            # Entity stats
            result = session.run("""
                MATCH (e:Entity)
                RETURN count(e) as entity_count,
                       collect(DISTINCT e.class) as entity_classes
            """).single()
            
            stats['total_entities'] = result['entity_count']
            stats['entity_classes'] = result['entity_classes']
            
            # Scene stats
            result = session.run("""
                MATCH (s:Scene)
                RETURN count(s) as scene_count,
                       collect(DISTINCT s.scene_type) as scene_types
            """).single()
            
            stats['total_scenes'] = result['scene_count']
            stats['scene_types'] = result['scene_types']
            
            # Relationship stats
            result = session.run("""
                MATCH ()-[r:SPATIAL_REL]->()
                RETURN count(r) as spatial_count,
                       collect(DISTINCT r.type) as spatial_types
            """).single()
            
            stats['spatial_relationships'] = result['spatial_count']
            stats['spatial_types'] = result['spatial_types']
            
            # Causal relationships
            result = session.run("""
                MATCH ()-[r:POTENTIALLY_CAUSED]->()
                RETURN count(r) as causal_count
            """).single()
            
            stats['causal_chains'] = result['causal_count']
            
            # Scene transitions
            result = session.run("""
                MATCH ()-[r:TRANSITIONS_TO]->()
                RETURN count(r) as transition_count
            """).single()
            
            stats['scene_transitions'] = result['transition_count']
        
        return stats
    
    def list_scenes(self) -> List[Dict[str, Any]]:
        """List all scenes with details"""
        if not self.driver:
            return []
        
        with self.driver.session() as session:
            results = session.run("""
                MATCH (s:Scene)
                RETURN s.id as id,
                       s.scene_type as type,
                       s.confidence as confidence,
                       s.timestamp_start as start,
                       s.timestamp_end as end,
                       s.dominant_objects as objects
                ORDER BY s.timestamp_start
            """).data()
            
            return results
    
    def list_entities(self, scene_type: str = None) -> List[Dict[str, Any]]:
        """List entities, optionally filtered by scene type"""
        if not self.driver:
            return []
        
        with self.driver.session() as session:
            if scene_type:
                results = session.run("""
                    MATCH (e:Entity)-[:APPEARS_IN]->(s:Scene)
                    WHERE s.scene_type = $scene_type
                    RETURN DISTINCT e.id as id,
                           e.class as class,
                           e.appearance_count as count,
                           e.description as description
                    ORDER BY e.appearance_count DESC
                """, {'scene_type': scene_type}).data()
            else:
                results = session.run("""
                    MATCH (e:Entity)
                    RETURN e.id as id,
                           e.class as class,
                           e.appearance_count as count,
                           e.description as description
                    ORDER BY e.appearance_count DESC
                """).data()
            
            return results
    
    def get_spatial_relationships(self, entity_class: str = None) -> List[Dict[str, Any]]:
        """Get spatial relationships, optionally filtered by entity class"""
        if not self.driver:
            return []
        
        with self.driver.session() as session:
            if entity_class:
                results = session.run("""
                    MATCH (e1:Entity)-[r:SPATIAL_REL]->(e2:Entity)
                    WHERE e1.class = $class OR e2.class = $class
                    RETURN e1.class as entity_a,
                           e2.class as entity_b,
                           r.type as relationship,
                           r.confidence as confidence,
                           r.co_occurrence as co_occurrence
                    ORDER BY r.confidence DESC
                """, {'class': entity_class}).data()
            else:
                results = session.run("""
                    MATCH (e1:Entity)-[r:SPATIAL_REL]->(e2:Entity)
                    RETURN e1.class as entity_a,
                           e2.class as entity_b,
                           r.type as relationship,
                           r.confidence as confidence,
                           r.co_occurrence as co_occurrence
                    ORDER BY r.confidence DESC
                    LIMIT 50
                """).data()
            
            return results
    
    def get_scene_timeline(self) -> List[Dict[str, Any]]:
        """Get chronological scene timeline"""
        if not self.driver:
            return []
        
        with self.driver.session() as session:
            results = session.run("""
                MATCH (s:Scene)
                OPTIONAL MATCH (s)-[:TRANSITIONS_TO]->(next:Scene)
                RETURN s.id as id,
                       s.scene_type as type,
                       s.timestamp_start as start,
                       s.timestamp_end as end,
                       s.dominant_objects as objects,
                       next.id as next_scene
                ORDER BY s.timestamp_start
            """).data()
            
            return results
    
    def export_subgraph(
        self,
        output_path: Path,
        scene_type: str = None,
        entity_class: str = None
    ):
        """Export subgraph to JSON"""
        if not self.driver:
            logger.error("Not connected to Neo4j")
            return
        
        subgraph = {
            'scenes': [],
            'entities': [],
            'spatial_relationships': [],
            'scene_transitions': []
        }
        
        with self.driver.session() as session:
            # Export scenes
            if scene_type:
                scenes = session.run("""
                    MATCH (s:Scene)
                    WHERE s.scene_type = $type
                    RETURN s
                """, {'type': scene_type}).data()
            else:
                scenes = session.run("MATCH (s:Scene) RETURN s").data()
            
            subgraph['scenes'] = [dict(s['s']) for s in scenes]
            
            # Export entities
            if entity_class:
                entities = session.run("""
                    MATCH (e:Entity)
                    WHERE e.class = $class
                    RETURN e
                """, {'class': entity_class}).data()
            else:
                entities = session.run("MATCH (e:Entity) RETURN e").data()
            
            subgraph['entities'] = [dict(e['e']) for e in entities]
            
            # Export relationships
            rels = session.run("""
                MATCH (e1:Entity)-[r:SPATIAL_REL]->(e2:Entity)
                RETURN e1.id as source, e2.id as target, properties(r) as props
            """).data()
            
            subgraph['spatial_relationships'] = rels
            
            # Export scene transitions
            trans = session.run("""
                MATCH (s1:Scene)-[r:TRANSITIONS_TO]->(s2:Scene)
                RETURN s1.id as source, s2.id as target, properties(r) as props
            """).data()
            
            subgraph['scene_transitions'] = trans
        
        with open(output_path, 'w') as f:
            json.dump(subgraph, f, indent=2, default=str)
        
        logger.info(f"✓ Exported subgraph to {output_path}")
        logger.info(f"  Scenes: {len(subgraph['scenes'])}")
        logger.info(f"  Entities: {len(subgraph['entities'])}")
        logger.info(f"  Spatial Relationships: {len(subgraph['spatial_relationships'])}")
        logger.info(f"  Scene Transitions: {len(subgraph['scene_transitions'])}")
    
    def print_statistics(self):
        """Print formatted statistics"""
        stats = self.get_statistics()
        
        if not stats:
            logger.error("No statistics available")
            return
        
        print("\n" + "="*80)
        print("KNOWLEDGE GRAPH STATISTICS")
        print("="*80)
        
        print(f"\n📊 Overview:")
        print(f"  Entities: {stats.get('total_entities', 0)}")
        print(f"  Scenes: {stats.get('total_scenes', 0)}")
        print(f"  Spatial Relationships: {stats.get('spatial_relationships', 0)}")
        print(f"  Causal Chains: {stats.get('causal_chains', 0)}")
        print(f"  Scene Transitions: {stats.get('scene_transitions', 0)}")
        
        print(f"\n🏷️  Entity Classes:")
        for cls in stats.get('entity_classes', [])[:10]:
            print(f"  - {cls}")
        if len(stats.get('entity_classes', [])) > 10:
            print(f"  ... and {len(stats['entity_classes']) - 10} more")
        
        print(f"\n🎬 Scene Types:")
        for scene_type in stats.get('scene_types', []):
            print(f"  - {scene_type}")
        
        print(f"\n🔗 Spatial Relationship Types:")
        for rel_type in stats.get('spatial_types', []):
            print(f"  - {rel_type}")
        
        print("\n" + "="*80 + "\n")
    
    def print_scene_timeline(self):
        """Print formatted scene timeline"""
        timeline = self.get_scene_timeline()
        
        if not timeline:
            logger.error("No timeline available")
            return
        
        print("\n" + "="*80)
        print("SCENE TIMELINE")
        print("="*80 + "\n")
        
        for i, scene in enumerate(timeline, 1):
            start = scene.get('start', 0)
            end = scene.get('end', 0)
            duration = end - start
            scene_type = scene.get('type', 'unknown')
            objects = scene.get('objects', [])
            
            print(f"{i}. {scene_type.upper()} ({start:.1f}s - {end:.1f}s, {duration:.1f}s)")
            print(f"   Objects: {', '.join(objects[:5])}")
            
            if scene.get('next_scene'):
                print(f"   ↓ transitions to →")
        
        print("\n" + "="*80 + "\n")
    
    def print_spatial_network(self, top_n: int = 20):
        """Print spatial relationship network"""
        relationships = self.get_spatial_relationships()[:top_n]
        
        if not relationships:
            logger.error("No spatial relationships available")
            return
        
        print("\n" + "="*80)
        print(f"SPATIAL RELATIONSHIP NETWORK (Top {top_n})")
        print("="*80 + "\n")
        
        for i, rel in enumerate(relationships, 1):
            entity_a = rel.get('entity_a', 'unknown')
            entity_b = rel.get('entity_b', 'unknown')
            rel_type = rel.get('relationship', 'unknown')
            confidence = rel.get('confidence', 0)
            co_occur = rel.get('co_occurrence', 0)
            
            print(f"{i}. {entity_a} --[{rel_type}]--> {entity_b}")
            print(f"   Confidence: {confidence:.2f}, Co-occurrence: {co_occur}")
        
        print("\n" + "="*80 + "\n")
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()


def main():
    parser = argparse.ArgumentParser(
        description="Explore the enhanced knowledge graph"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show graph statistics"
    )
    parser.add_argument(
        "--timeline",
        action="store_true",
        help="Show scene timeline"
    )
    parser.add_argument(
        "--spatial",
        action="store_true",
        help="Show spatial relationship network"
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Export subgraph to JSON file"
    )
    parser.add_argument(
        "--scene-type",
        type=str,
        help="Filter by scene type (for export)"
    )
    parser.add_argument(
        "--entity-class",
        type=str,
        help="Filter by entity class (for export)"
    )
    
    args = parser.parse_args()
    
    # Create explorer
    explorer = KnowledgeGraphExplorer()
    
    if not explorer.connect():
        logger.error("Cannot connect to Neo4j. Is it running?")
        return 1
    
    # Execute commands
    if args.stats or not any([args.timeline, args.spatial, args.export]):
        # Default: show stats
        explorer.print_statistics()
    
    if args.timeline:
        explorer.print_scene_timeline()
    
    if args.spatial:
        explorer.print_spatial_network()
    
    if args.export:
        output_path = Path(args.export)
        explorer.export_subgraph(
            output_path,
            scene_type=args.scene_type,
            entity_class=args.entity_class
        )
    
    explorer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
