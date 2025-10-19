"""
Temporal Knowledge Graph Builder
=================================

Builds rich temporal knowledge graph from tracked entities with:
- Entity nodes with full timeline
- Frame nodes for temporal anchoring
- Appearance relationships linking entities to frames
- Spatial relationships (co-occurrence, proximity)
- Temporal patterns (movement, state changes)
- Causal relationships (using CIS scoring)

Author: Orion Research Team
Date: October 2025
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import OrionConfig
from .config_manager import ConfigManager
from .neo4j_manager import Neo4jManager

logger = logging.getLogger('TemporalGraph')


class TemporalGraphBuilder:
    """Builds rich temporal knowledge graph in Neo4j"""
    
    def __init__(
        self,
        config: Optional[OrionConfig] = None,
        neo4j_manager: Optional[Neo4jManager] = None
    ):
        self.config = config or ConfigManager.get_config()
        self.neo4j_manager: Optional[Neo4jManager] = neo4j_manager
        self.driver: Optional[Any] = None
    
    def connect(self) -> bool:
        """Connect to Neo4j"""
        try:
            if self.neo4j_manager is None:
                neo4j_config = self.config.neo4j
                self.neo4j_manager = Neo4jManager(
                    uri=neo4j_config.uri,
                    user=neo4j_config.user,
                    password=neo4j_config.password,
                )

            if not self.neo4j_manager.connect():
                return False

            # Test connection
            self.driver = self.neo4j_manager.driver
            logger.info("✓ Connected to Neo4j")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to connect to Neo4j: {e}")
            return False
    
    def build_graph(
        self,
        entities: List[Any],
        observations: List[Any]
    ) -> Dict[str, int]:
        """Build complete temporal knowledge graph"""
        logger.info("="*80)
        logger.info("PHASE 4: TEMPORAL KNOWLEDGE GRAPH")
        logger.info("="*80)
        
        if not self.driver:
            if not self.connect():
                raise RuntimeError("Cannot connect to Neo4j")
        
        stats = {
            'entity_nodes': 0,
            'frame_nodes': 0,
            'appearance_rels': 0,
            'spatial_rels': 0,
            'movement_rels': 0,
            'state_change_rels': 0
        }
        
        with self.driver.session() as session:
            # Initialize schema
            self._initialize_schema(session)
            
            # 1. Create entity nodes
            logger.info("Creating entity nodes...")
            stats['entity_nodes'] = self._create_entity_nodes(session, entities)
            
            # 2. Create frame nodes and appearance relationships
            logger.info("Creating frame nodes and appearance relationships...")
            frame_stats = self._create_frame_relationships(session, entities)
            stats['frame_nodes'] = frame_stats['frames']
            stats['appearance_rels'] = frame_stats['appearances']
            
            # 3. Detect and create spatial relationships
            logger.info("Detecting spatial relationships...")
            stats['spatial_rels'] = self._create_spatial_relationships(session, entities)
            
            # 4. Detect and create temporal patterns
            logger.info("Detecting temporal patterns...")
            movement_stats = self._create_temporal_patterns(session, entities)
            stats['movement_rels'] = movement_stats['movements']
            stats['state_change_rels'] = movement_stats['state_changes']
        
        logger.info("\n" + "="*80)
        logger.info("KNOWLEDGE GRAPH COMPLETE")
        logger.info("="*80)
        logger.info(f"Entity nodes: {stats['entity_nodes']}")
        logger.info(f"Frame nodes: {stats['frame_nodes']}")
        logger.info(f"Appearance relationships: {stats['appearance_rels']}")
        logger.info(f"Spatial relationships: {stats['spatial_rels']}")
        logger.info(f"Movement relationships: {stats['movement_rels']}")
        logger.info(f"State change relationships: {stats['state_change_rels']}")
        logger.info("="*80 + "\n")
        
        return stats
    
    def _initialize_schema(self, session):
        """Initialize Neo4j schema with constraints and indexes"""
        # Constraints
        constraints = [
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT frame_number IF NOT EXISTS FOR (f:Frame) REQUIRE f.number IS UNIQUE"
        ]
        
        for constraint in constraints:
            try:
                session.run(constraint)
            except Exception:
                pass  # Constraint may already exist
        
        # Indexes
        indexes = [
            "CREATE INDEX entity_class IF NOT EXISTS FOR (e:Entity) ON (e.class)",
            "CREATE INDEX frame_timestamp IF NOT EXISTS FOR (f:Frame) ON (f.timestamp)",
            "CREATE INDEX entity_first_seen IF NOT EXISTS FOR (e:Entity) ON (e.first_seen)"
        ]
        
        for index in indexes:
            try:
                session.run(index)
            except Exception:
                pass  # Index may already exist
        
        # Vector index for embeddings
        try:
            session.run("""
                CREATE VECTOR INDEX entity_embedding IF NOT EXISTS
                FOR (e:Entity) ON (e.embedding)
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 2048,
                        `vector.similarity_function`: 'cosine'
                    }
                }
            """)
        except Exception:
            pass  # Vector index may not be supported
    
    def _create_entity_nodes(self, session, entities: List[Any]) -> int:
        """Create entity nodes with full metadata"""
        for entity in entities:
            session.run("""
                MERGE (e:Entity {id: $id})
                SET e.class = $class,
                    e.description = $description,
                    e.first_seen = $first_seen,
                    e.last_seen = $last_seen,
                    e.duration = $duration,
                    e.appearance_count = $count,
                    e.described_from_frame = $described_from,
                    e.embedding = $embedding
            """, {
                'id': entity.id,
                'class': entity.class_name,
                'description': entity.description,
                'first_seen': entity.first_seen,
                'last_seen': entity.last_seen,
                'duration': entity.duration,
                'count': entity.appearance_count,
                'described_from': entity.described_from_frame,
                'embedding': entity.average_embedding.tolist()
            })
        
        return len(entities)
    
    def _create_frame_relationships(
        self,
        session,
        entities: List[Any]
    ) -> Dict[str, int]:
        """Create frame nodes and APPEARS_IN relationships"""
        frames_created = set()
        appearance_count = 0
        
        for entity in entities:
            for obs in entity.observations:
                # Create frame node
                if obs.frame_number not in frames_created:
                    session.run("""
                        MERGE (f:Frame {number: $frame_num})
                        SET f.timestamp = $timestamp
                    """, {
                        'frame_num': obs.frame_number,
                        'timestamp': obs.timestamp
                    })
                    frames_created.add(obs.frame_number)
                
                # Create APPEARS_IN relationship
                session.run("""
                    MATCH (e:Entity {id: $entity_id})
                    MATCH (f:Frame {number: $frame_num})
                    MERGE (e)-[r:APPEARS_IN]->(f)
                    SET r.bbox = $bbox,
                        r.confidence = $confidence,
                        r.bbox_area = $area,
                        r.centrality = $centrality
                """, {
                    'entity_id': entity.id,
                    'frame_num': obs.frame_number,
                    'bbox': obs.bbox,
                    'confidence': obs.confidence,
                    'area': obs.get_bbox_area(),
                    'centrality': obs.get_centrality_score()
                })
                
                appearance_count += 1
        
        return {
            'frames': len(frames_created),
            'appearances': appearance_count
        }
    
    def _create_spatial_relationships(
        self,
        session,
        entities: List[Any]
    ) -> int:
        """Detect and create spatial relationships (co-occurrence, proximity)"""
        # Group entities by frame
        frame_entities = defaultdict(list)
        for entity in entities:
            for obs in entity.observations:
                frame_entities[obs.frame_number].append((entity, obs))
        
        spatial_rel_count = 0
        
        for frame_num, entity_obs_list in frame_entities.items():
            if len(entity_obs_list) < 2:
                continue
            
            # Check each pair of entities in this frame
            for i, (e1, obs1) in enumerate(entity_obs_list):
                for e2, obs2 in entity_obs_list[i+1:]:
                    # Calculate distance between centroids
                    c1 = obs1.get_bbox_center()
                    c2 = obs2.get_bbox_center()
                    distance = self._euclidean_distance(c1, c2)
                    
                    # Normalize by frame size
                    frame_diagonal = np.sqrt(obs1.frame_width**2 + obs1.frame_height**2)
                    normalized_distance = distance / frame_diagonal
                    
                    # Determine relationship type
                    if normalized_distance < 0.15:  # Very close
                        rel_type = "VERY_NEAR"
                    elif normalized_distance < 0.3:  # Nearby
                        rel_type = "NEAR"
                    elif normalized_distance < 0.5:  # Same region
                        rel_type = "SAME_REGION"
                    else:
                        continue  # Too far apart
                    
                    # Create spatial relationship
                    session.run(f"""
                        MATCH (e1:Entity {{id: $id1}}),
                              (e2:Entity {{id: $id2}})
                        MERGE (e1)-[r:{rel_type}]->(e2)
                        ON CREATE SET r.first_frame = $frame,
                                      r.first_time = $timestamp,
                                      r.count = 1,
                                      r.min_distance = $distance,
                                      r.avg_distance = $distance
                        ON MATCH SET r.count = r.count + 1,
                                     r.last_frame = $frame,
                                     r.last_time = $timestamp,
                                     r.min_distance = CASE WHEN $distance < r.min_distance 
                                                           THEN $distance 
                                                           ELSE r.min_distance END,
                                     r.avg_distance = (r.avg_distance * (r.count - 1) + $distance) / r.count
                    """, {
                        'id1': e1.id,
                        'id2': e2.id,
                        'frame': frame_num,
                        'timestamp': obs1.timestamp,
                        'distance': float(distance)
                    })
                    
                    spatial_rel_count += 1
        
        return spatial_rel_count
    
    def _create_temporal_patterns(
        self,
        session,
        entities: List[Any]
    ) -> Dict[str, int]:
        """Detect temporal patterns (movement, state changes)"""
        movement_count = 0
        state_change_count = 0
        
        for entity in entities:
            observations = entity.observations
            
            # Detect movements
            if len(observations) > 1:
                for i in range(len(observations) - 1):
                    curr_obs = observations[i]
                    next_obs = observations[i + 1]
                    
                    # Calculate movement
                    curr_center = curr_obs.get_bbox_center()
                    next_center = next_obs.get_bbox_center()
                    distance_moved = self._euclidean_distance(curr_center, next_center)
                    
                    # Normalize by frame size
                    frame_diagonal = np.sqrt(curr_obs.frame_width**2 + curr_obs.frame_height**2)
                    normalized_distance = distance_moved / frame_diagonal
                    
                    if normalized_distance > 0.05:  # Significant movement (5% of frame)
                        # Calculate velocity
                        time_delta = next_obs.timestamp - curr_obs.timestamp
                        velocity = distance_moved / max(time_delta, 0.001)
                        
                        # Calculate direction
                        dx = next_center[0] - curr_center[0]
                        dy = next_center[1] - curr_center[1]
                        direction_deg = np.degrees(np.arctan2(dy, dx))
                        
                        session.run("""
                            MATCH (e:Entity {id: $entity_id})
                            CREATE (m:Movement {
                                entity_id: $entity_id,
                                from_frame: $from_frame,
                                to_frame: $to_frame,
                                from_time: $from_time,
                                to_time: $to_time,
                                distance: $distance,
                                velocity: $velocity,
                                direction: $direction
                            })
                            CREATE (e)-[:HAS_MOVEMENT]->(m)
                        """, {
                            'entity_id': entity.id,
                            'from_frame': curr_obs.frame_number,
                            'to_frame': next_obs.frame_number,
                            'from_time': curr_obs.timestamp,
                            'to_time': next_obs.timestamp,
                            'distance': float(distance_moved),
                            'velocity': float(velocity),
                            'direction': float(direction_deg)
                        })
                        
                        movement_count += 1
            
            # Add state changes
            for change in entity.state_changes:
                session.run("""
                    MATCH (e:Entity {id: $entity_id})
                    CREATE (sc:StateChange {
                        entity_id: $entity_id,
                        from_frame: $from_frame,
                        to_frame: $to_frame,
                        from_time: $from_time,
                        to_time: $to_time,
                        similarity: $similarity,
                        new_description: $new_desc
                    })
                    CREATE (e)-[:HAD_STATE_CHANGE]->(sc)
                """, {
                    'entity_id': entity.id,
                    'from_frame': change['from_frame'],
                    'to_frame': change['to_frame'],
                    'from_time': change['from_time'],
                    'to_time': change['to_time'],
                    'similarity': change['similarity'],
                    'new_desc': change['new_description']
                })
                
                state_change_count += 1
        
        return {
            'movements': movement_count,
            'state_changes': state_change_count
        }
    
    @staticmethod
    def _euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    def close(self):
        """Close Neo4j connection"""
        if self.neo4j_manager:
            self.neo4j_manager.close()
            self.driver = None
            logger.info("Neo4j connection closed")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import json
    import logging
    from pathlib import Path
    
    logging.basicConfig(level=logging.INFO)
    
    # Load tracking results
    results_path = Path("data/testing/tracking_results.json")
    
    if not results_path.exists():
        logger.error(f"Results file not found: {results_path}")
        logger.info("Run tracking_engine.py first to generate results")
        exit(1)
    
    with open(results_path) as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data['entities'])} entities")
    
    # Note: This is a simplified example
    # In practice, you'd need to reconstruct Entity and Observation objects
    logger.info("To build graph, integrate with tracking_engine.py directly")
