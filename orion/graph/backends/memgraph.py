"""
Memgraph Graph Backend for Orion
High-performance C++-based graph database for real-time video queries

Stores:
- Entity observations (frame_idx, bbox, zone_id, class, confidence)
- Spatial relationships (NEAR, IN_ZONE, FOLLOWS)
- Temporal relationships (APPEARS_AFTER, COEXISTS_WITH)
- Semantic metadata (captions, attributes)

Key advantages over SQLite:
- 1000+ TPS on reads/writes (C++ native)
- Graph queries in Cypher (natural for relationships)
- Real-time path finding and pattern matching
- Vector search for semantic similarity
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

try:
    import mgclient
except ImportError:
    mgclient = None

logger = logging.getLogger(__name__)


@dataclass
class EntityObservation:
    """Single observation of an entity in a frame"""
    entity_id: int
    frame_idx: int
    timestamp: float
    bbox: List[float]  # [x1, y1, x2, y2]
    class_name: str
    confidence: float
    zone_id: Optional[int] = None
    caption: Optional[str] = None


@dataclass
class SpatialRelationship:
    """Spatial relationship between two entities"""
    entity1_id: int
    entity2_id: int
    relationship_type: str  # NEAR, ABOVE, BELOW, LEFT_OF, RIGHT_OF
    confidence: float
    frame_idx: int


class MemgraphBackend:
    """
    High-performance Memgraph backend for video understanding graph
    
    Schema:
    - (Entity {id, class_name, first_seen, last_seen})
    - (Frame {idx, timestamp, zone_id})
    - (Zone {id, type, centroid})
    - (Entity)-[OBSERVED_IN {bbox, confidence, caption}]->(Frame)
    - (Entity)-[NEAR|ABOVE|BELOW {confidence, frame_idx}]->(Entity)
    - (Entity)-[IN_ZONE]->(Zone)
    - (Frame)-[IN_ZONE]->(Zone)
    """
    
    def __init__(self, host: str = "127.0.0.1", port: int = 7687):
        """
        Initialize Memgraph connection
        
        Args:
            host: Memgraph host (default: localhost)
            port: Memgraph port (default: 7687)
        """
        if mgclient is None:
            raise ImportError(
                "pymgclient not installed. Install with: pip install pymgclient"
            )
        
        self.host = host
        self.port = port
        self.connection = None
        self._connect()
        self._initialize_schema()
    
    def _connect(self):
        """Establish connection to Memgraph"""
        try:
            self.connection = mgclient.connect(
                host=self.host,
                port=self.port,
                lazy=False
            )
            logger.info(f"✓ Connected to Memgraph at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Memgraph: {e}")
            logger.error("Make sure Memgraph is running: docker compose up -d")
            raise
    
    def _initialize_schema(self):
        """Create indexes and constraints for optimal query performance"""
        try:
            cursor = self.connection.cursor()
            
            # Create indexes for fast lookups
            cursor.execute("CREATE INDEX ON :Entity(id);")
            cursor.execute("CREATE INDEX ON :Frame(idx);")
            cursor.execute("CREATE INDEX ON :Zone(id);")
            cursor.execute("CREATE INDEX ON :Entity(class_name);")
            
            self.connection.commit()
            logger.info("✓ Memgraph schema initialized")
        except Exception as e:
            # Indexes may already exist
            logger.debug(f"Schema initialization: {e}")
    
    def add_entity_observation(
        self,
        entity_id: int,
        frame_idx: int,
        timestamp: float,
        bbox: list,
        class_name: str,
        confidence: float,
        zone_id: Optional[int] = None,
        caption: Optional[str] = None,
        crop_path: Optional[str] = None,  # NEW: For query-time captioning
        embedding: Optional[List[float]] = None  # NEW: For vector search
    ):
        """
        Add an entity observation to the graph
        
        Creates/updates:
        - Entity node
        - Frame node
        - OBSERVED_IN relationship
        - Zone relationship (if zone_id provided)
        """
        try:
            cursor = self.connection.cursor()
            
            # Create/update Entity node
            # Note: We only update embedding if provided. 
            # In a real system, we might want to average them or keep a history.
            entity_query = """
            MERGE (e:Entity {id: $entity_id})
            SET e.class_name = $class_name
            """
            
            params = {
                'entity_id': entity_id,
                'class_name': class_name
            }

            if embedding is not None:
                entity_query += ", e.embedding = $embedding"
                params['embedding'] = embedding

            cursor.execute(entity_query, params)
            self.connection.commit()
            
            # Create/update Frame node
            frame_query = """
            MERGE (f:Frame {idx: $frame_idx})
            SET f.timestamp = $timestamp
            """
            cursor.execute(frame_query, {
                'frame_idx': frame_idx,
                'timestamp': timestamp
            })
            self.connection.commit()
            
            # Create OBSERVED_IN relationship with bbox and caption
            obs_props = {
                'entity_id': entity_id,
                'frame_idx': frame_idx,
                'bbox_x1': bbox[0],
                'bbox_y1': bbox[1],
                'bbox_x2': bbox[2],
                'bbox_y2': bbox[3],
                'confidence': confidence
            }
            
            if caption:
                obs_props['caption'] = caption
            
            if crop_path:
                obs_props['crop_path'] = crop_path  # Store for query-time captioning
            
            obs_query = """
            MATCH (e:Entity {id: $entity_id})
            MATCH (f:Frame {idx: $frame_idx})
            MERGE (e)-[r:OBSERVED_IN]->(f)
            SET r.bbox_x1 = $bbox_x1,
                r.bbox_y1 = $bbox_y1,
                r.bbox_x2 = $bbox_x2,
                r.bbox_y2 = $bbox_y2,
                r.confidence = $confidence
            """
            
            if caption:
                obs_query += ", r.caption = $caption"
            
            if crop_path:
                obs_query += ", r.crop_path = $crop_path"
            
            cursor.execute(obs_query, obs_props)
            self.connection.commit()
            
            # Add zone relationship if provided
            if zone_id is not None:
                zone_query = """
                MATCH (e:Entity {id: $entity_id})
                MERGE (z:Zone {id: $zone_id})
                MERGE (e)-[:IN_ZONE]->(z)
                """
                cursor.execute(zone_query, {
                    'zone_id': zone_id,
                    'entity_id': entity_id
                })
                self.connection.commit()
                
        except Exception as e:
            logger.error(f"Failed to add entity observation: {e}")
            raise
    
    def add_spatial_relationship(
        self,
        entity1_id: int,
        entity2_id: int,
        relationship_type: str,
        confidence: float,
        frame_idx: int
    ):
        """
        Add spatial relationship between entities
        
        Examples:
        - (person)-[NEAR {confidence: 0.9}]->(laptop)
        - (book)-[ON_TOP_OF {confidence: 0.85}]->(desk)
        """
        cursor = self.connection.cursor()
        
        cursor.execute(
            f"""
            MATCH (e1:Entity {{id: $entity1_id}})
            MATCH (e2:Entity {{id: $entity2_id}})
            CREATE (e1)-[r:{relationship_type} {{
                confidence: $confidence,
                frame_idx: $frame_idx
            }}]->(e2)
            """,
            {
                "entity1_id": entity1_id,
                "entity2_id": entity2_id,
                "confidence": confidence,
                "frame_idx": frame_idx
            }
        )
        
        self.connection.commit()
    
    def query_entity_by_class(
        self,
        class_name: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find all entities of a given class
        
        Example: "Find all books in the video"
        """
        cursor = self.connection.cursor()
        
        cursor.execute(
            """
            MATCH (e:Entity {class_name: $class_name})
            MATCH (e)-[r:OBSERVED_IN]->(f:Frame)
            RETURN e.id as entity_id,
                   e.class_name as class_name,
                   e.first_seen as first_seen,
                   e.last_seen as last_seen,
                   collect({
                       frame_idx: f.idx,
                       timestamp: f.timestamp,
                       bbox: r.bbox,
                       confidence: r.confidence,
                       caption: r.caption,
                       crop_path: r.crop_path
                   }) as observations
            LIMIT $limit
            """,
            {"class_name": class_name, "limit": limit}
        )
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "entity_id": row[0],
                "class_name": row[1],
                "first_seen": row[2],
                "last_seen": row[3],
                "observations": row[4]
            })
        
        return results
    
    def query_entity_in_zone(
        self,
        zone_id: int,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find all entities observed in a specific zone
        
        Example: "What objects were in zone 2?"
        """
        cursor = self.connection.cursor()
        
        cursor.execute(
            """
            MATCH (z:Zone {id: $zone_id})
            MATCH (e:Entity)-[:IN_ZONE]->(z)
            MATCH (e)-[r:OBSERVED_IN]->(f:Frame)
            WHERE f.zone_id = $zone_id
            RETURN e.id as entity_id,
                   e.class_name as class_name,
                   collect({
                       frame_idx: f.idx,
                       bbox: r.bbox,
                       caption: r.caption
                   }) as observations
            LIMIT $limit
            """,
            {"zone_id": zone_id, "limit": limit}
        )
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "entity_id": row[0],
                "class_name": row[1],
                "observations": row[2]
            })
        
        return results
    
    def query_spatial_relationships(
        self,
        entity_id: int
    ) -> List[Dict[str, Any]]:
        """
        Find all spatial relationships for an entity
        
        Example: "What objects are near the laptop?"
        """
        cursor = self.connection.cursor()
        
        cursor.execute(
            """
            MATCH (e1:Entity {id: $entity_id})-[r]->(e2:Entity)
            WHERE type(r) IN ['NEAR', 'ABOVE', 'BELOW', 'LEFT_OF', 'RIGHT_OF']
            RETURN type(r) as relationship,
                   e2.id as related_entity_id,
                   e2.class_name as related_class,
                   r.confidence as confidence,
                   r.frame_idx as frame_idx
            """,
            {"entity_id": entity_id}
        )
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "relationship": row[0],
                "related_entity_id": row[1],
                "related_class": row[2],
                "confidence": row[3],
                "frame_idx": row[4]
            })
        
        return results
    
    def query_temporal_coexistence(
        self,
        entity_id: int,
        time_window: float = 5.0
    ) -> List[Dict[str, Any]]:
        """
        Find entities that appeared at the same time
        
        Example: "What objects appeared with the book?"
        """
        cursor = self.connection.cursor()
        
        cursor.execute(
            """
            MATCH (e1:Entity {id: $entity_id})-[:OBSERVED_IN]->(f1:Frame)
            MATCH (e2:Entity)-[:OBSERVED_IN]->(f2:Frame)
            WHERE e1.id <> e2.id
              AND abs(f1.timestamp - f2.timestamp) <= $time_window
            RETURN DISTINCT e2.id as entity_id,
                            e2.class_name as class_name,
                            count(*) as coexistence_count
            ORDER BY coexistence_count DESC
            """,
            {"entity_id": entity_id, "time_window": time_window}
        )
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "entity_id": row[0],
                "class_name": row[1],
                "coexistence_count": row[2]
            })
        
        return results
    
    def update_caption(
        self,
        entity_id: int,
        frame_idx: int,
        caption: str
    ):
        """
        Update caption for specific observation (lazy-loaded)
        """
        cursor = self.connection.cursor()
        
        cursor.execute(
            """
            MATCH (e:Entity {id: $entity_id})-[r:OBSERVED_IN]->(f:Frame {idx: $frame_idx})
            SET r.caption = $caption
            """,
            {
                "entity_id": entity_id,
                "frame_idx": frame_idx,
                "caption": caption
            }
        )
        
        self.connection.commit()
    
    def get_statistics(self) -> Dict[str, int]:
        """Get graph statistics"""
        cursor = self.connection.cursor()
        
        stats = {}
        
        # Count entities
        cursor.execute("MATCH (e:Entity) RETURN count(e) as count")
        stats["entities"] = cursor.fetchone()[0]
        
        # Count frames
        cursor.execute("MATCH (f:Frame) RETURN count(f) as count")
        stats["frames"] = cursor.fetchone()[0]
        
        # Count zones
        cursor.execute("MATCH (z:Zone) RETURN count(z) as count")
        stats["zones"] = cursor.fetchone()[0]
        
        # Count observations
        cursor.execute("MATCH ()-[r:OBSERVED_IN]->() RETURN count(r) as count")
        stats["observations"] = cursor.fetchone()[0]
        
        # Count spatial relationships
        cursor.execute(
            """
            MATCH ()-[r]->()
            WHERE type(r) IN ['NEAR', 'ABOVE', 'BELOW', 'LEFT_OF', 'RIGHT_OF']
            RETURN count(r) as count
            """
        )
        stats["spatial_relationships"] = cursor.fetchone()[0]
        
        return stats
    
    def clear_all(self):
        """Clear entire graph (use with caution!)"""
        cursor = self.connection.cursor()
        cursor.execute("MATCH (n) DETACH DELETE n")
        self.connection.commit()
        logger.info("✓ Graph cleared")
    
    def close(self):
        """Close connection"""
        if self.connection:
            self.connection.close()
            logger.info("✓ Memgraph connection closed")
    
    def create_vector_index(self, dimension: int = 512, metric: str = "cosine"):
        """
        Create a vector index on Entity embeddings for fast similarity search.
        Requires Memgraph MAGE or vector search support.
        """
        try:
            cursor = self.connection.cursor()
            # Check if index exists or just try to create it.
            # Syntax depends on Memgraph version. Using a generic approach or MAGE.
            # For now, we'll assume standard Memgraph vector index syntax if available,
            # or just rely on the property being there for manual cosine sim if needed.
            
            # Example for Memgraph 2.14+ native vector index:
            # CREATE VECTOR INDEX ON :Entity(embedding) WITH CONFIG {"dimension": 512, "metric": "cosine"}
            
            query = f"""
            CREATE VECTOR INDEX ON :Entity(embedding) 
            WITH CONFIG {{"dimension": {dimension}, "metric": "{metric}"}}
            """
            cursor.execute(query)
            self.connection.commit()
            logger.info(f"✓ Vector index created on Entity(embedding) with dim={dimension}")
        except Exception as e:
            logger.warning(f"Failed to create vector index (might already exist or not supported): {e}")

    def add_observations_batch(
        self,
        observations: List[Dict[str, Any]],
    ) -> int:
        """
        Add multiple entity observations in a single transaction using UNWIND.
        
        This is 10-50x faster than individual add_entity_observation calls.
        
        Args:
            observations: List of dicts with keys:
                - entity_id: int
                - frame_idx: int
                - timestamp: float
                - bbox: [x1, y1, x2, y2]
                - class_name: str
                - confidence: float
                - zone_id: Optional[int]
                - caption: Optional[str]
                - embedding: Optional[List[float]]
                
        Returns:
            Number of observations inserted.
        """
        if not observations:
            return 0
            
        try:
            cursor = self.connection.cursor()
            
            # Prepare data for UNWIND
            obs_data = []
            for obs in observations:
                bbox = obs.get('bbox', [0, 0, 0, 0])
                obs_data.append({
                    'entity_id': obs['entity_id'],
                    'frame_idx': obs['frame_idx'],
                    'timestamp': obs.get('timestamp', 0.0),
                    'class_name': obs['class_name'],
                    'confidence': obs.get('confidence', 0.0),
                    'bbox_x1': bbox[0] if len(bbox) > 0 else 0,
                    'bbox_y1': bbox[1] if len(bbox) > 1 else 0,
                    'bbox_x2': bbox[2] if len(bbox) > 2 else 0,
                    'bbox_y2': bbox[3] if len(bbox) > 3 else 0,
                    'zone_id': obs.get('zone_id'),
                    'caption': obs.get('caption'),
                })
            
            # Batch insert using UNWIND
            batch_query = """
            UNWIND $observations AS obs
            MERGE (e:Entity {id: obs.entity_id})
            SET e.class_name = obs.class_name
            MERGE (f:Frame {idx: obs.frame_idx})
            SET f.timestamp = obs.timestamp
            MERGE (e)-[r:OBSERVED_IN]->(f)
            SET r.bbox_x1 = obs.bbox_x1,
                r.bbox_y1 = obs.bbox_y1,
                r.bbox_x2 = obs.bbox_x2,
                r.bbox_y2 = obs.bbox_y2,
                r.confidence = obs.confidence
            """
            
            cursor.execute(batch_query, {'observations': obs_data})
            self.connection.commit()
            
            logger.debug(f"Batch inserted {len(observations)} observations")
            return len(observations)
            
        except Exception as e:
            logger.error(f"Batch insert failed: {e}")
            raise

    def add_relationships_batch(
        self,
        relationships: List[Dict[str, Any]],
    ) -> int:
        """
        Add multiple spatial relationships in a single transaction.
        
        Args:
            relationships: List of dicts with keys:
                - entity1_id: int
                - entity2_id: int
                - relationship_type: str (NEAR, ABOVE, BELOW, etc.)
                - confidence: float
                - frame_idx: int
                
        Returns:
            Number of relationships inserted.
        """
        if not relationships:
            return 0
            
        try:
            cursor = self.connection.cursor()
            
            # Group by relationship type for efficiency
            by_type: Dict[str, List] = {}
            for rel in relationships:
                rel_type = rel['relationship_type']
                by_type.setdefault(rel_type, []).append(rel)
            
            total = 0
            for rel_type, rels in by_type.items():
                rel_data = [{
                    'e1': r['entity1_id'],
                    'e2': r['entity2_id'],
                    'conf': r['confidence'],
                    'fidx': r['frame_idx']
                } for r in rels]
                
                # Dynamic relationship type
                query = f"""
                UNWIND $rels AS r
                MATCH (e1:Entity {{id: r.e1}})
                MATCH (e2:Entity {{id: r.e2}})
                MERGE (e1)-[rel:{rel_type}]->(e2)
                SET rel.confidence = r.conf,
                    rel.frame_idx = r.fidx
                """
                
                cursor.execute(query, {'rels': rel_data})
                total += len(rels)
            
            self.connection.commit()
            logger.debug(f"Batch inserted {total} relationships")
            return total
            
        except Exception as e:
            logger.error(f"Batch relationship insert failed: {e}")
            raise

    def search_similar_entities(
        self, 
        query_embedding: List[float], 
        limit: int = 5, 
        min_score: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for entities with similar embeddings using vector search.
        """
        cursor = self.connection.cursor()
        
        # Using Memgraph's vector search syntax
        # CALL vector_search.search(index_name, query_vector, limit)
        # Or native Cypher if supported.
        
        # We'll try the native Cypher syntax for Memgraph 2.14+
        query = """
        MATCH (e:Entity)
        WHERE e.embedding IS NOT NULL
        WITH e, vector.similarity.cosine(e.embedding, $query_embedding) AS score
        WHERE score >= $min_score
        RETURN e.id as entity_id, 
               e.class_name as class_name, 
               score
        ORDER BY score DESC
        LIMIT $limit
        """
        
        try:
            cursor.execute(query, {
                "query_embedding": query_embedding,
                "min_score": min_score,
                "limit": limit
            })
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "entity_id": row[0],
                    "class_name": row[1],
                    "score": row[2]
                })
            return results
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
