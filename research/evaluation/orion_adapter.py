"""
Orion Knowledge Graph Adapter for Evaluation
=============================================

Converts Orion's Neo4j knowledge graph to standardized format
for benchmark comparison.

This adapter:
1. Queries Neo4j for all entities, relationships, events
2. Extracts temporal and spatial information
3. Builds standardized PredictionGraph for evaluation

Author: Orion Research Team
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

import numpy as np
from neo4j import GraphDatabase

from .benchmark_evaluator import PredictionGraph

logger = logging.getLogger("orion.evaluation.adapter")


class OrionKGAdapter:
    """
    Adapter to export Orion's Neo4j knowledge graph to evaluation format
    """
    
    def __init__(self, uri: str, user: str, password: str):
        """
        Initialize Neo4j connection
        
        Args:
            uri: Neo4j URI (e.g., "bolt://localhost:7687")
            user: Neo4j username
            password: Neo4j password
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        """Close Neo4j connection"""
        self.driver.close()
    
    def export_prediction_graph(
        self,
        video_id: str,
        include_embeddings: bool = False,
    ) -> PredictionGraph:
        """
        Export Orion's knowledge graph for a video as PredictionGraph
        
        Args:
            video_id: Identifier for the video
            include_embeddings: Whether to include context embeddings
        
        Returns:
            PredictionGraph ready for evaluation
        """
        with self.driver.session() as session:
            # 1. Export entities
            entities = self._export_entities(session, video_id, include_embeddings)
            
            # 2. Export relationships
            relationships = self._export_relationships(session, video_id)
            
            # 3. Export events
            events = self._export_events(session, video_id)
            
            # 4. Export causal links
            causal_links = self._export_causal_links(session, video_id)
        
        return PredictionGraph(
            video_id=video_id,
            entities=entities,
            relationships=relationships,
            events=events,
            causal_links=causal_links,
        )
    
    def _export_entities(
        self,
        session,
        video_id: str,
        include_embeddings: bool,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Export Entity nodes with temporal and spatial info
        
        Returns:
            Dict mapping entity_id -> entity_data
        """
        query = """
        MATCH (e:Entity)
        WHERE e.video_id = $video_id OR $video_id IS NULL
        OPTIONAL MATCH (e)-[:APPEARS_IN]->(s:Scene)
        WITH e, collect(DISTINCT s.frame_number) AS frames
        RETURN 
            id(e) AS entity_id,
            e.entity_id AS external_id,
            e.class AS class,
            e.canonical_label AS label,
            e.description AS description,
            e.confidence AS confidence,
            e.bbox AS bbox,
            e.track_id AS track_id,
            e.first_seen_frame AS first_frame,
            e.last_seen_frame AS last_frame,
            e.scene_types AS scene_types,
            frames,
            CASE WHEN $include_embeddings THEN e.context_embedding ELSE NULL END AS embedding
        """
        
        result = session.run(query, video_id=video_id, include_embeddings=include_embeddings)
        
        entities = {}
        for record in result:
            entity_id = str(record["entity_id"])
            
            # Extract bounding boxes by frame
            bboxes = {}
            if record.get("bbox"):
                # If bbox is stored per-frame, extract
                # For now, assume single bbox or we need frame-level tracking
                pass
            
            # Build frames list
            frames = record.get("frames", [])
            if not frames and record.get("first_frame") is not None:
                # Generate range from first to last
                first = record["first_frame"]
                last = record.get("last_frame", first)
                frames = list(range(first, last + 1))
            
            entity_data = {
                "entity_id": entity_id,
                "external_id": record.get("external_id"),
                "class": record.get("class", "unknown"),
                "label": record.get("label"),
                "description": record.get("description"),
                "confidence": record.get("confidence", 1.0),
                "track_id": record.get("track_id"),
                "frames": frames,
                "first_frame": record.get("first_frame"),
                "last_frame": record.get("last_frame"),
                "scene_types": record.get("scene_types", []),
                "bboxes": bboxes,
            }
            
            if include_embeddings and record.get("embedding"):
                entity_data["embedding"] = record["embedding"]
            
            entities[entity_id] = entity_data
        
        logger.info(f"Exported {len(entities)} entities for video {video_id}")
        return entities
    
    def _export_relationships(
        self,
        session,
        video_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Export relationships (SPATIAL_REL, PART_OF, etc.)
        
        Returns:
            List of relationship dicts
        """
        query = """
        MATCH (e1:Entity)-[r]->(e2:Entity)
        WHERE (e1.video_id = $video_id OR $video_id IS NULL)
          AND type(r) IN ['SPATIAL_REL', 'PART_OF', 'INTERACTS_WITH']
        RETURN 
            id(e1) AS subject_id,
            id(e2) AS object_id,
            type(r) AS predicate,
            r.relation AS relation_detail,
            r.confidence AS confidence,
            r.frame_number AS frame_number
        """
        
        result = session.run(query, video_id=video_id)
        
        relationships = []
        for record in result:
            rel = {
                "subject": str(record["subject_id"]),
                "object": str(record["object_id"]),
                "predicate": self._normalize_predicate(
                    record["predicate"],
                    record.get("relation_detail")
                ),
                "confidence": record.get("confidence", 1.0),
                "frame_number": record.get("frame_number"),
            }
            relationships.append(rel)
        
        logger.info(f"Exported {len(relationships)} relationships for video {video_id}")
        return relationships
    
    @staticmethod
    def _normalize_predicate(predicate_type: str, detail: Optional[str]) -> str:
        """Normalize relationship predicates to standard names"""
        if predicate_type == "SPATIAL_REL":
            return detail or "near"
        elif predicate_type == "PART_OF":
            return "part_of"
        elif predicate_type == "INTERACTS_WITH":
            return "interacts_with"
        else:
            return predicate_type.lower()
    
    def _export_events(
        self,
        session,
        video_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Export Event nodes with temporal bounds and involved entities
        
        Returns:
            List of event dicts
        """
        query = """
        MATCH (ev:Event)
        WHERE ev.video_id = $video_id OR $video_id IS NULL
        OPTIONAL MATCH (ev)-[:INVOLVES]->(e:Entity)
        WITH ev, collect(id(e)) AS entity_ids
        RETURN 
            id(ev) AS event_id,
            ev.event_type AS event_type,
            ev.description AS description,
            ev.start_frame AS start_frame,
            ev.end_frame AS end_frame,
            ev.confidence AS confidence,
            entity_ids
        """
        
        result = session.run(query, video_id=video_id)
        
        events = []
        for record in result:
            event = {
                "event_id": str(record["event_id"]),
                "type": record.get("event_type", "unknown"),
                "description": record.get("description"),
                "start_frame": record.get("start_frame", 0),
                "end_frame": record.get("end_frame", 0),
                "confidence": record.get("confidence", 1.0),
                "entities": [str(eid) for eid in record.get("entity_ids", [])],
            }
            events.append(event)
        
        logger.info(f"Exported {len(events)} events for video {video_id}")
        return events
    
    def _export_causal_links(
        self,
        session,
        video_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Export causal links (POTENTIALLY_CAUSED, ENABLES, etc.)
        
        Returns:
            List of causal link dicts
        """
        query = """
        MATCH (cause)-[r:POTENTIALLY_CAUSED]->(effect)
        WHERE (cause.video_id = $video_id OR $video_id IS NULL)
          AND (cause:Event OR cause:Entity)
          AND (effect:Event OR effect:Entity)
        RETURN 
            id(cause) AS cause_id,
            id(effect) AS effect_id,
            r.confidence AS confidence,
            r.time_diff AS time_diff,
            r.cis_score AS cis_score
        """
        
        result = session.run(query, video_id=video_id)
        
        causal_links = []
        for record in result:
            link = {
                "cause": str(record["cause_id"]),
                "effect": str(record["effect_id"]),
                "confidence": record.get("confidence", 1.0),
                "time_diff": record.get("time_diff", 0),
                "cis_score": record.get("cis_score", 0.0),
            }
            causal_links.append(link)
        
        logger.info(f"Exported {len(causal_links)} causal links for video {video_id}")
        return causal_links
    
    def get_video_stats(self, video_id: Optional[str] = None) -> Dict[str, int]:
        """
        Get graph statistics for a video or entire database
        
        Returns:
            Dict with counts of entities, relationships, events, causal links
        """
        with self.driver.session() as session:
            query = """
            MATCH (e:Entity)
            WHERE $video_id IS NULL OR e.video_id = $video_id
            WITH count(e) AS entity_count
            
            MATCH (e1:Entity)-[r]->(e2:Entity)
            WHERE $video_id IS NULL OR e1.video_id = $video_id
            WITH entity_count, count(r) AS rel_count
            
            MATCH (ev:Event)
            WHERE $video_id IS NULL OR ev.video_id = $video_id
            WITH entity_count, rel_count, count(ev) AS event_count
            
            MATCH (cause)-[c:POTENTIALLY_CAUSED]->(effect)
            WHERE $video_id IS NULL OR cause.video_id = $video_id
            RETURN 
                entity_count,
                rel_count,
                event_count,
                count(c) AS causal_count
            """
            
            result = session.run(query, video_id=video_id)
            record = result.single()
            
            if record:
                return {
                    "entities": record["entity_count"],
                    "relationships": record["rel_count"],
                    "events": record["event_count"],
                    "causal_links": record["causal_count"],
                }
            else:
                return {
                    "entities": 0,
                    "relationships": 0,
                    "events": 0,
                    "causal_links": 0,
                }


if __name__ == "__main__":
    # Example usage
    adapter = OrionKGAdapter(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password"
    )
    
    try:
        # Get stats
        stats = adapter.get_video_stats()
        print(f"Graph stats: {stats}")
        
        # Export a video's graph
        pred_graph = adapter.export_prediction_graph(video_id="test_video_001")
        print(f"Exported {len(pred_graph.entities)} entities")
        print(f"Exported {len(pred_graph.relationships)} relationships")
        print(f"Exported {len(pred_graph.events)} events")
        print(f"Exported {len(pred_graph.causal_links)} causal links")
    
    finally:
        adapter.close()
