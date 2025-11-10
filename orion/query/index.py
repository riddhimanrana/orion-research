"""
Video Index - Lightweight storage for processed video metadata
================================================================

Stores entity detections, zones, and SLAM data for fast querying.

Author: Orion Research Team  
Date: November 2025
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class EntityObservation:
    """Single observation of an entity"""
    entity_id: int
    frame_idx: int
    timestamp: float
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    zone_id: Optional[int]
    pose: Optional[List[float]]  # [x, y, z] camera position
    clip_embedding: Optional[List[float]] = None
    caption: Optional[str] = None  # Lazy-loaded from FastVLM


@dataclass  
class SpatialZone:
    """Spatial zone metadata"""
    zone_id: int
    zone_type: str
    frame_indices: List[int]
    entity_ids: List[int]
    centroid: Optional[List[float]]  # Average camera position


class VideoIndex:
    """
    Lightweight SQLite index for processed video data.
    Enables fast spatial and temporal queries.
    """
    
    def __init__(self, index_path: Path, video_path: Path):
        self.index_path = Path(index_path)
        self.video_path = Path(video_path)
        self.conn: Optional[sqlite3.Connection] = None
        
    def create_schema(self):
        """Create database schema"""
        self.conn = sqlite3.connect(self.index_path)
        cursor = self.conn.cursor()
        
        # Metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        
        # Entity observations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id INTEGER,
                frame_idx INTEGER,
                timestamp REAL,
                class_name TEXT,
                confidence REAL,
                bbox_x1 REAL,
                bbox_y1 REAL,
                bbox_x2 REAL,
                bbox_y2 REAL,
                zone_id INTEGER,
                pose_x REAL,
                pose_y REAL,
                pose_z REAL,
                clip_embedding BLOB,
                caption TEXT
            )
        """)
        
        # Spatial zones
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS zones (
                zone_id INTEGER PRIMARY KEY,
                zone_type TEXT,
                frame_indices TEXT,
                entity_ids TEXT,
                centroid_x REAL,
                centroid_y REAL,
                centroid_z REAL
            )
        """)
        
        # Create indices for fast queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entity_id ON observations(entity_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_class_name ON observations(class_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_zone_id ON observations(zone_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_frame_idx ON observations(frame_idx)")
        
        self.conn.commit()
        
    def add_observation(self, obs: EntityObservation):
        """Add entity observation"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO observations 
            (entity_id, frame_idx, timestamp, class_name, confidence,
             bbox_x1, bbox_y1, bbox_x2, bbox_y2, zone_id,
             pose_x, pose_y, pose_z, clip_embedding, caption)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            obs.entity_id, obs.frame_idx, obs.timestamp, obs.class_name, obs.confidence,
            obs.bbox[0], obs.bbox[1], obs.bbox[2], obs.bbox[3], obs.zone_id,
            obs.pose[0] if obs.pose else None,
            obs.pose[1] if obs.pose else None,
            obs.pose[2] if obs.pose else None,
            np.array(obs.clip_embedding).tobytes() if obs.clip_embedding else None,
            obs.caption
        ))
        
    def add_zone(self, zone: SpatialZone):
        """Add spatial zone"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO zones 
            (zone_id, zone_type, frame_indices, entity_ids, centroid_x, centroid_y, centroid_z)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            zone.zone_id, zone.zone_type,
            json.dumps(zone.frame_indices), json.dumps(zone.entity_ids),
            zone.centroid[0] if zone.centroid else None,
            zone.centroid[1] if zone.centroid else None,
            zone.centroid[2] if zone.centroid else None
        ))
        
    def query_by_class(self, class_name: str) -> List[EntityObservation]:
        """Find all observations of a class"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT entity_id, frame_idx, timestamp, class_name, confidence,
                   bbox_x1, bbox_y1, bbox_x2, bbox_y2, zone_id,
                   pose_x, pose_y, pose_z, caption
            FROM observations
            WHERE class_name = ?
            ORDER BY frame_idx
        """, (class_name,))
        
        observations = []
        for row in cursor.fetchall():
            obs = EntityObservation(
                entity_id=row[0],
                frame_idx=row[1],
                timestamp=row[2],
                class_name=row[3],
                confidence=row[4],
                bbox=[row[5], row[6], row[7], row[8]],
                zone_id=row[9],
                pose=[row[10], row[11], row[12]] if row[10] is not None else None,
                caption=row[13]
            )
            observations.append(obs)
        
        return observations
    
    def query_by_zone(self, zone_id: int) -> List[EntityObservation]:
        """Find all observations in a zone"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT entity_id, frame_idx, timestamp, class_name, confidence,
                   bbox_x1, bbox_y1, bbox_x2, bbox_y2, zone_id,
                   pose_x, pose_y, pose_z, caption
            FROM observations
            WHERE zone_id = ?
            ORDER BY frame_idx
        """, (zone_id,))
        
        observations = []
        for row in cursor.fetchall():
            obs = EntityObservation(
                entity_id=row[0],
                frame_idx=row[1],
                timestamp=row[2],
                class_name=row[3],
                confidence=row[4],
                bbox=[row[5], row[6], row[7], row[8]],
                zone_id=row[9],
                pose=[row[10], row[11], row[12]] if row[10] is not None else None,
                caption=row[13]
            )
            observations.append(obs)
        
        return observations
    
    def update_caption(self, entity_id: int, frame_idx: int, caption: str):
        """Update caption for a specific observation"""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE observations
            SET caption = ?
            WHERE entity_id = ? AND frame_idx = ?
        """, (caption, entity_id, frame_idx))
        self.conn.commit()
        
    def get_frame_for_entity(self, entity_id: int) -> Tuple[int, List[float]]:
        """Get best frame to caption (largest bbox)"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT frame_idx, bbox_x1, bbox_y1, bbox_x2, bbox_y2
            FROM observations
            WHERE entity_id = ?
            ORDER BY (bbox_x2 - bbox_x1) * (bbox_y2 - bbox_y1) DESC
            LIMIT 1
        """, (entity_id,))
        
        row = cursor.fetchone()
        if row:
            return row[0], [row[1], row[2], row[3], row[4]]
        return None, None
    
    def commit(self):
        """Commit changes"""
        if self.conn:
            self.conn.commit()
            
    def close(self):
        """Close connection"""
        if self.conn:
            self.conn.close()
