"""
Perception Engine Data Types & Contracts
========================================

Clean, typed data structures for Phase 1 (Perception).

These are the canonical types that flow through:
    Frame → Observation → PerceptionEntity → Description

Author: Orion Research Team
Date: October 2025
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np


class ObjectClass(str, Enum):
    """COCO object classes (80 classes from YOLO11)"""
    # People & animals
    PERSON = "person"
    BICYCLE = "bicycle"
    CAR = "car"
    MOTORCYCLE = "motorcycle"
    AIRPLANE = "airplane"
    BUS = "bus"
    TRAIN = "train"
    TRUCK = "truck"
    BOAT = "boat"
    
    # Traffic infrastructure
    TRAFFIC_LIGHT = "traffic light"
    FIRE_HYDRANT = "fire hydrant"
    STOP_SIGN = "stop sign"
    PARKING_METER = "parking meter"
    BENCH = "bench"
    
    # Animals
    BIRD = "bird"
    CAT = "cat"
    DOG = "dog"
    HORSE = "horse"
    SHEEP = "sheep"
    COW = "cow"
    ELEPHANT = "elephant"
    BEAR = "bear"
    ZEBRA = "zebra"
    GIRAFFE = "giraffe"
    
    # Accessories
    BACKPACK = "backpack"
    UMBRELLA = "umbrella"
    HANDBAG = "handbag"
    TIE = "tie"
    SUITCASE = "suitcase"
    
    # Sports
    FRISBEE = "frisbee"
    SKIS = "skis"
    SNOWBOARD = "snowboard"
    SPORTS_BALL = "sports ball"
    KITE = "kite"
    BASEBALL_BAT = "baseball bat"
    BASEBALL_GLOVE = "baseball glove"
    SKATEBOARD = "skateboard"
    SURFBOARD = "surfboard"
    TENNIS_RACKET = "tennis racket"
    
    # Tableware & food
    BOTTLE = "bottle"
    WINE_GLASS = "wine glass"
    CUP = "cup"
    FORK = "fork"
    KNIFE = "knife"
    SPOON = "spoon"
    BOWL = "bowl"
    BANANA = "banana"
    APPLE = "apple"
    SANDWICH = "sandwich"
    ORANGE = "orange"
    BROCCOLI = "broccoli"
    CARROT = "carrot"
    HOT_DOG = "hot dog"
    PIZZA = "pizza"
    DONUT = "donut"
    CAKE = "cake"
    
    # Furniture
    CHAIR = "chair"
    COUCH = "couch"
    POTTED_PLANT = "potted plant"
    BED = "bed"
    DINING_TABLE = "dining table"
    TOILET = "toilet"
    
    # Electronics
    TV = "tv"
    LAPTOP = "laptop"
    MOUSE = "mouse"
    REMOTE = "remote"
    KEYBOARD = "keyboard"
    CELL_PHONE = "cell phone"
    MICROWAVE = "microwave"
    OVEN = "oven"
    TOASTER = "toaster"
    SINK = "sink"
    REFRIGERATOR = "refrigerator"
    
    # Misc
    BOOK = "book"
    CLOCK = "clock"
    VASE = "vase"
    SCISSORS = "scissors"
    TEDDY_BEAR = "teddy bear"
    HAIR_DRIER = "hair drier"
    TOOTHBRUSH = "toothbrush"
    
    UNKNOWN = "unknown"


class SpatialZone(str, Enum):
    """Coarse spatial zones in frame"""
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    TOP = "top"
    MIDDLE = "middle"
    BOTTOM = "bottom"
    UNKNOWN = "unknown"


@dataclass
class SpatialContext:
    """Rich spatial context for an observation"""
    zone_type: str  # ceiling, wall_upper, wall_middle, wall_lower, floor
    confidence: float
    x_position: str  # left, center, right
    y_position: str  # top, middle, bottom
    reasoning: List[str] = field(default_factory=list)


@dataclass
class BoundingBox:
    """Bounding box representation [x1, y1, x2, y2]"""
    x1: float
    y1: float
    x2: float
    y2: float
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)
    
    def to_list(self) -> List[float]:
        """Convert to [x1, y1, x2, y2] list"""
        return [self.x1, self.y1, self.x2, self.y2]
    
    @staticmethod
    def from_list(bbox: List[float]) -> "BoundingBox":
        """Create from [x1, y1, x2, y2] list"""
        return BoundingBox(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))


@dataclass
class Observation:
    """
    Single detection of an object in a frame.
    
    This is the atomic unit that flows through the perception pipeline.
    Multiple observations of the same object are later clustered into
    a PerceptionEntity.
    """
    
    # Spatial information
    bounding_box: BoundingBox
    centroid: Tuple[float, float]  # (x, y) center
    
    # Detection information
    object_class: ObjectClass
    confidence: float  # 0-1, from YOLO
    
    # Visual information
    visual_embedding: np.ndarray  # CLIP embedding (normalized L2)
    
    # Temporal information
    frame_number: int
    timestamp: float  # seconds since start
    
    # Metadata
    temp_id: str  # Temporary ID before clustering
    
    # Optional: Cropped region for description
    image_patch: Optional[np.ndarray] = None
    spatial_zone: Optional["SpatialContext"] = None
    raw_yolo_class: Optional[str] = None  # For debugging
    frame_width: Optional[float] = None
    frame_height: Optional[float] = None
    
    # Placeholder for Phase 2
    entity_id: Optional[str] = None
    rich_description: Optional[str] = None
    scene_id: Optional[str] = None
    location_id: Optional[str] = None
    
    def __post_init__(self):
        """Validate observation"""
        if not isinstance(self.visual_embedding, np.ndarray):
            raise TypeError("visual_embedding must be np.ndarray")
        
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        
        # Validate embedding is normalized
        norm = float(np.linalg.norm(self.visual_embedding))
        if norm > 1.01 or norm < 0.99:
            # Auto-normalize if slightly off
            self.visual_embedding = self.visual_embedding / norm if norm > 0 else self.visual_embedding

        if self.frame_width is not None and self.frame_width <= 0:
            raise ValueError("frame_width must be positive when provided")

        if self.frame_height is not None and self.frame_height <= 0:
            raise ValueError("frame_height must be positive when provided")


@dataclass
class PerceptionEntity:
    """
    Clustered entity from Phase 1.
    
    Multiple observations of the same object are merged into a single entity
    with averaged visual embedding and selected for description.
    """
    
    entity_id: str
    object_class: ObjectClass
    observations: List[Observation] = field(default_factory=list)
    
    # Aggregated properties
    average_embedding: Optional[np.ndarray] = None
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    appearance_count: int = 0
    
    # Description
    description: Optional[str] = None
    description_frame: Optional[int] = None  # Frame used for description
    
    # Quality improvements (optional)
    corrected_class: Optional[str] = None  # Class correction result
    correction_confidence: Optional[float] = None  # Correction confidence score
    
    def __post_init__(self):
        """Validate entity"""
        if not self.observations:
            raise ValueError("PerceptionEntity must have at least one observation")
        
        self.appearance_count = len(self.observations)
        self.first_seen_frame = min(obs.frame_number for obs in self.observations)
        self.last_seen_frame = max(obs.frame_number for obs in self.observations)
    
    def compute_average_embedding(self) -> np.ndarray:
        """Compute average visual embedding across all observations"""
        if not self.observations:
            raise ValueError("No observations to average")
        
        embeddings = np.array([obs.visual_embedding for obs in self.observations])
        avg = np.mean(embeddings, axis=0)
        
        # Normalize
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg = avg / norm
        
        self.average_embedding = avg
        return avg
    
    def get_best_observation(self) -> Observation:
        """
        Select best observation for description.
        
        Criteria (in order):
        1. Highest confidence
        2. Largest bbox area
        3. Closest to frame center
        """
        if not self.observations:
            raise ValueError("No observations available")
        
        best = self.observations[0]
        
        for obs in self.observations[1:]:
            # Compare by confidence
            if obs.confidence > best.confidence:
                best = obs
            elif obs.confidence == best.confidence:
                # Tie-break by bbox area
                if obs.bounding_box.area > best.bounding_box.area:
                    best = obs
        
        return best
    
    def get_timeline(self) -> List[Observation]:
        """Get observations in chronological order"""
        return sorted(self.observations, key=lambda obs: obs.timestamp)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "entity_id": self.entity_id,
            "object_class": self.object_class.value,
            "appearance_count": self.appearance_count,
            "first_seen_frame": self.first_seen_frame,
            "last_seen_frame": self.last_seen_frame,
            "description": self.description,
            "description_frame": self.description_frame,
            "observations": [
                {
                    "frame": obs.frame_number,
                    "timestamp": obs.timestamp,
                    "bbox": obs.bounding_box.to_list(),
                    "confidence": obs.confidence,
                }
                for obs in self.observations
            ],
        }


@dataclass
class PerceptionResult:
    """Output from Phase 1 (Perception Engine)"""
    
    entities: List[PerceptionEntity]
    raw_observations: List[Observation]
    
    # Metadata
    video_path: str
    total_frames: int
    fps: float
    duration_seconds: float
    
    # Statistics
    total_detections: int = 0
    unique_entities: int = 0
    processing_time_seconds: float = 0.0
    
    def __post_init__(self):
        """Compute statistics"""
        self.total_detections = len(self.raw_observations)
        self.unique_entities = len(self.entities)
