""""""

Unit tests for perception/semantic types and configurationUnit tests for perception/semantic types and configuration

==================================================================================================================



Tests all data structures and config validation for Phase 1 and Phase 2.Tests all data structures and config validation for Phase 1 and Phase 2.



Run: pytest tests/test_types_and_config.py -vRun: pytest tests/test_types_and_config.py -v

""""""



import pytestimport pytest

import numpy as npimport numpy as np

from datetime import datetime

# Perception types and config

from orion.perception.types import ObjectClass, BoundingBox, Observation, PerceptionEntity# Perception types and config

from orion.perception.config import (from orion.perception.types import (

    DetectionConfig, EmbeddingConfig, DescriptionConfig, PerceptionConfig,    ObjectClass, SpatialZone, BoundingBox, Observation, PerceptionEntity,

    get_fast_config, get_balanced_config, get_accurate_config,    PerceptionResult

))

from orion.perception.config import (

# Semantic types and config    DetectionConfig, EmbeddingConfig, DescriptionConfig, PerceptionConfig,

from orion.semantic.types import (    get_fast_config as get_fast_perception_config,

    ChangeType, StateChange, TemporalWindow, Event    get_balanced_config as get_balanced_perception_config,

)    get_accurate_config as get_accurate_perception_config,

from orion.semantic.config import ()

    StateChangeConfig, CausalConfig, SemanticConfig,

    get_fast_semantic_config,# Semantic types and config

    get_balanced_semantic_config,from orion.semantic.types import (

    get_accurate_semantic_config,    ChangeType, StateChange, TemporalWindow, CausalLink, SceneSegment,

)    LocationProfile, Event, SemanticEntity, SemanticResult

)

# Main configfrom orion.semantic.config import (

from orion.config import OrionConfig    StateChangeConfig, TemporalWindowConfig, EventCompositionConfig,

    CausalConfig, SemanticConfig,

    get_fast_semantic_config,

# ============================================================================    get_balanced_semantic_config,

# PERCEPTION TYPES TESTS    get_accurate_semantic_config,

# ============================================================================)



class TestObjectClass:# Main config

    """Test ObjectClass enum"""from orion.config import OrionConfig

    

    def test_common_classes_present(self):

        """Verify common COCO classes exist"""# ============================================================================

        assert ObjectClass.PERSON.value == "person"# PERCEPTION TYPES TESTS

        assert ObjectClass.CAR.value == "car"# ============================================================================

        assert ObjectClass.DOG.value == "dog"

class TestObjectClass:

    """Test ObjectClass enum"""

class TestBoundingBox:    

    """Test BoundingBox dataclass"""    def test_all_coco_classes_present(self):

            """Verify all 80 COCO classes are defined"""

    def test_bbox_properties(self):        # Should have at least common classes

        """Test bbox geometry"""        assert ObjectClass.PERSON

        bbox = BoundingBox(x1=10, y1=20, x2=110, y2=70)        assert ObjectClass.CAR

        assert bbox.width == 100        assert ObjectClass.DOG

        assert bbox.height == 50        assert ObjectClass.CAT

        assert bbox.area == 5000        assert ObjectClass.CUP

        assert bbox.center == (60, 45)        assert ObjectClass.BOTTLE

    

    def test_enum_values_unique(self):

class TestObservation:        """Verify all enum values are unique"""

    """Test Observation dataclass"""        values = [e.value for e in ObjectClass]

            assert len(values) == len(set(values)), "Duplicate enum values found"

    def test_observation_creation(self):

        """Create a valid observation"""

        bbox = BoundingBox(x1=10, y1=20, x2=60, y2=60)class TestBoundingBox:

        embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)    """Test BoundingBox dataclass"""

            

        obs = Observation(    def test_bbox_creation(self):

            bounding_box=bbox,        """Create a simple bounding box"""

            centroid=(35, 40),        bbox = BoundingBox(x1=10, y1=20, x2=110, y2=70)

            object_class=ObjectClass.PERSON,        assert bbox.x1 == 10

            confidence=0.95,        assert bbox.y1 == 20

            visual_embedding=embedding,        assert bbox.x2 == 110

            frame_number=0,        assert bbox.y2 == 70

            timestamp=0.0,    

            temp_id="temp_0",    def test_bbox_area(self):

        )        """Test area calculation"""

                bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)

        assert obs.frame_number == 0        assert bbox.area == 5000

        assert obs.confidence == 0.95    

    def test_bbox_center(self):

        """Test center point calculation"""

class TestPerceptionEntity:        bbox = BoundingBox(x1=10, y1=20, x2=110, y2=70)

    """Test PerceptionEntity dataclass"""        center = bbox.center

            assert center == (60, 45)

    def test_entity_from_observations(self):    

        """Create entity from observations"""    def test_bbox_width_height(self):

        bbox = BoundingBox(x1=10, y1=20, x2=60, y2=60)        """Test width/height properties"""

        emb = np.zeros(512)        bbox = BoundingBox(x1=10, y1=20, x2=110, y2=70)

        emb[0] = 1.0        assert bbox.width == 100

                assert bbox.height == 50

        obs = Observation(    

            bounding_box=bbox,    def test_bbox_to_list(self):

            centroid=(35, 40),        """Test conversion to list"""

            object_class=ObjectClass.PERSON,        bbox = BoundingBox(x1=10, y1=20, x2=110, y2=70)

            confidence=0.9,        lst = bbox.to_list()

            visual_embedding=emb,        assert lst == [10, 20, 110, 70]

            frame_number=0,

            timestamp=0.0,

            temp_id="temp_0",class TestObservation:

        )    """Test Observation dataclass"""

            

        entity = PerceptionEntity(    def test_observation_creation(self):

            entity_id="e1",        """Create a valid observation"""

            object_class=ObjectClass.PERSON,        bbox = BoundingBox(x1=10, y1=20, x2=60, y2=60)

            observations=[obs],        embedding = np.random.randn(512)

        )        embedding = embedding / np.linalg.norm(embedding)  # Normalize

                

        assert entity.appearance_count == 1        obs = Observation(

            bounding_box=bbox,

            centroid=(35, 40),

# ============================================================================            object_class=ObjectClass.PERSON,

# SEMANTIC TYPES TESTS            confidence=0.95,

# ============================================================================            visual_embedding=embedding,

            frame_number=0,

class TestStateChange:            timestamp=0.0,

    """Test StateChange dataclass"""            temp_id="temp_0",

            )

    def test_state_change_creation(self):        

        """Create a state change"""        assert obs.frame_number == 0

        change = StateChange(        assert obs.confidence == 0.95

            entity_id="e1",        assert len(obs.visual_embedding) == 512

            timestamp_before=1.0,    

            timestamp_after=1.5,    def test_observation_embedding_normalization(self):

            frame_before=4,        """Verify embedding is L2 normalized"""

            frame_after=6,        bbox = BoundingBox(x1=0, y1=0, x2=50, y2=50)

            description_before="standing",        embedding = np.array([3.0, 4.0, 0.0], dtype=np.float32)  # Not normalized (norm=5)

            description_after="sitting",        

            similarity_score=0.7,        obs = Observation(

        )            bounding_box=bbox,

                    centroid=(25, 25),

        assert change.entity_id == "e1"            object_class=ObjectClass.PERSON,

        assert abs(change.change_magnitude - 0.3) < 1e-6            confidence=0.9,

            visual_embedding=embedding,

            frame_number=0,

class TestTemporalWindow:            timestamp=0.0,

    """Test TemporalWindow dataclass"""            temp_id="temp_0",

            )

    def test_window_duration(self):        

        """Test duration calculation"""        # Should be normalized to unit length

        window = TemporalWindow(start_time=0.0, end_time=5.0)        norm = np.linalg.norm(obs.visual_embedding)

        assert window.duration == 5.0        assert abs(norm - 1.0) < 1e-5

    

    def test_observation_invalid_confidence(self):

class TestEvent:        """Confidence must be in [0, 1]"""

    """Test Event dataclass"""        bbox = BoundingBox(x1=0, y1=0, x2=50, y2=50)

            with pytest.raises(ValueError):

    def test_event_creation(self):            Observation(

        """Create an event"""                bounding_box=bbox,

        event = Event(                centroid=(25, 25),

            event_id="ev1",                object_class=ObjectClass.PERSON,

            description="person entered",                confidence=1.5,  # Invalid

            event_type="motion",                visual_embedding=np.zeros(512),

            start_timestamp=1.0,                frame_number=0,

            end_timestamp=3.5,                timestamp=0.0,

            start_frame=4,                temp_id="temp_0",

            end_frame=14,            )

            involved_entities=["e1"],

            confidence=0.92,

        )class TestPerceptionEntity:

            """Test PerceptionEntity dataclass"""

        assert event.duration == 2.5    

    def test_entity_creation_from_observations(self):

        """Create entity from multiple observations"""

# ============================================================================        observations = []

# CONFIG TESTS        

# ============================================================================        for i in range(3):

            bbox = BoundingBox(x1=10+i, y1=20, x2=60+i, y2=60)

class TestPerceptionConfig:            embedding = np.random.randn(512)

    """Test perception config"""            embedding = embedding / np.linalg.norm(embedding)

                

    def test_defaults(self):            obs = Observation(

        """Default values"""                bounding_box=bbox,

        cfg = DetectionConfig()                centroid=(35+i, 40),

        assert cfg.model == "yolo11x"                object_class=ObjectClass.PERSON,

                    confidence=0.9 + i * 0.01,

    def test_presets(self):                visual_embedding=embedding,

        """Presets work"""                frame_number=i,

        fast = get_fast_config()                timestamp=float(i) / 4.0,

        assert fast.detection.model == "yolo11n"                temp_id=f"temp_{i}",

                    )

        balanced = get_balanced_config()            observations.append(obs)

        assert balanced.detection.model == "yolo11m"        

        entity = PerceptionEntity(

            entity_id="entity_1",

class TestSemanticConfig:            object_class=ObjectClass.PERSON,

    """Test semantic config"""            observations=observations,

            )

    def test_defaults(self):        

        """Default values"""        assert entity.entity_id == "entity_1"

        cfg = StateChangeConfig()        assert len(entity.observations) == 3

        assert cfg.embedding_similarity_threshold == 0.85        assert entity.appearance_count == 3

        

    def test_presets(self):    def test_entity_average_embedding(self):

        """Presets work"""        """Test average embedding computation"""

        fast = get_fast_semantic_config()        observations = []

        assert fast.state_change.embedding_similarity_threshold == 0.80        

        for _ in range(2):

            bbox = BoundingBox(x1=10, y1=20, x2=60, y2=60)

class TestOrionConfig:            embedding = np.random.randn(512)

    """Test main config"""            embedding = embedding / np.linalg.norm(embedding)

                

    def test_composition(self):            obs = Observation(

        """Config composes correctly"""                bounding_box=bbox,

        cfg = OrionConfig()                centroid=(35, 40),

        assert cfg.perception is not None                object_class=ObjectClass.PERSON,

        assert cfg.semantic is not None                confidence=0.9,

                visual_embedding=embedding,

                frame_number=0,

if __name__ == "__main__":                timestamp=0.0,

    pytest.main([__file__, "-v"])                temp_id="temp_0",

            )
            observations.append(obs)
        
        entity = PerceptionEntity(
            entity_id="e1",
            object_class=ObjectClass.PERSON,
            observations=observations,
        )
        
        avg_emb = entity.compute_average_embedding()
        assert avg_emb is not None
        assert len(avg_emb) == 512
        # Should be normalized
        assert abs(np.linalg.norm(avg_emb) - 1.0) < 1e-5


# ============================================================================
# SEMANTIC TYPES TESTS
# ============================================================================

class TestStateChange:
    """Test StateChange dataclass"""
    
    def test_state_change_creation(self):
        """Create a state change"""
        change = StateChange(
            entity_id="e1",
            timestamp_before=1.0,
            timestamp_after=1.5,
            frame_before=4,
            frame_after=6,
            description_before="person standing",
            description_after="person sitting",
            similarity_score=0.7,
        )
        
        assert change.entity_id == "e1"
        assert abs(change.change_magnitude - 0.3) < 1e-6
    
    def test_state_change_invalid_similarity(self):
        """Similarity must be in [0, 1]"""
        with pytest.raises(ValueError):
            StateChange(
                entity_id="e1",
                timestamp_before=0.0,
                timestamp_after=1.0,
                frame_before=0,
                frame_after=4,
                description_before="before",
                description_after="after",
                similarity_score=1.5,  # Invalid
            )


class TestTemporalWindow:
    """Test TemporalWindow dataclass"""
    
    def test_window_creation(self):
        """Create a temporal window"""
        changes = []
        for i in range(2):
            change = StateChange(
                entity_id="e1",
                timestamp_before=float(i),
                timestamp_after=float(i) + 0.5,
                frame_before=i*4,
                frame_after=i*4+2,
                description_before=f"state {i}",
                description_after=f"state {i+1}",
                similarity_score=0.7,
            )
            changes.append(change)
        
        window = TemporalWindow(
            window_id="w1",
            start_time=0.0,
            end_time=2.0,
            state_changes=changes,
        )
        
        assert window.window_id == "w1"
        assert len(window.state_changes) == 2
    
    def test_window_duration(self):
        """Test duration property"""
        change = StateChange(
            entity_id="e1",
            timestamp_before=1.0,
            timestamp_after=1.5,
            frame_before=4,
            frame_after=6,
            description_before="before",
            description_after="after",
            similarity_score=0.7,
        )
        
        window = TemporalWindow(
            window_id="w1",
            start_time=0.0,
            end_time=5.0,
            state_changes=[change],
        )
        
        assert window.duration == 5.0


class TestEvent:
    """Test Event dataclass"""
    
    def test_event_creation(self):
        """Create an event"""
        event = Event(
            event_id="ev1",
            event_type="PERSON_SITS_DOWN",
            description="A person sat down on the chair",
            confidence=0.92,
            start_time=1.0,
            end_time=3.5,
            entities=["e1"],
        )
        
        assert event.event_id == "ev1"
        assert event.confidence == 0.92
    
    def test_event_duration(self):
        """Test event duration calculation"""
        event = Event(
            event_id="ev1",
            event_type="TEST",
            description="test event",
            confidence=0.9,
            start_time=0.0,
            end_time=5.0,
            entities=["e1"],
        )
        
        assert event.duration == 5.0
    
    def test_event_invalid_confidence(self):
        """Confidence must be in [0, 1]"""
        with pytest.raises(ValueError):
            Event(
                event_id="ev1",
                event_type="TEST",
                description="test",
                confidence=1.5,  # Invalid
                start_time=0.0,
                end_time=1.0,
                entities=["e1"],
            )


# ============================================================================
# PERCEPTION CONFIG TESTS
# ============================================================================

class TestDetectionConfig:
    """Test DetectionConfig"""
    
    def test_default_detection_config(self):
        """Test default values"""
        config = DetectionConfig()
        assert config.model == "yolo11x"
        assert config.confidence_threshold == 0.25
    
    def test_invalid_model(self):
        """Invalid model should raise error"""
        with pytest.raises(ValueError):
            DetectionConfig(model="yolo11xyz")
    
    def test_invalid_confidence_threshold(self):
        """Confidence must be in [0, 1]"""
        with pytest.raises(ValueError):
            DetectionConfig(confidence_threshold=1.5)


class TestEmbeddingConfig:
    """Test EmbeddingConfig"""
    
    def test_default_embedding_config(self):
        """Test default values"""
        config = EmbeddingConfig()
        assert config.embedding_dim == 512
        assert config.use_text_conditioning is True
    
    def test_invalid_embedding_dim(self):
        """Invalid dimension should raise error"""
        with pytest.raises(ValueError):
            EmbeddingConfig(embedding_dim=256)  # Not in {512, 768, 1024}


class TestPerceptionPresets:
    """Test perception config presets"""
    
    def test_fast_preset(self):
        """Fast preset should use small model"""
        config = get_fast_perception_config()
        assert config.detection.model == "yolo11n"
        assert config.target_fps == 2.0
    
    def test_balanced_preset(self):
        """Balanced preset should use medium model"""
        config = get_balanced_perception_config()
        assert config.detection.model == "yolo11m"
        assert config.target_fps == 4.0
    
    def test_accurate_preset(self):
        """Accurate preset should use largest model"""
        config = get_accurate_perception_config()
        assert config.detection.model == "yolo11x"
        assert config.target_fps == 8.0


# ============================================================================
# SEMANTIC CONFIG TESTS
# ============================================================================

class TestStateChangeConfig:
    """Test StateChangeConfig"""
    
    def test_default_state_change_config(self):
        """Test default values"""
        config = StateChangeConfig()
        assert config.embedding_similarity_threshold == 0.85
        assert config.embedding_model == "clip"
    
    def test_invalid_threshold(self):
        """Threshold must be in [0, 1]"""
        with pytest.raises(ValueError):
            StateChangeConfig(embedding_similarity_threshold=1.5)


class TestCausalConfig:
    """Test CausalConfig"""
    
    def test_weight_normalization(self):
        """Weights should auto-normalize if needed"""
        config = CausalConfig(
            weight_temporal=0.5,
            weight_spatial=0.5,
            weight_motion=0.5,
            weight_semantic=0.5,
        )
        
        # Should have been normalized
        total = (
            config.weight_temporal + config.weight_spatial +
            config.weight_motion + config.weight_semantic
        )
        assert abs(total - 1.0) < 0.01
    
    def test_invalid_cis_threshold(self):
        """CIS threshold must be in [0, 1]"""
        with pytest.raises(ValueError):
            CausalConfig(cis_threshold=1.5)


class TestSemanticPresets:
    """Test semantic config presets"""
    
    def test_fast_semantic_preset(self):
        """Fast preset should use loose thresholds"""
        config = get_fast_semantic_config()
        assert config.state_change.embedding_similarity_threshold == 0.80
        assert config.enable_graph_ingestion is False
    
    def test_balanced_semantic_preset(self):
        """Balanced preset should be default"""
        config = get_balanced_semantic_config()
        assert config.state_change.embedding_similarity_threshold == 0.85
        assert config.enable_graph_ingestion is True
    
    def test_accurate_semantic_preset(self):
        """Accurate preset should use tight thresholds"""
        config = get_accurate_semantic_config()
        assert config.state_change.embedding_similarity_threshold == 0.90
        assert config.verbose is True


# ============================================================================
# MAIN ORION CONFIG TESTS
# ============================================================================

class TestOrionConfig:
    """Test main OrionConfig composition"""
    
    def test_orion_config_creation(self):
        """Create main config"""
        config = OrionConfig()
        assert config.perception is not None
        assert config.semantic is not None
        assert config.neo4j is not None
    
    def test_perception_config_access(self):
        """Access perception config"""
        config = OrionConfig()
        assert config.perception.detection.model == "yolo11x"
        assert config.perception.detection.confidence_threshold == 0.25
    
    def test_semantic_config_access(self):
        """Access semantic config"""
        config = OrionConfig()
        assert config.semantic.causal.cis_threshold == 0.55
        assert config.semantic.enable_graph_ingestion is True
    
    def test_config_serialization(self):
        """Config should serialize to dict"""
        config = OrionConfig()
        # Configs have to_dict() methods in production
        # For now, just verify they're accessible
        assert hasattr(config.perception, 'detection')
        assert hasattr(config.semantic, 'causal')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

import pytest
import numpy as np
from datetime import datetime

# Perception types and config
from orion.perception.types import (
    ObjectClass, SpatialZone, BoundingBox, Observation, PerceptionEntity,
    PerceptionResult
)
from orion.perception.config import (
    DetectionConfig, EmbeddingConfig, DescriptionConfig, PerceptionConfig,
    get_fast_config as get_fast_perception_config,
    get_balanced_config as get_balanced_perception_config,
    get_accurate_config as get_accurate_perception_config,
)

# Semantic types and config
from orion.semantic.types import (
    ChangeType, StateChange, TemporalWindow, CausalLink, SceneSegment,
    LocationProfile, Event, SemanticEntity, SemanticResult
)
from orion.semantic.config import (
    StateChangeConfig, TemporalWindowConfig, EventCompositionConfig,
    CausalConfig, SemanticConfig,
    get_fast_semantic_config,
    get_balanced_semantic_config,
    get_accurate_semantic_config,
)

# Main config
from orion.config import OrionConfig


# ============================================================================
# PERCEPTION TYPES TESTS
# ============================================================================

class TestObjectClass:
    """Test ObjectClass enum"""
    
    def test_all_coco_classes_present(self):
        """Verify all 80 COCO classes are defined"""
        # Should have at least common classes
        assert ObjectClass.PERSON
        assert ObjectClass.CAR
        assert ObjectClass.DOG
        assert ObjectClass.CAT
        assert ObjectClass.CUP
        assert ObjectClass.BOTTLE
    
    def test_enum_values_unique(self):
        """Verify all enum values are unique"""
        values = [e.value for e in ObjectClass]
        assert len(values) == len(set(values)), "Duplicate enum values found"


class TestBoundingBox:
    """Test BoundingBox dataclass"""
    
    def test_bbox_creation(self):
        """Create a simple bounding box"""
        bbox = BoundingBox(x1=10, y1=20, x2=110, y2=70)
        assert bbox.x1 == 10
        assert bbox.y1 == 20
        assert bbox.x2 == 110
        assert bbox.y2 == 70
    
    def test_bbox_area(self):
        """Test area calculation"""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        assert bbox.area == 5000
    
    def test_bbox_center(self):
        """Test center point calculation"""
        bbox = BoundingBox(x1=10, y1=20, x2=110, y2=70)
        center = bbox.center
        assert center == (60, 45)
    
    def test_bbox_width_height(self):
        """Test width/height properties"""
        bbox = BoundingBox(x1=10, y1=20, x2=110, y2=70)
        assert bbox.width == 100
        assert bbox.height == 50


class TestObservation:
    """Test Observation dataclass"""
    
    def test_observation_creation(self):
        """Create a valid observation"""
        bbox = BoundingBox(x1=10, y1=20, x2=60, y2=60)
        embedding = np.random.randn(512)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize
        
        obs = Observation(
            bounding_box=bbox,
            centroid=(35, 40),
            object_class=ObjectClass.PERSON,
            confidence=0.95,
            visual_embedding=embedding,
            frame_number=0,
            timestamp=0.0,
            temp_id="temp_0",
        )
        
        assert obs.frame_number == 0
        assert obs.confidence == 0.95
        assert len(obs.visual_embedding) == 512
    
    def test_observation_embedding_normalization(self):
        """Verify embedding is L2 normalized"""
        bbox = BoundingBox(x1=0, y1=0, x2=50, y2=50)
        embedding = np.array([3.0, 4.0, 0.0], dtype=np.float32)  # Not normalized (norm=5)
        
        obs = Observation(
            bounding_box=bbox,
            centroid=(25, 25),
            object_class=ObjectClass.PERSON,
            confidence=0.9,
            visual_embedding=embedding,
            frame_number=0,
            timestamp=0.0,
            temp_id="temp_0",
        )
        
        # Should be normalized to unit length
        norm = np.linalg.norm(obs.visual_embedding)
        assert abs(norm - 1.0) < 1e-5
    
    def test_observation_invalid_confidence(self):
        """Confidence must be in [0, 1]"""
        bbox = BoundingBox(x1=0, y1=0, x2=50, y2=50)
        with pytest.raises(ValueError):
            Observation(
                bounding_box=bbox,
                centroid=(25, 25),
                object_class=ObjectClass.PERSON,
                confidence=1.5,  # Invalid
                visual_embedding=np.zeros(512),
                frame_number=0,
                timestamp=0.0,
                temp_id="temp_0",
            )


class TestPerceptionEntity:
    """Test PerceptionEntity dataclass"""
    
    def test_entity_creation_from_observations(self):
        """Create entity from multiple observations"""
        embeddings = [np.random.randn(512) for _ in range(3)]
        observations = []
        
        for i, emb in enumerate(embeddings):
            bbox = BoundingBox(x=10+i, y=20, width=50, height=40)
            obs = Observation(
                frame_idx=i,
                timestamp=float(i) / 4.0,
                bbox=bbox,
                class_label=ObjectClass.PERSON,
                confidence=0.9 + i * 0.01,
                embedding=emb,
            )
            observations.append(obs)
        
        entity = PerceptionEntity(
            entity_id="entity_1",
            class_label=ObjectClass.PERSON,
            observations=observations,
        )
        
        assert entity.entity_id == "entity_1"
        assert len(entity.observations) == 3
        assert entity.get_best_observation() is not None
    
    def test_entity_average_embedding(self):
        """Test average embedding computation"""
        emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        embeddings = [emb1, emb2]
        observations = []
        
        for emb in embeddings:
            bbox = BoundingBox(x=10, y=20, width=50, height=40)
            obs = Observation(
                frame_idx=0,
                timestamp=0.0,
                bbox=bbox,
                class_label=ObjectClass.PERSON,
                confidence=0.9,
                embedding=emb,
            )
            observations.append(obs)
        
        entity = PerceptionEntity(
            entity_id="e1",
            class_label=ObjectClass.PERSON,
            observations=observations,
        )
        
        avg_emb = entity.average_embedding
        assert avg_emb is not None
        assert len(avg_emb) == 3


# ============================================================================
# SEMANTIC TYPES TESTS
# ============================================================================

class TestStateChange:
    """Test StateChange dataclass"""
    
    def test_state_change_creation(self):
        """Create a state change"""
        change = StateChange(
            entity_id="e1",
            timestamp=1.5,
            before_description="person standing",
            after_description="person sitting",
            before_embedding=np.random.randn(512),
            after_embedding=np.random.randn(512),
            change_type=ChangeType.POSE,
        )
        
        assert change.entity_id == "e1"
        assert change.change_type == ChangeType.POSE
    
    def test_state_change_similarity(self):
        """Test similarity computation"""
        emb = np.random.randn(512)
        
        change = StateChange(
            entity_id="e1",
            timestamp=0.0,
            before_description="state A",
            after_description="state B",
            before_embedding=emb,
            after_embedding=emb,
            change_type=ChangeType.STATE,
        )
        
        # Same embedding should have high similarity
        assert change.similarity > 0.95


class TestTemporalWindow:
    """Test TemporalWindow dataclass"""
    
    def test_window_creation(self):
        """Create a temporal window"""
        changes = []
        for i in range(3):
            change = StateChange(
                entity_id="e1",
                timestamp=float(i),
                before_description=f"state {i}",
                after_description=f"state {i+1}",
                before_embedding=np.random.randn(512),
                after_embedding=np.random.randn(512),
            )
            changes.append(change)
        
        window = TemporalWindow(
            window_id="w1",
            start_time=0.0,
            end_time=2.0,
            state_changes=changes,
        )
        
        assert window.window_id == "w1"
        assert len(window.state_changes) == 3
    
    def test_window_duration(self):
        """Test duration property"""
        change = StateChange(
            entity_id="e1",
            timestamp=1.0,
            before_description="before",
            after_description="after",
            before_embedding=np.random.randn(512),
            after_embedding=np.random.randn(512),
        )
        
        window = TemporalWindow(
            window_id="w1",
            start_time=0.0,
            end_time=5.0,
            state_changes=[change],
        )
        
        assert window.duration == 5.0


class TestEvent:
    """Test Event dataclass"""
    
    def test_event_creation(self):
        """Create an event"""
        event = Event(
            event_id="ev1",
            event_type="PERSON_SITS_DOWN",
            description="A person sat down on the chair",
            confidence=0.92,
            start_time=1.0,
            end_time=3.5,
            entities=["e1"],
        )
        
        assert event.event_id == "ev1"
        assert event.confidence == 0.92
    
    def test_event_duration(self):
        """Test event duration calculation"""
        event = Event(
            event_id="ev1",
            event_type="TEST",
            description="test event",
            confidence=0.9,
            start_time=0.0,
            end_time=5.0,
            entities=["e1"],
        )
        
        assert event.duration == 5.0
    
    def test_event_invalid_confidence(self):
        """Confidence must be in [0, 1]"""
        with pytest.raises(ValueError):
            Event(
                event_id="ev1",
                event_type="TEST",
                description="test",
                confidence=1.5,  # Invalid
                start_time=0.0,
                end_time=1.0,
                entities=["e1"],
            )


# ============================================================================
# PERCEPTION CONFIG TESTS
# ============================================================================

class TestDetectionConfig:
    """Test DetectionConfig"""
    
    def test_default_detection_config(self):
        """Test default values"""
        config = DetectionConfig()
        assert config.model == "yolo11x"
        assert config.confidence_threshold == 0.25
    
    def test_invalid_model(self):
        """Invalid model should raise error"""
        with pytest.raises(ValueError):
            DetectionConfig(model="yolo11xyz")
    
    def test_invalid_confidence_threshold(self):
        """Confidence must be in [0, 1]"""
        with pytest.raises(ValueError):
            DetectionConfig(confidence_threshold=1.5)


class TestEmbeddingConfig:
    """Test EmbeddingConfig"""
    
    def test_default_embedding_config(self):
        """Test default values"""
        config = EmbeddingConfig()
        assert config.embedding_dim == 512
        assert config.use_text_conditioning is True
    
    def test_invalid_embedding_dim(self):
        """Invalid dimension should raise error"""
        with pytest.raises(ValueError):
            EmbeddingConfig(embedding_dim=256)  # Not in {512, 768, 1024}


class TestPerceptionPresets:
    """Test perception config presets"""
    
    def test_fast_preset(self):
        """Fast preset should use small model"""
        config = get_fast_perception_config()
        assert config.detection.model == "yolo11n"
        assert config.target_fps == 2.0
    
    def test_balanced_preset(self):
        """Balanced preset should use medium model"""
        config = get_balanced_perception_config()
        assert config.detection.model == "yolo11m"
        assert config.target_fps == 4.0
    
    def test_accurate_preset(self):
        """Accurate preset should use largest model"""
        config = get_accurate_perception_config()
        assert config.detection.model == "yolo11x"
        assert config.target_fps == 8.0


# ============================================================================
# SEMANTIC CONFIG TESTS
# ============================================================================

class TestStateChangeConfig:
    """Test StateChangeConfig"""
    
    def test_default_state_change_config(self):
        """Test default values"""
        config = StateChangeConfig()
        assert config.embedding_similarity_threshold == 0.85
        assert config.embedding_model == "clip"
    
    def test_invalid_threshold(self):
        """Threshold must be in [0, 1]"""
        with pytest.raises(ValueError):
            StateChangeConfig(embedding_similarity_threshold=1.5)


class TestCausalConfig:
    """Test CausalConfig"""
    
    def test_weight_normalization(self):
        """Weights should auto-normalize if needed"""
        config = CausalConfig(
            weight_temporal=0.5,
            weight_spatial=0.5,
            weight_motion=0.5,
            weight_semantic=0.5,
        )
        
        # Should have been normalized
        total = (
            config.weight_temporal + config.weight_spatial +
            config.weight_motion + config.weight_semantic
        )
        assert abs(total - 1.0) < 0.01
    
    def test_invalid_cis_threshold(self):
        """CIS threshold must be in [0, 1]"""
        with pytest.raises(ValueError):
            CausalConfig(cis_threshold=1.5)


class TestSemanticPresets:
    """Test semantic config presets"""
    
    def test_fast_semantic_preset(self):
        """Fast preset should use loose thresholds"""
        config = get_fast_semantic_config()
        assert config.state_change.embedding_similarity_threshold == 0.80
        assert config.enable_graph_ingestion is False
    
    def test_balanced_semantic_preset(self):
        """Balanced preset should be default"""
        config = get_balanced_semantic_config()
        assert config.state_change.embedding_similarity_threshold == 0.85
        assert config.enable_graph_ingestion is True
    
    def test_accurate_semantic_preset(self):
        """Accurate preset should use tight thresholds"""
        config = get_accurate_semantic_config()
        assert config.state_change.embedding_similarity_threshold == 0.90
        assert config.semantic.verbose is True


# ============================================================================
# MAIN ORION CONFIG TESTS
# ============================================================================

class TestOrionConfig:
    """Test main OrionConfig composition"""
    
    def test_orion_config_creation(self):
        """Create main config"""
        config = OrionConfig()
        assert config.perception is not None
        assert config.semantic is not None
        assert config.neo4j is not None
    
    def test_perception_config_access(self):
        """Access perception config"""
        config = OrionConfig()
        assert config.perception.detection.model == "yolo11x"
        assert config.perception.detection.confidence_threshold == 0.25
    
    def test_semantic_config_access(self):
        """Access semantic config"""
        config = OrionConfig()
        assert config.semantic.causal.cis_threshold == 0.55
        assert config.semantic.enable_graph_ingestion is True
    
    def test_config_serialization(self):
        """Config should serialize to dict"""
        config = OrionConfig()
        # Configs have to_dict() methods in production
        # For now, just verify they're accessible
        assert hasattr(config.perception, 'detection')
        assert hasattr(config.semantic, 'causal')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
