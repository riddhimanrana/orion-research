from dataclasses import dataclass


@dataclass
class VideoQAConfig:
    """Feature toggles and limits for the unified video QA system."""

    enable_semantic: bool = True
    enable_overview: bool = True
    enable_spatial: bool = True
    enable_scene: bool = True
    enable_temporal: bool = True
    enable_causal: bool = True
    enable_events: bool = True

    vector_top_k_entities: int = 5
    vector_top_k_scenes: int = 3
    overview_top_entities: int = 5
    overview_recent_events: int = 8

    conversation_history_size: int = 10
