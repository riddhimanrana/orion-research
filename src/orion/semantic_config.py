"""
Configuration Guide for Part 2: Semantic Uplift Engine
=======================================================

This module provides preset configurations for different semantic uplift scenarios.

Usage:
    from orion.semantic_config import apply_config, FAST_CONFIG, ACCURATE_CONFIG
    
    apply_config(FAST_CONFIG)
    results = run_semantic_uplift(perception_log)
"""

# mypy: ignore-errors
# pyright: reportGeneralTypeIssues=false, reportOptionalMemberAccess=false, reportArgumentType=false

import logging
from typing import Optional

from .semantic_uplift import Config


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

# Fast processing - loose clustering, fewer events
FAST_CONFIG = {
    "MIN_CLUSTER_SIZE": 5,
    "CLUSTER_SELECTION_EPSILON": 0.20,
    "STATE_CHANGE_THRESHOLD": 0.80,
    "TIME_WINDOW_SIZE": 60.0,
    "MIN_EVENTS_PER_WINDOW": 3,
    "LOG_LEVEL": logging.WARNING,
    "DESCRIPTION": "Fast mode - loose clustering, larger windows",
}

# Balanced - default settings
BALANCED_CONFIG = {
    "MIN_CLUSTER_SIZE": 3,
    "CLUSTER_SELECTION_EPSILON": 0.15,
    "STATE_CHANGE_THRESHOLD": 0.85,
    "TIME_WINDOW_SIZE": 30.0,
    "MIN_EVENTS_PER_WINDOW": 2,
    "LOG_LEVEL": logging.INFO,
    "DESCRIPTION": "Balanced mode - default settings",
}

# Accurate - tight clustering, more events
ACCURATE_CONFIG = {
    "MIN_CLUSTER_SIZE": 2,
    "CLUSTER_SELECTION_EPSILON": 0.10,
    "STATE_CHANGE_THRESHOLD": 0.90,
    "TIME_WINDOW_SIZE": 15.0,
    "MIN_EVENTS_PER_WINDOW": 1,
    "LOG_LEVEL": logging.DEBUG,
    "DESCRIPTION": "Accurate mode - tight clustering, small windows, sensitive to changes",
}

# High precision entity tracking
HIGH_PRECISION_TRACKING = {
    "MIN_CLUSTER_SIZE": 4,
    "MIN_SAMPLES": 3,
    "CLUSTER_SELECTION_EPSILON": 0.08,
    "STATE_CHANGE_THRESHOLD": 0.85,
    "TIME_WINDOW_SIZE": 30.0,
    "LOG_LEVEL": logging.INFO,
    "DESCRIPTION": "High precision tracking - stricter clustering criteria",
}

# Sensitive state detection
SENSITIVE_STATE_DETECTION = {
    "MIN_CLUSTER_SIZE": 3,
    "STATE_CHANGE_THRESHOLD": 0.92,
    "TIME_WINDOW_SIZE": 20.0,
    "MIN_EVENTS_PER_WINDOW": 1,
    "LOG_LEVEL": logging.INFO,
    "DESCRIPTION": "Sensitive state detection - catches subtle changes",
}

# Debug mode
DEBUG_CONFIG = {
    "MIN_CLUSTER_SIZE": 2,
    "CLUSTER_SELECTION_EPSILON": 0.15,
    "STATE_CHANGE_THRESHOLD": 0.85,
    "TIME_WINDOW_SIZE": 30.0,
    "MIN_EVENTS_PER_WINDOW": 1,
    "LOG_LEVEL": logging.DEBUG,
    "PROGRESS_LOGGING": True,
    "DESCRIPTION": "Debug mode - verbose logging, lower thresholds",
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def apply_config(config_dict: dict):
    """
    Apply a configuration dictionary to the Config class

    Args:
        config_dict: Dictionary of configuration parameters
    """
    description = config_dict.get("DESCRIPTION", "Custom configuration")
    print(f"Applying configuration: {description}")

    for key, value in config_dict.items():
        if key != "DESCRIPTION" and hasattr(Config, key):
            setattr(Config, key, value)
            print(f"  {key} = {value}")

    print("Configuration applied successfully!\n")


def print_current_config():
    """Print all current configuration parameters"""
    print("\n" + "=" * 80)
    print("CURRENT CONFIGURATION - SEMANTIC UPLIFT")
    print("=" * 80)

    print("\n[Entity Tracking - HDBSCAN]")
    print(f"  MIN_CLUSTER_SIZE: {Config.MIN_CLUSTER_SIZE}")
    print(f"  MIN_SAMPLES: {Config.MIN_SAMPLES}")
    print(f"  CLUSTER_METRIC: {Config.CLUSTER_METRIC}")
    print(f"  CLUSTER_SELECTION_METHOD: {Config.CLUSTER_SELECTION_METHOD}")
    print(f"  CLUSTER_SELECTION_EPSILON: {Config.CLUSTER_SELECTION_EPSILON}")

    print("\n[State Change Detection]")
    print(f"  STATE_CHANGE_THRESHOLD: {Config.STATE_CHANGE_THRESHOLD}")
    print(f"  EMBEDDING_MODEL_TYPE: {Config.EMBEDDING_MODEL_TYPE}")

    print("\n[Temporal Windowing]")
    print(f"  TIME_WINDOW_SIZE: {Config.TIME_WINDOW_SIZE}s")
    print(f"  MIN_EVENTS_PER_WINDOW: {Config.MIN_EVENTS_PER_WINDOW}")

    print("\n[Scene & Location Inference]")
    print(f"  SCENE_SIMILARITY_THRESHOLD: {Config.SCENE_SIMILARITY_THRESHOLD}")
    print(f"  SCENE_SIMILARITY_TOP_K: {Config.SCENE_SIMILARITY_TOP_K}")
    print(f"  SCENE_LOCATION_TOP_OBJECTS: {Config.SCENE_LOCATION_TOP_OBJECTS}")

    print("\n[LLM Event Composition - Ollama]")
    print(f"  OLLAMA_API_URL: {Config.OLLAMA_API_URL}")
    print(f"  OLLAMA_MODEL: {Config.OLLAMA_MODEL}")
    print(f"  OLLAMA_TEMPERATURE: {Config.OLLAMA_TEMPERATURE}")
    print(f"  OLLAMA_MAX_TOKENS: {Config.OLLAMA_MAX_TOKENS}")

    print("\n[Neo4j Configuration]")
    print(f"  NEO4J_URI: {Config.NEO4J_URI}")
    print(f"  NEO4J_USER: {Config.NEO4J_USER}")
    print(f"  NEO4J_DATABASE: {Config.NEO4J_DATABASE}")

    print("\n[Performance]")
    print(f"  BATCH_SIZE: {Config.BATCH_SIZE}")
    print(f"  LOG_LEVEL: {Config.LOG_LEVEL}")
    print(f"  PROGRESS_LOGGING: {Config.PROGRESS_LOGGING}")

    print("\n" + "=" * 80 + "\n")


def create_custom_config(**kwargs) -> dict:
    """
    Create a custom configuration dictionary

    Args:
        **kwargs: Configuration parameters to customize

    Returns:
        Dictionary of configuration parameters

    Example:
        custom = create_custom_config(
            MIN_CLUSTER_SIZE=4,
            STATE_CHANGE_THRESHOLD=0.88,
            DESCRIPTION='My custom config'
        )
        apply_config(custom)
    """
    # Start with balanced config as base
    custom_config = BALANCED_CONFIG.copy()

    # Override with provided parameters
    custom_config.update(kwargs)

    return custom_config


def recommend_config(num_objects: int, video_duration: float):
    """
    Recommend a configuration based on data characteristics

    Args:
        num_objects: Number of perception objects
        video_duration: Total video duration in seconds

    Returns:
        Recommended configuration dictionary
    """
    print(f"\nAnalyzing data: {num_objects} objects over {video_duration:.1f}s...")

    objects_per_second = num_objects / video_duration if video_duration > 0 else 0

    if num_objects < 50:
        print("Recommendation: ACCURATE_CONFIG (small dataset, use tight clustering)")
        return ACCURATE_CONFIG

    elif objects_per_second > 10:
        print("Recommendation: FAST_CONFIG (high density, use loose clustering)")
        return FAST_CONFIG

    elif video_duration > 300:
        print("Recommendation: FAST_CONFIG with larger windows (long video)")
        config = FAST_CONFIG.copy()
        config["TIME_WINDOW_SIZE"] = 90.0
        config["DESCRIPTION"] = "Fast mode adapted for long video"
        return config

    else:
        print("Recommendation: BALANCED_CONFIG (standard dataset)")
        return BALANCED_CONFIG


def configure_for_tracking_quality(quality: str):
    """
    Configure based on desired tracking quality

    Args:
        quality: 'high', 'medium', or 'low'

    Returns:
        Configuration dictionary
    """
    if quality.lower() == "high":
        return HIGH_PRECISION_TRACKING
    elif quality.lower() == "low":
        return FAST_CONFIG
    else:
        return BALANCED_CONFIG


def configure_for_state_sensitivity(sensitivity: str):
    """
    Configure based on desired state change sensitivity

    Args:
        sensitivity: 'high', 'medium', or 'low'

    Returns:
        Configuration dictionary
    """
    if sensitivity.lower() == "high":
        return SENSITIVE_STATE_DETECTION
    elif sensitivity.lower() == "low":
        config = FAST_CONFIG.copy()
        config["STATE_CHANGE_THRESHOLD"] = 0.75
        config["DESCRIPTION"] = "Low sensitivity - only major state changes"
        return config
    else:
        return BALANCED_CONFIG


# ============================================================================
# NEO4J CONNECTION HELPERS
# ============================================================================


def update_neo4j_credentials(
    uri: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
):
    """
    Update Neo4j connection credentials

    Args:
        uri: Neo4j URI
        user: Neo4j username
        password: Neo4j password
    """
    if uri:
        Config.NEO4J_URI = uri  # type: ignore[attr-defined]
        print(f"Neo4j URI updated: {uri}")

    if user:
        Config.NEO4J_USER = user  # type: ignore[attr-defined]
        print(f"Neo4j user updated: {user}")

    if password:
        Config.NEO4J_PASSWORD = password  # type: ignore[attr-defined]
        print("Neo4j password updated")


def use_neo4j_docker():
    """Configure for Neo4j running in Docker"""
    update_neo4j_credentials(
        uri="neo4j://127.0.0.1:7687", user="neo4j", password="orion123"
    )
    print("\nConfigured for Neo4j Docker (default setup)")


def use_neo4j_desktop():
    """Configure for Neo4j Desktop"""
    update_neo4j_credentials(
        uri="neo4j://127.0.0.1:7687", user="neo4j", password="orion123"
    )
    print("\nConfigured for Neo4j Desktop (default setup)")


def use_neo4j_aura(uri: str, password: str):
    """
    Configure for Neo4j Aura Cloud

    Args:
        uri: Aura instance URI
        password: Aura password
    """
    update_neo4j_credentials(uri=uri, user="neo4j", password=password)
    print(f"\nConfigured for Neo4j Aura: {uri}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Configuration Presets Demo - Part 2")
    print("=" * 80)

    print("\nAvailable presets:")
    print("  1. FAST_CONFIG - Quick processing")
    print("  2. BALANCED_CONFIG - Default (recommended)")
    print("  3. ACCURATE_CONFIG - High accuracy")
    print("  4. HIGH_PRECISION_TRACKING - Stricter clustering")
    print("  5. SENSITIVE_STATE_DETECTION - Catches subtle changes")
    print("  6. DEBUG_CONFIG - Development and debugging")

    print("\n" + "-" * 80)
    print("Example 1: Apply ACCURATE_CONFIG")
    print("-" * 80)
    apply_config(ACCURATE_CONFIG)

    print("\n" + "-" * 80)
    print("Example 2: Create custom configuration")
    print("-" * 80)
    custom = create_custom_config(
        MIN_CLUSTER_SIZE=4,
        STATE_CHANGE_THRESHOLD=0.88,
        TIME_WINDOW_SIZE=25.0,
        DESCRIPTION="Custom config for specific use case",
    )
    apply_config(custom)

    print("\n" + "-" * 80)
    print("Example 3: Get recommendation")
    print("-" * 80)
    recommended = recommend_config(num_objects=150, video_duration=45.0)
    apply_config(recommended)

    print("\n" + "-" * 80)
    print("Example 4: Configure Neo4j")
    print("-" * 80)
    use_neo4j_docker()

    print("\n" + "-" * 80)
    print("Current Configuration:")
    print("-" * 80)
    print_current_config()
