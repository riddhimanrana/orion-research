"""
Configuration Guide for Part 1: Asynchronous Perception Engine
===============================================================

This module provides easy-to-use configuration presets for different use cases.
Import and apply these configurations before running the perception engine.

Usage:
    from perception_config import apply_config, FAST_CONFIG, ACCURATE_CONFIG
    from perception_engine import DescriptionMode
    
    apply_config(FAST_CONFIG)
    perception_log = run_perception_engine(video_path)
    
    # Or switch description modes:
    from perception_config import set_description_mode
    set_description_mode(DescriptionMode.OBJECT)  # SCENE, OBJECT, or HYBRID
"""

from perception_engine import Config, DescriptionMode
import logging


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

# Fast processing - prioritizes speed over accuracy
FAST_CONFIG = {
    'TARGET_FPS': 2.0,
    'SCENE_SIMILARITY_THRESHOLD': 0.95,
    'FRAME_RESIZE_DIM': 512,
    'YOLO_CONFIDENCE_THRESHOLD': 0.35,
    'MIN_OBJECT_SIZE': 48,
    'NUM_WORKERS': 1,
    'LOG_LEVEL': logging.WARNING,
    'DESCRIPTION': 'Fast processing mode - 2 FPS, higher thresholds'
}

# Balanced - good trade-off between speed and accuracy (default)
BALANCED_CONFIG = {
    'TARGET_FPS': 4.0,
    'SCENE_SIMILARITY_THRESHOLD': 0.98,
    'FRAME_RESIZE_DIM': 768,
    'YOLO_CONFIDENCE_THRESHOLD': 0.25,
    'MIN_OBJECT_SIZE': 32,
    'NUM_WORKERS': 2,
    'LOG_LEVEL': logging.INFO,
    'DESCRIPTION': 'Balanced mode - default settings'
}

# Accurate - prioritizes accuracy over speed
ACCURATE_CONFIG = {
    'TARGET_FPS': 8.0,
    'SCENE_SIMILARITY_THRESHOLD': 0.99,
    'FRAME_RESIZE_DIM': 1024,
    'YOLO_CONFIDENCE_THRESHOLD': 0.15,
    'MIN_OBJECT_SIZE': 16,
    'NUM_WORKERS': 4,
    'LOG_LEVEL': logging.DEBUG,
    'DESCRIPTION': 'Accurate mode - 8 FPS, lower thresholds, more workers'
}

# High-quality scene detection - for videos with subtle changes
HIGH_QUALITY_SCENE_CONFIG = {
    'TARGET_FPS': 6.0,
    'SCENE_SIMILARITY_THRESHOLD': 0.99,
    'FRAME_RESIZE_DIM': 1024,
    'YOLO_CONFIDENCE_THRESHOLD': 0.25,
    'MIN_OBJECT_SIZE': 32,
    'NUM_WORKERS': 3,
    'LOG_LEVEL': logging.INFO,
    'DESCRIPTION': 'High-quality scene detection - captures subtle changes'
}

# Low-resource - for limited hardware
LOW_RESOURCE_CONFIG = {
    'TARGET_FPS': 1.0,
    'SCENE_SIMILARITY_THRESHOLD': 0.90,
    'FRAME_RESIZE_DIM': 384,
    'YOLO_CONFIDENCE_THRESHOLD': 0.40,
    'MIN_OBJECT_SIZE': 64,
    'NUM_WORKERS': 1,
    'QUEUE_MAX_SIZE': 500,
    'LOG_LEVEL': logging.WARNING,
    'DESCRIPTION': 'Low-resource mode - minimal memory and compute'
}

# Debug mode - verbose logging, slower processing
DEBUG_CONFIG = {
    'TARGET_FPS': 2.0,
    'SCENE_SIMILARITY_THRESHOLD': 0.98,
    'FRAME_RESIZE_DIM': 768,
    'YOLO_CONFIDENCE_THRESHOLD': 0.25,
    'MIN_OBJECT_SIZE': 32,
    'NUM_WORKERS': 1,
    'LOG_LEVEL': logging.DEBUG,
    'PROGRESS_BAR': True,
    'DESCRIPTION': 'Debug mode - verbose logging, single worker for easier debugging'
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
    description = config_dict.get('DESCRIPTION', 'Custom configuration')
    print(f"Applying configuration: {description}")
    
    for key, value in config_dict.items():
        if key != 'DESCRIPTION' and hasattr(Config, key):
            setattr(Config, key, value)
            print(f"  {key} = {value}")
    
    print("Configuration applied successfully!\n")


def print_current_config():
    """Print all current configuration parameters"""
    print("\n" + "="*80)
    print("CURRENT CONFIGURATION")
    print("="*80)
    
    print("\n[Video Processing]")
    print(f"  TARGET_FPS: {Config.TARGET_FPS}")
    print(f"  SCENE_SIMILARITY_THRESHOLD: {Config.SCENE_SIMILARITY_THRESHOLD}")
    print(f"  FRAME_RESIZE_DIM: {Config.FRAME_RESIZE_DIM}")
    
    print("\n[Object Detection]")
    print(f"  YOLO_CONFIDENCE_THRESHOLD: {Config.YOLO_CONFIDENCE_THRESHOLD}")
    print(f"  YOLO_IOU_THRESHOLD: {Config.YOLO_IOU_THRESHOLD}")
    print(f"  YOLO_MAX_DETECTIONS: {Config.YOLO_MAX_DETECTIONS}")
    print(f"  MIN_OBJECT_SIZE: {Config.MIN_OBJECT_SIZE}")
    print(f"  BBOX_PADDING_PERCENT: {Config.BBOX_PADDING_PERCENT}")
    
    print("\n[Visual Embedding]")
    print(f"  OSNET_INPUT_SIZE: {Config.OSNET_INPUT_SIZE}")
    print(f"  EMBEDDING_DIM: {Config.EMBEDDING_DIM}")
    
    print("\n[Multiprocessing]")
    print(f"  NUM_WORKERS: {Config.NUM_WORKERS}")
    print(f"  QUEUE_MAX_SIZE: {Config.QUEUE_MAX_SIZE}")
    print(f"  QUEUE_TIMEOUT: {Config.QUEUE_TIMEOUT}")
    print(f"  WORKER_SHUTDOWN_TIMEOUT: {Config.WORKER_SHUTDOWN_TIMEOUT}")
    
    print("\n[Description Generation]")
    print(f"  DESCRIPTION_MAX_TOKENS: {Config.DESCRIPTION_MAX_TOKENS}")
    print(f"  DESCRIPTION_TEMPERATURE: {Config.DESCRIPTION_TEMPERATURE}")
    
    print("\n[Performance & Logging]")
    print(f"  LOG_LEVEL: {Config.LOG_LEVEL}")
    print(f"  PROGRESS_BAR: {Config.PROGRESS_BAR}")
    print(f"  CHECKPOINT_INTERVAL: {Config.CHECKPOINT_INTERVAL}")
    
    print("\n" + "="*80 + "\n")


def create_custom_config(**kwargs) -> dict:
    """
    Create a custom configuration dictionary
    
    Args:
        **kwargs: Configuration parameters to customize
    
    Returns:
        Dictionary of configuration parameters
    
    Example:
        custom = create_custom_config(
            TARGET_FPS=6.0,
            NUM_WORKERS=3,
            DESCRIPTION='My custom config'
        )
        apply_config(custom)
    """
    # Start with balanced config as base
    custom_config = BALANCED_CONFIG.copy()
    
    # Override with provided parameters
    custom_config.update(kwargs)
    
    return custom_config


def recommend_config(video_duration: float, available_memory_gb: float = 8.0):
    """
    Recommend a configuration based on video characteristics and available resources
    
    Args:
        video_duration: Video duration in seconds
        available_memory_gb: Available system memory in GB
    
    Returns:
        Recommended configuration dictionary
    """
    print(f"\nAnalyzing requirements for {video_duration}s video with {available_memory_gb}GB RAM...")
    
    if video_duration < 30:
        if available_memory_gb >= 8:
            print("Recommendation: ACCURATE_CONFIG (short video, plenty of memory)")
            return ACCURATE_CONFIG
        else:
            print("Recommendation: BALANCED_CONFIG (short video, limited memory)")
            return BALANCED_CONFIG
    
    elif video_duration < 120:
        if available_memory_gb >= 16:
            print("Recommendation: BALANCED_CONFIG (medium video, plenty of memory)")
            return BALANCED_CONFIG
        elif available_memory_gb >= 8:
            print("Recommendation: FAST_CONFIG (medium video, moderate memory)")
            return FAST_CONFIG
        else:
            print("Recommendation: LOW_RESOURCE_CONFIG (medium video, limited memory)")
            return LOW_RESOURCE_CONFIG
    
    else:  # Long video
        if available_memory_gb >= 16:
            print("Recommendation: FAST_CONFIG (long video, plenty of memory)")
            return FAST_CONFIG
        else:
            print("Recommendation: LOW_RESOURCE_CONFIG (long video, limited memory)")
            return LOW_RESOURCE_CONFIG


# ============================================================================
# DESCRIPTION MODE CONFIGURATION
# ============================================================================

def set_description_mode(mode: DescriptionMode):
    """
    Set the FastVLM description generation mode
    
    Args:
        mode: DescriptionMode.SCENE, DescriptionMode.OBJECT, or DescriptionMode.HYBRID
    
    Modes:
        - SCENE: One description per frame, shared by all objects
          * Pros: Most efficient (1 FastVLM call per frame)
          * Cons: No object-specific details
          * Best for: Scene understanding, minimizing compute time
          
        - OBJECT: One description per object, focused on individual object
          * Pros: Detailed entity attributes, unique per object
          * Cons: More expensive (N FastVLM calls per frame)
          * Best for: Entity-centric knowledge graphs, object queries
          
        - HYBRID: Both scene + object descriptions  
          * Pros: Most comprehensive, both scene context + object details
          * Cons: Most expensive (1 + N FastVLM calls per frame)
          * Best for: Complete knowledge graphs, rich queries
    """
    Config.DESCRIPTION_MODE = mode
    print(f"Description mode set to: {mode.value.upper()}")
    print(f"  - FastVLM calls per frame: ", end="")
    if mode == DescriptionMode.SCENE:
        print("1 (shared by all objects)")
    elif mode == DescriptionMode.OBJECT:
        print("N (one per object)")
    else:  # HYBRID
        print("1 + N (scene + objects)")
    print()


def get_mode_config(mode: DescriptionMode) -> dict:
    """
    Get a configuration dict with the specified description mode
    
    Args:
        mode: DescriptionMode enum value
    
    Returns:
        Configuration dictionary
    """
    return {
        'DESCRIPTION_MODE': mode,
        'DESCRIPTION': f'Mode: {mode.value}'
    }


# Scene mode preset (most efficient)
SCENE_MODE_CONFIG = get_mode_config(DescriptionMode.SCENE)

# Object mode preset (balanced, recommended)
OBJECT_MODE_CONFIG = get_mode_config(DescriptionMode.OBJECT)

# Hybrid mode preset (most comprehensive)
HYBRID_MODE_CONFIG = get_mode_config(DescriptionMode.HYBRID)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Configuration Presets Demo")
    print("="*80)
    
    print("\nAvailable presets:")
    print("  1. FAST_CONFIG - Quick processing")
    print("  2. BALANCED_CONFIG - Default (recommended)")
    print("  3. ACCURATE_CONFIG - High accuracy")
    print("  4. HIGH_QUALITY_SCENE_CONFIG - Subtle scene changes")
    print("  5. LOW_RESOURCE_CONFIG - Limited hardware")
    print("  6. DEBUG_CONFIG - Development and debugging")
    
    print("\n" + "-"*80)
    print("Example 1: Apply FAST_CONFIG")
    print("-"*80)
    apply_config(FAST_CONFIG)
    
    print("\n" + "-"*80)
    print("Example 2: Create custom configuration")
    print("-"*80)
    custom = create_custom_config(
        TARGET_FPS=5.0,
        NUM_WORKERS=3,
        SCENE_SIMILARITY_THRESHOLD=0.97,
        DESCRIPTION='Custom config for my use case'
    )
    apply_config(custom)
    
    print("\n" + "-"*80)
    print("Example 3: Get recommendation")
    print("-"*80)
    recommended = recommend_config(video_duration=60, available_memory_gb=12)
    apply_config(recommended)
    
    print("\n" + "-"*80)
    print("Current Configuration:")
    print("-"*80)
    print_current_config()
