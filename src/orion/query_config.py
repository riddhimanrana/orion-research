"""
Part 3 Configuration Presets
=============================

Configuration presets for the Query & Evaluation Engine.
Provides different settings optimized for various use cases.

Author: Orion Research Team
Date: January 2025
"""

import os
from typing import Dict, Any, Optional
from query_evaluation import Config


# ============================================================================
# CONFIGURATION PRESETS
# ============================================================================

BASELINE_CONFIG = {
    "name": "Baseline",
    "description": "Minimal configuration for basic testing",
    # Gemini settings
    "GEMINI_MODEL": "gemini-2.0-flash-exp",
    "GEMINI_TEMPERATURE": 0.3,
    "GEMINI_MAX_TOKENS": 1024,
    # Video clip settings
    "CLIP_MAX_FRAMES": 5,
    "CLIP_FRAME_SAMPLE_RATE": 10,
    # Agent C settings
    "USE_VISION_CONTEXT": True,
    "USE_GRAPH_CONTEXT": True,
    "MAX_GRAPH_RESULTS": 10,
    "RERANK_RESULTS": False,
    # Evaluation
    "EC15_QUESTION_COUNT": 10,
    "LOTQ_QUESTION_COUNT": 3,
    "SIMILARITY_THRESHOLD": 0.75,
    "TEMPORAL_TOLERANCE_SECONDS": 2.0,
}

BALANCED_CONFIG = {
    "name": "Balanced",
    "description": "Balanced quality and performance (recommended)",
    # Gemini settings
    "GEMINI_MODEL": "gemini-2.0-flash-exp",
    "GEMINI_TEMPERATURE": 0.3,
    "GEMINI_MAX_TOKENS": 2048,
    # Video clip settings
    "CLIP_MAX_FRAMES": 10,
    "CLIP_FRAME_SAMPLE_RATE": 5,
    # Agent C settings
    "USE_VISION_CONTEXT": True,
    "USE_GRAPH_CONTEXT": True,
    "MAX_GRAPH_RESULTS": 20,
    "RERANK_RESULTS": True,
    # Evaluation
    "EC15_QUESTION_COUNT": 15,
    "LOTQ_QUESTION_COUNT": 5,
    "SIMILARITY_THRESHOLD": 0.75,
    "TEMPORAL_TOLERANCE_SECONDS": 2.0,
}

HIGH_QUALITY_CONFIG = {
    "name": "High Quality",
    "description": "Maximum quality for research evaluation",
    # Gemini settings
    "GEMINI_MODEL": "gemini-2.0-flash-exp",
    "GEMINI_TEMPERATURE": 0.2,  # More deterministic
    "GEMINI_MAX_TOKENS": 4096,
    # Video clip settings
    "CLIP_MAX_FRAMES": 20,
    "CLIP_FRAME_SAMPLE_RATE": 3,
    # Agent C settings
    "USE_VISION_CONTEXT": True,
    "USE_GRAPH_CONTEXT": True,
    "MAX_GRAPH_RESULTS": 50,
    "RERANK_RESULTS": True,
    # Evaluation
    "EC15_QUESTION_COUNT": 20,
    "LOTQ_QUESTION_COUNT": 10,
    "SIMILARITY_THRESHOLD": 0.80,
    "TEMPORAL_TOLERANCE_SECONDS": 1.5,
}

FAST_CONFIG = {
    "name": "Fast",
    "description": "Quick testing with minimal API calls",
    # Gemini settings
    "GEMINI_MODEL": "gemini-2.0-flash-exp",
    "GEMINI_TEMPERATURE": 0.5,
    "GEMINI_MAX_TOKENS": 512,
    # Video clip settings
    "CLIP_MAX_FRAMES": 3,
    "CLIP_FRAME_SAMPLE_RATE": 15,
    # Agent C settings
    "USE_VISION_CONTEXT": True,
    "USE_GRAPH_CONTEXT": True,
    "MAX_GRAPH_RESULTS": 5,
    "RERANK_RESULTS": False,
    # Evaluation
    "EC15_QUESTION_COUNT": 5,
    "LOTQ_QUESTION_COUNT": 2,
    "SIMILARITY_THRESHOLD": 0.70,
    "TEMPORAL_TOLERANCE_SECONDS": 3.0,
}

VISION_ONLY_CONFIG = {
    "name": "Vision Only",
    "description": "Agent A comparison - vision without graph",
    # Gemini settings
    "GEMINI_MODEL": "gemini-2.0-flash-exp",
    "GEMINI_TEMPERATURE": 0.3,
    "GEMINI_MAX_TOKENS": 2048,
    # Video clip settings
    "CLIP_MAX_FRAMES": 15,
    "CLIP_FRAME_SAMPLE_RATE": 3,
    # Agent C settings (disabled for Agent A comparison)
    "USE_VISION_CONTEXT": True,
    "USE_GRAPH_CONTEXT": False,
    "MAX_GRAPH_RESULTS": 0,
    "RERANK_RESULTS": False,
    # Evaluation
    "EC15_QUESTION_COUNT": 15,
    "LOTQ_QUESTION_COUNT": 5,
    "SIMILARITY_THRESHOLD": 0.75,
    "TEMPORAL_TOLERANCE_SECONDS": 2.0,
}

GRAPH_ONLY_CONFIG = {
    "name": "Graph Only",
    "description": "Agent B comparison - graph without vision",
    # Gemini settings (not used but required)
    "GEMINI_MODEL": "gemini-2.0-flash-exp",
    "GEMINI_TEMPERATURE": 0.3,
    "GEMINI_MAX_TOKENS": 2048,
    # Video clip settings (not used)
    "CLIP_MAX_FRAMES": 0,
    "CLIP_FRAME_SAMPLE_RATE": 1,
    # Agent C settings (disabled for Agent B comparison)
    "USE_VISION_CONTEXT": False,
    "USE_GRAPH_CONTEXT": True,
    "MAX_GRAPH_RESULTS": 30,
    "RERANK_RESULTS": False,
    # Evaluation
    "EC15_QUESTION_COUNT": 15,
    "LOTQ_QUESTION_COUNT": 5,
    "SIMILARITY_THRESHOLD": 0.75,
    "TEMPORAL_TOLERANCE_SECONDS": 2.0,
}


# Map of all presets
PRESETS = {
    "baseline": BASELINE_CONFIG,
    "balanced": BALANCED_CONFIG,
    "high_quality": HIGH_QUALITY_CONFIG,
    "fast": FAST_CONFIG,
    "vision_only": VISION_ONLY_CONFIG,
    "graph_only": GRAPH_ONLY_CONFIG,
}


# ============================================================================
# CONFIGURATION HELPERS
# ============================================================================


def apply_config(config: Dict[str, Any]) -> None:
    """
    Apply a configuration to the Config class.

    Args:
        config: Configuration dictionary
    """
    for key, value in config.items():
        if key in ["name", "description"]:
            continue
        if hasattr(Config, key):
            setattr(Config, key, value)

    print(f"✓ Applied configuration: {config.get('name', 'Custom')}")


def get_preset(preset_name: str) -> Dict[str, Any]:
    """
    Get a configuration preset by name.

    Args:
        preset_name: Name of preset (baseline, balanced, high_quality, fast, vision_only, graph_only)

    Returns:
        Configuration dictionary

    Raises:
        ValueError: If preset name is invalid
    """
    if preset_name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown preset: {preset_name}. Available: {available}")

    return PRESETS[preset_name]


def print_current_config() -> None:
    """Print current configuration settings."""
    print("Current Configuration:")
    print("=" * 60)
    print(f"  Gemini Model: {Config.GEMINI_MODEL}")
    print(f"  Gemini Temperature: {Config.GEMINI_TEMPERATURE}")
    print(f"  Gemini Max Tokens: {Config.GEMINI_MAX_TOKENS}")
    print()
    print(f"  Clip Max Frames: {Config.CLIP_MAX_FRAMES}")
    print(f"  Clip Sample Rate: 1/{Config.CLIP_FRAME_SAMPLE_RATE}")
    print()
    print(f"  Use Vision Context: {Config.USE_VISION_CONTEXT}")
    print(f"  Use Graph Context: {Config.USE_GRAPH_CONTEXT}")
    print(f"  Max Graph Results: {Config.MAX_GRAPH_RESULTS}")
    print(f"  Rerank Results: {Config.RERANK_RESULTS}")
    print()
    print(f"  EC-15 Questions: {Config.EC15_QUESTION_COUNT}")
    print(f"  LOT-Q Questions: {Config.LOTQ_QUESTION_COUNT}")
    print(f"  Similarity Threshold: {Config.SIMILARITY_THRESHOLD}")
    print(f"  Temporal Tolerance: {Config.TEMPORAL_TOLERANCE_SECONDS}s")
    print("=" * 60)


def create_custom_config(**kwargs) -> Dict[str, Any]:
    """
    Create a custom configuration by overriding balanced config.

    Args:
        **kwargs: Configuration parameters to override

    Returns:
        Custom configuration dictionary

    Example:
        config = create_custom_config(
            CLIP_MAX_FRAMES=20,
            GEMINI_TEMPERATURE=0.1
        )
        apply_config(config)
    """
    config = BALANCED_CONFIG.copy()
    config["name"] = "Custom"
    config["description"] = "User-defined configuration"
    config.update(kwargs)
    return config


def recommend_config(
    speed_priority: bool = False,
    quality_priority: bool = False,
    vision_only: bool = False,
    graph_only: bool = False,
) -> str:
    """
    Recommend a configuration based on priorities.

    Args:
        speed_priority: Prioritize speed over quality
        quality_priority: Prioritize quality over speed
        vision_only: Use only vision (Agent A comparison)
        graph_only: Use only graph (Agent B comparison)

    Returns:
        Recommended preset name
    """
    if vision_only:
        return "vision_only"
    elif graph_only:
        return "graph_only"
    elif quality_priority:
        return "high_quality"
    elif speed_priority:
        return "fast"
    else:
        return "balanced"


def update_gemini_api_key(api_key: str) -> None:
    """
    Update Gemini API key in configuration.

    Args:
        api_key: Google Gemini API key
    """
    Config.GEMINI_API_KEY = api_key
    os.environ["GEMINI_API_KEY"] = api_key
    print("✓ Gemini API key updated")


def update_neo4j_credentials(uri: str, user: str, password: str) -> None:
    """
    Update Neo4j connection credentials.

    Args:
        uri: Neo4j URI
        user: Username
        password: Password
    """
    Config.NEO4J_URI = uri
    Config.NEO4J_USER = user
    Config.NEO4J_PASSWORD = password
    print(f"✓ Neo4j credentials updated: {uri}")


# ============================================================================
# PRESET COMPARISON
# ============================================================================


def compare_presets() -> None:
    """Print a comparison of all configuration presets."""
    print("\nConfiguration Presets Comparison")
    print("=" * 100)
    print(
        f"{'Preset':<20} {'Frames':<10} {'Sample':<10} {'Graph':<10} {'EC-15':<10} {'LOT-Q':<10}"
    )
    print("-" * 100)

    for name, config in PRESETS.items():
        frames = config.get("CLIP_MAX_FRAMES", 0)
        sample = config.get("CLIP_FRAME_SAMPLE_RATE", 0)
        graph = config.get("MAX_GRAPH_RESULTS", 0)
        ec15 = config.get("EC15_QUESTION_COUNT", 0)
        lotq = config.get("LOTQ_QUESTION_COUNT", 0)

        print(
            f"{name:<20} {frames:<10} 1/{sample:<8} {graph:<10} {ec15:<10} {lotq:<10}"
        )

    print("=" * 100)
    print("\nRecommendations:")
    print("  • baseline: Basic testing, minimal resources")
    print("  • balanced: Production use (recommended)")
    print("  • high_quality: Research evaluation, best results")
    print("  • fast: Quick iteration during development")
    print("  • vision_only: Agent A baseline comparison")
    print("  • graph_only: Agent B baseline comparison")
    print()


# Example usage
if __name__ == "__main__":
    print("Part 3 Configuration Presets")
    print("=" * 80)
    print()

    # Show all presets
    compare_presets()

    # Example: Apply balanced config
    print("\nApplying BALANCED_CONFIG...")
    apply_config(BALANCED_CONFIG)
    print()

    # Show current config
    print_current_config()

    # Example: Create custom config
    print("\nCreating custom configuration...")
    custom = create_custom_config(
        CLIP_MAX_FRAMES=15, GEMINI_TEMPERATURE=0.1, EC15_QUESTION_COUNT=20
    )
    print(f"Custom config created: {custom['name']}")

    # Recommendation
    print("\nConfiguration recommendations:")
    print(f"  For speed: {recommend_config(speed_priority=True)}")
    print(f"  For quality: {recommend_config(quality_priority=True)}")
    print(f"  For vision-only: {recommend_config(vision_only=True)}")
    print(f"  For graph-only: {recommend_config(graph_only=True)}")
