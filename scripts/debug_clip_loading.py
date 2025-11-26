
import logging
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from orion.backends.clip_backend import CLIPEmbedder
    logger.info("Successfully imported CLIPEmbedder")
    
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    embedder = CLIPEmbedder(
        model_name="openai/clip-vit-base-patch32",
        device=device
    )
    logger.info("Successfully instantiated CLIPEmbedder")

except Exception as e:
    logger.error(f"Failed to load CLIP: {e}")
    import traceback
    traceback.print_exc()
