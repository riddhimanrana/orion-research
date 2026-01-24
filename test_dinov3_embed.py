# Minimal DINOv3 embedding test script
# Save as test_dinov3_embed.py and run: python test_dinov3_embed.py

from orion.backends.dino_backend import DINOEmbedder
from PIL import Image
import numpy as np

# Path to a sample image (replace with a real image path if needed)
SAMPLE_IMAGE = "tests/assets/sample.jpg"

try:
    # Load a sample image (create a dummy one if not found)
    try:
        img = Image.open(SAMPLE_IMAGE).convert("RGB")
    except Exception:
        img = Image.fromarray(np.uint8(np.random.rand(224,224,3)*255))
        print("[Info] Using random image as input.")

    # Initialize DINOv3 embedder (adjust device if needed)
    embedder = DINOEmbedder(local_weights_dir="models/dinov3-vitb16", device="cpu")
    print(f"[Info] DINOv3 embedder loaded on {embedder.device}")

    # Extract embedding
    emb = embedder.encode_image(img)
    print(f"[Success] DINOv3 embedding shape: {emb.shape}")
except Exception as e:
    print(f"[Error] DINOv3 embedding failed: {e}")
