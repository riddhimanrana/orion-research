from transformers import AutoModel, AutoImageProcessor
import torch
from PIL import Image

# Pick a valid device at runtime instead of hardcoding "cuda"
if torch.cuda.is_available():
    device = "cuda"
# Check for Apple Silicon MPS support (if available in this torch build)
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

model = AutoModel.from_pretrained(
    "kevin510/fast-vit-hd", trust_remote_code=True
)

# Try to move the model to the chosen device; if that fails, fall back to cpu
try:
    model = model.to(device)
except Exception as e:
    print(f"Warning: failed to move model to {device}: {e}. Falling back to cpu.")
    device = "cpu"
    model = model.to(device)

model = model.eval()

processor = AutoImageProcessor.from_pretrained(
    "kevin510/fast-vit-hd", trust_remote_code=True
)

img = Image.open("../data/examples/example1.jpg")
px = processor(img, do_center_crop=False, return_tensors="pt")["pixel_values"].to(device)   # (1,3,1024,1024)

emb = model(px)
print(emb.shape)   # (1, D, 3072)
