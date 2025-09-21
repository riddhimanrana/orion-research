from transformers import AutoModel, AutoImageProcessor
import torch, PIL.Image

device = "cuda"  # or "cpu" / "mps"

model = AutoModel.from_pretrained(
    "kevin510/fast-vit-hd", trust_remote_code=True
).to(device).eval()

processor = AutoImageProcessor.from_pretrained(
    "kevin510/fast-vit-hd", trust_remote_code=True
)

img = PIL.Image.open("your_image.jpg")
px  = processor(img, do_center_crop=False, return_tensors="pt")["pixel_values"].to(device)   # (1,3,1024,1024)

emb = model(px)
print(emb.shape)   # (1, D, 3072)
