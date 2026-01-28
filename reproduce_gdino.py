
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

def test_gdino_raw():
    print("=== Testing Raw GroundingDINO (Transformers) ===")
    
    # 1. Load Model
    model_id = "IDEA-Research/grounding-dino-tiny"
    device = "cpu" # Force CPU as per user environment
    print(f"Loading {model_id} on {device}...")
    
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 2. Load Frame
    video_path = "/Users/yogeshatluru/orion-research/datasets/PVSG/VidOR/mnt/lustre/jkyang/CVPR23/openpvsg/data/vidor/videos/0003_6141007489.mp4"
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Failed to read video from {video_path}")
        # Try creating a dummy image
        frame = np.zeros((384, 640, 3), dtype=np.uint8)
        frame.fill(128) # Grey image
        print("Using dummy grey frame.")
    else:
        print(f"Loaded frame from {video_path}: {frame.shape}")

    # 3. Prepare Input
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    prompt = "person . chair ."
    print(f"Prompt: '{prompt}'")
    
    inputs = processor(images=pil_image, text=prompt, return_tensors="pt").to(device)
    
    # 4. Inference
    print("Running inference...")
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 5. Post-process
    target_sizes = torch.tensor([pil_image.size[::-1]]).to(device)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.25,
        target_sizes=target_sizes
    )[0]
    
    print(f"Raw results keys: {results.keys()}")
    
    scores = results["scores"]
    labels = results.get("labels", [])
    text_labels = results.get("text_labels", []) # Depending on transformers version
    boxes = results["boxes"]
    
    print(f"Num detections: {len(scores)}")
    for i in range(len(scores)):
        lbl = text_labels[i] if len(text_labels) > i else labels[i]
        print(f"Det {i}: Score={scores[i]:.4f}, Label={lbl}, Box={boxes[i].tolist()}")

if __name__ == "__main__":
    test_gdino_raw()
