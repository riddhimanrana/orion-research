
import time
import torch
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_loading(model_name, device, vocab):
    print(f"\nTesting {model_name} on {device}...")
    
    t0 = time.time()
    model = YOLO(model_name)
    t1 = time.time()
    print(f"  Model load time: {t1 - t0:.2f}s")
    
    # Move to device if needed (YOLO usually handles this auto, but let's be explicit if we can, 
    # though Ultralytics API is high-level. We'll rely on auto for now but check internal device)
    # model.to(device) # Ultralytics YOLO doesn't always support .to() directly on the wrapper in the same way
    
    t2 = time.time()
    print(f"  Setting classes ({len(vocab)} classes)...")
    try:
        # Force device context if possible, or just run it
        model.set_classes(vocab)
        t3 = time.time()
        print(f"  set_classes time: {t3 - t2:.2f}s")
    except Exception as e:
        print(f"  ERROR in set_classes: {e}")
        return

    # Warmup inference
    print("  Running warmup inference...")
    try:
        import numpy as np
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        t4 = time.time()
        model.predict(dummy_frame, device=device, verbose=False)
        t5 = time.time()
        print(f"  Warmup inference time: {t5 - t4:.2f}s")
    except Exception as e:
        print(f"  ERROR in inference: {e}")

if __name__ == "__main__":
    vocab = [
        "monitor", "desk", "chair", "keyboard", "laptop", 
        "picture frame", "bottle", "mouse", "person", 
        "couch", "table", "tv", "phone", "cup", "book", "plant", ""
    ]
    
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Test Medium model
    test_loading("yolov8m-worldv2.pt", "cuda", vocab)
    
    # Test Large model
    test_loading("yolov8l-worldv2.pt", "cuda", vocab)
