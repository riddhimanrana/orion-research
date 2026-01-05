
import time
from ultralytics import YOLO
import os

def test_save_load():
    vocab = [
        "monitor", "desk", "chair", "keyboard", "laptop", 
        "picture frame", "bottle", "mouse", "person", 
        "couch", "table", "tv", "phone", "cup", "book", "plant", ""
    ]
    
    print("1. Loading original model...")
    model = YOLO("yolov8l-worldv2.pt")
    
    print("2. Setting classes (this is the slow part)...")
    t0 = time.time()
    model.set_classes(vocab)
    t1 = time.time()
    print(f"   Time: {t1 - t0:.2f}s")
    
    print("3. Saving custom model...")
    model.save("yolov8l-worldv2-custom.pt")
    
    print("4. Loading custom model...")
    t2 = time.time()
    custom_model = YOLO("yolov8l-worldv2-custom.pt")
    t3 = time.time()
    print(f"   Load time: {t3 - t2:.2f}s")
    
    print("5. Verifying classes...")
    print(f"   Classes: {custom_model.names}")
    
    print("6. Inference test...")
    import numpy as np
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    custom_model.predict(dummy, verbose=False)
    print("   Inference done.")

if __name__ == "__main__":
    test_save_load()
