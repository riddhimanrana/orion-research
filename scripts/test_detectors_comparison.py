
import cv2
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from ultralytics import YOLO
import time
from PIL import Image
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval_detectors")

def run_eval():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    video_path = "/lambda/nfs/orion-core-fs/test.mp4"
    
    # Load models
    logger.info("Loading YOLO11x...")
    yolo11x = YOLO("yolo11x.pt")
    
    logger.info("Loading YOLO-World v2...")
    yoloworld = YOLO("yolov8x-worldv2.pt")
    # Set same classes as COCO for fair comparison
    coco_classes = list(yolo11x.names.values())
    yoloworld.set_classes(coco_classes)
    
    logger.info("Loading Grounding DINO (transformers)...")
    gdino_id = "IDEA-Research/grounding-dino-tiny" # Using tiny for speed
    processor = AutoProcessor.from_pretrained(gdino_id)
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(gdino_id).to(device)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    # Grab frame at 30s
    cap.set(cv2.CAP_PROP_POS_MSEC, 30000)
    ret, frame = cap.read()
    if not ret:
        logger.error("Failed to read frame")
        return
    cap.release()
    
    # Convert to RGB for GDINO
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)
    
    # Run YOLO11x
    t0 = time.time()
    yolo_results = yolo11x(frame, verbose=False)[0]
    yolo_time = time.time() - t0
    logger.info(f"YOLO11x: {len(yolo_results.boxes)} detections in {yolo_time:.3f}s")
    
    # Run YOLO-World
    t0 = time.time()
    yw_results = yoloworld(frame, verbose=False)[0]
    yw_time = time.time() - t0
    logger.info(f"YOLO-World: {len(yw_results.boxes)} detections in {yw_time:.3f}s")
    
    # Run Grounding DINO
    # Construct prompt from COCO classes
    text_prompt = " . ".join(coco_classes) + " ."
    t0 = time.time()
    inputs = processor(images=pil_img, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = gdino_model(**inputs)
    
    # Post-process GDINO
    target_sizes = torch.tensor([pil_img.size[::-1]])
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.3,
        text_threshold=0.25,
        target_sizes=target_sizes.to(device)
    )[0]
    gdino_time = time.time() - t0
    logger.info(f"GroundingDINO: {len(results['boxes'])} detections in {gdino_time:.3f}s")
    
    # Output comparison
    print("\nDETECTION COMPARISON (Frame at 30s):")
    print(f"{'Model':<20} | {'Detections':<12} | {'Latency':<10}")
    print("-" * 50)
    print(f"{'YOLO11x':<20} | {len(yolo_results.boxes):<12} | {yolo_time:.3f}s")
    print(f"{'YOLO-World':<20} | {len(yw_results.boxes):<12} | {yw_time:.3f}s")
    print(f"{'GroundingDINO':<20} | {len(results['boxes']):<12} | {gdino_time:.3f}s")
    
    # Print what GDINO found that others might have missed
    gdino_labels = results['labels']
    print(f"\nGroundingDINO found: {', '.join(gdino_labels) if gdino_labels else 'Nothing'}")

if __name__ == "__main__":
    run_eval()
