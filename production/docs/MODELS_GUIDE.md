# Model Configuration Guide for Orion Research

This guide explains the models used in the Orion Research "From Moments to Memory" pipeline and how to configure them correctly.

## Overview

The pipeline uses two main models:

1. **YOLO11m**: Object detection and tracking (Part 1)
2. **FastVLM-0.5B**: Visual language model for scene understanding (Part 1)

## 1. YOLO11 Model

### Installation

```bash
pip install ultralytics
```

### Usage

```python
from ultralytics import YOLO

# Load YOLO11m model (medium size)
model = YOLO("yolo11m.pt")

# The model will be automatically downloaded on first use
# Saved to: ~/.ultralytics/weights/yolo11m.pt

# Run detection
results = model(image_path)

# Or with options
results = model(
    image_path,
    conf=0.25,      # Confidence threshold
    iou=0.45,       # NMS IoU threshold
    imgsz=640,      # Input image size
    device='mps'    # Device: 'cpu', 'mps', 'cuda'
)
```

### Model Variants

| Model | Size | mAP | Speed (CPU) | Speed (GPU) | Use Case |
|-------|------|-----|-------------|-------------|----------|
| yolo11n.pt | 2.6M | 39.5 | ~5ms | ~1ms | Mobile/Edge |
| yolo11s.pt | 9.4M | 47.0 | ~10ms | ~2ms | Mobile |
| **yolo11m.pt** | 20.1M | **51.5** | **~20ms** | **~3ms** | **Balanced (recommended)** |
| yolo11l.pt | 25.3M | 53.4 | ~30ms | ~4ms | High accuracy |
| yolo11x.pt | 56.9M | 54.7 | ~50ms | ~6ms | Best accuracy |

**Why YOLO11m?**
- **Balanced performance**: Good accuracy with reasonable speed
- **80 COCO classes**: Detects common objects (person, car, dog, etc.)
- **Strong tracking**: Works well with video sequences
- **Apple Silicon optimized**: Runs efficiently on MPS (Mac GPU)

### Configuration in Part 1

```python
# In production/part1_perception_engine.py
from ultralytics import YOLO

def load_yolo_model():
    """Load YOLO11m model for object detection."""
    model = YOLO("yolo11m.pt")
    return model

# Use in perception engine
yolo_model = load_yolo_model()
results = yolo_model(
    frame,
    conf=0.25,          # Minimum confidence
    iou=0.45,           # NMS threshold
    device='mps',       # Use Mac GPU
    verbose=False       # Suppress output
)
```

### YOLO11 Features

- **Multi-object tracking**: Persistent IDs across frames
- **Pose estimation**: Human pose keypoints (with yolo11m-pose.pt)
- **Segmentation**: Instance segmentation (with yolo11m-seg.pt)
- **Classification**: Image classification

For this project, we use standard detection (bounding boxes + classes).

## 2. FastVLM Model

### Your Custom Model

**Location**: `models/fastvlm-0.5b-captions/`

**Specifications**:
- **Base**: FastVLM-0.5B (Qwen2-0.5B language model)
- **Vision encoder**: FastViTHD (3072-dim features)
  - Same encoder across 0.5B, 1.5B, and 7B models
  - Based on MobileCLIP-L (1024 width)
  - Input: 336x336 RGB images
  - Patch size: 64x64
- **Fine-tuning**: Optimized for caption generation
- **Quantization**: 4-bit (group_size=64)
- **Size**: ~300MB (vs ~1GB for non-quantized)
- **Image token index**: 151646

### Installation

```bash
pip install transformers>=4.36.0 safetensors torch pillow
```

### Usage

```python
from production.fastvlm_wrapper import load_fastvlm

# Load your fine-tuned model
model = load_fastvlm()  # Auto-loads from models/fastvlm-0.5b-captions/

# Or specify path explicitly
model = load_fastvlm(model_path="models/fastvlm-0.5b-captions")

# Generate description
description = model(
    image_path="image.jpg",
    prompt="Describe this image in detail.",
    max_new_tokens=100,
    temperature=0.7
)
```

### Vision Encoder (FastViTHD)

The vision encoder is **already exported** in your model directory:

**File**: `models/fastvlm-0.5b-captions/fastvithd.mlpackage`

This is a CoreML-compiled version optimized for:
- Apple Neural Engine
- Low latency (~10-20ms)
- Low power consumption
- On-device inference (iOS/macOS)

#### Vision Encoder Details

**Same across all FastVLM sizes!**

The vision encoder is identical for:
- FastVLM-0.5B (your model)
- FastVLM-1.5B
- FastVLM-7B

Only the **language model** size changes. The vision encoder always produces:
- **Input**: 336x336 RGB image
- **Output**: 26 tokens × 3072 dimensions
  - 25 spatial patches (5×5 grid with 64px patches)
  - 1 CLS token
- **Architecture**: MobileCLIP-L based
  - Fused attention mechanisms
  - Efficient linear layers
  - Optimized for Apple hardware

The **projector (MLP)** then maps these features to the LLM's hidden size:
- 0.5B: 3072 → 896
- 1.5B: 3072 → 1536
- 7B: 3072 → 3584

#### If You Need to Re-export the Vision Encoder

```bash
# From ml-fastvlm/model_export directory
python export_vision_encoder.py \
    --model-path /path/to/models/fastvlm-0.5b-captions

# This will:
# 1. Load full FastVLM model
# 2. Extract vision tower: model.get_vision_tower()
# 3. Export to CoreML: fastvithd.mlpackage
# 4. Save to model directory
```

### Model Comparison

| Aspect | 0.5B (Your Model) | 1.5B | 7B |
|--------|-------------------|------|-----|
| **Vision Encoder** | FastViTHD (3072-dim) | Same | Same |
| **LLM** | Qwen2-0.5B (896 hidden) | Qwen2-1.5B (1536) | Qwen2-7B (3584) |
| **Projector** | MLP 3072→896 | MLP 3072→1536 | MLP 3072→3584 |
| **Size (4-bit)** | ~300MB | ~900MB | ~4GB |
| **Speed** | ~50ms/image | ~100ms/image | ~300ms/image |
| **Quality** | Good | Better | Best |
| **Use Case** | Real-time, mobile | Balanced | Research |

## Integration in Part 1

### Configuration

```python
# In production/part1_perception_engine.py

# YOLO11 for object detection
from ultralytics import YOLO
yolo_model = YOLO("yolo11m.pt")

# FastVLM for scene understanding
from production.fastvlm_wrapper import load_fastvlm
vlm_model = load_fastvlm()  # Uses models/fastvlm-0.5b-captions/

# Process frame
def process_frame(frame):
    # 1. Object detection with YOLO
    detections = yolo_model(
        frame,
        conf=0.25,
        device='mps',
        verbose=False
    )
    
    # 2. Scene description with FastVLM
    description = vlm_model(
        frame,
        prompt="Describe the scene, focusing on actions and interactions.",
        max_new_tokens=150
    )
    
    return {
        'detections': detections,
        'description': description
    }
```

### Device Configuration

**For Apple Silicon (M1/M2/M3):**

```python
# YOLO11
yolo_model = YOLO("yolo11m.pt")
results = yolo_model(frame, device='mps')  # Use Mac GPU

# FastVLM
vlm_model = load_fastvlm(device='mps')  # Use Mac GPU
```

**For NVIDIA GPU:**

```python
# YOLO11
results = yolo_model(frame, device='cuda')  # Use NVIDIA GPU

# FastVLM
vlm_model = load_fastvlm(device='cuda')  # Use NVIDIA GPU
```

**For CPU only:**

```python
# YOLO11
results = yolo_model(frame, device='cpu')

# FastVLM
vlm_model = load_fastvlm(device='cpu')
```

## Model Files Structure

```
models/
├── fastvlm-0.5b-captions/          # Your fine-tuned FastVLM
│   ├── config.json                  # Model configuration
│   ├── model.safetensors            # Model weights (quantized)
│   ├── model.safetensors.index.json # Weight index
│   ├── tokenizer.json               # Tokenizer
│   ├── vocab.json                   # Vocabulary
│   ├── preprocessor_config.json     # Image preprocessing
│   ├── processor_config.json        # Processor config
│   ├── fastvithd.mlpackage/        # CoreML vision encoder ✓
│   └── ... (other tokenizer files)
│
└── (YOLO models auto-downloaded)
    └── ~/.ultralytics/weights/
        └── yolo11m.pt              # Auto-downloaded
```

## Testing the Models

### Test YOLO11

```bash
# Test with image
python -c "
from ultralytics import YOLO
model = YOLO('yolo11m.pt')
results = model('data/examples/example1.jpg')
results[0].show()
"

# Test with video
python -c "
from ultralytics import YOLO
model = YOLO('yolo11m.pt')
results = model('video.mp4', stream=True)
for r in results:
    print(f'Frame: {len(r.boxes)} objects detected')
"
```

### Test FastVLM

```bash
# Test your fine-tuned model
python production/fastvlm_wrapper.py data/examples/example1.jpg

# Or with custom prompt
python production/fastvlm_wrapper.py \
    data/examples/example1.jpg \
    "What objects are visible in this image?"
```

### Test Both Together

```python
# test_models.py
from ultralytics import YOLO
from production.fastvlm_wrapper import load_fastvlm
from PIL import Image

# Load models
print("Loading YOLO11m...")
yolo = YOLO("yolo11m.pt")

print("Loading FastVLM...")
vlm = load_fastvlm()

# Test image
image_path = "data/examples/example1.jpg"
image = Image.open(image_path)

# YOLO detection
print("\nRunning YOLO detection...")
detections = yolo(image, verbose=False)
print(f"Detected {len(detections[0].boxes)} objects")
for box in detections[0].boxes:
    cls = int(box.cls[0])
    conf = float(box.conf[0])
    name = yolo.names[cls]
    print(f"  - {name}: {conf:.2f}")

# FastVLM description
print("\nGenerating description...")
description = vlm(image, "Describe this image in detail.")
print(f"Description: {description}")
```

## Performance Tips

### For YOLO11

1. **Batch processing**: Process multiple frames at once
   ```python
   results = model([frame1, frame2, frame3])
   ```

2. **Reduce image size**: For faster processing
   ```python
   results = model(frame, imgsz=320)  # Default: 640
   ```

3. **Adjust confidence**: Trade recall for precision
   ```python
   results = model(frame, conf=0.5)  # Higher = fewer detections
   ```

### For FastVLM

1. **Use quantized model**: Already using 4-bit (good!)

2. **Limit tokens**: For faster generation
   ```python
   description = vlm(image, max_new_tokens=50)  # Shorter responses
   ```

3. **Lower temperature**: For more deterministic output
   ```python
   description = vlm(image, temperature=0.3)  # Less random
   ```

4. **Batch processing**: Process multiple images
   ```python
   descriptions = vlm.batch_generate([img1, img2, img3])
   ```

## Troubleshooting

### YOLO11 Issues

**"Model not found"**
```bash
# Download manually
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11m.pt
```

**"MPS not available"**
- Check macOS version (requires 12.3+)
- Use `device='cpu'` instead

### FastVLM Issues

**"Model not found at models/fastvlm-0.5b-captions"**
- Check the path is correct
- Verify all model files are present

**"Out of memory"**
- Your model is 4-bit quantized, should be fine
- Try `device='cpu'` if MPS runs out of memory

**"Slow inference"**
- Check you're using MPS/CUDA: `device='mps'`
- Reduce `max_new_tokens`
- Consider using MLX-optimized version for Apple Silicon

## Next Steps

1. **Update Part 1**: Integrate YOLO11m and your FastVLM model
2. **Test pipeline**: Run end-to-end test
3. **Optimize**: Tune confidence thresholds and prompts
4. **Benchmark**: Measure FPS and accuracy

## References

- YOLO11: https://docs.ultralytics.com/models/yolo11/
- FastVLM: https://arxiv.org/abs/2411.11671
- HuggingFace Transformers: https://huggingface.co/docs/transformers
- Ultralytics: https://github.com/ultralytics/ultralytics
