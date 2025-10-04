# Part 1 Model Update Summary

**Date:** January 2025  
**Status:** ✅ Updated to use YOLO11m and custom fine-tuned FastVLM-0.5B

---

## Changes Made

### 1. Object Detection: YOLO11m

**Previous:** Generic YOLO reference  
**Updated:** YOLO11m (ultralytics)

**Configuration:**
```python
from ultralytics import YOLO
model = YOLO('yolo11m.pt')  # Auto-downloads to ~/.ultralytics/weights/
```

**Specifications:**
- **Size:** 20.1M parameters
- **Classes:** 80 COCO classes
- **mAP:** 51.5
- **Speed:** ~3ms on GPU, ~20ms on CPU
- **Use case:** Balanced performance for video processing

**Why YOLO11m?**
- Excellent balance between accuracy and speed
- Native support for video tracking (persistent IDs)
- Optimized for Apple Silicon (MPS) and CUDA
- Well-maintained by Ultralytics team

### 2. Description Generation: FastVLM-0.5B (Custom Fine-tuned)

**Previous:** Generic FastVLM reference  
**Updated:** Local fine-tuned model at `models/fastvlm-0.5b-captions/`

**Configuration:**
```python
from production.fastvlm_wrapper import load_fastvlm
model = load_fastvlm()  # Loads from models/fastvlm-0.5b-captions/
```

**Specifications:**
- **Base:** FastVLM-0.5B (Qwen2-0.5B language model)
- **Vision Encoder:** FastViTHD (3072-dim features, same across 0.5B/1.5B/7B)
- **Fine-tuning:** Optimized for caption generation
- **Quantization:** 4-bit (group_size=64)
- **Size:** ~300MB (vs ~1GB unquantized)
- **Image Token:** 151646
- **Location:** `models/fastvlm-0.5b-captions/`

**Model Files:**
```
models/fastvlm-0.5b-captions/
├── config.json                    # Model configuration
├── model.safetensors              # Quantized weights
├── model.safetensors.index.json   # Weight index
├── tokenizer.json                 # Tokenizer
├── vocab.json                     # Vocabulary
├── preprocessor_config.json       # Image preprocessing
├── processor_config.json          # Processor config
├── fastvithd.mlpackage/          # CoreML vision encoder (optional)
└── ... (other tokenizer files)
```

**Why Your Custom Model?**
- Fine-tuned specifically for caption generation
- 4-bit quantization for efficient inference
- Local deployment (no API calls)
- Optimized for your use case
- Faster than downloading from HuggingFace each time

### 3. Vision Encoder Details

**FastViTHD (same across all FastVLM sizes):**
- Input: 336x336 RGB images
- Patch size: 64x64
- Output: 26 tokens × 3072 dimensions (25 patches + 1 CLS token)
- Based on MobileCLIP-L architecture
- Optimized for Apple Neural Engine

**Projector (size-dependent):**
- 0.5B: MLP 3072 → 896 (your model)
- 1.5B: MLP 3072 → 1536
- 7B: MLP 3072 → 3584

---

## Code Changes

### File: `production/part1_perception_engine.py`

#### 1. Updated Module Docstring
Added model specifications:
```python
"""
Models Used:
- Object Detection: YOLO11m (ultralytics)
- Scene Detection: FastViT-T8 (timm)
- Visual Embeddings: ResNet50 (timm)
- Description Generation: FastVLM-0.5B (custom fine-tuned)
  * Location: models/fastvlm-0.5b-captions/
  * Quantization: 4-bit
  * Vision encoder: FastViTHD (3072-dim)
"""
```

#### 2. Updated YOLO11 Loading (Line ~203)
```python
def get_object_detector(self):
    """Load YOLO11m model for object detection"""
    from ultralytics import YOLO
    logger.info("Loading YOLO11m model...")
    
    # Load YOLO11m - will auto-download if not present
    model = YOLO('yolo11m.pt')
    
    logger.info("✓ YOLO11m model loaded successfully")
    logger.info(f"  Model: yolo11m.pt (20.1M params, 80 COCO classes)")
```

#### 3. Updated FastVLM Loading (Line ~606)
```python
def load_fastvlm_model():
    """
    Load custom fine-tuned FastVLM model
    
    Uses the locally fine-tuned FastVLM-0.5B model optimized for captions:
    - Location: models/fastvlm-0.5b-captions/
    - Quantization: 4-bit (group_size=64)
    - Vision encoder: FastViTHD (3072-dim features)
    - Size: ~300MB
    """
    from production.fastvlm_wrapper import load_fastvlm
    logger.info("Loading custom fine-tuned FastVLM-0.5B model...")
    logger.info("  Model: models/fastvlm-0.5b-captions/ (4-bit quantized)")
    
    # Load local fine-tuned model
    _FASTVLM_MODEL = load_fastvlm(
        model_path=None,  # Uses default: models/fastvlm-0.5b-captions/
        device=None,      # Auto-detect (MPS/CUDA/CPU)
        dtype=None,       # Auto-select based on device
    )
    
    logger.info("✓ FastVLM model loaded successfully")
    logger.info(f"  Device: {_FASTVLM_MODEL.device}")
    logger.info(f"  Vision encoder: FastViTHD (3072-dim)")
```

#### 4. Updated Description Generation (Line ~651)
```python
def generate_rich_description(image, object_class="", use_fastvlm=True):
    """
    Generate rich description using custom fine-tuned FastVLM-0.5B
    
    Model details:
    - Base: FastVLM-0.5B (Qwen2-0.5B language model)
    - Fine-tuned for caption generation
    - 4-bit quantized for efficiency
    - Vision encoder: FastViTHD (3072-dim features)
    
    Args:
        image: Cropped object image (BGR format from OpenCV)
        object_class: Object class from YOLO11 (for context)
        use_fastvlm: Whether to use FastVLM or fallback to placeholder
    """
    # ... (same generation logic)
```

---

## File Updates

### Updated Files:
1. ✅ `production/part1_perception_engine.py` - Uses YOLO11m and local FastVLM
2. ✅ `production/fastvlm_wrapper.py` - Updated to load local model
3. ✅ `production/FASTVLM_MODEL_GUIDE.md` - Comprehensive model documentation
4. ✅ `production/MODELS_GUIDE.md` - Complete configuration guide

### New Documentation:
- `production/FASTVLM_MODEL_GUIDE.md` (260 lines)
- `production/MODELS_GUIDE.md` (420 lines)
- `production/PART1_MODEL_UPDATE.md` (this file)

---

## Dependencies

### Required Packages:
```bash
# Core dependencies
pip install ultralytics          # YOLO11m
pip install transformers>=4.36.0 # FastVLM
pip install safetensors          # Model loading
pip install torch                # PyTorch
pip install opencv-python        # Video processing
pip install pillow               # Image processing
pip install numpy                # Numerical operations

# Optional but recommended
pip install timm                 # FastViT, ResNet50
pip install tqdm                 # Progress bars
```

### Install All:
```bash
pip install -r requirements.txt
```

---

## Testing the Updates

### 1. Test YOLO11m
```bash
python -c "
from ultralytics import YOLO
model = YOLO('yolo11m.pt')
print('✓ YOLO11m loaded successfully')
print(f'  Classes: {len(model.names)}')
print(f'  Model: {model.model_name}')
"
```

### 2. Test FastVLM
```bash
python production/fastvlm_wrapper.py data/examples/example1.jpg
```

### 3. Test Part 1
```bash
python production/part1_perception_engine.py \
    --video data/examples/example_video.mp4 \
    --output output/part1_test
```

---

## Performance Expectations

### YOLO11m:
- **Speed:** ~3-5ms per frame on Apple M1/M2 (MPS)
- **Speed:** ~2-3ms per frame on NVIDIA GPU (CUDA)
- **Speed:** ~15-20ms per frame on CPU
- **Accuracy:** mAP 51.5 on COCO dataset

### FastVLM-0.5B (4-bit quantized):
- **Loading:** ~2-3 seconds (first load)
- **Speed:** ~50-100ms per description on MPS
- **Speed:** ~30-50ms per description on CUDA
- **Speed:** ~200-300ms per description on CPU
- **Memory:** ~300MB model size
- **Quality:** Optimized for captions (your fine-tuning)

### Combined Pipeline:
- **Frame selection:** ~5ms per frame (scene detection)
- **Object detection:** ~3-5ms per frame (YOLO11m)
- **Embedding:** ~10ms per object (ResNet50)
- **Description:** ~50-100ms per object (FastVLM)
- **Overall:** Can process ~10-20 frames/second (depends on object count)

---

## Troubleshooting

### "YOLO11m not found"
```bash
# Will auto-download on first use
python -c "from ultralytics import YOLO; YOLO('yolo11m.pt')"
```

### "FastVLM model not found"
```bash
# Check model path
ls -la models/fastvlm-0.5b-captions/

# Should see:
# - config.json
# - model.safetensors
# - tokenizer files
# - fastvithd.mlpackage
```

### "Out of memory"
- Your FastVLM is 4-bit quantized (~300MB), should be fine
- Use `device='cpu'` if GPU memory is limited
- Reduce `Config.NUM_WORKERS` to 1 or 2

### "Slow inference"
- Check you're using GPU: `device='mps'` (Mac) or `device='cuda'` (NVIDIA)
- Reduce `Config.DESCRIPTION_MAX_TOKENS` (default: 150)
- Lower `Config.TARGET_FPS` (default: 4.0)

---

## Next Steps

1. ✅ **Models updated** - YOLO11m and custom FastVLM integrated
2. 📝 **Test pipeline** - Run Part 1 on sample video
3. 🔧 **Tune parameters** - Adjust confidence thresholds, prompts
4. 📊 **Benchmark** - Measure FPS and accuracy on your videos
5. 🔗 **Integrate Part 2** - Connect to knowledge graph generation

---

## References

- **YOLO11:** https://docs.ultralytics.com/models/yolo11/
- **FastVLM:** https://arxiv.org/abs/2411.11671
- **Ultralytics:** https://github.com/ultralytics/ultralytics
- **Transformers:** https://huggingface.co/docs/transformers
- **Model Guide:** `production/MODELS_GUIDE.md`
- **FastVLM Guide:** `production/FASTVLM_MODEL_GUIDE.md`

---

**Status:** ✅ Part 1 successfully updated to use YOLO11m and custom fine-tuned FastVLM-0.5B!
