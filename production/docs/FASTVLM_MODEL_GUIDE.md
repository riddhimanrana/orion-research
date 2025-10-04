"""
FastVLM Model Configuration and Setup Guide
============================================

This document explains the FastVLM model structure and how to use your fine-tuned model.

## Model Architecture

FastVLM consists of three components:
1. **Vision Encoder (FastViTHD)**: Processes images → 3072-dim features
2. **Projector (MLP)**: Maps vision features to LLM hidden size
3. **Language Model**: Generates text (0.5B/1.5B/7B Qwen2)

## Your Custom Model

Location: `models/fastvlm-0.5b-captions/`

**Specifications:**
- Base: FastVLM-0.5B (Qwen2-0.5B language model)
- Fine-tuned: Caption generation
- Quantized: 4-bit (group_size=64)
- Vision encoder: FastViTHD (mobileclip_l_1024)
- Vision features: 3072 dimensions
- LLM hidden size: 896 dimensions
- Image token index: 151646

**Files:**
- `model.safetensors` / `model.safetensors.index.json`: Model weights
- `fastvithd.mlpackage`: CoreML vision encoder (already exported!)
- `config.json`: Model configuration
- `tokenizer.json`, `vocab.json`: Tokenizer files
- `preprocessor_config.json`: Image preprocessing config
- `processor_config.json`: Processor config

## Vision Encoder Export

### Already Exported! ✓

Your model directory already contains `fastvithd.mlpackage`, which is the CoreML
exported vision encoder. This was created using the `export_vision_encoder.py` script.

### If You Need to Re-export

```bash
# From the ml-fastvlm/model_export directory
python export_vision_encoder.py \
    --model-path /path/to/models/fastvlm-0.5b-captions
```

This will:
1. Load the full FastVLM model
2. Extract the vision tower: `model.get_vision_tower()`
3. Export to CoreML: `fastvithd.mlpackage`
4. Save to model directory

## Vision Encoder Across Model Sizes

**Important**: The vision encoder is THE SAME across all FastVLM sizes!

| Model Size | Vision Encoder | Vision Output | LLM Hidden | Projector |
|------------|---------------|---------------|------------|-----------|
| 0.5B       | FastViTHD     | 3072-dim      | 896        | MLP 3072→896 |
| 1.5B       | FastViTHD     | 3072-dim      | 1536       | MLP 3072→1536 |
| 7B         | FastViTHD     | 3072-dim      | 3584       | MLP 3072→3584 |

**Why?**
- Vision processing is computationally expensive
- Using a shared, optimized vision encoder (FastViTHD) keeps inference fast
- Only the language understanding scales with model size
- The projector adapts vision features to the LLM's hidden size

## Using the Model

### Option 1: Use Full Model (Python with transformers)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

# Load your fine-tuned model
model_path = "models/fastvlm-0.5b-captions"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Generate caption
image = Image.open("image.jpg")
prompt = "<image>Describe this image."
inputs = tokenizer(prompt, return_tensors="pt")
# Add image to inputs...
outputs = model.generate(**inputs, max_new_tokens=100)
caption = tokenizer.decode(outputs[0])
```

### Option 2: Use Vision Encoder Only (CoreML)

```python
import coremltools as ct
import numpy as np
from PIL import Image

# Load CoreML vision encoder
vision_model = ct.models.MLModel("models/fastvlm-0.5b-captions/fastvithd.mlpackage")

# Process image
image = Image.open("image.jpg").resize((336, 336))
image_array = np.array(image).transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, 336, 336)

# Extract features
features = vision_model.predict({"images": image_array})["image_features"]
# features shape: (1, num_patches, 3072)
```

### Option 3: Use MLX for Apple Silicon (Fastest)

If you've exported the model to MLX format (using the mlx-vlm instructions):

```bash
# Generate with MLX (optimized for Apple Silicon)
python -m mlx_vlm.generate \
    --model models/fastvlm-0.5b-captions-mlx \
    --image image.jpg \
    --prompt "Describe this image." \
    --max-tokens 256 \
    --temp 0.0
```

## For Orion Research Pipeline

### Current Setup

Your `fastvlm_wrapper.py` should:

1. **Load the model from local path**:
   ```python
   model_path = "models/fastvlm-0.5b-captions"
   ```

2. **Handle quantized weights**:
   - Your model is 4-bit quantized
   - transformers library should handle this automatically with safetensors

3. **Use correct image token index**:
   ```python
   IMAGE_TOKEN_INDEX = 151646  # From your config.json
   ```

4. **Image preprocessing**:
   - Input size: 336x336 (from preprocessor_config.json)
   - Patch size: 64 (from processor_config.json)
   - Mean/std normalization (CLIP-style)

### Recommended Update

```python
class FastVLMModel:
    def __init__(
        self,
        model_path: str = "models/fastvlm-0.5b-captions",  # Local path
        device: Optional[str] = None,
    ):
        # Load from local directory
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.IMAGE_TOKEN_INDEX = 151646  # From config
```

## Vision Encoder Technical Details

### FastViTHD Architecture

- **Base**: MobileCLIP-L (1024 width)
- **Input**: RGB images, 336x336 pixels
- **Patches**: 64x64 patch size → 5x5 = 25 patches (+ CLS token)
- **Output**: 26 tokens × 3072 dimensions
- **Optimization**: 
  - Fused attention
  - Efficient linear layers
  - Optimized for Apple Neural Engine

### Why CoreML Export?

- **On-device inference**: Runs on iPhone/iPad/Mac
- **Neural Engine**: Uses Apple's dedicated ML hardware
- **Low latency**: ~10-20ms for vision encoding
- **Low power**: Efficient for mobile devices

The CoreML package (`fastvithd.mlpackage`) contains:
- Compiled model for Apple Neural Engine
- Optimized operations (conv, attention, etc.)
- Metadata for input/output shapes

## Next Steps for Integration

1. **Update fastvlm_wrapper.py**: Point to local model path
2. **Update YOLO**: Use `ultralytics` with `yolo11m.pt`
3. **Test vision encoder**: Verify features extraction works
4. **Test full model**: Verify caption generation works
5. **Integrate with Part 1**: Use in perception engine

## Troubleshooting

### "Cannot load safetensors"
```bash
pip install safetensors transformers>=4.36.0
```

### "Out of memory"
- Your model is 4-bit quantized, should be ~300MB
- Use device="mps" on Mac, device="cpu" if needed

### "Image token not recognized"
- Check `IMAGE_TOKEN_INDEX = 151646` in config
- Verify tokenizer has `<image>` token in vocab

### "Vision encoder not found"
- The vision encoder is embedded in the full model
- For CoreML: use `fastvithd.mlpackage`
- No need to load separately unless doing iOS deployment

## References

- FastVLM Paper: https://arxiv.org/abs/2411.11671
- HuggingFace: https://huggingface.co/apple/FastVLM-0.5B
- MobileCLIP: https://github.com/apple/ml-mobileclip
- Export Scripts: `ml-fastvlm/model_export/`
