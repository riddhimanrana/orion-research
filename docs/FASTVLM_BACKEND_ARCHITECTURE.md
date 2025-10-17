# FastVLM Backend Architecture

## Overview

The Orion FastVLM backend uses a **platform-aware architecture** with automatic backend selection based on the system's capabilities.

## Architecture

```
orion/backends/
├── torch_fastvlm.py           # Main entry point (router)
├── torch_fastvlm_legacy.py    # PyTorch/LLaVA implementation  
└── mlx_fastvlm.py             # MLX implementation (currently disabled)
```

## Backend Selection Logic

### Current State (MLX Disabled)

**All platforms**: PyTorch/LLaVA backend (`torch_fastvlm_legacy.py`)

### Future State (When MLX Issues Resolved)

1. **Apple Silicon (M1/M2/M3)**: MLX backend for optimized performance
2. **CUDA GPUs**: PyTorch backend with CUDA acceleration
3. **CPU**: PyTorch backend with CPU inference

## Implementation Details

### 1. Router (`torch_fastvlm.py`)

The main `FastVLMTorchWrapper` class acts as a router:

```python
class FastVLMTorchWrapper:
    def __init__(self, model_source=None, device=None, conv_mode="qwen_2", force_backend=None):
        # Platform detection
        use_mlx = False  # Currently disabled due to CoreML issues
        
        # Backend selection
        if use_mlx:
            self._backend = FastVLMMLXWrapper(...)
        else:
            self._backend = FastVLMTorchLegacyWrapper(...)
    
    # All methods delegate to the selected backend
    def generate_description(self, ...):
        return self._backend.generate_description(...)
```

### 2. PyTorch Backend (`torch_fastvlm_legacy.py`)

**Implementation**: Uses the vendored LLaVA package from repository root

**Features**:
- ✅ Works on MPS (Apple Metal), CUDA, and CPU
- ✅ Uses LLaVA conversation templates
- ✅ Handles generation_config.json backup/restore
- ✅ Supports `apple/FastVLM-0.5B` and `apple/FastVLM-0.5B-fp16`

**Device Selection**:
```python
def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```

### 3. MLX Backend (`mlx_fastvlm.py`) - DISABLED

**Status**: Currently disabled due to CoreML package compatibility issues

**Issue**: Apple's FastVLM models include CoreML vision tower packages (`.mlpackage`) that are corrupt or incompatible on some systems:
```
RuntimeError: Item does not exist for identifier: 058450D8-38E9-41BE-B9B5-C9865F6984F3
```

**Future Plan**: Re-enable when:
1. Apple releases fixed CoreML packages, or
2. MLX-VLM adds support for pure PyTorch vision towers

## Usage

### Basic Usage

```python
from orion.backends import FastVLMTorchWrapper

# Initialize (automatically selects backend)
model = FastVLMTorchWrapper()

# Generate description
result = model.generate_description(
    image="path/to/image.jpg",
    prompt="Describe this image",
    max_tokens=256,
    temperature=0.2
)
```

### Force Specific Backend

```python
# Force PyTorch (available)
model = FastVLMTorchWrapper(force_backend="pytorch")

# Force MLX (currently disabled, will fall back to PyTorch)
model = FastVLMTorchWrapper(force_backend="mlx")
```

### Custom Model

```python
# Use specific model
model = FastVLMTorchWrapper(model_source="apple/FastVLM-0.5B-fp16")

# Use local checkpoint
model = FastVLMTorchWrapper(model_source="./models/my-fastvlm")
```

## Performance

### Apple Silicon (M1/M2/M3)

- **Current**: PyTorch with MPS backend
- **Speed**: Good (~2-3 tok/s for 0.5B model)
- **Memory**: Moderate (~2GB VRAM)

### CUDA GPUs

- **Backend**: PyTorch with CUDA
- **Speed**: Excellent (~10-20 tok/s for 0.5B model on RTX 3090)
- **Memory**: Moderate (~2GB VRAM)

### CPU

- **Backend**: PyTorch CPU
- **Speed**: Slower (~0.5-1 tok/s)
- **Memory**: Low (~1GB RAM)

## Troubleshooting

### Issue: Model fails to load

**Solution**: Ensure LLaVA package is vendored in repository root:
```bash
ls llava/  # Should show llava package contents
```

### Issue: MPS errors on Apple Silicon

**Solution**: The fp16 variant may have quantization issues on MPS. Try the non-fp16 version:
```python
model = FastVLMTorchWrapper(model_source="apple/FastVLM-0.5B")
```

### Issue: "Module llava not found"

**Solution**: The backend automatically adds llava to sys.path. If issues persist:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3]))
```

## Future Enhancements

1. **MLX Re-enablement**: Once CoreML issues are resolved
2. **Quantization**: 4-bit/8-bit quantized models for lower memory
3. **Batch Inference**: Optimized batch processing
4. **Streaming**: Token-by-token generation
5. **LoRA Adapters**: Support for fine-tuned adapters

## References

- [Apple FastVLM Repository](https://github.com/apple/ml-fastvlm)
- [MLX-VLM](https://github.com/Blaizzy/mlx-vlm)
- [LLaVA](https://github.com/haotian-liu/LLaVA)
