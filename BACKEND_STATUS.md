# Orion Backend Status

## ✓ All Backends Working

All four core backends are now operational:

| Backend | Status | Details |
|---------|--------|---------|
| **YOLO** | ✓ PASS | Detection working (yolo11m auto-downloaded) |
| **CLIP** | ✓ PASS | Image + text embeddings (512-dim, L2-normalized) |
| **DINO** | ✓ PASS | Vision embeddings (768-dim, L2-normalized) |
| **FastVLM** | ✓ PASS | Image descriptions (MLX-optimized) |

## DINO Backend Details

### Current Setup
- **Active Model**: DINOv2 Base (`facebook/dinov2-base`)
- **Source**: Public Hugging Face model (no gating)
- **Embedding Dim**: 768
- **Backend**: Transformers (AutoModel)

### DINOv3 Upgrade (Optional)

DINOv3 models are gated and require manual download:

#### 1. Request Access
Visit: https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/

#### 2. Download Weights
After approval, you'll receive download URLs via email. Use `wget`:

```bash
# Example (replace with your actual URL)
wget -O dinov3_vitb16_pretrain.pth "YOUR_DINOV3_URL_FROM_EMAIL"
```

**Important**: Use `wget`, NOT a browser (browser downloads may corrupt the file).

#### 3. Organize Weights
```bash
python scripts/setup_dinov3_weights.py --checkpoint dinov3_vitb16_pretrain.pth
```

This creates `models/dinov3-vitb16/` with the checkpoint and minimal config files.

#### 4. Verify
ModelManager will auto-detect and load DINOv3 if the directory exists:

```bash
python scripts/test_all_backends.py --device auto
```

You should see:
```
✓ DINO loaded (backend: transformers-local)
```

### Available DINOv3 Models

| Model | Params | Recommended For |
|-------|--------|----------------|
| ViT-S/16 | 21M | Fastest, mobile |
| ViT-B/16 | 86M | **Balanced (recommended)** |
| ViT-L/16 | 300M | High accuracy |
| ViT-H+/16 | 840M | Maximum accuracy |
| ViT-7B/16 | 6.7B | Research-grade |

### Fallback Behavior

If DINOv3 weights are not found, ModelManager automatically falls back to:
- **DINOv2** (public, no gating required)
- Same API, slightly lower accuracy than DINOv3

## Testing

Run comprehensive backend test:
```bash
python scripts/test_all_backends.py --device auto
```

Skip specific backends:
```bash
python scripts/test_all_backends.py --skip fastvlm
```

## Notebook Evaluation

Compare CLIP vs DINO for Re-ID:
```bash
jupyter notebook notebooks/reid_dinov3_evaluation.ipynb
```

The notebook includes:
- Embedding quality comparison
- Cosine similarity heatmaps
- Optional SAM masking
- Basic VO baseline
- Performance profiling

## Next Steps

1. **For Re-ID improvements**: Run the evaluation notebook to compare DINO vs CLIP
2. **For DINOv3 access**: Follow the download instructions above
3. **For production**: DINOv2 works well; upgrade to DINOv3 if you need maximum accuracy
