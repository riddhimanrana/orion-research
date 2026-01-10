# Orion Detection Backend Benchmark Results

**Date:** January 10, 2026  
**Platform:** Lambda Cloud (NVIDIA A10 GPU, 24GB VRAM)  
**Test Videos:** 
- `video_short.mp4` (10.08s, 302 frames)
- `test.mp4` (60.96s, 1827 frames)

## Summary

| Backend | Detections | Unique Tracks | Time (s) | Det/Frame | Recommendation |
|---------|------------|---------------|----------|-----------|----------------|
| **hybrid** | 287 | 29 | 45.8 | 4.8 | ✅ **Best choice** |
| **groundingdino** | 217 | 31 | 44.9 | 3.6 | Good for open-vocab |
| yoloworld | 129 | 12 | 68.4 | 2.1 | Limited classes |
| openvocab | 72 | 11 | 271.7 | 1.2 | ❌ Too slow |

## Backend Descriptions

### Hybrid (YOLO + GroundingDINO)
- **Primary:** YOLO11m for fast initial detection
- **Secondary:** GroundingDINO-tiny for verification and open-vocab enhancement
- **Strategy:** Uses YOLO for common objects, falls back to GroundingDINO when detection count is low or for specified classes

### GroundingDINO Primary
- Uses Hugging Face `IDEA-Research/grounding-dino-tiny`
- Pure open-vocabulary detection with text prompts
- Good diversity but some merged/noisy labels

### YOLO-World
- YOLO-based open-vocabulary detector
- Limited to COCO-style classes without custom prompts
- Slower than expected for this architecture

### OpenVocab (CLIP-based)
- OWL-ViT proposer + CLIP classifier
- Very slow due to per-proposal CLIP inference
- Poor recall, not recommended

## Full Video Test Results (test.mp4 - 60s)

**Hybrid Backend Performance:**
- Processing time: 219.9s (~3.6x realtime)
- Total detections: 2,746
- Unique tracks: 135
- Average: 7.5 detections/frame

### Top 25 Detected Classes (Hybrid)
```
picture frame painting: 1546
picture frame: 466
window: 460
curtains blinds: 452
flowers: 419
door: 361
kitchen island: 316
ceiling light: 314
stairs staircase: 283
ceiling light lamp: 192
wall clock: 187
ottoman: 168
book: 160
railing banister: 156
wall art: 151
kitchen cabinets: 145
kitchen cabinets cabinet bookshelf: 137
blinds: 134
rug: 126
tv: 120
remote remote control: 103
cabinet bookshelf bookcase: 100
door doorway: 95
bookshelf bookcase: 92
oven: 77
```

## Recommendations

1. **For production use:** Use `--detector-backend hybrid` 
2. **For maximum track diversity:** Use `--detector-backend groundingdino`
3. **Avoid:** `openvocab` backend until CLIP inference is optimized

## Commands Used

```bash
# Hybrid (recommended)
python -m orion.cli.run_tracks --video data/examples/video.mp4 \
  --episode my_test --detector-backend hybrid

# GroundingDINO primary
python -m orion.cli.run_tracks --video data/examples/video.mp4 \
  --episode my_test --detector-backend groundingdino

# YOLO-World with custom prompt
python -m orion.cli.run_tracks --video data/examples/video.mp4 \
  --episode my_test --detector-backend yoloworld \
  --yoloworld-prompt "chair . table . person . laptop . book"
```

## Hardware Notes

- **GPU:** NVIDIA A10 (24GB VRAM)
- **PyTorch:** 2.6.0
- **Ultralytics:** 8.3.217
- **Transformers:** 4.55.4
- **CUDA available:** Yes
