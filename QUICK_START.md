# Quick Start: TAO Evaluation

## What I Added

1. ✅ **R@K Metrics** (`orion/evaluation/recall_at_k.py`) - HyperGLM comparison
2. ✅ **Download Script** (`scripts/download_tao_videos.py`) - Get videos from HuggingFace  
3. ✅ **Frame Adapter** (`orion/utils/frame_adapter.py`) - Read TAO frame folders

## What You Should Do

### VSGR Dataset:
- ✅ Use `data/validation.json` (35MB)
- ❌ Ignore `data/aspire_test.json` (259MB, corrupted)

### 2-Video Test:

```bash
# 1. Download
python scripts/download_tao_videos.py \
    --validation-file data/validation.json \
    --max-videos 2

# 2. Update pipeline (add this line)
from orion.utils import create_video_capture

# 3. Run evaluation
python scripts/run_orion_evaluation.py \
    --aspire-file data/validation.json \
    --output-dir results_2_test
```

## Full Details
- See `EVALUATION_SUMMARY.md` for complete guide
- See `TAO_EVALUATION_SETUP.md` for technical details
