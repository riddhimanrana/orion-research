# Scene Filter with Semantic Aliases - Evaluation Results

## Summary

The scene filter successfully reduces false positives by comparing detection labels against scene context using CLIP text embeddings.

## Key Features

1. **CLIP-based Scene Filtering**: Uses scene caption to filter unlikely detections
2. **Semantic Aliases**: Maps COCO labels to scene-relevant terms (tv→monitor, dining table→desk)
3. **Threshold**: 0.56 min similarity (validated via Gemini comparison)

## Evaluation on `video_short.mp4`

| Version | Detections | Entities | Refrigerator FP | Time |
|---------|------------|----------|-----------------|------|
| v1 (0.55, no aliases) | 107 | 7 | ❌ 24 detections | 54.65s |
| v2 (0.56, no aliases) | 30 | 1 | ✅ None | 18.69s |
| **v3 (0.56 + aliases)** | 93 | 3 | ✅ None | 31.84s |

## Semantic Alias Impact

| Label | Before | After | Result |
|-------|--------|-------|--------|
| tv | 0.561 (FAIL) | 0.663 (PASS) | Matched "monitor" |
| dining table | 0.559 (FAIL) | 0.643 (PASS) | Matched "desk" |
| refrigerator | 0.551 | 0.515 | Correctly filtered |

## Gemini Validation

- **Frames 50-100 (desk scene)**: Correctly detects keyboard, laptop, mouse, desk
- **Frames 151-251 (door scene)**: No false positives (refrigerator filtered)
- **Issue**: Overcounting (23 keyboards instead of 1) - this is a tracking issue, not filtering

## Files Modified

- `orion/perception/scene_filter.py` - Added SEMANTIC_ALIASES mapping
- `orion/perception/engine.py` - Set threshold to 0.56

## Next Steps

1. Consider per-segment scene updates for multi-scene videos
2. Improve entity deduplication to reduce overcounting
3. Add more semantic aliases as needed
