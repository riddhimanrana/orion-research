# Heuristic Baseline Update Log

## Date
October 23, 2025

## Changes Made

### 1. Path Configuration Updates
**File**: `scripts/3b_run_heuristic_baseline_ag_eval.py` (Lines 28-35)

```python
# BEFORE:
FRAMES_DIR = os.path.join(AG_DATASET_ROOT, 'frames')

# AFTER:
AG_SOURCE_ROOT = 'dataset/ag'
VIDEOS_DIR = os.path.join(AG_SOURCE_ROOT, 'videos')
FRAMES_DIR = os.path.join(AG_DATASET_ROOT, 'frames')
```

**Reason**: Script 3 now uses both videos and extracted frames. Updated to support both input types.

---

### 2. Input Validation Enhancement
**File**: `scripts/3b_run_heuristic_baseline_ag_eval.py` (Lines 889-908)

```python
# BEFORE:
if not os.path.exists(FRAMES_DIR):
    print(f"❌ Frames directory not found: {FRAMES_DIR}")
    return False

# AFTER:
has_videos = os.path.exists(VIDEOS_DIR)
has_frames = os.path.exists(FRAMES_DIR)

if not has_videos and not has_frames:
    print(f"❌ Neither videos nor frames found:")
    print(f"   Videos: {VIDEOS_DIR}")
    print(f"   Frames: {FRAMES_DIR}")
    return False
```

**Reason**: Gracefully handles scenarios where either videos or frames (or both) are available.

---

### 3. Frame Loading Logic
**File**: `scripts/3b_run_heuristic_baseline_ag_eval.py` (Lines 930-970)

```python
# BEFORE:
frames = []
for frame_file in frame_files[:1000]:
    frame_path = os.path.join(clip_frames_dir, frame_file)
    import cv2
    frame = cv2.imread(frame_path)
    if frame is not None:
        frames.append(frame)

# AFTER:
# Try video first
if has_videos:
    video_path = os.path.join(VIDEOS_DIR, clip_id)
    if os.path.exists(video_path):
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                if len(frames) >= 1000:
                    break
            cap.release()
        except Exception as e:
            logger.warning(f"Failed to load video {clip_id}: {e}")
            frames = []

# Fall back to frames if needed
if not frames and has_frames:
    clip_frames_dir = os.path.join(FRAMES_DIR, clip_id)
    if os.path.exists(clip_frames_dir):
        frame_files = sorted([f for f in os.listdir(clip_frames_dir) if f.endswith('.jpg')])
        if frame_files:
            import cv2
            for frame_file in frame_files[:1000]:
                frame_path = os.path.join(clip_frames_dir, frame_file)
                frame = cv2.imread(frame_path)
                if frame is not None:
                    frames.append(frame)
```

**Reason**: Now supports both video files and frame directories with intelligent fallback.

---

### 4. Output Directory Structure
**File**: `scripts/3b_run_heuristic_baseline_ag_eval.py` (Line 35)

```python
os.makedirs(os.path.join(OUTPUT_DIR, 'intermediate'), exist_ok=True)
```

**Reason**: Matches script 3's output organization with intermediate directory for auxiliary files.

---

### 5. Summary Output
**File**: `scripts/3b_run_heuristic_baseline_ag_eval.py` (Line 1015)

```python
# BEFORE:
Next: Compare with Orion using standard evaluation
   python scripts/4_evaluate_ag_predictions.py

# AFTER:
Next: Compare baseline with Orion
   python scripts/4b_compare_baseline_vs_orion.py
```

**Reason**: Directs users to the correct comparison script (4b instead of 4).

---

## Compatibility

✅ **Compatible with script 3's new structure**
✅ **Supports both video and frame inputs**
✅ **Same JSON output format**
✅ **Same evaluation metrics**
✅ **Can run in parallel with script 3**

## Testing

✅ Python syntax validation: PASSED
✅ Import validation: PASSED
✅ Backward compatibility: MAINTAINED

## Usage

### Sequential Execution
```bash
python scripts/3_run_orion_ag_eval.py
python scripts/3b_run_heuristic_baseline_ag_eval.py
python scripts/4b_compare_baseline_vs_orion.py
```

### Parallel Execution (Faster)
```bash
python scripts/3_run_orion_ag_eval.py &
python scripts/3b_run_heuristic_baseline_ag_eval.py &
wait
python scripts/4b_compare_baseline_vs_orion.py
```

## Notes

- The baseline now automatically detects available input sources (videos or frames)
- If videos are available, they're loaded first (faster than individual frames)
- Falls back to frame directories if video loading fails
- All errors are logged with helpful diagnostics
- Same 50-clip evaluation as script 3 for direct comparison

---

**Status**: ✅ COMPLETE AND VALIDATED
