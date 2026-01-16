#!/bin/bash
# Reprocess 5 test videos with improved YOLO-World vocabulary

YOLO_PROMPT=$(cat pvsg_yoloworld_prompt.txt)

# Select 5 videos for testing (from batch 1)
VIDEOS=(
    "0001_4164158586"
    "0003_3396832512"
    "0004_11566980553"
    "0008_6225185844"
    "0010_8610561401"
)

for vid in "${VIDEOS[@]}"; do
    VIDEO_PATH="datasets/PVSG/VidOR/mnt/lustre/jkyang/CVPR23/openpvsg/data/vidor/videos/${vid}.mp4"
    
    if [ ! -f "$VIDEO_PATH" ]; then
        echo "❌ Not found: $VIDEO_PATH"
        continue
    fi
    
    echo "=========================================="
    echo "Processing: $vid"
    echo "=========================================="
    
    conda run -n orion python -m orion.cli.run_showcase \
        --episode "$vid" \
        --video "$VIDEO_PATH" \
        --detector-backend yoloworld \
        --yoloworld-prompt "$YOLO_PROMPT" \
        --force-phase1
    
    echo "✓ Done: $vid"
done

echo ""
echo "✓ All videos reprocessed with improved vocabulary"
