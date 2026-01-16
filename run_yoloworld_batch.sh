#!/bin/bash
PROMPT=$(cat pvsg_yoloworld_prompt.txt)
VIDS=("0003_3396832512" "0003_6141007489" "0004_11566980553" "0005_2505076295")
BASE="datasets/PVSG/VidOR/mnt/lustre/jkyang/CVPR23/openpvsg/data/vidor/videos"

for vid in "${VIDS[@]}"; do
    echo "Starting: $vid"
    conda run -n orion python -m orion.cli.run_showcase \
      --episode "${vid}_yoloworld" \
      --video "$BASE/${vid}.mp4" \
      --detector-backend yoloworld \
      --yoloworld-prompt "$PROMPT" \
      --no-overlay 2>&1 | tail -5
    echo "✅ Done: $vid"
    sleep 5
done
