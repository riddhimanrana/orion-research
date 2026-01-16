#!/bin/bash
# Reprocess 10 PVSG test videos with YOLO-World using PVSG vocabulary

# Load PVSG YOLO-World prompt
PROMPT=$(tr '\n' ' ' < pvsg_yoloworld_prompt.txt)

# Remove newlines and clean up spacing
PROMPT=$(echo "$PROMPT" | sed 's/[[:space:]]\+/ /g' | tr -s ' ')

echo "PVSG YOLO-World Prompt:"
echo "$PROMPT"
echo ""

VIDEOS=(
    "0020_10793023296"
    "0020_5323209509"
    "0021_2446450580"
    "0021_4999665957"
    "0024_5224805531"
    "0026_2764832695"
    "0027_4571353789"
    "0028_3085751774"
    "0028_4021064662"
    "0029_5139813648"
)

for vid in "${VIDEOS[@]}"; do
    video_path="data/examples/episodes/${vid}/video.mp4"
    
    if [ ! -f "$video_path" ]; then
        echo "❌ Video not found: $video_path"
        continue
    fi
    
    echo ""
    echo "=========================================="
    echo "Processing: $vid"
    echo "=========================================="
    
    python -m orion.cli.run_showcase \
        --episode "$vid" \
        --video "$video_path" \
        --detector-backend yoloworld \
        --yoloworld-prompt "$PROMPT" \
        --force-phase1 \
        2>&1 | tail -30
done

echo ""
echo "✓ Reprocessing complete. Run eval_sgg_recall.py to check results."
