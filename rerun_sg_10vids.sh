#!/bin/bash

# Regenerate scene graphs for the 10 test videos
VID_DIR="/Users/yogeshatluru/orion-research/datasets/PVSG/VidOR/mnt/lustre/jkyang/CVPR23/openpvsg/data/vidor/videos"

VIDEO_IDS=(
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

conda activate orion
cd /Users/yogeshatluru/orion-research || exit 1

for VID in "${VIDEO_IDS[@]}"
do
    VIDEO_PATH="$VID_DIR/$VID.mp4"
    echo "Regenerating scene graph for $VID..."
    python -m orion.cli.run_showcase --episode "$VID" --video "$VIDEO_PATH" --skip-phase1 --skip-memory --force-graph --no-overlay
done

echo "Done! Running evaluation..."
python scripts/eval_sgg_recall.py
