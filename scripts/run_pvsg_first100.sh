#!/bin/bash

# PVSG VidOR video directory
VID_DIR="/Users/yogeshatluru/orion-research/datasets/PVSG/VidOR/mnt/lustre/jkyang/CVPR23/openpvsg/data/vidor/videos"
# Output results directory
RESULTS_DIR="/Users/yogeshatluru/orion-research/results/pvsg_batch_1_100"
mkdir -p "$RESULTS_DIR"


# List of the first 10 video IDs from pvsg.json (train split)
VIDEO_IDS=(
"0001_4164158586"
"0003_3396832512"
"0003_6141007489"
"0005_2505076295"
"0006_2889117240"
"0008_6225185844"
"0008_8890945814"
"0018_3057666738"
"0020_10793023296"
"0020_5323209509"
)


# Ensure the script runs from the cloned code directory (separate from datasets)
cd /Users/yogeshatluru/orion-research || exit 1

for VID in "${VIDEO_IDS[@]}"
do
    VIDEO_PATH="$VID_DIR/$VID.mp4"
    EPISODE_ID="$VID"
    OUT_DIR="$RESULTS_DIR/$EPISODE_ID"
    mkdir -p "$OUT_DIR"
    echo "Processing $VIDEO_PATH ..."
    PYTHONPATH=/Users/yogeshatluru/orion-research python -m orion.cli.run_showcase --episode "$EPISODE_ID" --video "$VIDEO_PATH"
done
