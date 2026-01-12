#!/bin/bash

# PVSG VidOR video directory
VID_DIR="/Users/yogeshatluru/orion-research/datasets/PVSG/VidOR/mnt/lustre/jkyang/CVPR23/openpvsg/data/vidor/videos"
# Output results directory (not directly used by run_showcase but kept for grouping)
RESULTS_DIR="/Users/yogeshatluru/orion-research/results/pvsg_batch_11_20"
mkdir -p "$RESULTS_DIR"

# Next 10 video IDs from pvsg.json (train split, following the first 10)
VIDEO_IDS=(
"0021_2446450580"
"0021_4999665957"
"0024_5224805531"
"0026_2764832695"
"0028_3085751774"
"0029_5139813648"
"0029_5290336869"
"0034_2445168413"
"0036_5322122291"
"0040_3400840679"
)

# Activate conda env and run from repo root
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate orion
fi

cd /Users/yogeshatluru/orion-research || exit 1

for VID in "${VIDEO_IDS[@]}"
do
    VIDEO_PATH="$VID_DIR/$VID.mp4"
    EPISODE_ID="$VID"
    echo "Processing $VIDEO_PATH ..."
    PYTHONPATH=/Users/yogeshatluru/orion-research python -m orion.cli.run_showcase --episode "$EPISODE_ID" --video "$VIDEO_PATH"
done
