#!/bin/bash
# Batch reprocess all 20 videos with YOLO-World detection and re-cluster with V-JEPA2 embeddings

conda activate orion || { echo "Failed to activate conda"; exit 1; }

VIDEOS=(
    "0001_4164158586"
    "0003_3396832512"
    "0003_6141007489"
    "0004_11566980553"
    "0005_2505076295"
    "0006_2889117240"
    "0008_6225185844"
    "0008_8890945814"
    "0010_8610561401"
    "0018_3057666738"
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

BASE_PATH="datasets/PVSG/VidOR/mnt/lustre/jkyang/CVPR23/openpvsg/data/vidor/videos"

echo "Reprocessing all 20 videos with YOLO-World detections..."
echo "This will re-cluster memory with V-JEPA2 embeddings"
echo ""

SUCCESS=0
FAILED=0

for i in "${!VIDEOS[@]}"; do
    VID="${VIDEOS[$i]}"
    VID_NUM=$((i + 1))
    
    echo "[${VID_NUM}/20] Processing $VID... " | tr -d '\n'
    
    # Re-cluster memory with --force-memory
    python -m orion.cli.run_showcase \
        --episode "$VID" \
        --video "$BASE_PATH/$VID.mp4" \
        --skip-phase1 \
        --force-memory \
        --skip-graph \
        --no-overlay \
        2>&1 | grep -q "✓ Built" && {
        echo "✓"
        ((SUCCESS++))
    } || {
        echo "✗"
        ((FAILED++))
    }
done

echo ""
echo "================================"
echo "Batch memory re-clustering complete"
echo "  Success: $SUCCESS/20"
echo "  Failed: $FAILED/20"
echo ""
echo "Next: Rebuild scene graphs with:"
echo "  python scripts/rebuild_all_graphs.py"
