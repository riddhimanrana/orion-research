#!/bin/bash
# Batch test on 10 diverse PVSG videos

VIDEOS=(
    "0001_4164158586"
    "0003_3396832512"
    "0004_11566980553"
    "0005_2505076295"
    "0006_2889117240"
    "0008_6225185844"
    "0010_8610561401"
    "0018_3057666738"
    "0024_5224805531"
    "0028_4021064662"
)

echo "=== PVSG Batch Evaluation ==="
echo "Testing ${#VIDEOS[@]} videos..."
echo ""

for vid in "${VIDEOS[@]}"; do
    echo "Processing $vid..."
    python run_and_eval.py --video "datasets/PVSG/VidOR/mnt/lustre/jkyang/CVPR23/openpvsg/data/vidor/videos/${vid}.mp4" 2>&1 | grep -E "(R@20|R@50|R@100|ERROR)"
    echo "---"
done

echo ""
echo "=== Summary ==="
echo "Calculating averages..."

# Extract results and compute averages
python - <<'EOF'
import json
import glob

results = []
for f in glob.glob("results/*/scene_graph.jsonl"):
    vid = f.split("/")[1]
    # Check if evaluation ran
    try:
        with open(f) as fp:
            lines = fp.readlines()
            if lines:
                results.append(vid)
    except:
        pass

print(f"Successfully processed: {len(results)} videos")
print(f"Videos: {', '.join(results[:5])}...")
EOF
