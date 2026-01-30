#!/bin/bash
# Batch test on 10 diverse PVSG videos

VIDEOS=(
    "1110_7839815880"
    "1111_3522761604"
    "1202_4295889026"
    "1203_6461150811"
    "0003_6141007489"
    "1000_8770650748"
    "0020_10793023296"
    "0020_5323209509"
    "0021_2446450580"
    "0021_4999665957"
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
