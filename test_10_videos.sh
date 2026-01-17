#!/bin/bash
# Test 10 PVSG videos with YOLO-World and DINOv3 backends
# Outputs: Structured results with SGG evaluation

set -e

VIDEO_DIR="datasets/PVSG/VidOR/mnt/lustre/jkyang/CVPR23/openpvsg/data/vidor/videos"
RESULTS_DIR="results"
PVSG_PROMPT="$(cat pvsg_yoloworld_prompt.txt)"

# 10 videos to test
VIDEOS=(
  "1019_3004044251"
  "1111_8115037154"
  "1160_8623061698"
  "1160_5942411333"
  "1102_12754064355"
  "0054_2612939953"
  "1052_8530515192"
  "1021_8339742435"
  "1018_6811493102"
  "0005_2505076295"
)

echo "════════════════════════════════════════════════════════════════"
echo "BATCH TEST: 10 PVSG Videos (YOLO-World + DINOv3)"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Test YOLO-World on all 10 videos
echo "PHASE 1: Running YOLO-World on 10 videos..."
echo "Starting background jobs..."
for vid in "${VIDEOS[@]}"; do
  episode="${vid}_yoloworld"
  video_path="${VIDEO_DIR}/${vid}.mp4"
  
  if [ -f "$video_path" ]; then
    echo "  [QUEUE] $episode"
    python -m orion.cli.run_showcase \
      --episode "$episode" \
      --video "$video_path" \
      --detector-backend yoloworld \
      --yoloworld-prompt "$PVSG_PROMPT" \
      --no-overlay \
      > "/tmp/${episode}.log" 2>&1 &
  else
    echo "  [SKIP] Video not found: $video_path"
  fi
done

echo "Waiting for YOLO-World runs to complete..."
wait
echo "✅ YOLO-World runs completed"
echo ""

# Test DINOv3 if weights available
echo "PHASE 2: Testing DINOv3 backend (3 videos sample)..."
DINOV3_WEIGHTS="models/dinov3-vitb16"
if [ -d "$DINOV3_WEIGHTS" ]; then
  echo "✅ DINOv3 weights found at $DINOV3_WEIGHTS"
  for vid in "${VIDEOS[@]:0:3}"; do
    episode="${vid}_dinov3"
    video_path="${VIDEO_DIR}/${vid}.mp4"
    
    if [ -f "$video_path" ]; then
      echo "  [QUEUE] $episode"
      python -m orion.cli.run_showcase \
        --episode "$episode" \
        --video "$video_path" \
        --embedding-backend dinov3 \
        --dinov3-weights "$DINOV3_WEIGHTS" \
        --no-overlay \
        > "/tmp/${episode}.log" 2>&1 &
    fi
  done
  echo "Waiting for DINOv3 runs to complete..."
  wait
  echo "✅ DINOv3 runs completed (sample)"
else
  echo "⚠️  DINOv3 weights not found at $DINOV3_WEIGHTS"
  echo "   Skipping DINOv3 tests"
fi
echo ""

# Run SGG evaluation
echo "PHASE 3: SGG Evaluation..."
python3 << 'PYEOF'
import os, json, sys
import numpy as np
from scripts.eval_sgg_recall import load_pvsg_ground_truth, evaluate_video

VIDEOS = [
  "1019_3004044251",
  "1111_8115037154",
  "1160_8623061698",
  "1160_5942411333",
  "1102_12754064355",
  "0054_2612939953",
  "1052_8530515192",
  "1021_8339742435",
  "1018_6811493102",
  "0005_2505076295"
]

gt = load_pvsg_ground_truth('datasets/PVSG/pvsg.json')

print("\n" + "="*80)
print("SGG EVALUATION RESULTS")
print("="*80 + "\n")

# YOLO-World Results
print("YOLO-WORLD BACKEND (PVSG Vocabulary)")
print("-" * 80)
print(f"{'Video ID':<30} {'Pred':>6} {'GT':>4} {'R@20':>8} {'R@50':>8} {'R@100':>8}")
print("-" * 80)

yolo_results = []
for vid in VIDEOS:
    episode = f"{vid}_yoloworld"
    sg_path = f"results/{episode}/scene_graph.jsonl"
    if os.path.exists(sg_path):
        result = evaluate_video(episode, 'results', gt)
        if 'error' not in result:
            yolo_results.append(result)
            print(f"{vid:<30} {result.get('pred_count',0):>6.0f} {result.get('gt_count',0):>4.0f} "
                  f"{result.get('R@20',0):>7.1f}% {result.get('R@50',0):>7.1f}% {result.get('R@100',0):>7.1f}%")

if yolo_results:
    r20_avg = np.mean([r['R@20'] for r in yolo_results])
    r50_avg = np.mean([r['R@50'] for r in yolo_results])
    r100_avg = np.mean([r['R@100'] for r in yolo_results])
    print("-" * 80)
    print(f"{'AVERAGE':<30} {'':<6} {'':<4} {r20_avg:>7.1f}% {r50_avg:>7.1f}% {r100_avg:>7.1f}%")
    print(f"Videos tested: {len(yolo_results)}/10")
else:
    print("No YOLO-World results found")

# DINOv3 Results (if any)
print("\n" + "="*80)
print("DINOv3 BACKEND (Sample - 3 videos)")
print("-" * 80)
print(f"{'Video ID':<30} {'Pred':>6} {'GT':>4} {'R@20':>8} {'R@50':>8} {'R@100':>8}")
print("-" * 80)

dinov3_results = []
for vid in VIDEOS[:3]:
    episode = f"{vid}_dinov3"
    sg_path = f"results/{episode}/scene_graph.jsonl"
    if os.path.exists(sg_path):
        result = evaluate_video(episode, 'results', gt)
        if 'error' not in result:
            dinov3_results.append(result)
            print(f"{vid:<30} {result.get('pred_count',0):>6.0f} {result.get('gt_count',0):>4.0f} "
                  f"{result.get('R@20',0):>7.1f}% {result.get('R@50',0):>7.1f}% {result.get('R@100',0):>7.1f}%")

if dinov3_results:
    r20_avg = np.mean([r['R@20'] for r in dinov3_results])
    r50_avg = np.mean([r['R@50'] for r in dinov3_results])
    r100_avg = np.mean([r['R@100'] for r in dinov3_results])
    print("-" * 80)
    print(f"{'AVERAGE':<30} {'':<6} {'':<4} {r20_avg:>7.1f}% {r50_avg:>7.1f}% {r100_avg:>7.1f}%")
    print(f"Videos tested: {len(dinov3_results)}/3 (sample)")
else:
    print("No DINOv3 results found (weights may not be available)")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
PYEOF

