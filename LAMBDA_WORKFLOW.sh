#!/usr/bin/env bash
# Lambda AI Workflow - Quick Reference
# Copy this to your Mac for quick command reference

# ============================================================================
# DAILY WORKFLOW
# ============================================================================

# 1. DEVELOP LOCALLY (Mac) - FREE
# ============================================================================
cd ~/Desktop/Coding/Orion/orion-research
# Edit code in VSCode
vim orion/perception/engine.py
git add .
git commit -m "improve tracking"
git push


# 2. QUICK TEST LOCALLY (Mac) - FREE (~30 sec)
# ============================================================================
python -m orion.cli.run_showcase --episode local_test --video data/examples/test.mp4 --max-frames 30
# If looks good → proceed to GPU


# 3. PULL & TEST ON GPU (Lambda) - $0.02 (~1 min)
# ============================================================================
ssh lambda
cd ~/orion-research  # or: cd /lambda/nfs/orion-core-fs/orion-research
git pull
python -m orion.cli.run_showcase --episode gpu_quick --video data/examples/test.mp4 --max-frames 100


# 4. RUN FULL PIPELINE (Lambda) - $0.10 (~3 min per video)
# ============================================================================
# Single video
python -m orion.cli.run_showcase --episode full_run --video data/videos/sample.mp4

# Batch: Multiple videos in parallel
for video in data/videos/*.mp4; do
    python -m orion.cli.run_showcase --episode $(basename $video .mp4) --video $video &
done
wait
echo "All done!"


# 4b. COMPARE DETECTOR BACKENDS (Lambda) - apples-to-apples hallucination check
# ============================================================================
# Run the same video with YOLO-only, GDINO-only, and HYBRID, keeping FPS and thresholds identical.
# Output goes to results/<episode>/tracks.jsonl.
# NOTE: Do NOT edit code on Lambda; only git pull.

# YOLO baseline
python -m orion.cli.run_tracks \
    --episode cmp_yolo \
    --video data/videos/test.mp4 \
    --fps 5 \
    --model yolo11x \
    --detector-backend yolo \
    --conf-threshold 0.25

# GroundingDINO baseline
python -m orion.cli.run_tracks \
    --episode cmp_gdino \
    --video data/videos/test.mp4 \
    --fps 5 \
    --model yolo11x \
    --detector-backend groundingdino \
    --gdino-model IDEA-Research/grounding-dino-tiny \
    --conf-threshold 0.25

# Hybrid (YOLO primary + conditional GDINO)
python -m orion.cli.run_tracks \
    --episode cmp_hybrid \
    --video data/videos/test.mp4 \
    --fps 5 \
    --model yolo11x \
    --detector-backend hybrid \
    --gdino-model IDEA-Research/grounding-dino-tiny \
    --hybrid-min-detections 3 \
    --hybrid-secondary-conf 0.30 \
    --conf-threshold 0.25


# 5. RUN EVALUATIONS (Lambda) - $0.50-2.00
# ============================================================================
# Sweep different configs
for threshold in 0.7 0.75 0.8 0.85; do
    python -m orion.cli.run_showcase \
        --episode reid_$threshold \
        --video data/videos/test.mp4 \
        --reid-threshold $threshold
done

# Or full quality sweep
python -m orion.cli.run_quality_sweep --episode eval_all --video data/videos/test.mp4


# 6. CHECK GPU STATUS (Lambda)
# ============================================================================
ssh lambda
nvidia-smi          # One-time snapshot
nvidia-smi -l 1     # Real-time (refresh every sec)


# 7. DOWNLOAD RESULTS (Mac)
# ============================================================================
# All results
rsync -avz lambda:~/orion-research/results/ ./results-gpu/

# Specific result
scp -r lambda:~/orion-research/results/eval_all ./results/

# Watch live progress
ssh lambda "tail -f ~/orion-research/results/full_run/perception_run.json"


# 8. ANALYZE LOCALLY (Mac) - FREE
# ============================================================================
# View metrics
python scripts/analyze_scene_graph.py results-gpu/eval_all/
python scripts/eval_perception_run.py results-gpu/eval_all/

# Jupyter analysis
jupyter lab
# Open: notebooks/reid_dinov3_evaluation.ipynb


# 9. STOP INSTANCE (Save Money!) - $0/hr
# ============================================================================
# Go to Lambda Dashboard → Click "Stop"
# Code is safe in GitHub
# Restart anytime
# (Don't forget! Running instance costs $1.29/hr)


# ============================================================================
# COST CHEAT SHEET
# ============================================================================

# $0.02  = 1 min     = quick test on GPU
# $0.10  = 5 min     = single video
# $0.50  = 30 min    = 10 video batch
# $1.29  = 1 hour    = full eval run
# 
# Your $400 = ~300 hours = ~6000 videos
# 
# RULE: If you forgot to STOP, you're burning $1.29/hr
# CHECK: https://cloud.lambdalabs.com/instances


# ============================================================================
# WORKFLOW DECISION TREE
# ============================================================================

# Want to edit code?
#   → Edit on Mac (free, slow M1)
#   → Push to GitHub
#   → Pull on Lambda to test (costs money only if running)

# Want to test something?
#   → Test locally first (free, 30 sec on M1)
#   → If looks good → SSH to Lambda → git pull → run on GPU

# Want to process many videos?
#   → SSH Lambda → batch loop with & (parallel)
#   → Let it run overnight or while you work
#   → Download results in morning

# Want to compare configs?
#   → Loop over thresholds/models on Lambda
#   → Download all results at once
#   → Analyze on Mac (free)

# Done working?
#   → STOP instance in Lambda Dashboard
#   → Go to zero cost
#   → Code is in GitHub (safe)


# ============================================================================
# TYPICAL SESSION (with costs)
# ============================================================================

# Session A: Quick iteration (Total: FREE + $0.20)
# 1. Edit code on Mac                          FREE
# 2. Test locally 30 frames                    FREE (~30 sec)
# 3. Push to GitHub                            FREE
# 4. Start Lambda instance                     FREE (first minute)
# 5. git pull + quick GPU test                 $0.02 (~1 min)
# 6. Run full pipeline on GPU                  $0.10 (~3 min)
# 7. Download results                          FREE
# 8. Analyze on Mac                            FREE
# 9. Stop instance                             FREE
# Total: $0.12 + 10 min work


# Session B: Batch processing (Total: $1.50)
# 1. Start instance                            FREE
# 2. git pull                                  FREE
# 3. Run batch: 15 videos × 3min = 45 min      $0.97
# 4. Let it run (avoid code edits on Lambda; make changes locally and git push)
# 5. Download all results                      FREE
# 6. Analyze on Mac                            FREE
# 7. Stop instance                             FREE
# Total: $0.97 + 15 videos processed


# Session C: Long evaluation (Total: $3.87)
# 1. Start instance                            FREE
# 2. Run quality_sweep (4 configs)             $1.29 × 3 = $3.87 (~3 hours)
# 3. Download results                          FREE
# 4. Compare locally (Jupyter)                 FREE
# 5. Stop instance                             FREE
# Total: $3.87 + comprehensive evaluation


echo "Lambda workflow reference loaded! 🚀"
echo "Full guide: cat STARTUP.md"
