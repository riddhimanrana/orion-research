# ORION STARTUP GUIDE

Simple setup for running Orion on Apple Silicon (M1/M2/M3) or Lambda AI GPUs.

---

## Quick Start (Local - Mac)

### 1. Clone & Install
```bash
git clone https://github.com/riddhimanrana/orion-research.git
cd orion-research
pip install -e .
```

### 2. Validate Setup

```bash
python scripts/validate_setup.py
# Should see: ✅ ALL CHECKS PASSED
```

### 2b. (NEW) Validate Phase 4 Improvements

```bash
# Test improvements made to remote filtering, spatial/temporal queries
python scripts/test_improvements.py
# Should see: ✅ ALL TESTS PASSED
```

### 3. Run Test

### 3. Run Test
```bash
python -m orion.cli.run_showcase --episode test --video data/examples/test.mp4
# Results in: results/test/
```

**Done!** Your Mac is ready (expect 1-3 fps on M1).

---

## Lambda AI GPU Setup (20-30x Faster)

### Why Lambda?
- **20-30 fps** vs 1-3 fps on M1
- **$1.29/hr** for A100 GPU
- Your **$400 credits** = ~300 hours

### Step 1: Get Lambda Account
1. Sign up: https://cloud.lambdalabs.com
2. Add your $400 credits
3. **SSH Keys**: Settings → SSH Keys → Add your `~/.ssh/id_rsa.pub`

### Step 2: Launch Instance
1. **Dashboard → Launch Instance**
2. Select:
   - **GPU**: `1x A100 (40GB)` - $1.29/hr
   - **Region**: `us-west-1` (closest to you)
   - **SSH Key**: Your key
3. Click **Launch** (takes 1-2 min)
4. **Copy IP address** (e.g., `165.232.xxx.xxx`)

### Step 3: Connect via SSH
```bash
# Add to ~/.ssh/config
cat >> ~/.ssh/config << 'EOF'
Host lambda
    HostName <PASTE_IP_HERE>
    User ubuntu
    IdentityFile ~/.ssh/id_rsa
EOF

# Connect
ssh lambda
```

### Step 4: Install Orion on Lambda
```bash
# Clone into instance storage (fast, simple)
# /home/ubuntu is instance ephemeral storage - loses data when instance stops
# But that's fine for code - pull latest from git each session

git clone https://github.com/riddhimanrana/orion-research.git
cd orion-research

# Install (takes ~5 min)
pip install -e .

# Validate (should see ✅ CUDA available)
python scripts/validate_setup.py
```

**Storage Guide:**
- **Code**: `/home/ubuntu/orion-research` (instance storage, fast)
  - Lost when instance stops, but you can `git clone` again (2 min)
- **Large datasets**: Use NFS mount if available (persistent across restarts)
  - Ask Lambda support about mounting NFS if doing lots of batch processing
  - For now, just use instance storage - simpler

### Step 5: Upload Video & Run
```bash
# From Mac: upload test video
scp data/examples/test.mp4 lambda:~/orion-research/data/examples/

# On Lambda: run pipeline (gets 20-30 fps!)
cd ~/orion-research
python -m orion.cli.run_showcase --episode gpu_test --video data/examples/test.mp4

# Download results back to Mac
# (from Mac terminal)
scp -r lambda:~/orion-research/results/gpu_test ./results/
```

**You're done!** Lambda processes at 20-30 fps vs 1-3 fps on M1.

---

## VSCode Remote SSH (Best Way to Work)

Code on your Mac, execute on Lambda GPU automatically:

### 1. Install Extension
- Open VSCode → Extensions → Install **"Remote - SSH"**

### 2. Connect to Lambda
1. Press `Cmd+Shift+P`
2. Type: **"Remote-SSH: Connect to Host"**
3. Select `lambda` (from your SSH config)
4. VSCode opens new window - **you're now on Lambda!**

### 3. Open Orion Folder
```
File → Open Folder → /home/ubuntu/orion-research
```

### 4. Code Normally
- Edit files (saves on Lambda)
- Terminal runs on Lambda GPU
- GitHub Copilot works
- Python IntelliSense sees Lambda packages

**Install Python extension** when VSCode prompts (needed for IntelliSense).

---

## Storage on Lambda (Simple Answer)

**TL;DR**: Just clone into `/home/ubuntu/orion-research` - it's simple and works.

### Why?
- **Fast**: Instance storage is NVMe SSD (thousands MB/s)
- **Simple**: No setup needed, just `git clone`
- **Ephemeral**: When instance stops, code is gone (but that's OK - takes 2 min to re-clone)

### If You're Doing Batch Processing (Optional)
If you're processing tons of videos and want to avoid re-cloning:
1. **Keep code on instance**: `/home/ubuntu/orion-research`
2. **Store results on NFS** (persistent): Ask Lambda to mount filesystem
3. **Workflow**:
   ```bash
   # Instance storage (fast, code)
   /home/ubuntu/orion-research
   
   # NFS storage (persistent, data)
   /mnt/nfs/videos/        # Input videos
   /mnt/nfs/results/       # Output results
   ```
4. Results survive instance restart, but code you re-clone (it's in git anyway)

### For Your First Run
```bash
# Just do this - don't overthink storage
cd /home/ubuntu
git clone https://github.com/riddhimanrana/orion-research.git
cd orion-research
pip install -e .
python -m orion.cli.run_showcase --episode test --video data/examples/test.mp4
```

**Done!** Results in `/home/ubuntu/orion-research/results/test/`

---

## Daily Workflow with Lambda

This is how you'll actually work with Orion on Lambda - it's simple!

### Setup (One-Time)
```bash
# On Mac: commit your code and push to GitHub
git push

# On Lambda: clone once, then always pull latest
ssh lambda
cd /home/ubuntu
git clone https://github.com/riddhimanrana/orion-research.git
cd orion-research
pip install -e .
```

### Workflow Loop (Every Day)

**1. Edit Code Locally (Mac)**
```bash
# Edit files in VSCode or Remote SSH
# Make your changes (improve tracking, add features, etc.)

vim orion/perception/engine.py
python -c "from orion.perception.engine import PerceptionEngine; print('OK')"
```

**2. Push to GitHub**
```bash
git add .
git commit -m "improve tracking confidence thresholds"
git push
```

**3. Pull on Lambda & Test**
```bash
# SSH into Lambda
ssh lambda
cd ~/orion-research

# Pull latest code
git pull

# Quick test on small video (1-2 min)
python -m orion.cli.run_showcase --episode quick_test --video data/examples/test.mp4 --max-frames 100
```

**4. If Good, Run Full Evaluation**
```bash
# Full pipeline on actual data (10-30 min)
python -m orion.cli.run_quality_sweep --episode eval_v1 --video data/videos/sample.mp4

# Or batch process multiple videos
for video in data/videos/*.mp4; do
    python -m orion.cli.run_showcase --episode $(basename $video .mp4) --video $video
done
```

**5. Download Results to Mac**
```bash
# From Mac terminal (NOT while SSH'd into Lambda)
rsync -avz lambda:~/orion-research/results/ ./results-lambda/

# Or specific result
scp -r lambda:~/orion-research/results/eval_v1 ./results/
```

**6. Analyze Results Locally**
```bash
# On Mac: view metrics, visualize
python scripts/analyze_scene_graph.py results/eval_v1/
python scripts/eval_perception_run.py results/eval_v1/

# Or in Jupyter
jupyter lab
# Open notebooks/reid_dinov3_evaluation.ipynb
```

**7. Stop Instance to Save Money**
```bash
# Go to Lambda Dashboard → Click "Stop" on your instance
# Costs $0 while stopped, but /home/ubuntu data disappears
# Code is safe in GitHub, just re-clone next time

# Restart anytime with same IP
```

---

## Common Scenarios

### Scenario 1: Quick Local Test (Free, 1 min)
```bash
# On Mac
python -m orion.cli.run_showcase --episode local_test --video data/examples/test.mp4 --max-frames 30

# Good? Push to GitHub
git push
```

### Scenario 2: Full GPU Pipeline (5 min, $0.10)
```bash
# On Lambda
git pull
python -m orion.cli.run_showcase --episode gpu_full --video data/videos/sample.mp4

# Download results
# (on Mac): scp -r lambda:~/orion-research/results/gpu_full ./results/
```

### Scenario 3: Batch Processing (30 min - 2 hours, $0.50-2.50)
```bash
# On Lambda
git pull

# Process all videos at once (GPU processes in parallel!)
for video in data/videos/*.mp4; do
    echo "Processing: $video"
    python -m orion.cli.run_showcase \
        --episode $(basename $video .mp4) \
        --video $video \
        --detection-model yolo11m
done

# Then download all results
# (on Mac): rsync -avz lambda:~/orion-research/results/ ./results-batch/
```

### Scenario 4: Run Evaluations (Compare configs)
```bash
# Test different settings
for threshold in 0.7 0.75 0.8 0.85; do
    python -m orion.cli.run_showcase \
        --episode reid_threshold_$threshold \
        --video data/videos/test.mp4 \
        --reid-threshold $threshold
done

# Download and compare locally
# (on Mac): rsync -avz lambda:~/orion-research/results/ ./results-eval/
python scripts/eval_reid.py results-eval/reid_threshold_*/
```

### Scenario 5: Code + Batch in Parallel
```bash
# On Lambda terminal 1: Start long batch job (don't wait)
for video in data/videos/batch1/*.mp4; do
    python -m orion.cli.run_showcase --episode $(basename $video .mp4) --video $video &
done
# (Runs in background with &)

# On Mac (or Lambda terminal 2): Edit code via VSCode Remote SSH
# Changes auto-save on Lambda
# When batch finishes, test new code on GPU
git pull
python -m orion.cli.run_showcase --episode final_test --video test.mp4
```

---

## Cost Guide

**Typical costs:**
- Quick test (30 frames): **$0.02** (1 min)
- Single video: **$0.05-0.10** (2-3 min)
- 10 video batch: **$0.50-0.65** (30 min)
- 1 hour continuous: **$1.29**

**Your $400 budget:**
- ~**300 hours** of GPU time = ~**6000 videos processed**
- Or: Mix development (Mac free) + GPU runs ($0.05-0.10 per video)

**Money-saving tips:**
1. **STOP instance when done** (not terminate)
   - Becomes free while stopped
   - Code is safe in GitHub
2. **Test locally first** (Mac is free!)
   - Edit + quick 30-frame test = free + fast feedback
   - Use Lambda GPU only for full/batch runs
3. **Batch process** - keep GPU busy
   - 10 videos = 30 min (not 30 min each!)
   - Schedule overnight runs

---

---

## Commands Reference

### Check GPU (Lambda only)
```bash
nvidia-smi          # GPU status
nvidia-smi -l 1     # Real-time updates
```

### Process Multiple Videos
```bash
# Batch processing
for video in data/videos/*.mp4; do
    python -m orion.cli.run_showcase --episode $(basename $video .mp4) --video $video
done
```

### Optional: Memgraph (Scene Graphs)
```bash
# Install Docker + Memgraph
docker run -d -p 7687:7687 memgraph/memgraph
pip install orion[memgraph]

# Run with graph export
python -m orion.cli.run_showcase --episode test --video data/examples/test.mp4 --export-memgraph
```

---

## Troubleshooting

### "CUDA out of memory"
- Use smaller model: add `--detection-model yolo11s` to command
- Reduce resolution

### "SSH connection refused"
- Check Lambda instance is **running** (Dashboard)
- IP may change on restart - update `~/.ssh/config`

### "Models downloading slowly"
- First run downloads 2-3GB (YOLO, CLIP models)
- Faster on Lambda (better internet)

---

## Cost Guide

**Lambda A100 40GB**: $1.29/hour
- Process 1hr video in ~2-3 min = **$0.05-0.10 per video**
- Your **$400** = ~6000 videos or ~300 hours

**Recommended**: Develop on Mac (free), run batches on Lambda (fast).

---

## That's It!

✅ Local Mac setup: `pip install -e . && python scripts/validate_setup.py`  
✅ Lambda GPU: Launch instance → SSH → Install → Run  
✅ VSCode Remote: Best for daily coding

**Questions?** Check `docs/` or GitHub issues.
