#!/bin/bash
set -e

# Test Full Pipeline: Detection -> Re-ID -> Semantic Filtering

EPISODE="test_demo"
VIDEO="data/examples/test.mp4"
RESULTS_DIR="results/${EPISODE}"

echo "======================================================================"
echo "PHASE 1 & 2: Detection + Re-ID (using pre-baked YOLO-World)"
echo "======================================================================"

# Run showcase (Phase 1+2)
# This will generate tracks.jsonl in results/test_demo/
# Note: We allow this to fail on Phase 3 (Graph) since we skip it but don't have previous results
python -m orion.cli.run_showcase \
    --episode "${EPISODE}" \
    --video "${VIDEO}" \
    --skip-graph || true  # Skip scene graph generation for now

echo ""
echo "======================================================================"
echo "PHASE 3: Semantic Filtering (FastVLM + SentenceTransformer)"
echo "======================================================================"

# Run semantic filter
# Input: results/test_demo/tracks.jsonl
# Output: results/test_demo/tracks_filtered.jsonl
python -m orion.cli.run_vlm_filter \
    --video "${VIDEO}" \
    --tracks "${RESULTS_DIR}/tracks.jsonl" \
    --out-tracks "${RESULTS_DIR}/tracks_filtered.jsonl" \
    --audit "${RESULTS_DIR}/vlm_audit.jsonl" \
    --threshold 0.25 \
    --device cuda

echo ""
echo "======================================================================"
echo "PIPELINE COMPLETE"
echo "======================================================================"
echo "Filtered tracks saved to: ${RESULTS_DIR}/tracks_filtered.jsonl"
echo "Audit log saved to: ${RESULTS_DIR}/vlm_audit.jsonl"

# Count tracks
ORIG_COUNT=$(wc -l < "${RESULTS_DIR}/tracks.jsonl")
FILT_COUNT=$(wc -l < "${RESULTS_DIR}/tracks_filtered.jsonl")
echo "Tracks: ${ORIG_COUNT} -> ${FILT_COUNT}"
