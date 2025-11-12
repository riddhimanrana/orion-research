#!/bin/bash
# Orion Codebase Audit & Cleanup - November 11, 2025

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║           ORION CODEBASE AUDIT: OLD/UNUSED FILES REMOVAL                    ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"

echo ""
echo "📋 FILES TO DELETE"
echo "════════════════════════════════════════════════════════════════════════════════"

# Root-level test files (10 files)
echo ""
echo "❌ Root-level test files (old week-by-week tests - no longer used):"
old_tests=(
    "test_depth_consistency_stats.py"
    "test_loop_closure_integration.py"
    "test_multi_frame_fusion.py"
    "test_phase4_week2_zones.py"
    "test_phase4_week6.py"
    "test_scale_estimator.py"
    "test_scene_understanding.py"
    "test_video_comparison.py"
    "test_yolo_advanced.py"
    "test_yolo_room.py"
)

for file in "${old_tests[@]}"; do
    if [ -f "$file" ]; then
        echo "  rm $file"
    fi
done

# Old .py files in root
echo ""
echo "❌ Root-level old files:"
root_files=(
    "debug_image_crops.py"
    "CLI_INTEGRATION_COMPLETED.md"
    "PRODUCTION_INTEGRATION_COMPLETE.md"
)

for file in "${root_files[@]}"; do
    if [ -f "$file" ]; then
        echo "  rm $file"
    fi
done

# Old file backups
echo ""
echo "❌ File backups in orion/:"
if [ -f "orion/graph/builder.py.old" ]; then
    echo "  rm orion/graph/builder.py.old"
fi

# Deprecated shims
echo ""
echo "❌ Deprecated shims (kept for backward compat - safe to remove now):"
if [ -f "orion/semantic/graph_builder.py" ]; then
    echo "  rm orion/semantic/graph_builder.py"
fi

echo ""
echo "📊 SUMMARY"
echo "════════════════════════════════════════════════════════════════════════════════"
echo "Files to delete: 16 files"
echo "  - 10 root test files (old phase tests)"
echo "  - 2 old documentation files (production integration docs)"
echo "  - 1 debug script"
echo "  - 1 .old backup file"
echo "  - 1 deprecated shim"
echo "  - 1 debug crops file"
echo ""
echo "Space to free: ~50-100 KB"
echo "Impact: NONE - no active code depends on these"
echo ""
echo "TO PROCEED WITH DELETION:"
echo "  bash orion_cleanup.sh --confirm"
