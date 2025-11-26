# Archived Modules

This directory contains deprecated or superseded modules from previous phases of Orion development. Files here are preserved for reference and potential restoration but are not actively used in the current pipeline.

## Why Archive?

The Orion Memory Engine refocus (Phase 0–7 roadmap) introduces a new architecture centered on:
- Persistent object memory with long-range re-ID
- Temporal scene graphs for state tracking
- Event-driven reasoning for QA

Many early experiments and prototype modules are no longer aligned with this direction. Rather than delete them, we archive for:
- Historical reference
- Potential future reuse of techniques
- Easier rollback if needed

## Archived Modules (Planned)

### Candidates for Phase 0 Archive

**Tracking Prototypes:**
- `orion/perception/tapnet_tracker.py` – TAP-Net point tracking integration (experimental; may revisit in Phase 8)
- `orion/perception/enhanced_tracker.py` – StrongSORT-inspired tracker (will be superseded by Memory Engine in Phase 3)

**Old Scripts (duplicates):**
- Scripts that duplicate new CLI functionality or are one-off experiments
- Consider: `scripts/check_dino.py` (superseded by backend tests)
- Consider: `scripts/phase2_visualize_reid.py` (legacy Phase 2 naming; will align with new phase numbering)

**Graph/Memory Prototypes:**
- `orion/graph/memgraph_backend.py` – MemGraph integration (optional; may keep if Neo4j alternative needed)
- `orion/graph/spatial_memory.py` – Early spatial memory prototype (will be replaced by Memory Engine)

### Not Archived (Keep Active)

**Core Perception:**
- `orion/perception/engine.py` – Main pipeline orchestrator
- `orion/perception/observer.py` – YOLO detection wrapper
- `orion/perception/embedder.py` – CLIP/DINO embedding layer
- `orion/perception/tracker.py` – Entity clustering (Phase 1)
- `orion/perception/types.py` – Data types and contracts
- `orion/perception/config.py` – Configuration management

**Backends:**
- `orion/backends/clip_backend.py`
- `orion/backends/dino_backend.py`
- `orion/backends/mlx_fastvlm.py`
- `orion/backends/torch_fastvlm.py`

**Managers:**
- `orion/managers/model_manager.py` – Model loading and device management
- `orion/managers/runtime.py` – Backend selection (MLX/Torch)

**CLI:**
- `orion/cli/commands/unified_pipeline.py` – Main pipeline CLI
- `orion/cli/commands/qa.py` – QA interface
- `orion/cli/commands/init.py` – System initialization

**Tests and Scripts:**
- `tests/test_phase1_*.py` – Core perception tests
- `scripts/test_full_pipeline.py` – End-to-end integration test
- `scripts/eval_perception_run.py` – Evaluation harness

## Archive Process

When moving a file to `_archive/`:

1. **Document reason:** Add entry to this README with date and rationale
2. **Update imports:** Check for any remaining imports and update/remove
3. **Preserve tests:** If tests exist, note their location and deprecate
4. **Tag commit:** Use `git tag archive-<module>-<date>` for easy restoration

## Restoration

To restore an archived module:

```bash
# Example: restore tapnet_tracker
cp orion/_archive/tapnet_tracker.py orion/perception/tapnet_tracker.py

# Update imports and dependencies
# Run tests to verify integration
```

## Archive Log

| Date | Module | Reason | Restored? |
|------|--------|--------|-----------|
| 2025-11-16 | (pending) | Phase 0 cleanup | - |

*Archive entries will be added as Phase 0 cleanup is executed.*

## Notes

- Files in `_archive/` are not included in package builds
- No active tests should depend on archived modules
- Deprecation warnings should guide users away from archived APIs
- Archive is version-controlled; old commits retain full history
