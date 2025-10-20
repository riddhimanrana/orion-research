"""
Orion Video Analysis Pipeline
==============================

Complete video understanding pipeline with:
- Visual perception & object detection
- Semantic knowledge graph construction  
- Interactive Q&A with local LLM

Author: Orion Research Team
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

# Use asset manager for model assets, runtime manager for LLM-enabled operations
from .models import AssetManager
from .model_manager import ModelManager as RuntimeModelManager
from .neo4j_manager import clear_neo4j_for_new_run
from .tracking_engine import run_tracking_engine
from .semantic_uplift import run_semantic_uplift
from .contextual_engine import apply_contextual_understanding

try:
    from .runtime import get_active_backend, select_backend, set_active_backend
except ImportError:  # pragma: no cover
    from .runtime import get_active_backend, select_backend, set_active_backend

VideoQASystem: Any

try:
    from .video_qa import VideoQASystem as _VideoQASystem

    VideoQASystem = _VideoQASystem
    QA_AVAILABLE = True
except ImportError:  # pragma: no cover
    VideoQASystem = None
    QA_AVAILABLE = False

# Configurations (Now in unified config.py)
apply_part1_config: Any
apply_part2_config: Any

try:
    from .config import (
        SEMANTIC_FAST_CONFIG, 
        SEMANTIC_BALANCED_CONFIG, 
        SEMANTIC_ACCURATE_CONFIG
    )
    
    # Helper functions for compatibility
    def apply_part1_config(cfg: dict) -> None:
        """Apply perception config (deprecated - configs now in config.py)"""
        pass
    
    def apply_part2_config(cfg: dict) -> None:
        """Apply semantic config (deprecated - configs now in config.py)"""
        pass

    CONFIGS_AVAILABLE = True
except ImportError:  # pragma: no cover
    apply_part1_config = None
    apply_part2_config = None
    CONFIGS_AVAILABLE = False

console = Console()


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    minutes, sec = divmod(int(seconds + 0.5), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


@dataclass
class StageState:
    title: str
    style: str = "cyan"
    total: Optional[int] = None
    current: int = 0
    status: str = "Pending"
    icon: str = "[dim]○[/]"
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    def start(self, message: Optional[str] = None) -> None:
        if self.start_time is None:
            self.start_time = time.time()
        self.status = message or self.status or "Running"
        self.icon = f"[{self.style}]⏳[/]"

    def set_total(self, total: Optional[int]) -> None:
        self.total = total
        if total is not None:
            self.current = min(self.current, total)

    def set_status(self, message: str) -> None:
        self.status = message

    def advance(self, *, current: Optional[int] = None, increment: Optional[int] = None) -> None:
        if current is not None:
            self.current = current
        elif increment is not None:
            self.current += increment
        elif self.total is not None:
            self.current += 1
        if self.total is not None:
            self.current = max(0, min(self.current, self.total))

    def complete(self, message: Optional[str] = None) -> None:
        if self.total not in (None, 0):
            self.current = self.total
        self.status = message or self.status or "Completed"
        self.icon = "[green]✓[/]"
        if self.end_time is None:
            self.end_time = time.time()

    def warn(self, message: str) -> None:
        self.status = message
        self.icon = "[yellow]![/]"
        if self.end_time is None:
            self.end_time = time.time()

    def fail(self, message: str) -> None:
        self.status = message
        self.icon = "[red]✗[/]"
        if self.end_time is None:
            self.end_time = time.time()

    def skip(self, message: str) -> None:
        if self.start_time is None:
            self.start_time = time.time()
        self.end_time = time.time()
        self.status = message
        self.icon = "[yellow]⏭[/]"

    def progress_ratio(self) -> Optional[float]:
        if self.total in (None, 0):
            return None
        return min(1.0, self.current / self.total if self.total else 0.0)

    def progress_display(self) -> str:
        ratio = self.progress_ratio()
        if ratio is None:
            return "—"
        percent = ratio * 100.0
        if self.total is None:
            return f"{percent:4.0f}%"
        return f"{self.current}/{self.total} ({percent:4.0f}%)"

    def elapsed_display(self) -> str:
        if self.start_time is None:
            return "—"
        end = self.end_time or time.time()
        return _format_duration(end - self.start_time)

    def eta_display(self) -> str:
        ratio = self.progress_ratio()
        if (
            ratio is None
            or ratio <= 0.0
            or ratio >= 1.0
            or self.start_time is None
            or self.end_time is not None
        ):
            return "—"
        elapsed = time.time() - self.start_time
        remaining = (1.0 - ratio) * elapsed / max(ratio, 1e-9)
        return _format_duration(remaining)


class PipelineUI:
    """Live-updating terminal dashboard for pipeline stages."""

    def __init__(self, console: Console, enabled: bool = True) -> None:
        self.console = console
        self.enabled = enabled
        self.stages: "OrderedDict[str, StageState]" = OrderedDict()
        self.live: Optional[Live] = None
        self._last_refresh = 0.0
        self._refresh_interval = 0.1

    def __enter__(self) -> "PipelineUI":
        if self.enabled:
            self.live = Live(
                self.render(),
                console=self.console,
                refresh_per_second=8,
                transient=False,
            )
            self.live.__enter__()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[Any],
    ) -> None:
        if self.enabled and self.live is not None:
            self.refresh(force=True)
            self.live.__exit__(exc_type, exc, tb)
            self.live = None

    # ------------------------------------------------------------------
    # Stage management helpers
    # ------------------------------------------------------------------
    def add_stage(
        self,
        key: str,
        title: str,
        *,
        style: str = "cyan",
        total: Optional[int] = None,
    ) -> None:
        if key not in self.stages:
            self.stages[key] = StageState(title=title, style=style, total=total)
            self.refresh(force=True)

    def start_stage(self, key: str, message: Optional[str] = None) -> None:
        stage = self.stages.get(key)
        if stage is None:
            return
        stage.start(message or "Running")
        self.refresh()

    def set_stage_total(self, key: str, total: Optional[int]) -> None:
        stage = self.stages.get(key)
        if stage is None:
            return
        stage.set_total(total)
        self.refresh()

    def advance_stage(
        self,
        key: str,
        *,
        current: Optional[int] = None,
        increment: Optional[int] = None,
        message: Optional[str] = None,
    ) -> None:
        stage = self.stages.get(key)
        if stage is None:
            return
        if stage.start_time is None:
            stage.start()
        stage.advance(current=current, increment=increment)
        if message is not None:
            stage.set_status(message)
        self.refresh()

    def complete_stage(self, key: str, message: Optional[str] = None) -> None:
        stage = self.stages.get(key)
        if stage is None:
            return
        stage.complete(message)
        self.refresh(force=True)

    def fail_stage(self, key: str, message: str) -> None:
        stage = self.stages.get(key)
        if stage is None:
            return
        stage.fail(message)
        self.refresh(force=True)

    def warn_stage(self, key: str, message: str) -> None:
        stage = self.stages.get(key)
        if stage is None:
            return
        stage.warn(message)
        self.refresh(force=True)

    def skip_stage(self, key: str, message: str) -> None:
        stage = self.stages.get(key)
        if stage is None:
            return
        stage.skip(message)
        self.refresh(force=True)

    def set_stage_status(self, key: str, message: str) -> None:
        stage = self.stages.get(key)
        if stage is None:
            return
        stage.set_status(message)
        self.refresh()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self) -> Panel:
        table = Table(box=box.SIMPLE_HEAVY, expand=True, show_edge=False)
        table.add_column("Stage", style="bold", no_wrap=True)
        table.add_column("Status", overflow="fold")
        table.add_column("Progress", justify="right", width=18)
        table.add_column("Elapsed", justify="right", width=8)
        table.add_column("ETA", justify="right", width=8)

        for stage in self.stages.values():
            table.add_row(
                f"{stage.icon} {stage.title}",
                stage.status,
                stage.progress_display(),
                stage.elapsed_display(),
                stage.eta_display(),
            )

        return Panel(
            table,
            title="[bold]Pipeline Status[/bold]",
            border_style="cyan",
            padding=(0, 1),
        )

    def refresh(self, force: bool = False) -> None:
        if not self.enabled or self.live is None:
            return
        now = time.time()
        if not force and now - self._last_refresh < self._refresh_interval:
            return
        self._last_refresh = now
        self.live.update(self.render(), refresh=True)


def setup_logging(verbose: bool = False):
    """Setup clean logging"""
    level = logging.DEBUG if verbose else logging.WARNING

    # Configure logging (console only, no file logging)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # Silence noisy libraries
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("coremltools").setLevel(logging.WARNING)


def _refresh_engine_loggers() -> None:
    """Re-evaluate engine loggers after toggling suppression flags."""

    try:
        from . import semantic_uplift

        semantic_uplift.logger = semantic_uplift.setup_logger("SemanticUplift")
    except Exception:  # pragma: no cover - defensive
        pass


def print_banner():
    """Print startup banner"""
    banner = """
[bold cyan]╔═══════════════════════════════════════════════════════════════╗
║                  ORION VIDEO ANALYSIS PIPELINE                ║
║                    From Moments to Memory                     ║
╚═══════════════════════════════════════════════════════════════╝[/bold cyan]
    """
    console.print(banner)


def _resolve_runtime(preferred: Optional[str]) -> Tuple[str, AssetManager]:
    normalized: Optional[str] = preferred.lower() if preferred else None
    if normalized == "auto":
        normalized = None
    if normalized is not None and normalized not in {"mlx", "torch"}:
        raise ValueError(f"Unsupported runtime preference '{preferred}'.")

    active_backend = get_active_backend()
    if normalized in {"mlx", "torch"}:
        backend = select_backend(normalized)
    elif active_backend is not None:
        backend = active_backend
    else:
        backend = select_backend(None)

    set_active_backend(backend)
    manager = AssetManager()
    if not manager.assets_ready(backend):
        console.print(
            f"[yellow]Preparing model assets for runtime '{backend}'...[/yellow]"
        )
        manager.ensure_runtime_assets(backend)
    return backend, manager


def _convert_tracking_results_to_perception_log(entities: List[Any], observations: List[Any]) -> List[Dict[str, Any]]:
    """
    Convert tracking engine output (entities, observations) to perception log format.

    The tracking engine returns Entity and Observation objects.
    We convert them to a list of dicts matching the perception log format.
    """
    perception_log = []

    # Create a mapping from observation to entity_id for lookup
    obs_to_entity_id = {}
    for entity in entities:
        for obs in entity.observations:
            obs_to_entity_id[id(obs)] = entity.id

    for obs in observations:
        # Get entity_id for this observation
        entity_id = obs_to_entity_id.get(id(obs), '')

        # Create perception object from observation
        perception_obj = {
            # Identity
            'entity_id': entity_id,
            'temp_id': entity_id,

            # Classification
            'object_class': obs.class_name,
            'detection_confidence': obs.confidence,

            # Description (from entity, not individual observation)
            'rich_description': '',  # Will be filled from entity later if needed

            # Temporal
            'timestamp': obs.timestamp,
            'frame_number': obs.frame_number,

            # Spatial
            'bounding_box': obs.bbox,
            'centroid': (
                (obs.bbox[0] + obs.bbox[2]) / 2, (obs.bbox[1] + obs.bbox[3]) / 2
            ) if len(obs.bbox) >= 4 else (0, 0),

            # Visual
            'visual_embedding': obs.embedding.tolist() if hasattr(obs.embedding, 'tolist') else obs.embedding,
            'crop_size': (obs.bbox[2] - obs.bbox[0], obs.bbox[3] - obs.bbox[1]) if len(obs.bbox) >= 4 else (0, 0),
        }

        perception_log.append(perception_obj)

    return perception_log


def run_pipeline(
    video_path: str,
    output_dir: str = "data/testing",
    neo4j_uri: Optional[str] = None,
    neo4j_user: Optional[str] = None,
    neo4j_password: Optional[str] = None,
    clear_db: bool = True,
    part1_config: str = "balanced",
    part2_config: str = "balanced",
    skip_part1: bool = False,
    skip_part2: bool = False,
    verbose: bool = False,
    runtime: Optional[str] = None,
    use_progress_ui: bool = True,
) -> Dict[str, Any]:
    """
    Run the complete video analysis pipeline with a streamlined terminal UI.

    Args:
        video_path: Path to input video.
        output_dir: Directory for output artifacts.
        neo4j_uri: Neo4j database URI (uses ConfigManager if not provided).
        neo4j_user: Neo4j username (uses ConfigManager if not provided).
        neo4j_password: Neo4j password (uses ConfigManager if not provided).
        clear_db: Whether to clear Neo4j before running Part 2.
        part1_config: Perception configuration preset (fast/balanced/accurate).
        part2_config: Semantic uplift configuration preset (fast/balanced/accurate).
        skip_part1: Skip perception engine (use existing results).
        skip_part2: Skip semantic uplift stage.
        verbose: Enable verbose logging.
        runtime: Preferred runtime backend.
        use_progress_ui: Render Rich progress bars instead of verbose logs.

    Returns:
        Results dictionary with statistics and file paths.
    """
    from .config_manager import ConfigManager
    
    # Use ConfigManager for defaults
    config = ConfigManager.get_config()
    neo4j_uri = neo4j_uri or config.neo4j.uri
    neo4j_user = neo4j_user or config.neo4j.user
    neo4j_password = neo4j_password or config.neo4j.password

    setup_logging(verbose)

    previous_suppress = None
    if not verbose:
        previous_suppress = os.environ.get("ORION_SUPPRESS_ENGINE_LOGS")
        os.environ["ORION_SUPPRESS_ENGINE_LOGS"] = "1"
        _refresh_engine_loggers()

    results: Dict[str, Any] = {
        "video_path": video_path,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "part1": {},
        "part2": {},
        "success": False,
        "errors": [],
    }

    perception_log: Optional[List[Dict[str, Any]]] = None
    perception_log_path: Optional[str] = None
    perception_duration: Optional[float] = None
    uplift_results: Dict[str, Any] = {}
    uplift_duration: Optional[float] = None
    neo4j_status: Optional[str] = None

    ui = PipelineUI(console, enabled=use_progress_ui and not verbose)

    try:
        with ui:
            if ui.enabled:
                ui.add_stage("runtime", "Runtime backend", style="cyan")
                ui.start_stage("runtime", "Preparing runtime backend")
            try:
                backend, _ = _resolve_runtime(runtime)
            except Exception as exc:
                if ui.enabled:
                    ui.fail_stage("runtime", f"Runtime preparation failed: {exc}")
                raise
            results["runtime"] = backend
            if ui.enabled:
                ui.complete_stage("runtime", f"Runtime ready ({backend})")

            if not skip_part2:
                if ui.enabled:
                    ui.add_stage("neo4j", "Neo4j database", style="yellow")
                    ui.start_stage("neo4j", "Preparing Neo4j database")
                if clear_db:
                    cleared = clear_neo4j_for_new_run(
                        neo4j_uri, neo4j_user, neo4j_password
                    )
                    if cleared:
                        neo4j_status = "Cleared"
                        if ui.enabled:
                            ui.complete_stage("neo4j", "Neo4j cleared")
                    else:
                        neo4j_status = "Unavailable (kept data)"
                        if ui.enabled:
                            ui.warn_stage(
                                "neo4j", "Neo4j unavailable – kept existing data"
                            )
                else:
                    neo4j_status = "Preserved"
                    if ui.enabled:
                        ui.skip_stage("neo4j", "Preserving existing Neo4j data")

            if skip_part1:
                results["part1"] = {"success": False, "skipped": True}
                if ui.enabled:
                    ui.add_stage("perception", "Perception engine", style="yellow")
                    ui.skip_stage("perception", "Perception engine skipped")
            else:
                if CONFIGS_AVAILABLE:
                    if part1_config == "fast":
                        from .config import get_fast_config
                        # Deprecated: apply_part1_config no-op
                    elif part1_config == "accurate":
                        from .config import get_accurate_config
                        # Deprecated: apply_part1_config no-op
                    else:
                        from .config import get_balanced_config
                        # Deprecated: apply_part1_config no-op

                if ui.enabled:
                    ui.add_stage(
                        "perception",
                        f"Smart Perception ({part1_config} mode)",
                        style="cyan",
                        total=3,
                    )
                    ui.start_stage(
                        "perception",
                        f"Smart tracking-based perception",
                    )
                    # Add sub-stages for smart perception
                    ui.add_stage(
                        "perception.phase1",
                        "Phase 1: Detection & CLIP Embeddings",
                        style="cyan",
                    )
                    ui.add_stage(
                        "perception.phase2",
                        "Phase 2: Entity Clustering (HDBSCAN)",
                        style="magenta",
                    )
                    ui.add_stage(
                        "perception.phase3",
                        "Phase 3: Smart Descriptions",
                        style="green",
                    )

                def handle_progress(event: str, payload: Dict[str, Any]) -> None:
                    if not ui.enabled:
                        return
                    
                    # Tracking engine events
                    if event == "tracking.start":
                        ui.set_stage_status("perception", "Starting tracking engine...")
                    
                    elif event == "tracking.phase1.start":
                        ui.start_stage("perception.phase1", "Detecting objects with YOLO11x")
                        ui.set_stage_status("perception", "Phase 1: Detection")
                    elif event == "tracking.phase1.complete":
                        observations = payload.get("observations", 0)
                        msg = payload.get("message", f"{observations} observations")
                        ui.complete_stage("perception.phase1", msg)
                        ui.advance_stage("perception", increment=1)
                        ui.set_stage_status("perception", "Phase 1 complete")
                        ui.start_stage("perception.phase2", "Clustering with HDBSCAN")
                        ui.set_stage_status("perception", "Phase 2: Clustering")
                    
                    elif event == "tracking.phase2.complete":
                        ui.advance_stage("perception", increment=1)
                        entities = payload.get("entities", 0)
                        observations = payload.get("observations", 0)
                        efficiency = payload.get("efficiency", "N/A")
                        msg = f"{observations} obs → {entities} entities ({efficiency})"
                        ui.complete_stage("perception.phase2", msg)
                        ui.set_stage_status("perception", "Phase 2 complete")
                        ui.start_stage("perception.phase3", f"Describing {entities} entities")
                        ui.set_stage_status("perception", "Phase 3: Descriptions")
                    
                    elif event == "tracking.phase3.complete":
                        entities = payload.get("entities", 0)
                        msg = payload.get("message", f"{entities} descriptions")
                        ui.complete_stage("perception.phase3", msg)
                        ui.advance_stage("perception", increment=1)
                        ui.set_stage_status("perception", "Phase 3 complete")
                    
                    elif event == "tracking.complete":
                        entities = payload.get("entities", 0)
                        observations = payload.get("observations", 0)
                        efficiency = payload.get("efficiency", "N/A")
                        ui.set_stage_status(
                            "perception",
                            f"Complete: {entities} entities from {observations} observations ({efficiency})"
                        )
                    
                    # Legacy perception events (fallback)
                    elif event == "perception.frames.start":
                        total = payload.get("total")
                        if total is not None:
                            ui.set_stage_total("perception.frames", total)
                        ui.start_stage("perception.frames", "Processing frames")
                        ui.set_stage_status("perception", "Processing frames")
                    elif event == "perception.frames.progress":
                        current = payload.get("current")
                        total = payload.get("total")
                        if total is not None:
                            ui.set_stage_total("perception.frames", total)
                        ui.advance_stage("perception.frames", current=current)
                        frame = payload.get("frame")
                        detections = payload.get("detections")
                        if frame is not None and detections is not None:
                            ui.set_stage_status(
                                "perception.frames",
                                f"Frame {frame}: +{detections} detections",
                            )
                        elif current is not None and total is not None:
                            ui.set_stage_status(
                                "perception.frames",
                                f"{current}/{total} frames processed",
                            )
                        if current is not None and total is not None:
                            ui.set_stage_status(
                                "perception",
                                f"{current}/{total} frames processed",
                            )
                    elif event == "perception.frames.complete":
                        detections = payload.get("detections")
                        total = payload.get("total")
                        if detections is not None and total is not None:
                            msg = f"{total} frames, {detections} detections"
                        elif total is not None:
                            msg = f"{total} frames processed"
                        else:
                            msg = "Frame processing complete"
                        ui.complete_stage("perception.frames", msg)
                        ui.set_stage_status("perception", "Awaiting descriptions")
                    elif event == "perception.descriptions.start":
                        total = payload.get("total")
                        if total is not None:
                            ui.set_stage_total("perception.descriptions", total)
                        ui.start_stage(
                            "perception.descriptions", "Generating descriptions"
                        )
                        ui.set_stage_status("perception", "Generating descriptions")
                    elif event == "perception.descriptions.progress":
                        current = payload.get("current")
                        total = payload.get("total")
                        if total is not None:
                            ui.set_stage_total("perception.descriptions", total)
                        ui.advance_stage(
                            "perception.descriptions", current=current
                        )
                        if current is not None and total:
                            ui.set_stage_status(
                                "perception.descriptions",
                                f"{current}/{total} complete",
                            )
                            ui.set_stage_status(
                                "perception",
                                f"{current}/{total} descriptions ready",
                            )
                        elif current is not None:
                            ui.set_stage_status(
                                "perception.descriptions",
                                f"{current} descriptions ready",
                            )
                    elif event == "perception.descriptions.complete":
                        total = payload.get("total")
                        if total is not None:
                            msg = f"{total} descriptions generated"
                        else:
                            msg = "Descriptions complete"
                        ui.complete_stage("perception.descriptions", msg)
                        ui.set_stage_status("perception", "Finalizing")

                start_time = time.time()
                try:
                    # Run tracking engine to detect, cluster, and describe entities
                    entities, observations = run_tracking_engine(
                        video_path,
                        config=None,  # Uses default config
                    )
                    
                    # Convert tracking results to perception log format
                    perception_log = _convert_tracking_results_to_perception_log(entities, observations)
                    
                except Exception as exc:
                    if ui.enabled:
                        ui.fail_stage("perception", "Tracking engine failed")
                    raise

                perception_duration = time.time() - start_time
                num_objects = len(perception_log)
                
                # Count unique entities
                unique_entities = len(set(obj.get('entity_id') for obj in perception_log if obj.get('entity_id')))

                if ui.enabled:
                    ui.complete_stage(
                        "perception",
                        f"Tracking complete: {unique_entities} entities, {num_objects} observations",
                    )

                perception_log_path = str(
                    Path(output_dir)
                    / f"perception_log_{Path(video_path).stem}_{results['timestamp']}.json"
                )
                Path(perception_log_path).parent.mkdir(parents=True, exist_ok=True)
                with open(perception_log_path, "w") as log_file:
                    json.dump(perception_log, log_file, indent=2)

                results["part1"] = {
                    "success": True,
                    "duration_seconds": perception_duration,
                    "num_objects": num_objects,
                    "num_entities": unique_entities,
                    "efficiency": f"{num_objects / max(unique_entities, 1):.1f}x",
                    "output_file": perception_log_path,
                    "mode": part1_config,
                }

            if skip_part2:
                results["part2"] = {"success": False, "skipped": True}
                if ui.enabled:
                    ui.add_stage("semantic", "Semantic uplift", style="yellow")
                    ui.skip_stage("semantic", "Semantic uplift skipped")
            else:
                if CONFIGS_AVAILABLE:
                    if part2_config == "fast":
                        from .config import SEMANTIC_FAST_CONFIG as SEM_FAST

                        apply_part2_config(SEM_FAST)
                    elif part2_config == "accurate":
                        from .config import SEMANTIC_ACCURATE_CONFIG as SEM_ACCURATE

                        apply_part2_config(SEM_ACCURATE)
                    else:
                        from .config import SEMANTIC_BALANCED_CONFIG as SEM_BALANCED

                        apply_part2_config(SEM_BALANCED)

                if ui.enabled:
                    ui.add_stage(
                        "semantic",
                        f"Semantic uplift ({part2_config} mode)",
                        style="cyan",
                    )
                    ui.start_stage(
                        "semantic",
                        f"Semantic uplift ({part2_config} mode)",
                    )

                if perception_log is None:
                    if perception_log_path is None:
                        logs = sorted(Path(output_dir).glob("perception_log_*.json"))
                        if not logs:
                            if ui.enabled:
                                ui.fail_stage(
                                    "semantic", "No perception logs available"
                                )
                            raise ValueError(
                                "No perception logs found. Run Part 1 or provide logs."
                            )
                        perception_log_path = str(logs[-1])

                    with open(perception_log_path, "r") as log_file:
                        perception_log = json.load(log_file)

                if perception_log is None:
                    if ui.enabled:
                        ui.fail_stage("semantic", "Failed to load perception log")
                    raise RuntimeError(
                        "Unable to load perception log for semantic uplift."
                    )

                # Apply contextual understanding before semantic uplift
                if ui.enabled:
                    ui.add_stage(
                        "contextual",
                        "Contextual Understanding",
                        style="magenta",
                        total=3,
                    )
                    ui.start_stage("contextual", "Analyzing spatial context & correcting classes")
                
                contextual_start = time.time()
                contextual_progress_tracker = {"heuristics": False, "llm": False}
                
                def handle_contextual_progress(event: str, payload: Dict[str, Any]) -> None:
                    message: Optional[str] = None

                    if event == "contextual.start":
                        total = payload.get("total", 0)
                        display_message = (
                            f"Processing {total} observations"
                            if total
                            else "Processing contextual corrections"
                        )
                        message = display_message
                        if ui.enabled:
                            ui.set_stage_status("contextual", display_message)
                    elif event == "contextual.heuristics":
                        display_message = payload.get("message") or "Computing spatial context"
                        message = display_message
                        if ui.enabled:
                            if not contextual_progress_tracker["heuristics"]:
                                ui.advance_stage("contextual", current=1)
                                contextual_progress_tracker["heuristics"] = True
                            ui.set_stage_status("contextual", display_message)
                    elif event == "contextual.llm":
                        total = payload.get("total", 0)
                        default_message = (
                            f"Correcting {total} classifications"
                            if total
                            else "Running LLM corrections"
                        )
                        display_message = payload.get("message") or default_message
                        message = display_message
                        if ui.enabled:
                            if not contextual_progress_tracker["llm"]:
                                ui.advance_stage("contextual", current=2)
                                contextual_progress_tracker["llm"] = True
                            ui.set_stage_status("contextual", display_message)
                    elif event == "contextual.llm.batch":
                        display_message = payload.get("message") or "Processing LLM batch"
                        message = display_message
                        if ui.enabled:
                            ui.set_stage_status("contextual", display_message)
                    elif event == "contextual.llm.cache_hit":
                        display_message = payload.get("message") or "Served contextual batch from cache"
                        message = display_message
                        if ui.enabled:
                            ui.set_stage_status("contextual", display_message)
                    elif event == "contextual.complete":
                        display_message = payload.get("message") or "Contextual analysis complete"
                        message = display_message
                        if ui.enabled:
                            ui.set_stage_status("contextual", display_message)

                    if message and not ui.enabled:
                        display_message = message
                        if not display_message.startswith("["):
                            display_message = f"[contextual] {display_message}"
                        console.log(display_message)
                
                try:
                    model_manager = RuntimeModelManager.get_instance()
                    perception_log = apply_contextual_understanding(
                        perception_log, model_manager, progress_callback=handle_contextual_progress
                    )
                    contextual_duration = time.time() - contextual_start
                    
                    # Count corrections
                    corrections = sum(1 for obj in perception_log if obj.get('was_corrected'))
                    spatial_zones = sum(1 for obj in perception_log if obj.get('spatial_zone') != 'unknown')
                    
                    if ui.enabled:
                        ui.complete_stage(
                            "contextual", 
                            f"Complete: {corrections} corrections, {spatial_zones} spatial zones"
                        )
                    
                    results["contextual"] = {
                        "success": True,
                        "duration_seconds": contextual_duration,
                        "corrections": corrections,
                        "spatial_zones": spatial_zones,
                    }
                except Exception as exc:
                    logging.getLogger(__name__).warning(f"Contextual analysis failed: {exc}")
                    if ui.enabled:
                        ui.warn_stage("contextual", "Contextual analysis failed (continuing)")

                semantic_progress_state = {"last_index": 0, "total": None}

                def handle_semantic_progress(event: str, payload: Dict[str, Any]) -> None:
                    message: Optional[str] = None

                    if event == "semantic.start":
                        total = payload.get("total")
                        if total:
                            semantic_progress_state["total"] = total
                            if ui.enabled:
                                ui.set_stage_total("semantic", total)
                        display_message = payload.get("message") or f"Initializing semantic uplift ({part2_config} mode)"
                        message = display_message
                        if ui.enabled:
                            ui.set_stage_status("semantic", display_message)
                    elif event == "semantic.step.start":
                        index = payload.get("index")
                        total = payload.get("total") or semantic_progress_state.get("total")
                        base_message = payload.get("message") or payload.get("name", "Semantic step")
                        if index and total:
                            display_message = f"Step {index}/{total}: {base_message}"
                        else:
                            display_message = base_message
                        message = display_message
                        if ui.enabled:
                            ui.set_stage_status("semantic", display_message)
                    elif event == "semantic.progress":
                        message = payload.get("message")
                        if ui.enabled and message:
                            ui.set_stage_status("semantic", message)
                    elif event == "semantic.step.complete":
                        index = payload.get("index")
                        total = payload.get("total") or semantic_progress_state.get("total")
                        detail = payload.get("detail")
                        name = payload.get("name", "Semantic step")
                        display_message = detail if detail is not None else f"{name} complete"
                        message = display_message
                        last_index = semantic_progress_state.get("last_index", 0)
                        if index is not None and index > last_index:
                            semantic_progress_state["last_index"] = index
                            if ui.enabled:
                                ui.advance_stage("semantic", current=index, message=display_message)
                        elif ui.enabled:
                            ui.set_stage_status("semantic", display_message)
                        if total and semantic_progress_state.get("total") is None:
                            semantic_progress_state["total"] = total
                            if ui.enabled:
                                ui.set_stage_total("semantic", total)
                    elif event == "semantic.warning":
                        warning_message = payload.get("message")
                        if warning_message:
                            message = f"⚠ {warning_message}"
                            if ui.enabled:
                                ui.set_stage_status("semantic", message)
                    elif event == "semantic.error":
                        error_message = payload.get("message") or "Semantic uplift error"
                        message = error_message
                        if ui.enabled:
                            ui.fail_stage("semantic", error_message)
                    elif event == "semantic.complete":
                        message = payload.get("message")
                        if ui.enabled and message:
                            ui.set_stage_status("semantic", message)

                    if message and not ui.enabled:
                        display_message = message
                        if not display_message.startswith("["):
                            display_message = f"[semantic] {display_message}"
                        console.log(display_message)

                start_time = time.time()
                try:
                    uplift_results = run_semantic_uplift(
                        perception_log,
                        neo4j_uri=neo4j_uri,
                        neo4j_user=neo4j_user,
                        neo4j_password=neo4j_password,
                        progress_callback=handle_semantic_progress,
                    )
                except Exception as exc:
                    if ui.enabled:
                        ui.fail_stage("semantic", "Semantic uplift failed")
                    raise

                uplift_duration = time.time() - start_time
                entities_tracked = uplift_results.get("num_entities", 0)

                if ui.enabled:
                    ui.complete_stage(
                        "semantic",
                        f"Semantic uplift complete ({entities_tracked} entities)",
                    )

                results["part2"] = {
                    "success": uplift_results.get("success", False),
                    "duration_seconds": uplift_duration,
                    "perception_log": perception_log_path,
                    "mode": part2_config,
                    **uplift_results,
                }

        results["success"] = True

        output_file = (
            Path(output_dir)
            / f"pipeline_results_{Path(video_path).stem}_{results['timestamp']}.json"
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        summary = Table(title="Pipeline Summary", box=box.ROUNDED, expand=False)
        summary.add_column("Stage", style="cyan", no_wrap=True)
        summary.add_column("Details", style="green")

        runtime_label = results.get("runtime", "-")
        summary.add_row("Runtime", runtime_label)

        if skip_part1:
            summary.add_row("Perception", "Skipped")
        elif perception_duration is not None and results["part1"].get("success"):
            summary.add_row(
                "Perception",
                f"{results['part1']['num_objects']} objects in {perception_duration:.1f}s",
            )
        else:
            summary.add_row("Perception", "Failed")

        if skip_part2:
            summary.add_row("Semantic Uplift", "Skipped")
        elif uplift_duration is not None and results["part2"].get("success"):
            summary.add_row(
                "Semantic Uplift",
                f"{uplift_results.get('num_entities', 0)} entities in {uplift_duration:.1f}s",
            )
        else:
            summary.add_row("Semantic Uplift", "Failed")

        if neo4j_status is not None:
            summary.add_row("Neo4j", neo4j_status)

        summary.add_row("Artifacts", str(output_file))

        console.print()
        console.print(summary)

        if uplift_results.get("success"):
            metrics_tables: List[Table] = []

            composer_rows: List[Tuple[str, str]] = []
            windows_total = uplift_results.get("llm_windows_total", 0)
            windows_composed = uplift_results.get("llm_windows_composed", 0)
            windows_skipped = uplift_results.get("llm_windows_skipped", 0)
            windows_capped = uplift_results.get("llm_windows_capped", 0)
            template_windows = uplift_results.get("llm_windows_template", 0)
            cached_windows = uplift_results.get("llm_windows_cached", 0)
            pruned_signal = uplift_results.get("llm_windows_pruned_signal", 0)
            llm_batches = uplift_results.get("llm_batches", 0)
            llm_calls = uplift_results.get("llm_calls", 0)
            llm_latency = uplift_results.get("llm_latency_seconds", 0.0)

            if windows_total:
                composed_display = f"{windows_composed}/{windows_total}"
                if windows_skipped:
                    composed_display += f" (skipped {windows_skipped})"
                composer_rows.append(("Windows composed", composed_display))
            if windows_capped:
                composer_rows.append(("Windows capped", str(windows_capped)))
            if template_windows:
                composer_rows.append(("Template fallbacks", str(template_windows)))
            if cached_windows:
                composer_rows.append(("Cache hits", str(cached_windows)))
            if pruned_signal:
                composer_rows.append(("Pruned low-signal", str(pruned_signal)))
            if llm_batches:
                composer_rows.append(("LLM batches", str(llm_batches)))
            if llm_calls:
                composer_rows.append(("LLM calls", str(llm_calls)))
            if llm_latency:
                composer_rows.append(("LLM latency (s)", f"{llm_latency:.2f}"))

            if composer_rows:
                composer_table = Table(title="Event Composition", box=box.ROUNDED, expand=False)
                composer_table.add_column("Metric", style="cyan", no_wrap=True)
                composer_table.add_column("Value", style="green")
                for label, value in composer_rows:
                    composer_table.add_row(label, value)
                metrics_tables.append(composer_table)

            causal_rows: List[Tuple[str, str]] = []
            causal_windows = uplift_results.get("windows_with_causal", 0)
            causal_links = uplift_results.get("num_causal_links", 0)
            causal_calls = uplift_results.get("causal_engine_calls", 0)
            causal_pairs = uplift_results.get("causal_candidate_pairs", 0)
            causal_retained = uplift_results.get("causal_pairs_retained", 0)

            if causal_links:
                causal_rows.append(("Causal links", str(causal_links)))
            if causal_windows:
                causal_rows.append(("Windows with causal signals", str(causal_windows)))
            if causal_calls:
                causal_rows.append(("Causal engine calls", str(causal_calls)))
            if causal_pairs:
                retained_display = str(causal_retained)
                causal_rows.append(("Candidate pairs", f"{causal_pairs} (retained {retained_display})"))

            if causal_rows:
                causal_table = Table(title="Causal Scoring", box=box.ROUNDED, expand=False)
                causal_table.add_column("Metric", style="cyan", no_wrap=True)
                causal_table.add_column("Value", style="green")
                for label, value in causal_rows:
                    causal_table.add_row(label, value)
                metrics_tables.append(causal_table)

            graph_stats = uplift_results.get("graph_stats") or {}
            if graph_stats:
                graph_table = Table(title="Graph Writes", box=box.ROUNDED, expand=False)
                graph_table.add_column("Metric", style="cyan", no_wrap=True)
                graph_table.add_column("Value", style="green")
                for key, value in sorted(graph_stats.items()):
                    label = key.replace("_", " ").title()
                    graph_table.add_row(label, str(value))
                metrics_tables.append(graph_table)

            if metrics_tables:
                console.print()
                for table in metrics_tables:
                    console.print(table)

        console.print(
            Panel(
                f"Pipeline complete\n[bold]{output_file}[/bold]",
                title="Status",
                subtitle="Results saved",
                style="green",
                expand=False,
            )
        )

        return results

    except Exception as exc:
        console.print(
            Panel(
                f"Pipeline failed: {exc}",
                title="Error",
                style="red",
                expand=False,
            )
        )
        results["success"] = False
        results["errors"].append(str(exc))
        import traceback

        logging.error(traceback.format_exc())
        return results

    finally:
        if previous_suppress is None:
            os.environ.pop("ORION_SUPPRESS_ENGINE_LOGS", None)
        else:
            os.environ["ORION_SUPPRESS_ENGINE_LOGS"] = previous_suppress
        _refresh_engine_loggers()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Orion Video Analysis Pipeline - Extract knowledge from video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python run_pipeline.py video.mp4
  
  # Run with custom config
  python run_pipeline.py video.mp4 --config accurate
  
  # Skip perception (use existing results)
  python run_pipeline.py video.mp4 --skip-part1
  
  # Run then start Q&A session
  python run_pipeline.py video.mp4 --interactive
        """,
    )

    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--output-dir", default="data/testing", help="Output directory")
    parser.add_argument(
        "--config",
        choices=["fast", "balanced", "accurate"],
        default="balanced",
        help="Processing configuration",
    )
    parser.add_argument(
        "--neo4j-uri", default=None, help="Neo4j URI (uses ConfigManager if not provided)"
    )
    parser.add_argument("--neo4j-password", default=None, help="Neo4j password (uses ConfigManager if not provided)")
    parser.add_argument(
        "--no-clear-db", action="store_true", help="Do not clear Neo4j before running"
    )
    parser.add_argument(
        "--skip-part1", action="store_true", help="Skip perception engine"
    )
    parser.add_argument(
        "--skip-part2", action="store_true", help="Skip semantic uplift"
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Start interactive Q&A after pipeline",
    )
    parser.add_argument(
        "--qa-only", action="store_true", help="Skip pipeline, only run Q&A"
    )
    parser.add_argument("--qa-model", default="gemma3:4b", help="Ollama model for Q&A")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )

    args = parser.parse_args()

    print_banner()

    # Q&A only mode
    if args.qa_only:
        if not QA_AVAILABLE:
            console.print(
                "[red]Error: Q&A system not available (install ollama package)[/red]"
            )
            console.print("[dim]Run: pip install ollama[/dim]")
            return
        console.print("\n[bold cyan]Starting Q&A session...[/bold cyan]\n")
        qa = VideoQASystem(neo4j_password=args.neo4j_password, model=args.qa_model)
        qa.start_interactive_session()
        return

    # Run pipeline
    results = run_pipeline(
        video_path=args.video_path,
        output_dir=args.output_dir,
        neo4j_uri=args.neo4j_uri,
        neo4j_password=args.neo4j_password,
        clear_db=not args.no_clear_db,
        part1_config=args.config,
        part2_config=args.config,
        skip_part1=args.skip_part1,
        skip_part2=args.skip_part2,
        verbose=args.verbose,
    )

    # Interactive Q&A if requested
    if args.interactive and results["success"]:
        if not QA_AVAILABLE:
            console.print("\n[yellow]Warning: Q&A system not available[/yellow]")
            console.print("[dim]Install with: pip install ollama[/dim]")
            return
        console.print("\n[bold cyan]Starting interactive Q&A session...[/bold cyan]")
        console.print("[dim]You can now ask questions about the video![/dim]\n")

        qa = VideoQASystem(neo4j_password=args.neo4j_password, model=args.qa_model)
        qa.start_interactive_session()


if __name__ == "__main__":
    main()
