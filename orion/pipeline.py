"""
Main Pipeline Orchestrator

Coordinates the complete video understanding pipeline across perception, semantic,
and graph ingestion stages.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from orion.perception.config import PerceptionConfig
from orion.semantic.config import SemanticConfig
# from orion.graph.database import Neo4jManager  # REMOVED: Neo4j deprecated, using Memgraph
from orion.perception.engine import PerceptionEngine
from orion.semantic.engine import SemanticEngine

if TYPE_CHECKING:
    from orion.perception.types import PerceptionResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the complete pipeline"""

    perception_config: PerceptionConfig = field(default_factory=PerceptionConfig)
    semantic_config: SemanticConfig = field(default_factory=SemanticConfig)

    # Graph configuration (deprecated Neo4j, now using Memgraph)
    # neo4j_uri: str = "neo4j://localhost:7687"
    # neo4j_user: str = "neo4j"
    # neo4j_password: str = "password"

    # Pipeline behavior
    skip_semantic: bool = False
    skip_graph_ingestion: bool = False

    # Logging
    log_level: str = "INFO"


class VideoPipeline:
    """
    Main orchestrator for video understanding pipeline.

    Manages the flow of video through perception, semantic, and graph stages.
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize the pipeline with configuration.

        Args:
            config: Pipeline configuration with all stage configs
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(config.log_level)

        # Initialize stages
        self.perception_engine = PerceptionEngine(config.perception_config)
        self.semantic_engine = SemanticEngine(config.semantic_config)

        # Graph ingestion disabled (Neo4j deprecated, Memgraph integration pending)
        # self.neo4j_manager: Optional[Neo4jManager] = None
        # if not config.skip_graph_ingestion:
        #     try:
        #         self.neo4j_manager = Neo4jManager(...)
        #         self.logger.info("Neo4j manager initialized successfully")
        #     except Exception as e:
        #         self.logger.warning(f"Failed to initialize Neo4j: {e}")
        #         self.logger.info("Continuing without graph ingestion")

        self.logger.info("Pipeline initialized successfully")
        
        # For CLI compatibility
        self._inspection_mode = None
        self._video_path = None
        self._perception_result_obj = None  # Store actual PerceptionResult object

    @classmethod
    def from_config(cls, config_dict: dict) -> "VideoPipeline":
        """
        Create pipeline from configuration dictionary (CLI compatibility).
        
        Args:
            config_dict: Dictionary with configuration parameters
            
        Returns:
            VideoPipeline instance
        """
        from orion.perception.config import get_fast_config, get_balanced_config, get_accurate_config
        from orion.semantic.config import get_fast_semantic_config, get_balanced_semantic_config, get_accurate_semantic_config
        
        # Extract perception config using preset
        perception_dict = config_dict.get("perception", {})
        mode = perception_dict.get("mode", "balanced")
        
        if mode == "fast":
            perception_config = get_fast_config()
        elif mode == "accurate":
            perception_config = get_accurate_config()
        else:  # balanced
            perception_config = get_balanced_config()
        
        # Override 3D perception settings from CLI
        if "enable_3d" in perception_dict:
            perception_config.enable_3d = perception_dict["enable_3d"]
        if "depth_model" in perception_dict and perception_dict["depth_model"]:
            perception_config.depth_model = perception_dict["depth_model"]
        if "enable_hands" in perception_dict:
            perception_config.enable_hands = perception_dict["enable_hands"]
        if "enable_occlusion" in perception_dict:
            perception_config.enable_occlusion = perception_dict["enable_occlusion"]
        
        # Extract semantic config using preset
        semantic_dict = config_dict.get("semantic", {})
        semantic_mode = semantic_dict.get("mode", "balanced")
        
        if semantic_mode == "fast":
            semantic_config = get_fast_semantic_config()
        elif semantic_mode == "accurate":
            semantic_config = get_accurate_semantic_config()
        else:  # balanced
            semantic_config = get_balanced_semantic_config()
        
        # Neo4j config deprecated (using Memgraph)
        # neo4j_dict = config_dict.get("neo4j", {})
        
        # Create pipeline config
        pipeline_config = PipelineConfig(
            perception_config=perception_config,
            semantic_config=semantic_config,
            # neo4j_uri=neo4j_dict.get("uri", "neo4j://localhost:7687"),
            # neo4j_user=neo4j_dict.get("user", "neo4j"),
            # neo4j_password=neo4j_dict.get("password", "password"),
            skip_graph_ingestion=True,  # Disabled until Memgraph integration
        )
        
        # Create pipeline and store video path
        pipeline = cls(pipeline_config)
        pipeline._video_path = config_dict.get("video_path")
        
        return pipeline

    def set_inspection_mode(self, stage: str) -> None:
        """Set inspection mode to stop after specified stage."""
        self._inspection_mode = stage

    def run(self, skip_perception: bool = False, skip_semantic: bool = False, skip_graph: bool = False) -> dict:
        """
        Run the pipeline (CLI compatibility method).
        
        Args:
            skip_perception: Skip perception stage
            skip_semantic: Skip semantic stage  
            skip_graph: Skip graph ingestion stage
            
        Returns:
            Dictionary with results
        """
        if not self._video_path:
            raise ValueError("No video path specified")
            
        # Generate scene ID from video path
        scene_id = Path(self._video_path).stem
        
        try:
            results = self.process_video(
                video_path=self._video_path,
                scene_id=scene_id,
                inspect_stage=self._inspection_mode,
            )
            results["success"] = True
            return results
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def process_video(
        self,
        video_path: str,
        scene_id: str,
        metadata: Optional[dict] = None,
        inspect_stage: Optional[str] = None,
    ) -> dict:
        """
        Process a video through the complete pipeline.

        Args:
            video_path: Path to video file
            scene_id: Unique scene identifier
            metadata: Optional metadata to attach to results
            inspect_stage: If provided, stop after this stage and return intermediate results
                          ('perception', 'semantic', or 'graph')

        Returns:
            Dictionary with results from each stage:
            {
                'perception': {...},
                'semantic': {...},
                'graph': {...} (if ingested)
            }
        """
        results = {}
        try:
            self.logger.info(f"Starting pipeline for video: {video_path}")

            # Stage 1: Perception
            self.logger.info("Stage 1: Running perception engine...")
            perception_result_dict = self._run_perception(video_path, scene_id, metadata)
            results["perception"] = perception_result_dict

            if inspect_stage == "perception":
                self.logger.info("Stopping after perception stage (inspect_stage='perception')")
                return results

            # Stage 2: Semantic (if not skipped)
            if not self.config.skip_semantic:
                self.logger.info("Stage 2: Running semantic engine...")
                # Pass the actual PerceptionResult object stored during _run_perception
                if self._perception_result_obj is None:
                    raise ValueError("Perception result not available for semantic stage")
                semantic_result = self._run_semantic(
                    self._perception_result_obj, scene_id, metadata
                )
                results["semantic"] = semantic_result

                if inspect_stage == "semantic":
                    self.logger.info(
                        "Stopping after semantic stage (inspect_stage='semantic')"
                    )
                    return results
            else:
                self.logger.info("Skipping semantic stage (skip_semantic=True)")

            # Stage 3: Graph Ingestion (disabled - Neo4j deprecated)
            # if not self.config.skip_graph_ingestion and self.neo4j_manager:
            #     self.logger.info("Stage 3: Ingesting to knowledge graph...")
            #     graph_result = self._ingest_to_graph(
            #         results, scene_id, metadata
            #     )
            #     results["graph"] = graph_result
            # 
            #     if inspect_stage == "graph":
            #         self.logger.info(
            #             "Stopping after graph stage (inspect_stage='graph')"
            #         )
            #         return results

            self.logger.info("Pipeline completed successfully")
            return results

        except Exception as e:
            self.logger.error(f"Pipeline error: {e}", exc_info=True)
            raise

    def _run_perception(
        self, video_path: str, scene_id: str, metadata: Optional[dict]
    ) -> dict:
        """
        Run perception stage (detection, embedding, description).

        Args:
            video_path: Path to video file
            scene_id: Scene identifier
            metadata: Optional metadata

        Returns:
            Perception stage results as dict (for display)
        """
        try:
            result = self.perception_engine.process_video(video_path)
            
            # Store the actual object for semantic stage
            self._perception_result_obj = result
            
            # Convert PerceptionResult to dict format (flattened for display)
            result_dict = {
                "entities": [
                    {
                        "entity_id": e.entity_id,
                        "class": e.object_class.name if hasattr(e.object_class, 'name') else str(e.object_class),
                        "observations_count": len(e.observations),
                        "appearance_count": e.appearance_count,
                        "description": e.description,
                    }
                    for e in result.entities
                ],
                "raw_observations": [
                    {
                        "timestamp": o.timestamp,
                        "frame_number": o.frame_number,
                    }
                    for o in result.raw_observations
                ],
                # Flatten metadata to top level for display
                "video_path": result.video_path,
                "total_frames": result.total_frames,
                "fps": result.fps,
                "duration_seconds": result.duration_seconds,
                "total_detections": result.total_detections,
                "unique_entities": result.unique_entities,
                "processing_time_seconds": result.processing_time_seconds,
                # Add display-specific keys
                "num_entities": result.unique_entities,
                "unique_classes": len(set(e.object_class for e in result.entities)),
            }
            
            self.logger.info(
                f"Perception complete: {result.total_detections} detections, "
                f"{result.unique_entities} entities"
            )
            return result_dict
        except Exception as e:
            self.logger.error(f"Perception stage failed: {e}", exc_info=True)
            raise

    def _run_semantic(
        self, perception_result: "PerceptionResult", scene_id: str, metadata: Optional[dict]
    ) -> dict:
        """
        Run semantic stage (state changes, events, causal relationships).

        Args:
            perception_result: Actual PerceptionResult object from perception stage
            scene_id: Scene identifier
            metadata: Optional metadata

        Returns:
            Semantic stage results as dict (for display)
        """
        try:
            # Now we have the actual PerceptionResult object, pass it directly
            semantic_result = self.semantic_engine.process(perception_result)
            
            # Convert to dict
            result_dict = {
                "entities": [
                    {
                        "entity_id": e.entity_id,
                        "class": e.object_class,
                        "first_timestamp": e.first_timestamp,
                        "last_timestamp": e.last_timestamp,
                        "description": e.description,
                    }
                    for e in semantic_result.entities
                ],
                "state_changes": len(semantic_result.state_changes),
                "events": len(semantic_result.events),
                "scenes": len(semantic_result.scenes),
                "causal_links": len(semantic_result.causal_links),
                "temporal_windows": len(semantic_result.temporal_windows),
                # Add display-specific keys
                "num_entities": len(semantic_result.entities),
                "num_described": len([e for e in semantic_result.entities if e.description]),
                "num_spatial_zones": len(semantic_result.locations),
                "num_events": len(semantic_result.events),
                "num_causal_links": len(semantic_result.causal_links),
            }
            
            self.logger.info(
                f"Semantic complete: {len(semantic_result.entities)} entities, "
                f"{len(semantic_result.state_changes)} state changes, "
                f"{len(semantic_result.events)} events"
            )
            return result_dict
        except Exception as e:
            self.logger.error(f"Semantic stage failed: {e}", exc_info=True)
            raise

    def _ingest_to_graph(
        self, results: dict, scene_id: str, metadata: Optional[dict]
    ) -> dict:
        """
        Ingest results to knowledge graph (DEPRECATED - Neo4j removed).

        Args:
            results: Combined results from perception and semantic stages
            scene_id: Scene identifier
            metadata: Optional metadata

        Returns:
            Graph ingestion results
        """
        # Neo4j deprecated - graph ingestion disabled
        self.logger.warning("Graph ingestion disabled (Neo4j deprecated, Memgraph integration pending)")
        return {"status": "skipped", "reason": "Neo4j deprecated"}

    def close(self):
        """Close all resources"""
        # Neo4j manager deprecated
        # if self.neo4j_manager:
        #     try:
        #         self.neo4j_manager.close()
        #     except Exception as e:
        #         self.logger.warning(f"Error closing Neo4j manager: {e}")

        self.logger.info("Pipeline resources closed")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
