"""
Asynchronous Entity Description Engine
========================================

Async FastVLM description generation for unique entities POST-clustering.

This is Phase 3 of the tracking pipeline, but with async workers:
- Input: List of unique entities (from HDBSCAN clustering)
- Process: Generate FastVLM descriptions in parallel using async workers  
- Output: Same entities with descriptions attached

KEY: FastVLM only runs on UNIQUE ENTITIES, not every detection.
This ensures we describe each object exactly once, from its best frame.

Author: Orion Research Team
Date: October 2025
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

try:
    from .config import OrionConfig, AsyncConfig
    from .model_manager import ModelManager
    from .tracking_engine import Entity, Observation
except ImportError:
    from config import OrionConfig, AsyncConfig  # type: ignore
    from model_manager import ModelManager  # type: ignore
    from tracking_engine import Entity, Observation  # type: ignore

logger = logging.getLogger("AsyncEntityDescriber")


@dataclass
class DescriptionTask:
    """Task for describing a unique entity"""
    entity: Entity
    best_observation: Observation
    task_id: int


@dataclass
class DescriptionResult:
    """Result from VLM description"""
    entity_id: str
    description: str
    frame_number: int
    generation_time: float


@dataclass
class AsyncDescriptionStats:
    """Statistics for async description"""
    total_entities: int = 0
    descriptions_generated: int = 0
    descriptions_pending: int = 0
    slow_loop_fps: float = 0.0
    total_time: float = 0.0
    queue_max_size: int = 0
    average_description_time: float = 0.0


class AsyncEntityDescriber:
    """
    Asynchronous entity description engine.
    
    Generates FastVLM descriptions for unique entities using parallel workers.
    This is a drop-in replacement for the synchronous EntityDescriber.
    """
    
    def __init__(
        self,
        config: Optional[OrionConfig] = None,
        async_config: Optional[AsyncConfig] = None,
        progress_callback: Optional[Callable[[str, Dict], None]] = None
    ):
        self.config = config or OrionConfig()
        self.async_config = async_config or self.config.async_processing
        self.progress_callback = progress_callback
        
        self.model_manager = ModelManager.get_instance()
        
        # Queues for async coordination
        self.task_queue: asyncio.Queue[Optional[DescriptionTask]] = asyncio.Queue(
            maxsize=self.async_config.max_queue_size
        )
        self.result_queue: asyncio.Queue[Optional[DescriptionResult]] = asyncio.Queue()
        
        # Statistics
        self.stats = AsyncDescriptionStats()
        
        logger.info("AsyncEntityDescriber initialized")
        logger.info(f"  Description workers: {self.async_config.num_description_workers}")
    
    def _select_best_observation(self, entity: Entity) -> Observation:
        """Select best observation for description (same as sync version)"""
        if len(entity.observations) == 1:
            return entity.observations[0]
        
        scored_obs = []
        
        for obs in entity.observations:
            # Calculate scores
            size_score = obs.get_bbox_area()
            centrality_score = obs.get_centrality_score()
            confidence_score = obs.confidence
            
            # Weighted combination
            total_score = (
                self.config.description.size_weight * size_score +
                self.config.description.centrality_weight * centrality_score +
                self.config.description.confidence_weight * confidence_score
            )
            
            scored_obs.append((total_score, obs))
        
        # Return observation with highest score
        return max(scored_obs, key=lambda x: x[0])[1]
    
    async def _task_producer(self, entities: List[Entity]) -> None:
        """
        Producer: Create description tasks for all entities.
        """
        logger.info(f"Creating description tasks for {len(entities)} entities...")
        
        for i, entity in enumerate(entities):
            # Select best observation
            best_obs = self._select_best_observation(entity)
            
            # Check confidence - skip low-confidence entities
            if best_obs.confidence < self.config.detection.low_confidence_threshold:
                entity.description = (
                    f"Low confidence detection (conf={best_obs.confidence:.2f}). "
                    f"Likely false positive - skipped description."
                )
                entity.described_from_frame = best_obs.frame_number
                continue
            
            # Create task
            task = DescriptionTask(
                entity=entity,
                best_observation=best_obs,
                task_id=i
            )
            
            # Enqueue
            await self.task_queue.put(task)
        
        # Signal end of tasks
        for _ in range(self.async_config.num_description_workers):
            await self.task_queue.put(None)
        
        logger.info(f"Task producer complete: {self.task_queue.qsize()} tasks queued")
    
    async def _description_worker(self, worker_id: int) -> None:
        """
        Worker: Generate VLM descriptions from tasks.
        """
        logger.info(f"Description worker {worker_id} started")
        
        # Load FastVLM (lazy loaded, shared across workers)
        vlm = self.model_manager.fastvlm
        
        descriptions_count = 0
        total_time = 0.0
        
        while True:
            # Get task from queue
            task = await self.task_queue.get()
            
            # Check for termination signal
            if task is None:
                logger.info(f"Worker {worker_id} received termination signal")
                break
            
            try:
                # Generate description
                desc_start = time.time()
                
                # Convert crop to PIL Image
                crop = task.best_observation.crop
                rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if crop.shape[2] == 3 else crop
                pil_image = Image.fromarray(rgb_crop)
                
                # Generate unbiased description
                prompt = f"""What do you see in this image? Provide a detailed description.

Focus on:
- What type of object this is
- Its appearance, color, and shape
- Any distinguishing features or characteristics
- The context or setting if visible

Be objective and describe exactly what you observe."""
                
                description = vlm.generate_description(
                    pil_image,
                    prompt,
                    max_tokens=self.config.description.max_tokens,
                    temperature=self.config.description.temperature
                )
                
                desc_time = time.time() - desc_start
                total_time += desc_time
                
                # Create result
                result = DescriptionResult(
                    entity_id=task.entity.id,
                    description=description,
                    frame_number=task.best_observation.frame_number,
                    generation_time=desc_time
                )
                
                # Enqueue result
                await self.result_queue.put(result)
                
                descriptions_count += 1
                
                if descriptions_count % 10 == 0:
                    avg_time = total_time / descriptions_count
                    logger.info(
                        f"Worker {worker_id}: {descriptions_count} descriptions "
                        f"(avg time: {avg_time:.2f}s)"
                    )
            
            except Exception as e:
                logger.error(f"Worker {worker_id} error processing entity {task.entity.id}: {e}")
                # Still enqueue a result with error description
                result = DescriptionResult(
                    entity_id=task.entity.id,
                    description=f"A {task.entity.class_name} (description failed: {str(e)})",
                    frame_number=task.best_observation.frame_number,
                    generation_time=0.0
                )
                await self.result_queue.put(result)
            
            finally:
                self.task_queue.task_done()
        
        avg_time = total_time / descriptions_count if descriptions_count > 0 else 0
        logger.info(
            f"Worker {worker_id} complete: {descriptions_count} descriptions "
            f"in {total_time:.2f}s (avg: {avg_time:.2f}s)"
        )
    
    async def _result_collector(self, entities: List[Entity]) -> None:
        """
        Collector: Collect description results and attach to entities.
        """
        logger.info("Result collector started")
        
        # Create entity lookup
        entity_map = {e.id: e for e in entities}
        
        results_collected = 0
        total_expected = sum(1 for e in entities if e.description is None)
        
        while results_collected < total_expected:
            try:
                # Get result with timeout
                result = await asyncio.wait_for(
                    self.result_queue.get(),
                    timeout=5.0
                )
                
                # Attach to entity
                if result.entity_id in entity_map:
                    entity = entity_map[result.entity_id]
                    entity.description = result.description
                    entity.described_from_frame = result.frame_number
                    results_collected += 1
                    
                    self.stats.descriptions_generated = results_collected
                    
                    if results_collected % 20 == 0:
                        logger.info(f"Collected {results_collected}/{total_expected} descriptions")
            
            except asyncio.TimeoutError:
                logger.warning("Result collector timeout - waiting for more results...")
                continue
            except Exception as e:
                logger.error(f"Result collector error: {e}")
                break
        
        logger.info(f"Result collector complete: {results_collected} descriptions collected")
    
    async def describe_entities_async(self, entities: List[Entity]) -> List[Entity]:
        """
        Describe entities asynchronously using parallel workers.
        
        Args:
            entities: List of unique entities from clustering
            
        Returns:
            Same entities with descriptions attached
        """
        logger.info("="*80)
        logger.info("PHASE 3: ASYNC DESCRIPTION GENERATION")
        logger.info("="*80)
        logger.info(f"Describing {len(entities)} unique entities...")
        logger.info(f"Workers: {self.async_config.num_description_workers}")
        logger.info("(Only describing each entity ONCE from its best frame)")
        
        start_time = time.time()
        self.stats.total_entities = len(entities)
        
        # Create tasks
        tasks = [
            # Producer
            asyncio.create_task(self._task_producer(entities)),
            
            # Workers
            *[
                asyncio.create_task(self._description_worker(i))
                for i in range(self.async_config.num_description_workers)
            ],
            
            # Collector
            asyncio.create_task(self._result_collector(entities))
        ]
        
        # Wait for all tasks
        await asyncio.gather(*tasks)
        
        end_time = time.time()
        self.stats.total_time = end_time - start_time
        self.stats.slow_loop_fps = (
            self.stats.descriptions_generated / self.stats.total_time
            if self.stats.total_time > 0 else 0
        )
        
        # Count results
        described_count = sum(1 for e in entities if e.description is not None and "skipped" not in e.description)
        skipped_count = sum(1 for e in entities if e.description is not None and "skipped" in e.description)
        
        logger.info("="*80)
        logger.info(f"✓ Described {described_count} entities")
        if skipped_count > 0:
            logger.info(f"  Skipped {skipped_count} low-confidence entities")
        logger.info(f"  Total time: {self.stats.total_time:.2f}s")
        logger.info(f"  Avg FPS: {self.stats.slow_loop_fps:.1f} descriptions/sec")
        logger.info(f"  Workers: {self.async_config.num_description_workers}")
        logger.info("="*80 + "\n")
        
        if self.progress_callback:
            self.progress_callback("async_description.complete", {
                "total_entities": len(entities),
                "described": described_count,
                "skipped": skipped_count,
                "time": self.stats.total_time,
                "fps": self.stats.slow_loop_fps
            })
        
        return entities
    
    def describe_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Synchronous wrapper for async description.
        
        This is a drop-in replacement for the synchronous EntityDescriber.
        """
        return asyncio.run(self.describe_entities_async(entities))


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def describe_entities_async(
    entities: List[Entity],
    config: Optional[OrionConfig] = None,
    async_config: Optional[AsyncConfig] = None,
    progress_callback: Optional[Callable[[str, Dict], None]] = None
) -> List[Entity]:
    """
    Convenience function to describe entities asynchronously.
    
    Args:
        entities: List of unique entities from clustering
        config: Orion configuration
        async_config: Async configuration  
        progress_callback: Optional progress callback
        
    Returns:
        Entities with descriptions attached
    """
    describer = AsyncEntityDescriber(config, async_config, progress_callback)
    return describer.describe_entities(entities)
