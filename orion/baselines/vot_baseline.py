"""
LLM-Only Captions Baseline (Video-of-Thought Style)

This baseline mimics the Video-of-Thought (VoT) paradigm by:
1. Generating dense video descriptions using FastVLM at 0.5 FPS
2. Feeding descriptions directly to Gemma3 for scene reasoning
3. Extracting relationships/events from free-form text output

This baseline omits structured embeddings and entity tracking,
exposing the limits of free-form caption reasoning in maintaining
entity continuity and causal consistency.

Reference: Fei et al., 2024 - Video of Thought
Author: Orion Research Team
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from rich.console import Console
from rich.progress import track

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class VOTConfig:
    """Configuration for VoT baseline"""
    fps: float = 0.5  # Frame sampling rate
    description_model: str = "fastvlm"  # Model for generating descriptions
    llm_model: str = "gemma3:4b"  # LLM for reasoning
    max_description_tokens: int = 150
    max_reasoning_tokens: int = 300
    scene_window_seconds: float = 5.0  # Group captions into scenes
    enable_temporal_reasoning: bool = True
    temperature: float = 0.7


@dataclass
class CaptionedFrame:
    """Represents a frame with its caption"""
    frame_idx: int
    timestamp: float
    image: np.ndarray
    description: str
    confidence: float = 1.0


@dataclass
class SceneDescription:
    """A scene composed of multiple captions"""
    scene_id: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    captions: List[str] = field(default_factory=list)
    reasoning: str = ""
    extracted_entities: List[Dict[str, Any]] = field(default_factory=list)
    extracted_relationships: List[Dict[str, Any]] = field(default_factory=list)
    extracted_events: List[Dict[str, Any]] = field(default_factory=list)


class FastVLMCaptioner:
    """Generates video descriptions using FastVLM"""

    def __init__(self, model_name: str = "fastvlm", device: str = "auto"):
        """Initialize FastVLM for captioning
        
        Args:
            model_name: Model identifier
            device: Device to use ("auto", "cuda", "cpu")
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None
        logger.info(f"FastVLMCaptioner initialized (will load on first use)")

    def _ensure_loaded(self):
        """Lazy load FastVLM model"""
        if self._model is not None:
            return

        try:
            logger.info("Loading FastVLM model...")
            # Try MLX first (Apple Silicon)
            try:
                from orion.backends.mlx_fastvlm import FastVLMMLXWrapper
                self._model = FastVLMMLXWrapper()
                logger.info("Using MLX backend for FastVLM")
            except (ImportError, RuntimeError):
                # Fall back to transformers
                logger.info("MLX not available, using transformers backend")
                from transformers import AutoProcessor, AutoModelForCausalLM
                import torch
                
                model_id = "microsoft/phi-3.5-vision-instruct"
                self._processor = AutoProcessor.from_pretrained(
                    model_id, trust_remote_code=True
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_id, trust_remote_code=True, torch_dtype=torch.float16
                )
                if self.device == "auto":
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self._model = self._model.to(self.device)
                self._model.eval()
        except Exception as e:
            logger.error(f"Failed to load FastVLM: {e}")
            raise

    def caption_frame(self, image: np.ndarray, prompt: str = "Describe what you see in detail:") -> str:
        """Generate a caption for a single frame
        
        Args:
            image: Frame as numpy array (H, W, 3) in BGR format
            prompt: Instruction prompt for the model
            
        Returns:
            Generated caption string
        """
        self._ensure_loaded()

        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Use MLX backend if available
            if hasattr(self._model, 'generate_caption'):
                caption = self._model.generate_caption(image_rgb, prompt)
            else:
                # Transformers backend
                from PIL import Image
                pil_image = Image.fromarray(image_rgb)
                
                messages = [
                    {
                        "role": "user",
                        "content": f"<|image_1|>\n{prompt}",
                    }
                ]
                
                prompt_text = self._processor.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                inputs = self._processor(
                    prompt_text, [pil_image], return_tensors="pt"
                )
                if self.device != "cpu":
                    for key in inputs:
                        if hasattr(inputs[key], 'to'):
                            inputs[key] = inputs[key].to(self.device)
                
                with __import__('torch').no_grad():
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=150,
                        temperature=0.7,
                    )
                
                caption = self._processor.decode(outputs[0], skip_special_tokens=True)
                # Extract just the caption part (after the prompt)
                if "Assistant:" in caption:
                    caption = caption.split("Assistant:")[-1].strip()

            return caption.strip()
        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return f"[Error: {str(e)[:50]}]"

    def caption_video(self, video_path: str, fps: float = 0.5) -> List[CaptionedFrame]:
        """Generate captions for video frames at specified FPS
        
        Args:
            video_path: Path to video file
            fps: Target frames per second to caption (0.5 = every 2 seconds at 30fps)
            
        Returns:
            List of CaptionedFrame objects
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return []

        source_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(source_fps / fps))
        
        captioned_frames = []
        frame_idx = 0
        
        logger.info(f"Captioning video at {fps} FPS (sampling every {frame_interval} frames)")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    
                    caption = self.caption_frame(frame)
                    captioned_frames.append(
                        CaptionedFrame(
                            frame_idx=frame_idx,
                            timestamp=timestamp,
                            image=frame.copy(),
                            description=caption,
                        )
                    )

                frame_idx += 1
        finally:
            cap.release()

        logger.info(f"Generated {len(captioned_frames)} captions")
        return captioned_frames


class Gemma3Reasoner:
    """Uses Gemma3 LLM for scene reasoning over captions"""

    def __init__(self, model_name: str = "gemma3:4b", base_url: str = "http://localhost:11434"):
        """Initialize Gemma3 reasoner
        
        Args:
            model_name: Ollama model name
            base_url: Ollama API base URL
        """
        self.model_name = model_name
        self.base_url = base_url
        self._client = None

    def _ensure_client(self):
        """Initialize Ollama client"""
        if self._client is not None:
            return

        try:
            import ollama
            self._client = ollama.Client(host=self.base_url)
            
            # Test connection
            self._client.list()
            logger.info(f"Connected to Ollama at {self.base_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            logger.info("Ensure Ollama is running: ollama serve")
            raise

    def reason_over_scene(self, captions: List[str], scene_context: str = "") -> str:
        """Use Gemma3 to reason over a sequence of captions
        
        Args:
            captions: List of frame descriptions
            scene_context: Additional context about the scene
            
        Returns:
            Reasoning output from Gemma3
        """
        self._ensure_client()

        caption_text = "\n".join(f"[Frame {i+1}] {cap}" for i, cap in enumerate(captions))
        
        prompt = f"""You are analyzing a video scene based on frame descriptions.

Scene Captions:
{caption_text}

{f"Context: {scene_context}" if scene_context else ""}

Please analyze these captions and:
1. Identify the main objects/entities in the scene
2. Describe the relationships between entities (e.g., "person holding object", "object on table")
3. Identify key actions or events happening
4. Describe any causal relationships (what causes what)
5. Summarize the overall scene understanding

Format your response as structured observations."""

        try:
            response = self._client.generate(
                model=self.model_name,
                prompt=prompt,
                stream=False,
                options={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 300,
                }
            )
            
            return response["response"].strip()
        except Exception as e:
            logger.error(f"Error calling Gemma3: {e}")
            return f"[Error: {str(e)[:100]}]"

    def extract_triplets_from_reasoning(self, reasoning: str) -> List[Dict[str, str]]:
        """Extract (subject, predicate, object) triplets from LLM reasoning
        
        Args:
            reasoning: LLM reasoning output
            
        Returns:
            List of relationship triplets
        """
        triplets = []
        
        # Parse reasoning for structured patterns
        lines = reasoning.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Look for patterns like "X is/does Y with Z"
            if ' is ' in line or ' does ' in line or ' with ' in line:
                triplets.append({
                    "text": line,
                    "confidence": 0.6,  # Lower confidence for free-form extraction
                    "source": "llm_reasoning"
                })
        
        # Also look for explicit relationship markers
        if 'relationship' in reasoning.lower():
            for line in lines:
                if 'relationship' in line.lower():
                    triplets.append({
                        "text": line.strip(),
                        "confidence": 0.7,
                        "source": "llm_explicit"
                    })

        return triplets


class VOTBaseline:
    """Video-of-Thought style baseline for comparison with Orion"""

    def __init__(self, config: Optional[VOTConfig] = None):
        """Initialize VoT baseline
        
        Args:
            config: Configuration object
        """
        self.config = config or VOTConfig()
        self.captioner = FastVLMCaptioner(self.config.description_model)
        self.reasoner = Gemma3Reasoner(self.config.llm_model)

    def process_video(self, video_path: str) -> Dict[str, Any]:
        """Process entire video using VoT pipeline
        
        Args:
            video_path: Path to input video
            
        Returns:
            Structured output with entities, relationships, events
        """
        logger.info(f"Starting VoT baseline processing: {video_path}")

        # Step 1: Generate captions at 0.5 FPS
        console.print("[cyan]Step 1: Generating captions at 0.5 FPS[/cyan]")
        captioned_frames = self.captioner.caption_video(
            video_path, fps=self.config.fps
        )
        
        if not captioned_frames:
            logger.error("No frames captioned")
            return {"error": "No frames processed"}

        # Step 2: Group captions into scenes
        console.print("[cyan]Step 2: Grouping captions into scenes[/cyan]")
        scenes = self._group_into_scenes(captioned_frames)
        logger.info(f"Created {len(scenes)} scenes")

        # Step 3: Reason over each scene with Gemma3
        console.print("[cyan]Step 3: Reasoning over scenes with Gemma3[/cyan]")
        for scene in track(scenes, description="Reasoning..."):
            reasoning = self.reasoner.reason_over_scene(
                [cf.description for cf in scene.captions],
                scene_context=f"Scene {scene.scene_id}"
            )
            scene.reasoning = reasoning
            
            # Extract relationships from reasoning
            relationships = self.reasoner.extract_triplets_from_reasoning(reasoning)
            scene.extracted_relationships = relationships

        # Step 4: Aggregate results
        console.print("[cyan]Step 4: Aggregating results[/cyan]")
        result = self._aggregate_scenes(scenes, captioned_frames)

        logger.info("VoT baseline processing complete")
        return result

    def _group_into_scenes(self, captioned_frames: List[CaptionedFrame]) -> List[SceneDescription]:
        """Group captioned frames into scenes based on temporal windows
        
        Args:
            captioned_frames: List of captioned frames
            
        Returns:
            List of SceneDescription objects
        """
        scenes = []
        window_duration = self.config.scene_window_seconds
        
        current_scene = None
        scene_id = 0

        for cf in captioned_frames:
            if current_scene is None:
                # Start new scene
                current_scene = SceneDescription(
                    scene_id=scene_id,
                    start_frame=cf.frame_idx,
                    end_frame=cf.frame_idx,
                    start_time=cf.timestamp,
                    end_time=cf.timestamp,
                )
                scene_id += 1

            # Check if frame belongs to current scene
            if cf.timestamp - current_scene.start_time < window_duration:
                current_scene.end_frame = cf.frame_idx
                current_scene.end_time = cf.timestamp
                current_scene.captions.append(cf.description)
            else:
                # Start new scene
                scenes.append(current_scene)
                current_scene = SceneDescription(
                    scene_id=scene_id,
                    start_frame=cf.frame_idx,
                    end_frame=cf.frame_idx,
                    start_time=cf.timestamp,
                    end_time=cf.timestamp,
                )
                scene_id += 1
                current_scene.captions.append(cf.description)

        # Add final scene
        if current_scene and current_scene.captions:
            scenes.append(current_scene)

        return scenes

    def _aggregate_scenes(
        self, scenes: List[SceneDescription], captioned_frames: List[CaptionedFrame]
    ) -> Dict[str, Any]:
        """Aggregate scene-level reasoning into video-level output
        
        Args:
            scenes: List of processed scenes
            captioned_frames: Original captioned frames
            
        Returns:
            Structured output dict
        """
        # Collect all entities and relationships
        all_entities = {}
        all_relationships = []
        all_events = []
        entity_id = 0

        for scene in scenes:
            # Add entities from reasoning
            for rel in scene.extracted_relationships:
                # Simple entity extraction from relationship text
                text = rel.get("text", "")
                
                # This is simplistic - in practice would need NER
                words = text.split()
                for word in words:
                    if word.lower() not in ['is', 'does', 'with', 'the', 'a', 'an', 'and', 'or']:
                        if word not in all_entities:
                            all_entities[str(entity_id)] = {
                                "class": word,
                                "description": f"From scene {scene.scene_id}",
                                "confidence": 0.5,
                                "frames": list(range(scene.start_frame, scene.end_frame + 1)),
                            }
                            entity_id += 1

                # Add relationship
                all_relationships.append({
                    "subject": 0,  # Simplified: would need proper linking
                    "predicate": rel.get("text", "unknown"),
                    "object": 1,
                    "confidence": rel.get("confidence", 0.5),
                    "frame_range": [scene.start_frame, scene.end_frame],
                })

            # Create event from scene reasoning
            if scene.reasoning:
                all_events.append({
                    "id": f"event_{scene.scene_id}",
                    "type": "scene_event",
                    "description": scene.reasoning[:200],
                    "start_frame": scene.start_frame,
                    "end_frame": scene.end_frame,
                    "confidence": 0.5,
                })

        # Compile output
        output = {
            "pipeline": "vot_baseline",
            "num_captions": len(captioned_frames),
            "num_scenes": len(scenes),
            "fps_sampled": self.config.fps,
            "entities": all_entities,
            "relationships": all_relationships,
            "events": all_events,
            "causal_links": [],  # VoT doesn't explicitly track causality
            "scenes": [
                {
                    "scene_id": s.scene_id,
                    "start_frame": s.start_frame,
                    "end_frame": s.end_frame,
                    "captions": s.captions,
                    "reasoning": s.reasoning[:500],  # Truncate for storage
                    "num_relationships": len(s.extracted_relationships),
                }
                for s in scenes
            ],
        }

        return output


def main(video_path: str, output_dir: str = "vot_results"):
    """Main entry point for VoT baseline
    
    Args:
        video_path: Input video path
        output_dir: Output directory for results
    """
    os.makedirs(output_dir, exist_ok=True)

    config = VOTConfig(
        fps=0.5,
        description_model="fastvlm",
        llm_model="gemma3:4b",
    )

    baseline = VOTBaseline(config)
    result = baseline.process_video(video_path)

    # Save results
    output_file = os.path.join(output_dir, "vot_predictions.json")
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    console.print(f"\n[green]Results saved to {output_file}[/green]")
    console.print(f"[cyan]Summary:[/cyan]")
    console.print(f"  Captions: {result.get('num_captions', 0)}")
    console.print(f"  Scenes: {result.get('num_scenes', 0)}")
    console.print(f"  Entities: {len(result.get('entities', {}))}")
    console.print(f"  Relationships: {len(result.get('relationships', []))}")
    console.print(f"  Events: {len(result.get('events', []))}")

    return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python vot_baseline.py <video_path> [output_dir]")
        sys.exit(1)

    video_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "vot_results"

    main(video_path, output_dir)
