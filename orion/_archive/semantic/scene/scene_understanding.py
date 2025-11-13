"""
Scene Understanding Module for ORION Spatial Historian

Provides real-time scene classification, room type detection, and semantic
context for spatial reasoning. Runs in background thread to maintain 30 FPS.

Features:
- Room type classification (living room, kitchen, bedroom, etc.)
- Scene attributes (indoor/outdoor, lighting, activity)
- Temporal scene changes (detecting transitions)
- Zero-shot classification using CLIP (no training needed)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import time
import threading
from queue import Queue
from PIL import Image

try:
    from transformers import CLIPProcessor, CLIPModel
    import torch
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


@dataclass
class SceneContext:
    """Rich scene context for spatial reasoning"""
    room_type: str  # "living_room", "kitchen", "bedroom", "office"
    confidence: float
    scene_attributes: List[str]  # ["indoor", "residential", "furniture"]
    lighting: str  # "bright", "dim", "natural"
    activity_level: str  # "static", "active", "busy"
    timestamp: float
    
    def to_natural_language(self) -> str:
        """Convert to friendly description"""
        article = "an" if self.room_type[0] in "aeiou" else "a"
        return f"{article} {self.room_type.replace('_', ' ')}"


class SceneUnderstanding:
    """
    Real-time scene understanding with background processing
    
    Runs classification every N frames in a separate thread to avoid
    blocking the main perception loop. Maintains current scene context.
    """
    
    # Room type categories (common residential spaces)
    ROOM_TYPES = [
        "living room",
        "kitchen", 
        "bedroom",
        "bathroom",
        "dining room",
        "office",
        "hallway",
        "garage",
        "outdoor patio"
    ]
    
    # Scene attributes for richer context
    SCENE_ATTRIBUTES = [
        "indoor",
        "outdoor", 
        "bright lighting",
        "dim lighting",
        "furniture visible",
        "people present",
        "cluttered",
        "organized"
    ]
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        classification_interval: float = 1.0,  # Classify every 1 second
        use_threading: bool = True
    ):
        """
        Initialize scene understanding
        
        Args:
            model_name: CLIP model to use
            classification_interval: Seconds between classifications
            use_threading: Run in background thread (recommended)
        """
        self.classification_interval = classification_interval
        self.use_threading = use_threading
        
        # Current scene state
        self.current_scene: Optional[SceneContext] = None
        self.last_classification_time = 0
        
        # Background processing
        self.classification_queue: Queue = Queue(maxsize=2)
        self.result_queue: Queue = Queue(maxsize=1)
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None
        
        # Load CLIP model
        if CLIP_AVAILABLE:
            print("  Loading CLIP model for scene understanding...")
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()
            print(f"  ✓ CLIP loaded on {self.device}")
        else:
            print("  ⚠️  CLIP not available. Install with: pip install transformers torch")
            self.model = None
            self.processor = None
        
        # Start background thread
        if self.use_threading and self.model:
            self.start_background_processing()
    
    def start_background_processing(self):
        """Start background classification thread"""
        if self.running:
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._classification_worker, daemon=True)
        self.worker_thread.start()
    
    def stop_background_processing(self):
        """Stop background thread"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=1.0)
    
    def _classification_worker(self):
        """Background thread for scene classification"""
        while self.running:
            try:
                # Get frame from queue (with timeout)
                frame_data = self.classification_queue.get(timeout=0.1)
                
                # Classify scene
                result = self._classify_scene_sync(frame_data['frame'])
                
                # Put result in queue (drop old results if full)
                if self.result_queue.full():
                    try:
                        self.result_queue.get_nowait()
                    except:
                        pass
                self.result_queue.put(result)
                
            except Exception as e:
                continue
    
    def classify_scene(self, frame: np.ndarray, timestamp: float) -> Optional[SceneContext]:
        """
        Classify scene (non-blocking if using threading)
        
        Args:
            frame: Input frame (BGR numpy array)
            timestamp: Current video timestamp
            
        Returns:
            SceneContext if available, else previous result
        """
        if not self.model:
            return None
        
        # Check if we should classify (time-based throttling)
        if timestamp - self.last_classification_time < self.classification_interval:
            return self.current_scene
        
        self.last_classification_time = timestamp
        
        if self.use_threading:
            # Submit frame to background thread (non-blocking)
            if not self.classification_queue.full():
                try:
                    self.classification_queue.put_nowait({
                        'frame': frame.copy(),
                        'timestamp': timestamp
                    })
                except:
                    pass
            
            # Check for results
            try:
                result = self.result_queue.get_nowait()
                self.current_scene = result
            except:
                pass
            
            return self.current_scene
        else:
            # Synchronous classification
            self.current_scene = self._classify_scene_sync(frame)
            return self.current_scene
    
    def _classify_scene_sync(self, frame: np.ndarray) -> SceneContext:
        """
        Synchronous scene classification using CLIP
        
        Uses zero-shot classification with predefined room types.
        """
        # Convert BGR → RGB
        frame_rgb = frame[:, :, ::-1]
        
        # Convert to PIL Image
        image = Image.fromarray(frame_rgb)
        
        # Prepare inputs for CLIP
        text_inputs = [f"a photo of {room}" for room in self.ROOM_TYPES]
        
        inputs = self.processor(
            text=text_inputs,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
        
        # Get top prediction
        top_idx = np.argmax(probs)
        room_type = self.ROOM_TYPES[top_idx].replace(" ", "_")
        confidence = float(probs[top_idx])
        
        # Classify attributes (in parallel with room type)
        attr_probs = self._classify_attributes(frame_rgb)
        
        # Detect lighting and activity
        lighting = self._detect_lighting(frame)
        activity_level = "static"  # TODO: Use motion history
        
        return SceneContext(
            room_type=room_type,
            confidence=confidence,
            scene_attributes=attr_probs,
            lighting=lighting,
            activity_level=activity_level,
            timestamp=time.time()
        )
    
    def _classify_attributes(self, frame_rgb: np.ndarray) -> List[str]:
        """Classify scene attributes using CLIP"""
        image = Image.fromarray(frame_rgb)
        
        text_inputs = [f"a photo with {attr}" for attr in self.SCENE_ATTRIBUTES]
        
        inputs = self.processor(
            text=text_inputs,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
        
        # Return attributes with >50% confidence
        attributes = [
            self.SCENE_ATTRIBUTES[i]
            for i, p in enumerate(probs)
            if p > 0.5
        ]
        
        return attributes or ["indoor"]  # Default
    
    def _detect_lighting(self, frame: np.ndarray) -> str:
        """Simple lighting detection based on brightness"""
        gray = np.mean(frame, axis=2)
        avg_brightness = np.mean(gray)
        
        if avg_brightness > 150:
            return "bright"
        elif avg_brightness > 80:
            return "natural"
        else:
            return "dim"
    
    def get_room_context(self) -> str:
        """Get natural language room context"""
        if not self.current_scene:
            return "the scene"
        
        return self.current_scene.to_natural_language()
    
    def __del__(self):
        """Cleanup"""
        self.stop_background_processing()


# Singleton instance for easy access
_scene_understanding_instance: Optional[SceneUnderstanding] = None

def get_scene_understanding() -> SceneUnderstanding:
    """Get or create singleton scene understanding instance"""
    global _scene_understanding_instance
    if _scene_understanding_instance is None:
        _scene_understanding_instance = SceneUnderstanding()
    return _scene_understanding_instance
