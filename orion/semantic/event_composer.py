"""
Event Composer
===============

Composes natural language event descriptions from temporal windows using LLM.

Responsibilities:
- Generate event descriptions from temporal windows
- Query Ollama/Gemma for narrative composition
- Track composed events
- Apply quality filters

Author: Orion Research Team
Date: October 2025
"""

import logging
from typing import List, Dict, Optional
import requests
import uuid

from orion.semantic.types import TemporalWindow, Event, CausalLink
from orion.semantic.config import EventCompositionConfig

logger = logging.getLogger(__name__)


class EventComposer:
    """
    Composes natural language events from temporal windows.
    
    Uses LLM (via Ollama) to generate coherent event descriptions
    from state changes and causal relationships.
    """
    
    def __init__(self, config: EventCompositionConfig):
        """
        Initialize event composer.
        
        Args:
            config: Event composition configuration
        """
        self.config = config
        self.events: List[Event] = []
        
        logger.debug(
            f"EventComposer initialized: model={config.model}, "
            f"max_tokens={config.max_tokens}"
        )
    
    def compose_events(
        self,
        windows: List[TemporalWindow],
    ) -> List[Event]:
        """
        Compose events from temporal windows.
        
        Args:
            windows: List of temporal windows
            
        Returns:
            List of composed events
        """
        logger.info("="*80)
        logger.info("PHASE 2F: EVENT COMPOSITION")
        logger.info("="*80)
        
        if not windows:
            logger.warning("No windows to compose events from")
            return []
        
        logger.info(f"Composing events from {len(windows)} temporal windows...")
        logger.info(f"  LLM model: {self.config.model}")
        logger.info(f"  Max tokens: {self.config.max_tokens}")
        
        events = []
        
        for i, window in enumerate(windows):
            # Skip insignificant windows
            if not self._should_compose_window(window):
                logger.debug(f"  Skipping window {i+1} - insufficient activity")
                continue
            
            # Generate event description
            event = self._compose_event_for_window(window, i)
            
            if event:
                events.append(event)
        
        logger.info(f"\n✓ Composed {len(events)} events from {len(windows)} windows")
        
        if events:
            logger.info("  Event statistics:")
            avg_duration = sum(e.duration for e in events) / len(events)
            avg_participants = sum(len(e.involved_entities) for e in events) / len(events)
            logger.info(f"    Average event duration: {avg_duration:.2f}s")
            logger.info(f"    Average participants per event: {avg_participants:.1f}")
        
        logger.info("="*80 + "\n")
        
        self.events = events
        return events
    
    def _should_compose_window(self, window: TemporalWindow) -> bool:
        """
        Determine if window warrants event composition.
        
        Args:
            window: Temporal window to check
            
        Returns:
            True if window should be composed
        """
        # Check minimum number of participants
        if len(window.active_entities) < self.config.min_participants:
            return False
        
        # Check minimum duration
        if window.duration < self.config.min_duration_seconds:
            return False
        
        # Check minimum state changes
        if len(window.state_changes) < 2:
            return False
        
        return True
    
    def _compose_event_for_window(
        self,
        window: TemporalWindow,
        window_idx: int,
    ) -> Optional[Event]:
        """
        Compose event for a single window.
        
        Args:
            window: Temporal window
            window_idx: Index of window
            
        Returns:
            Composed event or None if failed
        """
        # Create prompt
        prompt = self._create_prompt(window)
        
        # Query LLM
        description = self._query_llm(prompt)
        
        if not description:
            logger.warning(f"  Failed to compose event for window {window_idx+1}")
            return None
        
        # Infer event type
        event_type = self._infer_event_type(window, description)
        
        # Create event
        event = Event(
            event_id=str(uuid.uuid4()),
            description=description,
            event_type=event_type,
            start_timestamp=window.start_time,
            end_timestamp=window.end_time,
            start_frame=int(window.start_time * 30),  # Assuming 30 FPS
            end_frame=int(window.end_time * 30),
            involved_entities=list(window.active_entities),
            causal_links=window.causal_links,
            confidence=self._compute_confidence(window),
        )
        
        logger.debug(f"  Composed event {window_idx+1}: {event.event_type} - {description[:60]}...")
        
        return event
    
    def _create_prompt(self, window: TemporalWindow) -> str:
        """
        Create LLM prompt for event composition.
        
        Args:
            window: Temporal window
            
        Returns:
            Formatted prompt
        """
        prompt = f"""Describe the event that occurred in this time window ({window.start_time:.1f}s - {window.end_time:.1f}s).

Entities involved: {', '.join(window.active_entities)}

State changes ({len(window.state_changes)}):
"""
        
        for i, change in enumerate(window.state_changes[:5], 1):  # Limit to 5 changes
            prompt += f"{i}. {change.entity_id}: {change.description_before} → {change.description_after}\n"
        
        if window.causal_links:
            prompt += f"\nCausal relationships ({len(window.causal_links)}):\n"
            for link in window.top_causal_links(3):  # Top 3 links
                prompt += f"- {link.agent_id} influenced {link.patient_id} (score: {link.influence_score:.2f})\n"
        
        prompt += "\nProvide a concise 1-2 sentence description of what happened. Focus on the key actions and interactions."
        
        return prompt
    
    def _query_llm(self, prompt: str) -> str:
        """
        Query Ollama LLM.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated description
        """
        ollama_url = "http://localhost:11434/api/generate"
        
        try:
            payload = {
                "model": self.config.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                },
            }
            
            response = requests.post(
                ollama_url,
                json=payload,
                timeout=self.config.timeout_seconds,
            )
            
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return ""
        
        except requests.exceptions.ConnectionError:
            logger.error(f"Could not connect to Ollama at {ollama_url}")
            return ""
        except Exception as e:
            logger.error(f"Error querying LLM: {e}")
            return ""
    
    def _infer_event_type(self, window: TemporalWindow, description: str) -> str:
        """
        Infer event type from window and description.
        
        Args:
            window: Temporal window
            description: Event description
            
        Returns:
            Event type string
        """
        description_lower = description.lower()
        
        # Simple keyword-based classification
        if any(word in description_lower for word in ["move", "walk", "run", "travel"]):
            return "motion"
        elif any(word in description_lower for word in ["interact", "touch", "pick", "place"]):
            return "interaction"
        elif any(word in description_lower for word in ["appear", "enter", "arrive"]):
            return "appearance"
        elif any(word in description_lower for word in ["disappear", "leave", "exit"]):
            return "disappearance"
        else:
            return "state_change"
    
    def _compute_confidence(self, window: TemporalWindow) -> float:
        """
        Compute confidence score for event.
        
        Args:
            window: Temporal window
            
        Returns:
            Confidence score [0, 1]
        """
        # Base confidence on window properties
        confidence = 0.5
        
        # More state changes = higher confidence
        if len(window.state_changes) >= 5:
            confidence += 0.2
        elif len(window.state_changes) >= 3:
            confidence += 0.1
        
        # Causal links increase confidence
        if window.causal_links:
            avg_cis = sum(link.influence_score for link in window.causal_links) / len(window.causal_links)
            confidence += 0.3 * avg_cis
        
        return min(1.0, confidence)
