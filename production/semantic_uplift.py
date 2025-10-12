"""
Part 2: The Semantic Uplift Engine
===================================

This module transforms raw perception logs into a structured knowledge graph by:
1. Tracking entities across time using visual embeddings (object permanence)
2. Detecting state changes through semantic analysis
3. Composing events using LLM reasoning
4. Building a queryable Neo4j knowledge graph

Author: Orion Research Team
Date: October 3, 2025
"""

import os
import sys
import time
import logging
import json
import warnings
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from neo4j import GraphDatabase, Session
from neo4j.exceptions import ServiceUnavailable, SessionExpired
import requests

# Suppress warnings
warnings.filterwarnings('ignore')

# Optional imports with graceful fallback
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("Warning: hdbscan not available. Install with: pip install hdbscan")

try:
    from embedding_model import create_embedding_model
    EMBEDDING_MODEL_AVAILABLE = True
except ImportError:
    EMBEDDING_MODEL_AVAILABLE = False
    print("Warning: embedding_model not available. Ensure embedding_model.py is present.")


# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

class Config:
    """Configuration parameters for semantic uplift engine"""
    
    # Entity Tracking (HDBSCAN)
    MIN_CLUSTER_SIZE = 3  # Minimum appearances to be tracked entity
    MIN_SAMPLES = 2
    CLUSTER_METRIC = 'euclidean'  # or 'cosine'
    CLUSTER_SELECTION_METHOD = 'eom'  # Excess of Mass
    CLUSTER_SELECTION_EPSILON = 0.15
    
    # State Change Detection
    STATE_CHANGE_THRESHOLD = 0.85  # Cosine similarity threshold
    EMBEDDING_MODEL_TYPE = 'embeddinggemma'  # 'embeddinggemma' (Ollama) or 'sentence-transformer'
    SENTENCE_MODEL_FALLBACK = 'all-MiniLM-L6-v2'  # Fallback if embeddinggemma unavailable
    
    # Temporal Windowing
    TIME_WINDOW_SIZE = 30.0  # seconds
    MIN_EVENTS_PER_WINDOW = 2  # Minimum state changes to trigger event composition
    
    # LLM Event Composition (Ollama)
    USE_LLM_COMPOSITION = True  # Enable LLM composition with gemma3:4b
    OLLAMA_API_URL = "http://localhost:11434/api/generate"
    OLLAMA_MODEL = "gemma3:4b"  # Use gemma3:4b for better Cypher generation (was gemma3:1b)
    OLLAMA_TEMPERATURE = 0.3  # More deterministic for structured output
    OLLAMA_MAX_TOKENS = 2000
    OLLAMA_TIMEOUT = 60  # seconds
    
    # Neo4j Configuration
    NEO4J_URI = "neo4j://127.0.0.1:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "orion123"  # Local Neo4j password
    NEO4J_DATABASE = "neo4j"
    MAX_CONNECTION_LIFETIME = 3600
    MAX_CONNECTION_POOL_SIZE = 50
    CONNECTION_TIMEOUT = 30
    
    # Graph Schema
    VECTOR_DIMENSIONS = 512  # For visual embeddings
    VECTOR_SIMILARITY_FUNCTION = 'cosine'
    
    # Performance
    BATCH_SIZE = 100  # For bulk operations
    LOG_LEVEL = logging.INFO
    PROGRESS_LOGGING = True


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger(name: str, level: int = Config.LOG_LEVEL) -> logging.Logger:
    """Set up a logger with consistent formatting"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

logger = setup_logger('SemanticUplift')


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Entity:
    """Represents a tracked entity across time"""
    entity_id: str
    object_class: str
    appearances: List[Dict[str, Any]] = field(default_factory=list)
    first_timestamp: float = 0.0
    last_timestamp: float = 0.0
    average_embedding: Optional[np.ndarray] = None
    
    def add_appearance(self, perception_obj: Dict[str, Any]):
        """Add an appearance of this entity"""
        self.appearances.append(perception_obj)
        
        timestamp = perception_obj['timestamp']
        if not self.first_timestamp or timestamp < self.first_timestamp:
            self.first_timestamp = timestamp
        if not self.last_timestamp or timestamp > self.last_timestamp:
            self.last_timestamp = timestamp
    
    def compute_average_embedding(self):
        """Compute average embedding from all appearances"""
        if not self.appearances:
            return
        
        embeddings = [np.array(obj['visual_embedding']) for obj in self.appearances]
        self.average_embedding = np.mean(embeddings, axis=0)
        # Normalize
        norm = np.linalg.norm(self.average_embedding)
        if norm > 0:
            self.average_embedding = self.average_embedding / norm
    
    def get_timeline(self) -> List[Dict[str, Any]]:
        """Get chronological timeline of appearances"""
        return sorted(self.appearances, key=lambda x: x['timestamp'])


@dataclass
class StateChange:
    """Represents a detected state change for an entity"""
    entity_id: str
    timestamp_before: float
    timestamp_after: float
    description_before: str
    description_after: str
    similarity_score: float
    change_magnitude: float  # 1 - similarity
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'entity_id': self.entity_id,
            'timestamp_before': self.timestamp_before,
            'timestamp_after': self.timestamp_after,
            'description_before': self.description_before,
            'description_after': self.description_after,
            'similarity_score': self.similarity_score,
            'change_magnitude': self.change_magnitude
        }


@dataclass
class TemporalWindow:
    """Represents a time window with active entities and state changes"""
    start_time: float
    end_time: float
    active_entities: Set[str] = field(default_factory=set)
    state_changes: List[StateChange] = field(default_factory=list)
    
    def add_state_change(self, change: StateChange):
        """Add a state change to this window"""
        self.state_changes.append(change)
        self.active_entities.add(change.entity_id)
    
    def is_significant(self) -> bool:
        """Check if window has enough activity to warrant event composition"""
        return len(self.state_changes) >= Config.MIN_EVENTS_PER_WINDOW


# ============================================================================
# MODULE 1: ENTITY TRACKING (OBJECT PERMANENCE)
# ============================================================================

class EntityTracker:
    """Tracks entities across time using visual embedding clustering"""
    
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.detection_to_entity: Dict[str, str] = {}  # temp_id -> entity_id
        
    def cluster_embeddings(self, perception_log: List[Dict[str, Any]]) -> np.ndarray:
        """
        Cluster visual embeddings using HDBSCAN
        
        Args:
            perception_log: List of perception objects
            
        Returns:
            Array of cluster labels (same length as perception_log)
        """
        if not HDBSCAN_AVAILABLE:
            logger.error("HDBSCAN not available. Cannot perform clustering.")
            # Return all -1 (noise) labels
            return np.full(len(perception_log), -1)
        
        logger.info("Extracting visual embeddings for clustering...")
        embeddings = []
        valid_indices = []
        
        for i, obj in enumerate(perception_log):
            emb = obj.get('visual_embedding')
            if emb is not None and len(emb) > 0:
                embeddings.append(np.array(emb))
                valid_indices.append(i)
        
        if not embeddings:
            logger.error("No valid embeddings found in perception log")
            return np.full(len(perception_log), -1)
        
        embeddings = np.array(embeddings)
        logger.info(f"Clustering {len(embeddings)} embeddings...")
        
        # Create HDBSCAN clusterer
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=Config.MIN_CLUSTER_SIZE,
            min_samples=Config.MIN_SAMPLES,
            metric=Config.CLUSTER_METRIC,
            cluster_selection_method=Config.CLUSTER_SELECTION_METHOD,
            cluster_selection_epsilon=Config.CLUSTER_SELECTION_EPSILON
        )
        
        # Perform clustering
        cluster_labels = clusterer.fit_predict(embeddings)
        
        # Map back to full perception log
        full_labels = np.full(len(perception_log), -1)
        for i, idx in enumerate(valid_indices):
            full_labels[idx] = cluster_labels[i]
        
        # Log clustering statistics
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        logger.info(f"Clustering complete:")
        logger.info(f"  Number of clusters (tracked entities): {n_clusters}")
        logger.info(f"  Noise points (unique objects): {n_noise}")
        logger.info(f"  Total objects: {len(embeddings)}")
        
        return full_labels
    
    def assign_entity_ids(
        self,
        perception_log: List[Dict[str, Any]],
        cluster_labels: np.ndarray
    ):
        """
        Assign entity IDs based on cluster labels
        
        Args:
            perception_log: List of perception objects
            cluster_labels: Cluster assignments from HDBSCAN
        """
        logger.info("Assigning entity IDs...")
        
        # Create entities for each cluster
        cluster_to_entity_id = {}
        noise_counter = 0
        
        for i, (obj, label) in enumerate(zip(perception_log, cluster_labels)):
            temp_id = obj.get('temp_id', f'det_{i:06d}')
            
            if label == -1:
                # Noise point - create unique entity
                entity_id = f"entity_unique_{noise_counter:06d}"
                noise_counter += 1
            else:
                # Part of cluster
                if label not in cluster_to_entity_id:
                    cluster_to_entity_id[label] = f"entity_cluster_{label:04d}"
                entity_id = cluster_to_entity_id[label]
            
            # Update perception object
            obj['entity_id'] = entity_id
            
            # Track mapping
            self.detection_to_entity[temp_id] = entity_id
            
            # Create or update entity
            if entity_id not in self.entities:
                self.entities[entity_id] = Entity(
                    entity_id=entity_id,
                    object_class=obj.get('object_class', 'unknown')
                )
            
            self.entities[entity_id].add_appearance(obj)
        
        # Compute average embeddings
        logger.info("Computing average embeddings for entities...")
        for entity in self.entities.values():
            entity.compute_average_embedding()
        
        logger.info(f"Created {len(self.entities)} entities")
        logger.info(f"  Tracked entities (clusters): {len(cluster_to_entity_id)}")
        logger.info(f"  Unique entities (noise): {noise_counter}")
    
    def track_entities(self, perception_log: List[Dict[str, Any]]):
        """
        Main entity tracking pipeline
        
        Args:
            perception_log: List of perception objects (modified in-place)
        """
        logger.info("\n" + "="*80)
        logger.info("ENTITY TRACKING - OBJECT PERMANENCE")
        logger.info("="*80)
        
        # Cluster embeddings
        cluster_labels = self.cluster_embeddings(perception_log)
        
        # Assign entity IDs
        self.assign_entity_ids(perception_log, cluster_labels)
        
        logger.info("="*80 + "\n")
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID"""
        return self.entities.get(entity_id)
    
    def get_all_entities(self) -> List[Entity]:
        """Get all tracked entities"""
        return list(self.entities.values())


# ============================================================================
# MODULE 2: STATE CHANGE DETECTION
# ============================================================================

class StateChangeDetector:
    """Detects state changes in entity descriptions over time"""
    
    def __init__(self):
        self.model = None
        self.state_changes: List[StateChange] = []
        
    def load_model(self):
        """Load embedding model (EmbeddingGemma or SentenceTransformer)"""
        if not EMBEDDING_MODEL_AVAILABLE:
            logger.error("Embedding model not available")
            return False
        
        try:
            logger.info(f"Loading embedding model: {Config.EMBEDDING_MODEL_TYPE}")
            self.model = create_embedding_model(
                prefer_ollama=(Config.EMBEDDING_MODEL_TYPE == 'embeddinggemma')
            )
            model_info = self.model.get_model_info()
            logger.info(f"✓ Model loaded: {model_info['model_name']} (type={model_info['type']}, dim={model_info['dimension']})")
            return True
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return False
    
    def compute_similarity(self, desc1: str, desc2: str) -> float:
        """
        Compute semantic similarity between two descriptions
        
        Args:
            desc1: First description
            desc2: Second description
            
        Returns:
            Cosine similarity score (0-1)
        """
        if self.model is None:
            return 1.0  # No change if model not available
        
        try:
            # Use the unified embedding model's similarity method
            return self.model.compute_similarity(desc1, desc2)
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception as e:
            logger.warning(f"Failed to compute similarity: {e}")
            return 1.0
    
    def detect_state_changes_for_entity(self, entity: Entity) -> List[StateChange]:
        """
        Detect state changes for a single entity
        
        Args:
            entity: Entity to analyze
            
        Returns:
            List of detected state changes
        """
        timeline = entity.get_timeline()
        changes = []
        
        if len(timeline) < 2:
            return changes
        
        # Compare consecutive appearances
        for i in range(len(timeline) - 1):
            curr = timeline[i]
            next_app = timeline[i + 1]
            
            curr_desc = curr.get('rich_description', '')
            next_desc = next_app.get('rich_description', '')
            
            if not curr_desc or not next_desc:
                continue
            
            # Compute similarity
            similarity = self.compute_similarity(curr_desc, next_desc)
            
            # Detect state change
            if similarity < Config.STATE_CHANGE_THRESHOLD:
                change = StateChange(
                    entity_id=entity.entity_id,
                    timestamp_before=curr['timestamp'],
                    timestamp_after=next_app['timestamp'],
                    description_before=curr_desc,
                    description_after=next_desc,
                    similarity_score=similarity,
                    change_magnitude=1.0 - similarity
                )
                changes.append(change)
        
        return changes
    
    def detect_all_state_changes(self, entities: List[Entity]) -> List[StateChange]:
        """
        Detect state changes for all entities
        
        Args:
            entities: List of entities to analyze
            
        Returns:
            List of all detected state changes
        """
        logger.info("\n" + "="*80)
        logger.info("STATE CHANGE DETECTION")
        logger.info("="*80)
        
        if not self.load_model():
            logger.warning("Continuing without state change detection")
            return []
        
        logger.info(f"Analyzing {len(entities)} entities for state changes...")
        
        all_changes = []
        for entity in entities:
            changes = self.detect_state_changes_for_entity(entity)
            all_changes.extend(changes)
        
        self.state_changes = all_changes
        
        logger.info(f"Detected {len(all_changes)} state changes")
        
        # Log some statistics
        if all_changes:
            magnitudes = [c.change_magnitude for c in all_changes]
            logger.info(f"  Average change magnitude: {np.mean(magnitudes):.3f}")
            logger.info(f"  Max change magnitude: {np.max(magnitudes):.3f}")
            logger.info(f"  Min change magnitude: {np.min(magnitudes):.3f}")
        
        logger.info("="*80 + "\n")
        
        return all_changes


# ============================================================================
# MODULE 3: TEMPORAL WINDOWING
# ============================================================================

def create_temporal_windows(
    state_changes: List[StateChange],
    window_size: float = Config.TIME_WINDOW_SIZE
) -> List[TemporalWindow]:
    """
    Divide state changes into temporal windows
    
    Args:
        state_changes: List of state changes
        window_size: Window size in seconds
        
    Returns:
        List of temporal windows
    """
    if not state_changes:
        return []
    
    logger.info(f"Creating temporal windows (size: {window_size}s)...")
    
    # Find time range
    all_timestamps = []
    for change in state_changes:
        all_timestamps.append(change.timestamp_before)
        all_timestamps.append(change.timestamp_after)
    
    min_time = min(all_timestamps)
    max_time = max(all_timestamps)
    
    # Create windows
    windows = []
    current_time = min_time
    
    while current_time < max_time:
        window = TemporalWindow(
            start_time=current_time,
            end_time=current_time + window_size
        )
        
        # Add state changes that fall in this window
        for change in state_changes:
            if (change.timestamp_before >= window.start_time and 
                change.timestamp_before < window.end_time):
                window.add_state_change(change)
        
        if window.state_changes:
            windows.append(window)
        
        current_time += window_size
    
    # Filter to significant windows
    significant_windows = [w for w in windows if w.is_significant()]
    
    logger.info(f"Created {len(windows)} windows, {len(significant_windows)} significant")
    
    return significant_windows


# ============================================================================
# MODULE 4: EVENT COMPOSITION (LLM REASONING)
# ============================================================================

class EventComposer:
    """Composes events from state changes using LLM reasoning"""
    
    def __init__(self):
        self.generated_queries: List[str] = []
    
    def query_ollama(self, prompt: str) -> str:
        """
        Query local Ollama instance
        
        Args:
            prompt: Prompt for the LLM
            
        Returns:
            Generated text
        """
        try:
            payload = {
                "model": Config.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": Config.OLLAMA_TEMPERATURE,
                    "num_predict": Config.OLLAMA_MAX_TOKENS
                }
            }
            
            response = requests.post(
                Config.OLLAMA_API_URL,
                json=payload,
                timeout=Config.OLLAMA_TIMEOUT
            )
            
            if response.status_code == 200:
                return response.json().get('response', '')
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return ""
        
        except requests.exceptions.ConnectionError:
            logger.error("Could not connect to Ollama. Is it running at localhost:11434?")
            return ""
        except Exception as e:
            logger.error(f"Error querying Ollama: {e}")
            return ""
    
    def create_prompt_for_window(
        self,
        window: TemporalWindow,
        entity_tracker: EntityTracker
    ) -> str:
        """
        Create structured prompt for event composition
        
        Args:
            window: Temporal window with state changes
            entity_tracker: Entity tracker for getting entity info
            
        Returns:
            Formatted prompt
        """
        prompt = f"""You are a knowledge graph builder. Given entity tracking data and state changes, generate ONLY valid Cypher queries to represent this information in Neo4j.

SCHEMA:
- Node types: :Entity, :Event, :State
- Relationship types: [:PARTICIPATED_IN], [:CHANGED_TO], [:OCCURRED_AT]
- Entity properties: id (STRING), label (STRING), embedding (LIST<FLOAT>), first_description (TEXT)
- Event properties: type (STRING), timestamp (DATETIME), description (TEXT)
- State properties: description (TEXT), timestamp (FLOAT)

TIME WINDOW: {window.start_time:.2f}s - {window.end_time:.2f}s

ACTIVE ENTITIES:
"""
        
        for entity_id in window.active_entities:
            entity = entity_tracker.get_entity(entity_id)
            if entity:
                prompt += f"- {entity_id} ({entity.object_class}): {len(entity.appearances)} appearances\n"
        
        prompt += "\nSTATE CHANGES:\n"
        
        for i, change in enumerate(window.state_changes, 1):
            prompt += f"{i}. Entity {change.entity_id} at {change.timestamp_before:.2f}s → {change.timestamp_after:.2f}s\n"
            prompt += f"   Before: {change.description_before}\n"
            prompt += f"   After: {change.description_after}\n"
            prompt += f"   Change magnitude: {change.change_magnitude:.3f}\n\n"
        
        prompt += """
Generate Cypher queries to:
1. Create or merge entity nodes with their labels
2. Create event nodes for significant state changes
3. Create relationships between entities and events

CRITICAL SYNTAX RULES:
- Use MERGE for entities to avoid duplicates
- Use MERGE for events with unique IDs
- NEVER use WHERE after SET (invalid syntax!)
- Use literal values, not parameters
- Each query must end with semicolon
- No explanations, only Cypher code

CORRECT EXAMPLES:
MERGE (e:Entity {id: 'entity_cluster_0001'}) SET e.label = 'laptop';
MERGE (ev:Event {id: 'event_001'}) SET ev.type = 'movement', ev.timestamp = datetime({epochSeconds: 5}), ev.description = 'Laptop moved across desk';
MATCH (e:Entity {id: 'entity_cluster_0001'}), (ev:Event {id: 'event_001'}) MERGE (e)-[:PARTICIPATED_IN]->(ev);

WRONG EXAMPLES (DO NOT DO THIS):
MERGE (e:Entity {id: '1'}) SET e.label = 'Laptop' WHERE e.type = 'Laptop';  // WRONG: WHERE after SET
CREATE (e:Entity {id: '1'}) SET e.label = 'Laptop';  // WRONG: Use MERGE not CREATE for entities
MERGE (e:Entity {id: '1'})  // WRONG: Missing semicolon

Now generate Cypher queries for the state changes above:
"""
        
        return prompt
    
    def parse_cypher_queries(self, llm_output: str) -> List[str]:
        """
        Parse Cypher queries from LLM output
        
        Args:
            llm_output: Raw output from LLM
            
        Returns:
            List of valid Cypher queries
        """
        queries = []
        
        lines = llm_output.strip().split('\n')
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('//') or line.startswith('#'):
                continue
            
            # Look for Cypher keywords
            if any(keyword in line.upper() for keyword in ['MERGE', 'CREATE', 'MATCH', 'SET']):
                # Ensure semicolon
                if not line.endswith(';'):
                    line += ';'
                queries.append(line)
        
        return queries
    
    def validate_cypher_queries(self, queries: List[str]) -> List[str]:
        """
        Validate Cypher queries for common syntax errors
        
        Args:
            queries: List of Cypher queries to validate
            
        Returns:
            List of valid queries (invalid ones filtered out)
        """
        valid_queries = []
        
        for query in queries:
            query_upper = query.upper()
            
            # Check for invalid patterns
            invalid = False
            
            # Pattern 1: WHERE after SET (invalid in Cypher)
            if 'SET' in query_upper and 'WHERE' in query_upper:
                set_pos = query_upper.index('SET')
                where_pos = query_upper.index('WHERE')
                if where_pos > set_pos:
                    logger.warning(f"Invalid query (WHERE after SET): {query[:50]}...")
                    invalid = True
            
            # Pattern 2: Missing required keywords
            if not any(kw in query_upper for kw in ['MERGE', 'CREATE', 'MATCH', 'SET', 'RETURN']):
                logger.warning(f"Invalid query (no valid keywords): {query[:50]}...")
                invalid = True
            
            if not invalid:
                valid_queries.append(query)
        
        return valid_queries

    
    def compose_events_for_window(
        self,
        window: TemporalWindow,
        entity_tracker: EntityTracker
    ) -> List[str]:
        """
        Compose events for a single window
        
        Args:
            window: Temporal window
            entity_tracker: Entity tracker
            
        Returns:
            List of Cypher queries
        """
        logger.info(f"Composing events for window {window.start_time:.1f}s - {window.end_time:.1f}s")
        
        # Use fallback if LLM is disabled
        if not Config.USE_LLM_COMPOSITION:
            logger.info("Using fallback query generation (LLM disabled)")
            return self.generate_fallback_queries(window, entity_tracker)
        
        # Create prompt
        prompt = self.create_prompt_for_window(window, entity_tracker)
        
        # Query LLM
        llm_output = self.query_ollama(prompt)
        
        if not llm_output:
            logger.warning("No output from LLM, generating basic queries")
            return self.generate_fallback_queries(window, entity_tracker)
        
        # Parse queries
        queries = self.parse_cypher_queries(llm_output)
        
        # Validate queries before returning
        validated_queries = self.validate_cypher_queries(queries)
        
        if not validated_queries:
            logger.warning("LLM generated invalid queries, using fallback")
            return self.generate_fallback_queries(window, entity_tracker)
        
        logger.info(f"Generated {len(validated_queries)} valid Cypher queries")
        
        return validated_queries
    
    def generate_fallback_queries(
        self,
        window: TemporalWindow,
        entity_tracker: EntityTracker
    ) -> List[str]:
        """
        Generate basic Cypher queries without LLM
        
        Args:
            window: Temporal window
            entity_tracker: Entity tracker
            
        Returns:
            List of basic Cypher queries
        """
        queries = []
        
        # Create entity nodes (skip descriptions in fallback mode to avoid syntax errors)
        for entity_id in window.active_entities:
            entity = entity_tracker.get_entity(entity_id)
            if entity:
                # Escape apostrophes and special characters in label
                safe_label = entity.object_class.replace("'", "\\'").replace('"', '\\"')
                queries.append(
                    f"MERGE (e:Entity {{id: '{entity_id}'}}) "
                    f"SET e.label = '{safe_label}';"
                )
        
        # Create event nodes for state changes (use counter to ensure unique IDs)
        for idx, change in enumerate(window.state_changes):
            event_id = f"event_{change.entity_id}_{int(change.timestamp_after)}_{idx}"
            queries.append(
                f"MERGE (ev:Event {{id: '{event_id}'}}) "
                f"SET ev.type = 'state_change', "
                f"ev.timestamp = datetime({{epochSeconds: {int(change.timestamp_after)}}}), "
                f"ev.description = 'Entity changed state';"
            )
            
            # Link entity to event
            queries.append(
                f"MATCH (e:Entity {{id: '{change.entity_id}'}}), "
                f"(ev:Event {{id: '{event_id}'}}) "
                f"MERGE (e)-[:PARTICIPATED_IN]->(ev);"
            )
        
        return queries
    
    def compose_all_events(
        self,
        windows: List[TemporalWindow],
        entity_tracker: EntityTracker
    ) -> List[str]:
        """
        Compose events for all windows
        
        Args:
            windows: List of temporal windows
            entity_tracker: Entity tracker
            
        Returns:
            List of all Cypher queries
        """
        logger.info("\n" + "="*80)
        logger.info("EVENT COMPOSITION - LLM REASONING")
        logger.info("="*80)
        
        all_queries = []
        
        for i, window in enumerate(windows, 1):
            logger.info(f"Processing window {i}/{len(windows)}...")
            queries = self.compose_events_for_window(window, entity_tracker)
            all_queries.extend(queries)
        
        self.generated_queries = all_queries
        
        logger.info(f"Generated {len(all_queries)} total Cypher queries")
        logger.info("="*80 + "\n")
        
        return all_queries


# ============================================================================
# MODULE 5: NEO4J KNOWLEDGE GRAPH INGESTION
# ============================================================================

class KnowledgeGraphBuilder:
    """Builds and manages Neo4j knowledge graph"""
    
    def __init__(self, uri: str = None, user: str = None, password: str = None):
        self.uri = uri or Config.NEO4J_URI
        self.user = user or Config.NEO4J_USER
        self.password = password or Config.NEO4J_PASSWORD
        self.driver = None
        
    def connect(self) -> bool:
        """
        Connect to Neo4j database
        
        Returns:
            True if connection successful
        """
        try:
            logger.info(f"Connecting to Neo4j at {self.uri}...")
            
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_lifetime=Config.MAX_CONNECTION_LIFETIME,
                max_connection_pool_size=Config.MAX_CONNECTION_POOL_SIZE,
                connection_timeout=Config.CONNECTION_TIMEOUT
            )
            
            # Verify connection
            self.driver.verify_connectivity()
            
            logger.info("Connected to Neo4j successfully")
            return True
        
        except ServiceUnavailable:
            logger.error(f"Could not connect to Neo4j at {self.uri}")
            logger.error("Make sure Neo4j is running and credentials are correct")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            return False
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def initialize_schema(self):
        """Initialize Neo4j schema with constraints and indexes"""
        logger.info("Initializing Neo4j schema...")
        
        with self.driver.session() as session:
            # Create constraints
            constraints = [
                "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
                "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (ev:Event) REQUIRE ev.id IS UNIQUE"
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.info(f"  Created constraint: {constraint.split('FOR')[1].split('REQUIRE')[0].strip()}")
                except Exception as e:
                    logger.warning(f"  Constraint may already exist: {e}")
            
            # Create indexes
            indexes = [
                "CREATE INDEX entity_label IF NOT EXISTS FOR (e:Entity) ON (e.label)",
                "CREATE INDEX event_timestamp IF NOT EXISTS FOR (ev:Event) ON (ev.timestamp)",
                "CREATE INDEX event_type IF NOT EXISTS FOR (ev:Event) ON (ev.type)"
            ]
            
            for index in indexes:
                try:
                    session.run(index)
                    logger.info(f"  Created index: {index.split('ON')[1].strip()}")
                except Exception as e:
                    logger.warning(f"  Index may already exist: {e}")
            
            # Create vector index (if supported)
            try:
                vector_index = f"""
                CREATE VECTOR INDEX entity_embedding IF NOT EXISTS
                FOR (e:Entity) ON (e.embedding)
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: {Config.VECTOR_DIMENSIONS},
                        `vector.similarity_function`: '{Config.VECTOR_SIMILARITY_FUNCTION}'
                    }}
                }}
                """
                session.run(vector_index)
                logger.info("  Created vector index for embeddings")
            except Exception as e:
                logger.warning(f"  Vector index not supported or already exists: {e}")
        
        logger.info("Schema initialization complete")
    
    def execute_query(self, query: str, parameters: Dict = None) -> bool:
        """
        Execute a single Cypher query
        
        Args:
            query: Cypher query
            parameters: Query parameters
            
        Returns:
            True if successful
        """
        try:
            with self.driver.session() as session:
                session.run(query, parameters or {})
            return True
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            logger.debug(f"Query: {query}")
            return False
    
    def execute_queries_batch(self, queries: List[str], batch_size: int = Config.BATCH_SIZE) -> int:
        """
        Execute queries in batches
        
        Args:
            queries: List of Cypher queries
            batch_size: Number of queries per batch
            
        Returns:
            Number of successfully executed queries
        """
        logger.info(f"Executing {len(queries)} queries in batches of {batch_size}...")
        
        successful = 0
        failed = 0
        
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            
            try:
                with self.driver.session() as session:
                    with session.begin_transaction() as tx:
                        for query in batch:
                            try:
                                tx.run(query)
                                successful += 1
                            except Exception as e:
                                failed += 1
                                logger.warning(f"Query failed: {e}")
                                logger.debug(f"Failed query: {query[:100]}...")
                        
                        tx.commit()
            
            except Exception as e:
                logger.error(f"Batch transaction failed: {e}")
                failed += len(batch)
            
            if Config.PROGRESS_LOGGING and (i + batch_size) % (batch_size * 5) == 0:
                logger.info(f"  Processed {i + batch_size}/{len(queries)} queries...")
        
        logger.info(f"Query execution complete: {successful} successful, {failed} failed")
        
        return successful
    
    def ingest_entities(self, entities: List[Entity]) -> int:
        """
        Ingest entity nodes into Neo4j
        
        Args:
            entities: List of entities
            
        Returns:
            Number of entities ingested
        """
        logger.info(f"Ingesting {len(entities)} entities...")
        
        successful = 0
        
        with self.driver.session() as session:
            for entity in entities:
                try:
                    # Prepare embedding
                    embedding = entity.average_embedding
                    if embedding is not None:
                        embedding = embedding.tolist()
                    else:
                        embedding = []
                    
                    # Get first description safely
                    first_desc = ""
                    if entity.appearances and len(entity.appearances) > 0:
                        first_appearance = entity.appearances[0]
                        if isinstance(first_appearance, dict):
                            first_desc = first_appearance.get('rich_description', '')
                        elif hasattr(first_appearance, 'rich_description'):
                            first_desc = first_appearance.rich_description
                    
                    # Create/merge entity node
                    query = """
                    MERGE (e:Entity {id: $id})
                    SET e.label = $label,
                        e.first_seen = $first_seen,
                        e.last_seen = $last_seen,
                        e.appearance_count = $appearance_count,
                        e.embedding = $embedding,
                        e.first_description = $first_description
                    """
                    
                    session.run(query, {
                        'id': entity.entity_id,
                        'label': entity.object_class,
                        'first_seen': entity.first_timestamp,
                        'last_seen': entity.last_timestamp,
                        'appearance_count': len(entity.appearances) if entity.appearances else 0,
                        'embedding': embedding,
                        'first_description': first_desc[:500] if first_desc else ""  # Truncate long descriptions
                    })
                    
                    successful += 1
                
                except Exception as e:
                    logger.error(f"Failed to ingest entity {entity.entity_id}: {e}")
        
        logger.info(f"Ingested {successful}/{len(entities)} entities")
        
        return successful
    
    def get_graph_statistics(self) -> Dict[str, int]:
        """
        Get graph statistics
        
        Returns:
            Dictionary of statistics
        """
        stats = {}
        
        with self.driver.session() as session:
            # Count nodes
            result = session.run("MATCH (e:Entity) RETURN count(e) as count")
            stats['entity_nodes'] = result.single()['count']
            
            result = session.run("MATCH (ev:Event) RETURN count(ev) as count")
            stats['event_nodes'] = result.single()['count']
            
            # Count relationships
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            stats['relationships'] = result.single()['count']
        
        return stats


# ============================================================================
# MAIN SEMANTIC UPLIFT PIPELINE
# ============================================================================

def run_semantic_uplift(
    perception_log: List[Dict[str, Any]],
    neo4j_driver = None,
    neo4j_uri: str = None,
    neo4j_user: str = None,
    neo4j_password: str = None
) -> Dict[str, Any]:
    """
    Main semantic uplift pipeline
    
    Args:
        perception_log: List of RichPerceptionObject dictionaries from Part 1
        neo4j_driver: Optional pre-configured Neo4j driver
        neo4j_uri: Neo4j URI (if driver not provided)
        neo4j_user: Neo4j username (if driver not provided)
        neo4j_password: Neo4j password (if driver not provided)
        
    Returns:
        Dictionary with uplift results and statistics
    """
    logger.info("\n" + "="*80)
    logger.info("SEMANTIC UPLIFT ENGINE - PART 2")
    logger.info("="*80)
    logger.info(f"Processing {len(perception_log)} perception objects")
    
    start_time = time.time()
    results = {
        'success': False,
        'num_entities': 0,
        'num_state_changes': 0,
        'num_windows': 0,
        'num_queries': 0,
        'graph_stats': {}
    }
    
    # Step 1: Entity Tracking
    tracker = EntityTracker()
    tracker.track_entities(perception_log)
    entities = tracker.get_all_entities()
    results['num_entities'] = len(entities)
    
    # Step 2: State Change Detection
    detector = StateChangeDetector()
    state_changes = detector.detect_all_state_changes(entities)
    results['num_state_changes'] = len(state_changes)
    
    # Step 3: Temporal Windowing
    windows = create_temporal_windows(state_changes)
    results['num_windows'] = len(windows)
    
    # Step 4: Event Composition
    composer = EventComposer()
    cypher_queries = composer.compose_all_events(windows, tracker)
    results['num_queries'] = len(cypher_queries)
    
    # Step 5: Neo4j Ingestion
    logger.info("\n" + "="*80)
    logger.info("KNOWLEDGE GRAPH INGESTION")
    logger.info("="*80)
    
    # Use provided driver or create new one
    if neo4j_driver:
        graph_builder = KnowledgeGraphBuilder()
        graph_builder.driver = neo4j_driver
    else:
        graph_builder = KnowledgeGraphBuilder(neo4j_uri, neo4j_user, neo4j_password)
        if not graph_builder.connect():
            logger.error("Failed to connect to Neo4j")
            return results
    
    try:
        # Initialize schema
        graph_builder.initialize_schema()
        
        # Ingest entities
        graph_builder.ingest_entities(entities)
        
        # Execute generated queries
        if cypher_queries:
            graph_builder.execute_queries_batch(cypher_queries)
        
        # Get final statistics
        results['graph_stats'] = graph_builder.get_graph_statistics()
        results['success'] = True
        
        logger.info("\nGraph Statistics:")
        for key, value in results['graph_stats'].items():
            logger.info(f"  {key}: {value}")
    
    finally:
        if not neo4j_driver:  # Only close if we created the connection
            graph_builder.close()
    
    elapsed_time = time.time() - start_time
    
    logger.info("\n" + "="*80)
    logger.info("SEMANTIC UPLIFT COMPLETE")
    logger.info("="*80)
    logger.info(f"Entities tracked: {results['num_entities']}")
    logger.info(f"State changes detected: {results['num_state_changes']}")
    logger.info(f"Temporal windows: {results['num_windows']}")
    logger.info(f"Cypher queries generated: {results['num_queries']}")
    logger.info(f"Total time: {elapsed_time:.2f}s")
    logger.info("="*80 + "\n")
    
    return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import json
    
    # Load perception log from Part 1
    PERCEPTION_LOG_PATH = "/Users/riddhiman.rana/Desktop/Coding/Orion/orion-research/data/testing/perception_log.json"
    
    if not os.path.exists(PERCEPTION_LOG_PATH):
        logger.error(f"Perception log not found: {PERCEPTION_LOG_PATH}")
        logger.info("Please run Part 1 first to generate a perception log")
        sys.exit(1)
    
    try:
        with open(PERCEPTION_LOG_PATH, 'r') as f:
            perception_log = json.load(f)
        
        logger.info(f"Loaded perception log with {len(perception_log)} objects")
        
        # Run semantic uplift
        results = run_semantic_uplift(
            perception_log,
            neo4j_uri=Config.NEO4J_URI,
            neo4j_user=Config.NEO4J_USER,
            neo4j_password=Config.NEO4J_PASSWORD
        )
        
        # Save results
        results_path = os.path.join(os.path.dirname(PERCEPTION_LOG_PATH), 'uplift_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {results_path}")
        
        if results['success']:
            logger.info("\n✅ Semantic uplift completed successfully!")
        else:
            logger.warning("\n⚠️ Semantic uplift completed with errors")
    
    except Exception as e:
        logger.error(f"Semantic uplift failed: {e}", exc_info=True)
        sys.exit(1)
