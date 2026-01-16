"""
Query Intelligence for ORION Spatial Historian

Understands user intent, expands synonyms, and applies contextual reasoning
to provide helpful, conversational responses.

Features:
- Intent classification (location, count, existence, state)
- Synonym expansion (laptop = computer, notebook, macbook)
- Temporal reasoning ("where did I leave..." = last known location)
- Context-aware retrieval (smart entity matching)
"""

import re
from typing import Dict, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum


class QueryIntent(Enum):
    """Types of user queries"""
    OBJECT_LOCATION = "object_location"  # "where is X?"
    OBJECT_COUNT = "object_count"  # "how many X?"
    OBJECT_EXISTENCE = "object_existence"  # "is there X?"
    OBJECT_STATE = "object_state"  # "is the door open?"
    SCENE_DESCRIPTION = "scene_description"  # "what do you see?"
    SPATIAL_RELATIONSHIP = "spatial_relationship"  # "what's near X?"
    TEMPORAL_QUERY = "temporal_query"  # "where did I leave X?"


@dataclass
class ParsedQuery:
    """Structured query information"""
    raw_query: str
    intent: QueryIntent
    target_objects: list[str]  # Main objects of interest
    keywords: list[str]  # All relevant keywords
    temporal_context: Optional[str]  # "last", "currently", "earlier"
    requires_history: bool  # Need temporal data?
    synonyms: dict[str, list[str]]  # Expanded synonyms


class QueryIntelligence:
    """
    Intelligent query understanding and entity retrieval
    
    Makes the system feel like a helpful assistant that understands
    what you're really asking for, even with imperfect phrasing.
    """
    
    # Synonym mappings for common objects
    OBJECT_SYNONYMS = {
        'laptop': ['computer', 'notebook', 'macbook', 'laptop computer', 'pc'],
        'phone': ['mobile', 'smartphone', 'cell phone', 'cellphone', 'iphone', 'android'],
        'keys': ['key', 'keychain', 'car keys', 'house keys'],
        'remote': ['remote control', 'tv remote', 'controller'],
        'book': ['textbook', 'novel', 'magazine', 'reading material'],
        'glasses': ['eyeglasses', 'spectacles', 'reading glasses', 'sunglasses'],
        'wallet': ['billfold', 'purse', 'money clip'],
        'bag': ['backpack', 'purse', 'handbag', 'tote', 'satchel', 'luggage'],
        'cup': ['mug', 'glass', 'tumbler', 'coffee cup', 'tea cup'],
        'bottle': ['water bottle', 'drink', 'beverage'],
        'charger': ['power adapter', 'cable', 'charging cable', 'usb cable'],
        'tv': ['television', 'monitor', 'screen', 'display'],
        'couch': ['sofa', 'loveseat', 'sectional'],
        'chair': ['seat', 'armchair', 'recliner', 'stool'],
        'table': ['desk', 'counter', 'surface', 'coffee table', 'dining table'],
    }
    
    # Intent patterns (regex patterns for each intent type)
    INTENT_PATTERNS = {
        QueryIntent.OBJECT_LOCATION: [
            r'\b(where|location|find|locate|spot)\b.*\b(is|are|was|were|the|my|a|an)\b',
            r'\b(is|are)\s+(?:the|my|a|an)?\s*\w+\b',
        ],
        QueryIntent.OBJECT_COUNT: [
            r'\b(how many|count|number of)\b',
            r'\b(\d+)\s+\w+\b',
        ],
        QueryIntent.OBJECT_EXISTENCE: [
            r'\b(is there|are there|do you see|can you see|any)\b',
            r'\b(do|does)\s+(?:the|you)?\s*(see|have)\b',
        ],
        QueryIntent.SCENE_DESCRIPTION: [
            r'\b(what|describe|tell me about|show me)\b.*\b(scene|room|space|area|see|visible)\b',
            r'\b(what.*(?:see|visible|here|there))\b',
        ],
        QueryIntent.SPATIAL_RELATIONSHIP: [
            r'\b(near|next to|beside|around|close to|by)\b',
            r'\bwhat.*\b(near|around|by|next to)\b',
        ],
        QueryIntent.TEMPORAL_QUERY: [
            r'\b(did I|where did|when did|last|leave|left|put|placed)\b',
            r'\b(was|were).*\b(earlier|before|ago|last|previously)\b',
        ],
    }
    
    # Temporal keywords
    TEMPORAL_KEYWORDS = {
        'last': 'last_seen',
        'previously': 'previous',
        'earlier': 'earlier',
        'recently': 'recent',
        'just': 'just_now',
        'ago': 'past',
        'leave': 'last_location',
        'left': 'last_location',
        'put': 'last_location',
        'placed': 'last_location',
    }
    
    def __init__(self):
        """Initialize query intelligence"""
        # Build reverse synonym map for quick lookup
        self.reverse_synonyms = {}
        for canonical, synonyms in self.OBJECT_SYNONYMS.items():
            for synonym in synonyms:
                self.reverse_synonyms[synonym.lower()] = canonical
            self.reverse_synonyms[canonical.lower()] = canonical
    
    def parse_query(self, query: str) -> ParsedQuery:
        """
        Parse user query to understand intent and extract information
        
        Args:
            query: Raw user query string
            
        Returns:
            ParsedQuery with structured information
        """
        query_lower = query.lower()
        
        # Classify intent
        intent = self._classify_intent(query_lower)
        
        # Extract target objects
        target_objects = self._extract_objects(query_lower)
        
        # Extract all keywords
        keywords = self._extract_keywords(query_lower)
        
        # Detect temporal context
        temporal_context, requires_history = self._detect_temporal_context(query_lower)
        
        # Expand synonyms for better matching
        synonyms = self._expand_synonyms(target_objects)
        
        return ParsedQuery(
            raw_query=query,
            intent=intent,
            target_objects=target_objects,
            keywords=keywords,
            temporal_context=temporal_context,
            requires_history=requires_history,
            synonyms=synonyms
        )
    
    def _classify_intent(self, query: str) -> QueryIntent:
        """Classify query intent using pattern matching"""
        
        # Check each intent pattern
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return intent
        
        # Default: assume location query
        return QueryIntent.OBJECT_LOCATION
    
    def _extract_objects(self, query: str) -> list[str]:
        """
        Extract object names from query
        
        Handles:
        - "where is my laptop" → ["laptop"]
        - "how many chairs" → ["chair"]  
        - "is there a book" → ["book"]
        """
        objects = []
        
        # Common object-indicating words
        indicators = ['the', 'my', 'a', 'an', 'any', 'some']
        
        # Split into words
        words = query.split()
        
        for i, word in enumerate(words):
            # Check if word after indicator
            if i > 0 and words[i-1] in indicators:
                # Clean word (remove punctuation)
                clean_word = re.sub(r'[^\w\s]', '', word)
                if len(clean_word) > 2:  # Avoid very short words
                    objects.append(clean_word)
            
            # Also check if word is a known object synonym
            clean_word = re.sub(r'[^\w\s]', '', word)
            if clean_word in self.reverse_synonyms:
                canonical = self.reverse_synonyms[clean_word]
                if canonical not in objects:
                    objects.append(canonical)
        
        return objects
    
    def _extract_keywords(self, query: str) -> list[str]:
        """Extract all relevant keywords for semantic search"""
        # Remove common words
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'can', 'may', 'might', 'must',
            'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'my', 'your', 'his', 'her', 'its', 'our', 'their',
            'there', 'here', 'this', 'that', 'these', 'those',
            'what', 'where', 'when', 'why', 'how'
        }
        
        # Extract words
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords
    
    def _detect_temporal_context(self, query: str) -> tuple[Optional[str], bool]:
        """
        Detect temporal context in query
        
        Returns: (temporal_context, requires_history)
        """
        for keyword, context in self.TEMPORAL_KEYWORDS.items():
            if keyword in query:
                return (context, True)
        
        return (None, False)
    
    def _expand_synonyms(self, objects: list[str]) -> dict[str, list[str]]:
        """
        Expand object names to include synonyms
        
        Args:
            objects: list of object names
            
        Returns:
            dict mapping each object to its synonyms
        """
        expanded = {}
        
        for obj in objects:
            # Get canonical form
            canonical = self.reverse_synonyms.get(obj.lower(), obj.lower())
            
            # Get all synonyms
            synonyms = self.OBJECT_SYNONYMS.get(canonical, [canonical])
            
            # Add canonical form
            if canonical not in synonyms:
                synonyms = [canonical] + synonyms
            
            expanded[obj] = synonyms
        
        return expanded
    
    def get_search_terms(self, parsed_query: ParsedQuery) -> Set[str]:
        """
        Get all search terms for entity retrieval
        
        Combines target objects + synonyms + keywords
        """
        terms = set()
        
        # Add target objects
        terms.update(parsed_query.target_objects)
        
        # Add synonyms
        for obj, synonyms in parsed_query.synonyms.items():
            terms.update(synonyms)
        
        # Add keywords
        terms.update(parsed_query.keywords)
        
        return terms
    
    def format_response_context(self, parsed_query: ParsedQuery) -> str:
        """
        Generate context string for LLM based on query intent
        
        This helps the LLM understand what kind of response is expected
        """
        intent_contexts = {
            QueryIntent.OBJECT_LOCATION: "The user wants to know the location of an object. Provide spatial context like 'on the table near the couch'.",
            QueryIntent.OBJECT_COUNT: "The user wants to count objects. Provide an exact count if possible.",
            QueryIntent.OBJECT_EXISTENCE: "The user wants to know if something exists. Answer yes/no first, then provide details.",
            QueryIntent.SCENE_DESCRIPTION: "The user wants a general scene description. Describe what's visible and the overall context.",
            QueryIntent.SPATIAL_RELATIONSHIP: "The user wants to know what's near something. Focus on nearby objects and their relationships.",
            QueryIntent.TEMPORAL_QUERY: "The user is asking about a past state. Focus on the last known location or state.",
        }
        
        return intent_contexts.get(parsed_query.intent, "")


# Singleton instance
_query_intelligence_instance: Optional[QueryIntelligence] = None

def get_query_intelligence() -> QueryIntelligence:
    """Get or create singleton query intelligence instance"""
    global _query_intelligence_instance
    if _query_intelligence_instance is None:
        _query_intelligence_instance = QueryIntelligence()
    return _query_intelligence_instance
