"""
Embedding Model Wrapper
Supports both Sentence Transformers and Ollama EmbeddingGemma
"""

import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import both options
SENTENCE_TRANSFORMERS_AVAILABLE = False
OLLAMA_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    pass


class EmbeddingModel:
    """
    Unified embedding model interface
    Supports both Sentence Transformers and Ollama embeddings
    """
    
    def __init__(self, model_type: str = 'embeddinggemma', fallback: str = 'all-MiniLM-L6-v2'):
        """
        Initialize embedding model
        
        Args:
            model_type: 'embeddinggemma' (via Ollama) or 'sentence-transformer'
            fallback: Sentence transformer model to use if Ollama unavailable
        """
        self.model_type = model_type
        self.fallback = fallback
        self.model = None
        self.embedding_dim = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model"""
        
        # Try EmbeddingGemma first (preferred)
        if self.model_type == 'embeddinggemma' and OLLAMA_AVAILABLE:
            try:
                # Test if model is available
                test_response = ollama.embeddings(
                    model='embeddinggemma',
                    prompt='test'
                )
                self.embedding_dim = len(test_response['embedding'])
                self.model_type = 'embeddinggemma'
                logger.info(f"✓ Using EmbeddingGemma (dim={self.embedding_dim})")
                return
            except Exception as e:
                logger.warning(f"EmbeddingGemma not available: {e}")
                logger.info("Falling back to sentence transformers...")
        
        # Fallback to Sentence Transformers
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(self.fallback)
                self.model_type = 'sentence-transformer'
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                logger.info(f"✓ Using {self.fallback} (dim={self.embedding_dim})")
                return
            except Exception as e:
                logger.error(f"Failed to load sentence transformer: {e}")
        
        # No models available
        logger.error("No embedding models available!")
        logger.error("Install: pip install sentence-transformers OR ollama pull embeddinggemma")
        raise RuntimeError("No embedding models available")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings
        
        Args:
            texts: List of text strings
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if self.model_type == 'embeddinggemma':
            return self._encode_ollama(texts)
        else:
            return self._encode_sentence_transformer(texts)
    
    def _encode_ollama(self, texts: List[str]) -> np.ndarray:
        """Encode using Ollama EmbeddingGemma"""
        embeddings = []
        
        for text in texts:
            response = ollama.embeddings(
                model='embeddinggemma',
                prompt=text
            )
            embeddings.append(response['embedding'])
        
        return np.array(embeddings)
    
    def _encode_sentence_transformer(self, texts: List[str]) -> np.ndarray:
        """Encode using Sentence Transformers"""
        return self.model.encode(texts, convert_to_numpy=True)
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        embeddings = self.encode([text1, text2])
        
        # Cosine similarity
        dot_product = np.dot(embeddings[0], embeddings[1])
        norm_product = np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        
        return float(dot_product / norm_product)
    
    def get_embedding_dimension(self) -> int:
        """Return embedding dimension"""
        return self.embedding_dim
    
    def get_model_info(self) -> dict:
        """Return model information"""
        return {
            'type': self.model_type,
            'dimension': self.embedding_dim,
            'model_name': 'embeddinggemma' if self.model_type == 'embeddinggemma' else self.fallback
        }


# Convenience function for backward compatibility
def create_embedding_model(prefer_ollama: bool = True) -> EmbeddingModel:
    """
    Create embedding model with smart defaults
    
    Args:
        prefer_ollama: If True, try EmbeddingGemma first
    
    Returns:
        EmbeddingModel instance
    """
    if prefer_ollama:
        return EmbeddingModel(model_type='embeddinggemma', fallback='all-MiniLM-L6-v2')
    else:
        return EmbeddingModel(model_type='sentence-transformer', fallback='all-MiniLM-L6-v2')


if __name__ == "__main__":
    # Test the embedding model
    print("Testing embedding models...")
    
    try:
        model = create_embedding_model(prefer_ollama=True)
        print(f"\n✓ Model loaded: {model.get_model_info()}")
        
        # Test encoding
        texts = [
            "A person is walking down the street",
            "Someone is strolling on the sidewalk",
            "A car is driving on the road"
        ]
        
        print(f"\nTesting with {len(texts)} sentences...")
        embeddings = model.encode(texts)
        print(f"✓ Embeddings shape: {embeddings.shape}")
        
        # Test similarity
        sim1 = model.compute_similarity(texts[0], texts[1])
        sim2 = model.compute_similarity(texts[0], texts[2])
        
        print(f"\nSimilarity scores:")
        print(f"  '{texts[0]}' vs '{texts[1]}': {sim1:.3f}")
        print(f"  '{texts[0]}' vs '{texts[2]}': {sim2:.3f}")
        print(f"\n✓ Higher similarity for similar sentences: {sim1 > sim2}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTo use EmbeddingGemma:")
        print("  1. ollama pull embeddinggemma")
        print("  2. pip install ollama")
