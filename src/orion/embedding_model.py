"""
Embedding Model Wrapper
Supports both Sentence Transformers and Ollama EmbeddingGemma
"""

import numpy as np
from typing import Any, List, Optional, Tuple
import logging

try:  # Ensure shared model caches are configured before downloads
    from .models import ModelManager as AssetManager
except ImportError:  # pragma: no cover
    from models import ModelManager as AssetManager  # type: ignore

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_MODEL = "embeddinggemma"
DEFAULT_SENTENCE_MODEL = "all-MiniLM-L6-v2"

_ASSET_MANAGER: Optional[AssetManager] = None


def _ensure_asset_environment() -> None:
    global _ASSET_MANAGER
    if _ASSET_MANAGER is None:
        _ASSET_MANAGER = AssetManager()

# Try to import both options
SENTENCE_TRANSFORMERS_AVAILABLE = False
OLLAMA_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:  # pragma: no cover - dependency may be missing or misconfigured
    SentenceTransformer = None  # type: ignore[assignment]

try:
    import ollama
    OLLAMA_AVAILABLE = True
except Exception:  # pragma: no cover - dependency may be missing or misconfigured
    ollama = None  # type: ignore[assignment]


class EmbeddingModel:
    """Unified embedding model interface across Ollama and Sentence Transformers."""

    def __init__(self, backend: str = "auto", model: Optional[str] = None) -> None:
        self.requested_backend = backend
        self.primary_model = model
        self.model_type: Optional[str] = None
        self.embedding_dim: Optional[int] = None
        self.model: Optional[Any] = None
        self.ollama_model: Optional[str] = None
        self.sentence_model_name: Optional[str] = None

        _ensure_asset_environment()
        self._initialize_model()

    def _initialize_model(self) -> None:
        attempts: List[Tuple[str, str]] = []
        backend_choice = (self.requested_backend or "auto").lower()
        if backend_choice == "ollama":
            attempts.append(("ollama", self.primary_model or DEFAULT_OLLAMA_MODEL))
        elif backend_choice == "sentence-transformer":
            attempts.append(("sentence-transformer", self.primary_model or DEFAULT_SENTENCE_MODEL))
        else:
            attempts.append(("ollama", self.primary_model or DEFAULT_OLLAMA_MODEL))
            attempts.append(("sentence-transformer", DEFAULT_SENTENCE_MODEL))

        errors: List[str] = []
        for backend_name, model_name in attempts:
            if backend_name == "ollama":
                success, message = self._initialize_ollama(model_name)
            else:
                success, message = self._initialize_sentence_transformer(model_name)
            if success:
                return
            errors.append(message)

        message = "No embedding models available."
        if errors:
            message = f"{message} {' '.join(msg for msg in errors if msg)}"
        logger.error(message)
        raise RuntimeError(message)

    def _initialize_ollama(self, model_name: str) -> Tuple[bool, str]:
        if not OLLAMA_AVAILABLE or ollama is None:
            return False, "Ollama client is not available."
        try:
            probe = ollama.embeddings(model=model_name, prompt="ping")
        except Exception as exc:  # noqa: BLE001
            return False, f"Ollama model '{model_name}' unavailable: {exc}"

        self.model_type = "ollama"
        self.embedding_dim = len(probe.get("embedding", [])) or None
        self.ollama_model = model_name
        if self.embedding_dim is None:
            return False, f"Ollama model '{model_name}' did not return embeddings."
        logger.info("✓ Using Ollama embeddings (%s, dim=%d)", model_name, self.embedding_dim)
        return True, ""

    def _initialize_sentence_transformer(self, model_name: str) -> Tuple[bool, str]:
        if not SENTENCE_TRANSFORMERS_AVAILABLE or SentenceTransformer is None:
            return False, "sentence-transformers is not installed."

        try:
            transformer = SentenceTransformer(model_name)
        except Exception as exc:  # noqa: BLE001
            return False, f"SentenceTransformer '{model_name}' unavailable: {exc}"

        self.model_type = "sentence-transformer"
        self.model = transformer
        self.sentence_model_name = model_name
        self.embedding_dim = transformer.get_sentence_embedding_dimension()
        logger.info("✓ Using sentence transformer %s (dim=%s)", model_name, self.embedding_dim)
        return True, ""

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        if isinstance(texts, str):
            texts = [texts]

        if self.model_type == "ollama":
            return self._encode_ollama(texts)
        if self.model_type == "sentence-transformer":
            return self._encode_sentence_transformer(texts)
        raise RuntimeError("Embedding backend is not initialized.")

    def _encode_ollama(self, texts: List[str]) -> np.ndarray:
        if not OLLAMA_AVAILABLE or ollama is None or self.ollama_model is None:
            raise RuntimeError("Ollama embedding backend is not available")
        embeddings = []
        for text in texts:
            response = ollama.embeddings(model=self.ollama_model, prompt=text)
            embeddings.append(response["embedding"])
        return np.array(embeddings)

    def _encode_sentence_transformer(self, texts: List[str]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Sentence transformer backend is not initialized")
        return self.model.encode(texts, convert_to_numpy=True)

    def compute_similarity(self, text1: str, text2: str) -> float:
        embeddings = self.encode([text1, text2])
        dot_product = np.dot(embeddings[0], embeddings[1])
        norm_product = np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        return float(dot_product / norm_product)

    def get_embedding_dimension(self) -> int:
        if self.embedding_dim is None:
            raise RuntimeError("Embedding model dimension is not initialized")
        return self.embedding_dim

    def get_model_info(self) -> dict:
        model_name = None
        if self.model_type == "ollama":
            model_name = self.ollama_model
        elif self.model_type == "sentence-transformer":
            model_name = self.sentence_model_name
        return {
            "type": self.model_type,
            "dimension": self.embedding_dim,
            "model_name": model_name,
        }


# Convenience function for backward compatibility
def create_embedding_model(
    prefer_ollama: bool = True,
    *,
    backend: Optional[str] = None,
    model: Optional[str] = None,
) -> EmbeddingModel:
    """Factory helper with backward-compatible defaults."""

    if backend is None:
        resolved_backend = "auto" if prefer_ollama else "sentence-transformer"
    else:
        resolved_backend = backend

    resolved_model = model
    if resolved_backend == "sentence-transformer" and resolved_model is None:
        resolved_model = DEFAULT_SENTENCE_MODEL
    if resolved_backend == "ollama" and resolved_model is None:
        resolved_model = DEFAULT_OLLAMA_MODEL

    return EmbeddingModel(resolved_backend, resolved_model)


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
