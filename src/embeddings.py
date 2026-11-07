"""
Embedding module using FastEmbed multilingual ONNX runtime.
Supports batch embedding, BLOB serialization, and vectorized cosine similarity.
"""
import os
import numpy as np
from typing import List, Union
from src.config import EMBEDDING_MODEL_NAME, EMBEDDING_DIM

# Ensure clean proxy configuration for httpx / HuggingFace
if "no_proxy" in os.environ and "[::1]" in os.environ.get("no_proxy", ""):
    clean = os.environ["no_proxy"].replace(",[::1]", "").replace("[::1]", "")
    os.environ["no_proxy"] = clean
    os.environ["NO_PROXY"] = clean

from fastembed import TextEmbedding


class EmbeddingEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingEngine, cls).__new__(cls)
            cls._instance._init_model()
        return cls._instance

    def _init_model(self):
        """Initializes the FastEmbed model."""
        self.model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
        self.dim = EMBEDDING_DIM

    def embed_text(self, text: str) -> np.ndarray:
        """Embeds a single text into a float32 1D numpy array."""
        clean_text = text.strip() if text else " "
        embeddings = list(self.model.embed([clean_text]))
        vec = np.array(embeddings[0], dtype=np.float32)
        # Normalize vector for faster dot-product cosine similarity
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Embeds a batch of texts efficiently."""
        clean_texts = [t.strip() if t and t.strip() else " " for t in texts]
        embeddings = list(self.model.embed(clean_texts, batch_size=batch_size))
        normalized = []
        for emb in embeddings:
            vec = np.array(emb, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            normalized.append(vec)
        return normalized

    @staticmethod
    def vector_to_blob(vec: np.ndarray) -> bytes:
        """Serializes float32 array into raw bytes for SQLite BLOB."""
        return vec.astype(np.float32).tobytes()

    @staticmethod
    def blob_to_vector(blob: bytes) -> np.ndarray:
        """Deserializes raw bytes from SQLite BLOB back to float32 numpy array."""
        return np.frombuffer(blob, dtype=np.float32)

    @staticmethod
    def compute_cosine_similarity(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
        """
        Computes cosine similarities between query_vec (1D normalized)
        and doc_matrix (2D normalized, shape [N, dim]) in a single BLAS dot product.
        Returns 1D array of similarity scores in [-1.0, 1.0].
        """
        if doc_matrix.size == 0:
            return np.array([], dtype=np.float32)
        return np.dot(doc_matrix, query_vec)


def get_embedding_engine() -> EmbeddingEngine:
    return EmbeddingEngine()
