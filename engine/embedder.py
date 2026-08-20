from __future__ import annotations
from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class StubEmbedder:
    DIM = 384

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), self.DIM), dtype=np.float32)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install chp[embeddings]"
            ) from e
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._model.get_sentence_embedding_dimension()), dtype=np.float32)
        return self._model.encode(texts, convert_to_numpy=True).astype(np.float32)
