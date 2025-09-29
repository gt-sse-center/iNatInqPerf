"""Milvus adaptor"""

from collections.abc import Sequence

import numpy as np
from loguru import logger

from inatinqperf.adaptors.base import VectorDatabase
from inatinqperf.adaptors.Metric import Metric




"""
class VectorDatabase(ABC):
    """Abstract base class for a vector database."""

    @abstractmethod
    def train_index(self, x_train: np.ndarray) -> None:
        """Train the index with given training vectors, if needed."""

    @abstractmethod
    def drop_index(self) -> None:
        """Drop the index."""

    @abstractmethod
    def upsert(self, ids: np.ndarray, x: np.ndarray) -> None:
        """Upsert vectors with given IDs."""

    @abstractmethod
    def search(self, q: np.ndarray, topk: int, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """Search for top-k nearest neighbors."""

    @abstractmethod
    def delete(self, ids: Sequence[int]) -> None:
        """Delete vectors with given IDs."""

    @abstractmethod
    def stats(self) -> dict[str, object]:
        """Return index statistics."""
"""

class Milvus(VectorDatabase):
    """Milvus vector database."""

    def __init__(self, dim: int, metric: str = "ip"):
        super().__init__()
        self.collection = None
        self.dim = dim
        self.metric = metric
    
    def upsert(self, ids: np.ndarray, x: np.ndarray):
        """Upsert vectors with given IDs."""
        self.index.insert(x, ids)

    def delete(self, ids: Sequence[int]):
        """Delete vectors with given IDs."""
        self.index.delete(ids)
    
    def drop_index(self):
        """Drop the index."""
        self.index = None

    def stats(self):
        """Return index statistics."""
        return {
            "ntotal": int(self.index.ntotal),
            "kind": "milvus",
            "metric": self.metric,
        }
    
    def search(self, q: np.ndarray, topk: int, **kwargs):
        """Search for top-k nearest neighbors."""
        return self.index.search(q, topk)
    
    def train_index(self, x_train: np.ndarray):
        """Train the index with given vectors."""
        self.index.train(x_train)
    
        
        
        