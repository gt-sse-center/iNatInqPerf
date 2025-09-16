"""Abstract base class for vector database backends."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Any
from collections.abc import Sequence


class VectorBackend(ABC):
    """Abstract base class for vector database backends."""

    name: str

    @abstractmethod
    def init(self, dim: int, metric: str, **params) -> None:
        """Initialize the index with given dimension and metric."""

    @abstractmethod
    def train(self, x_train: np.ndarray) -> None:
        """Train the index with given training vectors, if needed."""

    @abstractmethod
    def upsert(self, ids: np.ndarray, x: np.ndarray) -> None:
        """Upsert vectors with given IDs."""
        ...

    @abstractmethod
    def search(self, q: np.ndarray, topk: int, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """Initialize the index with given dimension and metric."""
        ...

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        """Return index statistics."""
        ...

    @abstractmethod
    def drop(self) -> None:
        """Drop the index."""
        ...

    @abstractmethod
    def delete(self, ids: Sequence[int]) -> None:
        """Delete vectors with given IDs."""
        ...
