"""Abstract base class for vector database backends."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class VectorDatabase(ABC):
    """Abstract base class for a vector database."""

    @abstractmethod
    def train(self, x_train: np.ndarray) -> None:
        """Train the index with given training vectors, if needed."""

    @abstractmethod
    def upsert(self, ids: np.ndarray, x: np.ndarray) -> None:
        """Upsert vectors with given IDs."""

    @abstractmethod
    def search(self, q: np.ndarray, topk: int, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """Search for top-k nearest neighbors."""

    @abstractmethod
    def stats(self) -> dict[str, object]:
        """Return index statistics."""

    @abstractmethod
    def drop(self) -> None:
        """Drop the index."""

    @abstractmethod
    def delete(self, ids: Sequence[int]) -> None:
        """Delete vectors with given IDs."""
