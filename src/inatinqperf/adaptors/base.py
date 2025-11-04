"""Abstract base class for vector database backends."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from datasets import Dataset as HuggingFaceDataset

from inatinqperf.adaptors.enums import Metric


@dataclass
class DataPoint:
    """A single data point in the dataset, which includes the embedding vector and additional metadata."""

    id: int
    vector: Sequence[float]
    metadata: dict[str, object]


@dataclass
class Query:
    """A class encapsulating the query vector and optional filters."""

    vector: Sequence[float]
    filters: object | None = None


@dataclass
class SearchResult:
    """The result of a search query.

    Contains the data point ID and the similarity score.
    """

    id: int
    score: float


class VectorDatabase(ABC):
    """Abstract base class for an adaptor to a vector database.

    This class serves as the base class for wrappers around a vector database client,
    providing a clean and consistent interface which can be used by the benchmarking code.
    """

    @abstractmethod
    def __init__(
        self,
        dataset: HuggingFaceDataset,
        metric: str | Metric,
        *args,
        **kwargs,
    ) -> None:
        """Constructor for the vector database adaptor.

        Args:
            dataset (HuggingFaceDataset): The dataset which to load to the database.
            metric (str | Metric): The distance/similarity metric to use for the vector database.
            *args (Sequence[object]): Optional positional arguments.
            **kwargs (dict[object, object]): Optional key-word arguments.
        """
        # Will raise exception if `metric` is not valid.
        self.metric = Metric(metric)
        self.dim = np.asarray(dataset["embedding"]).shape[1]

    @staticmethod
    @abstractmethod
    def _translate_metric(metric: Metric) -> str:
        """Map the metric value to a string value which is used by the vector database client."""

    @abstractmethod
    def upsert(self, x: Sequence[DataPoint]) -> None:
        """Upsert vectors with given IDs.

        Args:
            x (Sequence[DataPoint]): A sequence of `DataPoints` from the dataset.
        """

    @abstractmethod
    def search(self, q: Query, topk: int, **kwargs) -> Sequence[SearchResult]:
        """Search for top-k nearest neighbors.

        Args:
            q (Query): A single query point.
            topk (int): The number of closest results to return.
            **kwargs (dict): Additional search parameters.

        Returns:
            Sequence[SearchResult]: A list of SearchResult objects.
        """

    @abstractmethod
    def delete(self, ids: Sequence[int]) -> None:
        """Delete data points associated with IDs `ids`.

        Args:
            ids (Sequence[int]): The IDs of the data points to delete.
        """

    @abstractmethod
    def stats(self) -> dict[str, object]:
        """Return database statistics."""

    def close(self) -> None:
        """Method to perform cleanup when the adaptor is about to be deleted."""
        return

    def __del__(self) -> None:
        """Destructor method, which automatically closes any open connections."""
        self.close()
