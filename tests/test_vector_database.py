"""Tests for vector database operations."""

import numpy as np
from collections.abc import Sequence
import pytest

from inatinqperf.adaptors.base import VectorDatabase, DataPoint, Query, SearchResult, HuggingFaceDataset
from inatinqperf.adaptors.enums import Metric


def test_vectordb_is_abstract():
    # You should not be able to instantiate the ABC directly
    with pytest.raises(TypeError):
        VectorDatabase()  # type: ignore[abstract]


def test_partial_implementation_rejected():
    class Partial(VectorDatabase):
        """Missing required abstract methods -> still abstract."""

        def __init__(self, dim: int = 0, metric: str = "", **params):
            super().__init__()

        # upsert/search/stats/drop/delete not implemented

    with pytest.raises(TypeError):
        _ = Partial()  # type: ignore[abstract]


class ConcreteVectorDatabase(VectorDatabase):
    """
    A minimal concrete vector database for exercising the VectorDatabase contract.
    Implements brute-force search in-memory to validate shapes & lifecycle.
    """

    def __init__(self, dataset, metric: str, **params):
        super().__init__(dataset, metric)

        dataset = dataset.with_format("numpy")
        dim = dataset["embedding"].shape[1]

        assert isinstance(dim, int) and dim > 0
        self._dim = dim
        self._metric = self._translate_metric(metric)

        # "Upsert" dataset
        self._X = dataset["embedding"]
        self._ids = dataset["id"]

    @staticmethod
    def _translate_metric(metric: Metric) -> str:
        return metric.lower()

    def train_index(self, x_train: np.ndarray):
        # Should be a no-op for this dummy; just validate dims
        assert self._dim == x_train.shape[1]

    def upsert(self, x: Sequence[DataPoint]):
        ids = np.asarray([d.id for d in x], dtype=np.int64)
        vecs = np.asarray([d.vector for d in x], dtype=np.float32)

        # Remove any existing ids first (upsert semantics)
        mask_keep = np.isin(self._ids, ids, invert=True)
        self._ids = self._ids[mask_keep]
        self._X = self._X[mask_keep]

        # Append new
        self._ids = np.concatenate([self._ids, ids])
        self._X = np.vstack([self._X, vecs])

    def _similarity(self, q: np.ndarray, X: np.ndarray) -> np.ndarray:
        if self._metric in ("ip", "inner_product", "dot", "cosine"):
            # cosine handled by normalized vectors
            normalizer = np.linalg.norm(X, axis=1) * np.linalg.norm(q)
            return np.divide(X.dot(q), normalizer)

        elif self._metric in ("l2", "euclidean"):
            # Convert L2 distance to a similarity score (higher is better)
            # score = -||q - x||^2 = - (q^2 + x^2 - 2 q·x)
            return -np.power(np.linalg.norm(X - q, axis=1, keepdims=True), 2).squeeze()
        else:
            raise ValueError(f"unsupported metric {self._metric}")

    def search(self, q: Query, topk: int, **kwargs):
        assert self._X is not None and self._ids is not None

        query_vector = np.asarray(q.vector, dtype=np.float32)
        assert query_vector.shape[0] == self._X.shape[1]

        scores = self._similarity(query_vector, self._X)

        # argsort descending (higher score = better)
        descending_sorted_idxs = np.argsort(scores)[::-1]

        # Get topk
        idxs = descending_sorted_idxs[:topk]

        return [SearchResult(id=self._ids[idx], score=scores[idx]) for idx in idxs]

    def stats(self):
        return {
            "ntotal": int(self._X.shape[0]) if self._X is not None else 0,
            "dim": int(self._dim) if self._dim is not None else None,
            "metric": self._metric,
        }

    def delete(self, ids):
        ids = np.asarray(list(ids), dtype=np.int64)
        keep = np.isin(self._ids, ids, invert=True)
        self._ids = self._ids[keep]
        self._X = self._X[keep]


@pytest.fixture(name="tiny_dataset")
def tiny_dataset_fixture():
    # 4 points in 2D, easy to reason about
    X = np.array(
        [
            [1.0, 0.0],  # id 10
            [0.0, 1.0],  # id 11
            [1.0, 1.0],  # id 12
            [-1.0, 0.0],  # id 13
        ],
        dtype=np.float32,
    )
    ids = np.array([10, 11, 12, 13], dtype=np.int64)

    data_dict = {"embedding": X, "id": ids}
    return HuggingFaceDataset.from_dict(data_dict)


@pytest.mark.parametrize("metric", [Metric.INNER_PRODUCT, Metric.L2])
def test_lifecycle_and_shapes(metric, tiny_dataset):
    db = ConcreteVectorDatabase(tiny_dataset, metric=metric)

    # search with two queries
    q = Query([1.0, 0.0])
    topk = 3
    results = db.search(q, topk=topk)

    # Shape checks
    assert len(results) == topk

    # Basic correctness checks (nearest to [1,0] should be id 10)
    assert results[0].id == 10

    # stats must be a dict with expected keys
    st = db.stats()
    assert isinstance(st, dict)
    assert st["ntotal"] == 4
    assert st["dim"] == 2
    assert st["metric"] == metric

    # delete should remove specified ids
    db.delete([11, 13])
    st2 = db.stats()
    assert st2["ntotal"] == 2


def test_invalid_delete(tiny_dataset):
    db = ConcreteVectorDatabase(tiny_dataset, metric=Metric.INNER_PRODUCT)

    # id=101 does not exist in tiny_dataset.
    db.delete([101])

    # There should be no change in the size of the database.
    st = db.stats()
    assert st["ntotal"] == 4


def test_upsert_replaces_existing(tiny_dataset):
    db = ConcreteVectorDatabase(tiny_dataset, metric=Metric.INNER_PRODUCT)

    # Upsert same ids with shifted vectors
    tiny_dataset = tiny_dataset.with_format("numpy")
    ids = tiny_dataset["id"]
    X = tiny_dataset["embedding"]
    X2 = X + 1.0
    data_points = [DataPoint(id=i, vector=v, metadata={}) for i, v in zip(ids, X2)]
    db.upsert(data_points)

    # Should contain exactly len(ids) vectors (deduped), not doubled
    assert db.stats()["ntotal"] == len(ids)

    # Query should reflect updated vectors
    q = Query([2.0, 1.0])
    results = db.search(q, topk=1)
    assert len(results) == 1
