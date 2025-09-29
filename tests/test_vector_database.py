"""Tests for vector database operations."""

import numpy as np
import pytest

from inatinqperf.adaptors.base import VectorDatabase


def test_vectordb_is_abstract():
    # You should not be able to instantiate the ABC directly
    with pytest.raises(TypeError):
        _ = VectorDatabase()  # type: ignore[abstract]


def test_partial_implementation_rejected():
    class Partial(VectorDatabase):
        """Missing required abstract methods -> still abstract."""

        def __init__(self, dim: int = 0, metric: str = "", **params):
            super().__init__()

        # upsert/search/stats/drop/delete not implemented

    with pytest.raises(TypeError):
        _ = Partial()  # type: ignore[abstract]


class DummyVectorDatabase(VectorDatabase):
    """
    A minimal concrete vector database for exercising the VectorDatabase contract.
    Implements brute-force search in-memory to validate shapes & lifecycle.
    """

    def __init__(self, dim: int, metric: str, **params):
        super().__init__()

        assert isinstance(dim, int) and dim > 0
        self._dim = dim
        self._metric = metric.lower()
        self._X = np.zeros((0, dim), dtype=np.float32)
        self._ids = np.zeros((0,), dtype=np.int64)
        self._dropped = False

    def train_index(self, x_train: np.ndarray):
        # Should be a no-op for this dummy; just validate dims
        assert self._dim == x_train.shape[1]

    def upsert(self, ids: np.ndarray, x: np.ndarray):
        assert self._dim == x.shape[1]
        assert ids.shape[0] == x.shape[0]
        # Remove any existing ids first (upsert semantics)
        mask_keep = np.isin(self._ids, ids, invert=True)
        self._ids = self._ids[mask_keep]
        self._X = self._X[mask_keep]
        # Append new
        self._ids = np.concatenate([self._ids, ids.astype(np.int64)])
        self._X = np.vstack([self._X, x.astype(np.float32)])

    def _sim(self, Q: np.ndarray, X: np.ndarray) -> np.ndarray:
        if self._metric in ("ip", "inner_product", "dot", "cosine"):
            # cosine handled by normalized vectors; for simplicity treat as dot here
            return Q @ X.T
        elif self._metric in ("l2", "euclidean"):
            # Convert L2 distance to a "higher is better" score
            # score = -||q - x||^2 = - (q^2 + x^2 - 2 q·x)
            q2 = np.sum(Q**2, axis=1, keepdims=True)
            x2 = np.sum(X**2, axis=1, keepdims=True).T
            return -(q2 + x2 - 2.0 * (Q @ X.T))
        else:
            raise ValueError(f"unsupported metric {self._metric}")

    def search(self, q: np.ndarray, topk: int, **kwargs):
        assert self._X is not None and self._ids is not None
        assert q.shape[1] == self._X.shape[1]
        sims = self._sim(q.astype(np.float32), self._X)
        # argsort descending (higher score = better)
        idx = np.argpartition(-sims, kth=min(topk - 1, sims.shape[1] - 1), axis=1)[:, :topk]
        # sort each row fully
        row_indices = np.arange(q.shape[0])[:, None]
        top_scores = sims[row_indices, idx]
        order = np.argsort(-top_scores, axis=1)
        idx_sorted = idx[row_indices, order]
        D = sims[row_indices, idx_sorted].astype(np.float32)
        I = self._ids[idx_sorted].astype(np.int64)
        return D, I

    def stats(self):
        return {
            "ntotal": int(self._X.shape[0]) if self._X is not None else 0,
            "dim": int(self._dim) if self._dim is not None else None,
            "metric": self._metric,
            "dropped": bool(self._dropped),
        }

    def drop_index(self):
        self._X = None
        self._ids = None
        self._dropped = True

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
    return ids, X


@pytest.mark.parametrize("metric", ["ip", "l2"])
def test_lifecycle_and_shapes(metric, tiny_dataset):
    ids, X = tiny_dataset

    db = DummyVectorDatabase(dim=2, metric=metric)

    # train should be a no-op but must accept input
    db.train_index(X)

    # upsert vectors
    db.upsert(ids, X)

    # search with two queries
    Q = np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
    topk = 3
    D, I = db.search(Q, topk=topk)

    # Shape checks
    assert D.shape == (Q.shape[0], topk)
    assert I.shape == (Q.shape[0], topk)
    assert D.dtype == np.float32
    assert I.dtype == np.int64

    # Basic correctness checks (nearest to [1,0] should include id 10)
    assert 10 in I[0]

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

    # drop should release data
    db.drop_index()
    assert db.stats()["dropped"] is True


def test_upsert_replaces_existing(tiny_dataset):
    ids, X = tiny_dataset
    db = DummyVectorDatabase(dim=2, metric="ip")
    db.upsert(ids, X)

    # Upsert same ids with shifted vectors
    X2 = X + 1.0
    db.upsert(ids, X2)

    # Should contain exactly len(ids) vectors (deduped), not doubled
    assert db.stats()["ntotal"] == len(ids)

    # Query should reflect updated vectors
    Q = np.array([[2.0, 1.0]], dtype=np.float32)
    _, I = db.search(Q, topk=1)
    assert I.shape == (1, 1)
    # With IP and shifted vectors, id 12 ([2,2]) should db the best match
    assert I[0, 0] in ids
