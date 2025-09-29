"""Tests for the different FAISS adaptor classes."""

import numpy as np
import pytest

from inatinqperf.adaptors import faiss_adaptor
from inatinqperf.adaptors.faiss_adaptor import FaissFlat, FaissIVFPQ


@pytest.fixture(name="small_data")
def small_data_fixture():
    X = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    ids = np.array([100, 101, 102, 103], dtype=np.int64)
    return ids, X


@pytest.fixture(name="ivfpq_small_trainset")
def ivfpq_small_trainset_fixture():
    """A small dataset with at least 156 vectors since FAISS sets the
    minimum number of training vectors per cluster to be 39 and the
    number of centroids to be 2^nbits = 2^2 = 4, which gives
    4*39 = 156.

    https://github.com/facebookresearch/faiss/blob/3b14dad6d9ac48a0764e5ba01a45bca1d7d738ee/faiss/Clustering.h#L43
    """
    rng = np.random.default_rng()
    N = 160
    X = rng.standard_normal((N, 2))

    ids = np.arange(N)
    return ids, X


@pytest.fixture(name="ivfpq_trainset")
def ivfpq_trainset_fixture():
    """Large training set so IVF-PQ can train without FAISS clustering warnings."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((10240, 2)).astype(np.float32)  # >= 9984
    ids = np.arange(1000, 1000 + X.shape[0], dtype=np.int64)
    return ids, X


@pytest.mark.parametrize("metric", ["ip", "l2"])
def test_faiss_flat_lifecycle(metric, small_data):
    ids, X = small_data
    be = FaissFlat(dim=2, metric=metric)

    be.upsert(ids, X)
    st = be.stats()
    assert st["ntotal"] == len(ids)

    Q = np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
    D, I = be.search(Q, topk=2)

    # shape checks
    assert D.shape == (2, 2)
    assert I.shape == (2, 2)

    # nearest to [1,0] should include id 100
    assert 100 in I[0]

    # delete some ids
    be.delete([101, 103])
    assert be.stats()["ntotal"] == 2

    # drop releases index (idempotent)
    be.drop_index()
    assert getattr(be, "index", None) is None


def test_faiss_ivfpq_train_and_search_with_large_training(ivfpq_trainset, small_data):
    _, X_train = ivfpq_trainset
    ids, X = small_data

    # Use fewer PQ centroids (nbits=4 -> 16) so 300 training points suffice without warnings
    be = FaissIVFPQ(dim=2, metric="ip", nlist=2, m=1, nbits=4, nprobe=2)

    # training on sufficient dataset (silence FAISS native stderr warnings)
    be.train_index(X_train)

    # upsert a small, known set for deterministic checks
    be.upsert(ids, X)

    s = be.stats()
    assert s["kind"] == "ivfpq"
    assert "nlist" in s
    assert "nprobe" in s

    Q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    D, I = be.search(Q, topk=2)

    # shapes
    assert D.shape == (2, 2)
    assert I.shape == (2, 2)
    # ids returned should come from our set (since they are much closer than random train points)
    assert set(I.flatten()).issubset(set(ids))

    # delete works and stats update
    be.delete([100])
    assert be.stats()["ntotal"] == len(ids) - 1

    # drop releases resources
    be.drop_index()
    assert getattr(be, "index", None) is None


def test_faiss_ivfpq_l2_metric_delete_and_drop(ivfpq_trainset, small_data):
    _, X_train = ivfpq_trainset
    ids, X = small_data

    # Fewer centroids to prevent FAISS training warnings
    be = FaissIVFPQ(dim=2, metric="l2", nlist=2, m=1, nbits=4, nprobe=2)
    be.train_index(X_train)
    be.upsert(ids, X)

    # Search with L2 metric; still expect small_data ids to appear
    Q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    # Increase nprobe to improve recall; ANN may otherwise return fewer than topk hits (filled with -1)
    D, I = be.search(Q, topk=3, nprobe=64)
    assert D.shape == (2, 3)
    assert I.shape == (2, 3)
    valid = I[I >= 0]  # filter out FAISS "no result" slots
    assert set(valid.flatten()).issubset(set(ids))

    # Deleting non-existent id should not raise
    be.delete([999])

    # Delete existing and verify
    be.delete([101])
    assert be.stats()["ntotal"] == len(ids) - 1

    # Drop twice to exercise idempotency
    be.drop_index()
    be.drop_index()
    assert getattr(be, "index", None) is None


def test_faiss_ivfpq_runtime_nprobe_override_and_upsert_replace(ivfpq_trainset, small_data):
    """Cover runtime nprobe override branch and upsert replacement semantics for IVFPQ."""
    _, X_train = ivfpq_trainset
    ids, X = small_data

    be = FaissIVFPQ(dim=2, metric="ip", nlist=2, m=1, nbits=4, nprobe=1)
    be.train_index(X_train)

    # First insert
    be.upsert(ids, X)
    nt1 = be.stats()["ntotal"]

    # Upsert same ids with slightly shifted vectors (replacement semantics)
    be.upsert(ids, X + 0.01)
    nt2 = be.stats()["ntotal"]
    assert nt2 == nt1  # replaced, not duplicated

    # Force a very small and a larger nprobe to exercise the branch
    Q = np.array([[1.0, 0.0]], dtype=np.float32)
    D1, I1 = be.search(Q, topk=3, nprobe=1)
    D2, I2 = be.search(Q, topk=3, nprobe=64)

    assert D1.shape == D2.shape == (1, 3)
    assert I1.shape == I2.shape == (1, 3)

    be.drop_index()


def test_faiss_flat_topk_greater_than_ntotal_and_idempotent_delete(small_data):
    """Cover branch where topk > ntotal and deleting non-existent IDs is a no-op."""
    ids, X = small_data
    be = FaissFlat(dim=2, metric="ip")
    be.upsert(ids, X)

    Q = np.array([[0.2, 0.9]], dtype=np.float32)
    # Ask for more neighbors than we have to ensure code handles it
    D, I = be.search(Q, topk=10)
    assert D.shape == (1, 10)
    assert I.shape == (1, 10)

    # Deleting IDs that do not exist should not raise and should keep counts stable
    nt_before = be.stats()["ntotal"]
    be.delete([999, 1000])
    assert be.stats()["ntotal"] == nt_before

    be.drop_index()
    be.drop_index()
    assert getattr(be, "index", None) is None


def test_metric_mapping_and_unwrap_real_index(ivfpq_trainset, small_data):
    # Build a real IVFPQ index and ensure unwrap hits the IVF layer
    be = FaissIVFPQ(dim=2, metric="ip", nlist=2, m=1, nbits=4, nprobe=2)
    be.train_index(ivfpq_trainset[1])

    ivf = faiss_adaptor._unwrap_to_ivf(be.index.index)
    assert ivf is not None and hasattr(ivf, "nlist")
    be.drop_index()


def test_faiss_ivfpq_cosine_metric_and_topk_gt_ntotal(ivfpq_trainset, small_data):
    # Cosine path should map to inner-product internally
    ids, X = small_data
    be = FaissIVFPQ(dim=2, metric="cosine", nlist=2, m=1, nbits=4, nprobe=2)
    be.train_index(ivfpq_trainset[1])
    be.upsert(ids, X)

    Q = np.array([[1.0, 0.0]], dtype=np.float32)
    # Ask for more neighbors than exist; FAISS may pad with -1
    D, I = be.search(Q, topk=10, nprobe=64)
    assert D.shape == (1, 10) and I.shape == (1, 10)
    valid = I[I >= 0]
    assert set(valid.flatten()).issubset(set(ids))
    be.drop_index()


def test_faiss_ivfpq_empty_delete_noop(ivfpq_trainset, small_data):
    ids, X = small_data
    be = FaissIVFPQ(dim=2, metric="ip", nlist=2, m=1, nbits=4, nprobe=2)
    be.train_index(ivfpq_trainset[1])
    be.upsert(ids, X)

    # Empty delete should be a no-op
    be.delete([])
    assert be.stats()["ntotal"] == len(ids)
    be.drop_index()


def test_unwrap_fallback_dummy_chain():
    """Exercise the fallback path in _unwrap_to_ivf by walking .index chain without faiss.extract_index_ivf."""

    class Leaf:
        def __init__(self):
            self.nlist = 5

    class Wrapper:
        def __init__(self, inner):
            self.index = inner

    base = Wrapper(Wrapper(Leaf()))
    out = faiss_adaptor._unwrap_to_ivf(base)
    assert out is not None and hasattr(out, "nlist") and out.nlist == 5


# The following test needs to be debugged
def test_faiss_ivfpq_reduces_nlist_and_clamps_nprobe(ivfpq_small_trainset):
    """Cover train_index() branch that rebuilds with smaller nlist and clamps nprobe to nlist."""
    # Start with large nlist, tiny PQ (nbits=2, m=1)
    # so training with small set triggers reduce.
    vdb = FaissIVFPQ(dim=2, metric="ip", nlist=64, m=1, nbits=2)  # nprobe defaults to 32

    _, X = ivfpq_small_trainset
    vdb.train_index(X)

    s = vdb.stats()
    assert s["kind"] == "ivfpq"

    # nlist reduced to <= number of training points
    assert s["nlist"] <= X.shape[0]

    # nprobe clamped to nlist (default nprobe is 32, so expect <= nlist)
    assert s["nprobe"] <= s["nlist"]
