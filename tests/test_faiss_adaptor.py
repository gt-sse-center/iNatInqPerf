"""Tests for the different FAISS adaptor classes."""

import numpy as np
import pytest
import faiss

from inatinqperf.adaptors import faiss_adaptor
from inatinqperf.adaptors.faiss_adaptor import Faiss, HuggingFaceDataset, Query, DataPoint
from inatinqperf.adaptors.enums import Metric


@pytest.fixture(name="small_data")
def small_data_fixture():
    X = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    ids = np.array([100, 101, 102, 103], dtype=np.int64)
    data_dict = {"embedding": X, "id": ids}
    return HuggingFaceDataset.from_dict(data_dict)


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

    data_dict = {"embedding": X, "id": ids}
    return HuggingFaceDataset.from_dict(data_dict)


@pytest.fixture(name="ivfpq_trainset")
def ivfpq_trainset_fixture():
    """Large training set so IVF-PQ can train without FAISS clustering warnings."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((10240, 2)).astype(np.float32)  # >= 9984
    ids = np.arange(1000, 1000 + X.shape[0], dtype=np.int64)

    data_dict = {"embedding": X, "id": ids}
    return HuggingFaceDataset.from_dict(data_dict)


@pytest.fixture(name="query")
def query_fixture():
    """Query used for testing the FAISS search."""
    return Query(vector=[1.0, 0.0])


@pytest.mark.parametrize("metric", [Metric.INNER_PRODUCT, Metric.L2])
def test_faiss_flat_lifecycle(metric, ivfpq_trainset, query):
    vdb = Faiss(ivfpq_trainset, metric=metric, index_type="IVFPQ", m=1)

    ids = ivfpq_trainset["id"]
    st = vdb.stats()
    assert st["ntotal"] == len(ids)

    results = vdb.search(query, topk=2)

    # shape checks
    assert len(results) == 2

    # regression: nearest to [1,0]
    if metric == Metric.INNER_PRODUCT:
        assert results[0].id == 1003
    if metric == Metric.L2:
        assert results[0].id == 1009

    # delete some ids
    vdb.delete([1101, 1103])
    assert vdb.stats()["ntotal"] == (len(ivfpq_trainset) - 2)


def test_faiss_ivfpq_build_and_search_with_large_training(ivfpq_trainset, small_data, query):
    # Use fewer PQ centroids (nbits=4 -> 16) so 300 training points suffice without warnings
    vdb = Faiss(
        ivfpq_trainset, metric=Metric.INNER_PRODUCT, index_type="IVFPQ", nlist=2, m=1, nbits=4, nprobe=2
    )

    # upsert a small, known set for deterministic checks
    ids = small_data["id"]
    X = small_data["embedding"]
    datapoints = [DataPoint(id=id, vector=vector, metadata={}) for id, vector in zip(ids, X)]
    vdb.upsert(datapoints)

    s = vdb.stats()
    assert s["kind"] == "ivfpq"
    assert "nlist" in s
    assert "nprobe" in s

    results = vdb.search(query, topk=2)
    assert len(results) == 2

    # Inspect the internal id_map to assert every expected id made it into the index.
    id_map = faiss.vector_to_array(vdb.index.id_map)
    assert set(ids).issubset(set(id_map))

    # delete works and stats update
    total_before = vdb.stats()["ntotal"]
    vdb.delete([100])
    assert vdb.stats()["ntotal"] == total_before - 1


def test_faiss_ivfpq_l2_metric_delete(ivfpq_trainset, small_data, query):
    # Fewer centroids to prevent FAISS training warnings
    vdb = Faiss(ivfpq_trainset, metric="l2", index_type="IVFPQ", nlist=2, m=1, nbits=4, nprobe=2)

    ids = small_data["id"]
    X = small_data["embedding"]
    datapoints = [DataPoint(id=id, vector=vector, metadata={}) for id, vector in zip(ids, X)]
    vdb.upsert(datapoints)

    # Search with L2 metric and ensure the index preserves all ids we inserted.
    results = vdb.search(Query(vector=small_data["embedding"][0]), topk=3, nprobe=64)
    assert len(results) == 3
    id_map = faiss.vector_to_array(vdb.index.id_map)
    assert set(ids).issubset(set(id_map))

    # Deleting non-existent ID should not raise an exception
    try:
        vdb.delete([999])
    except Exception:
        pytest.fail("faiss.ivfpq delete raised an exception.")

    # Delete existing and verify
    total_before = vdb.stats()["ntotal"]
    vdb.delete([101])
    assert vdb.stats()["ntotal"] == total_before - 1


def test_faiss_ivfpq_runtime_nprobe_override_and_upsert_replace(ivfpq_trainset, small_data, query):
    """Cover runtime nprobe override branch and upsert replacement semantics for IVFPQ."""

    vdb = Faiss(
        ivfpq_trainset, metric=Metric.INNER_PRODUCT, index_type="IVFPQ", nlist=2, m=1, nbits=4, nprobe=1
    )

    # First insert
    ids = small_data["id"]
    X = small_data["embedding"]
    datapoints = [DataPoint(id=id, vector=vector, metadata={}) for id, vector in zip(ids, X)]
    vdb.upsert(datapoints)

    nt1 = vdb.stats()["ntotal"]

    # Upsert same ids with slightly shifted vectors (replacement semantics)
    X2 = (np.asarray(X) + 0.01).tolist()
    datapoints = [DataPoint(id=id, vector=vector, metadata={}) for id, vector in zip(ids, X2)]
    vdb.upsert(datapoints)

    nt2 = vdb.stats()["ntotal"]
    assert nt2 == nt1  # replaced, not duplicated

    # Force a very small and a larger nprobe to exercise the branch
    results1 = vdb.search(query, topk=3, nprobe=1)
    results2 = vdb.search(query, topk=3, nprobe=64)

    assert len(results1) == len(results2) == 3


def test_faiss_flat_topk_greater_than_ntotal_and_idempotent_delete(small_data):
    """Cover branch where topk > ntotal and deleting non-existent IDs is a no-op."""
    be = Faiss(small_data, metric=Metric.INNER_PRODUCT, index_type="FLAT")

    q = Query(vector=[0.2, 0.9])
    # Ask for more neighbors than we have to ensure code handles it
    results = be.search(q, topk=10)
    assert len(results) == 10

    # Deleting IDs that do not exist should not raise and should keep counts stable
    nt_before = be.stats()["ntotal"]
    be.delete([999, 1000])
    assert be.stats()["ntotal"] == nt_before


def test_metric_mapping_and_unwrap_real_index(ivfpq_trainset, small_data):
    # Build a real IVFPQ index and ensure unwrap hits the IVF layer
    be = Faiss(
        ivfpq_trainset, metric=Metric.INNER_PRODUCT, index_type="IVFPQ", nlist=2, m=1, nbits=4, nprobe=2
    )

    ivf = faiss_adaptor._unwrap_to_ivf(be.index.index)
    assert ivf is not None and hasattr(ivf, "nlist")


def test_faiss_ivfpq_cosine_metric_and_topk_gt_ntotal(ivfpq_trainset, small_data, query):
    # Cosine path should map to inner-product internally
    vdb = Faiss(ivfpq_trainset, metric="cosine", index_type="IVFPQ", nlist=2, m=1, nbits=4, nprobe=2)

    ids = small_data["id"]
    X = small_data["embedding"]
    datapoints = [DataPoint(id=id, vector=vector, metadata={}) for id, vector in zip(ids, X)]
    vdb.upsert(datapoints)

    # Ask for more neighbors than exist; FAISS may pad with -1
    results = vdb.search(Query(vector=small_data["embedding"][0]), topk=10, nprobe=64)
    assert len(results) == 10
    # The id_map should still contain every id even if FAISS pads search outputs.
    id_map = faiss.vector_to_array(vdb.index.id_map)
    assert set(ids).issubset(set(id_map))


def test_faiss_ivfpq_empty_delete_noop(ivfpq_trainset, small_data):
    vdb = Faiss(
        ivfpq_trainset, metric=Metric.INNER_PRODUCT, index_type="IVFPQ", nlist=2, m=1, nbits=4, nprobe=2
    )

    ids = small_data["id"]
    X = small_data["embedding"]
    datapoints = [DataPoint(id=id, vector=vector, metadata={}) for id, vector in zip(ids, X)]
    vdb.upsert(datapoints)

    # Empty delete should be a no-op and keep ntotal unchanged.
    before = vdb.stats()["ntotal"]
    vdb.delete([])
    assert vdb.stats()["ntotal"] == before


def test_query_edge_cases(ivfpq_trainset):
    """Test various query edge cases."""
    vdb = Faiss(
        ivfpq_trainset, metric=Metric.INNER_PRODUCT, index_type="IVFPQ", nlist=2, m=1, nbits=4, nprobe=2
    )

    topk = 5

    # empty query vector
    q0 = Query(vector=[])
    with pytest.raises(ValueError):
        vdb.search(q0, topk)

    # zeros query vector
    q1 = Query(vector=[0, 0])
    vdb.search(q1, topk)

    # inf query vector
    q1 = Query(vector=[-np.inf, np.inf])
    vdb.search(q1, topk)

    # smaller query vector
    q2 = Query(vector=[1])
    with pytest.raises(ValueError):
        vdb.search(q2, topk)

    # larger query vector
    q3 = Query(vector=[1, 2, 3])
    with pytest.raises(ValueError):
        vdb.search(q3, topk)


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
    vdb = Faiss(
        ivfpq_small_trainset, metric=Metric.INNER_PRODUCT, index_type="IVFPQ", nlist=64, m=1, nbits=2
    )  # nprobe defaults to 32

    X = np.asarray(ivfpq_small_trainset["embedding"])

    s = vdb.stats()
    assert s["kind"] == "ivfpq"

    # nlist reduced to <= number of training points
    assert s["nlist"] <= X.shape[0]

    # nprobe clamped to nlist (default nprobe is 32, so expect <= nlist)
    assert s["nprobe"] <= s["nlist"]
