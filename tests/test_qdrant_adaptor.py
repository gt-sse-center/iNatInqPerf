"""Tests for the Qdrant vector database adaptor class."""

import numpy as np
import docker
import pytest

from inatinqperf.adaptors.metric import Metric
from inatinqperf.adaptors.qdrant_adaptor import Qdrant


@pytest.fixture(autouse=True)
def container_fixture():
    """Start the docker container with the vector DB."""
    client = docker.from_env()
    container = client.containers.run(
        "qdrant/qdrant",
        ports={"6333": "6333"},
        remove=True,
        detach=True,  # enabled so we get back a Container object
    )

    yield container
    container.stop()


@pytest.fixture(name="collection_name")
def collection_name_fixture():
    """Return the collection name for the vector database."""
    return "test_collection"


def test_constructor(collection_name):
    vectordb = Qdrant(dim=1024, metric=Metric.INNER_PRODUCT, url="localhost", collection_name=collection_name)
    assert vectordb.client.collection_exists(collection_name)


def test_constructor_different_metrics():
    for metric in (Metric.INNER_PRODUCT, Metric.COSINE, Metric.L2, Metric.MANHATTAN):
        vdb = Qdrant(dim=1024, metric=metric, url="localhost", collection_name=str(metric))
        assert vdb.client.collection_exists(str(metric))


def test_constructor_invalid_metric(collection_name):
    with pytest.raises(ValueError):
        Qdrant(dim=1024, metric="INVALID", url="localhost", collection_name=collection_name)


def test_upsert(collection_name):
    dim = 1024
    N = 300

    vectordb = Qdrant(dim=dim, metric=Metric.INNER_PRODUCT, url="localhost", collection_name=collection_name)

    rng = np.random.default_rng(117)
    ids = rng.integers(low=0, high=10**6, size=N).tolist()
    x = rng.random(size=(N, dim))

    vectordb.upsert(ids, x)

    count_result = vectordb.client.count(collection_name=collection_name, exact=True)

    assert count_result.count == N


def test_search(collection_name):
    dim = 1024
    N = 300

    vectordb = Qdrant(dim=dim, metric=Metric.COSINE, url="localhost", collection_name=collection_name)

    rng = np.random.default_rng(117)
    ids = rng.integers(low=0, high=10**6, size=N).tolist()
    x = rng.random(size=(N, dim))

    vectordb.upsert(ids, x)

    expected_id = ids[117]
    query = x[117]
    results = vectordb.search(q=query, topk=5)

    assert results[0].id == expected_id
    assert results[0].score == 1.0
    assert results[0].payload is None
    assert results[0].vector is None

    assert np.allclose(results[1].score, 0.7728137)


def test_delete(collection_name):
    dim = 1024
    N = 300

    vectordb = Qdrant(dim=dim, metric=Metric.COSINE, url="localhost", collection_name=collection_name)

    rng = np.random.default_rng(117)
    ids = rng.integers(low=0, high=10**6, size=N).tolist()
    x = rng.random(size=(N, dim))

    vectordb.upsert(ids, x)

    ids_to_delete = ids[117:118]
    vectordb.delete(ids_to_delete)

    assert vectordb.client.count(collection_name=collection_name).count == N - 1

    query = x[117]
    results = vectordb.search(q=query, topk=5)

    # We deleted the vector we are querying
    # so the score should not be 1.0
    assert results[0].score != 1.0
    assert results[0].payload is None
    assert results[0].vector is None


def test_drop_index(collection_name):
    dim = 1024
    N = 300

    vectordb = Qdrant(dim=dim, metric=Metric.COSINE, url="localhost", collection_name=collection_name)

    rng = np.random.default_rng(117)
    ids = rng.integers(low=0, high=10**6, size=N).tolist()
    x = rng.random(size=(N, dim))

    vectordb.upsert(ids, x)

    vectordb.drop_index()

    assert not vectordb.client.collection_exists(collection_name=collection_name)


def test_stats(collection_name):
    dim = 1024

    vectordb = Qdrant(dim=dim, metric=Metric.COSINE, url="localhost", collection_name=collection_name)

    assert vectordb.stats() == {"metric": "cosine", "m": 32, "ef_construct": 128}
