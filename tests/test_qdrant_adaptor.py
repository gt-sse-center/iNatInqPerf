"""Tests for the Qdrant vector database adaptor class."""

import docker
import numpy as np
import pytest

from inatinqperf.adaptors.enums import Metric
from inatinqperf.adaptors.qdrant_adaptor import DataPoint, Qdrant, Query, HuggingFaceDataset


@pytest.fixture(scope="module", autouse=True)
def container_fixture():
    """Start the docker container with the vector DB."""
    client = docker.from_env()
    container = client.containers.run(
        "qdrant/qdrant",
        ports={"6333": "6333"},
        remove=True,
        detach=True,  # enabled so we don't block on this
        healthcheck={
            "test": "curl -s http://localhost:6333/healthz | grep -q 'healthz check passed' || exit 1",
            "interval": 30 * 10**9,
            "timeout": 20 * 10**9,
            "retries": 3,
        },
    )

    yield container

    container.stop()


@pytest.fixture(name="collection_name")
def collection_name_fixture():
    """Return the collection name for the vector database."""
    return "test_collection"


@pytest.fixture(name="dim")
def dim_fixture():
    """The dimension of the vectors used."""
    return 1024


@pytest.fixture(name="N")
def num_datapoints_fixture():
    """The size of the dataset."""
    return 300


@pytest.fixture(name="dataset")
def dataset_fixture(dim, N):
    rng = np.random.default_rng(117)
    ids = rng.choice(10**4, size=N, replace=False).tolist()
    x = rng.random(size=(N, dim))

    # Create HuggingFace dataset
    dataset = HuggingFaceDataset.from_dict({"id": ids, "embedding": x.tolist()})

    return dataset


@pytest.fixture(name="vectordb")
def vectordb_fixture(dataset, collection_name):
    """Return an instance of the Qdrant vector database."""

    vectordb = Qdrant(dataset=dataset, metric=Metric.COSINE, url="localhost", collection_name=collection_name)

    yield vectordb

    vectordb.delete_collection()


@pytest.mark.parametrize("metric", [Metric.INNER_PRODUCT, Metric.COSINE, Metric.L2, Metric.MANHATTAN])
def test_constructor(dataset, collection_name, metric):
    vectordb = Qdrant(dataset, metric=metric, url="localhost", collection_name=collection_name)
    assert vectordb.client.collection_exists(collection_name)
    vectordb.delete_collection()


def test_constructor_invalid_metric(dataset, collection_name):
    with pytest.raises(ValueError):
        Qdrant(dataset, metric="INVALID", url="localhost", collection_name=collection_name)


def test_upsert(collection_name, vectordb, dataset, N):
    data_points = [DataPoint(point["id"], point["embedding"], metadata={}) for point in dataset]
    vectordb.upsert(data_points)

    count_result = vectordb.client.count(collection_name=collection_name, exact=True)

    assert count_result.count == N


def test_search(vectordb, dataset):
    data_points = [DataPoint(point["id"], point["embedding"], metadata={}) for point in dataset]
    vectordb.upsert(data_points)

    # query single point
    point = dataset[117]
    expected_id = point["id"]
    query = Query(vector=point["embedding"])
    results = vectordb.search(q=query, topk=5)

    assert results[0].id == expected_id

    # regression
    assert np.allclose(results[0].score, 1.0)
    assert np.allclose(results[1].score, 0.7783108)


def test_delete(collection_name, vectordb, dataset, N):
    data_points = [DataPoint(point["id"], point["embedding"], metadata={}) for point in dataset]
    vectordb.upsert(data_points)

    ids_to_delete = [p["id"] for p in dataset.select([117])]
    vectordb.delete(ids_to_delete)

    assert vectordb.client.count(collection_name=collection_name).count == N - 1

    query = Query(vector=dataset[117]["embedding"])
    results = vectordb.search(q=query, topk=5)

    # We deleted the vector we are querying
    # so the score should not be 1.0
    assert results[0].score != 1.0


def test_stats(vectordb):
    assert vectordb.stats() == {"metric": "cosine", "m": 32, "ef_construct": 128}
