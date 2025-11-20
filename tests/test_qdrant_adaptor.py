"""Tests for the Qdrant vector database adaptor class."""

import docker
import numpy as np
import pytest
from types import SimpleNamespace

from inatinqperf.adaptors.enums import Metric
from inatinqperf.adaptors.qdrant_adaptor import (
    DataPoint,
    HuggingFaceDataset,
    Qdrant,
    QdrantCluster,
    Query,
)


@pytest.fixture(scope="module", autouse=True)
def container_fixture():
    """Start the docker container with the vector DB."""
    client = docker.from_env()
    container = client.containers.run(
        "qdrant/qdrant",
        ports={"6333": "6333", "6334": "6334"},
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


def test_qdrant_cluster_uses_cluster_config(monkeypatch, dataset):
    recorded = {}

    def fake_wait(self, endpoint, timeout, container_names):
        recorded["wait_endpoint"] = endpoint
        recorded["wait_timeout"] = timeout
        recorded["wait_containers"] = container_names

    class FakeQdrantClient:
        def __init__(self, *, url, grpc_port, prefer_grpc, **kwargs):
            recorded["client_kwargs"] = {
                "url": url,
                "grpc_port": grpc_port,
                "prefer_grpc": prefer_grpc,
                "extra": kwargs,
            }
            self.created_collection = None

        def collection_exists(self, collection_name):
            return False

        def delete_collection(self, collection_name):
            recorded["deleted"] = collection_name

        def create_collection(self, **kwargs):
            self.created_collection = kwargs
            recorded["create_args"] = kwargs

        def upsert(self, **kwargs):  # noqa: D401,ARG002
            """Record upsert operations for inspection."""
            recorded.setdefault("upserts", []).append(kwargs)

        def count(self, **kwargs):  # noqa: D401,ARG002
            """Return a simple namespace mimicking Qdrant count response."""
            return SimpleNamespace(count=0)

    monkeypatch.setattr("inatinqperf.adaptors.qdrant_adaptor.QdrantClient", FakeQdrantClient)
    monkeypatch.setattr("inatinqperf.adaptors.qdrant_adaptor.QdrantCluster._wait_for_startup", fake_wait)

    cluster = QdrantCluster(
        dataset=dataset,
        metric=Metric.COSINE,
        url="http://primary",
        port="7000",
        collection_name="test_cluster",
        node_urls=["http://node-a:1234", "http://node-b:1234"],
        container_names=["node-a"],
        m=11,
        ef=55,
        shard_number=5,
        replication_factor=2,
        write_consistency_factor=3,
        grpc_port="9876",
        prefer_grpc=True,
        batch_size=1,
        startup_timeout=12.5,
    )

    assert recorded["wait_endpoint"] == cluster.node_urls[0]
    assert recorded["wait_containers"] == cluster.container_names
    assert pytest.approx(recorded["wait_timeout"], rel=0.01) == 12.5
    assert recorded["client_kwargs"]["grpc_port"] == 9876
    assert recorded["client_kwargs"]["prefer_grpc"] is True
    stats = cluster.stats()
    assert stats["nodes"] == ["http://node-a:1234", "http://node-b:1234"]
    assert stats["shard_number"] == 5
    assert stats["replication_factor"] == 2
