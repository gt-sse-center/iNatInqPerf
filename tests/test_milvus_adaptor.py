"""Tests for the Milvus vector database adaptor class."""

import docker
import numpy as np
import pytest
from datasets import Dataset as HuggingFaceDataset

from inatinqperf.adaptors.enums import Metric
from inatinqperf.adaptors.milvus_adaptor import (
    DataPoint,
    Milvus,
    MilvusIndexType,
    Query,
)

index_params = {
    MilvusIndexType.HNSW: {"M": 4, "efConstruction": 128},
    MilvusIndexType.HNSW_SQ: {"M": 4, "efConstruction": 128},
    MilvusIndexType.HNSW_PQ: {"M": 4, "efConstruction": 128},
    MilvusIndexType.IVF_FLAT: {"nlist": 100},
    MilvusIndexType.IVF_SQ8: {"nlist": 100},
    MilvusIndexType.IVF_PQ: {"nlist": 100, "m": 4},
}


@pytest.fixture(name="milvus_network", scope="module")
def milvus_network_fixture():
    client = docker.from_env()

    # If the network already exists, then remove it
    for existing_network in client.networks.list():
        if "milvus" == existing_network.name:
            existing_network.remove()

    network = client.networks.create("milvus", driver="bridge")

    yield network

    network.remove()


@pytest.fixture(name="etcd_container", scope="module")
def etcd_container_fixture(milvus_network):
    client = docker.from_env()
    container = client.containers.run(
        "quay.io/coreos/etcd:v3.5.18",
        name="milvus-etcd",
        environment={
            "ETCD_AUTO_COMPACTION_MODE": "revision",
            "ETCD_AUTO_COMPACTION_RETENTION": "1000",
            "ETCD_QUOTA_BACKEND_BYTES": "4294967296",
            "ETCD_SNAPSHOT_COUNT": "50000",
        },
        hostname="etcd",  # This is the trick to getting the milvus-standalone to connect
        ports={
            2379: 2379,
            2380: 2380,
        },
        volumes=["etcd:/etcd"],
        command="etcd -advertise-client-urls=http://etcd:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd",
        healthcheck={
            "test": ["CMD", "etcdctl", "endpoint", "health"],
            "interval": 30 * 10**9,
            "timeout": 20 * 10**9,
            "retries": 3,
        },
        network="milvus",
        remove=True,
        detach=True,  # enabled so we get back a Container object
    )

    yield container

    container.stop()


@pytest.fixture(name="minio_container", scope="module")
def minio_container_fixture(milvus_network):
    client = docker.from_env()
    container = client.containers.run(
        name="milvus-minio",
        image="minio/minio:RELEASE.2024-12-18T13-15-44Z",
        environment={
            "MINIO_ACCESS_KEY": "minioadmin",
            "MINIO_SECRET_KEY": "minioadmin",
        },
        hostname="minio",  # This is the trick to getting the milvus-standalone to connect
        ports={
            9001: 9001,
            9000: 9000,
        },
        volumes=["minio:/minio_data"],
        command='minio server /minio_data --console-address ":9001"',
        healthcheck={
            "test": ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"],
            "interval": 30 * 10**9,
            "timeout": 20 * 10**9,
            "retries": 3,
        },
        network="milvus",
        remove=True,
        detach=True,  # enabled so we get back a Container object
    )

    yield container

    container.stop()


@pytest.fixture(scope="module", autouse=True)
def container_fixture(etcd_container, minio_container, milvus_network):
    """Start the Milvus docker container with the vector DB."""

    client = docker.from_env()
    assert client.containers.get(etcd_container.name).status == "running"
    assert client.containers.get(minio_container.name).status == "running"

    container = client.containers.run(
        name="milvus-standalone",
        image="milvusdb/milvus:v2.6.3",
        command=["milvus", "run", "standalone"],
        security_opt=["seccomp:unconfined"],
        environment={
            "ETCD_ENDPOINTS": "etcd:2379",
            "MINIO_ADDRESS": "minio:9000",
            "MQ_TYPE": "woodpecker",
        },
        volumes=["milvus:/var/lib/milvus"],
        healthcheck={
            "test": ["CMD", "curl", "-f", "http://localhost:9091/healthz"],
            "interval": 30 * 10**9,
            "timeout": 20 * 10**9,
            "retries": 3,
        },
        ports={
            19530: 19530,
            9091: 9091,
        },
        network="milvus",
        remove=True,
        detach=True,  # enabled so we get back a Container object
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
    """Create a HuggingFace dataset for testing."""
    rng = np.random.default_rng(117)
    ids = rng.choice(10**4, size=N, replace=False).tolist()
    x = rng.random(size=(N, dim))

    # Create HuggingFace dataset
    dataset = HuggingFaceDataset.from_dict({"id": ids, "embedding": x.tolist()})

    return dataset


@pytest.fixture(name="vectordb")
def vectordb_fixture(dataset):
    """Return an instance of the Milvus vector database."""
    vectordb = Milvus(
        dataset=dataset,
        metric=Metric.L2,
        index_type=MilvusIndexType.IVF_FLAT,
        index_params=index_params[MilvusIndexType.IVF_FLAT],
    )

    # NOTE: this typically happens automatically when upserting, but we'll do it explicitly for testing purposes
    vectordb.client.flush(collection_name=vectordb.collection_name)

    yield vectordb


@pytest.mark.parametrize("metric", [Metric.INNER_PRODUCT, Metric.COSINE, Metric.L2])
@pytest.mark.parametrize(
    "index_type",
    [
        MilvusIndexType.HNSW,
        MilvusIndexType.HNSW_SQ,
        MilvusIndexType.HNSW_PQ,
        MilvusIndexType.IVF_FLAT,
        MilvusIndexType.IVF_SQ8,
        MilvusIndexType.IVF_PQ,
    ],
)
def test_constructor(dataset, metric, index_type):
    """Test Milvus constructor with different metrics."""
    vectordb = Milvus(
        dataset=dataset,
        metric=metric,
        index_type=index_type,
        index_params=index_params[index_type],
    )

    # NOTE: this typically happens automatically when upserting, but we'll do it explicitly for testing purposes
    vectordb.client.flush(collection_name=vectordb.collection_name)

    assert vectordb.client.has_collection(vectordb.collection_name)


def test_upsert(vectordb, N):
    """Test upserting vectors."""
    # Create new data points to upsert
    rng = np.random.default_rng(42)
    M = 10  # num of new data points to add
    new_ids = rng.choice(10**5, size=M, replace=False).tolist()
    new_vectors = rng.random(size=(M, vectordb.dim))

    data_points = [DataPoint(i, vector, metadata={}) for i, vector in zip(new_ids, new_vectors)]
    vectordb.upsert(data_points)

    # NOTE: this typically happens automatically when upserting, but we'll do it explicitly for testing purposes
    vectordb.client.flush(collection_name=vectordb.collection_name)

    # Check that the collection still exists and has data
    assert vectordb.client.has_collection(vectordb.collection_name)

    # Verify that the number of data points has increased correctly.
    assert vectordb.client.get_collection_stats(vectordb.collection_name)["row_count"] == N + M


def test_search(vectordb, dataset):
    """Test searching for nearest neighbors."""
    # Use a vector from the original dataset for querying
    query_vector = dataset[117]["embedding"]
    query = Query(vector=query_vector)
    results = vectordb.search(q=query, topk=5)

    assert len(results) == 5
    assert results[0].id == dataset[117]["id"]  # Should find the exact match


def test_delete(vectordb, dataset):
    """Test deleting vectors."""
    # Delete a specific vector
    id_to_delete = dataset[117]["id"]
    vectordb.delete([id_to_delete])

    # NOTE: this typically happens automatically when upserting, but we'll do it explicitly for testing purposes
    vectordb.client.flush(collection_name=vectordb.collection_name)

    # Verify deletion by searching for the deleted vector
    query_vector = dataset[117]["embedding"]
    query = Query(vector=query_vector)
    results = vectordb.search(q=query, topk=5)

    # The deleted vector should not be the top result
    assert results[0].id != id_to_delete


def test_stats(vectordb):
    """Test getting index statistics."""
    stats = vectordb.stats()
    assert isinstance(stats, dict)
    # Check that stats contain expected keys for Milvus index
    assert "index_type" in stats
    assert "metric_type" in stats

    # regression
    assert stats["index_type"] == "IVF_FLAT"
    assert stats["metric_type"] == "L2"
    assert stats["nlist"] == "100"


def test_translate_metric():
    """Test translating metric to Milvus metric type."""
    # disable pylint warning for private method access
    # pylint: disable=W0212
    assert Milvus._translate_metric(Metric.INNER_PRODUCT) == "IP"
    assert Milvus._translate_metric(Metric.COSINE) == "COSINE"
    assert Milvus._translate_metric(Metric.L2) == "L2"
    with pytest.raises(ValueError):
        Milvus._translate_metric(Metric.MANHATTAN)
