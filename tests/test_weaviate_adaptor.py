"""End-to-end tests for the Weaviate adaptor."""

import time


import docker
import numpy as np
import pytest
from datasets import Dataset
import weaviate
from weaviate.collections.classes import config

from inatinqperf.adaptors.base import DataPoint, Query
from inatinqperf.adaptors.enums import Metric
from inatinqperf.adaptors.weaviate_adaptor import Weaviate


@pytest.fixture(scope="module", autouse=True)
def container_fixture():
    """Ensure a Weaviate container is available for the duration of the tests."""
    client = docker.from_env()
    container = client.containers.run(
        "cr.weaviate.io/semitechnologies/weaviate:1.33.2",
        ports={
            "8080": "8080",
            "50051": "50051",
        },
        # Minimal command to expose the HTTP interface on localhost:8080
        command=["--host", "0.0.0.0", "--port", "8080", "--scheme", "http"],
        # Environment mirrors the configuration used during benchmark runs.
        environment={
            "QUERY_DEFAULTS_LIMIT": "25",
            "AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED": "true",
            "PERSISTENCE_DATA_PATH": "/var/lib/weaviate",
            "AUTOSCHEMA_ENABLED": "false",  # Disable auto-schema,
        },
        remove=True,
        detach=True,
        healthcheck={
            "test": ["CMD", "curl", "-f", "http://localhost:8080/v1/.well-known/ready"],
            "interval": 30 * 10**9,
            "timeout": 10 * 10**9,
            "retries": 3,
        },
    )

    # Wait for 3 seconds since the Weaviate container is slow to load.
    # This avoids connection timeout issues in tests.
    time.sleep(3)

    yield container

    container.stop()


@pytest.fixture(name="collection_name")
def collection_name_fixture() -> str:
    """Return the collection name for the vector database."""
    return "test_collection"


@pytest.fixture(name="N")
def num_datapoints_fixture():
    """The size of the dataset."""
    return 300


@pytest.fixture(name="dim")
def dim_fixture():
    """The dimension of the vectors used."""
    return 16


@pytest.fixture(name="dataset")
def dataset_fixture(N, dim) -> Dataset:
    """Create a simple dataset placeholder with the desired dimensionality."""
    rng = np.random.default_rng(117)
    ids = np.arange(N).tolist()
    x = rng.random(size=(N, dim))

    # Create HuggingFace dataset
    dataset = Dataset.from_dict({"id": ids, "embedding": x.tolist()})

    return dataset


def test_constructor(dataset, N):
    """Test the constructor of the Weaviate vector database."""
    adaptor = Weaviate(dataset, Metric.COSINE, "flat")
    collection = adaptor.client.collections.use(adaptor.collection_name)

    response = collection.aggregate.over_all(total_count=True)
    assert response.total_count == N


def test_invalid_constructor(dataset, caplog):
    # test invalid distance metric
    with pytest.raises(ValueError):
        Weaviate(dataset, "L4", "hnsw")

    # test invalid index type
    with pytest.raises(ValueError):
        # invalid distance metric
        Weaviate(dataset, "cosine", "haha")


def test_upsert(dataset, N, dim):
    adaptor = Weaviate(dataset, Metric.COSINE, "hnsw")

    adaptor.upsert([DataPoint(N, np.ones(dim), metadata={})])

    collection = adaptor.client.collections.use(adaptor.collection_name)
    response = collection.aggregate.over_all(total_count=True)
    assert response.total_count == N + 1

    id_to_replace = 10
    adaptor.upsert([DataPoint(id_to_replace, np.ones(dim), metadata={})])

    result = collection.query.fetch_object_by_id(
        weaviate.util.generate_uuid5(id_to_replace),
        include_vector=True,
    )
    assert np.ones(dim).tolist() == result.vector["default"]


def test_search(dataset):
    adaptor = Weaviate(dataset, Metric.COSINE, "hnsw")

    idx = 117
    vector = dataset["embedding"][idx]
    q = Query(vector=vector)

    results = adaptor.search(q, topk=3)

    assert results[0].id == dataset["id"][idx]


def test_delete(dataset, N):
    adaptor = Weaviate(dataset, Metric.COSINE, "hnsw")

    adaptor.delete([117])

    collection = adaptor.client.collections.use(adaptor.collection_name)
    response = collection.aggregate.over_all(total_count=True)
    assert response.total_count == N - 1


def test_delete_multiple(dataset, N):
    adaptor = Weaviate(dataset, Metric.COSINE, "hnsw")

    ids_to_delete = [117, 199, 222, 51]
    adaptor.delete(ids_to_delete)

    collection = adaptor.client.collections.use(adaptor.collection_name)
    response = collection.aggregate.over_all(total_count=True)
    assert response.total_count == N - len(ids_to_delete)


def test_delete_invalid(dataset, N):
    adaptor = Weaviate(dataset, Metric.COSINE, "hnsw")

    adaptor.delete([N + 100])

    collection = adaptor.client.collections.use(adaptor.collection_name)
    response = collection.aggregate.over_all(total_count=True)
    assert response.total_count == N


def test_stats(dataset, N, collection_name, dim):
    adaptor = Weaviate(dataset, Metric.COSINE, "hnsw", collection_name=collection_name)

    stats = adaptor.stats()

    assert stats["ntotal"] == N
    assert stats["metric"] == Metric.COSINE.value
    assert stats["collection_name"] == collection_name
    assert stats["dim"] == dim


def test_full_lifecycle(collection_name, dataset, N, dim):
    """Exercise the full lifecycle against a live Weaviate server instance."""
    adaptor = Weaviate(
        dataset=dataset,
        metric=Metric.COSINE,
        index_type="hnsw",
        url="http://localhost",
        collection_name=collection_name,
    )

    ids = np.arange(300, 304, dtype=np.int64)
    rng = np.random.default_rng(117)
    vectors = rng.random(size=(4, dim), dtype=np.float32)

    data_points = [
        DataPoint(id=int(i), vector=vector.tolist(), metadata={}) for i, vector in zip(ids, vectors)
    ]
    adaptor.upsert(data_points)

    stats = adaptor.stats()
    assert stats["ntotal"] == N + len(ids)
    assert stats["collection_name"] == collection_name

    query = Query(vector=rng.random(size=(dim,), dtype=np.float32).tolist())
    results = adaptor.search(query, topk=3)
    assert len(results) == 3
    # regression
    assert results[0].id == 240

    adaptor.delete([100, 999])
    stats_after_delete = adaptor.stats()
    assert stats_after_delete["ntotal"] == N + len(ids) - 1  # 999 is not a valid ID to delete, hence -1


@pytest.mark.parametrize(
    "metric,expected_metric",
    [(Metric.INNER_PRODUCT, "dot"), (Metric.COSINE, "cosine"), (Metric.L2, "l2-squared")],
)
def test_metric_mapping(collection_name, dataset, metric, expected_metric):
    """Confirm metric names map to Weaviate's expected distance types."""

    adaptor = Weaviate(
        dataset=dataset,
        metric=metric,
        url="http://localhost",
        collection_name=collection_name,
        index_type="hnsw",
    )

    collection = adaptor.client.collections.use(adaptor.collection_name)
    collection_config = collection.config.get()
    vector_index_config = collection_config.vector_config["default"].vector_index_config

    assert vector_index_config.distance_metric.value == expected_metric


@pytest.mark.parametrize(
    "index_type, expected_index_type",
    [
        ("hnsw", config.VectorIndexConfigHNSW),
        ("flat", config.VectorIndexConfigFlat),
    ],
)
def test_index_type_mapping(collection_name, dataset, index_type, expected_index_type):
    """Check if index type is properly configured."""

    adaptor = Weaviate(
        dataset=dataset,
        metric=Metric.COSINE,
        url="http://localhost",
        collection_name=collection_name,
        index_type=index_type,
    )

    collection = adaptor.client.collections.use(adaptor.collection_name)
    collection_config = collection.config.get()
    vector_index_config = collection_config.vector_config["default"].vector_index_config

    assert isinstance(vector_index_config, expected_index_type)
