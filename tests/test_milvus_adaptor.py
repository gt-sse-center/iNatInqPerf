"""Tests for the Milvus vector database adaptor class."""

import docker
import numpy as np
import pytest
import subprocess
from datasets import Dataset as HuggingFaceDataset

from inatinqperf.adaptors.metric import Metric
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


@pytest.fixture(scope="module", autouse=True)
def container_fixture():
    """Start the docker container with the vector DB."""

    subprocess.run(
        ["docker", "compose", "-f", "milvus-standalone-docker-compose.yml", "up", "-d"], check=True
    )
    yield
    subprocess.run(["docker", "compose", "-f", "milvus-standalone-docker-compose.yml", "down"], check=True)


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
        host="localhost",
    )

    # NOTE: this typically happens automatically when upserting, but we'll do it explicitly for testing purposes
    vectordb.client.flush(collection_name=vectordb.COLLECTION_NAME)

    yield vectordb

    vectordb.teardown()


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
        host="localhost",
    )

    # NOTE: this typically happens automatically when upserting, but we'll do it explicitly for testing purposes
    vectordb.client.flush(collection_name=vectordb.COLLECTION_NAME)

    assert vectordb.client.has_collection(vectordb.COLLECTION_NAME)

    vectordb.teardown()


def test_upsert(vectordb):
    """Test upserting vectors."""
    # Create new data points to upsert
    rng = np.random.default_rng(42)
    new_ids = rng.choice(10**5, size=10, replace=False).tolist()
    new_vectors = rng.random(size=(10, vectordb.dim))

    data_points = [DataPoint(i, vector, metadata={}) for i, vector in zip(new_ids, new_vectors)]
    vectordb.upsert(data_points)

    # NOTE: this typically happens automatically when upserting, but we'll do it explicitly for testing purposes
    vectordb.client.flush(collection_name=vectordb.COLLECTION_NAME)

    # Check that the collection still exists and has data
    assert vectordb.client.has_collection(vectordb.COLLECTION_NAME)


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
    vectordb.client.flush(collection_name=vectordb.COLLECTION_NAME)

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


def test_translate_metric():
    """Test translating metric to Milvus metric type."""
    # disable pylint warning for private method access
    # pylint: disable=W0212
    assert Milvus._translate_metric(Metric.INNER_PRODUCT) == "IP"
    assert Milvus._translate_metric(Metric.COSINE) == "COSINE"
    assert Milvus._translate_metric(Metric.L2) == "L2"
    with pytest.raises(ValueError):
        Milvus._translate_metric(Metric.MANHATTAN)
