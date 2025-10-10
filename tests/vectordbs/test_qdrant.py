"""Tests for verifying if the `qdrant` vector DB server is up and running."""

import docker
import numpy as np
import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


@pytest.fixture(scope="module", autouse=True)
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


@pytest.fixture(name="client")
def client_fixture():
    """Fixture to get QdrantClient."""
    # Connect to existing Qdrant instance
    client = QdrantClient("http://localhost:6333")
    return client


@pytest.fixture(name="client_with_collection")
def client_with_collection_fixture(client: QdrantClient):
    """Fixture to yield a client with a collection initialized."""
    collection_name = "test_collection"
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=100, distance=Distance.COSINE),
    )
    yield client
    client.delete_collection(collection_name)


@pytest.fixture(name="rng")
def rng_fixture():
    """Fixture for a random number generator with a fixed seed."""
    rng = np.random.default_rng(seed=101)
    return rng


def test_create_collection(client: QdrantClient, collection_name: str):
    """Test collection creation."""
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=100, distance=Distance.COSINE),
    )
    assert client.collection_exists(collection_name)

    # Clean up collection
    client.delete_collection(collection_name)


def test_vector_insertion(
    client_with_collection: QdrantClient,
    rng: np.random.Generator,
    collection_name: str,
):
    """Test insertion of vectors into the database."""
    num_vectors = 117  # Some number of vectors. Not of any significance.

    vectors = rng.random((num_vectors, 100))
    points = [
        PointStruct(id=idx, vector=vector, payload={"color": "red", "rand_number": idx % 10})
        for idx, vector in enumerate(vectors)
    ]
    client_with_collection.upsert(
        collection_name=collection_name,
        points=points,
    )

    count_result = client_with_collection.count(collection_name)
    assert count_result.count == num_vectors


@pytest.mark.regression
def test_search(
    client_with_collection: QdrantClient,
    rng: np.random.Generator,
    collection_name: str,
):
    """Test search capabilities of the vector db."""
    vectors = rng.random((101, 100))
    points = [
        PointStruct(id=idx, vector=vector, payload={"color": "red", "rand_number": idx % 10})
        for idx, vector in enumerate(vectors)
    ]
    client_with_collection.upsert(
        collection_name=collection_name,
        points=points,
    )

    # Perform search
    query_vector = rng.random((100,))
    hits = client_with_collection.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=5,  # Return 5 closest points
    )

    assert np.isclose(hits.points[0].score, 0.83825624)
