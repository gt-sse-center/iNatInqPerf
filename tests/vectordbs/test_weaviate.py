"""Tests for verifying if the `weaviate` vector DB server is up and running."""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager

import docker
import numpy as np
import pytest
import requests

import inatinqperf

BASE_URL = "http://localhost:8080"
SCHEMA_ENDPOINT = f"{BASE_URL}/v1/schema"
GRAPHQL_ENDPOINT = f"{BASE_URL}/v1/graphql"
READY_ENDPOINT = f"{BASE_URL}/v1/.well-known/ready"

# Pin to a concrete tag published in the official Weaviate registry.
# Registry doesn't have a latest tag
WEAVIATE_IMAGE = "semitechnologies/weaviate:1.31.16-ab5cb66.arm64"
WEAVIATE_COMMAND = ["--host", "0.0.0.0", "--port", "8080", "--scheme", "http"]
WEAVIATE_ENVIRONMENT = {
    "AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED": "true",
    "ENABLE_MODULES": "",
    "DEFAULT_VECTORIZER_MODULE": "none",
    "PERSISTENCE_DATA_PATH": "/var/lib/weaviate",
    "QUERY_DEFAULTS_LIMIT": "20",
}


def wait_for_weaviate(timeout_seconds: int = 60) -> None:
    """Poll the readiness endpoint until Weaviate responds."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            response = requests.get(READY_ENDPOINT, timeout=2)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    msg = "Weaviate service did not become ready within the allotted timeout."
    raise RuntimeError(msg)


def create_class(class_name: str) -> None:
    """Create a class schema in Weaviate."""
    payload = {
        "class": class_name,
        "description": "Collection used for integration tests.",
        "vectorizer": "none",
        "vectorIndexType": "hnsw",
        "vectorIndexConfig": {"distance": "cosine"},
        "properties": [
            {
                "name": "color",
                "description": "Color label for the vector.",
                "dataType": ["text"],
            },
            {
                "name": "randNumber",
                "description": "Random number bucket.",
                "dataType": ["int"],
            },
        ],
    }
    response = requests.post(SCHEMA_ENDPOINT, json=payload, timeout=10)
    response.raise_for_status()


def delete_class(class_name: str) -> None:
    """Remove a class schema from Weaviate, ignoring 404 responses."""
    response = requests.delete(f"{SCHEMA_ENDPOINT}/{class_name}", timeout=10)
    if response.status_code not in {200, 204, 404}:
        response.raise_for_status()


@contextmanager
def managed_class(class_name: str):
    create_class(class_name)
    try:
        yield
    finally:
        delete_class(class_name)


def insert_vectors(class_name: str, vectors: np.ndarray) -> None:
    """Insert vectors into Weaviate objects."""
    objects_endpoint = f"{BASE_URL}/v1/objects"
    for idx, vector in enumerate(vectors):
        body = {
            "id": uuid.uuid4().hex,
            "class": class_name,
            "properties": {
                "color": "red",
                "randNumber": int(idx % 10),
            },
            "vector": vector.tolist(),
        }
        response = requests.post(objects_endpoint, json=body, timeout=10)
        response.raise_for_status()


def aggregate_count(class_name: str) -> int:
    """Return the object count for a class via GraphQL aggregate."""
    query = f"""
    {{
      Aggregate {{
        {class_name} {{
          meta {{
            count
          }}
        }}
      }}
    }}
    """
    response = requests.post(GRAPHQL_ENDPOINT, json={"query": query}, timeout=10)
    response.raise_for_status()
    payload = response.json()
    if "errors" in payload:
        raise RuntimeError(payload["errors"])
    aggregates = payload.get("data", {}).get("Aggregate", {})
    class_results = aggregates.get(class_name, [])
    if not class_results:
        return 0
    return class_results[0]["meta"]["count"]


def query_near_vector(class_name: str, vector: np.ndarray, limit: int) -> list[dict]:
    """Query Weaviate for nearest neighbors using GraphQL."""
    vector_json = json.dumps(vector.tolist())
    query = f"""
    {{
      Get {{
        {class_name}(nearVector: {{ vector: {vector_json} }}, limit: {limit}) {{
          _additional {{
            distance
          }}
        }}
      }}
    }}
    """
    response = requests.post(GRAPHQL_ENDPOINT, json={"query": query}, timeout=10)
    response.raise_for_status()
    payload = response.json()
    if "errors" in payload:
        raise RuntimeError(payload["errors"])
    return payload.get("data", {}).get("Get", {}).get(class_name, [])


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance between two vectors."""
    similarity = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return 1.0 - similarity


@pytest.fixture(scope="module", autouse=True)
def container_fixture():
    """Start the docker container with the weaviate."""
    client = docker.from_env()
    container = client.containers.run(
        WEAVIATE_IMAGE,
        ports={"8080": "8080", "8081": "8081"},
        environment=WEAVIATE_ENVIRONMENT,
        command=WEAVIATE_COMMAND,
        remove=True,
        detach=True,
    )
    try:
        wait_for_weaviate()
        yield container
    finally:
        container.stop()


@pytest.fixture(name="class_name")
def class_name_fixture(collection_name: str):
    """Convert the common collection name fixture to a valid Weaviate class name."""
    parts = [part.capitalize() for part in collection_name.split("_") if part]
    return "".join(parts) or "TestCollection"


def test_create_collection(class_name: str):
    """Test class creation in Weaviate."""
    delete_class(class_name)
    with managed_class(class_name):
        response = requests.get(f"{SCHEMA_ENDPOINT}/{class_name}", timeout=10)
        response.raise_for_status()
        payload = response.json()
        assert payload.get("class") == class_name


def test_vector_insertion(class_name: str):
    """Test insertion of vectors into the weaviate."""
    num_vectors = 117
    rng = np.random.default_rng(seed=101)
    vectors = rng.random((num_vectors, 100))
    delete_class(class_name)
    with managed_class(class_name):
        insert_vectors(class_name, vectors)
        assert aggregate_count(class_name) == num_vectors


@pytest.mark.regression
def test_search(class_name: str):
    """Test search capabilities of the weaviate."""
    rng = np.random.default_rng(seed=101)
    vectors = rng.random((101, 100))
    query_vector = rng.random(100)
    delete_class(class_name)
    with managed_class(class_name):
        insert_vectors(class_name, vectors)
        hits = query_near_vector(class_name, query_vector, limit=5)
        assert len(hits) == 5
        distances = [result["_additional"]["distance"] for result in hits]
        expected = [cosine_distance(query_vector, vec) for vec in vectors]
        assert np.isclose(distances[0], min(expected), rtol=1e-5)
