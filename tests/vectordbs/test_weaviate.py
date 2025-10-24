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


"""Optional live Weaviate tests mirroring the benchmark flow."""

import pytest

pytestmark = pytest.mark.skip(reason="Qdrant tests disabled for .venv-only test runs.")

import time
import uuid

import docker
import numpy as np
import requests
from datasets import Dataset
from docker.errors import APIError, DockerException
from pathlib import Path

from inatinqperf.adaptors.base import DataPoint, Query
from inatinqperf.adaptors.metric import Metric
from inatinqperf.adaptors.weaviate_adaptor import Weaviate
from inatinqperf.benchmark.benchmark import Benchmarker
from inatinqperf.benchmark.configuration import (
    Config,
    DatasetConfig,
    EmbeddingParams,
    SearchParams,
    VectorDatabaseConfig,
    VectorDatabaseParams,
)


@pytest.fixture(scope="module")
def base_url() -> str:
    """Base URL for the live Weaviate instance."""
    return "http://localhost:8080"


@pytest.fixture(scope="module")
def weaviate_image() -> str:
    """Docker image used to launch the live Weaviate instance."""
    return "semitechnologies/weaviate:1.31.16-ab5cb66.arm64"


@pytest.fixture(scope="module")
def weaviate_command() -> list[str]:
    """Command-line arguments passed to the Weaviate container."""
    return ["--host", "0.0.0.0", "--port", "8080", "--scheme", "http"]


@pytest.fixture(scope="module")
def weaviate_environment() -> dict[str, str]:
    """Environment variables applied to the Weaviate container."""
    return {
        "AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED": "true",
        "ENABLE_MODULES": "",
        "DEFAULT_VECTORIZER_MODULE": "none",
        "PERSISTENCE_DATA_PATH": "/var/lib/weaviate",
        "QUERY_DEFAULTS_LIMIT": "20",
    }


@pytest.fixture(scope="module")
def weaviate_container_name() -> str:
    """Container name for the live Weaviate instance."""
    return "inatinqperf-weaviate-live-tests"


def _service_ready(base_url: str) -> bool:
    """Return True when the Weaviate readiness endpoint responds with HTTP 200."""
    try:
        response = requests.get(f"{base_url}/v1/.well-known/ready", timeout=2)
    except requests.RequestException:
        return False
    return response.status_code == 200


def _wait_for_weaviate(base_url: str, timeout_seconds: int = 120) -> None:
    """Poll the readiness endpoint until the service reports healthy."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if _service_ready(base_url):
            return
        time.sleep(1)
    msg = "Weaviate service did not become ready within the allotted timeout."
    raise RuntimeError(msg)


@pytest.fixture(scope="module", autouse=True)
def weaviate_container(
    base_url: str,
    weaviate_image: str,
    weaviate_command: list[str],
    weaviate_environment: dict[str, str],
    weaviate_container_name: str,
) -> None:
    """Ensure a Weaviate container equivalent to the provided docker command is running."""
    try:
        client = docker.from_env()
    except DockerException as exc:  # pragma: no cover - environment specific
        pytest.skip(f"Docker daemon unavailable: {exc}")

    container = None
    started = False
    try:
        if _service_ready(base_url):
            yield
            return

        try:
            container = client.containers.run(
                weaviate_image,
                name=weaviate_container_name,
                ports={"8080": "8080", "8081": "8081"},
                environment=weaviate_environment,
                command=weaviate_command,
                remove=True,
                detach=True,
            )
            started = True
        except APIError as exc:  # pragma: no cover - depends on daemon state
            if "port is already allocated" not in str(exc).lower():
                pytest.skip(f"Unable to start Weaviate container: {exc}")

        _wait_for_weaviate(base_url)
        yield
    finally:
        if started and container is not None:
            try:
                container.stop()
            except APIError:
                pass  # best-effort cleanup
        client.close()


def test_weaviate_live_single_vector(tmp_path, base_url: str) -> None:
    dataset = Dataset.from_dict(
        {
            "id": [123],
            "embedding": [np.random.default_rng(0).standard_normal(512).astype(np.float32).tolist()],
        }
    )
    # print("Dataset row:", dataset[0])

    config = Config(
        dataset=DatasetConfig(
            dataset_id="stub",
            splits="validation",
            directory=tmp_path / "raw",
            export_images=False,
        ),
        embedding=EmbeddingParams(
            model_id="stub",
            batch_size=1,
            directory=tmp_path / "emb",
        ),
        vectordb=VectorDatabaseConfig(
            type="weaviate.hnsw",
            params=VectorDatabaseParams(
                metric=Metric.COSINE,
                url=base_url,
                collection_name="BenchmarkLiveTest",
                timeout=10.0,
            ),
        ),
        search=SearchParams(
            topk=1,
            queries_file=tmp_path / "queries.txt",
        ),
        update={"add_count": 1, "delete_count": 1},
    )

    benchmarker = Benchmarker(config_file=Path("configs/inquire_benchmark_weaviate.yaml"))
    benchmarker.cfg = config
    benchmarker.base_path = tmp_path

    vectordb = benchmarker.build(dataset)
    build_stats = vectordb.stats()

    query_vec = list(dataset[0]["embedding"])
    query = Query(query_vec)
    weaviate_results = vectordb.search(query, topk=1)

    assert weaviate_results
    assert weaviate_results[0].id == 123

    vectordb.drop_index()


@pytest.fixture
def live_class_name() -> str:
    """Generate a unique Weaviate class name per test."""
    return f"Perf{uuid.uuid4().hex[:10].capitalize()}"


@pytest.fixture
def live_dataset() -> Dataset:
    """Return a small dataset suitable for live adaptor construction."""
    rng = np.random.default_rng(21)
    dim = 4
    embeddings = rng.standard_normal((6, dim)).astype(np.float32)
    ids = list(range(5000, 5000 + embeddings.shape[0]))
    return Dataset.from_dict({"id": ids, "embedding": embeddings.tolist()})


@pytest.fixture
def live_adaptor(live_class_name: str, live_dataset: Dataset, base_url: str) -> tuple[Weaviate, Dataset]:
    """Construct a Weaviate adaptor bound to the live instance."""
    adaptor = Weaviate(
        dataset=live_dataset,
        metric=Metric.COSINE,
        url=base_url,
        collection_name=live_class_name,
        batch_size=3,
        timeout=10.0,
    )
    try:
        yield adaptor, live_dataset
    finally:
        try:
            adaptor.drop_index()
        except Exception:  # pragma: no cover - defensive cleanup
            pass


def test_weaviate_live_initial_stats(live_adaptor: tuple[Weaviate, Dataset]) -> None:
    """Initial ingest should populate stats and class metadata."""
    adaptor, dataset = live_adaptor
    stats = adaptor.stats()
    assert stats["ntotal"] == len(dataset)
    assert stats["class_name"] == adaptor.collection_name
    assert stats["metric"] == adaptor.metric.value


def test_weaviate_live_upsert_search_delete(live_adaptor: tuple[Weaviate, Dataset]) -> None:
    """Exercise upsert, search, delete, stats, and drop_index against a live instance."""
    adaptor, dataset = live_adaptor
    rng = np.random.default_rng(99)
    dim = adaptor.dim
    new_vectors = rng.standard_normal((3, dim)).astype(np.float32)
    new_ids = [9100, 9101, 9102]
    datapoints = [
        DataPoint(id=int(identifier), vector=vector.tolist(), metadata={"source": "live-upsert"})
        for identifier, vector in zip(new_ids, new_vectors, strict=False)
    ]

    adaptor.upsert(datapoints)

    # Verify stats reflect the new inserts (allow brief propagation window).
    expected_total = len(dataset) + len(datapoints)
    deadline = time.time() + 10.0
    post_stats = adaptor.stats()
    while post_stats.get("ntotal") != expected_total and time.time() < deadline:
        time.sleep(0.5)
        post_stats = adaptor.stats()
    assert post_stats["ntotal"] == expected_total

    # Query near the first inserted vector and confirm ids are surfaced.
    query = Query(vector=new_vectors[0].tolist())
    results = adaptor.search(query, topk=3)
    returned_ids = {result.id for result in results}
    assert new_ids[0] in returned_ids

    # Delete a subset and confirm ntotal shrinks accordingly.
    adaptor.delete(new_ids[:2])
    expected_total -= 2
    post_delete = adaptor.stats()
    deadline = time.time() + 10.0
    while post_delete.get("ntotal") != expected_total and time.time() < deadline:
        time.sleep(0.5)
        post_delete = adaptor.stats()
    assert post_delete["ntotal"] == expected_total

    adaptor.drop_index()
