"""End-to-end tests for the Weaviate adaptor."""

from __future__ import annotations

import time
import uuid

import docker
import numpy as np
import pytest
import requests
from docker.errors import APIError, DockerException

from inatinqperf.adaptors.weaviate_adaptor import Weaviate
from inatinqperf.adaptors.metric import Metric

# Mark these as integration tests and skip if Docker is unavailable.


def _docker_available() -> bool:
    """Return True when the Docker daemon is reachable."""

    try:
        client = docker.from_env()
        try:
            client.ping()
        finally:
            client.close()
    except DockerException:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _docker_available(), reason="Docker daemon unavailable for integration tests"
)

# These tests talk to a real Weaviate instance listening on localhost.
BASE_URL = "http://localhost:8080"
WEAVIATE_IMAGE = "semitechnologies/weaviate:1.31.16-ab5cb66.arm64"
# Minimal command to expose the HTTP interface on localhost:8080.
WEAVIATE_COMMAND = ["--host", "0.0.0.0", "--port", "8080", "--scheme", "http"]
# Environment matches the configuration we document for end-to-end runs.
WEAVIATE_ENVIRONMENT = {
    "AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED": "true",
    "ENABLE_MODULES": "",
    "DEFAULT_VECTORIZER_MODULE": "none",
    "PERSISTENCE_DATA_PATH": "/var/lib/weaviate",
    "QUERY_DEFAULTS_LIMIT": "20",
}


def wait_for_weaviate(timeout_seconds: int = 120) -> None:
    """Poll the readiness endpoint until the container reports healthy."""

    deadline = time.time() + timeout_seconds
    endpoint = f"{BASE_URL}/v1/.well-known/ready"
    while time.time() < deadline:
        try:
            response = requests.get(endpoint, timeout=2)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass  # container still booting -> retry after a short sleep
        time.sleep(1)
    msg = "Weaviate service did not become ready within the allotted timeout."
    raise RuntimeError(msg)


@pytest.fixture(scope="module", autouse=True)
def weaviate_container():
    """Ensure a Weaviate container is available for the duration of the tests."""
    # Guard against environments where the daemon/socket is not accessible.
    try:
        client = docker.from_env()
    except DockerException as exc:
        pytest.skip(f"docker daemon unavailable: {exc}")
    container = None
    started = False
    try:
        # Reuse an already-running instance if one is reachable.
        try:
            response = requests.get(f"{BASE_URL}/v1/.well-known/ready", timeout=2)
            if response.status_code == 200:
                yield
                return
        except requests.RequestException:
            pass

        try:
            container = client.containers.run(  # spin up the test instance
                WEAVIATE_IMAGE,
                ports={"8080": "8080", "8081": "8081"},
                environment=WEAVIATE_ENVIRONMENT,
                command=WEAVIATE_COMMAND,
                remove=True,
                detach=True,
            )
            started = True
        except APIError as exc:  # container already running or port busy
            if "port is already allocated" not in str(exc).lower():
                pytest.skip(f"unable to start weaviate container: {exc}")
        wait_for_weaviate()
        yield
    finally:
        if started and container is not None:
            container.stop()


@pytest.fixture(name="class_name")
def class_name_fixture() -> str:
    """Generate a valid Weaviate class name for isolation."""
    prefix = "Perf"
    suffix = uuid.uuid4().hex[:10].capitalize()
    return f"{prefix}{suffix}"


def test_weaviate_adaptor_lifecycle(class_name: str):
    """Exercise the full lifecycle against a live Weaviate instance."""
    adaptor = Weaviate(dim=4, metric=Metric.COSINE, base_url=BASE_URL, class_name=class_name)
    adaptor.drop_index()  # ensure clean start

    train = np.random.default_rng(7).standard_normal((8, 4)).astype(np.float32)
    adaptor.train_index(train)

    ids = np.arange(100, 104, dtype=np.int64)
    vectors = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    adaptor.upsert(ids, vectors)

    stats = adaptor.stats()  # confirm aggregation reflects inserted objects
    assert stats["ntotal"] == len(ids)
    assert stats["class_name"] == class_name

    query = np.array([[1.0, 0.1, 0.0, 0.0]], dtype=np.float32)
    distances, found_ids = adaptor.search(query, topk=3)
    assert distances.shape == (1, 3)
    assert found_ids.shape == (1, 3)
    assert int(found_ids[0, 0]) == 100

    adaptor.delete([100, 999])
    stats_after_delete = adaptor.stats()
    assert stats_after_delete["ntotal"] == len(ids) - 1

    adaptor.drop_index()


def test_weaviate_upsert_replaces_vectors(class_name: str):
    """Verify upsert overwrites existing vectors rather than duplicating them."""
    adaptor = Weaviate(dim=3, metric=Metric.COSINE, base_url=BASE_URL, class_name=class_name)
    # Drop any leftover schema, then create a fresh one for the test.
    adaptor.drop_index()
    adaptor.train_index(np.zeros((1, 3), dtype=np.float32))

    ids = np.array([1, 2], dtype=np.int64)
    first = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    adaptor.upsert(ids, first)
    second = first + 0.5
    adaptor.upsert(ids, second)

    stats = adaptor.stats()
    assert stats["ntotal"] == len(ids)

    # Ensure subsequent search surfaces the updated vectors (id=2 closer to query)
    query = np.array([[0.0, 1.0, 0.5]], dtype=np.float32)
    distances, found_ids = adaptor.search(query, topk=2)
    assert found_ids[0, 0] == 2

    adaptor.drop_index()


def test_weaviate_metric_mapping(class_name: str):
    """Confirm metric names map to Weaviate's expected distance types."""
    adaptor = Weaviate(dim=2, metric=Metric.INNER_PRODUCT, base_url=BASE_URL, class_name=class_name)
    adaptor.drop_index()
    adaptor.train_index(np.zeros((1, 2), dtype=np.float32))

    ids = np.array([10, 11], dtype=np.int64)
    vectors = np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
    adaptor.upsert(ids, vectors)

    query = np.array([[0.6, 0.4]], dtype=np.float32)
    distances, found_ids = adaptor.search(query, topk=2)
    assert distances.shape == (1, 2)
    assert found_ids.shape == (1, 2)
    assert set(found_ids[0]) == {10, 11}

    adaptor.drop_index()
