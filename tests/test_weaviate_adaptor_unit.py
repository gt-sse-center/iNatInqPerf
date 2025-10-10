"""Unit tests for the Weaviate adaptor using stubbed HTTP sessions."""

from __future__ import annotations

from collections import defaultdict
import uuid

import numpy as np
import pytest

from inatinqperf.adaptors.metric import Metric
from inatinqperf.adaptors.weaviate_adaptor import Weaviate, WeaviateError


class FakeResponse:
    """Lightweight stand-in for requests.Response with JSON payload support."""

    def __init__(self, status_code: int = 200, json_data: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._json = json_data or {}
        self.text = text

    def json(self) -> dict:
        return self._json


class StubSession:
    """Minimal stub emulating the requests.Session API used by the adaptor."""

    def __init__(self, class_name: str) -> None:
        self.class_name = class_name
        self.calls = defaultdict(list)
        self.graphql_results: list[dict] = []
        self.aggregate_count = 0

    def post(self, url: str, json: dict | None = None, timeout: float | None = None) -> FakeResponse:  # noqa: ARG002
        """Record POST requests and return canned responses."""
        self.calls["post"].append((url, json))
        if url.endswith("/v1/schema"):
            return FakeResponse(200)
        if url.endswith("/v1/objects"):
            return FakeResponse(200)
        if url.endswith("/v1/graphql"):
            if "Get" in (json or {}).get("query", ""):
                data = {"data": {"Get": {self.class_name: self.graphql_results}}}
                return FakeResponse(200, data)
            data = {
                "data": {
                    "Aggregate": {
                        self.class_name: [
                            {
                                "meta": {
                                    "count": self.aggregate_count,
                                }
                            }
                        ]
                    }
                }
            }
            return FakeResponse(200, data)
        return FakeResponse(404, text="unexpected post")

    def delete(self, url: str, timeout: float | None = None) -> FakeResponse:  # noqa: ARG002
        """Track DELETE calls and respond with success by default."""
        self.calls["delete"].append(url)
        return FakeResponse(204)

    def get(self, url: str, timeout: float | None = None) -> FakeResponse:  # noqa: ARG002
        """Return ready/404 responses matching the adaptor's expectations."""
        self.calls["get"].append(url)
        if url.endswith("/.well-known/ready"):
            return FakeResponse(200)
        if url.endswith("/v1/schema/TestClass"):
            return FakeResponse(404)
        return FakeResponse(200)


@pytest.fixture(name="adaptor_with_stub")
def adaptor_with_stub_fixture():
    """Provide a Weaviate adaptor wired to the stub session for unit tests."""
    # Minimal stub adaptor -> every test reuses this to avoid real HTTP calls.
    adaptor = Weaviate(dim=3, metric=Metric.COSINE, base_url="http://example.com", class_name="TestClass")
    stub = StubSession("TestClass")
    adaptor._session = stub  # type: ignore[attr-defined]
    adaptor._check_ready = lambda: None  # type: ignore[attr-defined]
    adaptor._class_exists = lambda: False  # type: ignore[attr-defined]
    return adaptor, stub


def test_train_index_creates_class(adaptor_with_stub):
    """Creating the class should POST to the schema endpoint."""
    adaptor, stub = adaptor_with_stub
    adaptor.train_index(np.zeros((1, 3), dtype=np.float32))
    assert any(call[0].endswith("/v1/schema") for call in stub.calls["post"])


def test_train_index_ignores_existing_class():
    """Ensure 422 from Weaviate (already exists) is treated as success."""
    adaptor = Weaviate(dim=3, metric=Metric.COSINE, base_url="http://example.com", class_name="TestClass")
    stub = StubSession("TestClass")

    def already_exists_post(url, json=None, timeout=None):  # type: ignore[arg-type]
        stub.calls["post"].append((url, json))
        return FakeResponse(422, text="Already exists")

    stub.post = already_exists_post  # type: ignore[assignment]
    adaptor._session = stub  # type: ignore[attr-defined]
    adaptor._check_ready = lambda: None  # type: ignore[attr-defined]
    adaptor._class_exists = lambda: False  # type: ignore[attr-defined]
    adaptor.train_index(np.zeros((1, 3), dtype=np.float32))
    assert any(call[0].endswith("/v1/schema") for call in stub.calls["post"])


def test_upsert_and_search_with_stub(adaptor_with_stub):
    """Upsert should succeed and search should surface the deterministic ids."""
    adaptor, stub = adaptor_with_stub
    adaptor._class_exists = lambda: True  # type: ignore[attr-defined]

    ids = np.array([1, 2], dtype=np.int64)
    vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    adaptor.upsert(ids, vectors)
    object_posts = [url for url, _ in stub.calls["post"] if url.endswith("/v1/objects")]
    assert len(object_posts) == len(ids)

    stub.graphql_results = [
        {"originalId": 2, "_additional": {"id": "00000000-0000-0000-0000-000000000002", "distance": 0.1}},
        {"originalId": 1, "_additional": {"id": "00000000-0000-0000-0000-000000000001", "distance": 0.9}},
    ]
    query = np.array([[0.1, 1.0, 0.0]], dtype=np.float32)
    distances, found = adaptor.search(query, topk=2)
    assert distances.shape == (1, 2)
    assert found.shape == (1, 2)
    assert list(found[0]) == [2, 1]


def test_stats_returns_count(adaptor_with_stub):
    """Stats should reflect the aggregate count returned by GraphQL."""
    adaptor, stub = adaptor_with_stub
    adaptor._class_exists = lambda: True  # type: ignore[attr-defined]
    stub.aggregate_count = 5
    stats = adaptor.stats()
    assert stats["ntotal"] == 5
    assert stats["class_name"] == "TestClass"


def test_stats_handles_graphql_error(adaptor_with_stub):
    """Stats should handle GraphQL "errors" payloads by returning zero."""
    adaptor, stub = adaptor_with_stub

    def graphql_error_post(url: str, json: dict | None = None, timeout: float | None = None):  # noqa: ARG002
        if url.endswith("/v1/graphql"):
            return FakeResponse(200, {"errors": [{"message": "boom"}]})
        return FakeResponse(200)

    stub.post = graphql_error_post  # type: ignore[assignment]
    stats = adaptor.stats()
    assert stats["ntotal"] == 0


def test_delete_handles_multiple_ids(adaptor_with_stub):
    """Deleting multiple ids should issue one request per id."""
    adaptor, stub = adaptor_with_stub
    adaptor.delete([10, 20])
    assert len(stub.calls["delete"]) == 2


def test_invalid_metric_raises_value_error():
    """Constructor should reject unsupported distance metrics."""
    with pytest.raises(ValueError):
        Weaviate(dim=2, metric="manhattan")


def test_upsert_rejects_dimension_mismatch(adaptor_with_stub):
    adaptor, _ = adaptor_with_stub
    with pytest.raises(ValueError):
        adaptor.upsert(np.array([1]), np.array([[1.0, 2.0]], dtype=np.float32))


def test_upsert_rejects_length_mismatch(adaptor_with_stub):
    """ids and vectors length mismatch should raise."""
    adaptor, _ = adaptor_with_stub
    with pytest.raises(ValueError):
        adaptor.upsert(np.array([1, 2]), np.array([[1.0, 0.0, 0.0]], dtype=np.float32))


def test_search_rejects_invalid_queries(adaptor_with_stub):
    adaptor, _ = adaptor_with_stub
    with pytest.raises(ValueError):
        adaptor.search(np.array([1.0, 2.0], dtype=np.float32), topk=1)

    with pytest.raises(ValueError):
        adaptor.search(np.zeros((1, 3), dtype=np.float32), topk=0)


def test_search_handles_graphql_errors(adaptor_with_stub):
    """GraphQL errors should return placeholder results instead of raising."""
    adaptor, stub = adaptor_with_stub

    def graphql_error_post(url: str, json: dict | None = None, timeout: float | None = None):  # noqa: ARG002
        if url.endswith("/v1/graphql"):
            return FakeResponse(200, {"errors": [{"message": "no results"}]})
        return FakeResponse(200)

    stub.post = graphql_error_post  # type: ignore[assignment]
    distances, ids = adaptor.search(np.zeros((1, 3), dtype=np.float32), topk=2)
    assert np.isinf(distances).all()
    assert (ids == -1).all()


def test_error_responses_raise_weaviate_error(adaptor_with_stub):
    """Non-success POST responses should raise WeaviateError."""
    adaptor, stub = adaptor_with_stub

    def failing_post(url: str, json: dict | None = None, timeout: float | None = None):  # noqa: ARG002
        return FakeResponse(500, text="boom")

    stub.post = failing_post  # type: ignore[assignment]
    with pytest.raises(WeaviateError):
        adaptor.train_index(np.zeros((1, 3), dtype=np.float32))


def test_drop_index_raises_on_failure():
    """DELETE failures should propagate as WeaviateError."""
    adaptor = Weaviate(dim=2, metric=Metric.COSINE, base_url="http://example.com", class_name="TestClass")
    stub = StubSession("TestClass")
    stub.delete = lambda url, timeout=None: FakeResponse(500, text="kaboom")  # type: ignore[arg-type,assignment]
    adaptor._session = stub  # type: ignore[attr-defined]
    with pytest.raises(WeaviateError):
        adaptor.drop_index()


def test_drop_index_success():
    """Successful drop should issue a delete call."""
    adaptor = Weaviate(dim=2, metric=Metric.COSINE, base_url="http://example.com", class_name="TestClass")
    stub = StubSession("TestClass")
    adaptor._session = stub  # type: ignore[attr-defined]
    adaptor.drop_index()
    assert stub.calls["delete"]


def test_check_ready_times_out_quickly(monkeypatch):
    """Readiness polling should surface repeated non-200 responses."""
    adaptor = Weaviate(
        dim=2, metric=Metric.COSINE, base_url="http://example.com", class_name="TestClass", timeout=0.6
    )
    stub = StubSession("TestClass")

    def failing_get(url, timeout=None):  # noqa: ARG002
        return FakeResponse(503, text="not ready")

    stub.get = failing_get  # type: ignore[assignment]
    adaptor._session = stub  # type: ignore[attr-defined]
    with pytest.raises(WeaviateError):
        adaptor._check_ready()


def test_check_ready_success():
    """Healthy readiness endpoint should exit the polling loop."""
    adaptor = Weaviate(
        dim=2, metric=Metric.COSINE, base_url="http://example.com", class_name="TestClass", timeout=1.0
    )
    stub = StubSession("TestClass")

    def ready_get(url, timeout=None):  # noqa: ARG002
        return FakeResponse(200)

    stub.get = ready_get  # type: ignore[assignment]
    adaptor._session = stub  # type: ignore[attr-defined]
    adaptor._check_ready()


def test_class_exists_paths():
    """Class existence helper should handle 404/200 and error paths."""
    adaptor = Weaviate(dim=2, metric=Metric.COSINE, base_url="http://example.com", class_name="TestClass")
    stub = StubSession("TestClass")

    # Not found -> False
    adaptor._session = stub  # type: ignore[attr-defined]
    stub.get = lambda url, timeout=None: FakeResponse(404)  # type: ignore[assignment]
    assert adaptor._class_exists() is False

    # Found -> True
    stub.get = lambda url, timeout=None: FakeResponse(200)  # type: ignore[assignment]
    assert adaptor._class_exists() is True

    # Unexpected status -> raises
    stub.get = lambda url, timeout=None: FakeResponse(500, text="oops")  # type: ignore[assignment]
    with pytest.raises(WeaviateError):
        adaptor._class_exists()


def test_train_index_raises_on_failed_creation():
    """train_index should raise when schema creation fails."""
    adaptor = Weaviate(dim=2, metric=Metric.COSINE, base_url="http://example.com", class_name="TestClass")
    stub = StubSession("TestClass")

    def failing_post(url: str, json: dict | None = None, timeout: float | None = None):  # noqa: ARG002
        if url.endswith("/v1/schema"):
            return FakeResponse(500, text="cannot create")
        return FakeResponse(200)

    stub.post = failing_post  # type: ignore[assignment]
    adaptor._session = stub  # type: ignore[attr-defined]
    with pytest.raises(WeaviateError):
        adaptor.train_index(np.zeros((1, 2), dtype=np.float32))


def test_upsert_delete_tolerates_404(adaptor_with_stub):
    """404 deletes should be treated as successful no-ops."""
    adaptor, stub = adaptor_with_stub
    adaptor._class_exists = lambda: True  # type: ignore[attr-defined]

    def delete_returning_404(url: str, timeout: float | None = None) -> FakeResponse:  # noqa: ARG002
        return FakeResponse(404)

    stub.delete = delete_returning_404  # type: ignore[assignment]
    adaptor.upsert(np.array([1], dtype=np.int64), np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
    adaptor.delete([1])


def test_delete_raises_on_failure(adaptor_with_stub):
    """Non-success delete responses must raise."""
    adaptor, stub = adaptor_with_stub

    def failing_delete(url: str, timeout: float | None = None) -> FakeResponse:  # noqa: ARG002
        return FakeResponse(500, text="delete boom")

    stub.delete = failing_delete  # type: ignore[assignment]
    with pytest.raises(WeaviateError):
        adaptor.delete([1])


def test_search_raises_on_bad_status(adaptor_with_stub):
    """Non-200 GraphQL responses should bubble up as errors."""
    adaptor, stub = adaptor_with_stub

    def bad_status_post(url: str, json: dict | None = None, timeout: float | None = None):  # noqa: ARG002
        if url.endswith("/v1/graphql"):
            return FakeResponse(500, text="bad")
        return FakeResponse(200)

    stub.post = bad_status_post  # type: ignore[assignment]
    with pytest.raises(WeaviateError):
        adaptor.search(np.zeros((1, 3), dtype=np.float32), topk=1)


def test_search_handles_invalid_uuid_fallback(adaptor_with_stub):
    """Invalid UUIDs should fall back to -1 identifiers without crashing."""
    adaptor, stub = adaptor_with_stub

    def graphql_payload(url: str, json: dict | None = None, timeout: float | None = None):  # noqa: ARG002
        if url.endswith("/v1/graphql"):
            data = {
                "data": {
                    "Get": {adaptor.class_name: [{"_additional": {"id": "not-a-uuid", "distance": 0.2}}]}
                }
            }
            return FakeResponse(200, data)
        return FakeResponse(200)

    stub.post = graphql_payload  # type: ignore[assignment]
    distances, ids = adaptor.search(np.zeros((1, 3), dtype=np.float32), topk=1)
    assert np.isfinite(distances[0, 0])
    assert ids[0, 0] == -1


def test_stats_raises_on_bad_status(adaptor_with_stub):
    """Non-200 aggregate responses should raise."""
    adaptor, stub = adaptor_with_stub

    def bad_status_post(url: str, json: dict | None = None, timeout: float | None = None):  # noqa: ARG002
        if url.endswith("/v1/graphql"):
            return FakeResponse(500, text="bad")
        return FakeResponse(200)

    stub.post = bad_status_post  # type: ignore[assignment]
    with pytest.raises(WeaviateError):
        adaptor.stats()


def test_stats_handles_empty_entries(adaptor_with_stub):
    """Empty aggregate entries should return ntotal=0."""
    adaptor, stub = adaptor_with_stub

    def empty_aggregate_post(url: str, json: dict | None = None, timeout: float | None = None):  # noqa: ARG002
        if url.endswith("/v1/graphql"):
            data = {"data": {"Aggregate": {adaptor.class_name: []}}}
            return FakeResponse(200, data)
        return FakeResponse(200)

    stub.post = empty_aggregate_post  # type: ignore[assignment]
    stats = adaptor.stats()
    assert stats["ntotal"] == 0


def test_constructor_and_distance_mapping():
    """Ensure basic constructor validation and metric mapping."""
    with pytest.raises(ValueError):
        _ = Weaviate(dim=0, metric=Metric.COSINE)

    adaptor_ip = Weaviate(dim=2, metric=Metric.INNER_PRODUCT)
    assert adaptor_ip._distance_metric == "dot"

    adaptor_l2 = Weaviate(dim=2, metric=Metric.L2)
    assert adaptor_l2._distance_metric == "l2-squared"


def test_validate_uuid_helpers():
    """Recover helper should accept UUIDs and reject malformed values."""
    valid_uuid = str(uuid.uuid4())
    assert Weaviate._validate_uuid(valid_uuid) == -1

    with pytest.raises(ValueError):
        Weaviate._validate_uuid("not-a-uuid")


def test_upsert_raises_when_delete_fails(adaptor_with_stub):
    """Upsert should raise if the delete phase returns a non-success status."""
    adaptor, stub = adaptor_with_stub
    adaptor._class_exists = lambda: True  # type: ignore[attr-defined]

    def failing_delete(url: str, timeout: float | None = None) -> FakeResponse:  # noqa: ARG002
        return FakeResponse(500, text="delete boom")

    stub.delete = failing_delete  # type: ignore[assignment]
    with pytest.raises(WeaviateError):
        adaptor.upsert(np.array([1], dtype=np.int64), np.array([[1.0, 0.0, 0.0]], dtype=np.float32))


def test_upsert_raises_when_insert_fails(adaptor_with_stub):
    """Upsert should raise when the insert POST returns an error."""
    adaptor, stub = adaptor_with_stub
    adaptor._class_exists = lambda: True  # type: ignore[attr-defined]

    def failing_post(url: str, json: dict | None = None, timeout: float | None = None):  # noqa: ARG002
        if url.endswith("/v1/objects"):
            return FakeResponse(500, text="insert boom")
        return FakeResponse(200)

    stub.post = failing_post  # type: ignore[assignment]
    with pytest.raises(WeaviateError):
        adaptor.upsert(np.array([1], dtype=np.int64), np.array([[1.0, 0.0, 0.0]], dtype=np.float32))


def test_check_ready_with_zero_timeout():
    """Zero timeout should immediately raise when readiness never returns 200."""
    adaptor = Weaviate(
        dim=2, metric=Metric.COSINE, base_url="http://example.com", class_name="TestClass", timeout=0.0
    )
    stub = StubSession("TestClass")
    adaptor._session = stub  # type: ignore[attr-defined]
    with pytest.raises(WeaviateError):
        adaptor._check_ready()
