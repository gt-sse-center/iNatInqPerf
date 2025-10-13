"""Unit tests for the Weaviate adaptor using stubbed HTTP sessions."""

from collections import defaultdict
import uuid

import numpy as np
import requests
from unittest import mock
import pytest

from inatinqperf.adaptors.metric import Metric
from inatinqperf.adaptors.weaviate_adaptor import (
    Weaviate,
    WeaviateError,
    HuggingFaceDataset,
    DataPoint,
    Query,
)


class FakeResponse:
    """Lightweight stand-in for requests.Response with JSON payload support."""

    def __init__(self, status_code: int = 200, json_data: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._json = json_data or {}
        self.text = text

    def json(self) -> dict:
        return self._json


class MockSession:
    """Mock emulating the requests.Session API used by the adaptor."""

    def __init__(self, class_name="TestClass", aggregate_count=0) -> None:
        self.class_name = class_name
        self.calls = defaultdict(list)
        self.graphql_results: list[dict] = []
        self.aggregate_count = aggregate_count

    def post(self, url: str, json: dict | None = None, timeout: float | None = None) -> FakeResponse:  # noqa: ARG002
        """Record POST requests and return canned responses."""
        if url.endswith("/v1/schema") or url.endswith("/v1/objects"):
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
        return FakeResponse(204)

    def get(self, url: str, timeout: float | None = None) -> FakeResponse:  # noqa: ARG002
        """Return ready/404 responses matching the adaptor's expectations."""
        # self.calls["get"].append(url)
        if url.endswith("/.well-known/ready"):
            return FakeResponse(200)
        if url.endswith("/v1/schema/TestClass"):
            return FakeResponse(404)
        return FakeResponse(200)


@pytest.fixture(name="adaptor_with_stub")
def adaptor_with_stub_fixture(monkeypatch):
    """Provide a Weaviate adaptor wired to the stub session for unit tests."""
    # Minimal stub adaptor -> every test reuses this to avoid real HTTP calls.
    adaptor = Weaviate(dim=3, metric=Metric.COSINE, base_url="http://example.com", class_name="TestClass")
    stub = StubSession("TestClass")
    monkeypatch.setattr(adaptor, "_session", stub)
    monkeypatch.setattr(adaptor, "check_ready", lambda: None)
    monkeypatch.setattr(adaptor, "_class_exists", lambda: True)

    return adaptor, stub


def test_create_class(mocker, dataset, caplog):
    """Creating the class should POST to the schema endpoint."""
    mocker.patch.object(Weaviate, "_class_exists", lambda self: False)

    class_name = "TestClass"
    adaptor = Weaviate(dataset, metric=Metric.COSINE, base_url="http://example.com", class_name=class_name)

    adaptor._session.post.assert_called()
    assert f"Created class={class_name}" in caplog.text
    assert f"Initialized class={class_name}" in caplog.text


def test_train_index_ignores_existing_class(monkeypatch):
    """Ensure 422 from Weaviate (already exists) is treated as success."""

    def already_exists_post(self, url, json=None, timeout=None):  # type: ignore[arg-type]
        # self.calls["post"].append((url, json))
        return FakeResponse(422, text="Already exists")

    stub.post = already_exists_post  # type: ignore[assignment]
    monkeypatch.setattr(adaptor, "_session", stub)

    adaptor.train_index(np.zeros((1, 3), dtype=np.float32))
    assert any(call[0].endswith("/v1/schema") for call in stub.calls["post"])


def test_upsert_and_search(adaptor, mocker):
    """Upsert should succeed and search should surface the deterministic ids."""
    adaptor, stub = adaptor_with_stub

    mock_session = MockSession()
    mock_session.graphql_results = [
        {"originalId": 2, "_additional": {"id": "00000000-0000-0000-0000-000000000002", "distance": 0.1}},
        {"originalId": 1, "_additional": {"id": "00000000-0000-0000-0000-000000000001", "distance": 0.9}},
    ]
    query = np.array([0.1, 1.0, 0.0], dtype=np.float32)
    results = adaptor.search(query, topk=2)
    assert len(results) == 2
    assert [r.id for r in results] == [2, 1]


def test_stats_returns_count(adaptor, mocker):
    """Stats should reflect the aggregate count returned by GraphQL."""
    adaptor, stub = adaptor_with_stub

    stub.aggregate_count = 5
    stats = adaptor.stats()
    assert stats["ntotal"] == 5
    assert stats["class_name"] == "TestClass"


def test_stats_handles_graphql_error(adaptor, mocker):
    """Stats should handle GraphQL "errors" payloads by returning zero."""

    def graphql_error_post(url: str, json: dict | None = None, timeout: float | None = None):  # noqa: ARG002
        if url.endswith("/v1/graphql"):
            return FakeResponse(200, {"errors": [{"message": "boom"}]})
        return FakeResponse(200)

    mocker.patch.object(adaptor._session, "post", side_effect=graphql_error_post)

    stats = adaptor.stats()
    assert stats["ntotal"] == 0


def test_delete_multiple_ids(adaptor):
    """Deleting multiple ids should issue one request per id."""
    adaptor.delete([10, 20])

    adaptor._session.delete.assert_called()
    assert adaptor._session.delete.call_count == 2


def test_invalid_metric_raises_value_error(dataset):
    """Constructor should reject unsupported distance metrics."""
    with pytest.raises(ValueError):
        Weaviate(dataset, metric="manhattan")


def test_upsert_rejects_dimension_mismatch(adaptor):
    data_points = [DataPoint(id=1, vector=np.array([[1.0, 2.0]], dtype=np.float32), metadata={})]
    with pytest.raises(ValueError):
        adaptor.upsert(data_points)


def test_upsert_rejects_length_mismatch(adaptor):
    """ids and vectors length mismatch should raise."""
    data_points = [
        DataPoint(id=1, vector=np.array([[1.0, 0.0, 0.0]], dtype=np.float32), metadata={}),
        DataPoint(id=2, vector=[], metadata={}),
    ]
    with pytest.raises(ValueError):
        adaptor.upsert(data_points)


def test_search_rejects_invalid_queries(adaptor):
    with pytest.raises(ValueError):
        q = Query(np.array([1.0, 2.0], dtype=np.float32))
        adaptor.search(q, topk=1)

    with pytest.raises(ValueError):
        q = Query(np.zeros((1, 3), dtype=np.float32))
        adaptor.search(q, topk=0)


def test_search_handles_graphql_errors(adaptor, mocker):
    """GraphQL errors should return placeholder results instead of raising."""

    def graphql_error_post(url: str, json: dict | None = None, timeout: float | None = None):  # noqa: ARG002
        if url.endswith("/v1/graphql"):
            return FakeResponse(200, {"errors": [{"message": "no results"}]})
        return FakeResponse(200)

    stub.post = graphql_error_post  # type: ignore[assignment]
    results = adaptor.search(np.zeros((3,), dtype=np.float32), topk=2)

    distances = np.asarray([r.score for r in results])
    assert np.isinf(distances).all()

    ids = np.asarray([r.id for r in results])
    assert (ids == -1).all()


def test_error_responses_raise_weaviate_error(adaptor_with_stub):
    """Non-success POST responses should raise WeaviateError."""
    adaptor, stub = adaptor_with_stub

    def failing_post(url: str, json: dict | None = None, timeout: float | None = None):  # noqa: ARG002
        return FakeResponse(500, text="boom")

    stub.post = failing_post  # type: ignore[assignment]
    with pytest.raises(WeaviateError):
        adaptor.train_index(np.zeros((1, 3), dtype=np.float32))


def test_drop_index_raises_on_failure(monkeypatch):
    """DELETE failures should propagate as WeaviateError."""
    adaptor = Weaviate(dim=2, metric=Metric.COSINE, base_url="http://example.com", class_name="TestClass")
    stub = StubSession("TestClass")
    stub.delete = lambda url, timeout=None: FakeResponse(500, text="kaboom")  # type: ignore[arg-type,assignment]
    monkeypatch.setattr(adaptor, "_session", stub)

    with pytest.raises(WeaviateError):
        adaptor.drop_index()


def test_drop_index_success(monkeypatch):
    """Successful drop should issue a delete call."""
    adaptor = Weaviate(dim=2, metric=Metric.COSINE, base_url="http://example.com", class_name="TestClass")
    stub = StubSession("TestClass")
    monkeypatch.setattr(adaptor, "_session", stub)

    adaptor.drop_index()
    assert stub.calls["delete"]


def test_check_ready_times_out_quickly(monkeypatch):
    """Readiness polling should surface repeated non-200 responses."""
    adaptor = Weaviate(
        dataset, metric=Metric.COSINE, base_url="http://example.com", class_name="TestClass", timeout=0.6
    )

    def failing_get(url, timeout=None):  # noqa: ARG002
        return FakeResponse(503, text="not ready")

    stub.get = failing_get  # type: ignore[assignment]
    monkeypatch.setattr(adaptor, "_session", stub)

    with pytest.raises(WeaviateError):
        adaptor.check_ready()


def test_check_ready_success(monkeypatch):
    """Healthy readiness endpoint should exit the polling loop."""
    adaptor = Weaviate(
        dataset, metric=Metric.COSINE, base_url="http://example.com", class_name="TestClass", timeout=1.0
    )

    def ready_get(url, timeout=None):  # noqa: ARG002
        return FakeResponse(200)

    stub.get = ready_get  # type: ignore[assignment]
    monkeypatch.setattr(adaptor, "_session", stub)

    adaptor.check_ready()


def test_class_exists_paths(monkeypatch):
    """Class existence helper should handle 404/200 and error paths."""
    adaptor = Weaviate(dataset, metric=Metric.COSINE, base_url="http://example.com", class_name="TestClass")

    # Not found -> False
    monkeypatch.setattr(adaptor, "_session", stub)
    stub.get = lambda url, timeout=None: FakeResponse(404)  # type: ignore[assignment]
    assert adaptor._class_exists() is False

    # Found -> True
    mocker.patch.object(adaptor._session, "get", side_effect=lambda url, timeout=None: FakeResponse(200))
    assert adaptor._class_exists() is True

    # Unexpected status -> raises
    mocker.patch.object(
        adaptor._session, "get", side_effect=lambda url, timeout=None: FakeResponse(500, text="oops")
    )
    with pytest.raises(WeaviateError):
        adaptor._class_exists()


def test_train_index_raises_on_failed_creation(monkeypatch):
    """train_index should raise when schema creation fails."""
    adaptor = Weaviate(dim=2, metric=Metric.COSINE, base_url="http://example.com", class_name="TestClass")
    stub = StubSession("TestClass")

    def failing_post(url: str, json: dict | None = None, timeout: float | None = None):  # noqa: ARG002
        if url.endswith("/v1/schema"):
            return FakeResponse(500, text="cannot create")
        return FakeResponse(200)

    stub.post = failing_post  # type: ignore[assignment]
    monkeypatch.setattr(adaptor, "_session", stub)

    with pytest.raises(WeaviateError):
        adaptor.train_index(np.zeros((1, 2), dtype=np.float32))


def test_upsert_delete_tolerates_404(adaptor_with_stub):
    """404 deletes should be treated as successful no-ops."""
    adaptor, stub = adaptor_with_stub

    def delete_returning_404(url: str, timeout: float | None = None) -> FakeResponse:  # noqa: ARG002
        return FakeResponse(404)

    mocker.patch.object(adaptor._session, "delete", side_effect=delete_returning_404)

    data_points = [DataPoint(id=1, vector=np.array([1.0, 0.0, 0.0], dtype=np.float32), metadata={})]
    adaptor.upsert(data_points)

    try:
        adaptor.delete([1])
    except Exception:
        pytest.fail("Delete with 404 raised exception")


def test_delete_raises_on_failure(mocker, adaptor):
    """Non-successful delete responses must raise exception."""

    def failing_delete(url: str, timeout: float | None = None) -> FakeResponse:  # noqa: ARG002
        return FakeResponse(500, text="delete boom")

    mocker.patch.object(adaptor._session, "delete", side_effect=failing_delete)

    with pytest.raises(WeaviateError):
        adaptor.delete([1])


def test_search_raises_on_bad_status(mocker, adaptor):
    """Non-200 GraphQL responses should bubble up as errors."""

    def bad_status_post(url: str, json: dict | None = None, timeout: float | None = None):  # noqa: ARG002
        if url.endswith("/v1/graphql"):
            return FakeResponse(500, text="bad")
        return FakeResponse(200)

    mocker.patch.object(adaptor._session, "post", side_effect=bad_status_post)

    with pytest.raises(WeaviateError):
        adaptor.search(np.zeros((3,), dtype=np.float32), topk=1)


def test_search_handles_invalid_uuid_fallback(mocker, adaptor):
    """Invalid UUIDs should fall back to -1 identifiers without crashing."""

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
    results = adaptor.search(np.zeros((3,), dtype=np.float32), topk=1)
    assert np.isfinite(results[0].score)
    assert results[0].id == -1


def test_stats_raises_on_bad_status(mocker, adaptor):
    """Non-200 aggregate responses should raise."""

    def bad_status_post(url: str, json: dict | None = None, timeout: float | None = None):  # noqa: ARG002
        if url.endswith("/v1/graphql"):
            return FakeResponse(500, text="bad")
        return FakeResponse(200)

    mocker.patch.object(adaptor._session, "post", side_effect=bad_status_post)

    with pytest.raises(WeaviateError):
        adaptor.stats()


def test_stats_handles_empty_entries(mocker, adaptor):
    """Empty aggregate entries should return ntotal=0."""

    def empty_aggregate_post(url: str, json: dict | None = None, timeout: float | None = None):  # noqa: ARG002
        if url.endswith("/v1/graphql"):
            data = {"data": {"Aggregate": {adaptor.class_name: []}}}
            return FakeResponse(200, data)
        return FakeResponse(200)

    mocker.patch.object(adaptor._session, "post", side_effect=empty_aggregate_post)

    stats = adaptor.stats()
    assert stats["ntotal"] == 0


def test_constructor_and_distance_mapping(dataset):
    """Ensure basic constructor validation and metric mapping."""
    adaptor_ip = Weaviate(dataset, metric=Metric.INNER_PRODUCT)
    assert adaptor_ip.distance_metric == "dot"

    adaptor_ip = Weaviate(dim=2, metric=Metric.INNER_PRODUCT)
    assert adaptor_ip.distance_metric == "dot"

    adaptor_l2 = Weaviate(dim=2, metric=Metric.L2)
    assert adaptor_l2.distance_metric == "l2-squared"


def test_validate_uuid_helpers():
    """Recover helper should accept UUIDs and reject malformed values."""
    valid_uuid = str(uuid.uuid4())
    assert Weaviate._validate_uuid(valid_uuid) == -1

    with pytest.raises(ValueError):
        Weaviate._validate_uuid("not-a-uuid")


def test_upsert_raises_when_delete_fails(mocker, adaptor):
    """Upsert should raise if the delete phase returns a non-success status."""
    adaptor, stub = adaptor_with_stub

    def failing_delete(url: str, timeout: float | None = None) -> FakeResponse:  # noqa: ARG002
        return FakeResponse(500, text="delete boom")

    mocker.patch.object(adaptor._session, "delete", side_effect=failing_delete)

    with pytest.raises(WeaviateError):
        data_points = [DataPoint(id=1, vector=np.array([1.0, 0.0, 0.0], dtype=np.float32), metadata={})]
        adaptor.upsert(data_points)


def test_upsert_raises_when_insert_fails(mocker, adaptor):
    """Upsert should raise when the insert POST returns an error."""
    adaptor, stub = adaptor_with_stub

    def failing_post(url: str, json: dict | None = None, timeout: float | None = None):  # noqa: ARG002
        if url.endswith("/v1/objects"):
            return FakeResponse(500, text="insert boom")
        return FakeResponse(200)

    mocker.patch.object(adaptor._session, "post", side_effect=failing_post)

    with pytest.raises(WeaviateError):
        data_points = [DataPoint(id=1, vector=np.array([1.0, 0.0, 0.0], dtype=np.float32), metadata={})]
        adaptor.upsert(data_points)


def test_check_ready_with_zero_timeout(monkeypatch):
    """Zero timeout should immediately raise when readiness never returns 200."""
    adaptor = Weaviate(
        dim=2, metric=Metric.COSINE, base_url="http://example.com", class_name="TestClass", timeout=0.0
    )
    stub = StubSession("TestClass")
    monkeypatch.setattr(adaptor, "_session", stub)

    with pytest.raises(WeaviateError):
        adaptor.check_ready()
