"""Unit tests for the Weaviate adaptor using a stubbed client.

Each test focuses on a specific slice of the Weaviate client API:

* ``client.schema`` covers class creation, deletion, and existence checks.
* ``client.data_object`` simulates vector ingest and removal semantics.
* ``client.query`` exercises the GraphQL Get/Aggregate endpoints.
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest
from datasets import Dataset

from inatinqperf.adaptors.base import DataPoint, Query, SearchResult
from inatinqperf.adaptors.metric import Metric
from inatinqperf.adaptors.weaviate_adaptor import Weaviate, WeaviateError


class FakeStatusError(Exception):
    """Mimic weaviate exceptions that surface HTTP status codes."""

    def __init__(self, status_code: int, message: str = "") -> None:
        """Store the status code and optional message."""
        super().__init__(message or f"status {status_code}")
        self.status_code = status_code


class StubSchema:
    """Stub implementation of the client.schema interface."""

    def __init__(self, client: StubWeaviateClient) -> None:  # noqa: F821 - forward reference
        """Initialise storage for schema interactions."""
        self._client = client
        self.classes: dict[str, dict] = {}
        self.create_calls: list[dict] = []
        self.delete_calls: list[str] = []
        self.get_calls = 0
        self.raise_on_create: Exception | None = None
        self.raise_on_delete: Exception | None = None
        self.raise_on_get: Exception | None = None

    def create_class(self, payload: dict) -> None:
        """Record a schema creation request and return success unless configured otherwise."""
        self.create_calls.append(payload)
        if self.raise_on_create is not None:
            raise self.raise_on_create
        self.classes[payload["class"]] = payload

    def delete_class(self, class_name: str) -> None:
        """Delete the stored class metadata or raise if configured to fail."""
        self.delete_calls.append(class_name)
        if self.raise_on_delete is not None:
            raise self.raise_on_delete
        self.classes.pop(class_name, None)

    def get(self) -> dict:
        """Return the stored schema payload, raising if configured to do so."""
        self.get_calls += 1
        if self.raise_on_get is not None:
            raise self.raise_on_get
        return {"classes": list(self.classes.values())}


class StubDataObject:
    """Stub implementation of the client.data_object interface."""

    def __init__(self) -> None:
        """Initialise call tracking for create/delete operations."""
        self.create_calls: list[tuple[str, str, dict, list[float]]] = []
        self.delete_calls: list[tuple[str, str]] = []
        self.raise_on_create: Exception | None = None
        self.raise_on_delete: Exception | None = None

    def create(self, data_object: dict, class_name: str, uuid: str, vector: list[float]) -> None:  # noqa: A002
        """Store create parameters and optionally raise to simulate failure."""
        self.create_calls.append((class_name, uuid, data_object, vector))
        if self.raise_on_create is not None:
            raise self.raise_on_create

    def delete(self, uuid: str, class_name: str) -> None:  # noqa: A002
        """Record delete calls and optionally raise to simulate failure."""
        self.delete_calls.append((class_name, uuid))
        if self.raise_on_delete is not None:
            raise self.raise_on_delete


class StubQueryGetBuilder:
    """Query builder that simulates GraphQL GET requests."""

    def __init__(self, client: StubWeaviateClient, class_name: str, properties: list[str]) -> None:  # noqa: F821
        """Capture the client context and request metadata."""
        self._client = client
        self._class_name = class_name
        self._properties = properties
        self._near_vector: dict | None = None
        self._limit: int | None = None
        self._additional: list[str] | None = None
        self._where: dict | None = None

    def with_near_vector(self, payload: dict) -> StubQueryGetBuilder:
        """Record the near-vector payload and return self for chaining."""
        self._near_vector = payload
        return self

    def with_limit(self, limit: int) -> StubQueryGetBuilder:
        """Record the query limit and return self for chaining."""
        self._limit = limit
        return self

    def with_additional(self, additional: list[str]) -> StubQueryGetBuilder:
        """Record requested additional fields and return self for chaining."""
        self._additional = additional
        return self

    def with_where(self, where: dict) -> StubQueryGetBuilder:
        """Record the filter payload and return self for chaining."""
        self._where = where
        return self

    def do(self) -> dict:
        """Return the canned GraphQL response, recording the request details."""
        self._client.get_queries.append(
            {
                "class_name": self._class_name,
                "properties": self._properties,
                "near_vector": self._near_vector,
                "limit": self._limit,
                "additional": self._additional,
                "where": self._where,
            }
        )
        if self._client.graphql_get_exception is not None:
            raise self._client.graphql_get_exception
        return self._client.graphql_get_response


class StubQueryAggregateBuilder:
    """Query builder that simulates GraphQL aggregate requests."""

    def __init__(self, client: StubWeaviateClient, class_name: str) -> None:  # noqa: F821
        """Capture the client context and requested class name."""
        self._client = client
        self._class_name = class_name
        self._meta_count = False

    def with_meta_count(self) -> StubQueryAggregateBuilder:
        """Flag that meta count was requested and return self."""
        self._meta_count = True
        return self

    def do(self) -> dict:
        """Return the canned aggregate payload, recording request metadata."""
        self._client.aggregate_queries.append(
            {"class_name": self._class_name, "meta_count": self._meta_count}
        )
        if self._client.graphql_aggregate_exception is not None:
            raise self._client.graphql_aggregate_exception
        return self._client.graphql_aggregate_response


class StubQuery:
    """Stub implementation of the client.query interface."""

    def __init__(self, client: StubWeaviateClient) -> None:  # noqa: F821
        """Store the shared stub client."""
        self._client = client

    def get(self, class_name: str, properties: list[str]) -> StubQueryGetBuilder:
        """Return a builder for GET queries."""
        return StubQueryGetBuilder(self._client, class_name, properties)

    def aggregate(self, class_name: str) -> StubQueryAggregateBuilder:
        """Return a builder for aggregate queries."""
        return StubQueryAggregateBuilder(self._client, class_name)


class StubWeaviateClient:
    """Aggregate stub matching the subset of the weaviate client API used by the adaptor."""

    def __init__(self, class_name: str) -> None:
        """Initialise stub sub-clients and canned responses."""
        self.schema = StubSchema(self)
        self.data_object = StubDataObject()
        self.query = StubQuery(self)
        self.get_queries: list[dict] = []
        self.aggregate_queries: list[dict] = []
        self.graphql_get_response: dict = {"data": {"Get": {class_name: []}}}
        self.graphql_get_exception: Exception | None = None
        self.graphql_aggregate_response: dict = {
            "data": {"Aggregate": {class_name: [{"meta": {"count": 0}}]}}
        }
        self.graphql_aggregate_exception: Exception | None = None
        self.ready_responses: list[bool | Exception] = [True]

    def is_ready(self) -> bool:
        """Return the next readiness response, raising if configured to do so."""
        if not self.ready_responses:
            return True
        next_value = self.ready_responses.pop(0)
        if isinstance(next_value, Exception):
            raise next_value
        return bool(next_value)


def make_dataset(dim: int) -> Dataset:
    """Construct a minimal dataset with a single embedding vector of the given dimensionality."""

    return Dataset.from_dict({"embedding": [[0.0] * dim]})


@pytest.fixture(name="stub_client")
def stub_client_fixture() -> StubWeaviateClient:
    """Provide a fresh stub client for each test."""
    return StubWeaviateClient("TestClass")


@pytest.fixture(name="dataset")
def dataset_fixture() -> Dataset:
    """Supply a minimal dataset for adaptor construction."""
    return make_dataset(3)


@pytest.fixture(name="adaptor")
def adaptor_fixture(dataset: Dataset, stub_client: StubWeaviateClient) -> tuple[Weaviate, StubWeaviateClient]:
    """Construct a Weaviate adaptor bound to the stub client."""
    adaptor = Weaviate(
        dataset=dataset,
        metric=Metric.COSINE,
        url="http://example.com",
        class_name="TestClass",
        client=stub_client,
    )
    return adaptor, stub_client


def test_ensure_schema_exists_creates_class(dataset: Dataset) -> None:
    """_ensure_schema_exists should create the class when it does not already exist."""
    client = StubWeaviateClient("TestClass")
    adaptor = Weaviate(
        dataset=dataset,
        metric=Metric.COSINE,
        url="http://example.com",
        class_name="TestClass",
        client=client,
    )
    adaptor._ensure_schema_exists()
    assert client.schema.create_calls


def test_ensure_schema_exists_tolerates_already_exists_error(dataset: Dataset) -> None:
    """_ensure_schema_exists should tolerate a 422 already-exists response."""
    client = StubWeaviateClient("TestClass")
    client.schema.raise_on_create = FakeStatusError(422, "already exists")
    adaptor = Weaviate(
        dataset=dataset,
        metric=Metric.COSINE,
        url="http://example.com",
        class_name="TestClass",
        client=client,
    )
    adaptor._ensure_schema_exists()
    assert client.schema.create_calls


def test_ensure_schema_exists_ignores_existing_class(dataset: Dataset) -> None:
    """_ensure_schema_exists should tolerate a 422 already-exists response."""
    client = StubWeaviateClient("TestClass")
    client.schema.raise_on_create = FakeStatusError(422, "already exists")
    adaptor = Weaviate(
        dataset=dataset,
        metric=Metric.COSINE,
        url="http://example.com",
        class_name="TestClass",
        client=client,
    )
<<<<<<< HEAD
    adaptor.train_index(np.zeros((1, 3), dtype=np.float32))
    assert client.schema.create_calls
>>>>>>> 12120e9 (ruff check due to online merge)


def test_ensure_schema_exists_tolerates_already_exists_error(dataset: Dataset) -> None:
    """_ensure_schema_exists should tolerate a 422 already-exists response."""
    client = StubWeaviateClient("TestClass")
    client.schema.raise_on_create = FakeStatusError(422, "already exists")
    adaptor = Weaviate(
        dataset=dataset,
        metric=Metric.COSINE,
        url="http://example.com",
        class_name="TestClass",
        client=client,
    )
=======
>>>>>>> ea6d6fb (ruff check due to online merge)
    adaptor._ensure_schema_exists()
    assert client.schema.create_calls


def test_upsert_and_search_with_stub(adaptor: tuple[Weaviate, StubWeaviateClient]) -> None:
    """Upserting data points should make them retrievable via search."""
    weaviate_adaptor, client = adaptor
    datapoints = [
        DataPoint(id=1, vector=[1.0, 0.0, 0.0], metadata={"species": "a"}),
        DataPoint(id=2, vector=[0.0, 1.0, 0.0], metadata={"originalId": 2}),
    ]
    initial_creates = len(client.data_object.create_calls)
    weaviate_adaptor.upsert(datapoints)
    new_creates = len(client.data_object.create_calls) - initial_creates
    assert new_creates == len(datapoints)

    # GraphQL Get query should honour the nearVector ranking and expose _additional metadata.
    client.graphql_get_response = {
        "data": {
            "Get": {
                weaviate_adaptor.class_name: [
                    {
                        "originalId": 2,
                        "_additional": {"id": "00000000-0000-0000-0000-000000000002", "distance": 0.1},
                    },
                    {
                        "originalId": 1,
                        "_additional": {"id": "00000000-0000-0000-0000-000000000001", "distance": 0.9},
                    },
                ]
            }
        }
    }
    query = Query(vector=[0.1, 1.0, 0.0])
    results = weaviate_adaptor.search(query, topk=2)
    assert isinstance(results, list)
    assert all(isinstance(item, SearchResult) for item in results)
    assert [result.id for result in results] == [2, 1]


def test_search_short_result_list_requires_padding(adaptor: tuple[Weaviate, StubWeaviateClient]) -> None:
    """Weaviate can return fewer hits than requested; padding keeps downstream arrays consistent."""
    weaviate_adaptor, client = adaptor
    client.graphql_get_response = {
        "data": {
            "Get": {
                weaviate_adaptor.class_name: [
                    {
                        "originalId": 7,
                        "_additional": {
                            "id": str(uuid.uuid4()),
                            "distance": 0.42,
                        },
                    }
                ]
            }
        }
    }

    results = weaviate_adaptor.search(Query(vector=[0.2, 0.3, 0.5]), topk=3)
    assert len(results) == 1
    padded = np.full(3, -1.0, dtype=float)
    for idx, result in enumerate(results[:3]):
        padded[idx] = float(result.id)

    # Expect first position to match the single result and trailing slots to remain sentinel values.
    assert padded.tolist() == [7.0, -1.0, -1.0]

    container = np.full((1, 3), -1.0, dtype=float)
    container[0] = padded
    assert container[0, 0] == 7.0
    assert np.all(container[0, 1:] == -1.0)


def test_stats_returns_count(adaptor: tuple[Weaviate, StubWeaviateClient]) -> None:
    """stats should surface the meta count from the aggregate response."""
    weaviate_adaptor, client = adaptor
    client.graphql_aggregate_response = {
        "data": {"Aggregate": {weaviate_adaptor.class_name: [{"meta": {"count": 5}}]}}
    }
    stats = weaviate_adaptor.stats()
    assert stats["ntotal"] == 5
    assert stats["class_name"] == weaviate_adaptor.class_name


def test_stats_handles_graphql_error(adaptor: tuple[Weaviate, StubWeaviateClient]) -> None:
    """stats should return zero when GraphQL reports errors."""
    weaviate_adaptor, client = adaptor
    client.graphql_aggregate_response = {"errors": [{"message": "boom"}]}
    stats = weaviate_adaptor.stats()
    assert stats["ntotal"] == 0


def test_delete_handles_multiple_ids(adaptor: tuple[Weaviate, StubWeaviateClient]) -> None:
    """delete should invoke the client for each identifier."""
    weaviate_adaptor, client = adaptor
    weaviate_adaptor.delete([10, 20])
    recorded_ids = {uuid for _, uuid in client.data_object.delete_calls}
    expected_ids = {
        weaviate_adaptor._make_object_id(10),
        weaviate_adaptor._make_object_id(20),
    }
    assert expected_ids.issubset(recorded_ids)


def test_invalid_metric_raises_value_error(dataset: Dataset) -> None:
    """Constructing with an unsupported metric string should raise ValueError."""
    with pytest.raises(ValueError):
        Weaviate(dataset=dataset, metric="manhattan", url="http://example.com", class_name="TestClass")


def test_upsert_rejects_dimension_mismatch(adaptor: tuple[Weaviate, StubWeaviateClient]) -> None:
    """Upserting vectors of the wrong dimensionality must raise ValueError."""
    weaviate_adaptor, _ = adaptor
    bad_point = DataPoint(id=1, vector=[1.0, 2.0], metadata={})
    with pytest.raises(ValueError):
        weaviate_adaptor.upsert([bad_point])


def test_search_rejects_invalid_queries(adaptor: tuple[Weaviate, StubWeaviateClient]) -> None:
    """Search should validate query dimensionality and top-k bounds."""
    weaviate_adaptor, _ = adaptor
    with pytest.raises(ValueError):
        weaviate_adaptor.search(Query(vector=[1.0, 2.0]), topk=1)
    with pytest.raises(ValueError):
        weaviate_adaptor.search(Query(vector=[0.0, 0.0, 0.0]), topk=0)


def test_search_handles_graphql_errors(adaptor: tuple[Weaviate, StubWeaviateClient]) -> None:
    """GraphQL errors should return an empty result list."""
    weaviate_adaptor, client = adaptor
    # Simulate Weaviate returning an errors field rather than data.
    client.graphql_get_response = {"errors": [{"message": "no results"}]}
    results = weaviate_adaptor.search(Query(vector=[0.0, 0.0, 0.0]), topk=2)
    assert results == []


def test_search_applies_filters(adaptor: tuple[Weaviate, StubWeaviateClient]) -> None:
    """search should forward Query filters to the underlying GraphQL request."""
    weaviate_adaptor, client = adaptor
    filters = {"path": ["species"], "operator": "Equal", "valueString": "a"}
    results = weaviate_adaptor.search(Query(vector=[0.0, 0.0, 0.0], filters=filters), topk=1)
    assert results == []
    assert client.get_queries[-1]["where"] == filters


def test_error_responses_raise_weaviate_error(dataset: Dataset) -> None:
    """Constructor should surface schema creation failures immediately."""
    client = StubWeaviateClient("TestClass")
    client.schema.raise_on_create = FakeStatusError(500, "boom")
    with pytest.raises(WeaviateError):
        Weaviate(
            dataset=dataset,
            metric=Metric.COSINE,
            url="http://example.com",
            class_name="TestClass",
            client=client,
        )


def test_drop_index_raises_on_failure(adaptor: tuple[Weaviate, StubWeaviateClient]) -> None:
    """drop_index must surface errors when the client delete fails."""
    weaviate_adaptor, client = adaptor
    client.schema.raise_on_delete = FakeStatusError(500, "kaboom")
    with pytest.raises(WeaviateError):
        weaviate_adaptor.drop_index()


def test_drop_index_success(adaptor: tuple[Weaviate, StubWeaviateClient]) -> None:
    """drop_index should call the client delete when no errors occur."""
    weaviate_adaptor, client = adaptor
    weaviate_adaptor.drop_index()
    assert client.schema.delete_calls


def test_check_ready_times_out() -> None:
    """_check_ready should raise when readiness never returns success."""
    client = StubWeaviateClient("TestClass")
    adaptor = Weaviate(
        dataset=make_dataset(2),
        metric=Metric.COSINE,
        url="http://example.com",
        class_name="TestClass",
        timeout=0.2,
        client=client,
    )
    client.ready_responses = [Exception("not ready")] * 5
    with pytest.raises(WeaviateError):
        adaptor._check_ready()


def test_check_ready_success() -> None:
    """_check_ready should exit successfully when the client reports ready."""
    client = StubWeaviateClient("TestClass")
    client.ready_responses = [False, True]
    adaptor = Weaviate(
        dataset=make_dataset(2),
        metric=Metric.COSINE,
        url="http://example.com",
        class_name="TestClass",
        timeout=1.0,
        client=client,
    )
    adaptor._check_ready()
    assert not client.ready_responses  # Consumed all readiness responses from the stub.


def test_class_exists_paths() -> None:
    """_class_exists should handle found, not found, and error states."""
    client = StubWeaviateClient("TestClass")
    adaptor = Weaviate(
        dataset=make_dataset(2),
        metric=Metric.COSINE,
        url="http://example.com",
        class_name="TestClass",
        client=client,
    )

    adaptor._drop_class_if_exists()
    assert adaptor._class_exists() is False

    client.schema.classes["TestClass"] = {"class": "TestClass"}
    assert adaptor._class_exists() is True

    client.schema.raise_on_get = Exception("oops")
    with pytest.raises(WeaviateError):
        adaptor._class_exists()


def test_ensure_schema_exists_raises_on_failed_creation(dataset: Dataset) -> None:
    """Schema creation failures should propagate as WeaviateError."""
    client = StubWeaviateClient("TestClass")
    client.schema.raise_on_create = Exception("cannot create")
    with pytest.raises(WeaviateError):
        Weaviate(
            dataset=dataset,
            metric=Metric.COSINE,
            url="http://example.com",
            class_name="TestClass",
            client=client,
        )


def test_upsert_delete_tolerates_404(adaptor: tuple[Weaviate, StubWeaviateClient]) -> None:
    """delete should swallow 404 responses but still record the call."""
    weaviate_adaptor, client = adaptor
    client.data_object.raise_on_delete = FakeStatusError(404, "missing")
    weaviate_adaptor.upsert([DataPoint(id=1, vector=[1.0, 0.0, 0.0], metadata={})])
    weaviate_adaptor.delete([1])
    assert client.data_object.delete_calls  # at least one delete attempt recorded
    expected_id = weaviate_adaptor._make_object_id(1)
    assert client.data_object.delete_calls[-1][1] == expected_id


def test_delete_raises_on_failure(adaptor: tuple[Weaviate, StubWeaviateClient]) -> None:
    """delete should raise when the client reports a hard failure."""
    weaviate_adaptor, client = adaptor
    client.data_object.raise_on_delete = FakeStatusError(500, "delete boom")
    with pytest.raises(WeaviateError):
        weaviate_adaptor.delete([1])


def test_search_handles_invalid_uuid_fallback(adaptor: tuple[Weaviate, StubWeaviateClient]) -> None:
    """search results with invalid UUIDs should map ids to -1."""
    weaviate_adaptor, client = adaptor
    # Weaviate sometimes omits originalId; ensure we fall back to UUID validation.
    client.graphql_get_response = {
        "data": {
            "Get": {
                weaviate_adaptor.class_name: [
                    {"_additional": {"id": "not-a-uuid", "distance": 0.2}},
                ]
            }
        }
    }
    results = weaviate_adaptor.search(Query(vector=[0.0, 0.0, 0.0]), topk=1)
    assert np.isfinite(results[0].score)
    assert results[0].id == -1


def test_stats_raises_on_bad_status(adaptor: tuple[Weaviate, StubWeaviateClient]) -> None:
    """stats should raise when the aggregate query surfaces an error."""
    weaviate_adaptor, client = adaptor
    client.graphql_aggregate_exception = FakeStatusError(500, "bad")
    with pytest.raises(WeaviateError):
        weaviate_adaptor.stats()


def test_stats_handles_empty_entries(adaptor: tuple[Weaviate, StubWeaviateClient]) -> None:
    """stats should return zero when the aggregate result is empty."""
    weaviate_adaptor, client = adaptor
    client.graphql_aggregate_response = {"data": {"Aggregate": {weaviate_adaptor.class_name: []}}}
    stats = weaviate_adaptor.stats()
    assert stats["ntotal"] == 0


def test_stats_handles_empty_entries(adaptor: tuple[Weaviate, StubWeaviateClient]) -> None:
    """stats should return zero when the aggregate result is empty."""
    weaviate_adaptor, client = adaptor
    client.graphql_aggregate_response = {"data": {"Aggregate": {weaviate_adaptor.class_name: []}}}
    stats = weaviate_adaptor.stats()
    assert stats["ntotal"] == 0


def test_ingest_datapoints_uses_batch_context_manager() -> None:
    """_ingest_datapoints should leverage batch context managers when available."""

    class RecordingBatch:
        def __init__(self) -> None:
            self.batch_size: int | None = None
            self.records: list[tuple[str, str, dict, list[float]]] = []

        def __enter__(self) -> "RecordingBatch":
            return self

        def __exit__(self, exc_type, exc, traceback) -> bool:
            return False

        def add_data_object(
            self,
            data_object: dict,
            class_name: str,
            uuid: str,
            vector: list[float],
        ) -> None:
            self.records.append((class_name, uuid, data_object, vector))

    class ClientWithBatch(StubWeaviateClient):
        def __init__(self, class_name: str) -> None:
            super().__init__(class_name)
            self.batch = RecordingBatch()

    client = ClientWithBatch("TestClass")
    adaptor = Weaviate(
        dataset=make_dataset(3),
        metric=Metric.COSINE,
        url="http://example.com",
        class_name="TestClass",
        batch_size=3,
        client=client,
    )

    client.batch.records.clear()
    client.batch.batch_size = None
    datapoints = [DataPoint(id=idx, vector=[1.0, 0.0, 0.0], metadata={}) for idx in range(2)]
    adaptor._ingest_datapoints(datapoints)
    assert client.batch.batch_size == 3
    assert len(client.batch.records) == len(datapoints)


def test_ingest_datapoints_handles_batch_without_context() -> None:
    """_ingest_datapoints should call add_data_object on batch clients lacking __enter__."""

    class StatelessBatch:
        def __init__(self) -> None:
            self.batch_size: int | None = None
            self.records: list[tuple[str, str, dict, list[float]]] = []

        def add_data_object(
            self,
            data_object: dict,
            class_name: str,
            uuid: str,
            vector: list[float],
        ) -> None:
            self.records.append((class_name, uuid, data_object, vector))

    class ClientWithSimpleBatch(StubWeaviateClient):
        def __init__(self, class_name: str) -> None:
            super().__init__(class_name)
            self.batch = StatelessBatch()

    client = ClientWithSimpleBatch("TestClass")
    adaptor = Weaviate(
        dataset=make_dataset(3),
        metric=Metric.COSINE,
        url="http://example.com",
        class_name="TestClass",
        batch_size=2,
        client=client,
    )

    client.batch.records.clear()
    client.batch.batch_size = None
    datapoints = [DataPoint(id=idx, vector=[0.0, 1.0, 0.0], metadata={}) for idx in range(2)]
    adaptor._ingest_datapoints(datapoints)
    assert client.batch.batch_size == 2
    assert len(client.batch.records) == len(datapoints)


def test_ingest_datapoints_falls_back_to_upsert(monkeypatch: pytest.MonkeyPatch) -> None:
    """_ingest_datapoints should fall back to upsert when batch ingest fails."""

    class FailingBatch:
        def __init__(self) -> None:
            self.batch_size: int | None = None

        def __enter__(self) -> "FailingBatch":
            return self

        def __exit__(self, exc_type, exc, traceback) -> bool:
            return False

        def add_data_object(self, *_, **__) -> None:
            raise RuntimeError("boom")

    class ClientWithFailingBatch(StubWeaviateClient):
        def __init__(self, class_name: str) -> None:
            super().__init__(class_name)
            self.batch = FailingBatch()

    client = ClientWithFailingBatch("TestClass")
    adaptor = Weaviate(
        dataset=make_dataset(3),
        metric=Metric.COSINE,
        url="http://example.com",
        class_name="TestClass",
        batch_size=4,
        client=client,
    )

    recorded: dict[str, list[DataPoint]] = {}

    def fake_upsert(self: Weaviate, datapoints: list[DataPoint]) -> None:
        recorded["datapoints"] = list(datapoints)

    monkeypatch.setattr(adaptor, "upsert", MethodType(fake_upsert, adaptor))
    datapoints = [DataPoint(id=idx, vector=[0.0, 0.0, 1.0], metadata={}) for idx in range(3)]
    adaptor._ingest_datapoints(datapoints)
    assert recorded["datapoints"] == datapoints


def test_add_batch_object_requires_add_method(adaptor: tuple[Weaviate, StubWeaviateClient]) -> None:
    """_add_batch_object should raise when the batch client lacks add_data_object."""
    weaviate_adaptor, _ = adaptor

    class MissingAdd:
        pass

    datapoint = DataPoint(id=1, vector=[1.0, 0.0, 0.0], metadata={})
    with pytest.raises(TypeError):
        weaviate_adaptor._add_batch_object(MissingAdd(), datapoint)


<<<<<<< HEAD
def test_ingest_datapoints_batch_context_manager(
    adaptor: tuple[Weaviate, StubWeaviateClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    """_ingest_datapoints should honour batch context managers and ingest all datapoints."""
=======
def test_bulk_ingest_with_context_manager(adaptor: tuple[Weaviate, StubWeaviateClient]) -> None:
    """_bulk_ingest should honour batch context managers and ingest all datapoints."""
>>>>>>> ea6d6fb (ruff check due to online merge)
    weaviate_adaptor, _ = adaptor

    class RecordingBatch:
        def __init__(self) -> None:
            self.batch_size: int | None = None
            self.records: list[tuple[str, str, dict, list[float]]] = []

        def __enter__(self) -> "RecordingBatch":
            return self

        def __exit__(self, exc_type, exc, traceback) -> bool:
            return False

        def add_data_object(
            self,
            data_object: dict,
            class_name: str,
            uuid: str,
            vector: list[float],
        ) -> None:
            self.records.append((class_name, uuid, data_object, vector))

    batch = RecordingBatch()
<<<<<<< HEAD
    weaviate_adaptor._client.batch = batch
    weaviate_adaptor._batch_size = 4
    datapoints = [DataPoint(id=i, vector=[1.0, 0.0, 0.0], metadata={}) for i in range(3)]

    upsert_tracker: dict[str, list[DataPoint]] = {}

    def fake_upsert(self: Weaviate, payload: list[DataPoint]) -> None:
        upsert_tracker["datapoints"] = list(payload)

    monkeypatch.setattr(weaviate_adaptor, "upsert", MethodType(fake_upsert, weaviate_adaptor))

    weaviate_adaptor._ingest_datapoints(datapoints)

    assert "datapoints" not in upsert_tracker
=======
    weaviate_adaptor._batch_size = 4
    datapoints = [DataPoint(id=i, vector=[1.0, 0.0, 0.0], metadata={}) for i in range(3)]

    result = weaviate_adaptor._bulk_ingest(datapoints, batch)
    assert result is True
>>>>>>> ea6d6fb (ruff check due to online merge)
    assert batch.batch_size == 4
    assert len(batch.records) == len(datapoints)


<<<<<<< HEAD
def test_ingest_datapoints_batch_without_context(
    adaptor: tuple[Weaviate, StubWeaviateClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    """_ingest_datapoints should work with batch clients lacking context manager support."""
=======
def test_bulk_ingest_without_context_manager(adaptor: tuple[Weaviate, StubWeaviateClient]) -> None:
    """_bulk_ingest should work with batch clients lacking context manager support."""
>>>>>>> ea6d6fb (ruff check due to online merge)
    weaviate_adaptor, _ = adaptor

    class StatelessBatch:
        def __init__(self) -> None:
            self.batch_size: int | None = None
            self.records: list[tuple[str, str, dict, list[float]]] = []

        def add_data_object(
            self,
            data_object: dict,
            class_name: str,
            uuid: str,
            vector: list[float],
        ) -> None:
            self.records.append((class_name, uuid, data_object, vector))

    batch = StatelessBatch()
<<<<<<< HEAD
    weaviate_adaptor._client.batch = batch
    weaviate_adaptor._batch_size = 2
    datapoints = [DataPoint(id=i, vector=[0.0, 1.0, 0.0], metadata={}) for i in range(2)]

    upsert_tracker: dict[str, list[DataPoint]] = {}

    def fake_upsert(self: Weaviate, payload: list[DataPoint]) -> None:
        upsert_tracker["datapoints"] = list(payload)

    monkeypatch.setattr(weaviate_adaptor, "upsert", MethodType(fake_upsert, weaviate_adaptor))

    weaviate_adaptor._ingest_datapoints(datapoints)

    assert "datapoints" not in upsert_tracker
=======
    weaviate_adaptor._batch_size = 2
    datapoints = [DataPoint(id=i, vector=[0.0, 1.0, 0.0], metadata={}) for i in range(2)]

    result = weaviate_adaptor._bulk_ingest(datapoints, batch)
    assert result is True
>>>>>>> ea6d6fb (ruff check due to online merge)
    assert batch.batch_size == 2
    assert len(batch.records) == len(datapoints)


<<<<<<< HEAD
def test_ingest_datapoints_batch_handles_exceptions(
    adaptor: tuple[Weaviate, StubWeaviateClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    """_ingest_datapoints should fall back to upsert when batch ingest fails."""
=======
def test_bulk_ingest_handles_exceptions(adaptor: tuple[Weaviate, StubWeaviateClient]) -> None:
    """_bulk_ingest should return False and avoid raising when ingestion fails."""
>>>>>>> ea6d6fb (ruff check due to online merge)
    weaviate_adaptor, _ = adaptor

    class ExplodingBatch:
        def __init__(self) -> None:
            self.batch_size: int | None = None

        def __enter__(self) -> "ExplodingBatch":
            return self

        def __exit__(self, exc_type, exc, traceback) -> bool:
            return False

        def add_data_object(self, *args, **kwargs) -> None:
            raise RuntimeError("boom")

    batch = ExplodingBatch()
<<<<<<< HEAD
    weaviate_adaptor._client.batch = batch
    weaviate_adaptor._batch_size = 5
    datapoints = [DataPoint(id=i, vector=[0.0, 0.0, 1.0], metadata={}) for i in range(3)]

    fallback: dict[str, list[DataPoint]] = {}

    def fake_upsert(self: Weaviate, payload: list[DataPoint]) -> None:
        fallback["datapoints"] = list(payload)

    monkeypatch.setattr(weaviate_adaptor, "upsert", MethodType(fake_upsert, weaviate_adaptor))

    weaviate_adaptor._ingest_datapoints(datapoints)

    assert fallback["datapoints"] == datapoints
=======
    weaviate_adaptor._batch_size = 5
    datapoints = [DataPoint(id=i, vector=[0.0, 0.0, 1.0], metadata={}) for i in range(3)]

    result = weaviate_adaptor._bulk_ingest(datapoints, batch)
    assert result is False
>>>>>>> ea6d6fb (ruff check due to online merge)


def test_constructor_and_distance_mapping(dataset):
    """Ensure basic constructor validation and metric mapping."""
    adaptor_ip = Weaviate(dataset, metric=Metric.INNER_PRODUCT)
    assert adaptor_ip.distance_metric == "dot"

    adaptor_ip = Weaviate(dim=2, metric=Metric.INNER_PRODUCT)
    assert adaptor_ip.distance_metric == "dot"

    adaptor_l2 = Weaviate(dataset, metric=Metric.L2)
    assert adaptor_l2.distance_metric == "l2-squared"


def test_validate_uuid_helpers():
    """Recover helper should accept UUIDs and reject malformed values."""
    valid_uuid = str(uuid.uuid4())
    assert Weaviate._validate_uuid(valid_uuid) == -1


def test_constructor_and_distance_mapping() -> None:
    """Constructor should validate metrics and map them to client values."""
    with pytest.raises(ValueError):
        _ = Weaviate(
            dataset=make_dataset(0),
            metric=Metric.COSINE,
            url="http://example.com",
            class_name="TestClass",
            client=StubWeaviateClient("TestClass"),
        )
    adaptor_ip = Weaviate(
        dataset=make_dataset(2),
        metric=Metric.INNER_PRODUCT,
        url="http://example.com",
        class_name="TestClass",
        client=StubWeaviateClient("TestClass"),
    )
    assert adaptor_ip._distance_metric == "dot"
    adaptor_l2 = Weaviate(
        dataset=make_dataset(2),
        metric=Metric.L2,
        url="http://example.com",
        class_name="TestClass",
        client=StubWeaviateClient("TestClass"),
    )
    assert adaptor_l2._distance_metric == "l2-squared"
