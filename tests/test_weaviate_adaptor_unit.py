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
from inatinqperf.adaptors.enums import Metric
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
        self.batch: object | None = None
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
        collection_name="TestClass",
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
        collection_name="TestClass",
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
        collection_name="TestClass",
        client=client,
    )
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
                weaviate_adaptor.collection_name: [
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


def test_stats_returns_count(adaptor: tuple[Weaviate, StubWeaviateClient]) -> None:
    """stats should surface the meta count from the aggregate response."""
    weaviate_adaptor, client = adaptor
    client.graphql_aggregate_response = {
        "data": {"Aggregate": {weaviate_adaptor.collection_name: [{"meta": {"count": 5}}]}}
    }
    stats = weaviate_adaptor.stats()
    assert stats["ntotal"] == 5
    assert stats["class_name"] == weaviate_adaptor.collection_name


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


def test_search_handles_graphql_errors(adaptor: tuple[Weaviate, StubWeaviateClient]) -> None:
    """GraphQL errors should return an empty result list."""
    weaviate_adaptor, client = adaptor
    # Simulate Weaviate returning an errors field rather than data.
    client.graphql_get_response = {"errors": [{"message": "no results"}]}
    results = weaviate_adaptor.search(Query(vector=[0.0, 0.0, 0.0]), topk=2)
    assert results == []


def test_error_responses_raise_weaviate_error(dataset: Dataset) -> None:
    """Constructor should surface schema creation failures immediately."""
    client = StubWeaviateClient("TestClass")
    client.schema.raise_on_create = FakeStatusError(500, "boom")
    with pytest.raises(WeaviateError):
        Weaviate(
            dataset=dataset,
            metric=Metric.COSINE,
            url="http://example.com",
            collection_name="TestClass",
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
        collection_name="TestClass",
        timeout=0.2,
        client=client,
    )
    client.ready_responses = [Exception("not ready")] * 5
    with pytest.raises(WeaviateError):
        adaptor._check_ready()


def test_search_handles_invalid_uuid_fallback(adaptor: tuple[Weaviate, StubWeaviateClient]) -> None:
    """search results with invalid UUIDs should map ids to -1."""
    weaviate_adaptor, client = adaptor
    # Weaviate sometimes omits originalId; ensure we fall back to UUID validation.
    client.graphql_get_response = {
        "data": {
            "Get": {
                weaviate_adaptor.collection_name: [
                    {"_additional": {"id": "not-a-uuid", "distance": 0.2}},
                ]
            }
        }
    }
    results = weaviate_adaptor.search(Query(vector=[0.0, 0.0, 0.0]), topk=1)
    assert np.isfinite(results[0].score)
    assert results[0].id == -1


def test_schema_creation_raises_on_failure(dataset: Dataset) -> None:
    """Schema provisioning errors should propagate as WeaviateError."""
    client = StubWeaviateClient("TestClass")
    client.schema.raise_on_create = Exception("cannot create")
    with pytest.raises(WeaviateError):
        Weaviate(
            dataset=dataset,
            metric=Metric.COSINE,
            url="http://example.com",
            collection_name="TestClass",
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


def test_search_raises_on_bad_status(adaptor: tuple[Weaviate, StubWeaviateClient]) -> None:
    """search should raise when the client surfaces an unexpected status."""
    weaviate_adaptor, client = adaptor
    client.graphql_get_exception = FakeStatusError(500, "bad")
    with pytest.raises(WeaviateError):
        weaviate_adaptor.search(Query(vector=[0.0, 0.0, 0.0]), topk=1)


def test_search_handles_invalid_uuid_fallback(adaptor: tuple[Weaviate, StubWeaviateClient]) -> None:
    """search results with invalid UUIDs should map ids to -1."""
    weaviate_adaptor, client = adaptor
    # Weaviate sometimes omits originalId; ensure we fall back to UUID validation.
    client.graphql_get_response = {
        "data": {
            "Get": {
                weaviate_adaptor.collection_name: [
                    {"_additional": {"id": "not-a-uuid", "distance": 0.2}},
                ]
            }
        }
    }
    results = weaviate_adaptor.search(Query(vector=[0.0, 0.0, 0.0]), topk=1)
    assert np.isfinite(results[0].score)
    assert results[0].id == -1


def test_ingest_datapoints_with_context_manager(adaptor: tuple[Weaviate, StubWeaviateClient]) -> None:
    """_ingest_datapoints should honour batch context managers."""
    weaviate_adaptor, client = adaptor

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
    client.batch = batch
    weaviate_adaptor._batch_size = 4
    initial_creates = len(client.data_object.create_calls)
    datapoints = [DataPoint(id=i, vector=[1.0, 0.0, 0.0], metadata={}) for i in range(3)]

    weaviate_adaptor._ingest_datapoints(datapoints)
    assert batch.batch_size == 4
    assert len(batch.records) == len(datapoints)
    assert len(client.data_object.create_calls) == initial_creates
    client.batch = None


def test_ingest_datapoints_falls_back_to_upsert(adaptor: tuple[Weaviate, StubWeaviateClient]) -> None:
    """_ingest_datapoints should fall back to upsert when batch ingest fails."""
    weaviate_adaptor, client = adaptor

    class ExplodingBatch:
        def __init__(self) -> None:
            self.batch_size: int | None = None

        def add_data_object(
            self,
            data_object: dict,
            class_name: str,
            uuid: str,
            vector: list[float],
        ) -> None:
            raise RuntimeError("boom")

    client.batch = ExplodingBatch()
    weaviate_adaptor._batch_size = 5
    datapoints = [DataPoint(id=i, vector=[0.0, 0.0, 1.0], metadata={}) for i in range(3)]

    initial_creates = len(client.data_object.create_calls)
    weaviate_adaptor._ingest_datapoints(datapoints)
    assert len(client.data_object.create_calls) - initial_creates == len(datapoints)
    client.batch = None


def test_stats_raises_on_bad_status(adaptor: tuple[Weaviate, StubWeaviateClient]) -> None:
    """stats should raise when the aggregate query surfaces an error."""
    weaviate_adaptor, client = adaptor
    client.graphql_aggregate_exception = FakeStatusError(500, "bad")
    with pytest.raises(WeaviateError):
        weaviate_adaptor.stats()


def test_stats_handles_empty_entries(adaptor: tuple[Weaviate, StubWeaviateClient]) -> None:
    """stats should return zero when the aggregate result is empty."""
    weaviate_adaptor, client = adaptor
    client.graphql_aggregate_response = {"data": {"Aggregate": {weaviate_adaptor.collection_name: []}}}
    stats = weaviate_adaptor.stats()
    assert stats["ntotal"] == 0


def test_validate_uuid_helpers():
    """Recover helper should accept UUIDs and reject malformed values."""
    valid_uuid = str(uuid.uuid4())
    assert Weaviate._validate_uuid(valid_uuid) == -1
