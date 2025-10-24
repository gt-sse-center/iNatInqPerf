"""Weaviate adaptor built on top of the official client library.

This adaptor keeps the surface area used by the benchmark intentionally small:

* Schema management is handled through ``client.schema``.
* Vector ingest happens via ``client.data_object`` calls.
* Queries rely on the GraphQL endpoint exposed through ``client.query``.

All interactions log high-level intent so that long-running benchmark sessions
remain debuggable and predictable for developers.
"""

import time
import uuid
from collections.abc import Sequence
from http import HTTPStatus

import numpy as np
from datasets import Dataset as HuggingFaceDataset
from loguru import logger
from weaviate import Client as WeaviateClient

from inatinqperf.adaptors.base import DataPoint, Query, SearchResult, VectorDatabase
from inatinqperf.adaptors.enums import Metric


class WeaviateError(RuntimeError):
    """Raised when Weaviate responds with an unexpected status code or payload."""


class Weaviate(VectorDatabase):
    """Light-weight wrapper for managing a single class within a Weaviate instance.

    In Weaviate, the vectorizer tells the server how it should turn objects into vectors when
    you ingest them. If you pick one of Weaviate builtin vectorizer modules (e.g., text2vec-transformers),
    Weaviate will look at specified text properties and produce vectors automatically. In this adaptor
    we set vectorizer="none" so Weaviate leaves the vectors entirely up to us. We push precomputed
    embeddings (data_object.create(..., vector=...)) and just use Weaviate for storage/search.
    """

    _ORIGINAL_ID_FIELD = "originalId"
    _EXPECTED_VECTOR_RANK = 2

    def __init__(
        self,
        dataset: HuggingFaceDataset,
        metric: Metric = Metric.COSINE,
        url: str = "http://localhost",
        collection_name: str = "collection_name",
        *,
        timeout: float = 10.0,
        vectorizer: str = "none",
        client: WeaviateClient | None = None,
        **options: object,
    ) -> None:
        """Initialise the adaptor with a dataset template and connectivity details."""
        super().__init__(dataset=dataset, metric=metric.value)

        self.url = url.rstrip("/")
        self.collection_name = collection_name
        self.timeout = timeout
        self.vectorizer = vectorizer
        self._batch_size = options.pop("batch_size", None)
        self._index_type = self._normalise_index_type(options.pop("index_type", None))
        self._distance_metric = self._translate_metric(metric)
        self._schema_ready = False

        self._client: WeaviateClient = client or WeaviateClient(
            url=self.url,
            timeout_config=(self.timeout, self.timeout),
        )

        self._drop_class_if_exists()
        self._ensure_schema_exists()
        self._ingest_dataset(dataset)

    # ------------------------------------------------------------------
    # VectorDatabase implementation
    # ------------------------------------------------------------------
    def drop_index(self) -> None:
        """Delete the managed class entirely."""
        try:
            self._client.schema.delete_class(self.collection_name)
        except Exception as exc:  # pragma: no cover - defensive programming
            status_code = self._status_code(exc)
            if status_code == HTTPStatus.NOT_FOUND:
                return
            self._handle_exception("failed to delete class", exc)

        self._schema_ready = False

    def upsert(self, x: Sequence[DataPoint]) -> None:
        """Insert or update vectors and associated metadata."""
        if len(x) == 0:
            return

        # Ensure the class exists. This is a no-op if already provisioned.
        if not self._schema_ready:
            self._ensure_schema_exists()

        for dp in x:
            vector_list = dp.vector

            properties = dp.metadata
            if properties is None:
                properties = {}
                dp.metadata = properties
            properties.setdefault(self._ORIGINAL_ID_FIELD, dp.id)

            obj_id = self._make_object_id(dp.id)

            try:
                self._client.data_object.delete(obj_id, class_name=self.collection_name)
            except Exception as exc:  # pragma: no cover - defensive programming
                status_code = self._status_code(exc)
                if status_code != HTTPStatus.NOT_FOUND:
                    self._handle_exception("failed to delete existing object", exc)

            try:
                self._client.data_object.create(
                    data_object=properties,
                    class_name=self.collection_name,
                    uuid=obj_id,
                    vector=vector_list,
                )
            except Exception as exc:  # pragma: no cover - defensive programming
                self._handle_exception("failed to upsert object", exc)

    def search(self, q: Query, topk: int, **kwargs) -> Sequence[SearchResult]:  # NOQA: ARG002
        """Search for the ``topk`` nearest vectors using Weaviate's GraphQL API."""

        builder = (
            self._client.query.get(self.collection_name, [self._ORIGINAL_ID_FIELD])
            .with_near_vector({"vector": q.vector})
            .with_limit(topk)
            .with_additional(["id", "distance"])
        )

        try:
            data = builder.do()
        except Exception as exc:  # pragma: no cover - defensive programming
            status_code = self._status_code(exc)
            if status_code == HTTPStatus.UNPROCESSABLE_ENTITY:
                logger.warning(f" search requested before schema exists (class={self.collection_name}).")
                self._schema_ready = False
                return []
            self._handle_exception("failed to execute search", exc)

        return self._extract_results(data)[:topk]

    def delete(self, ids: Sequence[int]) -> None:
        """Delete objects corresponding to the provided identifiers."""
        if not self._schema_ready:
            return
        for identifier in ids:
            self._delete_object(self._make_object_id(identifier))

    def stats(self) -> dict[str, object]:
        """Return summary statistics sourced via a Weaviate aggregation query."""

        try:
            data = self._client.query.aggregate(self.collection_name).with_meta_count().do()
        except Exception as exc:  # pragma: no cover - defensive programming
            status_code = self._status_code(exc)
            if status_code == HTTPStatus.UNPROCESSABLE_ENTITY:
                logger.warning(f" stats requested before schema exists (class={self.collection_name}).")
                self._schema_ready = False
                count = 0
            else:  # unexpected failure should propagate as WeaviateError
                self._handle_exception("failed to fetch stats", exc)
        else:
            count = self._extract_count(data)

        return {
            "ntotal": count,
            "metric": self.metric.value,
            "class_name": self.collection_name,
            "url": self.url,
            "dim": self.dim,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _check_ready(self) -> None:
        """Poll the readiness endpoint until Weaviate reports healthy."""
        deadline = time.time() + self.timeout
        last_error: Exception | None = None

        while time.time() < deadline:
            try:
                if self._client.is_ready():
                    return
            except Exception as exc:  # pragma: no cover - depends on client internals
                last_error = exc
            time.sleep(0.5)

        if last_error is None:
            msg = "weaviate instance readiness probe failed to connect"
            raise WeaviateError(msg)
        self._handle_exception("weaviate instance not ready", last_error)

    def _ensure_schema_exists(self) -> None:
        """Create the Weaviate class if it does not already exist."""
        self._check_ready()
        payload = self._schema_payload()
        try:
            self._client.schema.create_class(payload)
        except Exception as exc:  # pragma: no cover - defensive programming
            status_code = self._status_code(exc)
            if status_code == HTTPStatus.UNPROCESSABLE_ENTITY:
                self._schema_ready = True
                return
            self._handle_exception("failed to create class", exc)

        self._schema_ready = True

    def _drop_class_if_exists(self) -> None:
        """Drop the target class if it already exists."""
        self._schema_ready = False
        try:
            self._client.schema.delete_class(self.collection_name)
        except Exception as exc:  # pragma: no cover - defensive programming
            status_code = self._status_code(exc)
            if status_code != HTTPStatus.NOT_FOUND:
                self._handle_exception("failed to drop existing class", exc)

    def _schema_payload(self) -> dict[str, object]:
        """Return the schema definition submitted to Weaviate."""

        return {
            "class": self.collection_name,
            "description": f"{self.collection_name}_iNatInqPerf",
            "vectorizer": self.vectorizer,
            "vectorIndexType": self._index_type,
            "vectorIndexConfig": {
                "distance": self._distance_metric,
                "vectorCacheMaxObjects": 1_000_000,
            },
            "properties": [
                {
                    "name": self._ORIGINAL_ID_FIELD,
                    "description": "Original integer identifier",
                    "dataType": ["int"],
                }
            ],
        }

    def _ingest_dataset(self, dataset: HuggingFaceDataset) -> None:
        """Load existing dataset vectors into the managed Weaviate class."""
        if "embedding" not in dataset.column_names:
            logger.warning(" dataset missing 'embedding' column; skipping ingest")
            return

        embeddings = np.asarray(dataset["embedding"], dtype=np.float32)
        if embeddings.size == 0:
            logger.warning(" dataset contains no embeddings; skipping ingest")
            return

        if "id" in dataset.column_names:
            ids = dataset["id"]
            if len(ids) != len(embeddings):
                msg = "Length of dataset ids must match number of embeddings"
                raise ValueError(msg)
        else:
            ids = list(range(len(embeddings)))

        datapoints = [
            DataPoint(id=idx, vector=vec, metadata={}) for idx, vec in zip(ids, embeddings, strict=False)
        ]

        self._ingest_datapoints(datapoints)
        try:
            post_stats = self.stats()
            if post_stats.get("ntotal", 0) < len(datapoints):
                post_stat_ntotal = len(datapoints) - post_stats.get("ntotal")
                logger.warning(f" {post_stat_ntotal}  smaller than ingested count {len(datapoints)}")
        except WeaviateError:
            logger.warning(" Unable to fetch stats immediately after ingest")

    def _ingest_datapoints(self, datapoints: Sequence[DataPoint]) -> None:
        """Ingest datapoints via batch API when configured, otherwise fall back to upsert."""
        if len(datapoints) == 0:
            return

        batch_setting = self._batch_size
        try:
            use_batch = int(batch_setting) > 0
        except (TypeError, ValueError):
            use_batch = False

        if use_batch:
            batch_client = self._client.batch
            if batch_client is not None and self._ingest_via_batch(batch_client, datapoints):
                return
        logger.warning(" Batch ingest failed; falling back to upsert.")
        self.upsert(datapoints)

    def _add_batch_object(self, batch: object, datapoint: DataPoint) -> None:
        """Helper to append a datapoint to a Weaviate batch request."""
        obj_id = self._make_object_id(datapoint.id)
        properties = datapoint.metadata
        if properties is None:
            properties = {}
            datapoint.metadata = properties
        properties.setdefault(self._ORIGINAL_ID_FIELD, datapoint.id)

        batch.add_data_object(
            data_object=properties,
            class_name=self.collection_name,
            uuid=obj_id,
            vector=datapoint.vector,
        )

    def _ingest_via_batch(self, batch_client: object, datapoints: Sequence[DataPoint]) -> bool:
        """Attempt to ingest datapoints via the client's batch API."""
        batch_size = max(1, int(self._batch_size or 0))
        try:
            with batch_client as batch:
                batch.batch_size = batch_size
                for dp in datapoints:
                    self._add_batch_object(batch, dp)
            return True
        except Exception as exc:  # pragma: no cover - defensive programming
            logger.error(f"[WeaviateAdaptor] Batch ingest failed: {exc}")
            return False

    @staticmethod
    def _translate_metric(metric: Metric) -> str:
        """Map internal metric names to Weaviate's expected identifiers."""
        if metric == Metric.INNER_PRODUCT:
            return "dot"
        if metric == Metric.COSINE:
            return "cosine"
        if metric == Metric.L2:
            return "l2-squared"

        msg = f"Unsupported metric '{metric}'"
        raise ValueError(msg)

    def _extract_results(self, payload: dict) -> list[SearchResult]:
        """Convert the GraphQL response into a list of ``SearchResult`` objects."""
        if payload.get("errors"):
            return []

        data = payload.get("data", {})
        get_section = data.get("Get", {})
        hits = get_section.get(self.collection_name, [])

        results: list[SearchResult] = []
        for hit in hits:
            additional = hit.get("_additional", {})
            distance = float(additional.get("distance", float("inf")))
            original = hit.get(self._ORIGINAL_ID_FIELD)

            if original is not None:
                candidate_id = original
            else:
                id_str = additional.get("id")
                try:
                    candidate_id = self._validate_uuid(id_str) if id_str is not None else -1
                except ValueError:
                    candidate_id = -1

            results.append(SearchResult(id=candidate_id, score=distance))

        return results

    def _extract_count(self, payload: dict) -> int:
        """Extract the ``meta.count`` field from an aggregate response."""
        if payload.get("errors"):
            return 0

        data = payload.get("data", {})
        aggregate = data.get("Aggregate", {})
        entries = aggregate.get(self.collection_name, [])
        if not entries:
            return 0

        meta = entries[0].get("meta", {})
        return meta.get("count", 0)

    @staticmethod
    def _make_object_id(identifier: int) -> str:
        """Derive a deterministic UUID from the integer identifier."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"inatinqperf:{identifier}"))

    # TODO: to kill this once we add the IndexType enum
    @staticmethod
    def _normalise_index_type(index_type: str | None) -> str:
        """Return the vector index type accepted by Weaviate."""
        if index_type:
            return index_type.strip()
        return "hnsw"

    def _delete_object(self, obj_id: str) -> None:
        """Delete a single object, tolerating 404 responses."""
        try:
            self._client.data_object.delete(obj_id, class_name=self.collection_name)
        except Exception as exc:  # pragma: no cover - defensive programming
            status_code = exc.status_code if hasattr(exc, "status_code") else None
            if status_code not in {HTTPStatus.NOT_FOUND, HTTPStatus.UNPROCESSABLE_ENTITY}:
                self._handle_exception("failed to delete object", exc)

    @staticmethod
    def _validate_uuid(object_id: str) -> int:
        """Ensure Weaviate returned a valid UUID; return -1 when validation fails."""
        try:
            uuid.UUID(object_id)
        except (ValueError, AttributeError) as exc:
            msg = "not a valid uuid"
            raise ValueError(msg) from exc
        return -1

    @staticmethod
    def _handle_exception(context: str, exc: Exception) -> None:
        """Wrap raw client exceptions in a ``WeaviateError`` for consumers."""
        msg = f"{context}: {exc}"
        raise WeaviateError(msg) from exc
