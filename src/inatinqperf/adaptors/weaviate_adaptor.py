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
        dataset: HuggingFaceDataset | None = None,
        metric: Metric = Metric.COSINE,
        url: str | None = "http://localhost",
        collection_name: str = "collection_name",
        *,
        timeout: float = 10.0,
        vectorizer: str = "none",
        client: WeaviateClient | None = None,
        dim: int | None = None,
        **options: object,
    ) -> None:
        """Initialise the adaptor with a dataset template and connectivity details."""
        metric_enum = metric if isinstance(metric, Metric) else Metric(metric)

        if dataset is not None:
            super().__init__(dataset=dataset, metric=metric_enum.value)
            if self.dim <= 0:
                msg = "Vector dimensionality must be positive."
                raise ValueError(msg)
        else:
            if dim is None:
                msg = "Either dataset or dim must be provided."
                raise ValueError(msg)
            if dim <= 0:
                msg = "Vector dimensionality must be positive."
                raise ValueError(msg)
            self.metric = metric_enum
            self.dim = int(dim)

        if dataset is not None:
            super().__init__(dataset=dataset, metric=metric_enum.value)
            if self.dim <= 0:
                msg = "Vector dimensionality must be positive."
                raise ValueError(msg)
        else:
            if dim is None:
                msg = "Either dataset or dim must be provided."
                raise ValueError(msg)
            if dim <= 0:
                msg = "Vector dimensionality must be positive."
                raise ValueError(msg)
            self.metric = metric_enum
            self.dim = int(dim)

        self._metric = metric_enum
        self.url = (url or "").rstrip("/")
        self.collection_name = collection_name
        self.timeout = timeout
        self.vectorizer = vectorizer
        self._batch_size = options.pop("batch_size", None)
        self._index_type = self._normalise_index_type(options.pop("index_type", None))
        self._distance_metric = self._translate_metric(metric_enum)

        if client is not None:
            self._client: WeaviateClient | None = client
        elif self.url:
            self._client = WeaviateClient(
                url=self.url,
                timeout_config=(self.timeout, self.timeout),
            )
        else:
            self._client = None

        if self._client is not None:
            self._drop_class_if_exists()
            self._ensure_schema_exists()
            if dataset is not None:
                self._ingest_dataset(dataset)

        logger.info(
            "[WeaviateAdaptor] init "
            f"url={self.url} "
            f"class={self.collection_name} "
            f"dim={self.dim} "
            f"metric={self._metric.value}"
        )

    # ------------------------------------------------------------------
    # VectorDatabase implementation
    # ------------------------------------------------------------------
    def drop_index(self) -> None:
        """Delete the managed class entirely."""
        if self._client is None:
            msg = "Weaviate client not configured; cannot drop index."
            raise WeaviateError(msg)
        try:
            self._client.schema.delete_class(self.collection_name)
        except Exception as exc:  # pragma: no cover - defensive programming
            status_code = self._status_code(exc)
            if status_code == HTTPStatus.NOT_FOUND:
                logger.info(f" Class {self.collection_name} already absent.")
                return
            self._handle_exception("failed to delete class", exc)

        logger.info(f" Dropped class={self.collection_name}")

    def upsert(self, x: Sequence[DataPoint]) -> None:
        """Insert or update vectors and associated metadata."""
        if self._client is None:
            msg = "Weaviate client not configured; cannot upsert datapoints."
            raise WeaviateError(msg)
        datapoints = list(x)
        if not datapoints:
            logger.debug(" upsert called with empty payload; skipping.")
            return

        logger.info(f" Upserting {len(datapoints)} datapoints into class {self.collection_name}")

        # Ensure the class exists. This is a no-op if already provisioned.
        self._ensure_schema_exists()

        for dp in datapoints:
            vector = np.asarray(dp.vector, dtype=np.float32)
            if vector.ndim != 1 or vector.shape[0] != self.dim:
                msg = "DataPoint vectors must be 1-D with dimensionality matching the adaptor."
                raise ValueError(msg)

            obj_id = self._make_object_id(int(dp.id))
            metadata = dp.metadata if hasattr(dp, "metadata") else None
            properties = dict(metadata or {})
            properties.setdefault(self._ORIGINAL_ID_FIELD, int(dp.id))

            try:
                # Re-using deterministic object IDs keeps the collection idempotent.
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
                    vector=vector.tolist(),
                )
            except Exception as exc:  # pragma: no cover - defensive programming
                self._handle_exception("failed to upsert object", exc)

        logger.info(f" Finished upsert for {len(datapoints)} datapoints (class={self.collection_name})")

    def search(self, q: Query, topk: int, **kwargs) -> Sequence[SearchResult]:  # NOQA: ARG002
        """Search for the ``topk`` nearest vectors using Weaviate's GraphQL API."""
        if self._client is None:
            msg = "Weaviate client not configured; cannot execute search."
            raise WeaviateError(msg)
        if topk <= 0:
            msg = "topk must be positive"
            raise ValueError(msg)

        if isinstance(q, Query):
            vector = np.asarray(q.vector, dtype=np.float32)
            filters = q.filters
        else:
            vector = np.asarray(q, dtype=np.float32)
            filters = None

        if vector.ndim != 1 or vector.shape[0] != self.dim:
            msg = "Query vector must be 1-D with dimensionality matching the index"
            raise ValueError(msg)

        builder = (
            self._client.query.get(self.collection_name, [self._ORIGINAL_ID_FIELD])
            .with_near_vector({"vector": vector.tolist()})
            .with_limit(topk)
            .with_additional(["id", "distance"])
        )

        if filters:
            # Weaviate's where clause enables server-side filtering.
            builder = builder.with_where(filters)

        try:
            data = builder.do()
        except Exception as exc:  # pragma: no cover - defensive programming
            status_code = self._status_code(exc)
            if status_code == HTTPStatus.UNPROCESSABLE_ENTITY:
                logger.warning(f" search requested before schema exists (class={self.collection_name}).")
                return []
            self._handle_exception("failed to execute search", exc)

        results = self._extract_results(data)
        return [SearchResult(id=result.id, score=result.score) for result in results[:topk]]

    def delete(self, ids: Sequence[int]) -> None:
        """Delete objects corresponding to the provided identifiers."""
        if self._client is None:
            msg = "Weaviate client not configured; cannot delete datapoints."
            raise WeaviateError(msg)
        if not self._class_exists():
            logger.info(f" Delete requested but class {self.collection_name} absent; skipping.")
            return
        object_ids = [self._make_object_id(int(identifier)) for identifier in ids]
        for obj_id in object_ids:
            self._delete_object(obj_id)

    def stats(self) -> dict[str, object]:
        """Return summary statistics sourced via a Weaviate aggregation query."""
        if self._client is None:
            msg = "Weaviate client not configured; cannot fetch stats."
            raise WeaviateError(msg)
        count = 0
        try:
            data = self._client.query.aggregate(self.collection_name).with_meta_count().do()
        except Exception as exc:  # pragma: no cover - defensive programming
            status_code = self._status_code(exc)
            if status_code == HTTPStatus.UNPROCESSABLE_ENTITY:
                logger.warning(f" stats requested before schema exists (class={self.collection_name}).")
            else:  # unexpected failure should propagate as WeaviateError
                self._handle_exception("failed to fetch stats", exc)
        else:
            count = self._extract_count(data)
        stats = {
            "ntotal": count,
            "metric": self.metric.value,
            "class_name": self.collection_name,
            "url": self.url,
            "dim": self.dim,
        }
        logger.debug(f" Stats for {self.collection_name} => {stats}")
        return stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _check_ready(self) -> None:
        """Poll the readiness endpoint until Weaviate reports healthy."""
        if self._client is None:
            msg = "Weaviate client not configured; cannot check readiness."
            raise WeaviateError(msg)
        deadline = time.time() + self.timeout
        last_error: Exception | None = None

        while time.time() < deadline:
            try:
                if self._client.is_ready():
                    logger.info(f" Weaviate ready at {self.url}")
                    return
            except Exception as exc:  # pragma: no cover - depends on client internals
                last_error = exc
            time.sleep(0.5)

        if last_error is None:
            msg = "weaviate instance readiness probe failed to connect"
            raise WeaviateError(msg)
        self._handle_exception("weaviate instance not ready", last_error)

    def _class_exists(self) -> bool:
        """Return ``True`` if the target class already exists."""
        if self._client is None:
            msg = "Weaviate client not configured; cannot probe class existence."
            raise WeaviateError(msg)
        try:
            schema = self._client.schema.get()
        except Exception as exc:  # pragma: no cover - defensive programming
            self._handle_exception("failed to probe class", exc)

        classes = schema.get("classes", [])
        return any(entry.get("class") == self.collection_name for entry in classes)

    def _ensure_schema_exists(self) -> None:
        """Create the Weaviate class if it does not already exist."""
        if self._client is None:
            msg = "Weaviate client not configured; cannot ensure schema."
            raise WeaviateError(msg)
        self._check_ready()
        if self._class_exists():
            return

        payload = self._schema_payload()
        try:
            self._client.schema.create_class(payload)
        except Exception as exc:  # pragma: no cover - defensive programming
            status_code = self._status_code(exc)
            if status_code == HTTPStatus.UNPROCESSABLE_ENTITY and "already exists" in str(exc).lower():
                logger.info(f" Class {self.collection_name} already exists; continuing.")
                return
            self._handle_exception("failed to create class", exc)

        logger.info(f" Created class {self.collection_name}")

    def _drop_class_if_exists(self) -> None:
        """Drop the target class if it already exists."""
        if self._client is None:
            msg = "Weaviate client not configured; cannot drop existing class."
            raise WeaviateError(msg)
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
        if self._client is None:
            msg = "Weaviate client not configured; cannot ingest dataset."
            raise WeaviateError(msg)

        if "embedding" not in dataset.column_names:
            logger.warning(" dataset missing 'embedding' column; skipping ingest")
            return

        embeddings = np.asarray(dataset["embedding"], dtype=np.float32)
        if embeddings.size == 0:
            logger.warning(" dataset contains no embeddings; skipping ingest")
            return

        if embeddings.ndim != self._EXPECTED_VECTOR_RANK or embeddings.shape[1] != self.dim:
            msg = (
                "Dataset embeddings must be 2-D with dimensionality matching the adaptor "
                f"(expected {self.dim}, received {embeddings.shape})"
            )
            raise ValueError(msg)

        if "id" in dataset.column_names:
            ids = dataset["id"]
            if len(ids) != len(embeddings):
                msg = "Length of dataset ids must match number of embeddings"
                raise ValueError(msg)
        else:
            ids = list(range(len(embeddings)))

        datapoints = [
            DataPoint(id=int(idx), vector=vec, metadata={}) for idx, vec in zip(ids, embeddings, strict=False)
        ]
        ids_provided = "id" in dataset.column_names
        logger.info(
            f" Ingesting {len(datapoints)} data points "
            f"(dim={embeddings.shape[1]}, ids_provided={ids_provided})"
        )
        self._ingest_datapoints(datapoints)
        try:
            post_stats = self.stats()
            logger.info(f" Stats after ingest => ntotal={post_stats['ntotal']}")
            if post_stats.get("ntotal", 0) < len(datapoints):
                post_stat_ntotal = len(datapoints) - post_stats.get("ntotal")
                logger.warning(f" {post_stat_ntotal}  smaller than ingested count {len(datapoints)}")
        except WeaviateError:
            logger.warning(" Unable to fetch stats immediately after ingest")

    def _ingest_datapoints(self, datapoints: Sequence[DataPoint]) -> None:
        """Ingest datapoints via batch API when configured, otherwise fall back to upsert."""
        if self._client is None:
            msg = "Weaviate client not configured; cannot ingest datapoints."
            raise WeaviateError(msg)
        items = list(datapoints)
        if not items:
            logger.debug(" _ingest_datapoints received no datapoints; skipping.")
            return

        use_batch = isinstance(self._batch_size, int) and self._batch_size > 0
        batch_client = getattr(self._client, "batch", None)

        if use_batch and batch_client is not None:
            logger.info(f" Batch ingest enabled (batch_size={self._batch_size}); using batch API.")
            if self._ingest_via_batch(batch_client, items):
                return
            logger.warning(" Batch ingest failed; falling back to upsert.")
        elif use_batch:
            logger.warning(" Batch ingest requested but client lacks batch interface; using upsert.")
        self.upsert(items)

    def _add_batch_object(self, batch: object, datapoint: DataPoint) -> None:
        """Helper to append a datapoint to a Weaviate batch request."""
        if self._client is None:
            msg = "Weaviate client not configured; cannot add batch object."
            raise WeaviateError(msg)
        obj_id = self._make_object_id(int(datapoint.id))
        metadata = datapoint.metadata if hasattr(datapoint, "metadata") else None
        properties = dict(metadata or {})
        properties.setdefault(self._ORIGINAL_ID_FIELD, int(datapoint.id))
        vector_arr = np.asarray(datapoint.vector, dtype=np.float32)
        if vector_arr.ndim != 1 or vector_arr.shape[0] != self.dim:
            msg = (
                "DataPoint vectors added via batch ingest must be 1-D with dimensionality "
                f"matching the adaptor (expected {self.dim}, received {vector_arr.shape})"
            )
            raise ValueError(msg)
        vector = vector_arr.tolist()
        add = batch.add_data_object if hasattr(batch, "add_data_object") else None
        if callable(add):
            add(
                data_object=properties,
                class_name=self.collection_name,
                uuid=obj_id,
                vector=vector,
            )
        else:  # pragma: no cover - defensive programming
            msg = "batch client missing add_data_object"
            raise TypeError(msg)

    def _ingest_via_batch(self, batch_client: object, datapoints: Sequence[DataPoint]) -> bool:
        """Attempt to ingest datapoints via the client's batch API."""
        if self._client is None:
            msg = "Weaviate client not configured; cannot perform batch ingest."
            raise WeaviateError(msg)
        batch_size = max(1, int(self._batch_size or 0))
        try:
            # Some client versions expose the batch context manager; prefer that when available.
            batch_ctx = getattr(batch_client, "__enter__", None)
            if callable(batch_ctx):
                with batch_client as batch:
                    if hasattr(batch, "batch_size"):
                        batch.batch_size = batch_size
                    for dp in datapoints:
                        self._add_batch_object(batch, dp)
            else:
                # Fallback to calling add_data_object on the batch client directly.
                if hasattr(batch_client, "batch_size"):
                    batch_client.batch_size = batch_size
                for dp in datapoints:
                    self._add_batch_object(batch_client, dp)
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
                candidate_id = int(original)
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
        return int(meta.get("count", 0))

    @staticmethod
    def _make_object_id(identifier: int) -> str:
        """Derive a deterministic UUID from the integer identifier."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"inatinqperf:{identifier}"))

    # TODO: to kill this once we add the IndexType enum
    @staticmethod
    def _normalise_index_type(index_type: str | None) -> str:
        """Return the vector index type accepted by Weaviate."""
        if isinstance(index_type, str) and index_type.strip():
            return index_type.strip()
        return "hnsw"

    def _delete_object(self, obj_id: str) -> None:
        """Delete a single object, tolerating 404 responses."""
        try:
            self._client.data_object.delete(obj_id, class_name=self.collection_name)
        except Exception as exc:  # pragma: no cover - defensive programming
            status_code = self._status_code(exc)
            if status_code not in {HTTPStatus.NOT_FOUND}:
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

    @staticmethod
    def _status_code(exc: Exception) -> int | None:
        """Return the status_code attribute when available."""
        return exc.status_code if hasattr(exc, "status_code") else None
