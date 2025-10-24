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
        metric: Metric,
        url: str,
        class_name: str = "collection_name",
        timeout: float = 10.0,
        vectorizer: str = "none",
        client: WeaviateClient | None = None,
        index_type: str | None = None,  # noqa: ARG002 - interface contract
    ) -> None:
        """Initialise the adaptor with a dataset template and connectivity details."""
        metric_enum = metric if isinstance(metric, Metric) else Metric(metric)
        super().__init__(dataset=dataset, metric=metric_enum.value)

        if self.dim <= 0:
            msg = "Vector dimensionality must be positive."
            raise ValueError(msg)

        self._metric = metric_enum
        self.url = url.rstrip("/")
        self.class_name = class_name
        self.timeout = timeout
        self.vectorizer = vectorizer
        self._distance_metric = self._translate_metric(metric_enum)

        # The official Weaviate client exposes the schema, data, and query APIs we rely on.
        self._client: WeaviateClient = client or WeaviateClient(
            url=self.url,
            timeout_config=(self.timeout, self.timeout),
        )

        self._drop_class_if_exists()
        self._ensure_schema_exists()
        self._ingest_dataset(dataset)

        logger.info(
            "[WeaviateAdaptor] init "
            f"url={self.url} "
            f"class={self.class_name} "
            f"dim={self.dim} "
            f"metric={self._metric.value}"
        )

    # ------------------------------------------------------------------
    # VectorDatabase implementation
    # ------------------------------------------------------------------
    def train_index(self, x_train: np.ndarray) -> None:  # noqa: ARG002 - interface contract
        """Ensure that the backing Weaviate class exists."""
        self._ensure_schema_exists()

    def drop_index(self) -> None:
        """Delete the managed class entirely."""
        try:
            self._client.schema.delete_class(self.class_name)
        except Exception as exc:  # pragma: no cover - defensive programming
            status_code = self._status_code(exc)
            if status_code == HTTPStatus.NOT_FOUND:
                logger.info(f" Class {self.class_name} already absent.")
                return
            self._handle_exception("failed to delete class", exc)

        logger.info(f" Dropped class={self.class_name}")

    def upsert(self, x: Sequence[DataPoint]) -> None:
        """Insert or update vectors and associated metadata."""
        datapoints = list(x)
        if not datapoints:
            logger.debug(" upsert called with empty payload; skipping.")
            return

        logger.info(f" Upserting {len(datapoints)} datapoints into class {self.class_name}")

        # Ensure the class exists. This is a no-op if already provisioned.
        self.train_index(vectors)

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
                self._client.data_object.delete(obj_id, class_name=self.class_name)
            except Exception as exc:  # pragma: no cover - defensive programming
                status_code = self._status_code(exc)
                if status_code != HTTPStatus.NOT_FOUND:
                    self._handle_exception("failed to delete existing object", exc)

            try:
                self._client.data_object.create(
                    data_object=properties,
                    class_name=self.class_name,
                    uuid=obj_id,
                    vector=vector.tolist(),
                )
            except Exception as exc:  # pragma: no cover - defensive programming
                self._handle_exception("failed to upsert object", exc)

        logger.info(f" Finished upsert for {len(datapoints)} datapoints (class={self.class_name})")

    def search(
        self,
        q: Query | Sequence[float] | np.ndarray,
        topk: int,
        **_: object,
    ) -> Sequence[SearchResult]:
        """Search for the ``topk`` nearest vectors using Weaviate's GraphQL API."""
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
            self._client.query.get(self.class_name, [self._ORIGINAL_ID_FIELD])
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
                logger.warning(f" search requested before schema exists (class={self.class_name}).")
                return []

        results = self._extract_results(data)
        return [SearchResult(id=result.id, score=result.score) for result in results[:topk]]

    def delete(self, ids: Sequence[int]) -> None:
        """Delete objects corresponding to the provided identifiers."""
        object_ids = [self._make_object_id(int(identifier)) for identifier in ids]
        for obj_id in object_ids:
            self._delete_object(obj_id)

    def stats(self) -> dict[str, object]:
        """Return summary statistics sourced via a Weaviate aggregation query."""
        try:
            data = self._client.query.aggregate(self.class_name).with_meta_count().do()
        except Exception as exc:  # pragma: no cover - defensive programming
            status_code = self._status_code(exc)
            if status_code == HTTPStatus.UNPROCESSABLE_ENTITY:
                logger.warning(f" stats requested before schema exists (class={self.class_name}).")
            else:
                logger.error(f" Failed to fetch stats for class={self.class_name}: {exc}")
        else:
            count = self._extract_count(data)
        stats = {
            "ntotal": count,
            "metric": self.metric.value,
            "metric": self.metric.value,
            "class_name": self.class_name,
            "url": self.url,
            "dim": self.dim,
        }
        logger.debug(f" Stats for {self.class_name} => {stats}")
        return stats

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
        try:
            schema = self._client.schema.get()
        except Exception as exc:  # pragma: no cover - defensive programming
            self._handle_exception("failed to probe class", exc)

        classes = schema.get("classes", [])
        return any(entry.get("class") == self.class_name for entry in classes)

    def _ensure_schema_exists(self) -> None:
        """Create the Weaviate class if it does not already exist."""
        self._check_ready()
        if self._class_exists():
            return

        payload = self._schema_payload()
        try:
            self._client.schema.create_class(payload)
        except Exception as exc:  # pragma: no cover - defensive programming
            status_code = self._status_code(exc)
            if status_code == HTTPStatus.UNPROCESSABLE_ENTITY and "already exists" in str(exc).lower():
                logger.info(f" Class {self.class_name} already exists; continuing.")
                return
            self._handle_exception("failed to create class", exc)

        logger.info(f" Created class {self.class_name}")

    def _drop_class_if_exists(self) -> None:
        """Drop the target class if it already exists."""
        try:
            self._client.schema.delete_class(self.class_name)
        except Exception as exc:  # pragma: no cover - defensive programming
            status_code = self._status_code(exc)
            if status_code != HTTPStatus.NOT_FOUND:
                self._handle_exception("failed to drop existing class", exc)

    def _schema_payload(self) -> dict[str, object]:
        """Return the schema definition submitted to Weaviate."""

        return {
            "class": self.class_name,
            "description": f"{self.class_name}_iNatInqPerf",
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
        items = list(datapoints)
        if not items:
            logger.debug(" _ingest_datapoints received no datapoints; skipping.")
            return

        use_batch = isinstance(self._batch_size, int) and self._batch_size > 0
        batch_client = self._client.batch if hasattr(self._client, "batch") else None

        if use_batch and batch_client is not None:
            logger.info(f" Batch ingest enabled (batch_size={self._batch_size}); using batch API.")
            batch_size = max(1, int(self._batch_size or 0))

            def configure(target: object) -> None:
                if hasattr(target, "batch_size"):
                    target.batch_size = batch_size

            def emit(target: object) -> None:
                for dp in items:
                    self._add_batch_object(target, dp)

            try:
                has_context = hasattr(batch_client, "__enter__") and callable(batch_client.__enter__)
                if has_context:
                    with batch_client as batch:
                        configure(batch)
                        emit(batch)
                else:
                    configure(batch_client)
                    emit(batch_client)
                return
            except Exception as exc:  # pragma: no cover - defensive programming
                logger.error(f" Batch ingest failed: {exc}")
                logger.warning(" Batch ingest failed; falling back to upsert.")
                self.upsert(items)
                return

        if use_batch:
            logger.warning(" Batch ingest requested but client lacks batch interface; using upsert.")
        self.upsert(items)

    def _add_batch_object(self, batch: object, datapoint: DataPoint) -> None:
        """Helper to append a datapoint to a Weaviate batch request."""
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
                class_name=self.class_name,
                uuid=obj_id,
                vector=vector,
            )
        else:  # pragma: no cover - defensive programming
            msg = "batch client missing add_data_object"
            raise TypeError(msg)

    def _bulk_ingest(self, datapoints: Sequence[DataPoint], batch_client: object) -> bool:
        """Attempt to ingest datapoints via the client's batch API."""
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

    def _add_batch_object(self, batch: object, datapoint: DataPoint) -> None:
        """Helper to append a datapoint to a Weaviate batch request."""
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
                class_name=self.class_name,
                uuid=obj_id,
                vector=vector,
            )
        else:  # pragma: no cover - defensive programming
            msg = "batch client missing add_data_object"
            raise TypeError(msg)

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
        hits = get_section.get(self.class_name, [])

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
        entries = aggregate.get(self.class_name, [])
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
        return "unsupported"

    def _delete_object(self, obj_id: str) -> None:
        """Delete a single object, tolerating 404 responses."""
        try:
            self._client.data_object.delete(obj_id, class_name=self.class_name)
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
