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
from inatinqperf.adaptors.metric import Metric


class WeaviateError(RuntimeError):
    """Raised when Weaviate responds with an unexpected status code or payload."""


class Weaviate(VectorDatabase):
    """Light-weight wrapper for managing a single class within a Weaviate instance."""

    _DEFAULT_DESCRIPTION = "Collection managed by iNatInqPerf"
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
            status_code = getattr(exc, "status_code", None)
            if status_code == HTTPStatus.NOT_FOUND:
                logger.info(f"[WeaviateAdaptor] Class {self.class_name} already absent.")
                return
            self._handle_exception("failed to delete class", exc)

        logger.info(f"[WeaviateAdaptor] Dropped class={self.class_name}")

    def upsert(self, x: Sequence[DataPoint]) -> None:
        """Insert or update vectors and associated metadata."""
        datapoints = list(x)
        if not datapoints:
            logger.debug("[WeaviateAdaptor] upsert called with empty payload; skipping.")
            return

        vectors = np.vstack([np.asarray(dp.vector, dtype=np.float32) for dp in datapoints])
        if vectors.ndim != self._EXPECTED_VECTOR_RANK or vectors.shape[1] != self.dim:
            msg = "DataPoint vectors must be 2-D with shape (n, dim)."
            raise ValueError(msg)

        # Ensure the class exists. This is a no-op if already provisioned.
        self.train_index(vectors)

        for dp, vector in zip(datapoints, vectors, strict=False):
            obj_id = self._make_object_id(int(dp.id))
            properties = dict(getattr(dp, "metadata", {}) or {})
            properties.setdefault(self._ORIGINAL_ID_FIELD, int(dp.id))

            try:
                # Re-using deterministic object IDs keeps the collection idempotent.
                self._client.data_object.delete(obj_id, class_name=self.class_name)
            except Exception as exc:  # pragma: no cover - defensive programming
                status_code = getattr(exc, "status_code", None)
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
            status_code = getattr(exc, "status_code", None)
            if status_code == HTTPStatus.UNPROCESSABLE_ENTITY:
                logger.warning(
                    f"[WeaviateAdaptor] search requested before schema exists (class={self.class_name})."
                )
                return []
            self._handle_exception("search request failed", exc)

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
            status_code = getattr(exc, "status_code", None)
            if status_code == HTTPStatus.UNPROCESSABLE_ENTITY:
                logger.warning(
                    f"[WeaviateAdaptor] stats requested before schema exists (class={self.class_name})."
                )
                return {
                    "ntotal": 0,
                    "metric": self._metric.value,
                    "class_name": self.class_name,
                    "url": self.url,
                    "dim": self.dim,
                }
            self._handle_exception("failed to fetch stats", exc)

        count = self._extract_count(data)
        stats = {
            "ntotal": count,
            "metric": self._metric.value,
            "class_name": self.class_name,
            "url": self.url,
            "dim": self.dim,
        }
        logger.debug(f"[WeaviateAdaptor] Stats for {self.class_name} => {stats}")
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
                    logger.info(f"[WeaviateAdaptor] Weaviate ready at {self.url}")
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
            status_code = getattr(exc, "status_code", None)
            if status_code == HTTPStatus.UNPROCESSABLE_ENTITY and "already exists" in str(exc).lower():
                logger.info(f"[WeaviateAdaptor] Class {self.class_name} already exists; continuing.")
                return
            self._handle_exception("failed to create class", exc)

        logger.info(f"[WeaviateAdaptor] Created class {self.class_name}")

    def _drop_class_if_exists(self) -> None:
        """Drop the target class if it already exists."""
        try:
            self._client.schema.delete_class(self.class_name)
        except Exception as exc:  # pragma: no cover - defensive programming
            status_code = getattr(exc, "status_code", None)
            if status_code != HTTPStatus.NOT_FOUND:
                self._handle_exception("failed to drop existing class", exc)

    def _schema_payload(self) -> dict[str, object]:
        """Return the schema definition submitted to Weaviate."""

        return {
            "class": self.class_name,
            "description": self._DEFAULT_DESCRIPTION,
            "vectorizer": self.vectorizer,
            "vectorIndexType": "hnsw",
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
            logger.warning("[WeaviateAdaptor] dataset missing 'embedding' column; skipping ingest")
            return

        embeddings = np.asarray(dataset["embedding"], dtype=np.float32)
        if embeddings.size == 0:
            logger.warning("[WeaviateAdaptor] dataset contains no embeddings; skipping ingest")
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
        self.upsert(datapoints)

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

    def _delete_object(self, obj_id: str) -> None:
        """Delete a single object, tolerating 404 responses."""
        try:
            self._client.data_object.delete(obj_id, class_name=self.class_name)
        except Exception as exc:  # pragma: no cover - defensive programming
            status_code = getattr(exc, "status_code", None)
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
