"""Adaptor for interacting with a running Weaviate instance via HTTP."""

import time
import uuid
from collections.abc import Sequence
from http import HTTPStatus

import numpy as np
from datasets import Dataset as HuggingFaceDataset
from loguru import logger
from weaviate import Client as WeaviateClient

from inatinqperf.adaptors.base import DataPoint, HuggingFaceDataset, Query, SearchResult, VectorDatabase
from inatinqperf.adaptors.metric import Metric


class WeaviateError(RuntimeError):
    """Raised when Weaviate responds with an unexpected status code."""


class Weaviate(VectorDatabase):
    """HTTP-based adaptor that manages a single Weaviate class/collection."""

    INATINQ_WEVIATE_QUERY_DIM = 2

    def __init__(
        self,
        dataset: HuggingFaceDataset,
        metric: Metric | str = Metric.COSINE,
        *,
        base_url: str = "http://localhost:8080",
        class_name: str = "collection_name",
        timeout: float = 10.0,
        vectorizer: str = "none",
        client: WeaviateClient | None = None,
    ) -> None:
        metric_enum = metric if isinstance(metric, Metric) else Metric(metric)
        super().__init__(dataset=dataset, metric=metric_enum.value)

        if self.dim <= 0:
            msg = "Vector dimensionality must be positive."
            raise ValueError(msg)

        self._metric = metric_enum
        self.base_url = base_url.rstrip("/")
        self.class_name = class_name
        self.timeout = timeout
        self.vectorizer = vectorizer
        self._distance_metric = self._translate_metric(metric_enum)

        self._client: WeaviateClient = client or WeaviateClient(
            url=self.base_url,
            timeout_config=(self.timeout, self.timeout),
        )

        self._configure()

        logger.info(
            f"""[WeaviateAdaptor] Initialized class='{self.class_name}' """
            f"""base_url={self.base_url} dim={self.dim} metric={self._metric.value}"""
        )

    @property
    def distance_metric(self) -> str:
        """The distance metric to use for similarity search."""
        return self._distance_metric

    # ------------------------------------------------------------------
    # VectorDatabase implementation
    # ------------------------------------------------------------------
    def _configure(self) -> None:
        """Ensure that the Weaviate class exists before ingesting data."""
        self.check_ready()

        if self._class_exists():
            return

        payload = {
            "class": self.class_name,
            "description": "Collection managed by iNatInqPerf",
            "vectorizer": self.vectorizer,
            "vectorIndexType": "hnsw",
            "vectorIndexConfig": {
                "distance": self._distance_metric,
                "vectorCacheMaxObjects": 1_000_000,
            },
            "properties": [
                {
                    "name": "originalId",
                    "description": "Original integer identifier",
                    "dataType": ["int"],
                }
            ],
        }
        response = self._session.post(self._schema_endpoint, json=payload, timeout=self.timeout)

        if response.status_code not in {HTTPStatus.OK, HTTPStatus.CREATED}:
            # Weaviate returns 422 if class already exists - handle gracefully.
            if (
                response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
                and "already exists" in response.text.lower()
            ):
                logger.info(f"[WeaviateAdaptor] Class {self.class_name} already exists.")
                return
            self._raise_error("failed to create class", response)

        logger.info(
            f"""[WeaviateAdaptor] Created class={self.class_name} """
            f"""(distance={self._distance_metric} vectorizer={self.vectorizer})"""
        )

    def upsert(self, x: Sequence[DataPoint]) -> None:
        """Insert or update vectors in Weaviate."""
        ids, vectors = np.empty((len(x),), dtype=np.int64), np.empty((len(x), self.dim), dtype=np.float32)
        for i, d in enumerate(x):
            ids[i] = d.id
            vectors[i] = np.asarray(d.vector, dtype=np.float32)

        if vectors.ndim != self.INATINQ_WEVIATE_QUERY_DIM or vectors.shape[1] != self.dim:
            msg = "Vectors must be 2-D with shape (n, dim)."
            raise ValueError(msg)

        if ids.shape[0] != vectors.shape[0]:
            msg = "ids and vectors must have matching length"
            raise ValueError(msg)

        # Prefer best-effort ingestion: we validate matching lengths above, but leave
        # strict=False so late mismatches drop extras instead of aborting the batch.
        for identifier, vector in zip(ids, vectors, strict=False):
            obj_id = self._make_object_id(int(identifier))
            # Delete any pre-existing object so POST remains consistent.
            delete_resp = self._session.delete(
                f"{self._objects_endpoint}/{self.class_name}/{obj_id}", timeout=self.timeout
            )
            if delete_resp.status_code not in {HTTPStatus.OK, HTTPStatus.NO_CONTENT, HTTPStatus.NOT_FOUND}:
                self._raise_error("failed to delete existing object", delete_resp)

        for dp, vector in zip(datapoints, vectors, strict=False):
            obj_id = self._make_object_id(int(dp.id))
            properties = dict(getattr(dp, "metadata", {}) or {})
            properties.setdefault("originalId", int(dp.id))


    def search(self, q: Query, topk: int, **_: object) -> Sequence[SearchResult]:
        """Run nearest-neighbor search using GraphQL."""
        if topk <= 0:
            msg = "topk must be positive"
            raise ValueError(msg)
        query_vector = np.asarray(q.vector, dtype=np.float32)

        if query_vector.ndim > 1 or query_vector.shape[0] != self.dim:
            msg = "Query vectors must be 1-D with correct dimensionality"
            raise ValueError(msg)

        vector_json = json.dumps(query_vector.tolist())

        if q.ndim > 1 or q.shape[0] != self.dim:
            msg = "Query vectors must be 1-D with correct dimensionality"
            raise ValueError(msg)

        vector_json = json.dumps(q.tolist())
        query_str = (
            "{\n  Get {\n    "
            f"{self.class_name}(nearVector: {{ vector: {vector_json} }}, limit: {topk}) "
            "{\n      originalId\n      _additional { id distance }\n    }\n  }\n}"
        )
        payload = {"query": query_str}
        response = self._session.post(self._graphql_endpoint, json=payload, timeout=self.timeout)

        if response.status_code != HTTPStatus.OK:
            self._raise_error("search request failed", response)

        data = response.json()
        results = self._extract_results(data)

        search_results = []

        for result in results[:topk]:
            additional = result.get("_additional", {})
            distance = float(additional.get("distance", "inf"))
            id_str = additional.get("id")
            original = result.get("originalId")
            if original is not None:
                data_id = int(original)

            else:
                try:
                    data_id = self._validate_uuid(id_str) if id_str is not None else -1
                except ValueError:
                    data_id = -1

            search_results.append(SearchResult(id=data_id, score=distance))

        return search_results

    def delete(self, ids: Sequence[int]) -> None:
        """Delete objects for the provided identifiers."""
        for identifier in ids:
            obj_id = self._make_object_id(int(identifier))
            try:
                self._client.data_object.delete(obj_id, class_name=self.class_name)
            except Exception as exc:  # pragma: no cover - defensively capture client errors
                status_code = getattr(exc, "status_code", None)
                if status_code not in {HTTPStatus.NOT_FOUND}:
                    self._handle_exception("failed to delete object", exc)

    def stats(self) -> dict[str, object]:
        """Return basic statistics derived from Weaviate aggregate queries."""
        try:
            data = self._client.query.aggregate(self.class_name).with_meta_count().do()
        except Exception as exc:
            self._handle_exception("failed to fetch stats", exc)
        count = self._extract_count(data)
        stats = {
            "ntotal": count,
            "metric": self._metric.value,
            "class_name": self.class_name,
            "base_url": self.base_url,
            "dim": self.dim,
        }
        logger.debug(f"[WeaviateAdaptor] Stats for {self.class_name}: {stats}")
        return stats

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def check_ready(self) -> None:
        """Check if instance is ready for operations."""
        deadline = time.time() + self.timeout
        last_error: Exception | None = None
        while time.time() < deadline:
            try:
                if self._client.is_ready():
                    logger.info(f"[WeaviateAdaptor] Weaviate instance ready at {self.base_url}")
                    return
            except Exception as exc:  # pragma: no cover - depends on client internals
                last_error = exc
            time.sleep(0.5)
        if last_error is None:
            error_msg = "weaviate instance readiness probe failed to connect"
            raise WeaviateError(error_msg)
        self._handle_exception("weaviate instance not ready", last_error)

    def _class_exists(self) -> bool:
        try:
            schema = self._client.schema.get()
        except Exception as exc:
            self._handle_exception("failed to probe class", exc)
        classes = schema.get("classes", [])
        return any(entry.get("class") == self.class_name for entry in classes)

    @staticmethod
    def _translate_metric(metric: Metric) -> str:
        if metric == Metric.INNER_PRODUCT:
            return "dot"
        if metric == Metric.COSINE:
            return "cosine"
        if metric == Metric.L2:
            return "l2-squared"

        msg = f"Unsupported metric '{metric}'"
        raise ValueError(msg)

    def _extract_results(self, payload: dict) -> list[SearchResult]:
        if payload.get("errors"):
            return []
        data = payload.get("data", {})
        get_section = data.get("Get", {})
        hits = get_section.get(self.class_name, [])
        results: list[SearchResult] = []
        for hit in hits:
            additional = hit.get("_additional", {})
            distance = float(additional.get("distance", float("inf")))
            original = hit.get("originalId")
            candidate_id: int
            if original is not None:
                candidate_id = int(original)
            else:
                id_str = additional.get("id")
                try:
                    candidate_id = self._validate_uuid(id_str) if id_str is not None else -1
                except ValueError:
                    candidate_id = -1
            results.append(
                SearchResult(
                    id=candidate_id,
                    score=float(distance),
                )
            )
        return results

    def _extract_count(self, payload: dict) -> int:
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
        """Map integer identifiers to deterministic UUIDs accepted by Weaviate."""

        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"inatinqperf:{identifier}"))

    @staticmethod
    def _validate_uuid(object_id: str) -> int:
        """Fallback when the search response omits the originalId property."""

        try:
            uuid.UUID(object_id)
        except (ValueError, AttributeError) as exc:
            msg = "not a valid uuid"
            raise ValueError(msg) from exc
        return -1

    @staticmethod
    def _handle_exception(context: str, exc: Exception) -> None:
        msg = f"{context}: {exc}"
        raise WeaviateError(msg) from exc
