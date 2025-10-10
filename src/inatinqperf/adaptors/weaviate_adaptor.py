"""Adaptor for interacting with a running Weaviate instance via HTTP."""

import json
import time
import uuid
from collections.abc import Sequence
from http import HTTPStatus

import numpy as np
import requests
from loguru import logger
from requests import Response, Session

from inatinqperf.adaptors.base import VectorDatabase
from inatinqperf.adaptors.metric import Metric


class WeaviateError(RuntimeError):
    """Raised when Weaviate responds with an unexpected status code."""


class Weaviate(VectorDatabase):
    """HTTP-based adaptor that manages a single Weaviate class/collection."""

    INATINQ_WEVIATE_QUERY_DIM = 2

    def __init__(
        self,
        dim: int,
        metric: str = Metric.COSINE,
        *,
        base_url: str = "http://localhost:8080",
        class_name: str = "collection_name",
        timeout: float = 10.0,
        vectorizer: str = "none",
    ) -> None:
        super().__init__()
        if dim <= 0:
            msg = "Vector dimensionality must be positive."
            raise ValueError(msg)

        self.dim = int(dim)
        self.metric = metric
        self.base_url = base_url.rstrip("/")
        self.class_name = class_name
        self.timeout = timeout
        self.vectorizer = vectorizer
        self._distance_metric = self._translate_metric(metric)

        self._session: Session = requests.Session()
        self._schema_endpoint = f"{self.base_url}/v1/schema"
        self._objects_endpoint = f"{self.base_url}/v1/objects"
        self._graphql_endpoint = f"{self.base_url}/v1/graphql"
        self._ready_endpoint = f"{self.base_url}/v1/.well-known/ready"

        logger.info(
            f"""[WeaviateAdaptor] Initialized class='{self.class_name}' """
            f"""base_url={self.base_url} dim={self.dim} metric={self.metric.value}"""
        )

    # ------------------------------------------------------------------
    # VectorDatabase implementation
    # ------------------------------------------------------------------
    def train_index(self, x_train: np.ndarray) -> None:  # noqa: ARG002 - interface contract
        """Ensure that the Weaviate class exists before ingest."""
        self._check_ready()
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

    def drop_index(self) -> None:
        """Delete the underlying Weaviate class."""
        response = self._session.delete(f"{self._schema_endpoint}/{self.class_name}", timeout=self.timeout)
        if response.status_code not in {HTTPStatus.OK, HTTPStatus.NO_CONTENT, HTTPStatus.NOT_FOUND}:
            self._raise_error("failed to delete class", response)
        logger.info(f"[WeaviateAdaptor] Dropped class {self.class_name}")

    def upsert(self, ids: np.ndarray, x: np.ndarray) -> None:
        """Insert or update vectors in Weaviate."""
        ids = np.asarray(ids).astype(np.int64)
        vectors = np.asarray(x, dtype=np.float32)
        if vectors.ndim != self.INATINQ_WEVIATE_QUERY_DIM or vectors.shape[1] != self.dim:
            msg = "Vectors must be 2-D with shape (n, dim)."
            raise ValueError(msg)
        if ids.shape[0] != vectors.shape[0]:
            msg = "ids and vectors must have matching length"
            raise ValueError(msg)

        self.train_index(vectors)
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

            body = {
                "id": obj_id,
                "class": self.class_name,
                "vector": vector.tolist(),
                "properties": {"originalId": int(identifier)},
            }
            response = self._session.post(self._objects_endpoint, json=body, timeout=self.timeout)
            if response.status_code not in {200, 201}:
                self._raise_error("failed to upsert object", response)

    def search(self, q: np.ndarray, topk: int, **kwargs: object) -> tuple[np.ndarray, np.ndarray]:
        """Run nearest-neighbor search using GraphQL."""
        if topk <= 0:
            msg = "topk must be positive"
            raise ValueError(msg)

        queries = np.asarray(q, dtype=np.float32)
        if queries.ndim != self.INATINQ_WEVIATE_QUERY_DIM or queries.shape[1] != self.dim:
            msg = "Query vectors must be 2-D with correct dimensionality"
            raise ValueError(msg)

        limit = topk
        n_queries = queries.shape[0]
        distances = np.full((n_queries, limit), np.inf, dtype=np.float32)
        ids = np.full((n_queries, limit), -1, dtype=np.int64)

        if kwargs:
            logger.debug(f"[WeaviateAdaptor] Ignoring unused search kwargs: {kwargs}")

        for row, vec in enumerate(queries):
            vector_json = json.dumps(vec.tolist())
            query_str = (
                "{\n  Get {\n    "
                f"{self.class_name}(nearVector: {{ vector: {vector_json} }}, limit: {limit}) "
                "{\n      originalId\n      _additional { id distance }\n    }\n  }\n}"
            )
            payload = {"query": query_str}
            response = self._session.post(self._graphql_endpoint, json=payload, timeout=self.timeout)
            if response.status_code != HTTPStatus.OK:
                self._raise_error("search request failed", response)
            data = response.json()
            results = self._extract_results(data)
            for col, result in enumerate(results[:limit]):
                additional = result.get("_additional", {})
                distance = float(additional.get("distance", float("inf")))
                id_str = additional.get("id")
                original = result.get("originalId")
                if original is not None:
                    ids[row, col] = int(original)
                else:
                    try:
                        ids[row, col] = self._validate_uuid(id_str) if id_str is not None else -1
                    except ValueError:
                        ids[row, col] = -1
                distances[row, col] = distance
        return distances, ids

    def delete(self, ids: Sequence[int]) -> None:
        """Delete objects for the provided identifiers."""
        for identifier in ids:
            obj_id = self._make_object_id(int(identifier))
            response = self._session.delete(
                f"{self._objects_endpoint}/{self.class_name}/{obj_id}", timeout=self.timeout
            )
            if response.status_code not in {200, 204, 404}:
                self._raise_error("failed to delete object", response)

    def stats(self) -> dict[str, object]:
        """Return basic statistics derived from Weaviate aggregate queries."""
        count_query = (
            f"{{\n  Aggregate {{\n    {self.class_name} {{\n      meta {{ count }}\n    }}\n  }}\n}}"
        )
        aggregate_query = {"query": count_query}
        response = self._session.post(self._graphql_endpoint, json=aggregate_query, timeout=self.timeout)
        if response.status_code != HTTPStatus.OK:
            self._raise_error("failed to fetch stats", response)
        data = response.json()
        count = self._extract_count(data)
        stats = {
            "ntotal": count,
            "metric": self.metric.value,
            "class_name": self.class_name,
            "base_url": self.base_url,
            "dim": self.dim,
        }
        logger.debug(f"[WeaviateAdaptor] Stats for {self.class_name}: {stats}")
        return stats

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _check_ready(self) -> None:
        deadline = time.time() + self.timeout
        last_response: Response | None = None
        while time.time() < deadline:
            response = self._session.get(self._ready_endpoint, timeout=2)
            last_response = response
            if response.status_code == HTTPStatus.OK:
                logger.info(f"[WeaviateAdaptor] Weaviate instance ready at {self.base_url}")
                return
            time.sleep(0.5)
        if last_response is None:
            error_msg = "weaviate instance readiness probe failed to connect"
            raise WeaviateError(error_msg)
        self._raise_error("weaviate instance not ready", last_response)

    def _class_exists(self) -> bool:
        response = self._session.get(f"{self._schema_endpoint}/{self.class_name}", timeout=self.timeout)
        if response.status_code == HTTPStatus.NOT_FOUND:
            return False
        if response.status_code != HTTPStatus.OK:
            self._raise_error("failed to probe class", response)
        return True

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

    def _build_count_query(self) -> str:
        return f"{{\n  Aggregate {{\n    {self.class_name} {{\n      meta {{ count }}\n    }}\n  }}\n}}"

    def _extract_results(self, payload: dict) -> list[dict]:
        if payload.get("errors"):
            return []
        data = payload.get("data", {})
        get_section = data.get("Get", {})
        return get_section.get(self.class_name, [])

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
    def _raise_error(context: str, response: Response) -> None:
        msg = f"{context}: {response.status_code} {response.text}"
        raise WeaviateError(msg)
