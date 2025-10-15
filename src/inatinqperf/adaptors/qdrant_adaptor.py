"""Qdrant vector database adaptor."""

from collections.abc import Generator, Sequence

import numpy as np
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    PointStruct,
    SearchParams,
    VectorParams,
)

from inatinqperf.adaptors.base import SearchResult, VectorDatabase
from inatinqperf.adaptors.metric import Metric


class Qdrant(VectorDatabase):
    """Qdrant vector database.

    Qdrant only supports a single dense vector index: HNSW.
    However, it supports indexes on the attributes (aka payload) associated with each vector.
    These payload indexes can greatly improve search efficiency.
    """

    def __init__(
        self,
        dim: int,
        metric: Metric,
        url: str,
        collection_name: str,
        m: int = 32,
        ef_construct: int = 128,
    ) -> None:
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name
        self.metric = metric
        self.m = m
        self.ef_construct = ef_construct

        if not self.client.collection_exists(collection_name=collection_name):
            logger.info(f"Creating collection {collection_name}")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=dim,
                    distance=self._translate_metric(metric),
                    hnsw_config=HnswConfigDiff(
                        m=m,
                        ef_construct=ef_construct,
                    ),
                ),
            )

    @staticmethod
    def _translate_metric(metric: Metric) -> Distance:
        """Helper method to convert from Metric enum to Qdrant Distance."""
        if metric == Metric.INNER_PRODUCT:
            return Distance.DOT
        if metric == Metric.COSINE:
            return Distance.COSINE
        if metric == Metric.L2:
            return Distance.EUCLID
        if metric == Metric.MANHATTAN:
            return Distance.MANHATTAN

        msg = f"{metric} metric specified is not a valid one for Qdrant."
        raise ValueError(msg)

    def train_index(self, x_train: np.ndarray) -> None:
        """For Qdrant, the HNSW index is created when the collection is created.

        Nothing to do here.
        """

    def drop_index(self) -> None:
        """Drop the index by deleting the collection."""
        self.client.delete_collection(self.collection_name)

    @staticmethod
    def _points_iterator(ids: Sequence[int], vectors: np.ndarray) -> Generator[PointStruct]:
        """A generator to help with creating PointStructs."""
        for idx, vector in zip(ids, vectors, strict=True):
            yield PointStruct(id=idx, vector=vector)

    def upsert(self, ids: np.ndarray, x: np.ndarray) -> None:
        """Upsert vectors with given IDs. This also builds the HNSW index."""
        # Qdrant requires list of ints
        ids = ids.tolist()

        # Qdrant will override points with the same ID if they already exist,
        # which is the same behavior as `upsert`.
        # Hence we use `upload_points` for performance.
        logger.info("Uploading points to database")
        self.client.upload_points(
            collection_name=self.collection_name,
            points=self._points_iterator(ids=ids, vectors=x),
            parallel=4,
            wait=True,
        )

    def search(self, q: np.ndarray, topk: int, **kwargs) -> Sequence[SearchResult]:
        """Search for top-k nearest neighbors."""
        # Has support for attribute filter: https://qdrant.tech/documentation/quickstart/#add-a-filter

        ef = kwargs.get("ef", 128)
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=q,
            with_payload=False,
            with_vectors=False,
            limit=topk,
            search_params=SearchParams(hnsw_ef=ef, exact=False),
        )

        return [SearchResult(point.id, point.score) for point in search_result.points]

    def delete(self, ids: np.ndarray[int]) -> None:
        """Delete vectors with given IDs."""
        self.client.delete(collection_name=self.collection_name, points_selector=ids.tolist())

    def delete_collection(self) -> None:
        """Delete the collection associated with this adaptor instance."""
        logger.info(f"Deleting collection {self.collection_name}")
        self.client.delete_collection(collection_name=self.collection_name)

    def stats(self) -> dict[str, object]:
        """Return index statistics."""
        return {
            "metric": self.metric.value,
            "m": self.m,
            "ef_construct": self.ef_construct,
        }
