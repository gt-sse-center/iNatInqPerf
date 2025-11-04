"""Qdrant vector database adaptor."""

from collections.abc import Generator, Sequence
from itertools import islice

from loguru import logger
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, PointStruct

from inatinqperf.adaptors.base import DataPoint, HuggingFaceDataset, Query, SearchResult, VectorDatabase
from inatinqperf.adaptors.enums import Metric


class Qdrant(VectorDatabase):
    """Qdrant vector database.

    Qdrant only supports a single dense vector index: HNSW.
    However, it supports indexes on the attributes (aka payload) associated with each vector.
    These payload indexes can greatly improve search efficiency.
    """

    def __init__(
        self,
        dataset: HuggingFaceDataset,
        metric: Metric,
        url: str = "localhost",
        port: str = "6333",
        collection_name: str = "default_collection",
        m: int = 32,
        ef: int = 128,
        batch_size: int = 1000,
        **params,  # noqa: ARG002
    ) -> None:
        super().__init__(dataset, metric)

        self.client = QdrantClient(url=url, port=port)
        self.collection_name = collection_name

        self.m = m
        # The ef value used during collection construction
        self.ef = ef

        if self.client.collection_exists(collection_name=collection_name):
            logger.info("Deleted existing collection")
            self.client.delete_collection(collection_name=collection_name)

        logger.patch(lambda r: r.update(function="constructor")).info(
            f"Creating collection {collection_name}"
        )

        qdrant_index_params = models.HnswConfigDiff(
            m=m,
            ef_construct=ef,
            max_indexing_threads=0,
            on_disk=False,
        )

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=self.dim,
                distance=self._translate_metric(metric),
                hnsw_config=qdrant_index_params,
            ),
            shard_number=2,  # reasonable default as per qdrant docs
        )

        # Batch insert dataset
        for batch in self._batched(dataset, batch_size):
            ids = [point.pop("id") for point in batch]
            vectors = [point.pop("embedding") for point in batch]

            self.client.upsert(
                collection_name=collection_name,
                points=models.Batch(
                    ids=ids,
                    vectors=vectors,
                    payloads=batch,
                ),
            )

        num_points_in_db = self.client.count(
            collection_name=collection_name,
            exact=True,
        ).count
        logger.info(f"Number of points in Qdrant database: {num_points_in_db}")

    @staticmethod
    def _batched(iterable: HuggingFaceDataset, n: int) -> Generator[object]:
        iterator = iter(iterable)
        while batch := list(islice(iterator, n)):
            yield batch

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

    @staticmethod
    def _points_iterator(data_points: Sequence[DataPoint]) -> Generator[PointStruct]:
        """A generator to help with creating PointStructs."""
        for data_point in data_points:
            yield PointStruct(id=data_point.id, vector=data_point.vector)

    def upsert(self, x: Sequence[DataPoint]) -> None:
        """Upsert vectors with given IDs. This also builds the HNSW index."""
        # Qdrant will override points with the same ID if they already exist,
        # which is the same behavior as `upsert`.
        # Hence we use `upload_points` for performance.
        logger.info("Uploading points to database")
        self.client.upload_points(
            collection_name=self.collection_name,
            points=self._points_iterator(data_points=x),
            parallel=4,
            wait=True,
        )

    def search(self, q: Query, topk: int, **kwargs) -> Sequence[SearchResult]:
        """Search for top-k nearest neighbors."""
        # Has support for attribute filter: https://qdrant.tech/documentation/quickstart/#add-a-filter

        ef = kwargs.get("ef", 128)
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=q.vector,
            with_payload=False,
            with_vectors=False,
            limit=topk,
            search_params=models.SearchParams(hnsw_ef=ef, exact=False),
        )

        return [SearchResult(point.id, point.score) for point in search_result.points]

    def delete(self, ids: Sequence[int]) -> None:
        """Delete vectors with given IDs."""
        self.client.delete(collection_name=self.collection_name, points_selector=ids)

    def delete_collection(self) -> None:
        """Delete the collection associated with this adaptor instance."""
        logger.info(f"Deleting collection {self.collection_name}")
        self.client.delete_collection(collection_name=self.collection_name)

    def stats(self) -> dict[str, object]:
        """Return index statistics."""
        return {
            "metric": self.metric.value,
            "m": self.m,
            "ef_construct": self.ef,
        }

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self, "client") and self.client:
            self.client.close()
