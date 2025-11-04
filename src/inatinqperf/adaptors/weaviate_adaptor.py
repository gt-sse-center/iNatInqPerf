"""Weaviate adaptor using the v4 client library.

Docs can be found here:
- https://weaviate-python-client.readthedocs.io/en/stable/weaviate.html
- https://docs.weaviate.io/weaviate/guides
"""

from collections.abc import Sequence

import numpy as np
import weaviate
from datasets import Dataset as HuggingFaceDataset
from loguru import logger
from weaviate.classes.config import Configure, DataType, Property, VectorDistances
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.connect import ConnectionParams

from inatinqperf.adaptors.base import DataPoint, Query, SearchResult, VectorDatabase
from inatinqperf.adaptors.enums import IndexTypeBase, Metric


class WeaviateIndexType(IndexTypeBase):
    """Enum for various index types supported by Weaviate."""

    FLAT = "flat"
    HNSW = "hnsw"
    DYNAMIC = "dynamic"


class Weaviate(VectorDatabase):
    """Adaptor to help work with the Weaviate vector database."""

    @logger.catch(reraise=True)
    def __init__(
        self,
        dataset: HuggingFaceDataset,
        metric: Metric,
        index_type: WeaviateIndexType,
        url: str = "http://localhost",
        port: str = "8080",
        collection_name: str = "collection_name",
        batch_size: int = 1000,
        **params: object,  # noqa: ARG002
    ) -> None:
        """Initialise the adaptor with a dataset template and connectivity details."""
        super().__init__(dataset=dataset, metric=metric)

        self.collection_name = collection_name

        self.client = weaviate.WeaviateClient(
            connection_params=ConnectionParams.from_url(url=f"{url}:{port}", grpc_port=50051),
            skip_init_checks=False,
        )
        self.client.connect()

        ## Create collection. If it exists, then delete the previous one.
        if self.client.collections.exists(self.collection_name):
            self.client.collections.delete(self.collection_name)

        index_type_func = self._get_index_type(WeaviateIndexType(index_type))

        # The `id` and `vector` properties are created by default
        self.client.collections.create(
            self.collection_name,
            vector_config=Configure.Vectors.self_provided(
                vector_index_config=index_type_func(distance_metric=self._translate_metric(metric)),
            ),
            properties=[
                Property(
                    name="dataset_id",
                    data_type=DataType.INT,
                    description="The original ID of this data point in the dataset",
                )
            ],
        )

        # Upload the dataset to prepare the database.
        self._upload_dataset(dataset, batch_size)

    @staticmethod
    def _get_index_type(index_type: WeaviateIndexType) -> callable:
        if index_type == WeaviateIndexType.FLAT:
            return Configure.VectorIndex.flat
        if index_type == WeaviateIndexType.HNSW:
            return Configure.VectorIndex.hnsw
        if index_type == WeaviateIndexType.DYNAMIC:
            return Configure.VectorIndex.dynamic

        msg = f"Unsupported index type: '{index_type}'"
        raise ValueError(msg)

    @staticmethod
    def _translate_metric(metric: Metric) -> str:
        """Map internal metric names to Weaviate's expected identifiers."""
        if metric == Metric.INNER_PRODUCT:
            return VectorDistances.DOT
        if metric == Metric.COSINE:
            return VectorDistances.COSINE
        if metric == Metric.L2:
            return VectorDistances.L2_SQUARED
        if metric == Metric.MANHATTAN:
            return VectorDistances.MANHATTAN

        msg = f"Unsupported metric '{metric}'"
        raise ValueError(msg)

    def _upload_dataset(self, dataset: HuggingFaceDataset, batch_size: int) -> None:
        """Upload the HuggingFaceDataset to the vector database."""

        embeddings = np.asarray(dataset["embedding"], dtype=np.float32)
        if embeddings.size == 0:
            logger.warning("Dataset contains no embeddings; skipping ingest")
            return

        ids = dataset["id"]
        if len(ids) != len(embeddings):
            msg = "Length of dataset ids must match number of embeddings"
            raise ValueError(msg)

        collection = self.client.collections.use(self.collection_name)

        with collection.batch.fixed_size(batch_size=batch_size) as batch:
            for data_row in dataset:
                dataset_id = data_row["id"]
                uuid = weaviate.util.generate_uuid5(dataset_id)
                vector = data_row["embedding"]

                batch.add_object(uuid=uuid, vector=vector, properties={"dataset_id": dataset_id})

                if batch.number_errors >= 10:  # noqa: PLR2004
                    raise RuntimeError("Batch import stopped due to excessive errors.")

    def upsert(self, x: Sequence[DataPoint]) -> None:
        """Insert or update vectors and associated metadata."""

        collection = self.client.collections.use(self.collection_name)

        for dp in x:
            uuid = weaviate.util.generate_uuid5(dp.id)

            # Check if data point exists in database. If not, returns None
            data_object = collection.query.fetch_object_by_id(uuid)

            if data_object is None:
                collection.data.insert(
                    uuid=uuid,
                    vector=dp.vector,
                    properties={"dataset_id": dp.id},
                )
            else:
                collection.data.replace(
                    uuid=uuid,
                    vector=dp.vector,
                    properties={"dataset_id": dp.id},
                )

    def search(self, q: Query, topk: int, **kwargs) -> Sequence[SearchResult]:  # NOQA: ARG002
        """Search for the `topk` nearest vectors based on the query point `q`."""

        collection = self.client.collections.use(self.collection_name)
        response = collection.query.near_vector(
            near_vector=q.vector, limit=topk, return_metadata=MetadataQuery(distance=True, score=True)
        )

        return [SearchResult(id=o.properties["dataset_id"], score=o.metadata.score) for o in response.objects]

    def delete(self, ids: Sequence[int]) -> None:
        """Delete objects corresponding to the provided `ids`.

        If an ID is not present, nothing happens, i.e. it is an idempotent operation.
        """

        collection = self.client.collections.get(self.collection_name)

        # https://docs.weaviate.io/weaviate/manage-objects/delete#delete-multiple-objects-by-id
        ids_to_delete = [weaviate.util.generate_uuid5(i) for i in ids]
        collection.data.delete_many(where=Filter.by_id().contains_any(ids_to_delete))

    def stats(self) -> dict[str, object]:
        """Return summary statistics sourced via a Weaviate aggregation query."""

        collection = self.client.collections.use(self.collection_name)
        total_count = collection.aggregate.over_all(total_count=True).total_count

        return {
            "ntotal": total_count,
            "metric": self.metric.value,
            "collection_name": self.collection_name,
            "dim": self.dim,
        }

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self, "client") and self.client:
            self.client.close()
