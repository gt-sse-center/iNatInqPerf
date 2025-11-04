"""Milvus adaptor."""

from collections.abc import Sequence

from datasets import Dataset as HuggingFaceDataset
from loguru import logger
from pymilvus import (
    DataType,
    MilvusClient,
    connections,
    utility,
)
from tqdm import tqdm

from inatinqperf.adaptors.base import DataPoint, Query, SearchResult, VectorDatabase
from inatinqperf.adaptors.enums import IndexTypeBase, Metric


class MilvusIndexType(IndexTypeBase):
    """Enum for various index types supported by Milvus.

    For more details, see https://milvus.io/docs/index.md?tab=floating.
    """

    IVF_FLAT = "IVF_FLAT"
    IVF_SQ8 = "IVF_SQ8"
    IVF_PQ = "IVF_PQ"
    HNSW = "HNSW"
    HNSW_SQ = "HNSW_SQ"
    HNSW_PQ = "HNSW_PQ"


class Milvus(VectorDatabase):
    """Adaptor to help work with Milvus vector database."""

    @logger.catch(reraise=True)
    def __init__(
        self,
        dataset: HuggingFaceDataset,
        metric: Metric,
        index_type: MilvusIndexType,
        index_params: dict | None = None,
        url: str = "localhost",
        port: str = "19530",
        collection_name: str = "default_collection",
        batch_size: int = 1000,
        **params,  # noqa: ARG002
    ) -> None:
        super().__init__(dataset, metric)

        self.index_type = MilvusIndexType(index_type)
        self.index_name: str = f"{collection_name}_index"
        self.collection_name = collection_name

        try:
            connections.connect(host=url, port=port)
            server_type = utility.get_server_type()
            logger.info(f"Milvus server is running. Server type: {server_type}")
        except Exception:
            logger.exception("Milvus server is not running or connection failed")

        # NOTE: pymilvus is very slow to connect, takes ~8 seconds as per profiling.
        self.client = MilvusClient(uri=f"http://{url}:{port}")

        # Remove collection if it already exists
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)

        # Define collection schema
        schema = (
            self.client.create_schema(
                auto_id=False,
                enable_dynamic_schema=True,
            )
            .add_field(
                field_name="id",
                datatype=DataType.INT64,
                is_primary=True,
            )
            .add_field(
                field_name="vector",
                datatype=DataType.FLOAT_VECTOR,
                dim=self.dim,
            )
        )

        # This calls `.add_index` internally
        milvus_index_params = self.client.prepare_index_params(
            field_name="vector",
            index_type=self.index_type.value,
            index_name=self.index_name,
            metric_type=self._translate_metric(self.metric),
            params=index_params if index_params else {},
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=milvus_index_params,
        )

        for i in tqdm(range(0, len(dataset), batch_size)):
            batch_data = []
            end = min(i + batch_size, len(dataset))
            for k in range(i, end):
                rid = int(dataset[k]["id"])
                vec = dataset[k]["embedding"]
                batch_data.append({"id": rid, "vector": vec})
            self.client.insert(collection_name=self.collection_name, data=batch_data)

        # loads the index files and fields raw data into memory for rapid response to searches and queries
        self.client.load_collection(collection_name=self.collection_name)

    @staticmethod
    def _translate_metric(metric: Metric) -> str:
        """Translate metric to Milvus metric type."""
        if metric == Metric.INNER_PRODUCT:
            return "IP"
        if metric == Metric.COSINE:
            return "COSINE"
        if metric == Metric.L2:
            return "L2"

        msg = f"{metric} metric specified is not a valid one for Milvus."
        raise ValueError(msg)

    def upsert(self, x: Sequence[DataPoint]) -> None:
        """Upsert vectors with given IDs."""

        data = [{"id": int(dp.id), "vector": dp.vector} for dp in x]

        self.client.upsert(collection_name=self.collection_name, data=data)

    def delete(self, ids: Sequence[int]) -> None:
        """Delete vectors with given IDs."""
        self.client.delete(collection_name=self.collection_name, ids=ids)

    def search(self, q: Query, topk: int, **kwargs) -> Sequence[SearchResult]:  # NOQA: ARG002
        """Search for top-k nearest neighbors.

        The score returned in this case is the distance, so smaller is better.
        """
        results = self.client.search(
            collection_name=self.collection_name,
            anns_field="vector",
            data=[q.vector],
            limit=topk,
            search_params={
                "metric_type": self._translate_metric(self.metric),
            },
        )

        search_results = []

        # `results` should only have a single value
        result = results[0]

        hit_ids = result.ids
        hit_distances = result.distances
        for hit_id, hit_distance in zip(hit_ids, hit_distances):
            search_results.append(SearchResult(id=hit_id, score=hit_distance))

        return search_results

    def stats(self) -> None:
        """Return index statistics."""
        return self.client.describe_index(collection_name=self.collection_name, index_name=self.index_name)

    def close(self) -> None:
        """Teardown the Milvus vector database."""
        if hasattr(self, "client") and self.client:
            self.client.drop_collection(self.collection_name)
            self.client.close()
        connections.disconnect(alias="default")
