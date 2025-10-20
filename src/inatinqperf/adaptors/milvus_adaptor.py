"""Milvus adaptor."""

from collections.abc import Sequence
from enum import Enum

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
from inatinqperf.adaptors.metric import Metric


class MilvusIndexType(str, Enum):
    """Enum for various index types supported by Milvus. For more details, see https://milvus.io/docs/index.md?tab=floating."""

    IVF_FLAT = "IVF_FLAT"
    IVF_SQ8 = "IVF_SQ8"
    IVF_PQ = "IVF_PQ"
    HNSW = "HNSW"
    HNSW_SQ = "HNSW_SQ"
    HNSW_PQ = "HNSW_PQ"

    @classmethod
    def _missing_(cls, value: str) -> "MilvusIndexType | None":
        value = value.lower()
        for member in cls:
            if member.value.lower() == value:
                return member
        return None


class Milvus(VectorDatabase):
    """Milvus vector database."""

    DATABASE_NAME: str = "default"
    COLLECTION_NAME: str = "collection_name"
    INDEX_NAME: str = f"{COLLECTION_NAME}_index"

    @logger.catch
    def __init__(self, dataset: HuggingFaceDataset, metric: Metric, index_type: MilvusIndexType | str, index_params: dict = {}, host: str = "localhost", port: str = "19530") -> None:
        super().__init__(dataset, metric)
        self.host = host
        self.port = port
        self.index_type = MilvusIndexType(index_type)
        try:
            connections.connect(host="localhost", port="19530")
            server_type = utility.get_server_type()
            logger.add(f"Milvus server is running. Server type: {server_type}")
        except Exception:
            logger.exception("Milvus server is not running or connection failed")

        self.client = MilvusClient(uri=f"http://{self.host}:{self.port}")
        self.metric = self._translate_metric(metric)

        # Remove collection if it already exists
        if self.client.has_collection(self.COLLECTION_NAME):
            self.client.drop_collection(self.COLLECTION_NAME)

        # Define collection schema
        self.schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_schema=True,
        )
        self.schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        self.schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.dim)

        self.index_params = self.client.prepare_index_params()
        self.index_params.add_index(
            field_name="vector",
            index_type=self.index_type.value,
            index_name=self.INDEX_NAME,
            metric_type=self.metric,
            params=index_params
        )

        self.client.create_collection(
            collection_name=self.COLLECTION_NAME,
            schema=self.schema,
            index_params=self.index_params,
        )

        batch_size = 1000

        for i in tqdm(range(0, len(dataset), batch_size)):
            batch_data = []
            end = min(i + batch_size, len(dataset))
            for k in range(i, end):
                rid = int(dataset[k]["id"])
                vec = dataset[k]["embedding"]
                batch_data.append({"id": rid, "vector": vec})
            self.client.insert(collection_name=self.COLLECTION_NAME, data=batch_data)

        self.client.load_collection(collection_name=self.COLLECTION_NAME)

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

        self.client.upsert(collection_name=self.COLLECTION_NAME, data=data)

    def delete(self, ids: Sequence[int]) -> None:
        """Delete vectors with given IDs."""
        self.client.delete(collection_name=self.COLLECTION_NAME, ids=ids)

    def search(self, q: Query, topk: int, **kwargs) -> Sequence[SearchResult]:  # NOQA: ARG002
        """Search for top-k nearest neighbors."""
        results = self.client.search(
            collection_name=self.COLLECTION_NAME, anns_field="vector", data=[q.vector], limit=topk, search_params={"metric_type": self.metric}
        )

        search_results = []

        logger.info(f"Results: {results}")

        for result in results:
            hit_ids = result.ids
            hit_distances = result.distances
            for hit_id, hit_distance in zip(hit_ids, hit_distances):
                search_results.append(SearchResult(id=hit_id, score=hit_distance))

        logger.info(f"Search results: {search_results}")
        # assert len(search_results) == topk
        return search_results

    def stats(self) -> None:
        """Return index statistics."""
        return self.client.describe_index(collection_name=self.COLLECTION_NAME, index_name=self.INDEX_NAME)
    
    def teardown(self) -> None:
        """Teardown the Milvus vector database."""
        self.client.drop_collection(self.COLLECTION_NAME)
        self.client.close()
        connections.disconnect(alias="default")
