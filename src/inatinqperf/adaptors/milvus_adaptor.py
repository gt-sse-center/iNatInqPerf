"""Milvus adaptor."""

from collections.abc import Sequence
from enum import Enum
import os
from tqdm import tqdm
from datasets import load_dataset, Dataset as HuggingFaceDataset

import numpy as np
from loguru import logger
from pymilvus import (
    Collection,
    DataType,
    FieldSchema,
    CollectionSchema,
    MilvusClient,
    connections,
    utility,
)

from pymilvus.bulk_writer import LocalBulkWriter, BulkFileType

from inatinqperf.adaptors.base import VectorDatabase, DataPoint, Query, SearchResult
from inatinqperf.adaptors.metric import Metric


class MilvusIndexType(Enum):
    """Enum for various index types supported by Milvus. For more details, see https://milvus.io/docs/index.md?tab=floating."""

    IVF_FLAT = "IVF_FLAT"
    IVF_SQ8 = "IVF_SQ8"
    IVF_PQ = "IVF_PQ"
    HNSW = "HNSW"
    HNSW_SQ = "HNSW_SQ"
    HNSW_PQ = "HNSW_PQ"


class Milvus(VectorDatabase):
    """Milvus vector database."""

    DATABASE_NAME: str = "default"
    COLLECTION_NAME: str = "collection_name"
    INDEX_NAME: str = f"{COLLECTION_NAME}_index"
    SERVER_HOST = "localhost"
    SERVER_PORT = "19530"  # default milvus server port

    @logger.catch
    def __init__(self, dataset: HuggingFaceDataset, metric: Metric, index_type: MilvusIndexType) -> None:
        super().__init__(dataset, metric)
        try:
            connections.connect(host="localhost", port="19530")
            server_type = utility.get_server_type()
            logger.add(f"Milvus server is running. Server type: {server_type}")
        except Exception:
            logger.exception("Milvus server is not running or connection failed")

        self.client = MilvusClient(uri=f"http://{self.SERVER_HOST}:{self.SERVER_PORT}")
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
            index_type=index_type,
            index_name=self.INDEX_NAME,
            metric_type=self.metric,
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

    def _translate_metric(self, metric: Metric) -> str:
        """Translate metric to Milvus metric type."""
        if metric == Metric.INNER_PRODUCT:
            return "IP"
        elif metric == Metric.COSINE:
            return "COSINE"
        elif metric == Metric.L2:
            return "L2"
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    
    def upsert(self, x: Sequence[DataPoint]) -> None:
        """Upsert vectors with given IDs."""

        data = []
        for dp in x:
            data.append({"id": dp.id, "vector": dp.vector})

        self.client.upsert(collection_name=self.COLLECTION_NAME, data=data)

    def delete(self, ids: Sequence[int]) -> None:
        """Delete vectors with given IDs."""
        self.client.delete(collection_name=self.COLLECTION_NAME, ids=ids)

    def search(self, q: Query, topk: int, **kwargs) -> Sequence[SearchResult]:
        """Search for top-k nearest neighbors."""
        results = self.client.search(
            collection_name=self.COLLECTION_NAME, data=[q.vector], topk=topk, metric=self.metric
        )

        search_results = []

        for result in results:
            hit_ids = result.ids
            hit_distances = result.distances
            for hit_id, hit_distance in zip(hit_ids, hit_distances):
                search_results.append(SearchResult(id=hit_id, score=hit_distance))
        
        return search_results

    def stats(self) -> None:
        """Return index statistics."""
        # return {"ntotal": 1}
        return self.client.describe_index(collection_name=self.COLLECTION_NAME, index_name=self.INDEX_NAME)
